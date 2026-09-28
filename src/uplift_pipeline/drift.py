"""Drift between a reference and a current batch of clients.

Run: python -m uplift_pipeline.drift REFERENCE CURRENT [--pushgateway HOST:PORT]

REFERENCE is the reference_profile.json that training logs to MLflow, or a
feature table to profile on the spot. CURRENT is a batch of clients, e.g. the
output of uplift_pipeline.simulate.

Reports, and optionally pushes to a Prometheus Pushgateway:
- PSI per feature on the reference deciles, with null as its own bin, so a
  feature that starts arriving empty shows up too. Above 0.2 is drift.
- PSI of the predicted uplift, when both sides have it.
- Lift in the top 20% (treated minus control conversion among the clients
  ranked highest), when CURRENT has treatment, y and uplift. Only this one
  sees concept drift; the normalized Qini breaks when the effect vanishes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BINS = 10
DRIFT = 0.2
EPS = 1e-4  # An empty bin would make PSI infinite.


def profile(df: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """Decile edges and bin shares per column; the last share is null."""
    out = {}
    for col in columns:
        values = df[col].dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, BINS + 1)[1:-1]))
        out[col] = {"edges": edges.tolist(), "shares": _shares(df[col], edges)}
    return out


def _shares(col: pd.Series, edges: np.ndarray) -> list[float]:
    values = col.to_numpy(dtype=float)
    present = values[~np.isnan(values)]
    counts = np.bincount(
        np.searchsorted(edges, present, side="right"), minlength=len(edges) + 1
    )
    counts = np.append(counts, np.isnan(values).sum())
    return (counts / max(len(values), 1)).tolist()


def psi(reference: dict, col: pd.Series) -> float:
    expected = np.clip(reference["shares"], EPS, None)
    actual = np.clip(_shares(col, np.asarray(reference["edges"])), EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def top_lift(df: pd.DataFrame, k: float = 0.2) -> float:
    """Treated minus control conversion among the top k by predicted uplift."""
    top = df.nlargest(max(int(len(df) * k), 1), "uplift")
    rates = top.groupby("treatment")["y"].mean()
    return float(rates.get(1, np.nan) - rates.get(0, np.nan))


def report(reference: dict[str, dict], current: pd.DataFrame) -> dict:
    features = {c: psi(reference[c], current[c]) for c in reference if c in current}
    out: dict = {"feature_psi": features, "rows": len(current)}
    if "uplift" in reference and "uplift" in current:
        out["uplift_psi"] = features.pop("uplift")
    if {"treatment", "y", "uplift"} <= set(current.columns):
        out["top20_lift"] = top_lift(current)
    return out


def push(result: dict, gateway: str, window: str) -> None:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    reg = CollectorRegistry()
    g = Gauge(
        "uplift_feature_psi",
        "PSI against the training reference",
        ["feature"],
        registry=reg,
    )
    for feature, value in result["feature_psi"].items():
        g.labels(feature).set(value)
    Gauge("uplift_drift_rows", "Rows in the drift window", registry=reg).set(
        result["rows"]
    )
    if "uplift_psi" in result:
        Gauge("uplift_score_psi", "PSI of the predicted uplift", registry=reg).set(
            result["uplift_psi"]
        )
    if "top20_lift" in result:
        Gauge(
            "uplift_top20_lift",
            "Treated minus control conversion, top 20%",
            registry=reg,
        ).set(result["top20_lift"])
    push_to_gateway(
        gateway, job="uplift_drift", grouping_key={"window": window}, registry=reg
    )


def _load_reference(path: str) -> dict[str, dict]:
    if path.endswith(".json"):
        return json.loads(Path(path).read_text())
    from uplift_pipeline.data import load_x5_features
    from uplift_pipeline.train import split

    df, features = load_x5_features(path)
    # The rows the model was trained on, as training profiles them.
    return profile(split(df)[0], features)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("reference", help="reference_profile.json or a feature table")
    p.add_argument("current", help="Parquet batch of clients")
    p.add_argument("--pushgateway", metavar="HOST:PORT")
    p.add_argument("--window", default="batch", help="grouping key for the push")
    args = p.parse_args()

    result = report(_load_reference(args.reference), pd.read_parquet(args.current))
    for feature, value in sorted(result["feature_psi"].items(), key=lambda kv: -kv[1]):
        print(f"{feature:28s} {value:7.3f}{'  DRIFT' if value > DRIFT else ''}")
    for key in ("uplift_psi", "top20_lift", "rows"):
        if key in result:
            print(f"{key:28s} {result[key]:7.3f}")
    if args.pushgateway:
        push(result, args.pushgateway, args.window)
