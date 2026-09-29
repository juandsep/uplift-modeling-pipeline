"""Drifted copies of the X5 feature table, to exercise drift monitoring.

Run: python -m uplift_pipeline.simulate SCENARIO --out scenario.parquet
     [--send http://localhost:8080 --api-key KEY]

Every scenario resamples real clients (so the joint distribution stays
realistic) and then breaks one thing on purpose:

- baseline: nothing; drift metrics should stay quiet.
- covariate: spend inflates and clients age; the inputs move, the effect of
  the campaign does not.
- missing: an upstream join fails for part of the rows and some features
  arrive null.
- concept: the inputs look the same, but the campaign stops working
  (treatment reassigned at random, so it no longer changes y). Only labelled
  data sees this one. Watch the lift in the top k% (treated minus control
  conversion), not the normalized Qini: causalml divides by the total effect,
  which goes to zero here, so the normalized Qini turns meaningless.

The CLI samples the rows training held out, so the Qini is not inflated by
clients the model has seen. The output keeps treatment and y for the Qini.
"""

import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

from uplift_pipeline.data import load_x5_features
from uplift_pipeline.train import split

SCENARIOS = ("baseline", "covariate", "missing", "concept")
SPEND = ["total_spend", "mean_spend", "max_spend", "spend_30d", "redeemed_amount"]
# Columns a failed join to the purchase history would leave null.
JOINED = ["age", "total_spend", "n_transactions", "recency_days", "spend_30d"]


def make_scenario(
    df: pd.DataFrame, scenario: str, n: int, seed: int = 0, strength: float = 1.0
) -> pd.DataFrame:
    """Resample n rows of df and apply the scenario; strength 0 is baseline."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    rng = np.random.default_rng(seed)
    out = df.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    if scenario == "covariate":
        spend = [c for c in SPEND if c in out]
        out[spend] = out[spend] * (1 + 0.5 * strength)
        if "age" in out:
            out["age"] = out["age"] + 10 * strength
    elif scenario == "missing":
        rows = rng.random(n) < 0.3 * strength
        out.loc[rows, [c for c in JOINED if c in out]] = np.nan
    elif scenario == "concept":
        # Shuffling y inside one arm would keep the other arm's signal and
        # inflate the Qini; shuffling treatment removes the effect everywhere.
        k = int(n * min(strength, 1.0))
        picked = rng.choice(n, size=k, replace=False)
        out.loc[picked, "treatment"] = rng.permutation(
            out.loc[picked, "treatment"].to_numpy()
        )
    return out


def send(
    df: pd.DataFrame, features: list[str], url: str, api_key: str, batch: int = 500
) -> list[float]:
    """POST the features to {url}/predict in batches; return every uplift."""
    uplift: list[float] = []
    for start in range(0, len(df), batch):
        chunk = df[features].iloc[start : start + batch]
        # to_json turns NaN into null, which the API reads as a missing value.
        body = json.dumps({"records": json.loads(chunk.to_json(orient="records"))})
        req = urllib.request.Request(
            url.rstrip("/") + "/predict",
            data=body.encode(),
            headers={"Content-Type": "application/json", "X-API-Key": api_key},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            uplift += json.load(resp)["uplift"]
    return uplift


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scenario", choices=SCENARIOS)
    p.add_argument("--features", default="data/features/x5/client_features.parquet")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--out", help="write the scenario (with treatment, y) here")
    p.add_argument("--send", metavar="URL", help="POST it to URL/predict")
    p.add_argument("--api-key", default="")
    args = p.parse_args()

    df, features = load_x5_features(args.features)
    out = make_scenario(split(df)[1], args.scenario, args.n, args.seed, args.strength)
    if args.send:
        out["uplift"] = send(out, features, args.send, args.api_key)
    if args.out:
        out.to_parquet(args.out, index=False)
    print(f"{args.scenario}: {len(out)} rows")
