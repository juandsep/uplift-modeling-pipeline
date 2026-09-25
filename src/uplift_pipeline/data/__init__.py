"""Data ingestion. Every loader returns features plus `treatment` (0/1) and `y`."""

from pathlib import Path

import pandas as pd
from causalml.dataset import make_uplift_classification

X5_KEYS = ["client_id", "treatment", "y"]


def load_training_data(
    n_samples: int = 10_000, seed: int = 42
) -> tuple[pd.DataFrame, list[str]]:
    # n_samples is per arm (total rows = 2 * n_samples).
    # ponytail: synthetic data, replace with the real source (e.g. BigQuery) here.
    df, features = make_uplift_classification(
        n_samples=n_samples,
        treatment_name=["control", "treatment1"],
        random_seed=seed,
    )
    df["treatment"] = (df["treatment_group_key"] == "treatment1").astype(int)
    df = df.rename(columns={"conversion": "y"})
    return df[[*features, "treatment", "y"]], features


def load_x5_features(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Per-client X5 feature table: client_id, treatment, y, numeric features."""
    df = pd.read_parquet(path)
    missing = [c for c in X5_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    if not df["client_id"].is_unique:
        raise ValueError(f"{path}: client_id is not unique")
    for col in ("treatment", "y"):
        if df[col].isna().any() or not df[col].isin([0, 1]).all():
            raise ValueError(f"{path}: {col!r} must be 0/1 with no nulls")
    features = [c for c in df.columns if c not in X5_KEYS]
    if not features:
        raise ValueError(f"{path}: no feature columns")
    bad = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"{path}: non-numeric feature columns {bad}")
    # float64 turns nulls into NaN (XGBoost handles it natively) and gives MLflow
    # a double-only signature, which matches the floats serving sends.
    out = df[features].astype("float64")
    out["treatment"] = df["treatment"].astype(int)
    out["y"] = df["y"].astype(int)
    return out, features
