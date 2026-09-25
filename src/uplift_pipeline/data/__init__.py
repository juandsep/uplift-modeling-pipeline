"""Data ingestion. Every loader returns features plus `treatment` (0/1) and `y`."""

import pandas as pd
from causalml.dataset import make_uplift_classification


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
