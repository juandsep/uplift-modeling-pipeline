"""Score every client in batch. Run: python -m uplift_pipeline.score

Reads FEATURES_PATH, the pinned model (MODEL_VERSION or MODEL_URI) and writes
SCORES_PATH, sorted by uplift so the top k% is the first rows.
"""

import os
from pathlib import Path

import mlflow
import pandas as pd

from uplift_pipeline import config
from uplift_pipeline.data import load_x5_features
from uplift_pipeline.train import split


def score(features_path: str | Path, model_uri: str, out_path: str | Path) -> Path:
    """Write client_id, uplift, treatment, y, split for every client, best first.

    split is "test" for the rows training held out (same seed): only those give
    an unbiased estimate of incremental conversions.
    """
    config.assert_model_uri_is_pinned(model_uri)
    df, features = load_x5_features(features_path)
    # load_x5_features drops client_id but keeps the row index, so it aligns.
    ids = pd.read_parquet(features_path, columns=["client_id"])["client_id"]
    config.refresh_mlflow_token(config.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(model_uri)
    scores = pd.DataFrame(
        {
            "client_id": ids,
            "uplift": model.predict(df[features]),
            "treatment": df["treatment"],
            "y": df["y"],
            "split": "train",
        }
    )
    scores.loc[split(df)[1].index, "split"] = "test"
    scores = scores.sort_values("uplift", ascending=False, ignore_index=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename so a crash never leaves a half-written file behind.
    tmp = out.with_suffix(".tmp")
    scores.to_parquet(tmp, index=False)
    tmp.replace(out)
    return out


if __name__ == "__main__":
    if not config.FEATURES_PATH:
        raise SystemExit("FEATURES_PATH is not set")
    out = os.getenv("SCORES_PATH", "data/scores/x5/scores.parquet")
    print(score(config.FEATURES_PATH, config.MODEL_URI, out))
