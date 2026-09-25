"""Runtime settings, read from environment variables."""

import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "uplift")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL", "uplift-model")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{REGISTERED_MODEL}/latest")
