"""Runtime settings, read from environment variables."""

import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "uplift")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL", "uplift-model")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{REGISTERED_MODEL}/latest")

# Serving guardrails. API_KEY is required: the API fails closed without it.
API_KEY = os.getenv("API_KEY", "")
MAX_RECORDS = int(os.getenv("MAX_RECORDS", "1000"))
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(1024 * 1024)))
