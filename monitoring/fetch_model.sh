#!/usr/bin/env bash
# Download the served model version into monitoring/model for the local API.
# Usage: monitoring/fetch_model.sh [VERSION]   (default: 2)
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-$(gh variable get MLFLOW_TRACKING_URI)}"
# The MLflow server is IAM-only; a user identity token is accepted.
MLFLOW_TRACKING_TOKEN="$(gcloud auth print-identity-token)"
export MLFLOW_TRACKING_TOKEN
rm -rf monitoring/model
uv run python -c "import mlflow, sys; mlflow.artifacts.download_artifacts('models:/uplift-model/' + sys.argv[1], dst_path='monitoring/model')" "${1:-2}"
