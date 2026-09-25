# uplift-pipeline

Automated uplift modeling pipeline: XGBoost T-learner (causalml), Qini/AUUC
evaluation, MLflow tracking and registry, Airflow retraining, FastAPI serving.

## Layout

- `src/uplift_pipeline/data` - ingestion (synthetic for now)
- `src/uplift_pipeline/models` - causal model, wrapped as MLflow pyfunc
- `src/uplift_pipeline/evaluation` - Qini and AUUC
- `src/uplift_pipeline/train.py` - train, evaluate, log, register
- `src/uplift_pipeline/serving` - FastAPI inference API
- `dags/` - Airflow DAG
- `docker/` - serving image

## Usage

```bash
uv sync
uv run pre-commit install
uv run python -m uplift_pipeline.train                     # trains and registers the model
API_KEY=dev-key uv run uvicorn uplift_pipeline.serving.app:app --reload
uv run pytest
```

`/predict` requires the `X-API-Key` header:

```bash
curl -s localhost:8000/predict -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"records": [{"x1_informative": 0.5, "x2_informative": -1.2}]}'
```

macOS: xgboost needs OpenMP (`brew install libomp`).

Settings (env vars):

| Variable | Default | Notes |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Use an authenticated server in prod |
| `MLFLOW_EXPERIMENT` | `uplift` | |
| `REGISTERED_MODEL` | `uplift-model` | |
| `MODEL_URI` | `models:/<model>/latest` | Pin a version in prod |
| `API_KEY` | none | Required by `/predict`; the API returns 503 without it |
| `MAX_RECORDS` | `1000` | Max rows per request |
| `MAX_BODY_BYTES` | `1048576` | Max request body |

The API authenticates but does not rate limit. Put it behind an ingress or
gateway that does, and terminate TLS there. `MLFLOW_TRACKING_USERNAME`,
`MLFLOW_TRACKING_PASSWORD` and `MLFLOW_TRACKING_TOKEN` are read by MLflow
directly when the registry requires auth.

