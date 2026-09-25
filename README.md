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
uv run uvicorn uplift_pipeline.serving.app:app --reload    # serves MODEL_URI
uv run pytest
```

macOS: xgboost needs OpenMP (`brew install libomp`).

Settings (env vars): `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`, `REGISTERED_MODEL`, `MODEL_URI`.
