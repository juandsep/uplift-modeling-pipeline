# uplift-pipeline

Uplift model that estimates how much a treatment (a campaign, a discount)
changes the chance that a customer converts. It trains an XGBoost T-learner
with causalml, scores it with Qini and AUUC, tracks runs in MLflow and serves
predictions through a FastAPI endpoint. Retraining runs weekly on Airflow.

Training data is synthetic for now (`src/uplift_pipeline/data`).

## Project layout

```
src/uplift_pipeline/
  config.py        settings from environment variables
  data/            data loading
  models/          T-learner, saved as an MLflow pyfunc model
  evaluation/      Qini and AUUC
  train.py         train, evaluate, register the model
  serving/app.py   FastAPI app
dags/              Airflow DAG (weekly retraining)
docker/            API image
tests/             unit and integration tests
```

## Run locally

Requires [uv](https://docs.astral.sh/uv/). On macOS, xgboost also needs
`brew install libomp`.

```bash
uv sync
uv run pre-commit install
```

Train and register a model. It prints the version it registered:

```bash
uv run python -m uplift_pipeline.train
# registered uplift-model version 1 (serve it with MODEL_VERSION=1)
```

Start the API with that version:

```bash
MODEL_VERSION=1 API_KEY=dev-key uv run uvicorn uplift_pipeline.serving.app:app --reload
```

Each record must include every feature the model was trained on
(`x1_informative` ... `x13_increase_mix` with the synthetic data):

```bash
curl -s localhost:8000/predict \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d @request.json
```

Run the checks:

```bash
uv run pytest
uv run pre-commit run --all-files
uv run mypy src
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow server |
| `MLFLOW_EXPERIMENT` | `uplift` | Experiment name |
| `REGISTERED_MODEL` | `uplift-model` | Registry model name |
| `MODEL_VERSION` | none | Model version to serve |
| `MODEL_URI` | none | Full model URI, overrides `MODEL_VERSION` |
| `ALLOW_UNPINNED_MODEL` | `false` | Local only: allow serving `latest` |
| `API_KEY` | none | Required by `/predict` (sent as `X-API-Key`) |
| `MAX_RECORDS` | `1000` | Max rows per request |
| `MAX_BODY_BYTES` | `1048576` | Max request size |

The API only serves a fixed model version, never `latest`. Loading a model
runs pickled code, so a moving alias would let anyone with registry write
access change what runs in production. To release a new model, set
`MODEL_VERSION` to the new version.

## Deploy on GCP

The API runs on Cloud Run and the DAG on Cloud Composer.

### API (Cloud Run)

The `Deploy` workflow runs on every push to `main`. It builds the image,
pushes it to Artifact Registry and deploys the `uplift-api` service.

One-time setup:

1. Create an Artifact Registry Docker repository.
2. Set up Workload Identity Federation for this GitHub repository and a
   deploy service account with `roles/run.admin`,
   `roles/artifactregistry.writer` and `roles/iam.serviceAccountUser`.
3. Create a runtime service account for the service with
   `roles/secretmanager.secretAccessor`.
4. Store the API key in Secret Manager as `uplift-api-key`.
5. Add these repository variables in GitHub (Settings > Variables):
   `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_ARTIFACT_REPO`, `GCP_WIF_PROVIDER`,
   `GCP_DEPLOY_SA`, `GCP_RUNTIME_SA`, `MLFLOW_TRACKING_URI`, `MODEL_VERSION`.
6. Create a `production` environment with required reviewers, so deploys
   wait for approval.

The service requires authenticated calls (Cloud Run IAM) on top of the API
key. Callers need `roles/run.invoker`.

### Training (Cloud Composer)

The Composer environment needs this package installed. Build it and publish
it to an Artifact Registry Python repository that the environment can
install from:

```bash
uv build
uv publish --publish-url https://REGION-python.pkg.dev/PROJECT/REPO/
gcloud composer environments update ENV --location REGION \
  --update-pypi-package "uplift-pipeline==0.1.0" \
  --update-env-variables MLFLOW_TRACKING_URI=https://your-mlflow-server
gcloud composer environments storage dags import --environment ENV \
  --location REGION --source dags/uplift_training_dag.py
```

## Contributing

Work goes on a branch cut from `dev`, is merged into `dev`, and `dev` is
merged into `main` to release. See [CONTRIBUTING.md](CONTRIBUTING.md).
