# uplift-pipeline

Uplift model that estimates how much a treatment (a campaign, a discount)
changes the chance that a customer converts. It trains an XGBoost T-learner
with causalml, scores it with Qini and AUUC, tracks runs in MLflow and serves
predictions through a FastAPI endpoint. An Airflow DAG runs ingestion,
sharded feature building and training on demand.

It trains on the X5 RetailHero dataset (see Data below). Without
`FEATURES_PATH` it falls back to synthetic data, which the tests use.

## Project layout

```
src/uplift_pipeline/
  config.py        settings from environment variables
  data/            data loading
  models/          T-learner, saved as an MLflow pyfunc model
  evaluation/      Qini and AUUC
  train.py         train, evaluate, register the model
  serving/app.py   FastAPI app
dags/              Airflow DAG (ingest, features, train)
infra/             Terraform for the GCP resources
scripts/           dataset download
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
| `FEATURES_PATH` | none | X5 feature table (Parquet) to train on; unset uses synthetic data |
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

## Data

The model will train on the X5 RetailHero uplift dataset: about 200k
clients from a randomized campaign plus their purchase history (about 45M
rows). Download it and check the checksums, optionally uploading to GCS:

```bash
scripts/fetch_x5.sh                         # to data/raw/x5
scripts/fetch_x5.sh gs://PROJECT-uplift-data  # and to GCS
```

## Deploy on GCP

`infra/main.tf` creates the base resources: APIs, the data
bucket, the Artifact Registry repository, service accounts, Workload
Identity Federation for GitHub and a monthly budget that unlinks billing
from the project once spend reaches it (default $15). `infra/airflow_vm.tf`
adds an optional spot VM for Airflow, off by default (see `airflow/README.md`).

```bash
gcloud auth application-default login
cd infra
cp terraform.tfvars.example terraform.tfvars   # set project and billing account
terraform init
terraform apply
terraform output github_variables
```

Then store the API key (the value never goes through Terraform):

```bash
printf '%s' "$API_KEY" | gcloud secrets versions add uplift-api-key --data-file=-
```

### API (Cloud Run)

The `Deploy` workflow runs on every push to `main`. It builds the image,
pushes it to Artifact Registry and deploys the `uplift-api` service.

One-time setup, after `terraform apply`:

1. Add the values from `terraform output github_variables` as repository
   variables in GitHub (Settings > Variables), plus `MLFLOW_TRACKING_URI`
   and `MODEL_VERSION`.
2. Create a `production` environment with required reviewers, so deploys
   wait for approval.

The service requires authenticated calls (Cloud Run IAM) on top of the API
key. Callers need `roles/run.invoker`.

## Contributing

Work goes on a branch cut from `dev`, is merged into `dev`, and `dev` is
merged into `main` to release. See [CONTRIBUTING.md](CONTRIBUTING.md).
