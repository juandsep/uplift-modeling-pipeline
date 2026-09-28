"""Retraining on X5: ingest -> feature shards -> merge -> train -> score.

Pipeline tasks run in /opt/venv, the package's own uv environment (see
airflow/Dockerfile), through @task.external_python. Only each function's source
is shipped there, so imports and helpers live inside the task bodies, which also
keeps DAG parsing light. Only paths and small dicts cross XCom.

The raw CSVs are not downloaded here: run scripts/fetch_x5.sh first.
"""

import os
from datetime import datetime, timedelta

from airflow.sdk import Variable, dag, task

DATA_DIR = os.getenv("UPLIFT_DATA_DIR", "/opt/airflow/data")
FEATURES_DIR = f"{DATA_DIR}/features/x5"
SCORES_PATH = f"{DATA_DIR}/scores/x5/scores.parquet"
# Set on the GCP VM (infra/airflow_vm_startup.sh): scores are copied there too.
DATA_BUCKET = os.getenv("UPLIFT_DATA_BUCKET", "")
SHARD_THREADS = int(os.getenv("UPLIFT_SHARD_THREADS", "2"))
# Shards running at once; also bounded by AIRFLOW__CORE__PARALLELISM.
MAX_PARALLEL_SHARDS = int(os.getenv("UPLIFT_MAX_PARALLEL_SHARDS", "8"))
VENV = {
    "python": os.getenv("UPLIFT_PYTHON", "/opt/venv/bin/python"),
    "expect_airflow": False,
}


@task.external_python(**VENV)
def ingest(data_dir: str) -> str:
    from pathlib import Path

    from uplift_pipeline.data.x5 import convert_x5

    raw, out = Path(data_dir, "raw/x5"), Path(data_dir, "processed/x5")
    # A marker, not the Parquet files: a crash mid-write must not look done.
    done = out / "_SUCCESS"
    if done.exists():
        print(f"{out} already converted, skipping")
        return str(out)
    if not raw.is_dir():
        raise FileNotFoundError(f"{raw} is missing; run scripts/fetch_x5.sh first")
    print(convert_x5(raw, out))
    done.touch()
    return str(out)


@task
def plan_shards() -> list[dict[str, int]]:
    # Read at run time, not parse time: no metadata DB hit on every parse, and
    # a changed Variable applies to the next run.
    default = os.getenv("UPLIFT_NUM_SHARDS", "8")
    n = int(Variable.get("uplift_num_shards", default=default))
    return [{"shard": i, "num_shards": n} for i in range(n)]


@task.external_python(**VENV, max_active_tis_per_dag=MAX_PARALLEL_SHARDS)
def features_shard(
    processed_dir: str, features_dir: str, shard: int, num_shards: int, threads: int
) -> str:
    from pathlib import Path

    from uplift_pipeline.features.x5 import build_shard

    path = build_shard(
        Path(processed_dir), Path(features_dir), shard, num_shards, threads
    )
    return str(path)


@task.external_python(**VENV)
def merge_features(features_dir: str, shards: list[dict[str, int]]) -> str:
    from pathlib import Path

    from uplift_pipeline.features.x5 import merge_shards

    return str(merge_shards(Path(features_dir), len(shards)))


@task.external_python(**VENV)
def train(features_path: str) -> dict:
    from uplift_pipeline.train import run

    metrics, version = run(features_path=features_path)
    if version is None:
        raise RuntimeError("the registry assigned no model version")
    # Plain floats: numpy scalars would not unpickle in Airflow's environment.
    return {"metrics": {k: float(v) for k, v in metrics.items()}, "version": version}


@task.external_python(**VENV)
def score(features_path: str, trained: dict, out_path: str, bucket: str) -> str:
    from uplift_pipeline import config
    from uplift_pipeline.score import score as score_clients

    # The version this run registered, pinned: a later registry write cannot swap it.
    uri = f"models:/{config.REGISTERED_MODEL}/{trained['version']}"
    path = score_clients(features_path, uri, out_path)
    if not bucket:
        return str(path)
    from google.cloud import storage

    blob = storage.Client().bucket(bucket).blob("scores/x5/scores.parquet")
    blob.upload_from_filename(str(path))
    return f"gs://{bucket}/{blob.name}"


@dag(
    # The dataset is static: manual triggers only.
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    # Runs share the same output paths; two at once would clobber each other.
    max_active_runs=1,
    # A task killed by a preemption or restart is retried instead of failing the run.
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["uplift"],
)
def uplift_training():
    processed = ingest(DATA_DIR)
    plan = plan_shards()
    shards = features_shard.partial(
        processed_dir=processed, features_dir=FEATURES_DIR, threads=SHARD_THREADS
    ).expand_kwargs(plan)
    features = merge_features(FEATURES_DIR, plan)
    shards >> features
    score(features, train(features), SCORES_PATH, DATA_BUCKET)


uplift_training()
