"""Retraining on X5: ingest -> feature shards -> merge -> train.

Pipeline tasks run in /opt/venv, the package's own uv environment (see
airflow/Dockerfile), through @task.external_python. Only each function's source
is shipped there, so imports and helpers live inside the task bodies, which also
keeps DAG parsing light. Only paths and small dicts cross XCom.

The raw CSVs are not downloaded here: run scripts/fetch_x5.sh first.
"""

import os
from datetime import datetime

from airflow.sdk import Variable, dag, task

DATA_DIR = os.getenv("UPLIFT_DATA_DIR", "/opt/airflow/data")
FEATURES_DIR = f"{DATA_DIR}/features/x5"
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
def train(features_path: str) -> dict[str, float]:
    from uplift_pipeline.train import run

    # Plain floats: numpy scalars would not unpickle in Airflow's environment.
    return {k: float(v) for k, v in run(features_path=features_path).items()}


@dag(
    # The dataset is static: manual triggers only.
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    # Runs share the same output paths; two at once would clobber each other.
    max_active_runs=1,
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
    train(features)


uplift_training()
