"""Weekly retraining. Requires the uplift-pipeline package in the Airflow env."""

from datetime import datetime

from airflow.sdk import dag, task


@dag(
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["uplift"],
)
def uplift_training():
    # ponytail: one task, since data lives in memory. Split into
    # extract/train/evaluate once data is staged in GCS/BigQuery.
    @task
    def train() -> dict[str, float]:
        from uplift_pipeline.train import run

        return run()

    train()


uplift_training()
