# Airflow

Runs `dags/uplift_training_dag.py` on Airflow 3 with Docker Compose:
LocalExecutor, Postgres metadata DB, one machine. Commands run from the repo
root.

1. Download the raw data: `scripts/fetch_x5.sh` (writes `data/raw/x5`).
2. Create `airflow/.env` from `airflow/.env.example` and fill in the secrets.
3. Build and start (the first build compiles causalml on arm64 and takes a
   while):

   ```sh
   docker compose -f airflow/docker-compose.yml up -d --build
   ```

4. Open http://localhost:8080 and log in as `admin` with
   `AIRFLOW_ADMIN_PASSWORD`. On a VM the port is bound to localhost only; use
   `gcloud compute ssh <vm> -- -L 8080:localhost:8080`.
5. Unpause and trigger `uplift_training` in the UI, or:

   ```sh
   docker compose -f airflow/docker-compose.yml exec airflow-scheduler \
     airflow dags unpause uplift_training
   docker compose -f airflow/docker-compose.yml exec airflow-scheduler \
     airflow dags trigger uplift_training
   ```

Outputs land in `data/processed/x5` and `data/features/x5`. The shard count
comes from the Airflow Variable `uplift_num_shards`, else `UPLIFT_NUM_SHARDS`.
Stop with `docker compose -f airflow/docker-compose.yml down` (add `-v` to
drop the metadata DB and logs).
