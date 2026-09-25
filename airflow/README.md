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

## Run on GCP

One spot VM (`infra/airflow_vm.tf`) runs the same Compose stack with an image
built by Cloud Build. It has no public ingress: SSH goes through IAP (your
account needs `roles/iap.tunnelResourceAccessor`; project owners have it) and
the UI through an SSH tunnel. From the repo root, with gcloud set to the
project:

1. Store the `.env` (the VM forces its own `AIRFLOW_IMAGE`, `AIRFLOW_UID`,
   data dir and MLflow settings, so the local file works as is):

   ```sh
   gcloud secrets versions add uplift-airflow-env --data-file=airflow/.env
   ```

2. Upload the raw data: `scripts/fetch_x5.sh gs://<project>-uplift-data`.
3. Build and push the image (x86, about 10 minutes):

   ```sh
   gcloud builds submit --config airflow/cloudbuild.yaml .
   ```

4. Create the VM: `cd infra && terraform apply -var airflow_vm_enabled=true`.
   The first boot installs Docker and pulls the image; check it with
   `scripts/airflow_vm.sh status`.
5. Run the DAG: `scripts/airflow_vm.sh trigger`.
6. Open the UI: `scripts/airflow_vm.sh tunnel`, then http://localhost:8080.

Every boot re-reads the secret, syncs `raw/x5` from the bucket and runs
`docker compose up`, so `scripts/airflow_vm.sh start` brings everything back.
The compose file and the DAG travel in the instance metadata: after changing
them, run `terraform apply` again and restart the VM. The weekly schedule only
fires while the VM is up. A spot preemption stops the VM and fails the running
task; start it and trigger again. Boot log:
`sudo journalctl -u google-startup-scripts` on the VM.

### Cost

Approximate us-central1 prices; check the pricing pages before relying on them.

- Running: e2-standard-4 spot about $0.04 to $0.05 per hour, plus about
  $0.0025 per hour for the ephemeral external IP.
- Stopped: only the 30 GB pd-balanced disk, about $3 per month.
- Artifact Registry storage for the image: cents per month. Cloud Build fits
  in its free tier.

The VM powers itself off once no DAG run has been running or queued for 30
minutes (`-var airflow_idle_minutes=N`) and it has been up at least that long.
`scripts/airflow_vm.sh stop` stops it at once;
`terraform apply -var airflow_vm_enabled=false` deletes it and its disk.
