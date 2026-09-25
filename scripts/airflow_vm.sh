#!/usr/bin/env bash
# Manage the Airflow spot VM (infra/airflow_vm.tf) over IAP.
# Usage: scripts/airflow_vm.sh start|stop|tunnel|trigger|status
# Uses gcloud's current project; override with CLOUDSDK_CORE_PROJECT.
set -euo pipefail

vm=${AIRFLOW_VM:-uplift-airflow}
zone=${AIRFLOW_VM_ZONE:-us-central1-a}
dc="sudo docker compose -f /opt/uplift/airflow/docker-compose.yml"
ssh() { gcloud compute ssh "$vm" --zone "$zone" --tunnel-through-iap "$@"; }

case "${1:-}" in
  start) gcloud compute instances start "$vm" --zone "$zone" ;;
  stop) gcloud compute instances stop "$vm" --zone "$zone" ;;
  # UI at http://localhost:8080 while this runs.
  tunnel) ssh -- -N -L 8080:localhost:8080 ;;
  # Unpause first: a run of a paused DAG stays queued forever. Waits up to 10
  # minutes for the stack and the DAG to come up after a boot.
  trigger)
    ssh --command "for i in \$(seq 60); do
        $dc exec -T airflow-scheduler airflow dags unpause uplift_training && break
        [ \$i = 60 ] && exit 1; sleep 10
      done && $dc exec -T airflow-scheduler airflow dags trigger uplift_training"
    ;;
  status)
    state=$(gcloud compute instances describe "$vm" --zone "$zone" --format='value(status)')
    echo "$vm: $state"
    if [ "$state" = RUNNING ]; then ssh --command "$dc ps"; fi
    ;;
  *)
    echo "usage: $0 start|stop|tunnel|trigger|status" >&2
    exit 2
    ;;
esac
