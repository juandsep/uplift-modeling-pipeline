# Airflow on one spot VM (Debian 12 + Docker Compose). Off by default:
#   terraform apply -var airflow_vm_enabled=true
# No public ingress: SSH only from IAP, UI through an SSH tunnel. The VM powers
# itself off when Airflow is idle. See airflow/README.md, "Run on GCP".

variable "airflow_vm_enabled" {
  description = "Create the Airflow VM. False costs nothing."
  type        = bool
  default     = false
}

variable "airflow_machine_type" {
  type    = string
  default = "e2-standard-4"
}

variable "airflow_zone" {
  type    = string
  default = "us-central1-a"
}

variable "airflow_image_tag" {
  description = "Tag of us-central1-docker.pkg.dev/<project>/uplift/airflow the VM pulls."
  type        = string
  default     = "latest"
}

variable "airflow_idle_minutes" {
  description = "Power off after this long with no DAG run running or queued."
  type        = number
  default     = 30
}

variable "mlflow_tracking_uri" {
  description = "Shared MLflow server. Empty: a SQLite store on the VM disk."
  type        = string
  default     = "https://mlflow-168267984831.us-central1.run.app"
}

# Holds the contents of airflow/.env. Added by hand, never through Terraform:
#   gcloud secrets versions add uplift-airflow-env --data-file=airflow/.env
resource "google_secret_manager_secret" "airflow_env" {
  secret_id = "uplift-airflow-env"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "training_reads_airflow_env" {
  secret_id = google_secret_manager_secret.airflow_env.id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.sa["training"].member
}

resource "google_artifact_registry_repository_iam_member" "training_pulls_images" {
  repository = google_artifact_registry_repository.images.name
  location   = google_artifact_registry_repository.images.location
  role       = "roles/artifactregistry.reader"
  member     = google_service_account.sa["training"].member
}

resource "google_project_iam_member" "training_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = google_service_account.sa["training"].member
}

# gcloud builds submit runs as the Compute Engine default account on new
# projects; make sure it may push the image (airflow/cloudbuild.yaml).
resource "google_project_iam_member" "cloudbuild_builder" {
  project    = var.project_id
  role       = "roles/cloudbuild.builds.builder"
  member     = "serviceAccount:${data.google_project.this.number}-compute@developer.gserviceaccount.com"
  depends_on = [google_project_service.apis]
}

# Own network: the default one allows SSH from anywhere. Networks and firewall
# rules are free, so they are not gated.
resource "google_compute_network" "airflow" {
  name                    = "uplift-airflow"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "airflow" {
  name          = "uplift-airflow"
  network       = google_compute_network.airflow.id
  region        = var.region
  ip_cidr_range = "10.10.0.0/24"
}

resource "google_compute_firewall" "airflow_iap_ssh" {
  name          = "uplift-airflow-iap-ssh"
  network       = google_compute_network.airflow.id
  source_ranges = ["35.235.240.0/20"] # IAP TCP forwarding
  target_tags   = ["uplift-airflow"]
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
}

resource "google_compute_instance" "airflow" {
  count        = var.airflow_vm_enabled ? 1 : 0
  name         = "uplift-airflow"
  machine_type = var.airflow_machine_type
  zone         = var.airflow_zone
  tags         = ["uplift-airflow"]

  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      # pd-standard: it is billed while the VM is stopped, and boot speed
      # does not matter for a batch job.
      size = 20
      type = "pd-standard"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.airflow.id
    # Ephemeral external IP for egress (Docker Hub, apt). No Cloud NAT or
    # static IP: both bill while the VM is stopped.
    access_config {}
  }

  scheduling {
    provisioning_model          = "SPOT"
    preemptible                 = true
    automatic_restart           = false
    instance_termination_action = "STOP"
  }

  service_account {
    email  = google_service_account.sa["training"].email
    scopes = ["cloud-platform"]
  }

  # The startup script reads everything else from these keys. A changed DAG or
  # compose file needs terraform apply (in place) and a reboot.
  # ponytail: one DAG file; add a key per file, or sync dags/ from GCS, if more appear.
  metadata = {
    enable-oslogin = "TRUE"
    startup-script = file("${path.module}/airflow_vm_startup.sh")
    compose-yaml   = file("${path.module}/../airflow/docker-compose.yml")
    dag-py         = file("${path.module}/../dags/uplift_training_dag.py")
    airflow-image  = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/airflow:${var.airflow_image_tag}"
    data-bucket    = google_storage_bucket.data.name
    mlflow-uri     = var.mlflow_tracking_uri
    idle-minutes   = tostring(var.airflow_idle_minutes)
  }

  depends_on = [
    google_secret_manager_secret_iam_member.training_reads_airflow_env,
    google_artifact_registry_repository_iam_member.training_pulls_images,
    google_project_iam_member.training_logs,
  ]
}
