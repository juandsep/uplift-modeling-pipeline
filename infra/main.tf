# Base GCP resources: APIs, data bucket, image registry, service accounts,
# GitHub Workload Identity Federation and a budget alert.
# State is local (terraform.tfstate, git-ignored). Move it to a GCS backend
# if more than one person applies this.

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }
}

variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "github_repo" {
  description = "owner/name of the repository allowed to deploy."
  type        = string
  default     = "juandsep/uplift-modeling-pipeline"
}

variable "billing_account" {
  description = "Billing account ID (XXXXXX-XXXXXX-XXXXXX) for the budget alert."
  type        = string
}

variable "monthly_budget_usd" {
  description = "Billing is cut at this amount. Kept below the real limit ($20) because billing data lags."
  type        = number
  default     = 15
}

provider "google" {
  project               = var.project_id
  region                = var.region
  user_project_override = true
  billing_project       = var.project_id
}

data "google_project" "this" {}

resource "google_project_service" "apis" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "iap.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "sts.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# Storage

resource "google_storage_bucket" "data" {
  name                        = "${var.project_id}-uplift-data"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  depends_on                  = [google_project_service.apis]
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "uplift"
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]

  # Every deploy pushes a new tag; keep the registry from growing forever.
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 10
    }
  }
  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      older_than = "2592000s" # 30 days
    }
  }
}

# The value is added by hand, never through Terraform, so it stays out of state:
#   printf '%s' "$KEY" | gcloud secrets versions add uplift-api-key --data-file=-
resource "google_secret_manager_secret" "api_key" {
  secret_id = "uplift-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

# Service accounts

locals {
  service_accounts = {
    deploy   = "GitHub Actions deploys"
    api      = "Cloud Run API runtime"
    training = "ETL and training VMs and jobs"
  }
}

resource "google_service_account" "sa" {
  for_each     = local.service_accounts
  account_id   = "uplift-${each.key}"
  display_name = each.value
  depends_on   = [google_project_service.apis]
}

resource "google_project_iam_member" "deploy" {
  for_each = toset(["roles/run.admin", "roles/artifactregistry.writer"])
  project  = var.project_id
  role     = each.value
  member   = google_service_account.sa["deploy"].member
}

# Deploy may only act as the API runtime account, not every account in the project.
resource "google_service_account_iam_member" "deploy_acts_as_api" {
  service_account_id = google_service_account.sa["api"].name
  role               = "roles/iam.serviceAccountUser"
  member             = google_service_account.sa["deploy"].member
}

resource "google_secret_manager_secret_iam_member" "api_reads_key" {
  secret_id = google_secret_manager_secret.api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_service_account.sa["api"].member
}

resource "google_storage_bucket_iam_member" "training_data" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = google_service_account.sa["training"].member
}

# MLflow (tracking server and artifact bucket) lives in the shared project,
# managed by the portfolio-infra repository. It grants the api and training
# accounts below access there.

# GitHub Workload Identity Federation: no service account keys.

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  # Only tokens from this repository are accepted.
  attribute_condition = "assertion.repository == '${var.github_repo}'"
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "github_impersonates_deploy" {
  service_account_id = google_service_account.sa["deploy"].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}

# Monthly budget that unlinks billing once spend reaches it. Shared module
# from portfolio-infra, pinned to a commit.
module "budget_guard" {
  source          = "git::https://github.com/juandsep/portfolio-infra.git//modules/budget-guard?ref=a46ece80773fa45aeeb0bd3109b2268ca05949ce"
  project_id      = var.project_id
  region          = var.region
  billing_account = var.billing_account
  amount_usd      = var.monthly_budget_usd
  source_bucket   = google_storage_bucket.data.name
}

# Values for the GitHub repository variables (see README).

output "github_variables" {
  value = {
    GCP_PROJECT_ID    = var.project_id
    GCP_REGION        = var.region
    GCP_ARTIFACT_REPO = google_artifact_registry_repository.images.repository_id
    GCP_WIF_PROVIDER  = google_iam_workload_identity_pool_provider.github.name
    GCP_DEPLOY_SA     = google_service_account.sa["deploy"].email
    GCP_RUNTIME_SA    = google_service_account.sa["api"].email
  }
}

output "data_bucket" {
  value = google_storage_bucket.data.name
}

# Hand these to portfolio-infra so it can grant MLflow access.
output "mlflow_clients" {
  value = {
    api      = google_service_account.sa["api"].email
    training = google_service_account.sa["training"].email
  }
}
