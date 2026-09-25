# Hard spending cap. A budget alone only notifies, so the budget publishes to
# Pub/Sub and this function unlinks billing once cost reaches the budget.
# ponytail: billing data lags by hours, so spend can pass the budget by a few
# dollars before the cut. Lower monthly_budget_usd for a tighter margin.

resource "google_project_service" "billing_guard_apis" {
  for_each = toset([
    "cloudbilling.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudfunctions.googleapis.com",
    "eventarc.googleapis.com",
    "pubsub.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

resource "google_pubsub_topic" "budget" {
  name       = "budget-alerts"
  depends_on = [google_project_service.billing_guard_apis]
}

resource "google_service_account" "billing_guard" {
  account_id   = "uplift-billing-guard"
  display_name = "Unlinks billing when the budget is exceeded"
}

# Project Billing Manager can unlink billing from this project only.
resource "google_project_iam_member" "billing_guard" {
  project = var.project_id
  role    = "roles/billing.projectManager"
  member  = google_service_account.billing_guard.member
}

data "archive_file" "billing_guard" {
  type        = "zip"
  source_dir  = "${path.module}/billing_guard"
  output_path = "${path.module}/.terraform/billing_guard.zip"
}

resource "google_storage_bucket_object" "billing_guard" {
  name   = "functions/billing_guard-${data.archive_file.billing_guard.output_md5}.zip"
  bucket = google_storage_bucket.data.name
  source = data.archive_file.billing_guard.output_path
}

resource "google_cloudfunctions2_function" "billing_guard" {
  name     = "billing-guard"
  location = var.region

  build_config {
    runtime     = "python312"
    entry_point = "stop_billing"
    source {
      storage_source {
        bucket = google_storage_bucket.data.name
        object = google_storage_bucket_object.billing_guard.name
      }
    }
  }

  service_config {
    max_instance_count    = 1
    available_memory      = "256M"
    service_account_email = google_service_account.billing_guard.email
    environment_variables = {
      PROJECT_ID = var.project_id
    }
  }

  event_trigger {
    trigger_region        = var.region
    event_type            = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic          = google_pubsub_topic.budget.id
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.billing_guard.email
  }

  depends_on = [google_project_service.billing_guard_apis]
}

# The Eventarc trigger calls the function as the guard account.
resource "google_project_iam_member" "billing_guard_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = google_service_account.billing_guard.member
}
