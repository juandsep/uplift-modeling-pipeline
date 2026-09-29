# Cloud Monitoring alerts on the production API, from the built-in Cloud Run
# request metrics: no Prometheus server to run. Staging is not watched.

variable "alert_email" {
  description = "Address that receives the API alerts."
  type        = string
}

resource "google_project_service" "monitoring" {
  service            = "monitoring.googleapis.com"
  disable_on_destroy = false
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "uplift alerts"
  type         = "email"
  labels = {
    email_address = var.alert_email
  }
  depends_on = [google_project_service.monitoring]
}

locals {
  api_filter = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"uplift-api\""
}

resource "google_monitoring_alert_policy" "api_p99" {
  display_name          = "uplift-api p99 latency above 1 s"
  combiner              = "OR"
  notification_channels = [google_monitoring_notification_channel.email.id]

  conditions {
    display_name = "p99 request latency > 1000 ms for 5 min"
    condition_threshold {
      # Measured locally: p99 about 70 ms for 500-record batches.
      filter          = "${local.api_filter} AND metric.type = \"run.googleapis.com/request_latencies\""
      comparison      = "COMPARISON_GT"
      threshold_value = 1000
      duration        = "300s"
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_PERCENTILE_99"
        cross_series_reducer = "REDUCE_MAX"
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "api_5xx" {
  display_name          = "uplift-api 5xx responses"
  combiner              = "OR"
  notification_channels = [google_monitoring_notification_channel.email.id]

  conditions {
    display_name = "more than 5 responses with 5xx in 5 min"
    condition_threshold {
      # Includes the 503 answered when the model failed to load.
      filter          = "${local.api_filter} AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"5xx\""
      comparison      = "COMPARISON_GT"
      threshold_value = 5
      duration        = "0s"
      aggregations {
        alignment_period     = "300s"
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  alert_strategy {
    auto_close = "1800s"
  }
}
