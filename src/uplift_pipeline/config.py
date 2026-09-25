"""Runtime settings, read from environment variables."""

import os
import urllib.parse
import urllib.request

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "uplift")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL", "uplift-model")
# Per-client X5 feature table (Parquet). Unset: train on synthetic data.
FEATURES_PATH = os.getenv("FEATURES_PATH") or None

# Serving guardrails. API_KEY is required: the API fails closed without it.
API_KEY = os.getenv("API_KEY", "")
MAX_RECORDS = int(os.getenv("MAX_RECORDS", "1000"))
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(1024 * 1024)))

# Model pinning. Serving refuses a floating alias (`latest`, `staging`, ...)
# unless dev opts out explicitly.
ALLOW_UNPINNED_MODEL = os.getenv("ALLOW_UNPINNED_MODEL", "").strip().lower() in {
    "1",
    "true",
    "yes",
}


def resolve_model_uri() -> str:
    """Pick the model to serve: explicit URI, pinned version, else the alias."""
    explicit = os.getenv("MODEL_URI")
    if explicit:
        return explicit
    version = os.getenv("MODEL_VERSION", "")
    if version:
        return f"models:/{REGISTERED_MODEL}/{version}"
    return f"models:/{REGISTERED_MODEL}/latest"


MODEL_URI = resolve_model_uri()


def is_pinned(uri: str) -> bool:
    """True when the URI cannot be repointed by a new registry write."""
    if not uri.startswith("models:"):
        # runs:/<id>/model and absolute paths are immutable.
        return True
    return uri.rstrip("/").rsplit("/", 1)[-1].isdigit()


def assert_model_uri_is_pinned(uri: str) -> None:
    """Refuse to serve a floating alias: a registry write must not swap a live model."""
    if is_pinned(uri) or ALLOW_UNPINNED_MODEL:
        return
    raise RuntimeError(
        f"MODEL_URI {uri!r} is a floating alias; a registry write could swap the "
        "model under a running service. Set MODEL_VERSION (e.g. MODEL_VERSION=3), "
        "or ALLOW_UNPINNED_MODEL=1 for local dev."
    )


def refresh_mlflow_token(uri: str) -> None:
    """On GCE, mint a fresh identity token for an IAM-protected MLflow server.

    Tokens expire after an hour, so mint right before talking to MLflow.
    Elsewhere, MLFLOW_TRACKING_TOKEN is left as is.
    """
    if not uri.startswith("https://"):
        return
    req = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "service-accounts/default/identity?audience="
        + urllib.parse.quote(uri, safe=""),
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            os.environ["MLFLOW_TRACKING_TOKEN"] = resp.read().decode()
    except OSError:
        pass  # Not on GCE.
