"""Runtime config: model pinning and the MLflow identity token."""

import io
import os
import urllib.error
import urllib.request

import pytest

from uplift_pipeline import config


@pytest.mark.parametrize(
    "uri",
    [
        "models:/uplift-model/1",
        "models:/uplift-model/42",
        "runs:/abc123/model",
        "/tmp/local-model",
    ],
)
def test_pinned_uris_accepted(uri, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_UNPINNED_MODEL", False)
    config.assert_model_uri_is_pinned(uri)


@pytest.mark.parametrize(
    "uri",
    ["models:/uplift-model/latest", "models:/uplift-model/staging"],
)
def test_floating_aliases_rejected(uri, monkeypatch):
    monkeypatch.setattr(config, "ALLOW_UNPINNED_MODEL", False)
    with pytest.raises(RuntimeError, match="floating alias"):
        config.assert_model_uri_is_pinned(uri)


def test_dev_can_opt_out(monkeypatch):
    monkeypatch.setattr(config, "ALLOW_UNPINNED_MODEL", True)
    config.assert_model_uri_is_pinned("models:/uplift-model/latest")


def test_model_version_env_builds_pinned_uri(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "7")
    monkeypatch.delenv("MODEL_URI", raising=False)
    assert config.resolve_model_uri() == f"models:/{config.REGISTERED_MODEL}/7"


def test_model_uri_env_wins(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "7")
    monkeypatch.setenv("MODEL_URI", "runs:/abc/model")
    assert config.resolve_model_uri() == "runs:/abc/model"


def test_mlflow_token_minted_on_gce(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    calls = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout):
        calls.append(req)
        return Resp(b"fresh-token")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config.refresh_mlflow_token("https://mlflow.example.run.app")

    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "fresh-token"
    assert calls[0].get_header("Metadata-flavor") == "Google"
    assert "audience=https%3A%2F%2Fmlflow.example.run.app" in calls[0].full_url


def test_mlflow_token_kept_off_gce(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "pasted")

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("no metadata server")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config.refresh_mlflow_token("https://mlflow.example.run.app")
    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "pasted"


def test_mlflow_token_skipped_for_local_store(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("metadata server must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    config.refresh_mlflow_token("sqlite:///mlflow.db")
