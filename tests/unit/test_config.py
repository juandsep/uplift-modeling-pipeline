"""Model pinning: serving must not follow a floating registry alias."""

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
