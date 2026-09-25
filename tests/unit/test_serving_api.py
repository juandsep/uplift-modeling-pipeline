"""Serving guards: auth, payload limits, and error-message hygiene."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from uplift_pipeline import config
from uplift_pipeline.serving import app as serving

API_KEY = "test-key"  # pragma: allowlist secret
HEADERS = {"X-API-Key": API_KEY}


class StubModel:
    """Minimal stand-in for the MLflow pyfunc model."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def predict(self, model_input):
        if self.error is not None:
            raise self.error
        return np.array([0.1] * len(model_input))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", API_KEY)
    monkeypatch.setattr(serving, "get_model", lambda: StubModel())
    return TestClient(serving.app)


def test_health_needs_no_key(client):
    assert client.get("/health").status_code == 200


def test_predict_rejects_missing_key(client):
    assert client.post("/predict", json={"records": [{"x": 1.0}]}).status_code == 401


def test_predict_rejects_wrong_key(client):
    resp = client.post(
        "/predict", json={"records": [{"x": 1.0}]}, headers={"X-API-Key": "nope"}
    )
    assert resp.status_code == 401


def test_predict_fails_closed_without_configured_key(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "")
    assert client.post("/predict", json={"records": [{"x": 1.0}]}).status_code == 503


def test_predict_rejects_too_many_records(client):
    payload = {"records": [{"x": 1.0}] * (config.MAX_RECORDS + 1)}
    assert client.post("/predict", json=payload, headers=HEADERS).status_code == 422


def test_predict_rejects_oversized_body(client):
    payload = {"records": [{"x": 1.0}], "pad": "0" * (config.MAX_BODY_BYTES + 1)}
    resp = client.post("/predict", json=payload, headers=HEADERS)
    assert resp.status_code == 413


def test_predict_succeeds_with_key(client):
    resp = client.post(
        "/predict", json={"records": [{"x": 1.0}, {"x": 2.0}]}, headers=HEADERS
    )
    assert resp.status_code == 200
    assert len(resp.json()["uplift"]) == 2


def test_internal_error_does_not_leak_schema(client, monkeypatch):
    leaked = "x1_informative"
    monkeypatch.setattr(serving, "get_model", lambda: StubModel(KeyError(leaked)))
    resp = client.post("/predict", json={"records": [{"x": 1.0}]}, headers=HEADERS)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "invalid input"
    assert leaked not in resp.text
