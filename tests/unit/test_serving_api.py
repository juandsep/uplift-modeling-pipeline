"""Serving guards: auth, payload limits, and error-message hygiene."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from mlflow.models import Model, ModelSignature
from mlflow.types import ColSpec, Schema

from uplift_pipeline import config
from uplift_pipeline.serving import app as serving

API_KEY = "test-key"  # pragma: allowlist secret
HEADERS = {"X-API-Key": API_KEY}


class StubModel:
    """Minimal stand-in for the MLflow pyfunc model."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.metadata = Model(
            signature=ModelSignature(inputs=Schema([ColSpec("double", "x")]))
        )

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


def test_ready_once_the_model_is_loaded(client, monkeypatch):
    assert client.get("/ready").status_code == 200
    monkeypatch.setattr(serving, "get_model", lambda: None)
    assert client.get("/ready").status_code == 503


def test_predict_unavailable_without_model(client, monkeypatch):
    monkeypatch.setattr(serving, "get_model", lambda: None)
    resp = client.post("/predict", json={"records": [{"x": 1.0}]}, headers=HEADERS)
    assert resp.status_code == 503
    assert resp.json()["detail"] == "service unavailable"


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


def test_predict_rejects_all_null_record(client):
    records = [{"x": 1.0}, {"x": None}]
    resp = client.post("/predict", json={"records": records}, headers=HEADERS)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "records with all features null: [1]"


def test_internal_error_does_not_leak_schema(client, monkeypatch):
    leaked = "x1_informative"
    monkeypatch.setattr(serving, "get_model", lambda: StubModel(KeyError(leaked)))
    resp = client.post("/predict", json={"records": [{"x": 1.0}]}, headers=HEADERS)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "invalid input"
    assert leaked not in resp.text


def test_predict_rejects_missing_feature(client):
    resp = client.post("/predict", json={"records": [{}]}, headers=HEADERS)
    assert resp.status_code == 422
    assert resp.json()["detail"] == "features do not match the model: ['missing x']"


def test_predict_rejects_unknown_features(client):
    record = {"x": 1.0} | {f"k{i:02}": 1.0 for i in range(20)}
    resp = client.post("/predict", json={"records": [record]}, headers=HEADERS)
    assert resp.status_code == 422
    keys = [f"unknown k{i:02}" for i in range(10)]
    assert resp.json()["detail"] == f"features do not match the model: {keys}"


def _sample(client, name, **labels):
    """Current value of one Prometheus sample from /metrics."""
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            if all(f'{k}="{v}"' in line for k, v in labels.items()):
                return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_count_predictions_and_rejections(client):
    records = {"records": [{"x": 1.0}, {"x": 2.0}]}
    before = _sample(client, "uplift_prediction_count")
    sum_before = _sample(client, "uplift_prediction_value_sum")
    ok_before = _sample(
        client, "uplift_request_seconds_count", path="/predict", status="200"
    )
    mismatch_before = _sample(
        client, "uplift_predict_errors_total", reason="feature_mismatch"
    )

    assert client.post("/predict", json=records, headers=HEADERS).status_code == 200
    bad = {"records": [{"y": 1.0}]}
    assert client.post("/predict", json=bad, headers=HEADERS).status_code == 422
    client.get("/wp-login.php")

    assert _sample(client, "uplift_prediction_count") == before + 2
    assert _sample(client, "uplift_prediction_value_sum") == pytest.approx(
        sum_before + 0.2
    )
    assert (
        _sample(client, "uplift_request_seconds_count", path="/predict", status="200")
        == ok_before + 1
    )
    assert (
        _sample(client, "uplift_predict_errors_total", reason="feature_mismatch")
        == mismatch_before + 1
    )
    # Unknown paths share one label instead of creating a series each.
    assert 'path="/wp-login.php"' not in client.get("/metrics").text


def test_predict_rate_limited(client, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_RPS", 2.0)
    monkeypatch.setattr(serving, "_bucket", serving.TokenBucket(2.0))
    body = {"records": [{"x": 1.0}]}
    # Rejected keys do not spend tokens.
    for _ in range(3):
        assert client.post("/predict", json=body).status_code == 401
    codes = [
        client.post("/predict", json=body, headers=HEADERS).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]
