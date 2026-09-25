from fastapi.testclient import TestClient
from mlflow import MlflowClient

from uplift_pipeline import config
from uplift_pipeline.data import load_training_data
from uplift_pipeline.serving import app as serving
from uplift_pipeline.train import run

HEADERS = {"X-API-Key": "test-key"}


def test_train_register_and_serve(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path}/m.db"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(config, "API_KEY", "test-key")
    monkeypatch.setattr(config, "ALLOW_UNPINNED_MODEL", False)
    serving.get_model.cache_clear()

    metrics = run(n_samples=2000)
    assert set(metrics) == {"qini", "auuc"}

    versions = MlflowClient(tracking_uri=tracking_uri).search_model_versions(
        f"name = '{config.REGISTERED_MODEL}'"
    )
    assert len(versions) == 1
    monkeypatch.setattr(
        config, "MODEL_URI", f"models:/{config.REGISTERED_MODEL}/{versions[0].version}"
    )

    df, features = load_training_data(n_samples=10, seed=1)
    records = df[features].head(3).to_dict("records")
    client = TestClient(serving.app)
    ok = client.post("/predict", json={"records": records}, headers=HEADERS)
    assert ok.status_code == 200
    assert len(ok.json()["uplift"]) == 3

    bad = client.post("/predict", json={"records": [{"nope": 1.0}]}, headers=HEADERS)
    assert bad.status_code == 422
    assert bad.json()["detail"] == "invalid input"

    unauth = client.post("/predict", json={"records": records})
    assert unauth.status_code == 401


def test_serving_refuses_floating_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    monkeypatch.setattr(config, "API_KEY", "test-key")
    monkeypatch.setattr(config, "ALLOW_UNPINNED_MODEL", False)
    monkeypatch.setattr(config, "MODEL_URI", "models:/uplift-model/latest")
    serving.get_model.cache_clear()

    client = TestClient(serving.app, raise_server_exceptions=False)
    resp = client.post("/predict", json={"records": [{"x": 1.0}]}, headers=HEADERS)

    assert resp.status_code == 503
    assert resp.json()["detail"] == "service unavailable"
    assert "uplift" not in resp.text
