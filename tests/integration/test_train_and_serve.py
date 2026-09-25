from fastapi.testclient import TestClient

from uplift_pipeline import config
from uplift_pipeline.data import load_training_data
from uplift_pipeline.serving import app as serving
from uplift_pipeline.train import run

HEADERS = {"X-API-Key": "test-key"}


def test_train_register_and_serve(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    monkeypatch.setattr(config, "API_KEY", "test-key")
    serving.get_model.cache_clear()

    metrics = run(n_samples=2000)
    assert set(metrics) == {"qini", "auuc"}

    df, features = load_training_data(n_samples=10, seed=1)
    records = df[features].head(3).to_dict("records")
    client = TestClient(serving.app)
    ok = client.post("/predict", json={"records": records}, headers=HEADERS)
    assert ok.status_code == 200
    assert len(ok.json()["uplift"]) == 3

    bad = client.post("/predict", json={"records": [{"nope": 1.0}]}, headers=HEADERS)
    assert bad.status_code == 422

    unauth = client.post("/predict", json={"records": records})
    assert unauth.status_code == 401
