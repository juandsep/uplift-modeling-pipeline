import numpy as np
import pandas as pd
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
    monkeypatch.setattr(config, "FEATURES_PATH", None)
    serving.get_model.cache_clear()

    metrics = run(n_samples=2000, learners=["t_xgb", "x_xgb"])
    assert set(metrics) == {"qini", "auuc", "ate", "mean_uplift"}

    mlflow_client = MlflowClient(tracking_uri=tracking_uri)
    versions = mlflow_client.search_model_versions(
        f"name = '{config.REGISTERED_MODEL}'"
    )
    assert len(versions) == 1
    runs = mlflow_client.search_runs(
        [mlflow_client.get_experiment_by_name(config.EXPERIMENT_NAME).experiment_id]
    )
    [parent] = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = {r.data.params["learner"]: r for r in runs if r is not parent}
    assert set(children) == {"t_xgb", "x_xgb"}
    assert all(
        r.data.tags["mlflow.parentRunId"] == parent.info.run_id
        for r in children.values()
    )
    best = parent.data.params["best_learner"]
    assert best == max(children, key=lambda n: parent.data.metrics[f"{n}_qini"])
    assert metrics["qini"] == parent.data.metrics[f"{best}_qini"]
    assert versions[0].run_id == children[best].info.run_id
    assert parent.data.tags["registered_version"] == str(versions[0].version)
    parent_files = {a.path for a in mlflow_client.list_artifacts(parent.info.run_id)}
    assert {"qini_curves.png", "qini_curves.csv"} <= parent_files
    for r in children.values():
        files = {a.path for a in mlflow_client.list_artifacts(r.info.run_id)}
        assert "qini_curve.png" in files
        assert len(r.outputs.model_outputs) == 1
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


def test_train_on_x5_features_with_nulls(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path}/m.db"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(config, "API_KEY", "test-key")
    serving.get_model.cache_clear()
    rng = np.random.default_rng(0)
    n = 400
    w = rng.integers(0, 2, n)
    age = pd.array(rng.integers(18, 80, n), dtype="Int64")
    age[rng.uniform(size=n) < 0.2] = pd.NA
    table = pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(n)],
            "treatment": w,
            "y": (rng.uniform(size=n) < 0.6 + 0.05 * w).astype(int),
            "age": age,
            "spend": np.where(rng.uniform(size=n) < 0.1, np.nan, rng.gamma(2, 50, n)),
        }
    )
    path = tmp_path / "client_features.parquet"
    table.to_parquet(path)

    metrics = run(features_path=str(path), learners=["t_xgb"])

    assert set(metrics) == {"qini", "auuc", "ate", "mean_uplift"}
    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.search_model_versions(f"name = '{config.REGISTERED_MODEL}'")[0]
    params = client.get_run(version.run_id).data.params
    assert params["dataset"] == "x5"
    assert params["n_rows"] == str(n)
    assert params["n_features"] == "2"
    assert params["features_path"] == str(path)

    monkeypatch.setattr(
        config, "MODEL_URI", f"models:/{config.REGISTERED_MODEL}/{version.version}"
    )
    records = [{"age": None, "spend": 30.0}, {"age": 40, "spend": 12.5}]
    resp = TestClient(serving.app).post(
        "/predict", json={"records": records}, headers=HEADERS
    )
    assert resp.status_code == 200
    assert len(resp.json()["uplift"]) == 2


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
