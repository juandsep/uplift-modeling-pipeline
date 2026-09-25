"""Train, evaluate, register the model. Run: python -m uplift_pipeline.train"""

import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

from uplift_pipeline import config
from uplift_pipeline.data import load_training_data, load_x5_features
from uplift_pipeline.evaluation import uplift_metrics
from uplift_pipeline.models import UpliftModel


def registered_version(run_id: str) -> str | None:
    """Version the registry assigned to the model logged by this run."""
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(
        f"name = '{config.REGISTERED_MODEL}' and run_id = '{run_id}'"
    )
    return versions[0].version if versions else None


def run(
    n_samples: int = 10_000, seed: int = 42, features_path: str | None = None
) -> dict[str, float]:
    """Train on the X5 table at features_path (or FEATURES_PATH), else synthetic.

    n_samples only applies to synthetic data.
    """
    features_path = features_path or config.FEATURES_PATH
    if features_path:
        df, features = load_x5_features(features_path)
        dataset: dict[str, str | int] = {
            "dataset": "x5",
            "features_path": features_path,
        }
    else:
        df, features = load_training_data(n_samples, seed)
        dataset = {"dataset": "synthetic", "n_samples": n_samples}
    # Stratify on both so each split keeps the arm sizes and the base rates.
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=df[["treatment", "y"]]
    )
    model = UpliftModel(features).fit(train_df)
    metrics = uplift_metrics(
        test_df["y"], test_df["treatment"], model.predict(None, test_df)
    )

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    version = None
    with mlflow.start_run() as active_run:
        mlflow.log_params(
            {
                **dataset,
                "n_rows": len(df),
                "n_features": len(features),
                "seed": seed,
                "learner": "t_xgb",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            input_example=test_df[features].head(5),
            registered_model_name=config.REGISTERED_MODEL,
        )
        version = registered_version(active_run.info.run_id)
        if version is not None:
            mlflow.set_tag("registered_version", version)

    if version is not None:
        print(
            f"registered {config.REGISTERED_MODEL} version {version} "
            f"(serve it with MODEL_VERSION={version})"
        )
    return metrics


if __name__ == "__main__":
    print(run())
