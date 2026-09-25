"""Train, evaluate, register the model. Run: python -m uplift_pipeline.train"""

import mlflow
from sklearn.model_selection import train_test_split

from uplift_pipeline import config
from uplift_pipeline.data import load_training_data
from uplift_pipeline.evaluation import uplift_metrics
from uplift_pipeline.models import UpliftModel


def run(n_samples: int = 10_000, seed: int = 42) -> dict[str, float]:
    df, features = load_training_data(n_samples, seed)
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=df["treatment"]
    )
    model = UpliftModel(features).fit(train_df)
    metrics = uplift_metrics(
        test_df["y"], test_df["treatment"], model.predict(None, test_df)
    )

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params({"n_samples": n_samples, "seed": seed, "learner": "t_xgb"})
        mlflow.log_metrics(metrics)
        mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            input_example=test_df[features].head(5),
            registered_model_name=config.REGISTERED_MODEL,
        )
    return metrics


if __name__ == "__main__":
    print(run())
