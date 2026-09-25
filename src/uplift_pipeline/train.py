"""Train, evaluate, register the model. Run: python -m uplift_pipeline.train"""

import time

import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

from uplift_pipeline import config
from uplift_pipeline.data import load_training_data, load_x5_features
from uplift_pipeline.evaluation import COLORS, plot_qini, qini_curve, uplift_metrics
from uplift_pipeline.models import LEARNERS, UpliftModel


def registered_version(run_id: str) -> str | None:
    """Version the registry assigned to the model logged by this run."""
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(
        f"name = '{config.REGISTERED_MODEL}' and run_id = '{run_id}'"
    )
    return versions[0].version if versions else None


def run(
    n_samples: int = 10_000,
    seed: int = 42,
    features_path: str | None = None,
    learners: list[str] | None = None,
) -> dict[str, float]:
    """Train and compare learners on the X5 table at features_path (or
    FEATURES_PATH), else synthetic. Registers the best by Qini, returns its metrics.

    n_samples only applies to synthetic data. learners defaults to LEARNERS.
    MLflow: one parent run, one nested run per learner.
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
    dataset.update(n_rows=len(df), n_features=len(features), seed=seed)
    # Built up front so an unknown learner fails before any training.
    models = {name: UpliftModel(features, name) for name in learners or config.LEARNERS}
    # One split for every learner, so the comparison is fair.
    # Stratify on both so each split keeps the arm sizes and the base rates.
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=df[["treatment", "y"]]
    )
    uplifts, metrics, fit_seconds = {}, {}, {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(train_df)
        fit_seconds[name] = time.perf_counter() - start
        uplifts[name] = model.predict(None, test_df)
        metrics[name] = uplift_metrics(
            test_df["y"], test_df["treatment"], uplifts[name]
        )
    best = max(metrics, key=lambda name: metrics[name]["qini"])
    curve = qini_curve(test_df["y"], test_df["treatment"], uplifts)
    colors = {name: COLORS[LEARNERS.index(name) % len(COLORS)] for name in models}

    config.refresh_mlflow_token(config.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    version = None
    with mlflow.start_run():
        mlflow.log_params(
            {
                **dataset,
                "learners": ",".join(models),
                "best_learner": best,
            }
        )
        mlflow.log_metrics(
            {f"{n}_{k}": m[k] for n, m in metrics.items() for k in ("qini", "auuc")}
        )
        mlflow.log_figure(plot_qini(curve, colors), "qini_curves.png")
        mlflow.log_text(curve.to_csv(index=False), "qini_curves.csv")
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True) as child:
                mlflow.log_params({**dataset, "learner": name})
                mlflow.log_metrics({**metrics[name], "fit_seconds": fit_seconds[name]})
                mlflow.log_figure(
                    plot_qini(curve[["fraction", name, "random"]], colors),
                    "qini_curve.png",
                )
                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=model,
                    input_example=test_df[features].head(5),
                    registered_model_name=config.REGISTERED_MODEL
                    if name == best
                    else None,
                )
                if name == best:
                    version = registered_version(child.info.run_id)
                    if version is not None:
                        mlflow.set_tag("registered_version", version)
        if version is not None:
            mlflow.set_tag("registered_version", version)

    if version is not None:
        print(
            f"registered {config.REGISTERED_MODEL} version {version} ({best}; "
            f"serve it with MODEL_VERSION={version})"
        )
    return metrics[best]


if __name__ == "__main__":
    print(run())
