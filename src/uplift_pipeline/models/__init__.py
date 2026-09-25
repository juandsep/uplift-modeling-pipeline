"""Causal models, wrapped as MLflow pyfunc so serving stays model-agnostic."""

import numpy as np
import pandas as pd
from causalml.inference.meta import BaseTClassifier
from mlflow.pyfunc import PythonModel
from xgboost import XGBClassifier


class UpliftModel(PythonModel):
    """T-learner with XGBoost base learners. Predicts P(y|treated) - P(y|control)."""

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.model = BaseTClassifier(
            learner=XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05)
        )

    def fit(self, df: pd.DataFrame) -> "UpliftModel":
        self.model.fit(
            X=df[self.features].to_numpy(),
            treatment=df["treatment"].to_numpy(),
            y=df["y"].to_numpy(),
        )
        return self

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        X = model_input[self.features].to_numpy()
        return self.model.predict(X, verbose=False)[:, 0]
