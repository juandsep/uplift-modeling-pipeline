"""Causal models, wrapped as MLflow pyfunc so serving stays model-agnostic."""

import numpy as np
import pandas as pd
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier
from mlflow.pyfunc import PythonModel
from xgboost import XGBClassifier, XGBRegressor

XGB = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
# Order fixes each learner's color in the Qini plots.
LEARNERS = ("t_xgb", "x_xgb", "s_xgb")


class UpliftModel(PythonModel):
    """Meta-learner with XGBoost base models. Predicts P(y|treated) - P(y|control).

    learner: t_xgb (T-learner), x_xgb (X-learner) or s_xgb (S-learner).
    """

    def __init__(self, features: list[str], learner: str = "t_xgb") -> None:
        self.features = features
        self.learner = learner
        self.propensity: float | None = None
        if learner == "t_xgb":
            self.model = BaseTClassifier(learner=XGBClassifier(**XGB))
        elif learner == "x_xgb":
            self.model = BaseXClassifier(
                outcome_learner=XGBClassifier(**XGB),
                effect_learner=XGBRegressor(**XGB),
            )
        elif learner == "s_xgb":
            self.model = BaseSClassifier(learner=XGBClassifier(**XGB))
        else:
            raise ValueError(f"unknown learner {learner!r}; expected one of {LEARNERS}")

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        # Nulls (None, pd.NA) become NaN, which XGBoost treats as missing.
        return df[self.features].to_numpy(dtype=float, na_value=np.nan)

    def fit(self, df: pd.DataFrame) -> "UpliftModel":
        X, w = self._matrix(df), df["treatment"].to_numpy()
        kwargs = {}
        if self.learner == "x_xgb":
            # X5 is a randomized trial (~50/50), so the true propensity is the
            # constant treatment rate: fitting a propensity model would only add
            # noise. An observational dataset would need a fitted one.
            self.propensity = float(w.mean())
            kwargs["p"] = np.full(len(X), self.propensity)
        self.model.fit(X=X, treatment=w, y=df["y"].to_numpy(), **kwargs)
        return self

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        X = self._matrix(model_input)
        # getattr: versions registered before this attribute existed are T-learners.
        p = getattr(self, "propensity", None)
        kwargs = {} if p is None else {"p": np.full(len(X), p)}
        return self.model.predict(X, verbose=False, **kwargs)[:, 0]
