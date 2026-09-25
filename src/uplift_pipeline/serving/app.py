"""Inference API. Serves the model at MODEL_URI from the MLflow registry."""

from functools import lru_cache

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from pydantic import BaseModel, Field

from uplift_pipeline import config

app = FastAPI(title="Uplift API")


class PredictRequest(BaseModel):
    records: list[dict[str, float]] = Field(min_length=1)


class PredictResponse(BaseModel):
    uplift: list[float]


@lru_cache
def get_model() -> PyFuncModel:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(config.MODEL_URI)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> PredictResponse:
    try:
        uplift = get_model().predict(pd.DataFrame(req.records))
    except (MlflowException, KeyError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PredictResponse(uplift=np.asarray(uplift, dtype=float).tolist())
