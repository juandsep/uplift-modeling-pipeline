"""Inference API. Serves the model at MODEL_URI from the MLflow registry."""

import logging
import secrets
from collections.abc import Awaitable, Callable
from functools import lru_cache

import mlflow
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from pydantic import BaseModel, Field

from uplift_pipeline import config

logger = logging.getLogger(__name__)

app = FastAPI(title="Uplift API")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Depends(_api_key_header)) -> None:
    """Authenticate the caller. Fails closed when no key is configured."""
    if not config.API_KEY:
        logger.error("API_KEY is not set: refusing to serve predictions")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service unavailable",
        )
    if key is None or not secrets.compare_digest(key, config.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key"
        )


@app.middleware("http")
async def reject_oversized_body(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject oversized bodies before anything parses them."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            too_big = int(declared) > config.MAX_BODY_BYTES
        except ValueError:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "invalid content-length"},
            )
        if too_big:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "payload too large"},
            )
    return await call_next(request)


class PredictRequest(BaseModel):
    # null = missing feature value (e.g. unknown age); the model handles NaN.
    records: list[dict[str, float | None]] = Field(
        min_length=1, max_length=config.MAX_RECORDS
    )


class PredictResponse(BaseModel):
    uplift: list[float]


@lru_cache
def get_model() -> PyFuncModel:
    config.assert_model_uri_is_pinned(config.MODEL_URI)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.pyfunc.load_model(config.MODEL_URI)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(require_api_key)],
)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        uplift = get_model().predict(pd.DataFrame(req.records))
    except RuntimeError:
        # The configured model is refused (see config.assert_model_uri_is_pinned).
        logger.exception("refusing to serve the configured model")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service unavailable",
        ) from None
    except (MlflowException, KeyError, ValueError, TypeError):
        # Never echo internal errors back to the caller: they leak the feature
        # schema and the tracking URI. Details go to the log instead.
        logger.exception("prediction failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="invalid input"
        ) from None
    return PredictResponse(uplift=np.asarray(uplift, dtype=float).tolist())
