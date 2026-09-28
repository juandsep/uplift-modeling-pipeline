"""Inference API. Serves the model at MODEL_URI from the MLflow registry."""

import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

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

# Loaded once per instance, before it takes traffic. None: /predict answers 503.
_model: PyFuncModel | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _model
    _model = None
    try:
        config.assert_model_uri_is_pinned(config.MODEL_URI)
        # The shared MLflow server is IAM-only.
        config.init_mlflow()
        _model = mlflow.pyfunc.load_model(config.MODEL_URI)
    except Exception:
        # Start anyway: /health stays up, /ready and /predict answer 503.
        logger.exception("model not loaded: refusing to serve predictions")
    yield


def get_model() -> PyFuncModel | None:
    return _model


app = FastAPI(title="Uplift API", lifespan=lifespan)

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if get_model() is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model not loaded"
        )
    return {"status": "ready"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(require_api_key)],
)
def predict(req: PredictRequest) -> PredictResponse:
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service unavailable",
        )
    schema = model.metadata.get_input_schema()
    if schema is not None:
        features = set(schema.input_names())
        missing = sorted({k for r in req.records for k in features - r.keys()})
        unknown = sorted({k for r in req.records for k in r.keys() - features})
        if missing or unknown:
            # Feature names are not secret to a caller holding the API key.
            keys = [f"missing {k}" for k in missing] + [f"unknown {k}" for k in unknown]
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"features do not match the model: {keys[:10]}",
            )
    empty = [i for i, r in enumerate(req.records) if all(v is None for v in r.values())]
    if empty:
        # Some nulls are fine (the model handles NaN); all null carries no signal.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"records with all features null: {empty[:10]}",
        )
    try:
        uplift = model.predict(pd.DataFrame(req.records))
    except (MlflowException, KeyError, ValueError, TypeError):
        # Never echo internal errors back to the caller: they leak the tracking
        # URI and stack details. Details go to the log instead.
        logger.exception("prediction failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="invalid input"
        ) from None
    return PredictResponse(uplift=np.asarray(uplift, dtype=float).tolist())
