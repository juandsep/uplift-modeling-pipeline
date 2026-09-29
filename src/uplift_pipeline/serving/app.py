"""Inference API. Serves the model at MODEL_URI from the MLflow registry."""

import logging
import secrets
import threading
import time
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
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from uplift_pipeline import config

logger = logging.getLogger(__name__)

# Per instance: Prometheus scrapes each process and sums across them.
LATENCY = Histogram(
    "uplift_request_seconds",
    "Request latency",
    ["path", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
BATCH = Histogram(
    "uplift_batch_records",
    "Records per /predict request",
    buckets=(1, 10, 50, 100, 250, 500, 1000),
)
UPLIFT = Histogram(
    "uplift_prediction",
    "Predicted uplift per record",
    buckets=(-0.1, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05, 0.1, 0.2),
)
# The histogram drops _sum because its buckets go negative (the sum can fall);
# a gauge carries it so the dashboard can plot the mean uplift.
UPLIFT_SUM = Gauge("uplift_prediction_value_sum", "Sum of predicted uplift")
PREDICT_ERRORS = Counter(
    "uplift_predict_errors", "Rejected /predict requests", ["reason"]
)
MODEL = Gauge("uplift_model_info", "1 for the model this instance serves", ["uri"])
# Fixed label set: raw paths from scanners would blow up the series count.
_PATHS = {"/predict", "/ready", "/health", "/metrics"}

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
        MODEL.labels(uri=config.MODEL_URI).set(1)
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


class TokenBucket:
    """Requests per second with a burst of the same size. Thread-safe."""

    def __init__(self, rate: float) -> None:
        self.rate = rate
        self.tokens = rate
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens < 1:
                return False
            self.tokens -= 1
            return True


# ponytail: one bucket per instance, not per caller: there is one API key per
# environment. The global cap is RATE_LIMIT_RPS x Cloud Run max-instances; move
# to per-key buckets (or API Gateway) if callers get their own keys.
_bucket = TokenBucket(config.RATE_LIMIT_RPS)


def rate_limit() -> None:
    if config.RATE_LIMIT_RPS > 0 and not _bucket.take():
        PREDICT_ERRORS.labels("rate_limited").inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": "1"},
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


@app.middleware("http")
async def record_latency(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path if request.url.path in _PATHS else "other"
    LATENCY.labels(path, str(response.status_code)).observe(time.perf_counter() - start)
    return response


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


@app.get("/metrics")
def metrics() -> Response:
    # Unauthenticated like /health; on Cloud Run, IAM still guards it.
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
    # Auth first: unauthenticated calls must not drain the bucket.
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
def predict(req: PredictRequest) -> PredictResponse:
    model = get_model()
    if model is None:
        PREDICT_ERRORS.labels("no_model").inc()
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
            PREDICT_ERRORS.labels("feature_mismatch").inc()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"features do not match the model: {keys[:10]}",
            )
    empty = [i for i, r in enumerate(req.records) if all(v is None for v in r.values())]
    if empty:
        # Some nulls are fine (the model handles NaN); all null carries no signal.
        PREDICT_ERRORS.labels("all_null").inc()
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
        PREDICT_ERRORS.labels("model_error").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="invalid input"
        ) from None
    values: np.ndarray = np.asarray(uplift, dtype=float)
    BATCH.observe(len(values))
    UPLIFT_SUM.inc(float(values.sum()))
    for v in values:
        UPLIFT.observe(v)
    return PredictResponse(uplift=values.tolist())
