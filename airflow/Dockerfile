# Airflow 3 plus the pipeline. The pipeline gets its own venv (/opt/venv, from
# uv.lock) instead of being installed next to Airflow: its pins (mlflow,
# fastapi, pydantic, ...) must not fight Airflow's constraints. Tasks reach it
# through @task.external_python. Build from the repo root:
#   docker build -f airflow/Dockerfile .
ARG AIRFLOW_VERSION=3.3.2

# Builder: same base as the runtime so the venv links the same Python. causalml
# has no linux/aarch64 wheel, so on arm64 it compiles here.
FROM apache/airflow:slim-${AIRFLOW_VERSION}-python3.12 AS builder
USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/python/bin/python3.12 UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# xgboost pulls ~360 MB of NCCL on Linux, used only for multi-GPU training;
# this image is CPU only.
RUN uv sync --locked --no-install-project --no-install-package nvidia-nccl-cu13
COPY src ./src
# Non-editable: /opt/venv must not point back at /app/src.
RUN uv sync --locked --no-editable --no-install-package nvidia-nccl-cu13

FROM apache/airflow:slim-${AIRFLOW_VERSION}-python3.12
ARG AIRFLOW_VERSION

# xgboost needs OpenMP.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# The slim image has no Postgres driver for the metadata DB. Airflow's own
# constraints pin it; the airflow pin stops pip from moving Airflow itself.
RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" apache-airflow-providers-postgres \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"

COPY --from=builder /opt/venv /opt/venv
