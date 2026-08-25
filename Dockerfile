# syntax=docker/dockerfile:1

# =====================================================================
# Multi-stage build for the S&P 500 prediction stack.
# NOTE: base is python:3.12-slim (not 3.11) because pyproject.toml pins
# requires-python = ">=3.12" and uv.lock is resolved for CPython 3.12.
# =====================================================================

# ---------------------------------------------------------------------
# Stage 1: builder — compile/install dependencies into /opt/venv
# ---------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

COPY pyproject.toml uv.lock README.md ./

# Export the locked dependency set and install it (compiled wheels kept).
RUN uv export --frozen --format requirements-txt --no-dev -o requirements.txt \
    && python -m venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# ---------------------------------------------------------------------
# Stage 2: runtime — slim image with prebuilt venv + application code
# ---------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp1 is required by scikit-learn/lightgbm-style native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ ./src/
COPY main.py pyproject.toml README.md ./

RUN mkdir -p /app/data /app/reports /app/models \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Default entrypoint: the Streamlit dashboard.
# The API service overrides this with uvicorn (see docker-compose.yml).
CMD ["python", "-m", "streamlit", "run", "src/sp500_stock_prediction/dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
