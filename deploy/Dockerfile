FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# system deps (bake in only what the service truly needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# install python deps first for layer caching
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e .
# uvicorn + fastapi power the /generate HTTP surface in deployed mode;
# prometheus + opentelemetry hooks come from monitoring/observability.py.
RUN pip install \
      "uvicorn[standard]>=0.30" \
      "fastapi>=0.110" \
      "prometheus-client>=0.20" \
      "opentelemetry-api>=1.25" \
      "opentelemetry-sdk>=1.25"

# copy the rest
COPY src ./src
COPY monitoring/observability.py ./src/observability.py
COPY platform.yaml principles.yaml ./

# non-root
RUN useradd --create-home --uid 10001 svc && chown -R svc:svc /app
USER svc

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=2s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Module path: src/ml_training/serving/local_server.py exposes `app` (FastAPI)
# wrapping LocalServer + POST /generate. Two workers — model load is heavy
# enough that more replicas should come from k8s HPA, not in-pod workers.
CMD ["uvicorn", "ml_training.serving.local_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
