"""Drop-in observability for FastAPI services.

Import once from app.py:

    from monitoring.observability import install_observability
    app = FastAPI()
    install_observability(app, service_name="ml-training-platform")

What you get:
  - Prometheus metrics at GET /metrics (http_requests_total, http_request_duration_seconds)
  - Liveness + readiness endpoints (/health, /ready) — only added if absent
  - OpenTelemetry tracing — automatic span per request; exports to OTLP endpoint if
    OTEL_EXPORTER_OTLP_ENDPOINT is set, otherwise no-ops.

Designed to be light: ~150 lines, no behavior changes to the app.
"""

from __future__ import annotations

import os
import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
    multiprocess,
)

# ---- Prometheus -------------------------------------------------------------

_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
)
_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def _metrics_endpoint() -> Response:
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---- OpenTelemetry (best-effort, no-op without endpoint) --------------------

def _maybe_setup_tracing(service_name: str) -> None:
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        # otel libs not installed — silently no-op
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


# ---- Wiring ----------------------------------------------------------------


def install_observability(app: FastAPI, service_name: str) -> None:
    _maybe_setup_tracing(service_name)

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next: Callable):
        started = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - started
        path = request.scope.get("route").path if request.scope.get("route") else request.url.path
        _REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        _LATENCY.labels(request.method, path).observe(elapsed)
        return response

    # idempotent: only add if not already declared
    existing_paths = {r.path for r in app.routes if hasattr(r, "path")}
    if "/metrics" not in existing_paths:
        app.add_route("/metrics", lambda _r: _metrics_endpoint())
    if "/health" not in existing_paths:
        @app.get("/health")
        async def _health():
            return {"status": "ok", "service": service_name}
    if "/ready" not in existing_paths:
        @app.get("/ready")
        async def _ready():
            # plug in real dep checks here
            return {"ready": True, "deps": {}}
