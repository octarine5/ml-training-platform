"""FastAPI wrapper around `LocalServer` exposing the same surface as the
stdlib `http.server` handler in `local_server.py`.

Why a separate module?
----------------------
`local_server.py` was written to depend only on the Python stdlib so the
training/eval/packaging stack can be unit-tested without FastAPI installed.
This wrapper is loaded by the deployed image (see `deploy/Dockerfile`):

    uvicorn ml_training.serving.local_server:app

For backwards-compat with that CMD, `local_server.py` re-exports `app` from
this module (lazy import — only triggered when uvicorn imports the attribute,
so test runs without FastAPI installed are unaffected).

Endpoints
---------
- POST /generate          — same dataclass shape as `LocalServer.generate`
- GET  /health            — liveness
- GET  /ready             — readiness (model + tokenizer attached)
- GET  /registry/bundles  — list bundles (optional `?alias=` resolves alias)
- GET  /metrics           — Prometheus exposition (added by observability)
- GET  /docs, /openapi.json — Swagger UI / schema

Bootstrap
---------
On startup we *try* to load a bundle. Source order:
1. env var `ML_TRAINING_BUNDLE_DIR`
2. env var `ML_TRAINING_BUNDLE_ID` (resolved via `WeightRegistry`)
3. `WeightRegistry.get("production")`
4. `WeightRegistry.latest()`

If none of those resolve, the server still boots — `/health` returns ok,
`/ready` returns 503, and `/generate` returns 503 with a clear message. This
matches how k8s `readinessProbe` is wired in `deploy/k8s/`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ml_training.serving.local_server import (
    GenerationRequest,
    LocalServer,
    ServingConfig,
    ServingMode,
)

log = logging.getLogger("ml_training.serving.fastapi_app")


# ---------------------------------------------------------------- pydantic I/O


class GenerateIn(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(32, ge=1, le=512)
    temperature: float = Field(1.0, gt=0.0, le=2.0)
    top_k: Optional[int] = Field(50, ge=1, le=1000)


class GenerateOut(BaseModel):
    text: str
    token_ids: list[int]
    truncated: bool
    blocks_used: int
    quant: str
    extra: dict


class BundleInfo(BaseModel):
    bundle_id: str
    arch_preset: str
    quant: str
    created_at: str
    compressed_size_bytes: int


# ---------------------------------------------------------------- bootstrap


def _bootstrap_server() -> tuple[Optional[LocalServer], Optional[str]]:
    """Try to load a bundle into a `LocalServer`. Returns (server, error)."""
    mode_str = os.environ.get("ML_TRAINING_MODE", "full")
    partial_blocks = os.environ.get("ML_TRAINING_PARTIAL_BLOCKS")
    cfg = ServingConfig(
        mode=ServingMode(mode_str),
        partial_blocks=int(partial_blocks) if partial_blocks else None,
    )

    bundle_dir = os.environ.get("ML_TRAINING_BUNDLE_DIR")
    bundle_id = os.environ.get("ML_TRAINING_BUNDLE_ID")
    tokenizer_path = os.environ.get(
        "ML_TRAINING_TOKENIZER", "artifacts/tokenizer/tokenizer.json"
    )

    if not bundle_dir:
        try:
            from ml_training.control_plane.registry import WeightRegistry

            reg = WeightRegistry(root=os.environ.get(
                "ML_TRAINING_REGISTRY_ROOT", "artifacts/registry"
            ))
            entry = None
            if bundle_id:
                entry = reg.get(bundle_id)
            else:
                try:
                    entry = reg.get("production")
                except KeyError:
                    entry = reg.latest()
            if entry is not None:
                bundle_dir = str(entry.bundle_dir)
        except Exception as e:  # noqa: BLE001
            return None, f"registry lookup failed: {e}"

    if not bundle_dir:
        return None, "no bundle configured (set ML_TRAINING_BUNDLE_DIR or register one)"

    try:
        server = LocalServer(cfg).load(bundle_dir)
    except Exception as e:  # noqa: BLE001
        return None, f"bundle load failed: {e}"

    try:
        from ml_training.tokenization import BPETokenizer, TokenizerConfig

        tk = BPETokenizer(TokenizerConfig(cache_path=tokenizer_path)).load()
        server.attach_tokenizer(tk)
    except Exception as e:  # noqa: BLE001
        return server, f"tokenizer attach failed: {e}"

    return server, None


# ---------------------------------------------------------------- app factory


def create_app() -> FastAPI:
    app = FastAPI(
        title="ml-training-platform",
        version="0.2.0",
        description="FastAPI wrapper for LocalServer; partial-N / int8 modes supported.",
    )

    state: dict = {"server": None, "boot_error": None}

    @app.on_event("startup")
    def _on_startup() -> None:
        srv, err = _bootstrap_server()
        state["server"] = srv
        state["boot_error"] = err
        if err:
            log.warning("server bootstrap incomplete: %s", err)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "ml-training-platform"}

    @app.get("/ready")
    def ready() -> dict:
        srv: Optional[LocalServer] = state["server"]
        deps = {
            "model": bool(srv and srv.model is not None),
            "tokenizer": bool(srv and srv.tokenizer is not None),
        }
        ok = all(deps.values())
        if not ok:
            raise HTTPException(
                status_code=503,
                detail={"ready": False, "deps": deps, "boot_error": state["boot_error"]},
            )
        return {"ready": True, "deps": deps}

    @app.post("/generate", response_model=GenerateOut)
    def generate(body: GenerateIn) -> GenerateOut:
        srv: Optional[LocalServer] = state["server"]
        if srv is None or srv.model is None or srv.tokenizer is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "model not loaded",
                    "boot_error": state["boot_error"],
                },
            )
        try:
            req = GenerationRequest(
                prompt=body.prompt,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
            )
            resp = srv.generate(req)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail={"error": str(e)})
        return GenerateOut(**asdict(resp))

    @app.get("/registry/bundles")
    def list_bundles(alias: Optional[str] = Query(None)) -> list[BundleInfo]:
        try:
            from ml_training.control_plane.registry import WeightRegistry

            reg = WeightRegistry(root=os.environ.get(
                "ML_TRAINING_REGISTRY_ROOT", "artifacts/registry"
            ))
            if alias:
                try:
                    entry = reg.get(alias)
                    entries = [entry]
                except KeyError:
                    entries = []
            else:
                entries = reg.list_entries()
            return [
                BundleInfo(
                    bundle_id=e.bundle_id,
                    arch_preset=e.arch_preset,
                    quant=e.quant,
                    created_at=e.created_at,
                    compressed_size_bytes=e.compressed_size_bytes,
                )
                for e in entries
            ]
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail={"error": str(e)})

    # ---- optional: install Prometheus + OTel middleware if available
    try:
        import sys as _sys
        from pathlib import Path as _Path

        # monitoring/ lives outside src/; ensure repo root is importable
        _repo = _Path(__file__).resolve().parents[3]
        if str(_repo) not in _sys.path:
            _sys.path.insert(0, str(_repo))
        from monitoring.observability import install_observability  # type: ignore

        install_observability(app, service_name="ml-training-platform")
    except Exception as e:  # noqa: BLE001
        log.info("observability not installed: %s", e)

    return app


# Module-level `app` for `uvicorn ml_training.serving.fastapi_app:app`
app = create_app()
