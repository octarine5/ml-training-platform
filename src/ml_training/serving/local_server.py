"""Local model server.

Supports three serving modes:
- "full": all blocks loaded
- "partial-N": only the first N transformer blocks loaded (rest dropped). The final
  LayerNorm + LM head still run on the truncated stack. This lets us deploy a model
  that does not fully fit. No tokens are "missing" in the output sequence - quality
  drops measurably and we surface a `truncated=True` flag.
- "int8": full model, weights stored quantized in the bundle, dequantized on load.

HTTP interface uses stdlib http.server (no extra deps) - POST /generate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import torch

from ml_training.architecture import TransformerSpec
from ml_training.models.transformer import MiniTransformer
from ml_training.packaging import WeightPackager


class ServingMode(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    INT8 = "int8"


@dataclass
class ServingConfig:
    mode: ServingMode = ServingMode.FULL
    partial_blocks: Optional[int] = None  # used when mode=PARTIAL; how many to keep
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


@dataclass
class GenerationRequest:
    prompt: str
    max_tokens: int = 32
    temperature: float = 1.0
    top_k: Optional[int] = 50


@dataclass
class GenerationResponse:
    text: str
    token_ids: list[int]
    truncated: bool = False
    blocks_used: int = 0
    quant: str = "fp32"
    extra: dict = field(default_factory=dict)


class LocalServer:
    """Loads a WeightBundle into a MiniTransformer and serves generations."""

    def __init__(self, config: ServingConfig) -> None:
        self.config = config
        self.device = config.resolve_device()
        self.model: Optional[MiniTransformer] = None
        self.spec: Optional[TransformerSpec] = None
        self.manifest: dict = {}
        self.tokenizer = None  # set via attach_tokenizer
        self._blocks_used: int = 0

    # ---------------------------------------------------------------- load

    def load(self, bundle_dir: str | Path) -> "LocalServer":
        pkg = WeightPackager()
        bundle = pkg.load(bundle_dir)
        spec = TransformerSpec(**bundle.manifest["arch_spec"])
        self.spec = spec
        self.manifest = bundle.manifest
        model = MiniTransformer(spec)
        # If partial, drop the tail blocks BEFORE loading state to avoid loading them.
        if self.config.mode == ServingMode.PARTIAL:
            n = self.config.partial_blocks or max(1, spec.num_layers // 2)
            n = min(n, spec.num_layers)
            model.blocks = torch.nn.ModuleList(list(model.blocks)[:n])
            # Filter state_dict to only include surviving blocks
            kept_sd = {}
            prefix_drop_thresholds = set()
            for k, v in bundle.state_dict.items():
                if k.startswith("blocks."):
                    idx = int(k.split(".")[1])
                    if idx >= n:
                        prefix_drop_thresholds.add(idx)
                        continue
                kept_sd[k] = v
            model.load_state_dict(kept_sd, strict=False)
            self._blocks_used = n
        else:
            model.load_state_dict(bundle.state_dict, strict=False)
            self._blocks_used = spec.num_layers

        model.to(self.device).eval()
        self.model = model
        return self

    def attach_tokenizer(self, tokenizer) -> "LocalServer":
        self.tokenizer = tokenizer
        return self

    # ---------------------------------------------------------------- generate

    def generate(self, req: GenerationRequest) -> GenerationResponse:
        if self.model is None or self.spec is None:
            raise RuntimeError("Server not loaded; call .load(bundle_dir) first.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not attached; call .attach_tokenizer(tok).")

        ids = self.tokenizer.encode(req.prompt)
        if not ids:
            ids = [0]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        out_ids = self.model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )[0].tolist()
        new_ids = out_ids[len(ids):]
        text = self.tokenizer.decode(new_ids)
        return GenerationResponse(
            text=text,
            token_ids=out_ids,
            truncated=(self.config.mode == ServingMode.PARTIAL),
            blocks_used=self._blocks_used,
            quant=self.manifest.get("quant", "fp32"),
            extra={
                "prompt_token_count": len(ids),
                "generated_token_count": len(new_ids),
                "bundle_id": self.manifest.get("bundle_id"),
            },
        )

    # ---------------------------------------------------------------- HTTP

    def serve_http(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Block forever, serving POST /generate."""
        server_ref = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return  # silence

            def do_POST(self):  # noqa: N802
                if self.path != "/generate":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(raw)
                    req = GenerationRequest(**payload)
                    resp = server_ref.generate(req)
                    body = json.dumps(asdict(resp)).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as e:  # noqa: BLE001
                    err = json.dumps({"error": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)

        httpd = HTTPServer((host, port), _Handler)
        print(f"Serving on http://{host}:{port}/generate (mode={self.config.mode.value})")
        httpd.serve_forever()


# ---------------------------------------------------------------- FastAPI shim
#
# The deployed image runs `uvicorn ml_training.serving.local_server:app` (see
# deploy/Dockerfile). For unit tests we don't want FastAPI as a hard import,
# so `app` is resolved lazily via __getattr__: the FastAPI wrapper lives in
# `ml_training.serving.fastapi_app` and is only imported when an attribute
# named `app` is accessed on this module.


def __getattr__(name):  # PEP 562
    if name == "app":
        from ml_training.serving.fastapi_app import app as _app
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
