"""Package a converged state_dict into a compressed .safetensors.zst bundle.

Bundle layout (a single tar-less directory under registry root):
    artifacts/registry/<bundle_id>/
        weights.safetensors.zst
        manifest.json

Manifest fields:
    bundle_id        sha256 of compressed weights (truncated)
    arch_preset      preset name or "custom"
    arch_spec        full TransformerSpec as dict
    quant            "fp32" | "int8" | "fp16"
    base_hash        sha256 of base weights (before personalization), or None
    principles_hash  sha256 of principles file content, or None
    source_uris      dict of dataset/tokenizer/principles URIs
    created_at       ISO8601 timestamp
    tensors          list of {name, shape, dtype}
    raw_size_bytes   uncompressed safetensors size
    compressed_size_bytes
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import zstandard as zstd
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from ml_training.architecture import TransformerSpec


@dataclass
class WeightBundle:
    bundle_id: str
    bundle_dir: Path
    manifest: dict
    state_dict: dict[str, torch.Tensor]

    @property
    def quant(self) -> str:
        return self.manifest.get("quant", "fp32")

    @property
    def raw_size_bytes(self) -> int:
        return int(self.manifest.get("raw_size_bytes", 0))

    @property
    def compressed_size_bytes(self) -> int:
        return int(self.manifest.get("compressed_size_bytes", 0))


def quantize_int8_symmetric(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Per-tensor symmetric int8 quantization. Returns (quant_dict, scales).

    quant_dict has int8 tensors for all 2-D+ weights; 1-D (norms/biases) pass through.
    """
    qsd: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    for k, v in state_dict.items():
        if v.dim() < 2 or not v.is_floating_point():
            qsd[k] = v.clone()
            continue
        abs_max = v.abs().max().item()
        if abs_max == 0:
            qsd[k] = torch.zeros_like(v, dtype=torch.int8)
            scales[k] = 1.0
            continue
        scale = abs_max / 127.0
        q = torch.clamp(torch.round(v / scale), -127, 127).to(torch.int8)
        qsd[k] = q
        scales[k] = float(scale)
    return qsd, scales


def dequantize_int8_symmetric(
    qsd: dict[str, torch.Tensor],
    scales: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Inverse of quantize_int8_symmetric."""
    out: dict[str, torch.Tensor] = {}
    for k, v in qsd.items():
        if k in scales:
            out[k] = v.to(torch.float32) * scales[k]
        else:
            out[k] = v.clone()
    return out


class WeightPackager:
    """Save and load compressed weight bundles."""

    SAFE_EXT = ".safetensors.zst"

    def __init__(self, registry_root: str = "artifacts/registry", level: int = 10) -> None:
        self.registry_root = Path(registry_root)
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.level = level

    # ---------------------------------------------------------------- save

    def save(
        self,
        state_dict: dict[str, torch.Tensor],
        arch_spec: TransformerSpec,
        arch_preset: str | None = None,
        quant: str = "fp32",
        base_hash: Optional[str] = None,
        principles_hash: Optional[str] = None,
        source_uris: Optional[dict[str, str]] = None,
    ) -> WeightBundle:
        if quant == "int8":
            qsd, scales = quantize_int8_symmetric(state_dict)
            tensors_to_save = qsd
            extra_meta = {"quant_scales": scales}
        elif quant == "fp16":
            tensors_to_save = {k: v.to(torch.float16) if v.is_floating_point() else v.clone()
                               for k, v in state_dict.items()}
            extra_meta = {}
        elif quant == "fp32":
            tensors_to_save = {k: v.clone() for k, v in state_dict.items()}
            extra_meta = {}
        else:
            raise ValueError(f"Unsupported quant: {quant}")

        # Serialize via safetensors (in-memory bytes)
        raw_bytes = st_save(tensors_to_save)

        # Compress with zstd
        cctx = zstd.ZstdCompressor(level=self.level)
        compressed = cctx.compress(raw_bytes)

        bundle_id = hashlib.sha256(compressed).hexdigest()[:16]
        bundle_dir = self.registry_root / bundle_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / ("weights" + self.SAFE_EXT)).write_bytes(compressed)

        manifest = {
            "bundle_id": bundle_id,
            "arch_preset": arch_preset or "custom",
            "arch_spec": asdict(arch_spec),
            "quant": quant,
            "base_hash": base_hash,
            "principles_hash": principles_hash,
            "source_uris": source_uris or {},
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
            "tensors": [
                {"name": k, "shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in tensors_to_save.items()
            ],
            "raw_size_bytes": len(raw_bytes),
            "compressed_size_bytes": len(compressed),
            **extra_meta,
        }
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        return WeightBundle(
            bundle_id=bundle_id,
            bundle_dir=bundle_dir,
            manifest=manifest,
            state_dict=state_dict,
        )

    # ---------------------------------------------------------------- load

    def load(self, bundle_dir: str | Path) -> WeightBundle:
        bundle_dir = Path(bundle_dir)
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        compressed = (bundle_dir / ("weights" + self.SAFE_EXT)).read_bytes()
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(compressed)
        loaded = st_load(raw)

        if manifest.get("quant") == "int8":
            scales = manifest.get("quant_scales", {})
            loaded = dequantize_int8_symmetric(loaded, scales)
        elif manifest.get("quant") == "fp16":
            loaded = {k: v.to(torch.float32) if v.is_floating_point() else v
                      for k, v in loaded.items()}
        return WeightBundle(
            bundle_id=manifest["bundle_id"],
            bundle_dir=bundle_dir,
            manifest=manifest,
            state_dict=loaded,
        )

    def list_bundles(self) -> list[dict]:
        out = []
        for d in sorted(self.registry_root.iterdir()):
            if not d.is_dir():
                continue
            m = d / "manifest.json"
            if m.exists():
                out.append(json.loads(m.read_text()))
        return out
