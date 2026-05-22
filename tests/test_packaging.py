"""Tests for weight packaging (safetensors + zstd)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml_training.architecture import TransformerArchitecture
from ml_training.models.transformer import MiniTransformer
from ml_training.packaging import (
    WeightPackager,
    dequantize_int8_symmetric,
    quantize_int8_symmetric,
)


@pytest.fixture
def small_model():
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    return arch, model


def test_fp32_round_trip(tmp_path: Path, small_model):
    arch, model = small_model
    pkg = WeightPackager(registry_root=str(tmp_path))
    sd = model.state_dict()
    bundle = pkg.save(sd, arch.spec, arch_preset="local-default", quant="fp32")
    loaded = pkg.load(bundle.bundle_dir)
    for k in sd:
        assert torch.allclose(sd[k], loaded.state_dict[k], atol=1e-6)


def test_fp16_round_trip_within_tolerance(tmp_path: Path, small_model):
    arch, model = small_model
    pkg = WeightPackager(registry_root=str(tmp_path))
    sd = model.state_dict()
    bundle = pkg.save(sd, arch.spec, arch_preset="local-default", quant="fp16")
    loaded = pkg.load(bundle.bundle_dir)
    for k in sd:
        if sd[k].is_floating_point():
            assert (sd[k] - loaded.state_dict[k]).abs().max().item() < 1e-2


def test_int8_round_trip_within_tolerance(tmp_path: Path, small_model):
    arch, model = small_model
    pkg = WeightPackager(registry_root=str(tmp_path))
    sd = model.state_dict()
    bundle = pkg.save(sd, arch.spec, arch_preset="local-default", quant="int8")
    loaded = pkg.load(bundle.bundle_dir)
    # Per-weight relative MAE should be small (<5%)
    for k in sd:
        if sd[k].is_floating_point() and sd[k].dim() >= 2:
            ref = sd[k].abs().mean().item()
            mae = (sd[k] - loaded.state_dict[k]).abs().mean().item()
            assert mae / max(ref, 1e-8) < 0.05, f"{k} rel MAE too large: {mae/max(ref,1e-8)}"


def test_int8_smaller_than_fp32(tmp_path: Path, small_model):
    arch, model = small_model
    pkg = WeightPackager(registry_root=str(tmp_path))
    sd = model.state_dict()
    b32 = pkg.save(sd, arch.spec, quant="fp32")
    b8 = pkg.save(sd, arch.spec, quant="int8")
    assert b8.compressed_size_bytes < b32.compressed_size_bytes * 0.5


def test_quantize_dequantize_symmetric():
    t = torch.randn(64, 64) * 2.5
    qsd, scales = quantize_int8_symmetric({"w": t})
    assert qsd["w"].dtype == torch.int8
    out = dequantize_int8_symmetric(qsd, scales)
    assert out["w"].dtype == torch.float32
    assert (t - out["w"]).abs().max().item() < t.abs().max().item() / 60  # within ~1/127 scale


def test_manifest_contains_source_links(tmp_path: Path, small_model):
    arch, model = small_model
    pkg = WeightPackager(registry_root=str(tmp_path))
    bundle = pkg.save(
        model.state_dict(),
        arch.spec,
        quant="fp32",
        source_uris={"dataset": "motionlabs/fineweb-ultra-mini", "principles": "principles.yaml"},
        base_hash="deadbeef",
        principles_hash="cafebabe",
    )
    m = bundle.manifest
    assert m["source_uris"]["dataset"] == "motionlabs/fineweb-ultra-mini"
    assert m["base_hash"] == "deadbeef"
    assert m["principles_hash"] == "cafebabe"
    assert "tensors" in m and len(m["tensors"]) > 0


def test_list_bundles(tmp_path: Path, small_model):
    arch, model = small_model
    pkg = WeightPackager(registry_root=str(tmp_path))
    pkg.save(model.state_dict(), arch.spec, quant="fp32")
    pkg.save(model.state_dict(), arch.spec, quant="int8")
    bundles = pkg.list_bundles()
    assert len(bundles) == 2
