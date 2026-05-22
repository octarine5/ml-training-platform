"""Tests for control plane: config, registry, sources, hardware planner."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from ml_training.architecture import TransformerArchitecture, TransformerSpec
from ml_training.control_plane import (
    HardwarePool,
    PlatformConfig,
    SourceLinks,
    WeightRegistry,
    load_platform_config,
    plan_deployment,
)
from ml_training.control_plane.hardware import DeviceSpec
from ml_training.control_plane.sources import hash_file
from ml_training.models.transformer import MiniTransformer


def test_platform_config_defaults():
    cfg = PlatformConfig()
    assert cfg.data_plane.dataset == "motionlabs/fineweb-ultra-mini"
    assert cfg.training.base_preset == "local-default"
    assert cfg.personalization.lora_rank == 8


def test_load_platform_config_from_yaml(tmp_path: Path):
    p = tmp_path / "platform.yaml"
    p.write_text(yaml.safe_dump({
        "training": {"num_phases": 5, "steps_per_phase": 100},
        "personalization": {"lora_rank": 16},
    }))
    cfg = load_platform_config(p)
    assert cfg.training.num_phases == 5
    assert cfg.training.steps_per_phase == 100
    assert cfg.personalization.lora_rank == 16
    # Untouched defaults remain
    assert cfg.data_plane.dataset == "motionlabs/fineweb-ultra-mini"


def test_load_platform_config_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_platform_config(tmp_path / "nonexistent.yaml")


def test_weight_registry_round_trip(tmp_path: Path):
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    reg = WeightRegistry(root=str(tmp_path))
    bundle = reg.packager.save(model.state_dict(), arch.spec, arch_preset="local-default", quant="fp32")
    entries = reg.list_entries()
    assert len(entries) == 1
    fetched = reg.get(bundle.bundle_id)
    assert fetched.bundle_id == bundle.bundle_id


def test_weight_registry_alias(tmp_path: Path):
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    reg = WeightRegistry(root=str(tmp_path))
    bundle = reg.packager.save(model.state_dict(), arch.spec, quant="fp32")
    reg.promote(bundle.bundle_id, alias="production")
    fetched = reg.get("production")
    assert fetched.bundle_id == bundle.bundle_id


def test_registry_get_unknown_raises(tmp_path: Path):
    reg = WeightRegistry(root=str(tmp_path))
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_source_links_serialization():
    src = SourceLinks(
        dataset_uri="motionlabs/fineweb-ultra-mini",
        tokenizer_uri="/tmp/tok.json",
        principles_uri="/tmp/principles.yaml",
        base_model_uri=None,
    )
    d = src.as_dict()
    assert d["dataset"] == "motionlabs/fineweb-ultra-mini"
    assert d["tokenizer"] == "/tmp/tok.json"
    assert "base_model" not in d


def test_hash_file_stable(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    h1 = hash_file(p)
    h2 = hash_file(p)
    assert h1 == h2
    assert hash_file(tmp_path / "nonexistent") is None


def test_hardware_pool_detect_returns_some_device():
    spec = HardwarePool.detect()
    assert spec.kind in ("cuda", "mps", "cpu")
    assert spec.memory_gb > 0


def test_plan_deployment_fp32_fits_local_default():
    spec = TransformerArchitecture.from_preset("local-default").spec
    fake_device = DeviceSpec(name="fake", kind="cuda", memory_gb=24.0,
                             supports_fp16=True, supports_int8=True)
    plan = plan_deployment(spec, device=fake_device, prefer_quant="auto")
    assert plan.fits is True
    assert plan.serving_mode == "full"


def test_plan_deployment_falls_back_to_partial_when_too_big():
    spec = TransformerArchitecture.from_preset("256L-base").spec
    tight_device = DeviceSpec(name="tight", kind="cpu", memory_gb=2.0,
                              supports_fp16=False, supports_int8=True)
    plan = plan_deployment(spec, device=tight_device, prefer_quant="auto",
                           max_memory_gb=0.5, allow_partial_fallback=True)
    # Should not fit at any quant within 0.5 GB; partial fallback engages.
    assert plan.fits is False
    assert plan.serving_mode == "partial"
    assert plan.partial_blocks is not None and plan.partial_blocks < 256


def test_plan_deployment_int8_when_fp32_too_big():
    spec = TransformerArchitecture.from_preset("256L-base").spec
    medium_device = DeviceSpec(name="medium", kind="cpu", memory_gb=2.0,
                               supports_fp16=False, supports_int8=True)
    plan = plan_deployment(spec, device=medium_device, prefer_quant="auto", max_memory_gb=1.5)
    # ~822M params; fp32=3.2GB, int8=0.8GB → int8 fits in 1.5GB budget
    assert plan.quant == "int8"
    assert plan.fits is True
