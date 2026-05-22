"""Pydantic-typed platform config loaded from platform.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class DataPlaneConfig(BaseModel):
    dataset: str = "motionlabs/fineweb-ultra-mini"
    max_records: int = 200
    tokenizer_cache: str = "artifacts/tokenizer/tokenizer.json"
    vocab_size: int = 8000
    batch_size: int = 8
    seq_len: int = 64


class TrainingSection(BaseModel):
    base_preset: Literal["local-default", "medium", "large", "xl", "256L-base"] = "local-default"
    num_phases: int = 3
    steps_per_phase: int = 50
    learning_rate: float = 3e-4
    shard_fraction: float = 0.10
    seed: int = 42
    # Optimizations (opt-in)
    gradient_checkpointing: bool = False
    mixed_precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    cpu_offload_frozen: bool = False
    disable_shard_mask: bool = False
    end_to_end: bool = False


class PersonalizationConfig(BaseModel):
    principles_file: str = "principles.yaml"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    epochs: int = 2
    learning_rate: float = 1e-4
    merge_at_packaging: bool = True


class HardwareConfig(BaseModel):
    prefer_quant: Literal["fp32", "fp16", "int8", "auto"] = "auto"
    max_memory_gb: Optional[float] = None  # if set, used instead of detected free memory
    allow_partial_fallback: bool = True


class ServingConfigSection(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    partial_blocks: Optional[int] = None


class RegistrySection(BaseModel):
    root: str = "artifacts/registry"


class PlatformConfig(BaseModel):
    data_plane: DataPlaneConfig = Field(default_factory=DataPlaneConfig)
    training: TrainingSection = Field(default_factory=TrainingSection)
    personalization: PersonalizationConfig = Field(default_factory=PersonalizationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    serving: ServingConfigSection = Field(default_factory=ServingConfigSection)
    registry: RegistrySection = Field(default_factory=RegistrySection)


def load_platform_config(path: str | Path | None = None) -> PlatformConfig:
    if path is None:
        return PlatformConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    return PlatformConfig.model_validate(raw)
