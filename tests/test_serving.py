"""Tests for the local serving server (full, partial-N, int8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml_training.architecture import TransformerArchitecture
from ml_training.models.transformer import MiniTransformer
from ml_training.packaging import WeightPackager
from ml_training.serving import (
    GenerationRequest,
    LocalServer,
    ServingConfig,
    ServingMode,
)
from ml_training.tokenization import BPETokenizer, TokenizerConfig


@pytest.fixture
def small_bundle_and_tokenizer(tmp_path: Path):
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    pkg = WeightPackager(registry_root=str(tmp_path / "registry"))
    bundle = pkg.save(model.state_dict(), arch.spec, arch_preset="local-default", quant="fp32")

    # Train a tiny tokenizer
    tk = BPETokenizer(TokenizerConfig(vocab_size=200, cache_path=str(tmp_path / "tok.json")))
    tk.train_from_iterator(iter([
        "the quick brown fox jumps over the lazy dog",
        "machine learning transforms data into predictions",
        "attention is all you need",
    ] * 20))
    return arch, bundle, tk


def test_full_serving_generates(small_bundle_and_tokenizer):
    _, bundle, tk = small_bundle_and_tokenizer
    server = LocalServer(ServingConfig(mode=ServingMode.FULL))
    server.load(bundle.bundle_dir).attach_tokenizer(tk)
    resp = server.generate(GenerationRequest(prompt="hello", max_tokens=4))
    assert len(resp.token_ids) == len(tk.encode("hello")) + 4 or len(resp.token_ids) >= 4
    assert resp.truncated is False
    assert resp.blocks_used == 6


def test_partial_serving_drops_blocks(small_bundle_and_tokenizer):
    _, bundle, tk = small_bundle_and_tokenizer
    server = LocalServer(ServingConfig(mode=ServingMode.PARTIAL, partial_blocks=2))
    server.load(bundle.bundle_dir).attach_tokenizer(tk)
    resp = server.generate(GenerationRequest(prompt="hi", max_tokens=3, temperature=0))
    assert resp.truncated is True
    assert resp.blocks_used == 2
    assert server.model is not None
    assert len(server.model.blocks) == 2


def test_int8_serving_round_trip(small_bundle_and_tokenizer, tmp_path: Path):
    arch, _, tk = small_bundle_and_tokenizer
    model = MiniTransformer.from_arch(arch)
    pkg = WeightPackager(registry_root=str(tmp_path / "registry2"))
    bundle = pkg.save(model.state_dict(), arch.spec, quant="int8")
    server = LocalServer(ServingConfig(mode=ServingMode.INT8))
    server.load(bundle.bundle_dir).attach_tokenizer(tk)
    resp = server.generate(GenerationRequest(prompt="test", max_tokens=4, temperature=0))
    assert resp.quant == "int8"
    assert resp.truncated is False


def test_generation_response_serializes_json(small_bundle_and_tokenizer):
    from dataclasses import asdict
    _, bundle, tk = small_bundle_and_tokenizer
    server = LocalServer(ServingConfig(mode=ServingMode.FULL))
    server.load(bundle.bundle_dir).attach_tokenizer(tk)
    resp = server.generate(GenerationRequest(prompt="x", max_tokens=2))
    # Must be JSON-serializable for HTTP
    blob = json.dumps(asdict(resp))
    parsed = json.loads(blob)
    assert "text" in parsed
    assert "token_ids" in parsed
    assert "truncated" in parsed


def test_requires_loaded_model_before_generate():
    server = LocalServer(ServingConfig())
    with pytest.raises(RuntimeError):
        server.generate(GenerationRequest(prompt="x"))
