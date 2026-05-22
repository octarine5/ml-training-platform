"""Tests for TransformerArchitecture + MiniTransformer."""

from __future__ import annotations

import pytest
import torch

from ml_training.architecture import (
    ArchitectureAnalyzer,
    TransformerArchitecture,
    TransformerSpec,
)
from ml_training.models.transformer import MiniTransformer


def test_local_default_preset_loads():
    arch = TransformerArchitecture.from_preset("local-default")
    assert arch.spec.num_layers == 6
    assert arch.spec.d_model == 128
    assert len(arch.layers) == 6 * 6  # 6 sublayers per block


def test_256l_preset_profilable_no_materialization():
    arch = TransformerArchitecture.from_preset("256L-base")
    assert arch.spec.num_layers == 256
    assert len(arch.layers) == 256 * 6
    # Analytic param count is roughly 822M
    assert arch.parameter_count() > 800_000_000


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        TransformerArchitecture.from_preset("nonexistent-preset")


def test_invalid_head_count_rejected():
    with pytest.raises(ValueError):
        TransformerSpec(num_layers=2, num_heads=3, d_model=128, d_ff=256,
                        vocab_size=100, max_seq_len=32)


def test_minitransformer_forward_shapes():
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    B, T = 2, 16
    ids = torch.randint(0, arch.spec.vocab_size, (B, T))
    out = model(ids, labels=ids)
    assert out["logits"].shape == (B, T, arch.spec.vocab_size)
    assert out["loss"].dim() == 0
    assert out["loss"].item() > 0


def test_minitransformer_causal_mask():
    """Logits at position t should not depend on tokens at positions > t."""
    arch = TransformerArchitecture(TransformerSpec(
        num_layers=2, num_heads=2, d_model=32, d_ff=64,
        vocab_size=100, max_seq_len=16,
    ))
    model = MiniTransformer.from_arch(arch).eval()
    ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits_a = model(ids)["logits"]
        ids2 = ids.clone()
        ids2[0, 6] = (ids2[0, 6] + 1) % 100
        logits_b = model(ids2)["logits"]
    # Position 0..5 logits should be identical (changes at pos 6 cannot influence past)
    diff = (logits_a[0, :6] - logits_b[0, :6]).abs().max().item()
    assert diff < 1e-5


def test_generate_produces_new_tokens():
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    ids = torch.tensor([[1, 2, 3]])
    out = model.generate(ids, max_new_tokens=5, temperature=0)
    assert out.shape == (1, 8)


def test_partial_serving_via_stop_after_block():
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    ids = torch.randint(0, arch.spec.vocab_size, (1, 8))
    with torch.no_grad():
        out_full = model(ids)["logits"]
        out_partial = model(ids, stop_after_block=2)["logits"]
    assert out_full.shape == out_partial.shape
    # Should differ
    assert (out_full - out_partial).abs().max().item() > 0


def test_architecture_analyzer_works_on_transformer_arch():
    arch = TransformerArchitecture.from_preset("local-default")
    analyzer = ArchitectureAnalyzer(arch)
    splits = analyzer.recommend_split_points(4)
    assert len(splits) == 4
    assert splits[0] == 0


def test_block_layer_ids_consistent():
    arch = TransformerArchitecture.from_preset("local-default")
    for b in range(arch.spec.num_layers):
        ids = arch.block_layer_ids(b)
        assert len(ids) == 6
        # All ids should resolve to layers in the arch
        for i in ids:
            assert 0 <= i < len(arch.layers)
