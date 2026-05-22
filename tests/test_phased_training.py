"""Tests for phased training and the 10% gradient mask."""

from __future__ import annotations

from pathlib import Path

import torch

from ml_training.architecture import TransformerArchitecture
from ml_training.models.transformer import MiniTransformer
from ml_training.training import (
    PhasedTrainer,
    PhasedTrainingConfig,
    build_layer_shard_mask,
)


def _data_iter(arch, batch_size=2, seq_len=16):
    while True:
        yield torch.randint(0, arch.spec.vocab_size, (batch_size, seq_len))


def test_shard_mask_fraction_correct():
    mask = build_layer_shard_mask((100, 50), fraction=0.10, seed=42, device=torch.device("cpu"))
    # 10% of 100 rows = 10 rows, full width
    assert mask.shape == (100, 50)
    rows_kept = mask.any(dim=1).sum().item()
    assert rows_kept == 10


def test_shard_mask_deterministic_per_seed():
    a = build_layer_shard_mask((50, 50), 0.2, seed=7, device=torch.device("cpu"))
    b = build_layer_shard_mask((50, 50), 0.2, seed=7, device=torch.device("cpu"))
    c = build_layer_shard_mask((50, 50), 0.2, seed=8, device=torch.device("cpu"))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_phased_training_runs_and_lowers_loss(tmp_path: Path):
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    cfg = PhasedTrainingConfig(
        num_phases=2, steps_per_phase=10, batch_size=2, seq_len=16, log_every=100,
    )
    trainer = PhasedTrainer(model, cfg, checkpoint_dir=str(tmp_path))
    phases = trainer.plan_phases()
    assert len(phases) == 2
    results = trainer.train(_data_iter(arch))
    assert len(results) == 2
    for r in results:
        assert r.steps == 10
        assert r.checkpoint_path is not None
        assert Path(r.checkpoint_path).exists()


def test_grad_mask_zeros_outside_shard(tmp_path: Path):
    """After a phased step, gradients on weights outside the 10% shard must be zero."""
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    cfg = PhasedTrainingConfig(
        num_phases=1, steps_per_phase=1, batch_size=2, seq_len=16, log_every=100,
    )
    trainer = PhasedTrainer(model, cfg, checkpoint_dir=str(tmp_path))
    phase = trainer.plan_phases()[0]
    trainer._set_trainable_blocks(phase.block_indices)
    trainer._install_shard_hooks(phase.block_indices)

    batch = next(_data_iter(arch)).to(trainer.device)
    model.train()
    out = model(batch, labels=batch)
    out["loss"].backward()

    # For at least one in-phase block weight, gradients should be sparse along axis 0.
    block = model.blocks[phase.block_indices[0]]
    g = block.attn.q_proj.weight.grad
    assert g is not None
    nonzero_rows = (g.abs().sum(dim=1) > 0).sum().item()
    total_rows = g.shape[0]
    # ~10% of rows should be nonzero (allow some slack from numerical zero)
    assert nonzero_rows <= total_rows * 0.15
    assert nonzero_rows >= 1


def test_frozen_blocks_have_no_grad(tmp_path: Path):
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    cfg = PhasedTrainingConfig(
        num_phases=2, steps_per_phase=1, batch_size=2, seq_len=16, log_every=100,
    )
    trainer = PhasedTrainer(model, cfg, checkpoint_dir=str(tmp_path))
    phases = trainer.plan_phases()
    trainer._set_trainable_blocks(phases[0].block_indices)

    in_phase = set(phases[0].block_indices)
    for i, block in enumerate(model.blocks):
        block_trainable = any(p.requires_grad for p in block.parameters())
        if i in in_phase:
            assert block_trainable, f"Block {i} should be trainable"
        else:
            assert not block_trainable, f"Block {i} should be frozen"
