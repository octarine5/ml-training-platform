"""Tests for outlier-trimmed checkpoint averaging."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ml_training.training import CheckpointAverager, average_state_dicts


def test_average_state_dicts_no_outliers():
    sds = [
        {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
    ]
    avg = average_state_dicts(sds)
    assert torch.allclose(avg["w"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_average_state_dicts_rejects_outliers():
    # Three values close, one wild outlier — outlier should be excluded.
    sds = [
        {"w": torch.tensor([1.0])},
        {"w": torch.tensor([1.1])},
        {"w": torch.tensor([0.9])},
        {"w": torch.tensor([100.0])},   # outlier
    ]
    avg = average_state_dicts(sds, z_threshold=1.5)
    # Should not be the naive mean (~25.75); should be close to 1.0
    assert avg["w"].item() < 5.0
    naive = (1.0 + 1.1 + 0.9 + 100.0) / 4
    assert abs(avg["w"].item() - naive) > 1.0


def test_average_state_dicts_preserves_dtype():
    sds = [{"w": torch.tensor([1.0, 2.0], dtype=torch.float16)}] * 3
    avg = average_state_dicts(sds)
    assert avg["w"].dtype == torch.float16


def test_average_state_dicts_single():
    sd = [{"w": torch.tensor([[1.0, 2.0]])}]
    avg = average_state_dicts(sd)
    assert torch.allclose(avg["w"], sd[0]["w"])


def test_average_state_dicts_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        average_state_dicts([])


def test_checkpoint_averager_loads_from_disk(tmp_path: Path):
    # Synthesize phase checkpoints with a manifest
    for i in range(3):
        sd = {"w": torch.full((4, 4), float(i + 1))}
        torch.save({"state_dict": sd, "phase_id": i, "block_indices": [i]}, tmp_path / f"phase_{i}.pt")
    # Manifest in same order
    with (tmp_path / "phases.jsonl").open("w") as f:
        for i in range(3):
            f.write(json.dumps({"checkpoint": str(tmp_path / f"phase_{i}.pt")}) + "\n")

    avg = CheckpointAverager(checkpoint_dir=str(tmp_path))
    paths = avg.list_phase_checkpoints()
    assert len(paths) == 3

    averaged, stats = avg.converge()
    # mean of [1,2,3] is 2.0; std is small enough that none should be rejected at z=2
    assert torch.allclose(averaged["w"], torch.full((4, 4), 2.0), atol=0.01)
    assert stats["num_checkpoints_averaged"] == 3
