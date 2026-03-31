"""Tests for distributed training strategies."""

from __future__ import annotations

import numpy as np
import pytest

from ml_training.architecture import ModelArchitecture
from ml_training.distribution import (
    DataParallelism,
    GradientSyncManager,
    PipelineParallelism,
    RatioBasedDistributor,
)


@pytest.fixture
def four_layer_arch() -> ModelArchitecture:
    """Build a 4-layer architecture for distribution tests."""
    arch = ModelArchitecture(name="dist-test")
    arch.add_layer(num_nodes=64, cardinality=128)
    arch.add_layer(num_nodes=32, cardinality=64)
    arch.add_layer(num_nodes=16, cardinality=32)
    arch.add_layer(num_nodes=8, cardinality=16, activation="sigmoid")
    arch.build()
    return arch


class TestPipelineParallelism:
    """Tests for PipelineParallelism."""

    def test_partition_creates_stages(self, four_layer_arch: ModelArchitecture) -> None:
        pp = PipelineParallelism(four_layer_arch, num_gpus=2)
        stages = pp.partition()
        assert len(stages) >= 1
        # All layers should be assigned
        total_layers = sum(len(s.layers) for s in stages)
        assert total_layers == len(four_layer_arch.layers)

    def test_forward_schedule(self, four_layer_arch: ModelArchitecture) -> None:
        pp = PipelineParallelism(four_layer_arch, num_gpus=2)
        pp.partition()
        schedule = pp.forward_schedule(num_microbatches=4)
        assert len(schedule) > 0
        # Each entry should be a list of (stage_id, microbatch_id) tuples
        for step in schedule:
            for stage_id, mb_id in step:
                assert 0 <= stage_id < len(pp.stages)
                assert 0 <= mb_id < 4

    def test_pipeline_bubble_ratio(self, four_layer_arch: ModelArchitecture) -> None:
        pp = PipelineParallelism(four_layer_arch, num_gpus=2)
        pp.partition()
        ratio = pp.pipeline_bubble_ratio(num_microbatches=8)
        assert 0.0 <= ratio < 1.0

    def test_pipeline_bubble_ratio_zero_microbatches(
        self, four_layer_arch: ModelArchitecture
    ) -> None:
        pp = PipelineParallelism(four_layer_arch, num_gpus=2)
        pp.partition()
        assert pp.pipeline_bubble_ratio(0) == 1.0


class TestDataParallelism:
    """Tests for DataParallelism."""

    def test_split_batch_even(self) -> None:
        dp = DataParallelism(num_replicas=4)
        sizes = dp.split_batch(100)
        assert sum(sizes) == 100
        assert len(sizes) == 4

    def test_split_batch_uneven(self) -> None:
        dp = DataParallelism(num_replicas=3)
        sizes = dp.split_batch(10)
        assert sum(sizes) == 10

    def test_split_data(self, rng: np.random.Generator) -> None:
        dp = DataParallelism(num_replicas=3)
        data = rng.standard_normal((30, 5))
        splits = dp.split_data(data)
        assert len(splits) == 3
        total = sum(s.shape[0] for s in splits)
        assert total == 30

    def test_effective_batch_size(self) -> None:
        dp = DataParallelism(num_replicas=8)
        assert dp.effective_batch_size(32) == 256


class TestRatioBasedDistributor:
    """Tests for RatioBasedDistributor."""

    def test_distribute_fits_all_layers(self, four_layer_arch: ModelArchitecture) -> None:
        memory_limits = [100_000, 100_000]  # Large enough
        dist = RatioBasedDistributor(four_layer_arch, memory_limits)
        assignment = dist.distribute()
        total_layers = sum(len(layers) for layers in assignment.values())
        assert total_layers == len(four_layer_arch.layers)

    def test_distribute_raises_on_insufficient_memory(
        self, four_layer_arch: ModelArchitecture
    ) -> None:
        memory_limits = [1, 1]  # Too small
        dist = RatioBasedDistributor(four_layer_arch, memory_limits)
        with pytest.raises(ValueError, match="does not fit"):
            dist.distribute()


class TestGradientSyncManager:
    """Tests for GradientSyncManager."""

    def test_allreduce(self, rng: np.random.Generator) -> None:
        grads = [rng.standard_normal((10, 5)) for _ in range(4)]
        manager = GradientSyncManager(num_workers=4, sync_mode="allreduce")
        result = manager.allreduce(grads)
        expected = np.mean(np.stack(grads), axis=0)
        np.testing.assert_allclose(result, expected)

    def test_ring_reduce(self, rng: np.random.Generator) -> None:
        grads = [rng.standard_normal((10, 5)) for _ in range(4)]
        manager = GradientSyncManager(num_workers=4, sync_mode="ring")
        result = manager.ring_reduce(grads)
        expected = np.mean(np.stack(grads), axis=0)
        np.testing.assert_allclose(result, expected)

    def test_sync_dispatches_correctly(self, rng: np.random.Generator) -> None:
        grads = [rng.standard_normal((5, 3)) for _ in range(2)]
        for mode in ("allreduce", "ring", "tree"):
            manager = GradientSyncManager(num_workers=2, sync_mode=mode)
            result = manager.sync(grads)
            assert result.shape == (5, 3)

    def test_invalid_sync_mode(self) -> None:
        with pytest.raises(ValueError, match="Unknown sync_mode"):
            GradientSyncManager(num_workers=2, sync_mode="invalid")

    def test_empty_gradients(self) -> None:
        manager = GradientSyncManager(num_workers=2)
        with pytest.raises(ValueError, match="No gradients"):
            manager.allreduce([])

    def test_estimated_comm_time(self) -> None:
        manager = GradientSyncManager(num_workers=4, sync_mode="allreduce")
        t = manager.estimated_comm_time(param_count=1_000_000)
        assert t > 0
