"""Tests for model architecture definition and analysis."""

from __future__ import annotations

import numpy as np
import pytest

from ml_training.architecture import ArchitectureAnalyzer, LayerConfig, ModelArchitecture


class TestLayerConfig:
    """Tests for LayerConfig."""

    def test_memory_bytes_float32(self) -> None:
        layer = LayerConfig(layer_id=0, num_nodes=64, cardinality=128)
        assert layer.memory_bytes == 64 * 128 * 4

    def test_memory_bytes_float16(self) -> None:
        layer = LayerConfig(layer_id=0, num_nodes=64, cardinality=128, dtype="float16")
        assert layer.memory_bytes == 64 * 128 * 2

    def test_memory_bytes_int8(self) -> None:
        layer = LayerConfig(layer_id=0, num_nodes=64, cardinality=128, dtype="int8")
        assert layer.memory_bytes == 64 * 128 * 1

    def test_flops(self) -> None:
        layer = LayerConfig(layer_id=0, num_nodes=64, cardinality=128, compute_intensity=1.5)
        expected = 2.0 * 64 * 128 * 1.5
        assert layer.flops == expected


class TestModelArchitecture:
    """Tests for ModelArchitecture."""

    def test_add_layer(self) -> None:
        arch = ModelArchitecture(name="test")
        layer = arch.add_layer(num_nodes=32, cardinality=64)
        assert layer.layer_id == 0
        assert len(arch.layers) == 1

    def test_build_validates_connectivity(self) -> None:
        arch = ModelArchitecture(name="test")
        arch.add_layer(num_nodes=32, cardinality=64)
        arch.add_layer(num_nodes=16, cardinality=32)
        arch.build()  # Should pass

    def test_build_rejects_mismatched_dims(self) -> None:
        arch = ModelArchitecture(name="test")
        arch.add_layer(num_nodes=32, cardinality=64)
        arch.add_layer(num_nodes=16, cardinality=50)  # Mismatch: 50 != 32
        with pytest.raises(ValueError, match="cardinality"):
            arch.build()

    def test_build_requires_layers(self) -> None:
        arch = ModelArchitecture(name="empty")
        with pytest.raises(ValueError, match="at least one layer"):
            arch.build()

    def test_total_parameters(self, small_architecture: ModelArchitecture) -> None:
        expected = 32 * 16 + 16 * 8 + 8 * 4
        assert small_architecture.total_parameters == expected

    def test_total_memory_bytes(self, small_architecture: ModelArchitecture) -> None:
        expected = (32 * 16 + 16 * 8 + 8 * 4) * 4  # float32
        assert small_architecture.total_memory_bytes == expected

    def test_summary_length(self, small_architecture: ModelArchitecture) -> None:
        summary = small_architecture.summary()
        assert len(summary) == 3


class TestArchitectureAnalyzer:
    """Tests for ArchitectureAnalyzer."""

    def test_compute_per_layer(self, small_architecture: ModelArchitecture) -> None:
        analyzer = ArchitectureAnalyzer(small_architecture)
        flops = analyzer.compute_per_layer()
        assert len(flops) == 3
        assert all(f > 0 for f in flops)

    def test_compute_ratios_sum_to_one(self, small_architecture: ModelArchitecture) -> None:
        analyzer = ArchitectureAnalyzer(small_architecture)
        ratios = analyzer.compute_ratios()
        assert abs(sum(ratios) - 1.0) < 1e-6

    def test_memory_ratios_sum_to_one(self, small_architecture: ModelArchitecture) -> None:
        analyzer = ArchitectureAnalyzer(small_architecture)
        ratios = analyzer.memory_ratios()
        assert abs(sum(ratios) - 1.0) < 1e-6

    def test_recommend_split_points_single_gpu(
        self, small_architecture: ModelArchitecture
    ) -> None:
        analyzer = ArchitectureAnalyzer(small_architecture)
        splits = analyzer.recommend_split_points(1)
        assert splits == [0]

    def test_recommend_split_points_multi_gpu(
        self, small_architecture: ModelArchitecture
    ) -> None:
        analyzer = ArchitectureAnalyzer(small_architecture)
        splits = analyzer.recommend_split_points(2)
        assert len(splits) == 2
        assert splits[0] == 0

    def test_bottleneck_layers(self, small_architecture: ModelArchitecture) -> None:
        analyzer = ArchitectureAnalyzer(small_architecture)
        bottlenecks = analyzer.bottleneck_layers(top_k=2)
        assert len(bottlenecks) == 2
        # First bottleneck should have highest FLOPs
        assert bottlenecks[0].flops >= bottlenecks[1].flops
