"""Distributed training strategies: pipeline parallelism, data parallelism, gradient sync."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ml_training.architecture import ArchitectureAnalyzer, LayerConfig, ModelArchitecture


@dataclass
class PipelineStage:
    """A stage in the pipeline, holding a subset of layers assigned to one GPU."""

    stage_id: int
    gpu_id: int
    layers: list[LayerConfig]

    @property
    def total_flops(self) -> float:
        return sum(l.flops for l in self.layers)

    @property
    def total_memory(self) -> int:
        return sum(l.memory_bytes for l in self.layers)


class PipelineParallelism:
    """Split model layers across GPUs in pipeline stages."""

    def __init__(self, architecture: ModelArchitecture, num_gpus: int) -> None:
        self.architecture = architecture
        self.num_gpus = num_gpus
        self.stages: list[PipelineStage] = []

    def partition(self) -> list[PipelineStage]:
        """Partition model layers into pipeline stages balanced by compute."""
        analyzer = ArchitectureAnalyzer(self.architecture)
        split_points = analyzer.recommend_split_points(self.num_gpus)

        self.stages = []
        layers = self.architecture.layers
        for i in range(len(split_points)):
            start = split_points[i]
            end = split_points[i + 1] if i + 1 < len(split_points) else len(layers)
            if start < len(layers):
                stage = PipelineStage(
                    stage_id=i,
                    gpu_id=i,
                    layers=layers[start:end],
                )
                self.stages.append(stage)
        return self.stages

    def forward_schedule(self, num_microbatches: int) -> list[list[tuple[int, int]]]:
        """Generate a 1F1B pipeline schedule.

        Returns a list of time steps, each containing (stage_id, microbatch_id) tuples
        that can execute concurrently.
        """
        if not self.stages:
            self.partition()

        num_stages = len(self.stages)
        schedule: list[list[tuple[int, int]]] = []

        # Warmup: fill the pipeline
        for step in range(num_stages + num_microbatches - 1):
            concurrent = []
            for s in range(num_stages):
                mb = step - s
                if 0 <= mb < num_microbatches:
                    concurrent.append((s, mb))
            if concurrent:
                schedule.append(concurrent)
        return schedule

    def pipeline_bubble_ratio(self, num_microbatches: int) -> float:
        """Calculate the fraction of idle time (bubble) in the pipeline."""
        num_stages = len(self.stages) if self.stages else self.num_gpus
        if num_microbatches == 0:
            return 1.0
        total_slots = (num_stages + num_microbatches - 1) * num_stages
        active_slots = num_microbatches * num_stages
        return 1.0 - (active_slots / total_slots)


class DataParallelism:
    """Replicate model across GPUs and split data."""

    def __init__(self, num_replicas: int) -> None:
        self.num_replicas = num_replicas

    def split_batch(self, batch_size: int) -> list[int]:
        """Split a batch evenly across replicas."""
        base = batch_size // self.num_replicas
        remainder = batch_size % self.num_replicas
        sizes = [base + (1 if i < remainder else 0) for i in range(self.num_replicas)]
        return sizes

    def split_data(self, data: np.ndarray) -> list[np.ndarray]:
        """Split a data array across replicas along the first axis."""
        return list(np.array_split(data, self.num_replicas, axis=0))

    def effective_batch_size(self, per_replica_batch: int) -> int:
        """Calculate the effective global batch size."""
        return per_replica_batch * self.num_replicas


class RatioBasedDistributor:
    """Distribute model layers based on cardinality ratios."""

    def __init__(self, architecture: ModelArchitecture, gpu_memory_limits: list[int]) -> None:
        self.architecture = architecture
        self.gpu_memory_limits = gpu_memory_limits

    def distribute(self) -> dict[int, list[LayerConfig]]:
        """Assign layers to GPUs based on cardinality ratio and memory constraints."""
        assignment: dict[int, list[LayerConfig]] = {
            i: [] for i in range(len(self.gpu_memory_limits))
        }
        gpu_usage = [0] * len(self.gpu_memory_limits)

        # Sort layers by cardinality descending (larger layers first)
        sorted_layers = sorted(
            self.architecture.layers,
            key=lambda l: l.cardinality,
            reverse=True,
        )

        for layer in sorted_layers:
            # Find the GPU with most remaining memory
            best_gpu = -1
            best_remaining = -1
            for g in range(len(self.gpu_memory_limits)):
                remaining = self.gpu_memory_limits[g] - gpu_usage[g]
                if remaining >= layer.memory_bytes and remaining > best_remaining:
                    best_gpu = g
                    best_remaining = remaining
            if best_gpu == -1:
                raise ValueError(
                    f"Layer {layer.layer_id} ({layer.memory_bytes} bytes) "
                    "does not fit in any GPU"
                )
            assignment[best_gpu].append(layer)
            gpu_usage[best_gpu] += layer.memory_bytes

        return assignment


class GradientSyncManager:
    """Synchronize gradients across pipeline stages or data-parallel replicas."""

    def __init__(self, num_workers: int, sync_mode: str = "allreduce") -> None:
        if sync_mode not in ("allreduce", "ring", "tree"):
            raise ValueError(f"Unknown sync_mode: {sync_mode}")
        self.num_workers = num_workers
        self.sync_mode = sync_mode

    def allreduce(self, gradients: list[np.ndarray]) -> np.ndarray:
        """Average gradients across all workers."""
        if not gradients:
            raise ValueError("No gradients to synchronize")
        stacked = np.stack(gradients)
        return np.mean(stacked, axis=0)

    def ring_reduce(self, gradients: list[np.ndarray]) -> np.ndarray:
        """Simulate ring all-reduce by sequentially accumulating and averaging."""
        if not gradients:
            raise ValueError("No gradients to synchronize")
        result = gradients[0].copy()
        for g in gradients[1:]:
            result += g
        return result / len(gradients)

    def sync(self, gradients: list[np.ndarray]) -> np.ndarray:
        """Synchronize gradients using the configured mode."""
        if self.sync_mode == "allreduce":
            return self.allreduce(gradients)
        elif self.sync_mode == "ring":
            return self.ring_reduce(gradients)
        elif self.sync_mode == "tree":
            # Tree reduce: pair-wise reduction
            current = list(gradients)
            while len(current) > 1:
                next_level = []
                for i in range(0, len(current), 2):
                    if i + 1 < len(current):
                        next_level.append((current[i] + current[i + 1]) / 2.0)
                    else:
                        next_level.append(current[i])
                current = next_level
            return current[0]
        raise ValueError(f"Unknown sync_mode: {self.sync_mode}")

    def estimated_comm_time(self, param_count: int, bandwidth_gbps: float = 100.0) -> float:
        """Estimate communication time in seconds for gradient sync.

        Args:
            param_count: Number of parameters (float32).
            bandwidth_gbps: Network bandwidth in Gbps.
        """
        data_bytes = param_count * 4  # float32
        data_bits = data_bytes * 8
        bandwidth_bps = bandwidth_gbps * 1e9

        if self.sync_mode == "allreduce":
            # Ring allreduce: 2 * (N-1)/N * data_size
            factor = 2.0 * (self.num_workers - 1) / self.num_workers
        elif self.sync_mode == "ring":
            factor = 2.0 * (self.num_workers - 1) / self.num_workers
        else:
            # Tree: log2(N) steps
            import math
            factor = math.log2(self.num_workers) if self.num_workers > 1 else 0
        return factor * data_bits / bandwidth_bps
