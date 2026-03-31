"""Model architecture definition and analysis for distribution planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LayerConfig:
    """Configuration for a single model layer."""

    layer_id: int
    num_nodes: int
    cardinality: int
    compute_intensity: float = 1.0
    activation: str = "relu"
    dtype: str = "float32"

    @property
    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes for this layer's parameters."""
        dtype_sizes = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}
        element_size = dtype_sizes.get(self.dtype, 4)
        return self.num_nodes * self.cardinality * element_size

    @property
    def flops(self) -> float:
        """Estimate FLOPs for a forward pass through this layer."""
        return 2.0 * self.num_nodes * self.cardinality * self.compute_intensity


class ModelArchitecture:
    """Define model layers, node counts, and cardinalities per layer."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.layers: list[LayerConfig] = []
        self._built = False

    def add_layer(
        self,
        num_nodes: int,
        cardinality: int,
        compute_intensity: float = 1.0,
        activation: str = "relu",
        dtype: str = "float32",
    ) -> LayerConfig:
        """Add a layer to the architecture."""
        layer = LayerConfig(
            layer_id=len(self.layers),
            num_nodes=num_nodes,
            cardinality=cardinality,
            compute_intensity=compute_intensity,
            activation=activation,
            dtype=dtype,
        )
        self.layers.append(layer)
        self._built = False
        return layer

    def build(self) -> ModelArchitecture:
        """Validate and finalize the architecture."""
        if not self.layers:
            raise ValueError("Architecture must have at least one layer")
        for i in range(1, len(self.layers)):
            prev = self.layers[i - 1]
            curr = self.layers[i]
            if prev.num_nodes != curr.cardinality:
                raise ValueError(
                    f"Layer {i} cardinality ({curr.cardinality}) must match "
                    f"layer {i-1} num_nodes ({prev.num_nodes})"
                )
        self._built = True
        return self

    @property
    def total_parameters(self) -> int:
        """Total number of parameters across all layers."""
        return sum(l.num_nodes * l.cardinality for l in self.layers)

    @property
    def total_memory_bytes(self) -> int:
        """Total estimated memory in bytes."""
        return sum(l.memory_bytes for l in self.layers)

    def summary(self) -> list[dict]:
        """Return a summary of each layer."""
        return [
            {
                "layer_id": l.layer_id,
                "num_nodes": l.num_nodes,
                "cardinality": l.cardinality,
                "params": l.num_nodes * l.cardinality,
                "memory_mb": l.memory_bytes / (1024 * 1024),
                "flops": l.flops,
            }
            for l in self.layers
        ]


class ArchitectureAnalyzer:
    """Analyze model architecture for compute/memory distribution planning."""

    def __init__(self, architecture: ModelArchitecture) -> None:
        self.architecture = architecture

    def compute_per_layer(self) -> list[float]:
        """Return estimated FLOPs per layer."""
        return [layer.flops for layer in self.architecture.layers]

    def memory_per_layer(self) -> list[int]:
        """Return estimated memory bytes per layer."""
        return [layer.memory_bytes for layer in self.architecture.layers]

    def compute_ratios(self) -> list[float]:
        """Return the fraction of total compute each layer requires."""
        flops = self.compute_per_layer()
        total = sum(flops)
        if total == 0:
            return [0.0] * len(flops)
        return [f / total for f in flops]

    def memory_ratios(self) -> list[float]:
        """Return the fraction of total memory each layer requires."""
        mem = self.memory_per_layer()
        total = sum(mem)
        if total == 0:
            return [0.0] * len(mem)
        return [m / total for m in mem]

    def recommend_split_points(self, num_gpus: int) -> list[int]:
        """Recommend layer indices to split the model across GPUs.

        Tries to balance compute evenly across pipeline stages.
        Returns indices where each new stage begins.
        """
        if num_gpus <= 1:
            return [0]
        flops = self.compute_per_layer()
        total = sum(flops)
        target_per_gpu = total / num_gpus

        split_points = [0]
        running = 0.0
        for i, f in enumerate(flops):
            running += f
            if running >= target_per_gpu and len(split_points) < num_gpus:
                split_points.append(i + 1)
                running = 0.0

        # Ensure we have exactly num_gpus split points
        while len(split_points) < num_gpus:
            split_points.append(len(flops))
        return split_points[:num_gpus]

    def bottleneck_layers(self, top_k: int = 3) -> list[LayerConfig]:
        """Identify the top-k layers with highest compute intensity."""
        sorted_layers = sorted(
            self.architecture.layers,
            key=lambda l: l.flops,
            reverse=True,
        )
        return sorted_layers[:top_k]
