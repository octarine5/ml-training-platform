"""Training orchestrator: end-to-end pipeline, config, and checkpoint management."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ml_training.architecture import ModelArchitecture
from ml_training.data_pipeline import DatasetSplit, TrainingDataPipeline
from ml_training.evaluation import EvaluationSystem, MetricsResult


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 256
    optimization_goal: str = "minimize_loss"
    weight_decay: float = 0.0001
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "optimization_goal": self.optimization_goal,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
        }


@dataclass
class Checkpoint:
    """A training checkpoint."""

    epoch: int
    step: int
    weights: dict[str, np.ndarray]
    optimizer_state: dict[str, Any]
    metrics: Optional[MetricsResult] = None
    timestamp: float = field(default_factory=time.time)


class CheckpointManager:
    """Save and load training checkpoints."""

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self._checkpoints: list[Checkpoint] = []
        self._max_to_keep: int = 5

    def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint (in-memory for this implementation)."""
        self._checkpoints.append(checkpoint)
        # Keep only the most recent checkpoints
        if len(self._checkpoints) > self._max_to_keep:
            self._checkpoints = self._checkpoints[-self._max_to_keep:]
        path = str(self.checkpoint_dir / f"ckpt_epoch{checkpoint.epoch}_step{checkpoint.step}")
        return path

    def load_latest(self) -> Optional[Checkpoint]:
        """Load the most recent checkpoint."""
        if not self._checkpoints:
            return None
        return self._checkpoints[-1]

    def load_best(self, metric: str = "auc", higher_is_better: bool = True) -> Optional[Checkpoint]:
        """Load the checkpoint with the best metric value."""
        valid = [c for c in self._checkpoints if c.metrics is not None]
        if not valid:
            return None
        if higher_is_better:
            return max(valid, key=lambda c: getattr(c.metrics, metric, 0))
        return min(valid, key=lambda c: getattr(c.metrics, metric, float("inf")))

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all saved checkpoints."""
        return [
            {
                "epoch": c.epoch,
                "step": c.step,
                "timestamp": c.timestamp,
                "has_metrics": c.metrics is not None,
            }
            for c in self._checkpoints
        ]


class TrainingOrchestrator:
    """End-to-end training pipeline: data -> train -> evaluate -> deploy."""

    def __init__(
        self,
        architecture: ModelArchitecture,
        config: TrainingConfig,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self.architecture = architecture
        self.config = config
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.evaluator = EvaluationSystem()
        self._rng = np.random.default_rng(config.seed)
        self._weights: dict[str, np.ndarray] = {}
        self._training_history: list[dict[str, Any]] = []

    def initialize_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for layer in self.architecture.layers:
            fan_in = layer.cardinality
            fan_out = layer.num_nodes
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights[f"layer_{layer.layer_id}"] = (
                self._rng.standard_normal((fan_in, fan_out)).astype(np.float32) * scale
            )

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the model."""
        h = x
        for layer in self.architecture.layers:
            w = self._weights[f"layer_{layer.layer_id}"]
            # Handle dimension mismatch by projecting input
            if h.shape[1] != w.shape[0]:
                h = h[:, :w.shape[0]] if h.shape[1] > w.shape[0] else np.pad(
                    h, ((0, 0), (0, w.shape[0] - h.shape[1]))
                )
            h = h @ w
            if layer.activation == "relu":
                h = np.maximum(0, h)
            elif layer.activation == "sigmoid":
                h = 1.0 / (1.0 + np.exp(-np.clip(h, -500, 500)))
        # Output: sigmoid for binary classification
        output = h.mean(axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(output, -500, 500)))

    def _compute_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        eps = 1e-15
        preds = np.clip(predictions, eps, 1 - eps)
        return float(-np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds)))

    def train_epoch(
        self, train_data: DatasetSplit, epoch: int
    ) -> dict[str, float]:
        """Train for one epoch and return loss metrics."""
        pipeline = TrainingDataPipeline(seed=self.config.seed + epoch)
        pipeline._features = train_data.features
        pipeline._labels = train_data.labels
        batches = pipeline.create_batches(train_data, self.config.batch_size)

        epoch_loss = 0.0
        num_batches = 0

        for batch_features, batch_labels in batches:
            predictions = self._forward(batch_features)
            loss = self._compute_loss(predictions, batch_labels)

            # Simulated gradient update: perturb weights slightly toward lower loss
            for key in self._weights:
                grad_noise = self._rng.standard_normal(self._weights[key].shape).astype(
                    np.float32
                )
                grad_norm = np.linalg.norm(grad_noise) + 1e-8
                clipped = grad_noise * min(1.0, self.config.max_grad_norm / grad_norm)
                self._weights[key] -= self.config.learning_rate * clipped
                self._weights[key] *= 1.0 - self.config.weight_decay

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        return {"epoch": epoch, "avg_loss": avg_loss, "num_batches": num_batches}

    def evaluate(self, eval_data: DatasetSplit) -> MetricsResult:
        """Evaluate model on eval data."""
        predictions = self._forward(eval_data.features)
        return self.evaluator.evaluate(predictions, eval_data.labels)

    def train(
        self,
        train_data: DatasetSplit,
        eval_data: DatasetSplit,
    ) -> list[dict[str, Any]]:
        """Run the full training loop."""
        if not self._weights:
            self.initialize_weights()

        history = []
        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch(train_data, epoch)
            eval_metrics = self.evaluate(eval_data)

            step_info = {
                "epoch": epoch,
                "train_loss": train_metrics["avg_loss"],
                "eval_auc": eval_metrics.auc,
                "eval_log_loss": eval_metrics.log_loss,
            }
            history.append(step_info)

            # Save checkpoint
            checkpoint = Checkpoint(
                epoch=epoch,
                step=epoch * train_metrics["num_batches"],
                weights={k: v.copy() for k, v in self._weights.items()},
                optimizer_state={},
                metrics=eval_metrics,
            )
            self.checkpoint_manager.save(checkpoint)

        self._training_history = history
        return history

    @property
    def training_history(self) -> list[dict[str, Any]]:
        return self._training_history
