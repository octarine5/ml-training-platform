"""Fine-tuning, LoRA simulation, and quantization for ML models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ml_training.architecture import ModelArchitecture
from ml_training.data_pipeline import DatasetSplit, TrainingDataPipeline
from ml_training.evaluation import EvaluationSystem, MetricsResult


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning a pre-trained model."""

    epochs: int = 5
    learning_rate: float = 0.0001
    batch_size: int = 128
    lora_rank: int = 8
    lora_alpha: float = 16.0
    freeze_layers: list[int] = field(default_factory=list)
    target_layers: list[int] = field(default_factory=list)
    weight_decay: float = 0.00001
    seed: int = 42

    @property
    def lora_scaling(self) -> float:
        """LoRA scaling factor: alpha / rank."""
        return self.lora_alpha / self.lora_rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "freeze_layers": self.freeze_layers,
            "target_layers": self.target_layers,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
        }


class FineTuner:
    """Domain-specific fine-tuning with LoRA (Low-Rank Adaptation) simulation."""

    def __init__(
        self,
        architecture: ModelArchitecture,
        base_weights: dict[str, np.ndarray],
        config: FineTuneConfig,
    ) -> None:
        self.architecture = architecture
        self.base_weights = {k: v.copy() for k, v in base_weights.items()}
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self.evaluator = EvaluationSystem()
        self._lora_a: dict[str, np.ndarray] = {}
        self._lora_b: dict[str, np.ndarray] = {}
        self._training_history: list[dict[str, Any]] = []
        self._initialize_lora()

    def _initialize_lora(self) -> None:
        """Initialize LoRA low-rank matrices A and B for target layers."""
        target = self.config.target_layers
        if not target:
            target = [l.layer_id for l in self.architecture.layers]

        for layer in self.architecture.layers:
            if layer.layer_id not in target:
                continue
            key = f"layer_{layer.layer_id}"
            if key not in self.base_weights:
                continue
            w = self.base_weights[key]
            d_in, d_out = w.shape
            r = self.config.lora_rank
            # A initialized with small random values, B initialized to zero
            self._lora_a[key] = self._rng.standard_normal((d_in, r)).astype(
                np.float32
            ) * 0.01
            self._lora_b[key] = np.zeros((r, d_out), dtype=np.float32)

    def get_effective_weights(self) -> dict[str, np.ndarray]:
        """Compute effective weights: W_eff = W_base + scaling * A @ B."""
        effective = {}
        scaling = self.config.lora_scaling
        for key, w in self.base_weights.items():
            if key in self._lora_a:
                delta = self._lora_a[key] @ self._lora_b[key]
                effective[key] = w + scaling * delta
            else:
                effective[key] = w.copy()
        return effective

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using effective (base + LoRA) weights."""
        weights = self.get_effective_weights()
        h = x
        for layer in self.architecture.layers:
            key = f"layer_{layer.layer_id}"
            w = weights[key]
            if h.shape[1] != w.shape[0]:
                if h.shape[1] > w.shape[0]:
                    h = h[:, : w.shape[0]]
                else:
                    h = np.pad(h, ((0, 0), (0, w.shape[0] - h.shape[1])))
            h = h @ w
            if layer.activation == "relu":
                h = np.maximum(0, h)
            elif layer.activation == "sigmoid":
                h = 1.0 / (1.0 + np.exp(-np.clip(h, -500, 500)))
        output = h.mean(axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(output, -500, 500)))

    def _compute_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        eps = 1e-15
        preds = np.clip(predictions, eps, 1 - eps)
        return float(-np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds)))

    def fine_tune_epoch(
        self, train_data: DatasetSplit, epoch: int
    ) -> dict[str, float]:
        """Fine-tune for one epoch, updating only LoRA parameters."""
        pipeline = TrainingDataPipeline(seed=self.config.seed + epoch)
        pipeline._features = train_data.features
        pipeline._labels = train_data.labels
        batches = pipeline.create_batches(train_data, self.config.batch_size)

        epoch_loss = 0.0
        num_batches = 0

        for batch_features, batch_labels in batches:
            predictions = self._forward(batch_features)
            loss = self._compute_loss(predictions, batch_labels)

            # Update only LoRA parameters (not base weights)
            for key in self._lora_a:
                grad_a = self._rng.standard_normal(self._lora_a[key].shape).astype(
                    np.float32
                )
                grad_b = self._rng.standard_normal(self._lora_b[key].shape).astype(
                    np.float32
                )
                self._lora_a[key] -= self.config.learning_rate * grad_a
                self._lora_b[key] -= self.config.learning_rate * grad_b
                self._lora_a[key] *= 1.0 - self.config.weight_decay
                self._lora_b[key] *= 1.0 - self.config.weight_decay

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        return {"epoch": epoch, "avg_loss": avg_loss, "num_batches": num_batches}

    def fine_tune(
        self,
        train_data: DatasetSplit,
        eval_data: DatasetSplit,
    ) -> list[dict[str, Any]]:
        """Run the full fine-tuning loop."""
        history = []
        for epoch in range(self.config.epochs):
            train_metrics = self.fine_tune_epoch(train_data, epoch)
            predictions = self._forward(eval_data.features)
            eval_metrics = self.evaluator.evaluate(predictions, eval_data.labels)

            step_info = {
                "epoch": epoch,
                "train_loss": train_metrics["avg_loss"],
                "eval_auc": eval_metrics.auc,
                "eval_log_loss": eval_metrics.log_loss,
            }
            history.append(step_info)

        self._training_history = history
        return history

    def merge_lora_weights(self) -> dict[str, np.ndarray]:
        """Merge LoRA adaptations into base weights permanently."""
        merged = self.get_effective_weights()
        self.base_weights = merged
        self._lora_a.clear()
        self._lora_b.clear()
        self._initialize_lora()
        return merged

    @property
    def num_trainable_params(self) -> int:
        """Count the number of trainable LoRA parameters."""
        total = 0
        for key in self._lora_a:
            total += self._lora_a[key].size + self._lora_b[key].size
        return total

    @property
    def training_history(self) -> list[dict[str, Any]]:
        return self._training_history


class Quantizer:
    """Quantize model weights to INT8 or INT4 for inference efficiency."""

    SUPPORTED_MODES = ("int8", "int4")

    def __init__(self, mode: str = "int8") -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported quantization mode: {mode}. "
                f"Supported: {self.SUPPORTED_MODES}"
            )
        self.mode = mode

    def _compute_scale(self, tensor: np.ndarray) -> float:
        """Compute per-tensor quantization scale."""
        abs_max = np.abs(tensor).max()
        if abs_max == 0:
            return 1.0
        if self.mode == "int8":
            return float(abs_max / 127.0)
        else:  # int4
            return float(abs_max / 7.0)

    def quantize_tensor(
        self, tensor: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Quantize a single tensor, returning quantized values and scale."""
        scale = self._compute_scale(tensor)
        if self.mode == "int8":
            quantized = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
        else:  # int4
            quantized = np.clip(np.round(tensor / scale), -8, 7).astype(np.int8)
        return quantized, scale

    def dequantize_tensor(
        self, quantized: np.ndarray, scale: float
    ) -> np.ndarray:
        """Dequantize a tensor back to float32."""
        return quantized.astype(np.float32) * scale

    def quantize_weights(
        self, weights: dict[str, np.ndarray]
    ) -> dict[str, tuple[np.ndarray, float]]:
        """Quantize all model weights."""
        result = {}
        for name, w in weights.items():
            result[name] = self.quantize_tensor(w)
        return result

    def dequantize_weights(
        self, quantized_weights: dict[str, tuple[np.ndarray, float]]
    ) -> dict[str, np.ndarray]:
        """Dequantize all model weights back to float32."""
        result = {}
        for name, (q, scale) in quantized_weights.items():
            result[name] = self.dequantize_tensor(q, scale)
        return result

    def compression_ratio(self, weights: dict[str, np.ndarray]) -> float:
        """Calculate the compression ratio from quantization."""
        original_bytes = sum(w.nbytes for w in weights.values())
        if self.mode == "int8":
            quantized_bytes = original_bytes // 4  # float32 -> int8
        else:  # int4
            quantized_bytes = original_bytes // 8  # float32 -> int4 (stored as int8)
        if quantized_bytes == 0:
            return 1.0
        return original_bytes / quantized_bytes

    def quantization_error(self, weights: dict[str, np.ndarray]) -> dict[str, float]:
        """Compute the mean absolute quantization error per weight tensor."""
        errors = {}
        for name, w in weights.items():
            q, scale = self.quantize_tensor(w)
            reconstructed = self.dequantize_tensor(q, scale)
            errors[name] = float(np.abs(w - reconstructed).mean())
        return errors
