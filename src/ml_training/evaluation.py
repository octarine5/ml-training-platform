"""Evaluation system: metrics, drift detection, and retraining triggers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""

    ctr: float
    cvr: float
    auc: float
    log_loss: float
    num_samples: int
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ctr": self.ctr,
            "cvr": self.cvr,
            "auc": self.auc,
            "log_loss": self.log_loss,
            "num_samples": self.num_samples,
            **self.extra,
        }


class EvaluationSystem:
    """Compute CTR/CVR metrics on evaluation sets."""

    def __init__(self, positive_threshold: float = 0.5) -> None:
        self.positive_threshold = positive_threshold

    def compute_ctr(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute click-through rate: fraction of positives in labels."""
        return float(labels.mean())

    def compute_cvr(
        self, predictions: np.ndarray, labels: np.ndarray, clicks: np.ndarray
    ) -> float:
        """Compute conversion rate among clicked items."""
        clicked_mask = clicks > 0
        if clicked_mask.sum() == 0:
            return 0.0
        return float(labels[clicked_mask].mean())

    def compute_auc(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute AUC-ROC using the trapezoidal rule."""
        # Sort by prediction score descending
        sorted_indices = np.argsort(-predictions)
        sorted_labels = labels[sorted_indices]

        tp = 0
        fp = 0
        total_pos = int(labels.sum())
        total_neg = len(labels) - total_pos

        if total_pos == 0 or total_neg == 0:
            return 0.5

        tpr_prev = 0.0
        fpr_prev = 0.0
        auc = 0.0

        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / total_pos
            fpr = fp / total_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
            tpr_prev = tpr
            fpr_prev = fpr

        return float(auc)

    def compute_log_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute binary cross-entropy log loss."""
        eps = 1e-15
        preds = np.clip(predictions, eps, 1.0 - eps)
        loss = -np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))
        return float(loss)

    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        clicks: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        """Run full evaluation and return metrics."""
        ctr = self.compute_ctr(predictions, labels)
        auc = self.compute_auc(predictions, labels)
        log_loss = self.compute_log_loss(predictions, labels)
        cvr = 0.0
        if clicks is not None:
            cvr = self.compute_cvr(predictions, labels, clicks)
        return MetricsResult(
            ctr=ctr,
            cvr=cvr,
            auc=auc,
            log_loss=log_loss,
            num_samples=len(labels),
        )


class ModelDriftDetector:
    """Detect performance drift between training and production metrics."""

    def __init__(
        self,
        baseline_metrics: MetricsResult,
        auc_threshold: float = 0.02,
        log_loss_threshold: float = 0.1,
    ) -> None:
        self.baseline = baseline_metrics
        self.auc_threshold = auc_threshold
        self.log_loss_threshold = log_loss_threshold
        self._history: list[MetricsResult] = []

    def record(self, metrics: MetricsResult) -> None:
        """Record a new set of production metrics."""
        self._history.append(metrics)

    def check_drift(self, current: MetricsResult) -> dict[str, Any]:
        """Check if current metrics have drifted from baseline."""
        auc_drop = self.baseline.auc - current.auc
        log_loss_increase = current.log_loss - self.baseline.log_loss

        drifted = False
        reasons = []

        if auc_drop > self.auc_threshold:
            drifted = True
            reasons.append(f"AUC dropped by {auc_drop:.4f} (threshold: {self.auc_threshold})")

        if log_loss_increase > self.log_loss_threshold:
            drifted = True
            reasons.append(
                f"Log loss increased by {log_loss_increase:.4f} "
                f"(threshold: {self.log_loss_threshold})"
            )

        return {
            "drifted": drifted,
            "auc_drop": auc_drop,
            "log_loss_increase": log_loss_increase,
            "reasons": reasons,
        }

    def trend(self) -> dict[str, list[float]]:
        """Return metric trends over recorded history."""
        return {
            "auc": [m.auc for m in self._history],
            "log_loss": [m.log_loss for m in self._history],
            "ctr": [m.ctr for m in self._history],
        }


class RetrainingTrigger:
    """Auto-trigger retraining when metrics drop below threshold."""

    def __init__(
        self,
        min_auc: float = 0.70,
        max_log_loss: float = 1.0,
        cooldown_steps: int = 5,
    ) -> None:
        self.min_auc = min_auc
        self.max_log_loss = max_log_loss
        self.cooldown_steps = cooldown_steps
        self._steps_since_retrain = cooldown_steps  # Allow immediate trigger

    def should_retrain(self, metrics: MetricsResult) -> bool:
        """Determine if retraining should be triggered."""
        self._steps_since_retrain += 1
        if self._steps_since_retrain < self.cooldown_steps:
            return False
        if metrics.auc < self.min_auc or metrics.log_loss > self.max_log_loss:
            self._steps_since_retrain = 0
            return True
        return False

    def reset(self) -> None:
        """Reset the trigger state."""
        self._steps_since_retrain = self.cooldown_steps
