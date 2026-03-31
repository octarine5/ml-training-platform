"""Training data pipeline: loading, sampling, splitting, and distribution matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class DatasetSplit:
    """A split of a dataset with features and labels."""

    features: np.ndarray
    labels: np.ndarray
    split_name: str

    @property
    def num_samples(self) -> int:
        return self.features.shape[0]


class TrainingDataPipeline:
    """Load, sample, and split data into train/eval sets."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def load(self, features: np.ndarray, labels: np.ndarray) -> TrainingDataPipeline:
        """Load raw features and labels."""
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Feature rows ({features.shape[0]}) != label rows ({labels.shape[0]})"
            )
        self._features = features
        self._labels = labels
        return self

    def sample(self, fraction: float) -> TrainingDataPipeline:
        """Subsample the loaded data."""
        if not hasattr(self, "_features"):
            raise RuntimeError("No data loaded. Call load() first.")
        n = self._features.shape[0]
        k = max(1, int(n * fraction))
        indices = self._rng.choice(n, size=k, replace=False)
        self._features = self._features[indices]
        self._labels = self._labels[indices]
        return self

    def split(
        self, train_ratio: float = 0.8, eval_ratio: float = 0.2
    ) -> tuple[DatasetSplit, DatasetSplit]:
        """Split data into train and eval sets."""
        if not hasattr(self, "_features"):
            raise RuntimeError("No data loaded. Call load() first.")
        total = train_ratio + eval_ratio
        train_frac = train_ratio / total

        n = self._features.shape[0]
        indices = self._rng.permutation(n)
        split_idx = int(n * train_frac)

        train_idx = indices[:split_idx]
        eval_idx = indices[split_idx:]

        train = DatasetSplit(
            features=self._features[train_idx],
            labels=self._labels[train_idx],
            split_name="train",
        )
        eval_split = DatasetSplit(
            features=self._features[eval_idx],
            labels=self._labels[eval_idx],
            split_name="eval",
        )
        return train, eval_split

    def create_batches(
        self, data: DatasetSplit, batch_size: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create mini-batches from a dataset split."""
        batches = []
        n = data.num_samples
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batches.append((data.features[start:end], data.labels[start:end]))
        return batches


class DistributionMatcher:
    """Ensure training distribution matches production ratios."""

    def __init__(self, target_distribution: dict[int, float]) -> None:
        """Initialize with target class distribution.

        Args:
            target_distribution: Mapping of class label to desired fraction.
        """
        self.target_distribution = target_distribution
        total = sum(target_distribution.values())
        self._normalized = {k: v / total for k, v in target_distribution.items()}

    def current_distribution(self, labels: np.ndarray) -> dict[int, float]:
        """Compute the current class distribution."""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        return {int(u): c / total for u, c in zip(unique, counts)}

    def divergence(self, labels: np.ndarray) -> float:
        """Compute KL divergence between current and target distribution."""
        current = self.current_distribution(labels)
        kl = 0.0
        for cls, target_p in self._normalized.items():
            current_p = current.get(cls, 1e-10)
            if target_p > 0:
                kl += target_p * np.log(target_p / (current_p + 1e-10))
        return float(kl)

    def resample(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        total_samples: int,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample data to match the target distribution."""
        rng = np.random.default_rng(seed)
        new_features = []
        new_labels = []

        for cls, frac in self._normalized.items():
            n_target = max(1, int(total_samples * frac))
            cls_mask = labels == cls
            cls_features = features[cls_mask]
            cls_labels = labels[cls_mask]
            if len(cls_features) == 0:
                continue
            indices = rng.choice(len(cls_features), size=n_target, replace=True)
            new_features.append(cls_features[indices])
            new_labels.append(cls_labels[indices])

        return np.concatenate(new_features), np.concatenate(new_labels)


class DataSampler:
    """Automatic sampling from production data for training."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def reservoir_sample(
        self, stream: list[np.ndarray], k: int
    ) -> np.ndarray:
        """Reservoir sampling: select k items from a stream of arrays uniformly."""
        reservoir: list[np.ndarray] = []
        count = 0
        for batch in stream:
            for i in range(batch.shape[0]):
                item = batch[i]
                count += 1
                if len(reservoir) < k:
                    reservoir.append(item)
                else:
                    j = self._rng.integers(0, count)
                    if j < k:
                        reservoir[j] = item
        return np.array(reservoir)

    def stratified_sample(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        samples_per_class: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample equal numbers of samples per class."""
        unique_classes = np.unique(labels)
        sampled_features = []
        sampled_labels = []
        for cls in unique_classes:
            mask = labels == cls
            cls_features = features[mask]
            n = min(samples_per_class, len(cls_features))
            indices = self._rng.choice(len(cls_features), size=n, replace=False)
            sampled_features.append(cls_features[indices])
            sampled_labels.append(np.full(n, cls))
        return np.concatenate(sampled_features), np.concatenate(sampled_labels)

    def time_decay_sample(
        self,
        features: np.ndarray,
        timestamps: np.ndarray,
        sample_size: int,
        decay_rate: float = 0.01,
    ) -> np.ndarray:
        """Sample with higher probability for more recent data."""
        max_time = timestamps.max()
        weights = np.exp(-decay_rate * (max_time - timestamps))
        probs = weights / weights.sum()
        indices = self._rng.choice(len(features), size=sample_size, replace=False, p=probs)
        return features[indices]
