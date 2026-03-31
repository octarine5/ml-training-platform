"""Shared fixtures for ML Training Platform tests."""

from __future__ import annotations

import numpy as np
import pytest

from ml_training.architecture import ModelArchitecture
from ml_training.data_pipeline import DatasetSplit, TrainingDataPipeline
from ml_training.evaluation import MetricsResult


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_features(rng: np.random.Generator) -> np.ndarray:
    """Generate sample feature matrix (200 samples x 32 features)."""
    return rng.standard_normal((200, 32)).astype(np.float32)


@pytest.fixture
def sample_labels(rng: np.random.Generator) -> np.ndarray:
    """Generate binary labels for 200 samples."""
    return rng.integers(0, 2, size=200).astype(np.float32)


@pytest.fixture
def sample_predictions(rng: np.random.Generator) -> np.ndarray:
    """Generate prediction scores for 200 samples."""
    return rng.uniform(0, 1, size=200).astype(np.float32)


@pytest.fixture
def small_architecture() -> ModelArchitecture:
    """Build a small 3-layer architecture for testing."""
    arch = ModelArchitecture(name="test-model")
    arch.add_layer(num_nodes=16, cardinality=32, activation="relu")
    arch.add_layer(num_nodes=8, cardinality=16, activation="relu")
    arch.add_layer(num_nodes=4, cardinality=8, activation="sigmoid")
    arch.build()
    return arch


@pytest.fixture
def train_eval_splits(
    sample_features: np.ndarray, sample_labels: np.ndarray
) -> tuple[DatasetSplit, DatasetSplit]:
    """Create train/eval splits from sample data."""
    pipeline = TrainingDataPipeline(seed=42)
    pipeline.load(sample_features, sample_labels)
    return pipeline.split()


@pytest.fixture
def baseline_metrics() -> MetricsResult:
    """Baseline metrics for drift detection tests."""
    return MetricsResult(
        ctr=0.15,
        cvr=0.05,
        auc=0.85,
        log_loss=0.45,
        num_samples=1000,
    )
