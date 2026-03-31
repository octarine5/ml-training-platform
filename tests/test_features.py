"""Tests for feature engineering and data pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ml_training.data_pipeline import (
    DataSampler,
    DatasetSplit,
    DistributionMatcher,
    TrainingDataPipeline,
)
from ml_training.features import FeatureMatrix, FeatureStore, FeatureTransformer


class TestTrainingDataPipeline:
    """Tests for TrainingDataPipeline."""

    def test_load_and_split(
        self, sample_features: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        pipeline = TrainingDataPipeline(seed=42)
        pipeline.load(sample_features, sample_labels)
        train, eval_split = pipeline.split()
        assert train.num_samples + eval_split.num_samples == len(sample_features)
        assert train.split_name == "train"
        assert eval_split.split_name == "eval"

    def test_load_mismatched_raises(self) -> None:
        pipeline = TrainingDataPipeline()
        with pytest.raises(ValueError, match="Feature rows"):
            pipeline.load(np.zeros((10, 5)), np.zeros(8))

    def test_sample_reduces_data(
        self, sample_features: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        pipeline = TrainingDataPipeline(seed=42)
        pipeline.load(sample_features, sample_labels)
        pipeline.sample(0.5)
        train, eval_split = pipeline.split()
        total = train.num_samples + eval_split.num_samples
        assert total < len(sample_features)

    def test_sample_without_load_raises(self) -> None:
        pipeline = TrainingDataPipeline()
        with pytest.raises(RuntimeError, match="No data loaded"):
            pipeline.sample(0.5)

    def test_create_batches(self) -> None:
        data = DatasetSplit(
            features=np.zeros((100, 10)),
            labels=np.zeros(100),
            split_name="test",
        )
        pipeline = TrainingDataPipeline()
        batches = pipeline.create_batches(data, batch_size=30)
        assert len(batches) == 4  # 30 + 30 + 30 + 10
        assert batches[-1][0].shape[0] == 10


class TestDistributionMatcher:
    """Tests for DistributionMatcher."""

    def test_current_distribution(self) -> None:
        labels = np.array([0, 0, 0, 1, 1])
        matcher = DistributionMatcher({0: 0.5, 1: 0.5})
        dist = matcher.current_distribution(labels)
        assert abs(dist[0] - 0.6) < 1e-6
        assert abs(dist[1] - 0.4) < 1e-6

    def test_divergence_zero_for_matching(self) -> None:
        labels = np.array([0, 0, 1, 1])
        matcher = DistributionMatcher({0: 0.5, 1: 0.5})
        kl = matcher.divergence(labels)
        assert kl < 0.01

    def test_resample_matches_target(self) -> None:
        rng = np.random.default_rng(42)
        features = rng.standard_normal((1000, 5))
        labels = np.array([0] * 900 + [1] * 100)
        matcher = DistributionMatcher({0: 0.5, 1: 0.5})
        new_features, new_labels = matcher.resample(features, labels, 500)
        dist = matcher.current_distribution(new_labels)
        assert abs(dist[0] - 0.5) < 0.05
        assert abs(dist[1] - 0.5) < 0.05


class TestDataSampler:
    """Tests for DataSampler."""

    def test_reservoir_sample(self) -> None:
        stream = [np.arange(10).reshape(10, 1), np.arange(10, 20).reshape(10, 1)]
        sampler = DataSampler(seed=42)
        result = sampler.reservoir_sample(stream, k=5)
        assert result.shape[0] == 5

    def test_stratified_sample(self) -> None:
        features = np.zeros((100, 3))
        labels = np.array([0] * 50 + [1] * 50)
        sampler = DataSampler(seed=42)
        sf, sl = sampler.stratified_sample(features, labels, samples_per_class=10)
        assert len(sl) == 20
        assert (sl == 0).sum() == 10
        assert (sl == 1).sum() == 10

    def test_time_decay_sample(self) -> None:
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 5))
        timestamps = np.arange(100, dtype=np.float64)
        sampler = DataSampler(seed=42)
        result = sampler.time_decay_sample(features, timestamps, sample_size=20)
        assert result.shape == (20, 5)


class TestFeatureStore:
    """Tests for FeatureStore."""

    def test_register_and_get(self) -> None:
        store = FeatureStore()
        data = np.array([1.0, 2.0, 3.0])
        store.register("feature_a", data, version=1)
        retrieved = store.get("feature_a")
        np.testing.assert_array_equal(retrieved, data)

    def test_get_specific_version(self) -> None:
        store = FeatureStore()
        store.register("f", np.array([1.0]), version=1)
        store.register("f", np.array([2.0]), version=2)
        v1 = store.get("f", version=1)
        assert v1[0] == 1.0

    def test_get_latest_version(self) -> None:
        store = FeatureStore()
        store.register("f", np.array([1.0]), version=1)
        store.register("f", np.array([2.0]), version=2)
        latest = store.get("f")
        assert latest[0] == 2.0

    def test_get_missing_raises(self) -> None:
        store = FeatureStore()
        with pytest.raises(KeyError, match="not found"):
            store.get("nonexistent")

    def test_list_features(self) -> None:
        store = FeatureStore()
        store.register("a", np.array([1.0]))
        store.register("b", np.array([2.0, 3.0]))
        features = store.list_features()
        assert len(features) == 2

    def test_delete(self) -> None:
        store = FeatureStore()
        store.register("f", np.array([1.0]))
        store.delete("f")
        with pytest.raises(KeyError):
            store.get("f")


class TestFeatureTransformer:
    """Tests for FeatureTransformer."""

    def test_text_embedding_shape(self) -> None:
        transformer = FeatureTransformer()
        embeddings = transformer.text_embedding(["hello", "world"], dim=64)
        assert embeddings.shape == (2, 64)

    def test_text_embedding_deterministic(self) -> None:
        t1 = FeatureTransformer()
        t2 = FeatureTransformer()
        e1 = t1.text_embedding(["test"])
        e2 = t2.text_embedding(["test"])
        np.testing.assert_array_equal(e1, e2)

    def test_text_embedding_normalized(self) -> None:
        transformer = FeatureTransformer()
        embeddings = transformer.text_embedding(["normalize me"], dim=32)
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 0.01

    def test_normalize_standard(self) -> None:
        transformer = FeatureTransformer()
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normed = transformer.normalize(data, method="standard")
        assert abs(normed.mean()) < 0.01

    def test_normalize_minmax(self) -> None:
        transformer = FeatureTransformer()
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normed = transformer.normalize(data, method="minmax")
        assert normed.min() >= -0.01
        assert normed.max() <= 1.01

    def test_normalize_invalid_method(self) -> None:
        transformer = FeatureTransformer()
        with pytest.raises(ValueError, match="Unknown normalization"):
            transformer.normalize(np.zeros((3, 3)), method="bad")


class TestFeatureMatrix:
    """Tests for FeatureMatrix."""

    def test_concatenated(self) -> None:
        groups = {
            "a": np.ones((10, 3)),
            "b": np.ones((10, 5)),
        }
        fm = FeatureMatrix(groups)
        cat = fm.concatenated()
        assert cat.shape == (10, 8)

    def test_pairwise_interactions(self) -> None:
        groups = {
            "a": np.ones((10, 4)),
            "b": np.full((10, 4), 2.0),
        }
        fm = FeatureMatrix(groups)
        result = fm.pairwise_interactions("a", "b")
        assert result.shape == (10, 4)
        np.testing.assert_array_equal(result, np.full((10, 4), 2.0))

    def test_inconsistent_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="Inconsistent"):
            FeatureMatrix({"a": np.ones((10, 3)), "b": np.ones((8, 3))})

    def test_cross_features(self) -> None:
        groups = {
            "a": np.ones((5, 2)),
            "b": np.ones((5, 3)),
            "c": np.ones((5, 4)),
        }
        fm = FeatureMatrix(groups)
        cross = fm.cross_features()
        assert cross.shape[0] == 5
        assert cross.shape[1] > 0
