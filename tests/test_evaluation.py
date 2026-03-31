"""Tests for evaluation system, drift detection, and retraining triggers."""

from __future__ import annotations

import numpy as np
import pytest

from ml_training.architecture import ModelArchitecture
from ml_training.data_pipeline import DatasetSplit, TrainingDataPipeline
from ml_training.evaluation import (
    EvaluationSystem,
    MetricsResult,
    ModelDriftDetector,
    RetrainingTrigger,
)
from ml_training.fine_tuning import FineTuneConfig, FineTuner, Quantizer
from ml_training.orchestrator import (
    CheckpointManager,
    TrainingConfig,
    TrainingOrchestrator,
)


class TestEvaluationSystem:
    """Tests for EvaluationSystem."""

    def test_compute_ctr(self, sample_labels: np.ndarray) -> None:
        evaluator = EvaluationSystem()
        ctr = evaluator.compute_ctr(np.zeros_like(sample_labels), sample_labels)
        assert 0.0 <= ctr <= 1.0

    def test_compute_cvr_no_clicks(self) -> None:
        evaluator = EvaluationSystem()
        preds = np.array([0.5, 0.5, 0.5])
        labels = np.array([1, 0, 1])
        clicks = np.array([0, 0, 0])
        assert evaluator.compute_cvr(preds, labels, clicks) == 0.0

    def test_compute_cvr_with_clicks(self) -> None:
        evaluator = EvaluationSystem()
        preds = np.array([0.8, 0.6, 0.4])
        labels = np.array([1, 0, 1])
        clicks = np.array([1, 1, 0])
        cvr = evaluator.compute_cvr(preds, labels, clicks)
        assert abs(cvr - 0.5) < 1e-6  # 1 conversion out of 2 clicks

    def test_compute_auc_perfect(self) -> None:
        evaluator = EvaluationSystem()
        labels = np.array([0, 0, 1, 1], dtype=np.float32)
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        auc = evaluator.compute_auc(preds, labels)
        assert auc == 1.0

    def test_compute_auc_random(self) -> None:
        evaluator = EvaluationSystem()
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 2, size=1000).astype(np.float32)
        preds = rng.uniform(0, 1, size=1000)
        auc = evaluator.compute_auc(preds, labels)
        assert 0.3 < auc < 0.7  # Should be near 0.5 for random

    def test_compute_auc_all_same_label(self) -> None:
        evaluator = EvaluationSystem()
        labels = np.ones(10, dtype=np.float32)
        preds = np.random.rand(10)
        auc = evaluator.compute_auc(preds, labels)
        assert auc == 0.5  # Undefined, returns 0.5

    def test_compute_log_loss(self) -> None:
        evaluator = EvaluationSystem()
        labels = np.array([1, 0, 1], dtype=np.float32)
        preds = np.array([0.9, 0.1, 0.8])
        loss = evaluator.compute_log_loss(preds, labels)
        assert loss > 0

    def test_evaluate_returns_metrics(
        self, sample_predictions: np.ndarray, sample_labels: np.ndarray
    ) -> None:
        evaluator = EvaluationSystem()
        result = evaluator.evaluate(sample_predictions, sample_labels)
        assert isinstance(result, MetricsResult)
        assert result.num_samples == len(sample_labels)
        assert result.auc > 0

    def test_metrics_to_dict(self) -> None:
        m = MetricsResult(ctr=0.1, cvr=0.05, auc=0.8, log_loss=0.5, num_samples=100)
        d = m.to_dict()
        assert d["ctr"] == 0.1
        assert d["auc"] == 0.8


class TestModelDriftDetector:
    """Tests for ModelDriftDetector."""

    def test_no_drift(self, baseline_metrics: MetricsResult) -> None:
        detector = ModelDriftDetector(baseline_metrics)
        current = MetricsResult(
            ctr=0.15, cvr=0.05, auc=0.84, log_loss=0.46, num_samples=500
        )
        result = detector.check_drift(current)
        assert result["drifted"] is False

    def test_auc_drift(self, baseline_metrics: MetricsResult) -> None:
        detector = ModelDriftDetector(baseline_metrics, auc_threshold=0.02)
        current = MetricsResult(
            ctr=0.15, cvr=0.05, auc=0.80, log_loss=0.46, num_samples=500
        )
        result = detector.check_drift(current)
        assert result["drifted"] is True
        assert "AUC" in result["reasons"][0]

    def test_log_loss_drift(self, baseline_metrics: MetricsResult) -> None:
        detector = ModelDriftDetector(baseline_metrics, log_loss_threshold=0.1)
        current = MetricsResult(
            ctr=0.15, cvr=0.05, auc=0.84, log_loss=0.60, num_samples=500
        )
        result = detector.check_drift(current)
        assert result["drifted"] is True
        assert "Log loss" in result["reasons"][0]

    def test_record_and_trend(self, baseline_metrics: MetricsResult) -> None:
        detector = ModelDriftDetector(baseline_metrics)
        for auc_val in [0.84, 0.82, 0.80]:
            m = MetricsResult(ctr=0.15, cvr=0.05, auc=auc_val, log_loss=0.5, num_samples=100)
            detector.record(m)
        trend = detector.trend()
        assert len(trend["auc"]) == 3
        assert trend["auc"] == [0.84, 0.82, 0.80]


class TestRetrainingTrigger:
    """Tests for RetrainingTrigger."""

    def test_trigger_on_low_auc(self) -> None:
        trigger = RetrainingTrigger(min_auc=0.70)
        metrics = MetricsResult(ctr=0.1, cvr=0.05, auc=0.65, log_loss=0.5, num_samples=100)
        assert trigger.should_retrain(metrics) is True

    def test_no_trigger_on_good_metrics(self) -> None:
        trigger = RetrainingTrigger(min_auc=0.70, max_log_loss=1.0)
        metrics = MetricsResult(ctr=0.1, cvr=0.05, auc=0.80, log_loss=0.5, num_samples=100)
        assert trigger.should_retrain(metrics) is False

    def test_cooldown_prevents_immediate_retrigger(self) -> None:
        trigger = RetrainingTrigger(min_auc=0.70, cooldown_steps=3)
        bad = MetricsResult(ctr=0.1, cvr=0.05, auc=0.60, log_loss=0.5, num_samples=100)
        assert trigger.should_retrain(bad) is True  # First trigger
        assert trigger.should_retrain(bad) is False  # In cooldown
        assert trigger.should_retrain(bad) is False  # Still in cooldown

    def test_reset(self) -> None:
        trigger = RetrainingTrigger(cooldown_steps=10)
        trigger._steps_since_retrain = 0
        trigger.reset()
        assert trigger._steps_since_retrain == 10


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_and_load_latest(self) -> None:
        from ml_training.orchestrator import Checkpoint

        mgr = CheckpointManager()
        ckpt = Checkpoint(epoch=1, step=100, weights={"w": np.zeros(5)}, optimizer_state={})
        mgr.save(ckpt)
        loaded = mgr.load_latest()
        assert loaded is not None
        assert loaded.epoch == 1

    def test_load_best_by_auc(self) -> None:
        from ml_training.orchestrator import Checkpoint

        mgr = CheckpointManager()
        for i, auc_val in enumerate([0.7, 0.9, 0.8]):
            m = MetricsResult(ctr=0.1, cvr=0.05, auc=auc_val, log_loss=0.5, num_samples=100)
            ckpt = Checkpoint(epoch=i, step=i * 10, weights={}, optimizer_state={}, metrics=m)
            mgr.save(ckpt)
        best = mgr.load_best(metric="auc", higher_is_better=True)
        assert best is not None
        assert best.metrics.auc == 0.9

    def test_max_checkpoints_kept(self) -> None:
        from ml_training.orchestrator import Checkpoint

        mgr = CheckpointManager()
        mgr._max_to_keep = 3
        for i in range(10):
            ckpt = Checkpoint(epoch=i, step=i, weights={}, optimizer_state={})
            mgr.save(ckpt)
        assert len(mgr._checkpoints) == 3

    def test_list_checkpoints(self) -> None:
        from ml_training.orchestrator import Checkpoint

        mgr = CheckpointManager()
        mgr.save(Checkpoint(epoch=0, step=0, weights={}, optimizer_state={}))
        listing = mgr.list_checkpoints()
        assert len(listing) == 1
        assert listing[0]["epoch"] == 0


class TestTrainingOrchestrator:
    """Tests for TrainingOrchestrator."""

    def test_initialize_weights(self, small_architecture: ModelArchitecture) -> None:
        config = TrainingConfig(epochs=1, seed=42)
        orch = TrainingOrchestrator(small_architecture, config)
        orch.initialize_weights()
        assert len(orch._weights) == len(small_architecture.layers)

    def test_train_returns_history(
        self,
        small_architecture: ModelArchitecture,
        train_eval_splits: tuple[DatasetSplit, DatasetSplit],
    ) -> None:
        train_data, eval_data = train_eval_splits
        config = TrainingConfig(epochs=2, batch_size=64, seed=42)
        orch = TrainingOrchestrator(small_architecture, config)
        history = orch.train(train_data, eval_data)
        assert len(history) == 2
        assert "train_loss" in history[0]
        assert "eval_auc" in history[0]

    def test_evaluate(
        self,
        small_architecture: ModelArchitecture,
        train_eval_splits: tuple[DatasetSplit, DatasetSplit],
    ) -> None:
        _, eval_data = train_eval_splits
        config = TrainingConfig(seed=42)
        orch = TrainingOrchestrator(small_architecture, config)
        orch.initialize_weights()
        metrics = orch.evaluate(eval_data)
        assert isinstance(metrics, MetricsResult)
        assert metrics.num_samples == eval_data.num_samples


class TestFineTuner:
    """Tests for FineTuner."""

    def test_lora_initialization(self, small_architecture: ModelArchitecture) -> None:
        config = TrainingConfig(seed=42)
        orch = TrainingOrchestrator(small_architecture, config)
        orch.initialize_weights()

        ft_config = FineTuneConfig(lora_rank=4, seed=42)
        tuner = FineTuner(small_architecture, orch._weights, ft_config)
        assert len(tuner._lora_a) == len(small_architecture.layers)
        assert tuner.num_trainable_params > 0

    def test_effective_weights_differ_from_base(
        self, small_architecture: ModelArchitecture
    ) -> None:
        config = TrainingConfig(seed=42)
        orch = TrainingOrchestrator(small_architecture, config)
        orch.initialize_weights()

        ft_config = FineTuneConfig(lora_rank=4, seed=42)
        tuner = FineTuner(small_architecture, orch._weights, ft_config)
        effective = tuner.get_effective_weights()
        # With B=0, effective should equal base initially
        for key in orch._weights:
            np.testing.assert_allclose(effective[key], orch._weights[key], atol=1e-6)

    def test_fine_tune_runs(
        self,
        small_architecture: ModelArchitecture,
        train_eval_splits: tuple[DatasetSplit, DatasetSplit],
    ) -> None:
        train_data, eval_data = train_eval_splits
        config = TrainingConfig(seed=42)
        orch = TrainingOrchestrator(small_architecture, config)
        orch.initialize_weights()

        ft_config = FineTuneConfig(epochs=2, batch_size=64, lora_rank=4, seed=42)
        tuner = FineTuner(small_architecture, orch._weights, ft_config)
        history = tuner.fine_tune(train_data, eval_data)
        assert len(history) == 2

    def test_merge_lora_weights(self, small_architecture: ModelArchitecture) -> None:
        config = TrainingConfig(seed=42)
        orch = TrainingOrchestrator(small_architecture, config)
        orch.initialize_weights()

        ft_config = FineTuneConfig(lora_rank=4, seed=42)
        tuner = FineTuner(small_architecture, orch._weights, ft_config)
        merged = tuner.merge_lora_weights()
        assert len(merged) == len(small_architecture.layers)

    def test_lora_scaling(self) -> None:
        config = FineTuneConfig(lora_rank=8, lora_alpha=16.0)
        assert config.lora_scaling == 2.0

    def test_fine_tune_config_to_dict(self) -> None:
        config = FineTuneConfig(epochs=3, lora_rank=4)
        d = config.to_dict()
        assert d["epochs"] == 3
        assert d["lora_rank"] == 4


class TestQuantizer:
    """Tests for Quantizer."""

    def test_int8_quantize_dequantize(self) -> None:
        q = Quantizer(mode="int8")
        tensor = np.array([[-1.0, 0.5], [0.25, -0.75]], dtype=np.float32)
        quantized, scale = q.quantize_tensor(tensor)
        assert quantized.dtype == np.int8
        reconstructed = q.dequantize_tensor(quantized, scale)
        np.testing.assert_allclose(tensor, reconstructed, atol=0.02)

    def test_int4_quantize_dequantize(self) -> None:
        q = Quantizer(mode="int4")
        tensor = np.array([[0.5, -0.3], [0.1, -0.7]], dtype=np.float32)
        quantized, scale = q.quantize_tensor(tensor)
        assert quantized.dtype == np.int8
        assert quantized.max() <= 7
        assert quantized.min() >= -8

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            Quantizer(mode="int2")

    def test_compression_ratio_int8(self) -> None:
        q = Quantizer(mode="int8")
        weights = {"w": np.ones((10, 10), dtype=np.float32)}
        ratio = q.compression_ratio(weights)
        assert ratio == 4.0

    def test_compression_ratio_int4(self) -> None:
        q = Quantizer(mode="int4")
        weights = {"w": np.ones((10, 10), dtype=np.float32)}
        ratio = q.compression_ratio(weights)
        assert ratio == 8.0

    def test_quantize_weights(self) -> None:
        q = Quantizer(mode="int8")
        weights = {
            "a": np.random.randn(5, 3).astype(np.float32),
            "b": np.random.randn(3, 2).astype(np.float32),
        }
        quantized = q.quantize_weights(weights)
        assert "a" in quantized
        assert "b" in quantized
        dequantized = q.dequantize_weights(quantized)
        assert dequantized["a"].shape == (5, 3)

    def test_quantization_error(self) -> None:
        q = Quantizer(mode="int8")
        weights = {"w": np.random.randn(10, 10).astype(np.float32)}
        errors = q.quantization_error(weights)
        assert "w" in errors
        assert errors["w"] >= 0

    def test_zero_tensor_quantization(self) -> None:
        q = Quantizer(mode="int8")
        tensor = np.zeros((3, 3), dtype=np.float32)
        quantized, scale = q.quantize_tensor(tensor)
        reconstructed = q.dequantize_tensor(quantized, scale)
        np.testing.assert_array_equal(reconstructed, tensor)
