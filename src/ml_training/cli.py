"""Command-line interface for the ML Training Platform."""

from __future__ import annotations

import json
from typing import Optional

import click
import numpy as np

from ml_training.architecture import ArchitectureAnalyzer, ModelArchitecture
from ml_training.data_pipeline import TrainingDataPipeline
from ml_training.evaluation import EvaluationSystem, ModelDriftDetector, RetrainingTrigger
from ml_training.fine_tuning import FineTuneConfig, FineTuner, Quantizer
from ml_training.orchestrator import CheckpointManager, TrainingConfig, TrainingOrchestrator


def _build_demo_architecture(
    num_layers: int = 4, input_dim: int = 64
) -> ModelArchitecture:
    """Build a demo architecture for CLI commands."""
    arch = ModelArchitecture(name="demo-model")
    dims = [input_dim]
    for i in range(num_layers):
        out_dim = max(16, input_dim // (2 ** (i + 1)))
        dims.append(out_dim)

    for i in range(num_layers):
        activation = "sigmoid" if i == num_layers - 1 else "relu"
        arch.add_layer(
            num_nodes=dims[i + 1],
            cardinality=dims[i],
            activation=activation,
        )
    arch.build()
    return arch


def _generate_demo_data(
    num_samples: int, input_dim: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic demo data."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((num_samples, input_dim)).astype(np.float32)
    labels = rng.integers(0, 2, size=num_samples).astype(np.float32)
    return features, labels


@click.group()
@click.version_option(package_name="ml-training-platform")
def cli() -> None:
    """ML Training Platform - distributed training, evaluation, and fine-tuning."""
    pass


@cli.command()
@click.option("--epochs", default=5, help="Number of training epochs.")
@click.option("--batch-size", default=128, help="Batch size for training.")
@click.option("--learning-rate", default=0.001, type=float, help="Learning rate.")
@click.option("--num-samples", default=1000, help="Number of synthetic training samples.")
@click.option("--input-dim", default=64, help="Input feature dimension.")
@click.option("--seed", default=42, help="Random seed.")
def train(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    num_samples: int,
    input_dim: int,
    seed: int,
) -> None:
    """Train a model on synthetic data."""
    click.echo(f"Building architecture (input_dim={input_dim})...")
    arch = _build_demo_architecture(input_dim=input_dim)

    click.echo(f"Generating {num_samples} synthetic samples...")
    features, labels = _generate_demo_data(num_samples, input_dim, seed)

    pipeline = TrainingDataPipeline(seed=seed)
    pipeline.load(features, labels)
    train_data, eval_data = pipeline.split()

    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    click.echo(f"Training for {epochs} epochs...")
    orchestrator = TrainingOrchestrator(arch, config)
    history = orchestrator.train(train_data, eval_data)

    click.echo("\nTraining complete. Results:")
    for entry in history:
        click.echo(
            f"  Epoch {entry['epoch']}: "
            f"loss={entry['train_loss']:.4f}, "
            f"auc={entry['eval_auc']:.4f}, "
            f"log_loss={entry['eval_log_loss']:.4f}"
        )


@cli.command()
@click.option("--num-samples", default=500, help="Number of evaluation samples.")
@click.option("--input-dim", default=64, help="Input feature dimension.")
@click.option("--seed", default=42, help="Random seed.")
def evaluate(num_samples: int, input_dim: int, seed: int) -> None:
    """Evaluate a model on synthetic data."""
    click.echo("Building architecture and generating data...")
    arch = _build_demo_architecture(input_dim=input_dim)
    features, labels = _generate_demo_data(num_samples, input_dim, seed)

    config = TrainingConfig(epochs=3, seed=seed)
    orchestrator = TrainingOrchestrator(arch, config)
    orchestrator.initialize_weights()

    pipeline = TrainingDataPipeline(seed=seed)
    pipeline.load(features, labels)
    _, eval_data = pipeline.split(train_ratio=0.5, eval_ratio=0.5)

    click.echo("Running evaluation...")
    metrics = orchestrator.evaluate(eval_data)

    click.echo("\nEvaluation Results:")
    for key, value in metrics.to_dict().items():
        click.echo(f"  {key}: {value}")


@cli.command(name="fine-tune")
@click.option("--epochs", default=3, help="Number of fine-tuning epochs.")
@click.option("--batch-size", default=64, help="Batch size.")
@click.option("--learning-rate", default=0.0001, type=float, help="Learning rate.")
@click.option("--lora-rank", default=8, help="LoRA rank.")
@click.option("--num-samples", default=500, help="Number of samples.")
@click.option("--input-dim", default=64, help="Input feature dimension.")
@click.option("--quantize", type=click.Choice(["none", "int8", "int4"]), default="none",
              help="Quantization mode after fine-tuning.")
@click.option("--seed", default=42, help="Random seed.")
def fine_tune(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_rank: int,
    num_samples: int,
    input_dim: int,
    quantize: str,
    seed: int,
) -> None:
    """Fine-tune a pre-trained model with LoRA."""
    click.echo("Building architecture and initializing base weights...")
    arch = _build_demo_architecture(input_dim=input_dim)

    # Create base weights via a brief pre-training
    base_config = TrainingConfig(epochs=1, seed=seed)
    orchestrator = TrainingOrchestrator(arch, base_config)
    orchestrator.initialize_weights()
    base_weights = orchestrator._weights

    click.echo(f"Generating {num_samples} domain-specific samples...")
    features, labels = _generate_demo_data(num_samples, input_dim, seed + 100)

    pipeline = TrainingDataPipeline(seed=seed)
    pipeline.load(features, labels)
    train_data, eval_data = pipeline.split()

    ft_config = FineTuneConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lora_rank=lora_rank,
        seed=seed,
    )

    click.echo(f"Fine-tuning with LoRA (rank={lora_rank}) for {epochs} epochs...")
    tuner = FineTuner(arch, base_weights, ft_config)
    click.echo(f"Trainable parameters: {tuner.num_trainable_params}")

    history = tuner.fine_tune(train_data, eval_data)

    click.echo("\nFine-tuning complete. Results:")
    for entry in history:
        click.echo(
            f"  Epoch {entry['epoch']}: "
            f"loss={entry['train_loss']:.4f}, "
            f"auc={entry['eval_auc']:.4f}"
        )

    if quantize != "none":
        click.echo(f"\nApplying {quantize} quantization...")
        merged = tuner.merge_lora_weights()
        quantizer = Quantizer(mode=quantize)
        ratio = quantizer.compression_ratio(merged)
        errors = quantizer.quantization_error(merged)
        click.echo(f"Compression ratio: {ratio:.1f}x")
        click.echo("Quantization error per layer:")
        for name, err in errors.items():
            click.echo(f"  {name}: {err:.6f}")


@cli.command(name="profile-model")
@click.option("--num-layers", default=4, help="Number of model layers.")
@click.option("--input-dim", default=128, help="Input feature dimension.")
@click.option("--num-gpus", default=4, help="Number of GPUs for distribution analysis.")
def profile_model(num_layers: int, input_dim: int, num_gpus: int) -> None:
    """Profile a model architecture for compute and memory analysis."""
    click.echo(f"Building {num_layers}-layer architecture (input_dim={input_dim})...")
    arch = _build_demo_architecture(num_layers=num_layers, input_dim=input_dim)

    analyzer = ArchitectureAnalyzer(arch)

    click.echo("\nModel Summary:")
    click.echo(f"  Total parameters: {arch.total_parameters:,}")
    click.echo(f"  Total memory: {arch.total_memory_bytes / 1024:.1f} KB")

    click.echo("\nPer-layer breakdown:")
    for info in arch.summary():
        click.echo(
            f"  Layer {info['layer_id']}: "
            f"params={info['params']:,}, "
            f"memory={info['memory_mb']:.4f} MB, "
            f"FLOPs={info['flops']:.0f}"
        )

    click.echo(f"\nCompute ratios: {[f'{r:.2%}' for r in analyzer.compute_ratios()]}")
    click.echo(f"Memory ratios:  {[f'{r:.2%}' for r in analyzer.memory_ratios()]}")

    split_points = analyzer.recommend_split_points(num_gpus)
    click.echo(f"\nRecommended split points for {num_gpus} GPUs: {split_points}")

    bottlenecks = analyzer.bottleneck_layers(top_k=min(3, num_layers))
    click.echo("\nBottleneck layers (by FLOPs):")
    for layer in bottlenecks:
        click.echo(f"  Layer {layer.layer_id}: {layer.flops:.0f} FLOPs")


if __name__ == "__main__":
    cli()
