"""Command-line interface for the ML Training Platform."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import click
import numpy as np

from ml_training.architecture import ArchitectureAnalyzer, ModelArchitecture, TransformerArchitecture
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


@cli.command(name="profile-transformer")
@click.option("--preset", default="local-default", help="Architecture preset name.")
@click.option("--num-gpus", default=1, help="Number of GPUs for sharding analysis.")
def profile_transformer(preset: str, num_gpus: int) -> None:
    """Profile a transformer preset (e.g. 256L-base) — no real model materialization."""
    arch = TransformerArchitecture.from_preset(preset)
    spec = arch.spec
    click.echo(f"Preset: {preset}")
    click.echo(
        f"  num_layers={spec.num_layers} num_heads={spec.num_heads} "
        f"d_model={spec.d_model} d_ff={spec.d_ff} vocab={spec.vocab_size}"
    )
    click.echo(f"  total params (analytical): {arch.parameter_count():,}")
    fp32_gb = arch.parameter_count() * 4 / 1e9
    int8_gb = arch.parameter_count() / 1e9
    click.echo(f"  weight memory: fp32={fp32_gb:.3f} GB, int8={int8_gb:.3f} GB")

    analyzer = ArchitectureAnalyzer(arch)
    splits = analyzer.recommend_split_points(num_gpus)
    click.echo(f"  recommended split points for {num_gpus} GPUs: {splits}")


@cli.command(name="personalize")
@click.option("--config", "config_path", default=None, help="Path to platform.yaml.")
@click.option("--principles", "principles_path", default=None, help="Override principles file.")
@click.option("--base-preset", default=None, help="Override base preset.")
@click.option("--dataset", default=None, help="Override dataset name.")
@click.option("--max-records", default=None, type=int, help="Override dataset max records.")
@click.option("--quant", default=None, help="Override quant (fp32|fp16|int8|auto).")
@click.option("--no-train", is_flag=True, help="Skip phased training (use random-init weights).")
@click.option("--no-personalize", is_flag=True, help="Skip principles LoRA fine-tune.")
@click.option(
    "--mock-dataset",
    is_flag=True,
    help="Use a small synthetic corpus instead of pulling from HuggingFace (offline mode).",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume phased training from the latest checkpoint in artifacts/phase_checkpoints/.",
)
def personalize(
    config_path: Optional[str],
    principles_path: Optional[str],
    base_preset: Optional[str],
    dataset: Optional[str],
    max_records: Optional[int],
    quant: Optional[str],
    no_train: bool,
    no_personalize: bool,
    mock_dataset: bool,
    resume: bool,
) -> None:
    """End-to-end: data → tokenize → phased train → personalize via LoRA → package → eval."""
    # Imports kept local so `ml-train --help` does not pay torch's import cost.
    import torch

    from ml_training.architecture import TransformerArchitecture
    from ml_training.control_plane import (
        SourceLinks,
        WeightRegistry,
        load_platform_config,
        plan_deployment,
    )
    from ml_training.control_plane.sources import hash_file
    from ml_training.data_sources import FineWebLoader
    from ml_training.evaluation_judge import (
        ModelJudge,
        aggregate_judge,
        compute_perplexity,
        load_judge_corpus,
    )
    from ml_training.models.transformer import MiniTransformer
    from ml_training.packaging import WeightPackager
    from ml_training.personalization import (
        LoRAConfig,
        LoRATrainer,
        load_principles,
        merge_lora,
        principles_to_pairs,
    )
    from ml_training.personalization.principles_to_dataset import principles_hash
    from ml_training.serving import LocalServer, ServingConfig, ServingMode
    from ml_training.tokenization import BPETokenizer, TokenizerConfig
    from ml_training.training import (
        CheckpointAverager,
        PhasedTrainer,
        PhasedTrainingConfig,
    )

    cfg = load_platform_config(config_path) if config_path else load_platform_config(None)
    if principles_path:
        cfg.personalization.principles_file = principles_path
    if base_preset:
        cfg.training.base_preset = base_preset
    if dataset:
        cfg.data_plane.dataset = dataset
    if max_records is not None:
        cfg.data_plane.max_records = max_records
    if quant:
        cfg.hardware.prefer_quant = quant  # type: ignore[assignment]

    click.echo("=" * 72)
    click.echo("Customized Model Factory — personalize")
    click.echo("=" * 72)

    # 1. Architecture
    arch = TransformerArchitecture.from_preset(cfg.training.base_preset)
    click.echo(f"[arch] preset={cfg.training.base_preset} params={arch.parameter_count():,}")

    # 2. Hardware plan
    plan = plan_deployment(
        arch.spec,
        prefer_quant=cfg.hardware.prefer_quant,
        max_memory_gb=cfg.hardware.max_memory_gb,
        allow_partial_fallback=cfg.hardware.allow_partial_fallback,
    )
    click.echo(f"[hw] device={plan.device.name} ({plan.device.kind}, {plan.device.memory_gb:.1f}GB)")
    click.echo(f"[hw] chosen quant={plan.quant} mode={plan.serving_mode} fits={plan.fits}")

    # 3. Data source
    def _real_texts():
        loader = FineWebLoader(dataset_name=cfg.data_plane.dataset)
        return loader.texts(max_records=cfg.data_plane.max_records)

    def _mock_texts():
        sample = (
            "The personal model factory adapts a base transformer to user preferences. "
            "Concise prose is preferred over bullet lists. Cite sources for factual claims. "
            "Avoid speculative phrasing unless the user invites it. Training is phased and "
            "uses a small fraction of parameters per step to fit on a single local GPU. "
        )
        return iter([sample] * max(cfg.data_plane.max_records, 50))

    text_factory = _mock_texts if mock_dataset else _real_texts

    # 4. Tokenizer (train-or-load)
    tk = BPETokenizer(TokenizerConfig(
        vocab_size=cfg.data_plane.vocab_size,
        cache_path=cfg.data_plane.tokenizer_cache,
    ))
    tk.load_or_train(text_factory)
    click.echo(f"[tokenizer] vocab_size={tk.vocab_size}")

    # Adjust arch vocab to match real tokenizer vocab
    if tk.vocab_size != arch.spec.vocab_size:
        from dataclasses import replace
        new_spec = replace(arch.spec, vocab_size=tk.vocab_size)
        arch = TransformerArchitecture(new_spec, name=cfg.training.base_preset)
        click.echo(f"[arch] vocab adjusted to {tk.vocab_size}")

    # 5. Materialize real model
    model = MiniTransformer.from_arch(arch)
    click.echo(f"[model] real params={model.num_parameters():,}")

    # 6. Phased training
    ckpt_dir = Path("artifacts/phase_checkpoints")
    if resume:
        click.echo(f"[train] resume mode — keeping existing checkpoints in {ckpt_dir}")
    else:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not no_train:
        ptc = PhasedTrainingConfig(
            num_phases=cfg.training.num_phases,
            steps_per_phase=cfg.training.steps_per_phase,
            batch_size=cfg.data_plane.batch_size,
            seq_len=cfg.data_plane.seq_len,
            learning_rate=cfg.training.learning_rate,
            shard_fraction=cfg.training.shard_fraction,
            seed=cfg.training.seed,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            mixed_precision=cfg.training.mixed_precision,
            cpu_offload_frozen=cfg.training.cpu_offload_frozen,
            disable_shard_mask=cfg.training.disable_shard_mask,
            end_to_end=cfg.training.end_to_end,
        )
        trainer = PhasedTrainer(model, ptc, checkpoint_dir=str(ckpt_dir))

        def _batches():
            while True:
                for b in tk.stream_token_batches(
                    text_factory(), batch_size=ptc.batch_size, seq_len=ptc.seq_len,
                ):
                    yield b

        phase_results = trainer.train(_batches(), resume=resume)
        click.echo("[train] phase results:")
        for r in phase_results:
            click.echo(
                f"  phase {r.phase_id} blocks={r.block_indices} "
                f"steps={r.steps} final_loss={r.final_loss:.4f} t={r.wall_time_sec:.1f}s"
            )

        # 7. Checkpoint averaging
        averager = CheckpointAverager(str(ckpt_dir))
        averaged, _stats = averager.converge()
        # Filter to keys that exist in the current model state_dict (defensive)
        current_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in averaged.items() if k in current_keys}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        click.echo(f"[converge] averaged {len(averaged)} tensors; "
                   f"missing={len(missing.missing_keys) if hasattr(missing, 'missing_keys') else 0}")

    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    import hashlib
    base_hash = hashlib.sha256(
        b"".join(v.numpy().tobytes() for v in base_state.values())
    ).hexdigest()[:16]

    # 8. Personalization (LoRA)
    p_hash: Optional[str] = None
    if not no_personalize:
        principles = load_principles(cfg.personalization.principles_file)
        p_hash = principles_hash(principles)
        click.echo(f"[personalize] {len(principles)} principles, hash={p_hash}")
        ds = principles_to_pairs(principles)
        click.echo(f"[personalize] preferred examples: {len(ds.preferred)}")

        lora_trainer = LoRATrainer(
            model,
            LoRAConfig(rank=cfg.personalization.lora_rank, alpha=cfg.personalization.lora_alpha),
            learning_rate=cfg.personalization.learning_rate,
        )
        click.echo(f"[personalize] trainable params: {lora_trainer.trainable_param_count:,}")

        def _principle_batches():
            return tk.stream_token_batches(
                iter(ds.preferred * 8), batch_size=cfg.data_plane.batch_size,
                seq_len=cfg.data_plane.seq_len,
            )

        losses = lora_trainer.train(_principle_batches(), epochs=cfg.personalization.epochs)
        if losses:
            click.echo(f"[personalize] LoRA losses: first={losses[0]:.3f} last={losses[-1]:.3f}")

        if cfg.personalization.merge_at_packaging:
            merge_lora(model)
            click.echo("[personalize] merged LoRA into base weights")

    # 9. Packaging
    src = SourceLinks(
        dataset_uri=cfg.data_plane.dataset,
        tokenizer_uri=str(Path(cfg.data_plane.tokenizer_cache).resolve()),
        principles_uri=str(Path(cfg.personalization.principles_file).resolve()),
    )
    registry = WeightRegistry(root=cfg.registry.root)
    bundle = registry.packager.save(
        model.state_dict(),
        arch.spec,
        arch_preset=cfg.training.base_preset,
        quant=plan.quant,
        base_hash=base_hash,
        principles_hash=p_hash,
        source_uris=src.as_dict(),
    )
    click.echo(
        f"[package] bundle_id={bundle.bundle_id} quant={plan.quant} "
        f"raw={bundle.raw_size_bytes/1e6:.2f}MB cmp={bundle.compressed_size_bytes/1e6:.2f}MB"
    )
    registry.set_alias("latest", bundle.bundle_id)

    # 10. Local serving probe + judge eval
    serving_mode = ServingMode.PARTIAL if plan.serving_mode == "partial" else (
        ServingMode.INT8 if plan.quant == "int8" else ServingMode.FULL
    )
    serving_cfg = ServingConfig(mode=serving_mode, partial_blocks=plan.partial_blocks)
    server = LocalServer(serving_cfg).load(bundle.bundle_dir).attach_tokenizer(tk)

    judge_corpus_path = Path("tests/data/judge_corpus.jsonl")
    if judge_corpus_path.exists():
        corpus = load_judge_corpus(judge_corpus_path)
        judge = ModelJudge(principles=[
            p.strip() for p in Path(cfg.personalization.principles_file).read_text().splitlines()
            if p.strip() and not p.strip().startswith("#") and not p.strip().startswith("principles:")
        ])
        from ml_training.serving import GenerationRequest
        results = []
        for item in corpus[:10]:
            resp = server.generate(GenerationRequest(prompt=item.prompt, max_tokens=24))
            results.append(judge.score(
                prompt=item.prompt, response=resp.text,
                reference=item.reference, applicable_principles=item.applicable_principles,
            ))
        agg = aggregate_judge(results)
        click.echo(f"[judge] avg_score={agg.extra['judge_score']:.3f} ({len(results)} prompts)")

    # Perplexity probe (cheap)
    try:
        device = next(model.parameters()).device
        sample_batches = list(tk.stream_token_batches(
            text_factory(), batch_size=cfg.data_plane.batch_size, seq_len=cfg.data_plane.seq_len,
        ))[:5]
        if sample_batches:
            ppl = compute_perplexity(model, sample_batches, device=device)
            click.echo(f"[eval] held-out perplexity (5 batches): {ppl:.2f}")
    except Exception as e:  # noqa: BLE001
        click.echo(f"[eval] perplexity skipped: {e}")

    click.echo("=" * 72)
    click.echo(f"Bundle at: {bundle.bundle_dir}")
    click.echo(f"Serve: ml-train serve --bundle {bundle.bundle_dir}")
    click.echo("=" * 72)


@cli.command(name="serve")
@click.option("--bundle", "bundle_dir", required=True, help="Path to bundle directory.")
@click.option("--mode", default="full", type=click.Choice(["full", "partial", "int8"]))
@click.option("--partial-blocks", default=None, type=int)
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8080)
@click.option("--tokenizer", "tokenizer_path", default="artifacts/tokenizer/tokenizer.json")
def serve(
    bundle_dir: str,
    mode: str,
    partial_blocks: Optional[int],
    host: str,
    port: int,
    tokenizer_path: str,
) -> None:
    """Serve a packaged bundle over HTTP."""
    from ml_training.serving import LocalServer, ServingConfig, ServingMode
    from ml_training.tokenization import BPETokenizer, TokenizerConfig

    tk = BPETokenizer(TokenizerConfig(cache_path=tokenizer_path)).load()
    cfg = ServingConfig(mode=ServingMode(mode), partial_blocks=partial_blocks)
    server = LocalServer(cfg).load(bundle_dir).attach_tokenizer(tk)
    server.serve_http(host=host, port=port)


if __name__ == "__main__":
    cli()
