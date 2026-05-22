"""Phased training with 10% layer-shard gradient masking.

The trainer divides transformer blocks into phases. In each phase:
- Blocks outside the phase are frozen (requires_grad=False).
- Within each in-phase block, parameters get a gradient-mask hook that zeros
  gradients outside a 10% slice. Optimizer state stays full; only the slice updates.

Phase boundaries come from ml_training.distribution.PipelineParallelism so the
existing sharding math is reused.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from ml_training.architecture import TransformerArchitecture
from ml_training.distribution import PipelineParallelism
from ml_training.models.transformer import MiniTransformer, TransformerBlock


@dataclass
class PhaseConfig:
    """Identifies which transformer blocks (by index) are trainable in a phase."""

    phase_id: int
    block_indices: list[int]


@dataclass
class PhasedTrainingConfig:
    num_phases: int = 3
    steps_per_phase: int = 50
    batch_size: int = 8
    seq_len: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    shard_fraction: float = 0.10  # 10% of hidden-dim rows updated per layer
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 10
    # Optimizations (all opt-in; defaults preserve original behavior)
    gradient_checkpointing: bool = False    # ~25% slower, ~4-8x activation mem reduction
    mixed_precision: str = "fp32"           # "fp32" | "bf16" | "fp16"
    cpu_offload_frozen: bool = False        # move out-of-phase blocks to CPU between phases
    disable_shard_mask: bool = False        # full backprop within in-phase blocks
    end_to_end: bool = False                # train ALL blocks in every phase (disables phase freeze)


@dataclass
class PhaseResult:
    phase_id: int
    block_indices: list[int]
    steps: int
    final_loss: float
    losses: list[float] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    wall_time_sec: float = 0.0


def build_layer_shard_mask(
    shape: tuple[int, ...],
    fraction: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a boolean mask the same shape as a weight; True where gradient survives.

    Selects a deterministic 10% of output rows (axis 0). Per-tensor seed makes the
    choice reproducible.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    if len(shape) < 1:
        return torch.ones(shape, dtype=torch.bool, device=device)
    n_rows = shape[0]
    k = max(1, math.ceil(n_rows * fraction))
    perm = torch.randperm(n_rows, generator=g)[:k]
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[perm] = True
    return mask.to(device)


class PhasedTrainer:
    """Train a MiniTransformer in phases with a per-phase 10% gradient mask."""

    def __init__(
        self,
        model: MiniTransformer,
        config: PhasedTrainingConfig,
        checkpoint_dir: str = "artifacts/phase_checkpoints",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or _pick_device()
        # Apply optimization flags before moving model to device
        if config.gradient_checkpointing:
            self.model.use_gradient_checkpointing = True
        self.model.to(self.device)
        torch.manual_seed(config.seed)
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._autocast_dtype, self._autocast_supported = _resolve_autocast(
            config.mixed_precision, self.device
        )
        # If mixed precision requested but autocast not supported on this device,
        # fall back to a manual model cast (less safe than autocast but works on MPS).
        if self._autocast_dtype is not None and not self._autocast_supported:
            self.model.to(dtype=self._autocast_dtype)
            print(
                f"[trainer] autocast not supported on {self.device.type}; "
                f"manually cast model to {self._autocast_dtype} (no fp32 master weights)"
            )

    # ------------------------------------------------------------------ phases

    def plan_phases(self) -> list[PhaseConfig]:
        """Group transformer blocks into phases using PipelineParallelism partition."""
        arch = TransformerArchitecture(self.model.spec, name=self.model.spec.__class__.__name__)
        pp = PipelineParallelism(arch, num_gpus=self.config.num_phases)
        pp.partition()
        sublayer_count = arch.sublayers_per_block

        phases: list[PhaseConfig] = []
        for stage in pp.stages:
            block_set: set[int] = set()
            for layer in stage.layers:
                block_set.add(layer.layer_id // sublayer_count)
            phases.append(PhaseConfig(
                phase_id=stage.stage_id,
                block_indices=sorted(block_set),
            ))
        # Make sure every block is covered (defensive: if PP under-allocates due to rounding)
        covered = {b for ph in phases for b in ph.block_indices}
        missing = [i for i in range(self.model.spec.num_layers) if i not in covered]
        if missing:
            phases[-1].block_indices = sorted(set(phases[-1].block_indices) | set(missing))
        return phases

    # ------------------------------------------------------------------ masking

    def _set_trainable_blocks(self, block_indices: list[int]) -> None:
        """Freeze all params, then unfreeze params in the chosen blocks.

        If config.end_to_end is set, every block is trainable in every phase
        (the phase mechanism becomes a no-op freezer, useful for ablations).
        """
        in_phase = set(block_indices) if not self.config.end_to_end else set(range(len(self.model.blocks)))
        for p in self.model.parameters():
            p.requires_grad = False
        for i, block in enumerate(self.model.blocks):
            if i in in_phase:
                for p in block.parameters():
                    p.requires_grad = True
        # Always train the final LayerNorm + (tied) lm_head/embedding so the LM head can adapt
        for p in self.model.ln_f.parameters():
            p.requires_grad = True
        # Tied lm_head.weight == token_emb.weight; unfreeze once.
        self.model.token_emb.weight.requires_grad = True

        if self.config.cpu_offload_frozen and not self.config.end_to_end:
            self._cpu_offload(in_phase)

    def _cpu_offload(self, in_phase: set[int]) -> None:
        """Move out-of-phase block params to CPU, in-phase blocks to GPU.

        Saves working-set GPU memory at the cost of CPU↔GPU copies between phases.
        Frozen blocks still need to run forward for activations to flow, so they get
        loaded back per-step lazily — that's why this only pays for itself across
        many phases or very tight memory.
        """
        for i, block in enumerate(self.model.blocks):
            target = self.device if i in in_phase else torch.device("cpu")
            block.to(target)

    def _install_shard_hooks(self, block_indices: list[int]) -> None:
        """Attach gradient hooks zeroing gradients outside the 10% shard per weight.

        Skipped entirely when `disable_shard_mask` or `end_to_end` is set.
        """
        self._remove_hooks()
        if self.config.disable_shard_mask or self.config.end_to_end:
            return
        in_phase = set(block_indices)
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx not in in_phase:
                continue
            for name, p in block.named_parameters():
                if not p.requires_grad or p.dim() < 2:
                    continue  # skip biases / LayerNorms (1-D)
                seed = self.config.seed + 1000 * block_idx + hash(name) % 997
                mask = build_layer_shard_mask(
                    tuple(p.shape), self.config.shard_fraction, seed, p.device
                )
                handle = p.register_hook(
                    lambda grad, m=mask: grad * m.to(grad.dtype)
                )
                self._hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------ loop

    def train(
        self,
        data_iter: Iterable[torch.Tensor] | Iterator[torch.Tensor],
        resume: bool = False,
    ) -> list[PhaseResult]:
        """Run all phases. `data_iter` yields [B, T] long-tensor batches indefinitely.

        When resume=True, scans checkpoint_dir for the latest phase_N.pt, loads its
        state into the model, and continues training from phase N+1. Phase results
        for previously-completed phases are reconstructed from saved metadata so
        the returned list still covers all phases.
        """
        phases = self.plan_phases()
        it = iter(data_iter)
        results: list[PhaseResult] = []

        resume_from_phase = 0
        if resume:
            resume_from_phase, prior_results = self._load_resume_state(phases)
            results.extend(prior_results)
            if resume_from_phase >= len(phases):
                print(f"[trainer] all {len(phases)} phases already complete; nothing to do")
                return results
            print(
                f"[trainer] resuming from phase {resume_from_phase} "
                f"({len(phases) - resume_from_phase} phases remaining)"
            )

        for phase in phases[resume_from_phase:]:
            self._set_trainable_blocks(phase.block_indices)
            self._install_shard_hooks(phase.block_indices)

            trainable = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = AdamW(
                trainable,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.model.train()
            losses: list[float] = []
            t0 = time.time()
            for step in range(self.config.steps_per_phase):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                if self._autocast_dtype is not None and self._autocast_supported:
                    with torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype):
                        out = self.model(batch, labels=batch)
                        loss = out["loss"]
                else:
                    out = self.model(batch, labels=batch)
                    loss = out["loss"]
                # Always backward in fp32 if we manually cast; loss is already a scalar.
                if loss.dtype != torch.float32:
                    loss = loss.float()
                loss.backward()
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable, self.config.grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if step % self.config.log_every == 0:
                    print(
                        f"[phase {phase.phase_id} blocks={phase.block_indices}] "
                        f"step {step:>4} loss={losses[-1]:.4f}"
                    )

            ckpt_path = self.checkpoint_dir / f"phase_{phase.phase_id}.pt"
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                    "phase_id": phase.phase_id,
                    "block_indices": phase.block_indices,
                    "final_loss": losses[-1] if losses else float("nan"),
                },
                ckpt_path,
            )
            # Manifest line per phase
            with (self.checkpoint_dir / "phases.jsonl").open("a") as f:
                f.write(json.dumps({
                    "phase_id": phase.phase_id,
                    "block_indices": phase.block_indices,
                    "final_loss": losses[-1] if losses else None,
                    "checkpoint": str(ckpt_path),
                }) + "\n")

            results.append(PhaseResult(
                phase_id=phase.phase_id,
                block_indices=phase.block_indices,
                steps=len(losses),
                final_loss=losses[-1] if losses else float("nan"),
                losses=losses,
                checkpoint_path=str(ckpt_path),
                wall_time_sec=time.time() - t0,
            ))
            self._remove_hooks()
        return results


    def _load_resume_state(
        self, phases: list[PhaseConfig]
    ) -> tuple[int, list[PhaseResult]]:
        """Find the latest phase_N.pt, load its weights, and return (next_phase, prior_results)."""
        existing: list[Path] = sorted(
            self.checkpoint_dir.glob("phase_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not existing:
            return 0, []
        latest = existing[-1]
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        last_phase_id = int(ck.get("phase_id", -1))
        # Filter state dict to keys the current model expects (defensive against
        # arch tweaks between runs).
        current_keys = set(self.model.state_dict().keys())
        sd = {k: v for k, v in ck.get("state_dict", {}).items() if k in current_keys}
        missing, _ = self.model.load_state_dict(sd, strict=False)
        n_missing = len(missing.missing_keys) if hasattr(missing, "missing_keys") else 0
        print(
            f"[trainer] loaded {latest.name}: phase_id={last_phase_id} "
            f"loss={ck.get('final_loss')} tensors={len(sd)} missing={n_missing}"
        )

        # Reconstruct prior PhaseResults from disk metadata
        prior_results: list[PhaseResult] = []
        for p in existing:
            d = torch.load(p, map_location="cpu", weights_only=False)
            prior_results.append(PhaseResult(
                phase_id=int(d.get("phase_id", 0)),
                block_indices=list(d.get("block_indices", [])),
                steps=self.config.steps_per_phase,
                final_loss=float(d.get("final_loss", float("nan"))),
                losses=[],
                checkpoint_path=str(p),
                wall_time_sec=0.0,
            ))
        # Next phase to run is one past the highest completed phase_id
        next_phase = last_phase_id + 1
        return next_phase, prior_results


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_autocast(
    precision: str, device: torch.device
) -> tuple[Optional[torch.dtype], bool]:
    """Return (target_dtype, autocast_supported).

    Falls back gracefully:
    - autocast not supported on this device.type → manual cast (returns supported=False)
    - requested dtype not supported on this device  → tries fp16, then fp32 (with warning)
    """
    if precision == "fp32":
        return None, True
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if precision not in dtype_map:
        raise ValueError(f"Unknown mixed_precision: {precision}")
    dtype = dtype_map[precision]

    # Probe dtype on device: can we allocate + matmul?
    def _dtype_works(dt: torch.dtype) -> bool:
        try:
            x = torch.randn(2, 2, dtype=dt, device=device)
            _ = x @ x
            return True
        except (TypeError, RuntimeError):
            return False

    if not _dtype_works(dtype):
        # Try fp16 as a fallback if bf16 was requested
        if dtype == torch.bfloat16 and _dtype_works(torch.float16):
            print(
                f"[trainer] {precision} unsupported on {device.type}; falling back to fp16"
            )
            dtype = torch.float16
        else:
            print(
                f"[trainer] {precision} unsupported on {device.type}; falling back to fp32"
            )
            return None, True

    # MPS in torch < 2.4 has known mixed-dtype bugs (mps.select op fails when
    # f16 logits meet f32 loss accumulators). Detect torch version + MPS device
    # and force fp32 with a warning rather than crashing mid-training.
    if device.type == "mps":
        try:
            major, minor = (int(x) for x in torch.__version__.split(".")[:2])
            if (major, minor) < (2, 4):
                print(
                    f"[trainer] mixed precision on MPS is unreliable in torch {torch.__version__}; "
                    "falling back to fp32. Upgrade to torch >= 2.4 for fp16 on MPS."
                )
                return None, True
        except (ValueError, AttributeError):
            pass

    # Probe autocast for that device
    try:
        with torch.autocast(device_type=device.type, dtype=dtype):
            pass
        supported = True
    except (RuntimeError, AssertionError):
        supported = False
    return dtype, supported
