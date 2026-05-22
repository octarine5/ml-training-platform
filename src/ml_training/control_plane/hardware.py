"""Hardware detection + deployment planner.

Detects local accelerators (CUDA, Apple MPS, CPU), measures or estimates available
memory, and picks a quantization mode + serving mode that fits.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import torch

from ml_training.architecture import TransformerSpec


QuantMode = Literal["fp32", "fp16", "int8"]


@dataclass
class DeviceSpec:
    name: str
    kind: Literal["cuda", "mps", "cpu"]
    memory_gb: float
    supports_fp16: bool
    supports_int8: bool


@dataclass
class DeploymentPlan:
    device: DeviceSpec
    quant: QuantMode
    serving_mode: Literal["full", "partial"]
    partial_blocks: Optional[int]
    estimated_weight_gb: float
    fits: bool
    reasoning: list[str]

    def as_dict(self) -> dict:
        return {
            "device": asdict(self.device),
            "quant": self.quant,
            "serving_mode": self.serving_mode,
            "partial_blocks": self.partial_blocks,
            "estimated_weight_gb": round(self.estimated_weight_gb, 3),
            "fits": self.fits,
            "reasoning": self.reasoning,
        }


class HardwarePool:
    """Detect available local devices."""

    @staticmethod
    def detect() -> DeviceSpec:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return DeviceSpec(
                name=props.name,
                kind="cuda",
                memory_gb=props.total_memory / 1e9,
                supports_fp16=True,
                supports_int8=True,
            )
        if torch.backends.mps.is_available():
            # MPS doesn't expose memory directly; use Apple-Silicon unified memory as a proxy.
            try:
                import subprocess
                out = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2
                ).stdout.strip()
                mem_gb = float(out) / 1e9 if out.isdigit() else 8.0
            except Exception:
                mem_gb = 8.0
            return DeviceSpec(
                name=f"Apple {platform.machine()} (MPS)",
                kind="mps",
                memory_gb=mem_gb,
                supports_fp16=True,
                supports_int8=False,  # MPS int8 matmul not generally supported
            )
        # CPU fallback
        try:
            import subprocess
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2
            ).stdout.strip()
            mem_gb = float(out) / 1e9 if out.isdigit() else 8.0
        except Exception:
            mem_gb = 8.0
        return DeviceSpec(
            name=platform.processor() or "CPU",
            kind="cpu",
            memory_gb=mem_gb,
            supports_fp16=False,
            supports_int8=True,
        )


def _weight_bytes(spec: TransformerSpec, quant: QuantMode) -> float:
    # Param count (same formula as TransformerArchitecture.parameter_count)
    embed = spec.vocab_size * spec.d_model
    per_block = 4 * spec.d_model * spec.d_model + 2 * spec.d_model * spec.d_ff + 4 * spec.d_model
    final_ln = 2 * spec.d_model
    params = embed + per_block * spec.num_layers + final_ln
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1}[quant]
    # Add ~20% overhead for optimizer state during inference is 0, but activations + kv cache
    return params * bytes_per_param * 1.2


def plan_deployment(
    spec: TransformerSpec,
    device: Optional[DeviceSpec] = None,
    prefer_quant: QuantMode | Literal["auto"] = "auto",
    max_memory_gb: Optional[float] = None,
    allow_partial_fallback: bool = True,
) -> DeploymentPlan:
    """Pick a quantization + serving mode that fits the available memory."""
    device = device or HardwarePool.detect()
    budget = max_memory_gb if max_memory_gb is not None else device.memory_gb * 0.5
    reasoning: list[str] = [f"Detected {device.kind}: {device.name} ({device.memory_gb:.1f} GB)"]
    reasoning.append(f"Memory budget for weights: {budget:.2f} GB")

    if prefer_quant == "auto":
        candidates: list[QuantMode] = ["fp32", "fp16", "int8"]
        if not device.supports_fp16:
            candidates = [c for c in candidates if c != "fp16"]
        if not device.supports_int8:
            candidates = [c for c in candidates if c != "int8"]
    else:
        candidates = [prefer_quant]

    chosen: QuantMode = candidates[0]
    fits = False
    est = 0.0
    for q in candidates:
        est_bytes = _weight_bytes(spec, q)
        est_gb = est_bytes / 1e9
        reasoning.append(f"Estimated weight footprint at {q}: {est_gb:.2f} GB")
        if est_gb <= budget:
            chosen = q
            est = est_gb
            fits = True
            break
    else:
        chosen = candidates[-1]
        est = _weight_bytes(spec, chosen) / 1e9

    serving_mode: Literal["full", "partial"] = "full"
    partial_blocks: Optional[int] = None
    if not fits and allow_partial_fallback:
        # Choose the largest N blocks that fit at the chosen quant
        budget_per_block = (
            _weight_bytes(TransformerSpec(
                num_layers=1, num_heads=spec.num_heads, d_model=spec.d_model,
                d_ff=spec.d_ff, vocab_size=spec.vocab_size, max_seq_len=spec.max_seq_len,
                dtype=spec.dtype,
            ), chosen) / 1e9
        )
        # Embeddings count once; remaining budget after embeddings goes to blocks
        embed_only = TransformerSpec(
            num_layers=0, num_heads=spec.num_heads, d_model=spec.d_model,
            d_ff=spec.d_ff, vocab_size=spec.vocab_size, max_seq_len=spec.max_seq_len,
            dtype=spec.dtype,
        )
        embed_gb = _weight_bytes(embed_only, chosen) / 1e9
        per_block_gb = max((budget_per_block - embed_gb), 1e-6)
        partial_blocks = max(1, int((budget - embed_gb) / per_block_gb))
        partial_blocks = min(partial_blocks, spec.num_layers)
        serving_mode = "partial"
        reasoning.append(
            f"Full model does not fit; falling back to partial-{partial_blocks} of {spec.num_layers} blocks."
        )

    return DeploymentPlan(
        device=device,
        quant=chosen,
        serving_mode=serving_mode,
        partial_blocks=partial_blocks,
        estimated_weight_gb=est,
        fits=fits,
        reasoning=reasoning,
    )
