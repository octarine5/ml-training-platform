"""Real LoRA: inject low-rank adapters on Q and V projections of every block.

Trainable params = 2 * (d_model * r + r * d_model) per block. Base weights are
frozen. Merge-into-base is supported for serving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ml_training.models.transformer import CausalSelfAttention, MiniTransformer


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target_projections: tuple[str, ...] = ("q_proj", "v_proj")

    @property
    def scaling(self) -> float:
        return self.alpha / max(self.rank, 1)


class LoRALinear(nn.Module):
    """Wraps an nn.Linear with `W x + scaling * (B @ A) x`. Base weight is frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base.in_features
        out_f = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.empty(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def scaling(self) -> float:
        return self.alpha / max(self.rank, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = F.linear(F.linear(x, self.A), self.B) * self.scaling
        return base_out + delta

    def merge_into_base(self) -> nn.Linear:
        """Fold the LoRA delta into the base linear and return the merged module."""
        with torch.no_grad():
            delta_W = self.B @ self.A * self.scaling
            self.base.weight.data = self.base.weight.data + delta_W
        return self.base


def inject_lora(model: MiniTransformer, config: LoRAConfig) -> list[LoRALinear]:
    """Replace q_proj / v_proj on each block with LoRALinear wrappers. Returns adapters."""
    adapters: list[LoRALinear] = []
    for block in model.blocks:
        attn: CausalSelfAttention = block.attn
        for proj_name in config.target_projections:
            base = getattr(attn, proj_name)
            if isinstance(base, LoRALinear):
                continue  # already injected
            wrapped = LoRALinear(base, rank=config.rank, alpha=config.alpha)
            wrapped.to(base.weight.device)
            setattr(attn, proj_name, wrapped)
            adapters.append(wrapped)
    # Freeze everything except adapter params; unfreeze the tied embedding + final LN
    for p in model.parameters():
        p.requires_grad = False
    for a in adapters:
        a.A.requires_grad = True
        a.B.requires_grad = True
    for p in model.ln_f.parameters():
        p.requires_grad = True
    return adapters


def merge_lora(model: MiniTransformer) -> None:
    """Merge all LoRA wrappers in-place; model becomes a plain transformer again."""
    for block in model.blocks:
        attn: CausalSelfAttention = block.attn
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(attn, proj_name, None)
            if isinstance(mod, LoRALinear):
                base = mod.merge_into_base()
                setattr(attn, proj_name, base)


class LoRATrainer:
    """Supervised next-token training, updating only LoRA params."""

    def __init__(
        self,
        model: MiniTransformer,
        config: LoRAConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> None:
        self.model = model
        self.config = config
        self.adapters = inject_lora(model, config)
        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train(self, token_batches: Iterable[torch.Tensor], epochs: int = 1) -> list[float]:
        losses: list[float] = []
        device = next(self.model.parameters()).device
        self.model.train()
        for _epoch in range(epochs):
            for batch in token_batches:
                batch = batch.to(device)
                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(batch, labels=batch)
                out["loss"].backward()
                self.optimizer.step()
                losses.append(float(out["loss"].detach().cpu()))
        return losses

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
