"""Convergence step: average phase checkpoints with outlier rejection.

For each tensor, stack the values across the last-K phase checkpoints, compute
per-element mean and std, drop any value where |x - mean| > z_threshold * std,
and take the mean of the survivors. Falls back to the simple mean when std=0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


def average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    z_threshold: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Per-element robust mean across N state_dicts, rejecting |z| > z_threshold."""
    if not state_dicts:
        raise ValueError("Need at least one state_dict to average")
    keys = state_dicts[0].keys()
    averaged: dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        if stacked.shape[0] == 1:
            averaged[k] = stacked[0].clone()
            continue
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)
        # Where std == 0, every value is the mean — no outliers to reject.
        z = torch.where(
            std > 1e-12,
            (stacked - mean.unsqueeze(0)).abs() / (std.unsqueeze(0) + 1e-12),
            torch.zeros_like(stacked),
        )
        keep = z <= z_threshold  # bool, same shape as stacked
        keep_count = keep.sum(dim=0).clamp(min=1)
        masked_sum = (stacked * keep.float()).sum(dim=0)
        robust = masked_sum / keep_count.float()
        averaged[k] = robust.to(state_dicts[0][k].dtype)
    return averaged


@dataclass
class CheckpointAverager:
    """Load phase checkpoints from disk and produce a converged state_dict."""

    checkpoint_dir: str
    z_threshold: float = 2.0

    def list_phase_checkpoints(self) -> list[Path]:
        d = Path(self.checkpoint_dir)
        manifest = d / "phases.jsonl"
        if manifest.exists():
            paths: list[Path] = []
            for line in manifest.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                paths.append(Path(entry["checkpoint"]))
            return paths
        # Fallback: glob
        return sorted(d.glob("phase_*.pt"))

    def load_state_dicts(self, last_k: int | None = None) -> list[dict[str, torch.Tensor]]:
        paths = self.list_phase_checkpoints()
        if last_k is not None:
            paths = paths[-last_k:]
        return [torch.load(p, map_location="cpu")["state_dict"] for p in paths]

    def converge(
        self, last_k: int | None = None
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Return (averaged_state_dict, stats)."""
        sds = self.load_state_dicts(last_k=last_k)
        if not sds:
            raise FileNotFoundError(f"No phase checkpoints in {self.checkpoint_dir}")
        averaged = average_state_dicts(sds, z_threshold=self.z_threshold)
        stats: dict = {
            "num_checkpoints_averaged": len(sds),
            "z_threshold": self.z_threshold,
            "per_tensor_stats": {},
        }
        for k in averaged:
            stacked = torch.stack([sd[k].float() for sd in sds], dim=0)
            stats["per_tensor_stats"][k] = {
                "shape": list(averaged[k].shape),
                "max_std": float(stacked.std(dim=0, unbiased=False).max()),
                "mean_abs": float(stacked.mean(dim=0).abs().mean()),
            }
        return averaged, stats
