"""Model-judge layer over the existing EvaluationSystem.

Adds LM-relevant metrics:
- Perplexity on a held-out token stream (real CE with the real model).
- Judge score per response: rubric-based (default, offline) or external_llm (HTTP plug).

Judge score flows into ModelDriftDetector via the existing `extra` field.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F

from ml_training.evaluation import MetricsResult
from ml_training.models.transformer import MiniTransformer


# --------------------------------------------------------------- perplexity


def compute_perplexity(
    model: MiniTransformer,
    token_batches: Iterable[torch.Tensor],
    device: torch.device | None = None,
) -> float:
    """Average per-token cross-entropy across batches; returns perplexity = exp(loss)."""
    device = device or next(model.parameters()).device
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in token_batches:
            batch = batch.to(device)
            out = model(batch, labels=batch)
            n = batch.numel() - batch.shape[0]  # next-token targets per row = T - 1
            total_loss += float(out["loss"]) * n
            total_tokens += n
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_loss / total_tokens)


# --------------------------------------------------------------- model judge


@dataclass
class JudgeResult:
    prompt: str
    response: str
    reference: Optional[str]
    principles: list[str]
    score: float                # in [0, 1]
    sub_scores: dict[str, float] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class JudgeCorpusItem:
    prompt: str
    reference: Optional[str] = None
    applicable_principles: list[str] = field(default_factory=list)


def load_judge_corpus(path: str | Path) -> list[JudgeCorpusItem]:
    items: list[JudgeCorpusItem] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        items.append(JudgeCorpusItem(
            prompt=d["prompt"],
            reference=d.get("reference"),
            applicable_principles=d.get("applicable_principles", []),
        ))
    return items


class ModelJudge:
    """Rubric-based judge (offline). Pluggable `external_scorer` for an LLM judge."""

    def __init__(
        self,
        principles: list[str],
        external_scorer: Optional[Callable[[str, str, str, list[str]], float]] = None,
    ) -> None:
        self.principles = principles
        self.external_scorer = external_scorer

    # ------------ rubric sub-scores

    def _length_score(self, response: str) -> float:
        """Reward concise responses; penalize empty or runaway outputs."""
        n = len(response.split())
        if n == 0:
            return 0.0
        if n > 300:
            return max(0.0, 1.0 - (n - 300) / 300)
        if n < 5:
            return 0.4
        return 1.0

    def _principle_overlap(self, response: str, applicable_principles: list[str]) -> float:
        """Lexical/keyword overlap between response and principles' key terms."""
        if not applicable_principles:
            return 0.5  # neutral when no principle applies
        resp_tokens = set(re.findall(r"[a-zA-Z]+", response.lower()))
        hits = 0
        total = 0
        for p in applicable_principles:
            terms = [t for t in re.findall(r"[a-zA-Z]+", p.lower()) if len(t) > 3]
            terms = [t for t in terms if t not in {
                "prefer", "avoid", "should", "must", "when", "uncertain",
                "answers", "claims", "with", "that", "this", "have", "will",
                "over", "from", "into", "than",
            }]
            for t in terms:
                total += 1
                if t in resp_tokens or any(t in r for r in resp_tokens):
                    hits += 1
        return hits / max(total, 1)

    def _reference_similarity(self, response: str, reference: Optional[str]) -> float:
        """Token-set Jaccard between response and reference."""
        if not reference:
            return 0.5
        a = set(re.findall(r"[a-zA-Z]+", response.lower()))
        b = set(re.findall(r"[a-zA-Z]+", reference.lower()))
        if not a and not b:
            return 1.0
        return len(a & b) / max(len(a | b), 1)

    def score(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        applicable_principles: Optional[list[str]] = None,
    ) -> JudgeResult:
        principles = applicable_principles if applicable_principles is not None else self.principles
        if self.external_scorer is not None:
            ext = float(self.external_scorer(prompt, response, reference or "", principles))
            ext = max(0.0, min(1.0, ext))
            return JudgeResult(
                prompt=prompt, response=response, reference=reference,
                principles=principles, score=ext,
                sub_scores={"external": ext},
                rationale="external_llm",
            )
        s_len = self._length_score(response)
        s_pri = self._principle_overlap(response, principles)
        s_ref = self._reference_similarity(response, reference)
        # Weighted blend; principles matter most when they exist
        weights = {"length": 0.2, "principles": 0.5 if principles else 0.1, "reference": 0.3 if reference else 0.1}
        wsum = sum(weights.values()) or 1.0
        score = (s_len * weights["length"] + s_pri * weights["principles"] + s_ref * weights["reference"]) / wsum
        return JudgeResult(
            prompt=prompt, response=response, reference=reference,
            principles=principles, score=float(max(0.0, min(1.0, score))),
            sub_scores={"length": s_len, "principles": s_pri, "reference": s_ref},
            rationale=f"len={s_len:.2f} pri={s_pri:.2f} ref={s_ref:.2f}",
        )


def aggregate_judge(results: list[JudgeResult]) -> MetricsResult:
    """Bundle judge results into a MetricsResult compatible with the existing drift detector."""
    if not results:
        return MetricsResult(ctr=0, cvr=0, auc=0, log_loss=0, num_samples=0)
    avg = sum(r.score for r in results) / len(results)
    sub_means: dict[str, float] = {}
    for key in ("length", "principles", "reference", "external"):
        vals = [r.sub_scores[key] for r in results if key in r.sub_scores]
        if vals:
            sub_means[f"judge_{key}"] = sum(vals) / len(vals)
    return MetricsResult(
        ctr=avg, cvr=avg, auc=avg, log_loss=-math.log(max(avg, 1e-6)),
        num_samples=len(results),
        extra={"judge_score": avg, **sub_means},
    )
