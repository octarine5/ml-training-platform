"""Tests for evaluation_judge: perplexity + model-judge."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ml_training.architecture import TransformerArchitecture
from ml_training.evaluation_judge import (
    JudgeCorpusItem,
    ModelJudge,
    aggregate_judge,
    compute_perplexity,
    load_judge_corpus,
)
from ml_training.models.transformer import MiniTransformer


def test_perplexity_finite_on_random_model():
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    batches = [torch.randint(0, arch.spec.vocab_size, (2, 16)) for _ in range(3)]
    ppl = compute_perplexity(model, batches)
    assert ppl > 0
    assert ppl == ppl  # not NaN


def test_perplexity_lower_after_overfit_on_constant():
    """Sanity check: training on a single repeated batch should decrease perplexity."""
    torch.manual_seed(0)
    arch = TransformerArchitecture(
        TransformerArchitecture.from_preset("local-default").spec.__class__(
            num_layers=2, num_heads=2, d_model=64, d_ff=128,
            vocab_size=200, max_seq_len=32,
        )
    )
    model = MiniTransformer.from_arch(arch)
    fixed_batch = torch.randint(0, 200, (2, 16))
    before = compute_perplexity(model, [fixed_batch])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(20):
        opt.zero_grad()
        out = model(fixed_batch, labels=fixed_batch)
        out["loss"].backward()
        opt.step()
    after = compute_perplexity(model, [fixed_batch])
    assert after < before


def test_judge_load_corpus(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    items = [
        {"prompt": "q1", "reference": "r1", "applicable_principles": ["be concise"]},
        {"prompt": "q2", "applicable_principles": []},
    ]
    p.write_text("\n".join(json.dumps(i) for i in items))
    loaded = load_judge_corpus(p)
    assert len(loaded) == 2
    assert loaded[0].prompt == "q1"
    assert loaded[1].reference is None


def test_judge_rubric_scoring_bounds():
    j = ModelJudge(principles=["Prefer concise answers"])
    res = j.score(
        prompt="What is 2+2?",
        response="four",
        reference="four",
        applicable_principles=["Prefer concise answers"],
    )
    assert 0.0 <= res.score <= 1.0


def test_judge_higher_score_when_response_matches_reference():
    j = ModelJudge(principles=[])
    matching = j.score("Q", "this exactly matches the reference", "this exactly matches the reference", [])
    non_matching = j.score("Q", "completely different unrelated text", "this exactly matches the reference", [])
    assert matching.score > non_matching.score


def test_judge_principle_overlap_rewards_keyword_match():
    j = ModelJudge(principles=["Always cite the source when stating a factual claim."])
    with_kw = j.score("Q", "I cite the source carefully.", None,
                      ["Always cite the source when stating a factual claim."])
    without_kw = j.score("Q", "random unrelated answer.", None,
                         ["Always cite the source when stating a factual claim."])
    assert with_kw.score >= without_kw.score


def test_aggregate_judge_extracts_judge_score():
    j = ModelJudge(principles=[])
    results = [
        j.score("q", "a perfect match", "a perfect match", []),
        j.score("q", "a perfect match", "a perfect match", []),
    ]
    agg = aggregate_judge(results)
    assert "judge_score" in agg.extra
    assert agg.num_samples == 2


def test_external_scorer_override():
    j = ModelJudge(principles=[], external_scorer=lambda *_args: 0.42)
    res = j.score("q", "anything", None, [])
    assert res.score == 0.42
    assert "external" in res.rationale
