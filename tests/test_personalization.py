"""Tests for principles_to_dataset + real LoRA."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from ml_training.architecture import TransformerArchitecture
from ml_training.models.transformer import MiniTransformer
from ml_training.personalization import (
    LoRAConfig,
    LoRALinear,
    LoRATrainer,
    inject_lora,
    load_principles,
    merge_lora,
    principles_to_pairs,
)
from ml_training.personalization.principles_to_dataset import principles_hash


def test_load_principles_yaml(tmp_path: Path):
    p = tmp_path / "principles.yaml"
    p.write_text(yaml.safe_dump({"principles": ["Be concise.", "Cite sources."]}))
    pr = load_principles(p)
    assert len(pr) == 2
    assert pr[0].text == "Be concise."


def test_principles_hash_stable(tmp_path: Path):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump({"principles": ["a", "b"]}))
    h1 = principles_hash(load_principles(p))
    h2 = principles_hash(load_principles(p))
    assert h1 == h2
    p.write_text(yaml.safe_dump({"principles": ["a", "c"]}))
    h3 = principles_hash(load_principles(p))
    assert h3 != h1


def test_principles_to_pairs_expansion(tmp_path: Path):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump({"principles": ["x", "y"]}))
    pr = load_principles(p)
    ds = principles_to_pairs(pr)
    # 10 preferred templates × 2 principles = 20
    assert len(ds.preferred) == 20
    # 5 dispreferred templates × 2 principles = 10
    assert len(ds.dispreferred) == 10
    assert "x" in ds.preferred[0] or "x" in ds.preferred[1]


def test_inject_lora_adds_adapters():
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    base_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapters = inject_lora(model, LoRAConfig(rank=8))
    # 2 projections × 6 blocks = 12 LoRA wrappers
    assert len(adapters) == 12
    # Only adapter params + final LN + token_emb should be trainable
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0
    # Wrapped projections should be LoRALinear
    for block in model.blocks:
        assert isinstance(block.attn.q_proj, LoRALinear)
        assert isinstance(block.attn.v_proj, LoRALinear)


def test_lora_forward_changes_after_training():
    torch.manual_seed(0)
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    ids = torch.randint(0, arch.spec.vocab_size, (2, 16))
    model.eval()
    with torch.no_grad():
        before = model(ids)["logits"].clone()

    trainer = LoRATrainer(model, LoRAConfig(rank=4), learning_rate=1e-3)

    def _iter():
        for _ in range(5):
            yield ids

    trainer.train(_iter(), epochs=1)
    model.eval()
    with torch.no_grad():
        after = model(ids)["logits"]
    # Outputs should differ after LoRA training
    assert (after - before).abs().max().item() > 1e-4


def test_merge_lora_back_to_plain_linear():
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    inject_lora(model, LoRAConfig(rank=4))
    merge_lora(model)
    for block in model.blocks:
        # After merge, projections should be plain nn.Linear again
        assert not isinstance(block.attn.q_proj, LoRALinear)
        assert not isinstance(block.attn.v_proj, LoRALinear)


def test_lora_only_lora_params_trainable():
    arch = TransformerArchitecture.from_preset("local-default")
    model = MiniTransformer.from_arch(arch)
    adapters = inject_lora(model, LoRAConfig(rank=4))
    trainable_param_ids = {id(p) for p in model.parameters() if p.requires_grad}
    # Every adapter A and B must be in the trainable set
    for a in adapters:
        assert id(a.A) in trainable_param_ids
        assert id(a.B) in trainable_param_ids
        # Base linear weight must NOT be trainable
        assert not a.base.weight.requires_grad
