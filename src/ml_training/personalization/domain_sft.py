"""Domain SFT: LoRA fine-tune a HuggingFace causal LM on a domain Q&A dataset.

Wraps PEFT for LoRA application and HuggingFace `datasets` for loading the
domain corpus. Reuses the platform's WeightPackager for the final bundle so
domain-tuned models flow through the same registry/serving paths as everything
else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.optim import AdamW


# Required so Qwen + MPS can use torch ops that aren't yet MPS-native (e.g. isin).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass
class DomainSFTConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "gbharti/finance-alpaca"
    dataset_split: str = "train"
    domain_tag: str = "finance"           # used in bundle manifest + output naming
    max_examples: int = 2000              # cap for reasonable wall time on MPS
    max_seq_len: int = 512
    batch_size: int = 2
    grad_accum: int = 4                   # effective batch = batch_size * grad_accum
    steps: int = 800
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    seed: int = 42
    log_every: int = 20


@dataclass
class DomainSFTResult:
    base_model: str
    domain_tag: str
    bundle_dir: Path
    trainable_params: int
    final_loss: float
    losses: list[float] = field(default_factory=list)
    wall_time_sec: float = 0.0


# ---------------------------------------------------------------- dataset loaders


def _format_alpaca(example: dict) -> Optional[tuple[str, str]]:
    """Return (user_message, assistant_response) for Alpaca-format records."""
    instr = example.get("instruction") or example.get("question") or example.get("input")
    inp = example.get("input") if example.get("instruction") else None
    out = example.get("output") or example.get("answer") or example.get("response")
    if not instr or not out:
        return None
    user = instr.strip()
    if inp and inp.strip() and inp.strip() != instr.strip():
        user = f"{user}\n\n{inp.strip()}"
    return user, str(out).strip()


def iter_domain_pairs(
    dataset_name: str,
    split: str = "train",
    max_examples: int = 2000,
    seed: int = 42,
) -> Iterable[tuple[str, str]]:
    """Yield (user, assistant) pairs from a HuggingFace dataset.

    Tries the standard Alpaca schema first; falls back to (question/answer) and
    (input/output) keys. Skips records that can't be parsed.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=1000)
    yielded = 0
    for example in ds:
        pair = _format_alpaca(example)
        if pair is None:
            continue
        yield pair
        yielded += 1
        if yielded >= max_examples:
            return


# ---------------------------------------------------------------- trainer


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DomainSFTTrainer:
    """LoRA SFT of a HuggingFace causal LM on a domain Q&A dataset."""

    def __init__(self, config: DomainSFTConfig) -> None:
        # Import here so module-level import doesn't pay transformers' cost.
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.device = _pick_device()
        torch.manual_seed(config.seed)

        print(f"[domain-sft] loading {config.base_model} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model, torch_dtype=torch.float32
        )
        peft_cfg = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, peft_cfg).to(self.device)
        self.trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[domain-sft] trainable: {self.trainable_params:,} of {total:,} "
            f"({100*self.trainable_params/total:.2f}%)"
        )

    # ------------------------------------------------------------ batching

    def _format_chat(self, user: str, assistant: str) -> str:
        """Apply the base model's chat template."""
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _encode_batch(
        self, pairs: list[tuple[str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a list of (user, assistant) pairs into (input_ids, labels) tensors.

        Labels mask the user prompt + special tokens with -100 so the loss is
        computed only over the assistant span — standard SFT practice.
        """
        max_len = self.config.max_seq_len
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        for user, assistant in pairs:
            full = self._format_chat(user, assistant)
            full_ids = self.tokenizer(full, add_special_tokens=False).input_ids
            # Tokenize the user-only prefix to find the boundary
            prefix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
            n_prefix = len(prefix_ids)
            # Truncate from the right (keep prefix, drop tail of assistant)
            full_ids = full_ids[:max_len]
            labels = [-100] * min(n_prefix, len(full_ids)) + full_ids[n_prefix:]
            labels = labels[:max_len]
            # Pad
            pad_id = self.tokenizer.pad_token_id
            pad_len = max_len - len(full_ids)
            full_ids = full_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            input_ids_list.append(full_ids)
            labels_list.append(labels)
        return (
            torch.tensor(input_ids_list, dtype=torch.long),
            torch.tensor(labels_list, dtype=torch.long),
        )

    # ------------------------------------------------------------ train

    def train(self, pairs_iter: Iterable[tuple[str, str]]) -> DomainSFTResult:
        import itertools, time
        cfg = self.config
        pairs = list(itertools.islice(pairs_iter, cfg.max_examples))
        print(f"[domain-sft] loaded {len(pairs)} domain examples")

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.model.train()

        losses: list[float] = []
        t0 = time.time()
        rng = torch.Generator().manual_seed(cfg.seed)
        n = len(pairs)
        global_step = 0

        for step in range(cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for accum_idx in range(cfg.grad_accum):
                # Sample a random batch (with replacement; fine for SFT at our scale)
                idxs = torch.randint(0, n, (cfg.batch_size,), generator=rng).tolist()
                batch = [pairs[i] for i in idxs]
                input_ids, labels = self._encode_batch(batch)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss / cfg.grad_accum
                loss.backward()
                accum_loss += float(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad), 1.0
            )
            optimizer.step()
            losses.append(accum_loss)
            global_step += 1
            if step % cfg.log_every == 0:
                print(
                    f"[domain-sft {cfg.domain_tag}] step {step:>4}/{cfg.steps} "
                    f"loss={accum_loss:.4f}"
                )

        wall = time.time() - t0
        print(
            f"[domain-sft {cfg.domain_tag}] done: {cfg.steps} steps in {wall:.0f}s "
            f"(final_loss={losses[-1]:.4f})"
        )

        bundle_dir = self._save_bundle(losses[-1])
        return DomainSFTResult(
            base_model=cfg.base_model,
            domain_tag=cfg.domain_tag,
            bundle_dir=bundle_dir,
            trainable_params=self.trainable_params,
            final_loss=losses[-1] if losses else float("nan"),
            losses=losses,
            wall_time_sec=wall,
        )

    # ------------------------------------------------------------ save

    def _save_bundle(self, final_loss: float) -> Path:
        """Save the LoRA adapters (not the merged model — keeps bundles tiny)."""
        out_dir = Path("artifacts/domain_adapters") / self.config.domain_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out_dir))
        self.tokenizer.save_pretrained(str(out_dir))
        # Side-car manifest so the registry can introspect without loading the model
        import json, datetime as dt
        (out_dir / "domain_manifest.json").write_text(json.dumps({
            "base_model": self.config.base_model,
            "domain_tag": self.config.domain_tag,
            "dataset": self.config.dataset_name,
            "steps": self.config.steps,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "trainable_params": self.trainable_params,
            "final_loss": final_loss,
            "max_seq_len": self.config.max_seq_len,
            "created_at": dt.datetime.utcnow().isoformat() + "Z",
        }, indent=2))
        return out_dir


# ---------------------------------------------------------------- inference helper


def load_domain_model(adapter_dir: str):
    """Load a base model + apply the saved LoRA adapter for inference."""
    import json
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = json.loads((Path(adapter_dir) / "domain_manifest.json").read_text())
    base_name = manifest["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_dir)
    return model, tokenizer, manifest
