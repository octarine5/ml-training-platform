"""Real decoder-only transformer (PyTorch nn.Module).

Materialized from a TransformerArchitecture spec. Real multi-head self-attention
with explicit Q, K, V projections, scaled dot-product, causal mask, residual +
pre-LayerNorm, GELU FFN, and tied input/output embeddings.

This is the runnable model. The 256-layer preset should NOT be instantiated here
by default; use the planning spec in architecture.TransformerArchitecture for that.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from ml_training.architecture import TransformerArchitecture, TransformerSpec


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with explicit Q, K, V projections.

    Uses torch.nn.functional.scaled_dot_product_attention (SDPA) when available:
    fused, faster, and avoids MPS-specific masked_fill dtype bugs in fp16.
    Falls back to manual attention only if SDPA is missing (older torch).
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Explicit Q, K, V (not fused) so per-projection LoRA / masking is straightforward.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Kept for backward compatibility with code that inspects this buffer
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)
        self._has_sdpa = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self._has_sdpa:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Use additive -inf mask cast to the score dtype to avoid mixed-dtype masked_fill
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(self.causal_mask[:T, :T], neg_inf)
            attn = F.softmax(scores, dim=-1)
            out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """Standard transformer FFN: Linear → GELU → Linear."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.ffn1 = nn.Linear(d_model, d_ff, bias=False)
        self.ffn2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn2(F.gelu(self.ffn1(x)))


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: attn → ffn with residuals."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    """Decoder-only transformer with tied input/output embeddings."""

    def __init__(self, spec: TransformerSpec) -> None:
        super().__init__()
        self.spec = spec
        self.token_emb = nn.Embedding(spec.vocab_size, spec.d_model)
        self.pos_emb = nn.Embedding(spec.max_seq_len, spec.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(spec.d_model, spec.num_heads, spec.d_ff, spec.max_seq_len)
            for _ in range(spec.num_layers)
        ])
        self.ln_f = nn.LayerNorm(spec.d_model)
        # LM head tied to token embedding (weight shared)
        self.lm_head = nn.Linear(spec.d_model, spec.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying
        # Optimization flags (opt-in; default off so existing behavior unchanged)
        self.use_gradient_checkpointing: bool = False

    @classmethod
    def from_arch(cls, arch: TransformerArchitecture) -> "MiniTransformer":
        return cls(arch.spec)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        stop_after_block: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [B, T] long tensor of token ids.
            labels: optional [B, T] labels for next-token cross-entropy (shifted internally).
            stop_after_block: if set, only run blocks[:stop_after_block+1] (partial serving).

        Returns:
            dict with "logits" [B, T, V] and optional "loss".
        """
        B, T = input_ids.shape
        if T > self.spec.max_seq_len:
            raise ValueError(f"seq_len {T} > max_seq_len {self.spec.max_seq_len}")
        pos = torch.arange(T, device=input_ids.device)
        h = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        last = stop_after_block if stop_after_block is not None else len(self.blocks) - 1
        for i, block in enumerate(self.blocks):
            if i > last:
                break
            if self.use_gradient_checkpointing and self.training:
                h = ckpt.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            # Standard next-token prediction: shift labels by one
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.spec.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_after_block: int | None = None,
    ) -> torch.Tensor:
        """Greedy / temperature sampling. Returns [B, T+new] long tensor."""
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            ctx = ids[:, -self.spec.max_seq_len :]
            logits = self(ctx, stop_after_block=stop_after_block)["logits"][:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

    def num_parameters(self) -> int:
        # Note: tied weights only count once.
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total
