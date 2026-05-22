"""BPE tokenizer wrapping `tokenizers` library.

Train-once-on-sample-or-load-from-cache pattern. Persists to artifacts/tokenizer/.
Produces torch long tensors compatible with MiniTransformer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


@dataclass
class TokenizerConfig:
    vocab_size: int = 8000
    cache_path: str = "artifacts/tokenizer/tokenizer.json"
    special_tokens: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")


class BPETokenizer:
    """Byte-pair-encoding tokenizer using HF `tokenizers`."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self._tk: Tokenizer | None = None

    # ----------------------------------------------------------------- train

    def train_from_iterator(self, text_iter: Iterable[str]) -> None:
        tk = Tokenizer(models.BPE(unk_token="<unk>"))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=list(self.config.special_tokens),
        )
        tk.train_from_iterator(text_iter, trainer=trainer)
        self._tk = tk
        self.save()

    def load_or_train(self, text_iter_factory) -> "BPETokenizer":
        """Load from cache if present; otherwise call factory() to get an iterator and train."""
        if Path(self.config.cache_path).exists():
            self.load()
            return self
        self.train_from_iterator(text_iter_factory())
        return self

    # ----------------------------------------------------------------- io

    def save(self) -> None:
        if self._tk is None:
            raise RuntimeError("Tokenizer not trained / loaded")
        path = Path(self.config.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tk.save(str(path))

    def load(self) -> "BPETokenizer":
        self._tk = Tokenizer.from_file(self.config.cache_path)
        return self

    # ----------------------------------------------------------------- encode / decode

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tk is None:
            raise RuntimeError("Tokenizer not initialized")
        return self._tk

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids)

    def encode_batch(
        self,
        texts: list[str],
        max_length: int | None = None,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """Return a padded [B, T] long tensor."""
        encs = self.tokenizer.encode_batch(texts)
        if max_length is None:
            max_length = max(len(e.ids) for e in encs)
        out = torch.full((len(encs), max_length), pad_id, dtype=torch.long)
        for i, e in enumerate(encs):
            ids = e.ids[:max_length]
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def stream_token_batches(
        self,
        text_iter: Iterable[str],
        batch_size: int,
        seq_len: int,
    ) -> Iterable[torch.Tensor]:
        """Concatenate streamed texts, chunk into [batch_size, seq_len] long tensors."""
        buf: list[int] = []
        need = batch_size * seq_len
        for text in text_iter:
            buf.extend(self.encode(text))
            buf.append(self.tokenizer.token_to_id("<eos>") or 0)
            while len(buf) >= need:
                arr = torch.tensor(buf[:need], dtype=torch.long).view(batch_size, seq_len)
                buf = buf[need:]
                yield arr
