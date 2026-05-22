"""Tests for the BPE tokenizer."""

from __future__ import annotations

from pathlib import Path

import torch

from ml_training.tokenization import BPETokenizer, TokenizerConfig


def _corpus():
    return [
        "the quick brown fox jumps over the lazy dog",
        "machine learning transforms data into predictions",
        "attention is all you need in a transformer",
        "language models learn from large text corpora",
    ] * 20


def test_train_and_encode_decode(tmp_path: Path):
    tk = BPETokenizer(TokenizerConfig(vocab_size=200, cache_path=str(tmp_path / "tok.json")))
    tk.train_from_iterator(iter(_corpus()))
    assert tk.vocab_size > 10
    ids = tk.encode("hello world")
    decoded = tk.decode(ids)
    assert "hello" in decoded


def test_persisted_tokenizer_roundtrip(tmp_path: Path):
    cache = tmp_path / "tok.json"
    tk = BPETokenizer(TokenizerConfig(vocab_size=200, cache_path=str(cache)))
    tk.train_from_iterator(iter(_corpus()))
    ids = tk.encode("a sample sentence")

    tk2 = BPETokenizer(TokenizerConfig(vocab_size=200, cache_path=str(cache)))
    tk2.load()
    assert tk2.encode("a sample sentence") == ids


def test_load_or_train_creates_when_missing(tmp_path: Path):
    cache = tmp_path / "tok.json"
    assert not cache.exists()
    tk = BPETokenizer(TokenizerConfig(vocab_size=200, cache_path=str(cache)))
    tk.load_or_train(lambda: iter(_corpus()))
    assert cache.exists()


def test_encode_batch_padding(tmp_path: Path):
    tk = BPETokenizer(TokenizerConfig(vocab_size=200, cache_path=str(tmp_path / "tok.json")))
    tk.train_from_iterator(iter(_corpus()))
    batch = tk.encode_batch(["short", "a longer sentence here"], max_length=16)
    assert batch.shape == (2, 16)
    assert batch.dtype == torch.long


def test_stream_token_batches_shape(tmp_path: Path):
    tk = BPETokenizer(TokenizerConfig(vocab_size=200, cache_path=str(tmp_path / "tok.json")))
    tk.train_from_iterator(iter(_corpus()))
    batches = list(tk.stream_token_batches(iter(_corpus()), batch_size=2, seq_len=16))
    assert len(batches) > 0
    for b in batches:
        assert b.shape == (2, 16)
