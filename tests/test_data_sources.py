"""Tests for the FineWebLoader; HF dataset is mocked so tests stay offline."""

from __future__ import annotations

from unittest.mock import patch

from ml_training.data_sources import FineWebLoader, TextRecord


def _fake_dataset(samples: list[dict]):
    class _Fake:
        def __init__(self, items): self._items = items
        def shuffle(self, **kwargs): return self
        def __iter__(self): return iter(self._items)
    return _Fake(samples)


def test_fineweb_loader_yields_text_records():
    samples = [
        {"text": "hello world", "url": "http://a", "id": "1"},
        {"text": "another doc", "url": "http://b", "id": "2"},
    ]
    with patch("datasets.load_dataset", return_value=_fake_dataset(samples)):
        loader = FineWebLoader()
        records = list(loader.stream())
    assert len(records) == 2
    assert isinstance(records[0], TextRecord)
    assert records[0].text == "hello world"
    assert records[0].source_url == "http://a"


def test_fineweb_loader_max_records():
    samples = [{"text": f"doc{i}"} for i in range(50)]
    with patch("datasets.load_dataset", return_value=_fake_dataset(samples)):
        loader = FineWebLoader()
        records = list(loader.stream(max_records=10))
    assert len(records) == 10


def test_fineweb_loader_resolves_alternate_text_field():
    samples = [{"content": "from content field"}]
    with patch("datasets.load_dataset", return_value=_fake_dataset(samples)):
        loader = FineWebLoader()
        records = list(loader.stream())
    assert records[0].text == "from content field"


def test_fineweb_loader_skips_non_string_text():
    samples = [
        {"text": "good"},
        {"text": ""},  # empty - should skip
        {"text": "also good"},
    ]
    with patch("datasets.load_dataset", return_value=_fake_dataset(samples)):
        loader = FineWebLoader()
        records = list(loader.stream())
    # First and third yielded
    assert {r.text for r in records} == {"good", "also good"}


def test_fineweb_loader_texts_helper():
    samples = [{"text": "one"}, {"text": "two"}]
    with patch("datasets.load_dataset", return_value=_fake_dataset(samples)):
        loader = FineWebLoader()
        texts = list(loader.texts())
    assert texts == ["one", "two"]
