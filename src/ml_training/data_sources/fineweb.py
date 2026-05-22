"""HuggingFace fineweb-ultra-mini loader.

Real `datasets.load_dataset(...)` streaming, so we do not fully materialize the
dataset in RAM. Network and HF cache required to run end-to-end; unit tests
mock `load_dataset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Protocol


@dataclass
class TextRecord:
    text: str
    source_url: Optional[str] = None
    record_id: Optional[str] = None


class DataSource(Protocol):
    """Anything that yields TextRecord."""

    def stream(self, max_records: Optional[int] = None) -> Iterator[TextRecord]: ...


class FineWebLoader:
    """Streams text from `motionlabs/fineweb-ultra-mini` via HuggingFace `datasets`."""

    DEFAULT_DATASET = "motionlabs/fineweb-ultra-mini"

    def __init__(
        self,
        dataset_name: str = DEFAULT_DATASET,
        split: str = "train",
        seed: int = 42,
        text_field_candidates: tuple[str, ...] = ("text", "content", "raw_content"),
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.seed = seed
        self.text_field_candidates = text_field_candidates

    def _resolve_text_field(self, sample: dict) -> str:
        for k in self.text_field_candidates:
            if k in sample and isinstance(sample[k], str):
                return k
        # Fallback: first str-valued field
        for k, v in sample.items():
            if isinstance(v, str) and len(v) > 0:
                return k
        raise KeyError(f"Could not find a text field in sample keys: {list(sample.keys())}")

    def stream(self, max_records: Optional[int] = None) -> Iterator[TextRecord]:
        # Lazy import so tests that mock datasets.load_dataset don't pay the cost.
        from datasets import load_dataset

        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        ds = ds.shuffle(seed=self.seed, buffer_size=1000)
        text_field: Optional[str] = None
        for i, sample in enumerate(ds):
            if max_records is not None and i >= max_records:
                break
            if text_field is None:
                text_field = self._resolve_text_field(sample)
            text = sample.get(text_field)
            if not isinstance(text, str) or not text:
                continue
            yield TextRecord(
                text=text,
                source_url=sample.get("url") or sample.get("source_url"),
                record_id=str(sample.get("id") or i),
            )

    def texts(self, max_records: Optional[int] = None) -> Iterator[str]:
        for rec in self.stream(max_records=max_records):
            yield rec.text
