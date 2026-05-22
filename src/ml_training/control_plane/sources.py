"""Source link manifest, embedded into each weight bundle's manifest.json."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SourceLinks:
    dataset_uri: str
    tokenizer_uri: Optional[str] = None
    principles_uri: Optional[str] = None
    base_model_uri: Optional[str] = None
    extras: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, str]:
        d = {"dataset": self.dataset_uri}
        if self.tokenizer_uri:
            d["tokenizer"] = self.tokenizer_uri
        if self.principles_uri:
            d["principles"] = self.principles_uri
        if self.base_model_uri:
            d["base_model"] = self.base_model_uri
        d.update(self.extras)
        return d


def hash_file(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
