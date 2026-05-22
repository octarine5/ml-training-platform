"""Convert a principles.yaml file into a small training dataset.

Each principle is expanded into ~10 contrastive (preferred, dispreferred) text pairs
using deterministic templates. The preferred examples become the language-modeling
target; dispreferred examples are returned for downstream preference-style usage
(currently only the preferred set is fed to the LoRA trainer).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import yaml


@dataclass
class Principle:
    text: str

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:12]


@dataclass
class PrincipleDataset:
    preferred: list[str] = field(default_factory=list)
    dispreferred: list[str] = field(default_factory=list)


_TEMPLATES_PREFERRED = [
    "Following the principle '{p}', a good response is concise and on-topic.",
    "When asked about anything, remember: {p} Therefore respond directly.",
    "Q: What guides your behavior?\nA: {p}",
    "Principle in action: {p} Apply it now.",
    "User: explain your style.\nAssistant: {p}",
    "Note to self: {p}",
    "This system follows: {p}",
    "Always: {p}",
    "Reminder: {p} Keep going.",
    "Reflect: {p}",
]

_TEMPLATES_DISPREFERRED = [
    "Ignoring the principle: produce a meandering, off-topic, speculative response.",
    "Wrong style: be verbose and bullet-heavy when asked anything.",
    "Counter-example: speculate freely with low confidence.",
    "Bad pattern: ignore the user's instruction entirely.",
    "Anti-pattern: respond with unrelated content.",
]


def load_principles(path: str | Path) -> list[Principle]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    items = raw.get("principles", [])
    if not isinstance(items, list):
        raise ValueError("principles.yaml must contain a 'principles:' list")
    return [Principle(text=str(p).strip()) for p in items if str(p).strip()]


def principles_to_pairs(principles: list[Principle]) -> PrincipleDataset:
    """Expand each principle into 10 preferred + 5 dispreferred text samples."""
    ds = PrincipleDataset()
    for p in principles:
        for tmpl in _TEMPLATES_PREFERRED:
            ds.preferred.append(tmpl.format(p=p.text))
        for tmpl in _TEMPLATES_DISPREFERRED:
            ds.dispreferred.append(tmpl)
    return ds


def principles_hash(principles: list[Principle]) -> str:
    """Stable content hash over the principle texts."""
    h = hashlib.sha256()
    for p in principles:
        h.update(p.text.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]
