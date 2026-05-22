"""Local weight registry.

Wraps WeightPackager and adds list/lookup operations and an alias table
(`registry/aliases.json`) so callers can refer to bundles by stable names
like 'latest' or 'production' without needing the sha hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ml_training.packaging import WeightPackager


@dataclass
class RegistryEntry:
    bundle_id: str
    bundle_dir: Path
    quant: str
    arch_preset: str
    created_at: str
    base_hash: Optional[str]
    principles_hash: Optional[str]
    compressed_size_bytes: int


class WeightRegistry:
    """Disk-backed registry of weight bundles."""

    def __init__(self, root: str = "artifacts/registry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.aliases_path = self.root / "aliases.json"
        if not self.aliases_path.exists():
            self.aliases_path.write_text("{}")
        self._packager = WeightPackager(registry_root=str(self.root))

    @property
    def packager(self) -> WeightPackager:
        return self._packager

    def list_entries(self) -> list[RegistryEntry]:
        entries: list[RegistryEntry] = []
        for m in self._packager.list_bundles():
            entries.append(RegistryEntry(
                bundle_id=m["bundle_id"],
                bundle_dir=self.root / m["bundle_id"],
                quant=m.get("quant", "fp32"),
                arch_preset=m.get("arch_preset", "custom"),
                created_at=m.get("created_at", ""),
                base_hash=m.get("base_hash"),
                principles_hash=m.get("principles_hash"),
                compressed_size_bytes=int(m.get("compressed_size_bytes", 0)),
            ))
        return entries

    def get(self, bundle_id_or_alias: str) -> RegistryEntry:
        aliases = json.loads(self.aliases_path.read_text())
        bundle_id = aliases.get(bundle_id_or_alias, bundle_id_or_alias)
        for entry in self.list_entries():
            if entry.bundle_id == bundle_id:
                return entry
        raise KeyError(f"Bundle not found: {bundle_id_or_alias}")

    def set_alias(self, alias: str, bundle_id: str) -> None:
        # Validate the bundle exists
        self.get(bundle_id)
        aliases = json.loads(self.aliases_path.read_text())
        aliases[alias] = bundle_id
        self.aliases_path.write_text(json.dumps(aliases, indent=2))

    def promote(self, bundle_id: str, alias: str = "production") -> None:
        self.set_alias(alias, bundle_id)

    def latest(self) -> Optional[RegistryEntry]:
        entries = self.list_entries()
        if not entries:
            return None
        return max(entries, key=lambda e: e.created_at)
