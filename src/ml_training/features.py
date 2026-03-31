"""Feature store, transformers, and interaction matrices for ML training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


class FeatureStore:
    """Store and retrieve training features by name and version."""

    def __init__(self) -> None:
        self._store: dict[str, dict[int, np.ndarray]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        data: np.ndarray,
        version: int = 1,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a feature array under a name and version."""
        if name not in self._store:
            self._store[name] = {}
        self._store[name][version] = data
        if metadata:
            self._metadata[name] = metadata

    def get(self, name: str, version: Optional[int] = None) -> np.ndarray:
        """Retrieve a feature by name. Returns the latest version if not specified."""
        if name not in self._store:
            raise KeyError(f"Feature '{name}' not found in store")
        versions = self._store[name]
        if version is not None:
            if version not in versions:
                raise KeyError(f"Feature '{name}' version {version} not found")
            return versions[version]
        latest = max(versions.keys())
        return versions[latest]

    def list_features(self) -> list[dict[str, Any]]:
        """List all registered features with their versions."""
        result = []
        for name, versions in self._store.items():
            result.append({
                "name": name,
                "versions": sorted(versions.keys()),
                "latest_shape": versions[max(versions.keys())].shape,
                "metadata": self._metadata.get(name, {}),
            })
        return result

    def delete(self, name: str, version: Optional[int] = None) -> None:
        """Delete a feature or specific version."""
        if name not in self._store:
            return
        if version is not None:
            self._store[name].pop(version, None)
            if not self._store[name]:
                del self._store[name]
        else:
            del self._store[name]
            self._metadata.pop(name, None)


class FeatureTransformer:
    """Transform raw features into embeddings and normalized representations."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def text_embedding(self, texts: list[str], dim: int = 128) -> np.ndarray:
        """Generate deterministic pseudo-embeddings for text inputs.

        Uses a hash-based approach so the same text always maps to the same vector.
        """
        embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            text_hash = hashlib.sha256(text.encode()).digest()
            seed_val = int.from_bytes(text_hash[:8], "little")
            rng = np.random.default_rng(seed_val)
            vec = rng.standard_normal(dim).astype(np.float32)
            embeddings[i] = vec / (np.linalg.norm(vec) + 1e-8)
        return embeddings

    def image_embedding(self, pixel_arrays: list[np.ndarray], dim: int = 256) -> np.ndarray:
        """Generate embeddings from image pixel arrays using average pooling + projection."""
        embeddings = np.zeros((len(pixel_arrays), dim), dtype=np.float32)
        for i, pixels in enumerate(pixel_arrays):
            flat = pixels.flatten().astype(np.float32)
            # Chunk and average pool to reduce to target dim
            if len(flat) >= dim:
                chunk_size = len(flat) // dim
                pooled = np.array([flat[j * chunk_size:(j + 1) * chunk_size].mean()
                                   for j in range(dim)])
            else:
                pooled = np.zeros(dim, dtype=np.float32)
                pooled[:len(flat)] = flat
            norm = np.linalg.norm(pooled) + 1e-8
            embeddings[i] = pooled / norm
        return embeddings

    def user_sequence_embedding(
        self, sequences: list[list[int]], vocab_size: int, dim: int = 64
    ) -> np.ndarray:
        """Embed user action sequences by averaging learned item embeddings."""
        embedding_table = self._rng.standard_normal((vocab_size, dim)).astype(np.float32)
        result = np.zeros((len(sequences), dim), dtype=np.float32)
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            valid_ids = [s % vocab_size for s in seq]
            seq_embeds = embedding_table[valid_ids]
            result[i] = seq_embeds.mean(axis=0)
        return result

    def normalize(self, features: np.ndarray, method: str = "standard") -> np.ndarray:
        """Normalize features using standard or min-max scaling."""
        if method == "standard":
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-8
            return (features - mean) / std
        elif method == "minmax":
            fmin = features.min(axis=0)
            fmax = features.max(axis=0)
            denom = (fmax - fmin) + 1e-8
            return (features - fmin) / denom
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class FeatureMatrix:
    """Build feature interaction matrices for recommendation models."""

    def __init__(self, feature_groups: dict[str, np.ndarray]) -> None:
        """Initialize with named feature groups.

        Args:
            feature_groups: Mapping of feature group name to 2D array (samples x features).
        """
        self.feature_groups = feature_groups
        self._validate()

    def _validate(self) -> None:
        """Ensure all feature groups have the same number of samples."""
        lengths = {name: arr.shape[0] for name, arr in self.feature_groups.items()}
        unique = set(lengths.values())
        if len(unique) > 1:
            raise ValueError(f"Inconsistent sample counts across groups: {lengths}")

    @property
    def num_samples(self) -> int:
        if not self.feature_groups:
            return 0
        return next(iter(self.feature_groups.values())).shape[0]

    def concatenated(self) -> np.ndarray:
        """Return all feature groups concatenated along the feature axis."""
        arrays = list(self.feature_groups.values())
        return np.concatenate(arrays, axis=1)

    def pairwise_interactions(self, group_a: str, group_b: str) -> np.ndarray:
        """Compute element-wise pairwise interactions between two feature groups.

        If groups have different feature dimensions, the smaller is padded with zeros.
        """
        a = self.feature_groups[group_a]
        b = self.feature_groups[group_b]
        max_dim = max(a.shape[1], b.shape[1])
        if a.shape[1] < max_dim:
            a = np.pad(a, ((0, 0), (0, max_dim - a.shape[1])))
        if b.shape[1] < max_dim:
            b = np.pad(b, ((0, 0), (0, max_dim - b.shape[1])))
        return a * b

    def cross_features(self) -> np.ndarray:
        """Build cross-feature interactions across all groups via outer products.

        Returns a matrix where each sample has flattened pairwise interactions
        between all groups.
        """
        group_names = sorted(self.feature_groups.keys())
        interactions = []
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                inter = self.pairwise_interactions(group_names[i], group_names[j])
                interactions.append(inter)
        if not interactions:
            return self.concatenated()
        return np.concatenate(interactions, axis=1)
