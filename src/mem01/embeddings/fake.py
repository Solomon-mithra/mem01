"""Deterministic embedder for tests (no API keys).

Maps text to a stable unit-ish vector via hashing tokens into buckets.
Similar strings are not guaranteed close — for tests we often set expectations
on ops/status, not semantic neighbors. When search order matters, tests can
hand-seed embeddings via the store.
"""

from __future__ import annotations

import hashlib
import math
import re


class FakeEmbedder:
    def __init__(self, dimensions: int = 32) -> None:
        if dimensions < 4:
            raise ValueError("dimensions must be >= 4")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vec = [0.0] * self._dimensions
        if not tokens:
            # empty → tiny bias so vector is non-zero
            vec[0] = 1.0
            return vec
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self._dimensions
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
