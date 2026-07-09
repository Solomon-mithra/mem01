"""Embedder protocol — text → vector.

Why an interface:
- Tests use FakeEmbedder (no network, deterministic)
- Production can use OpenAI / local models without changing apply_ops
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        """Return a fixed-dimension embedding for *text*."""
        ...

    @property
    def dimensions(self) -> int:
        """Vector size this embedder produces."""
        ...
