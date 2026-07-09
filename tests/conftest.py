"""Shared pytest fixtures.

Fakes (store, embedder, LLM) live here so tests run without API keys or network.
"""

from __future__ import annotations

import pytest

from mem01.store.memory_store import InMemoryBeliefStore


@pytest.fixture
def memory_store() -> InMemoryBeliefStore:
    """Fresh in-memory store per test."""
    return InMemoryBeliefStore()
