"""Store factory."""

from __future__ import annotations

import pytest

from mem01.store.factory import create_belief_store
from mem01.store.memory_store import InMemoryBeliefStore


def test_factory_memory_backend(monkeypatch):
    monkeypatch.setenv("MEM01_STORE", "memory")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MEM01_DATABASE_URL", raising=False)
    s = create_belief_store()
    assert isinstance(s, InMemoryBeliefStore)


def test_factory_requires_database_url(monkeypatch):
    monkeypatch.delenv("MEM01_STORE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MEM01_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        create_belief_store()
