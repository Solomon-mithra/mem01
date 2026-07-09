"""Store factory without Postgres."""

from __future__ import annotations

from mem01.store.factory import create_belief_store
from mem01.store.memory_store import InMemoryBeliefStore
from mem01.store.sqlite_store import SqliteBeliefStore


def test_factory_memory_backend(monkeypatch):
    monkeypatch.setenv("MEM01_STORE", "memory")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MEM01_DATABASE_URL", raising=False)
    s = create_belief_store("")
    assert isinstance(s, InMemoryBeliefStore)


def test_factory_sqlite_path(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MEM01_DATABASE_URL", raising=False)
    monkeypatch.setenv("MEM01_STORE", "sqlite")
    path = tmp_path / "t.db"
    s = create_belief_store(f"sqlite:///{path}")
    assert isinstance(s, SqliteBeliefStore)
    s.close()
