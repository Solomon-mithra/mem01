"""Create a BeliefStore from env / URL.

Production (self-hosted or Neon):
  DATABASE_URL=postgresql://...
  or MEM01_DATABASE_URL=...

Local dev (default):
  sqlite file or in-memory if unset
"""

from __future__ import annotations

import os
from pathlib import Path

from mem01.env import load_env
from mem01.store.memory_store import InMemoryBeliefStore
from mem01.store.sqlite_store import SqliteBeliefStore


def create_belief_store(
    url: str | None = None,
    *,
    embedding_dim: int | None = None,
):
    """Return InMemory, Sqlite, or Postgres store based on *url* / env.

    URL schemes:
      - unset / empty → InMemoryBeliefStore (tests) if MEM01_STORE=memory,
        else SqliteBeliefStore at ./data/mem01.db
      - sqlite:///:memory: or :memory: → SQLite memory
      - sqlite:///path or path ending .db → SQLite file
      - postgresql://... or postgres://... → PostgresBeliefStore + pgvector
    """
    load_env()
    if url is None:
        url = (
            os.environ.get("MEM01_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
            or ""
        ).strip()

    dim = embedding_dim
    if dim is None:
        raw = os.environ.get("MEM01_EMBEDDING_DIM", "1536").strip()
        dim = int(raw) if raw else 1536

    if not url:
        backend = os.environ.get("MEM01_STORE", "sqlite").strip().lower()
        if backend in ("memory", "mem", "inmemory"):
            return InMemoryBeliefStore()
        path = os.environ.get("MEM01_SQLITE_PATH", "data/mem01.db")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteBeliefStore(path)

    lower = url.lower()
    if lower in (":memory:", "sqlite:///:memory:", "sqlite://memory"):
        return SqliteBeliefStore(":memory:")

    if lower.startswith("sqlite:///"):
        path = url[len("sqlite:///") :]
        if path not in (":memory:",):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteBeliefStore(path)

    if lower.endswith(".db") or lower.startswith("sqlite:"):
        path = url.removeprefix("sqlite:")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return SqliteBeliefStore(path)

    if lower.startswith("postgres://") or lower.startswith("postgresql://"):
        # Neon sometimes uses postgres:// — psycopg accepts both
        from mem01.store.postgres_store import PostgresBeliefStore

        dsn = url
        if dsn.startswith("postgres://"):
            dsn = "postgresql://" + dsn[len("postgres://") :]
        return PostgresBeliefStore(dsn, embedding_dim=dim)

    raise ValueError(
        f"Unsupported store URL: {url!r}. "
        "Use postgresql://... (or Neon), sqlite:///path, or leave unset for local SQLite."
    )
