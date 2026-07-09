"""Create a BeliefStore from env / URL.

Production & dev (same stack):
  DATABASE_URL=postgresql://...   # local docker or Neon

Tests only:
  MEM01_STORE=memory              # InMemoryBeliefStore (no DB)
"""

from __future__ import annotations

import os

from mem01.env import load_env
from mem01.store.memory_store import InMemoryBeliefStore


def create_belief_store(
    url: str | None = None,
    *,
    embedding_dim: int | None = None,
):
    """Return Postgres+pgvector store, or in-memory for unit tests.

    SQLite is not supported — use docker compose (or Neon) for real runs.
    """
    load_env()

    backend = os.environ.get("MEM01_STORE", "").strip().lower()
    if url is None and backend in ("memory", "mem", "inmemory"):
        return InMemoryBeliefStore()

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
        raise RuntimeError(
            "DATABASE_URL is required (Postgres + pgvector).\n\n"
            "  docker compose up -d\n"
            "  # .env:\n"
            "  DATABASE_URL=postgresql://mem01:mem01@localhost:5433/mem01\n\n"
            "Or Neon:\n"
            "  DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require\n\n"
            "Unit tests only: MEM01_STORE=memory\n"
        )

    lower = url.lower()
    if lower.startswith("postgres://") or lower.startswith("postgresql://"):
        from mem01.store.postgres_store import PostgresBeliefStore

        dsn = url
        if dsn.startswith("postgres://"):
            dsn = "postgresql://" + dsn[len("postgres://") :]
        return PostgresBeliefStore(dsn, embedding_dim=dim)

    raise ValueError(
        f"Unsupported store URL: {url!r}. "
        "Use postgresql://... (Docker Postgres or Neon)."
    )
