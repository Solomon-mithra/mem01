"""BeliefStore protocol — the only door to persistence.

Why a protocol (interface) instead of talking to SQLite everywhere:
- apply_ops and recall stay pure logic; we can unit-test with RAM
- Later swap InMemory → SQLite → Postgres+pgvector without rewriting pipelines
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from mem01.types import Belief, BeliefStatus, ScopeIds, ScoredBelief


# Alias for call-site clarity: a filter is just ScopeIds used as "wanted fields"
ScopeFilter = ScopeIds


@runtime_checkable
class BeliefStore(Protocol):
    """Repository for beliefs and their embedding vectors."""

    def get(self, belief_id: str) -> Belief | None:
        """Return belief by id, or None if missing."""
        ...

    def upsert(self, belief: Belief) -> None:
        """Insert or replace a belief row (does not touch embeddings)."""
        ...

    def set_status(
        self,
        belief_id: str,
        status: BeliefStatus,
        *,
        valid_to: datetime | None = None,
    ) -> Belief | None:
        """Update status (and optional valid_to). Returns updated belief or None."""
        ...

    def list_by_scope(
        self,
        scope_filter: ScopeFilter,
        *,
        statuses: set[BeliefStatus] | None = None,
    ) -> list[Belief]:
        """List beliefs matching scope filter and status set (default: active only)."""
        ...

    def save_embedding(self, belief_id: str, vector: list[float]) -> None:
        """Store or replace the embedding for a belief id."""
        ...

    def get_embedding(self, belief_id: str) -> list[float] | None:
        """Return embedding vector or None."""
        ...

    def similarity_search(
        self,
        vector: list[float],
        scope_filter: ScopeFilter,
        *,
        k: int = 20,
        statuses: set[BeliefStatus] | None = None,
    ) -> list[ScoredBelief]:
        """Top-k cosine similarity within scope + status filters (higher score = closer)."""
        ...
