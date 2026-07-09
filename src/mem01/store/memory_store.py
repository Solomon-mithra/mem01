"""In-memory BeliefStore — fast tests and reference behavior.

Why this exists before SQLite:
- Proves repository semantics in milliseconds without files or vector extensions
- Golden behavior for later backends to match in tests
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable

from mem01.types import (
    Belief,
    BeliefStatus,
    ScopeIds,
    ScoredBelief,
    utc_now,
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero-length."""
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class InMemoryBeliefStore:
    """Dict-backed store implementing the BeliefStore protocol."""

    def __init__(self) -> None:
        self._beliefs: dict[str, Belief] = {}
        self._embeddings: dict[str, list[float]] = {}

    def get(self, belief_id: str) -> Belief | None:
        return self._beliefs.get(belief_id)

    def upsert(self, belief: Belief) -> None:
        # Store a copy so callers cannot mutate internal state by accident
        self._beliefs[belief.id] = belief.model_copy(deep=True)

    def set_status(
        self,
        belief_id: str,
        status: BeliefStatus,
        *,
        valid_to: datetime | None = None,
    ) -> Belief | None:
        belief = self._beliefs.get(belief_id)
        if belief is None:
            return None
        updates: dict = {
            "status": status,
            "updated_at": utc_now(),
        }
        if valid_to is not None:
            updates["valid_to"] = valid_to
        updated = belief.model_copy(update=updates)
        self._beliefs[belief_id] = updated
        return updated.model_copy(deep=True)

    def list_by_scope(
        self,
        scope_filter: ScopeIds,
        *,
        statuses: set[BeliefStatus] | None = None,
    ) -> list[Belief]:
        wanted = statuses if statuses is not None else {BeliefStatus.ACTIVE}
        out: list[Belief] = []
        for belief in self._beliefs.values():
            if belief.status not in wanted:
                continue
            if not scope_filter.matches(belief.scope_ids):
                continue
            out.append(belief.model_copy(deep=True))
        return out

    def save_embedding(self, belief_id: str, vector: list[float]) -> None:
        self._embeddings[belief_id] = list(vector)

    def get_embedding(self, belief_id: str) -> list[float] | None:
        vec = self._embeddings.get(belief_id)
        return list(vec) if vec is not None else None

    def similarity_search(
        self,
        vector: list[float],
        scope_filter: ScopeIds,
        *,
        k: int = 20,
        statuses: set[BeliefStatus] | None = None,
    ) -> list[ScoredBelief]:
        if k <= 0:
            return []
        candidates = self.list_by_scope(scope_filter, statuses=statuses)
        scored: list[ScoredBelief] = []
        for belief in candidates:
            emb = self._embeddings.get(belief.id)
            if emb is None:
                continue
            score = cosine_similarity(vector, emb)
            scored.append(ScoredBelief(belief=belief, score=score))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    # --- test / debug helpers (not part of protocol) ---

    def clear(self) -> None:
        self._beliefs.clear()
        self._embeddings.clear()

    def __len__(self) -> int:
        return len(self._beliefs)

    def all_ids(self) -> Iterable[str]:
        return self._beliefs.keys()
