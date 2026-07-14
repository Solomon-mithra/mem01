"""In-memory BeliefStore — fast tests and reference behavior.

Why this exists before SQLite:
- Proves repository semantics in milliseconds without files or vector extensions
- Golden behavior for later backends to match in tests
"""

from __future__ import annotations

import math
import threading
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
        self._lock = threading.RLock()
        self._beliefs: dict[str, Belief] = {}
        self._embeddings: dict[str, list[float]] = {}

    def get(self, belief_id: str) -> Belief | None:
        with self._lock:
            belief = self._beliefs.get(belief_id)
            return belief.model_copy(deep=True) if belief is not None else None

    def delete_by_user(self, user_id: str) -> int:
        if not user_id.strip():
            raise ValueError("user_id must be non-empty")
        with self._lock:
            belief_ids = [
                belief_id
                for belief_id, belief in self._beliefs.items()
                if belief.scope_ids.user_id == user_id
            ]
            for belief_id in belief_ids:
                del self._beliefs[belief_id]
                self._embeddings.pop(belief_id, None)
            return len(belief_ids)

    def upsert(self, belief: Belief) -> None:
        # Store a copy so callers cannot mutate internal state by accident
        with self._lock:
            self._beliefs[belief.id] = belief.model_copy(deep=True)

    def set_status(
        self,
        belief_id: str,
        status: BeliefStatus,
        *,
        valid_to: datetime | None = None,
    ) -> Belief | None:
        with self._lock:
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

    def supersede_if_owned(
        self,
        target_id: str,
        replacement: Belief,
        replacement_embedding: list[float],
        *,
        expected_user_id: str,
        superseded_at: datetime,
    ) -> bool:
        with self._lock:
            target = self._beliefs.get(target_id)
            if (
                target is None
                or target.status != BeliefStatus.ACTIVE
                or target.scope_ids.user_id != expected_user_id
                or replacement.scope_ids.user_id != expected_user_id
            ):
                return False
            self._beliefs[replacement.id] = replacement.model_copy(deep=True)
            self._embeddings[replacement.id] = list(replacement_embedding)
            self._beliefs[target_id] = target.model_copy(
                update={
                    "status": BeliefStatus.SUPERSEDED,
                    "valid_to": superseded_at,
                    "updated_at": superseded_at,
                }
            )
            return True

    def invalidate_if_owned(
        self,
        belief_id: str,
        *,
        expected_user_id: str,
        reason: str | None,
        invalidated_at: datetime,
    ) -> bool:
        with self._lock:
            target = self._beliefs.get(belief_id)
            if (
                target is None
                or target.status != BeliefStatus.ACTIVE
                or target.scope_ids.user_id != expected_user_id
            ):
                return False
            metadata = dict(target.metadata)
            if reason:
                metadata["invalidate_reason"] = reason
            self._beliefs[belief_id] = target.model_copy(
                update={
                    "status": BeliefStatus.INVALIDATED,
                    "valid_to": invalidated_at,
                    "updated_at": invalidated_at,
                    "metadata": metadata,
                }
            )
            return True

    def list_by_scope(
        self,
        scope_filter: ScopeIds,
        *,
        statuses: set[BeliefStatus] | None = None,
    ) -> list[Belief]:
        with self._lock:
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
        with self._lock:
            self._embeddings[belief_id] = list(vector)

    def get_embedding(self, belief_id: str) -> list[float] | None:
        with self._lock:
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
        with self._lock:
            self._beliefs.clear()
            self._embeddings.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._beliefs)

    def all_ids(self) -> Iterable[str]:
        with self._lock:
            return tuple(self._beliefs.keys())
