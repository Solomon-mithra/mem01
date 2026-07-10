"""Conflict-safe filtering on read candidates — no LLM.

Why this exists (product wedge):
- Even with SUPERSEDE on write, near-duplicates or slipped ADDs can both be active
- Default recall must not inject two opposite truths when we can detect them
- Deterministic rules only — keeps hot path cheap (mem0-class latency/$)

v1 rules (mode=current):
1. Drop non-active statuses
2. Drop outside validity window (valid_from / valid_to)
3. If multiple candidates share metadata topic_key, keep best (score, then confidence, then newer)

mode=history (audit / temporal questions):
- Keep active + superseded (+ optional invalidated)
- Do NOT collapse topic_key to one winner — both SF (active) and NY (superseded) surface
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from mem01.types import BeliefStatus, ScoredBelief, utc_now

HistoryMode = Literal["current", "history"]

_HISTORY_STATUSES = frozenset(
    {
        BeliefStatus.ACTIVE,
        BeliefStatus.SUPERSEDED,
        BeliefStatus.INVALIDATED,
    }
)


def filter_conflicts(
    candidates: list[ScoredBelief],
    *,
    at: datetime | None = None,
    mode: HistoryMode = "current",
) -> list[ScoredBelief]:
    """Return a filtered subset of *candidates* (order preserved among survivors)."""
    when = at or utc_now()
    if mode == "history":
        return [c for c in candidates if c.belief.status in _HISTORY_STATUSES]
    current = [c for c in candidates if _is_eligible(c, when)]
    return _dedupe_by_topic_key(current)


def _is_eligible(scored: ScoredBelief, when: datetime) -> bool:
    b = scored.belief
    if b.status != BeliefStatus.ACTIVE:
        return False
    # Reuse Belief.is_current for validity window
    return b.is_current(when)


def _dedupe_by_topic_key(candidates: list[ScoredBelief]) -> list[ScoredBelief]:
    """For each topic_key, keep a single winner; beliefs without topic_key all pass."""
    best_by_topic: dict[str, ScoredBelief] = {}
    no_topic: list[ScoredBelief] = []

    for c in candidates:
        key = c.belief.metadata.get("topic_key")
        if not key or not isinstance(key, str):
            no_topic.append(c)
            continue
        prev = best_by_topic.get(key)
        if prev is None or _better(c, prev):
            best_by_topic[key] = c

    # Preserve original relative order: walk candidates, emit if selected
    winners = set(id(v) for v in best_by_topic.values())
    winners.update(id(c) for c in no_topic)
    return [c for c in candidates if id(c) in winners]


def _better(a: ScoredBelief, b: ScoredBelief) -> bool:
    """True if *a* should replace *b* for the same topic_key."""
    if a.score != b.score:
        return a.score > b.score
    if a.belief.confidence != b.belief.confidence:
        return a.belief.confidence > b.belief.confidence
    # Newer updated_at wins
    return a.belief.updated_at >= b.belief.updated_at
