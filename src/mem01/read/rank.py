"""Rank memory candidates for packing.

Why not pure cosine alone:
- Recency and confidence matter for long-lived agents (stale near-matches lose)
- Single tunable score keeps pack() simple: sort desc, greedy fill
"""

from __future__ import annotations

import math
from datetime import datetime

from mem01.types import ScoredBelief, utc_now


def score_candidate(
    scored: ScoredBelief,
    *,
    now: datetime | None = None,
    similarity_weight: float = 0.6,
    recency_weight: float = 0.25,
    confidence_weight: float = 0.15,
) -> float:
    """Combine similarity, recency, and confidence into one rank score."""
    when = now or utc_now()
    sim = max(0.0, float(scored.score))  # cosine can be negative; floor for ranking
    conf = max(0.0, min(1.0, scored.belief.confidence))

    # Recency: half-life ~30 days (smooth decay, never zero)
    age_seconds = max(0.0, (when - scored.belief.updated_at).total_seconds())
    half_life = 30.0 * 24 * 3600
    recency = math.exp(-math.log(2) * age_seconds / half_life)

    return (
        similarity_weight * sim
        + recency_weight * recency
        + confidence_weight * conf
    )


def rank_candidates(
    candidates: list[ScoredBelief],
    *,
    now: datetime | None = None,
) -> list[ScoredBelief]:
    """Return candidates sorted by composite score (highest first).

    Note: returns new ScoredBeliefs with .score set to the composite rank
    so downstream pack can sort on score alone if needed. Original similarity
    is replaced — call rank only after conflict filter.
    """
    when = now or utc_now()
    ranked: list[ScoredBelief] = []
    for c in candidates:
        s = score_candidate(c, now=when)
        ranked.append(ScoredBelief(belief=c.belief, score=s))
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked
