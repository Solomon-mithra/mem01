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


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mmr_select(
    ranked: list[ScoredBelief],
    *,
    lambda_weight: float = 0.75,
) -> list[ScoredBelief]:
    """Greedy maximal-marginal-relevance reorder to reduce near-duplicate picks.

    Multi-hop and listing questions need *different* beliefs packed together;
    pure score ordering lets one topic cluster crowd out the rest of the
    budget. Uses token Jaccard between belief contents so it works with any
    store and never touches embeddings on the hot path. O(n^2), n <= recall k.
    """
    if len(ranked) <= 2:
        return list(ranked)

    from mem01.read.search import tokenize

    tokens = {c.belief.id: set(tokenize(c.belief.content)) for c in ranked}
    remaining = list(ranked)
    selected: list[ScoredBelief] = [remaining.pop(0)]
    while remaining:
        best_idx = 0
        best_val = float("-inf")
        for i, cand in enumerate(remaining):
            redundancy = max(
                _jaccard(tokens[cand.belief.id], tokens[s.belief.id])
                for s in selected
            )
            val = lambda_weight * cand.score - (1.0 - lambda_weight) * redundancy
            if val > best_val:
                best_val = val
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected
