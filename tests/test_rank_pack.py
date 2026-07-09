"""Rank + token pack — read path, no LLM."""

from __future__ import annotations

from datetime import timedelta

from mem01.read.pack import pack_beliefs
from mem01.read.rank import rank_candidates, score_candidate
from mem01.tokens import estimate_tokens
from mem01.types import Belief, ScopeIds, ScoredBelief, utc_now


def _scored(
    content: str,
    *,
    score: float,
    confidence: float = 0.7,
    updated_at=None,
) -> ScoredBelief:
    b = Belief(
        content=content,
        confidence=confidence,
        scope_ids=ScopeIds(user_id="u1"),
        updated_at=updated_at or utc_now(),
    )
    return ScoredBelief(belief=b, score=score)


def test_estimate_tokens_positive():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 8) == 2


def test_rank_prefers_higher_similarity():
    low = _scored("a", score=0.2)
    high = _scored("b", score=0.9)
    ranked = rank_candidates([low, high])
    assert ranked[0].belief.content == "b"
    assert ranked[0].score > ranked[1].score


def test_rank_boosts_recent_over_stale_same_sim():
    now = utc_now()
    stale = _scored("old", score=0.8, updated_at=now - timedelta(days=90))
    fresh = _scored("new", score=0.8, updated_at=now)
    # raw similarity equal — composite should prefer fresh
    assert score_candidate(fresh, now=now) > score_candidate(stale, now=now)


def test_pack_never_exceeds_budget():
    # Many medium lines
    cands = [
        _scored(f"memory fact number {i} with some extra words", score=1.0 - i * 0.01)
        for i in range(20)
    ]
    packed = pack_beliefs(cands, max_memory_tokens=50)
    assert packed.tokens_used <= 50
    assert packed.max_memory_tokens == 50
    assert estimate_tokens(packed.text) == packed.tokens_used


def test_pack_prefers_higher_scores():
    low = _scored("low priority uniquezzz", score=0.1)
    high = _scored("high priority uniqueaaa", score=0.99)
    packed = pack_beliefs([low, high], max_memory_tokens=100)
    assert packed.beliefs[0].content.startswith("high")


def test_pack_zero_budget_empty():
    packed = pack_beliefs([_scored("x", score=1.0)], max_memory_tokens=0)
    assert packed.beliefs == []
    assert packed.tokens_used == 0


def test_pack_records_candidate_count():
    cands = [_scored("a", score=1.0), _scored("b", score=0.5)]
    packed = pack_beliefs(cands, max_memory_tokens=800, candidate_count=10)
    assert packed.candidate_count == 10
