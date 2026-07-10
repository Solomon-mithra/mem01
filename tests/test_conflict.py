"""Conflict filter — read path, no LLM."""

from __future__ import annotations

from datetime import timedelta

from mem01.read.conflict import filter_conflicts
from mem01.types import Belief, BeliefStatus, ScopeIds, ScoredBelief, utc_now


def _scored(
    content: str,
    *,
    score: float = 0.5,
    status: BeliefStatus = BeliefStatus.ACTIVE,
    topic_key: str | None = None,
    confidence: float = 0.7,
    valid_to=None,
    valid_from=None,
) -> ScoredBelief:
    meta = {}
    if topic_key:
        meta["topic_key"] = topic_key
    b = Belief(
        content=content,
        status=status,
        confidence=confidence,
        valid_from=valid_from,
        valid_to=valid_to,
        scope_ids=ScopeIds(user_id="u1"),
        metadata=meta,
    )
    return ScoredBelief(belief=b, score=score)


def test_drops_non_active():
    active = _scored("live in SF", status=BeliefStatus.ACTIVE, score=0.9)
    old = _scored("live in NY", status=BeliefStatus.SUPERSEDED, score=0.95)
    out = filter_conflicts([old, active])
    assert len(out) == 1
    assert out[0].belief.content == "live in SF"


def test_history_mode_keeps_superseded():
    active = _scored("live in SF", status=BeliefStatus.ACTIVE, score=0.9)
    old = _scored("live in NY", status=BeliefStatus.SUPERSEDED, score=0.95)
    out = filter_conflicts([old, active], mode="history")
    assert len(out) == 2
    contents = {c.belief.content for c in out}
    assert contents == {"live in SF", "live in NY"}


def test_drops_expired_valid_to():
    now = utc_now()
    expired = _scored(
        "old job",
        valid_to=now - timedelta(days=1),
        score=0.99,
    )
    current = _scored("new job", score=0.5)
    out = filter_conflicts([expired, current], at=now)
    assert len(out) == 1
    assert out[0].belief.content == "new job"


def test_topic_key_keeps_higher_score():
    low = _scored("lives in NY", topic_key="location", score=0.4, confidence=0.9)
    high = _scored("lives in SF", topic_key="location", score=0.8, confidence=0.5)
    other = _scored("likes tea", topic_key="drink", score=0.7)
    out = filter_conflicts([low, high, other])
    contents = {c.belief.content for c in out}
    assert contents == {"lives in SF", "likes tea"}


def test_topic_key_tie_break_confidence():
    a = _scored("pref dark", topic_key="theme", score=0.5, confidence=0.4)
    b = _scored("pref light", topic_key="theme", score=0.5, confidence=0.9)
    out = filter_conflicts([a, b])
    assert len(out) == 1
    assert out[0].belief.content == "pref light"


def test_no_topic_key_all_kept():
    a = _scored("fact a", score=0.5)
    b = _scored("fact b", score=0.6)
    out = filter_conflicts([a, b])
    assert len(out) == 2


def test_preserves_relative_order_of_survivors():
    a = _scored("a", topic_key="t1", score=0.9)
    b = _scored("b", score=0.5)  # no topic
    c = _scored("c", topic_key="t1", score=0.1)  # loses to a
    d = _scored("d", topic_key="t2", score=0.7)
    out = filter_conflicts([a, b, c, d])
    assert [x.belief.content for x in out] == ["a", "b", "d"]
