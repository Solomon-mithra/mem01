"""SQLite BeliefStore — same behavioral contract as in-memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from mem01.store.memory_store import InMemoryBeliefStore
from mem01.store.sqlite_store import SqliteBeliefStore
from mem01.types import Belief, BeliefStatus, ScopeIds, utc_now


def _belief(
    content: str,
    *,
    user_id: str = "u1",
    project_id: str | None = None,
    status: BeliefStatus = BeliefStatus.ACTIVE,
) -> Belief:
    return Belief(
        content=content,
        status=status,
        scope_ids=ScopeIds(user_id=user_id, project_id=project_id),
    )


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path: Path):
    """Run each contract test against both backends."""
    if request.param == "memory":
        yield InMemoryBeliefStore()
    else:
        path = tmp_path / "mem01.db"
        s = SqliteBeliefStore(path)
        yield s
        s.close()


def test_upsert_get_roundtrip(store):
    b = _belief("User prefers TypeScript")
    store.upsert(b)
    got = store.get(b.id)
    assert got is not None
    assert got.content == b.content
    assert got.scope_ids.user_id == "u1"
    assert got.status == BeliefStatus.ACTIVE


def test_persistence_across_reconnect(tmp_path: Path):
    path = tmp_path / "persist.db"
    b = _belief("survives restart")
    with SqliteBeliefStore(path) as s1:
        s1.upsert(b)
        s1.save_embedding(b.id, [0.1, 0.2, 0.3])

    with SqliteBeliefStore(path) as s2:
        got = s2.get(b.id)
        assert got is not None
        assert got.content == "survives restart"
        emb = s2.get_embedding(b.id)
        assert emb == pytest.approx([0.1, 0.2, 0.3])


def test_set_status_and_list_active(store):
    a = _belief("active")
    store.upsert(a)
    store.set_status(a.id, BeliefStatus.SUPERSEDED, valid_to=utc_now())
    assert store.list_by_scope(ScopeIds(user_id="u1")) == []
    all_statuses = store.list_by_scope(
        ScopeIds(user_id="u1"),
        statuses={BeliefStatus.SUPERSEDED},
    )
    assert len(all_statuses) == 1
    assert all_statuses[0].status == BeliefStatus.SUPERSEDED


def test_scope_isolation(store):
    store.upsert(_belief("mine", user_id="u1"))
    store.upsert(_belief("theirs", user_id="u2"))
    listed = store.list_by_scope(ScopeIds(user_id="u1"))
    assert len(listed) == 1
    assert listed[0].content == "mine"


def test_similarity_search_order(store):
    close = _belief("lives in SF")
    far = _belief("likes pizza")
    store.upsert(close)
    store.upsert(far)
    store.save_embedding(close.id, [1.0, 0.0, 0.0])
    store.save_embedding(far.id, [0.0, 1.0, 0.0])
    results = store.similarity_search([1.0, 0.0, 0.0], ScopeIds(user_id="u1"), k=2)
    assert results[0].belief.id == close.id
    assert results[0].score > results[1].score


def test_save_embedding_unknown_id_raises():
    s = SqliteBeliefStore(":memory:")
    try:
        with pytest.raises(KeyError):
            s.save_embedding("bel_missing", [1.0])
    finally:
        s.close()
