"""In-memory BeliefStore behavior."""

from __future__ import annotations

from datetime import timedelta

import pytest

from mem01.store.memory_store import InMemoryBeliefStore, cosine_similarity
from mem01.types import Belief, BeliefStatus, ScopeIds, utc_now


@pytest.fixture
def store() -> InMemoryBeliefStore:
    return InMemoryBeliefStore()


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


def test_upsert_and_get(store: InMemoryBeliefStore):
    b = _belief("User prefers TypeScript")
    store.upsert(b)
    got = store.get(b.id)
    assert got is not None
    assert got.content == "User prefers TypeScript"
    assert got.id == b.id


def test_get_missing_returns_none(store: InMemoryBeliefStore):
    assert store.get("bel_missing") is None


def test_upsert_replaces_existing(store: InMemoryBeliefStore):
    b = _belief("v1")
    store.upsert(b)
    updated = b.model_copy(update={"content": "v2"})
    store.upsert(updated)
    assert store.get(b.id).content == "v2"
    assert len(store) == 1


def test_set_status_supersede(store: InMemoryBeliefStore):
    b = _belief("lives in NY")
    store.upsert(b)
    when = utc_now()
    out = store.set_status(b.id, BeliefStatus.SUPERSEDED, valid_to=when)
    assert out is not None
    assert out.status == BeliefStatus.SUPERSEDED
    assert out.valid_to == when
    assert store.get(b.id).status == BeliefStatus.SUPERSEDED


def test_set_status_missing_returns_none(store: InMemoryBeliefStore):
    assert store.set_status("bel_nope", BeliefStatus.INVALIDATED) is None


def test_list_by_scope_defaults_to_active_only(store: InMemoryBeliefStore):
    a = _belief("active one")
    s = _belief("old one", status=BeliefStatus.SUPERSEDED)
    store.upsert(a)
    store.upsert(s)
    listed = store.list_by_scope(ScopeIds(user_id="u1"))
    assert [x.id for x in listed] == [a.id]


def test_list_by_scope_isolates_users(store: InMemoryBeliefStore):
    mine = _belief("mine", user_id="u1")
    theirs = _belief("theirs", user_id="u2")
    store.upsert(mine)
    store.upsert(theirs)
    listed = store.list_by_scope(ScopeIds(user_id="u1"))
    assert len(listed) == 1
    assert listed[0].id == mine.id


def test_list_by_scope_project_filter(store: InMemoryBeliefStore):
    p1 = _belief("repo a", project_id="proj-a")
    p2 = _belief("repo b", project_id="proj-b")
    store.upsert(p1)
    store.upsert(p2)
    listed = store.list_by_scope(ScopeIds(user_id="u1", project_id="proj-a"))
    assert [x.id for x in listed] == [p1.id]


def test_cosine_similarity_identical_and_orthogonal():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_similarity_search_orders_by_score(store: InMemoryBeliefStore):
    close = _belief("lives in San Francisco")
    far = _belief("prefers dark mode")
    store.upsert(close)
    store.upsert(far)
    # Hand-built vectors: query ~ close
    store.save_embedding(close.id, [1.0, 0.0, 0.0])
    store.save_embedding(far.id, [0.0, 1.0, 0.0])
    results = store.similarity_search(
        [0.9, 0.1, 0.0],
        ScopeIds(user_id="u1"),
        k=2,
    )
    assert len(results) == 2
    assert results[0].belief.id == close.id
    assert results[0].score > results[1].score


def test_similarity_search_skips_missing_embeddings(store: InMemoryBeliefStore):
    with_vec = _belief("has vec")
    no_vec = _belief("no vec")
    store.upsert(with_vec)
    store.upsert(no_vec)
    store.save_embedding(with_vec.id, [1.0, 0.0])
    results = store.similarity_search([1.0, 0.0], ScopeIds(user_id="u1"), k=10)
    assert [r.belief.id for r in results] == [with_vec.id]


def test_similarity_search_respects_k(store: InMemoryBeliefStore):
    for i in range(5):
        b = _belief(f"fact {i}")
        store.upsert(b)
        store.save_embedding(b.id, [1.0, float(i)])
    results = store.similarity_search([1.0, 0.0], ScopeIds(user_id="u1"), k=2)
    assert len(results) == 2


def test_get_returns_copy_not_alias(store: InMemoryBeliefStore):
    b = _belief("immutable inside")
    store.upsert(b)
    got = store.get(b.id)
    assert got is not None
    # Changing a copy must not affect the store (we deep-copy on get/upsert)
    mutated = got.model_copy(update={"content": "hacked"})
    assert store.get(b.id).content == "immutable inside"
    assert mutated.content == "hacked"
