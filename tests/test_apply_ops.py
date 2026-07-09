"""apply_ops — product write semantics without an LLM."""

from __future__ import annotations

import pytest

from mem01.embeddings.fake import FakeEmbedder
from mem01.store.memory_store import InMemoryBeliefStore
from mem01.types import (
    Belief,
    BeliefOp,
    BeliefOpType,
    BeliefStatus,
    ScopeIds,
)
from mem01.write.apply_ops import apply_ops


@pytest.fixture
def store() -> InMemoryBeliefStore:
    return InMemoryBeliefStore()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimensions=16)


def _scope(user_id: str = "u1") -> ScopeIds:
    return ScopeIds(user_id=user_id)


def test_add_creates_active_belief_with_embedding(store, embedder):
    result = apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.ADD,
                content="User prefers TypeScript",
                scope_ids=_scope(),
                topic_key="language_pref",
            )
        ],
        embedder,
    )
    assert result.ok
    assert len(result.created_ids) == 1
    bid = result.created_ids[0]
    b = store.get(bid)
    assert b is not None
    assert b.status == BeliefStatus.ACTIVE
    assert b.content == "User prefers TypeScript"
    assert b.metadata.get("topic_key") == "language_pref"
    assert store.get_embedding(bid) is not None
    active = store.list_by_scope(_scope())
    assert len(active) == 1


def test_supersede_deactivates_old_activates_new(store, embedder):
    # Seed old belief
    add = apply_ops(
        store,
        [BeliefOp(op=BeliefOpType.ADD, content="User lives in New York", scope_ids=_scope())],
        embedder,
    )
    old_id = add.created_ids[0]

    result = apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.SUPERSEDE,
                target_id=old_id,
                content="User lives in San Francisco",
                scope_ids=_scope(),
                topic_key="location",
            )
        ],
        embedder,
    )
    assert result.ok
    assert old_id in result.superseded_ids
    assert len(result.created_ids) == 1
    new_id = result.created_ids[0]

    old = store.get(old_id)
    new = store.get(new_id)
    assert old is not None and new is not None
    assert old.status == BeliefStatus.SUPERSEDED
    assert old.valid_to is not None
    assert new.status == BeliefStatus.ACTIVE
    assert new.supersedes_id == old_id
    assert new.content == "User lives in San Francisco"

    active = store.list_by_scope(_scope())
    assert len(active) == 1
    assert active[0].id == new_id


def test_invalidate_hides_from_active_list(store, embedder):
    add = apply_ops(
        store,
        [BeliefOp(op=BeliefOpType.ADD, content="temp wrong fact", scope_ids=_scope())],
        embedder,
    )
    bid = add.created_ids[0]
    result = apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.INVALIDATE,
                target_id=bid,
                reason="user said forget that",
            )
        ],
        embedder,
    )
    assert result.ok
    assert bid in result.invalidated_ids
    b = store.get(bid)
    assert b is not None
    assert b.status == BeliefStatus.INVALIDATED
    assert store.list_by_scope(_scope()) == []


def test_update_changes_content_and_reembeds(store, embedder):
    add = apply_ops(
        store,
        [BeliefOp(op=BeliefOpType.ADD, content="likes coffee", scope_ids=_scope())],
        embedder,
    )
    bid = add.created_ids[0]
    old_emb = store.get_embedding(bid)

    result = apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.UPDATE,
                target_id=bid,
                content="likes tea",
                confidence=0.95,
            )
        ],
        embedder,
    )
    assert result.ok
    assert bid in result.updated_ids
    b = store.get(bid)
    assert b is not None
    assert b.content == "likes tea"
    assert b.confidence == 0.95
    assert b.status == BeliefStatus.ACTIVE
    new_emb = store.get_embedding(bid)
    assert new_emb is not None
    assert new_emb != old_emb


def test_two_adds_remain_two_active_beliefs(store, embedder):
    """Extractor's job to merge; apply only does what it's told."""
    result = apply_ops(
        store,
        [
            BeliefOp(op=BeliefOpType.ADD, content="fact A", scope_ids=_scope()),
            BeliefOp(op=BeliefOpType.ADD, content="fact B", scope_ids=_scope()),
        ],
        embedder,
    )
    assert result.ok
    assert len(result.created_ids) == 2
    assert len(store.list_by_scope(_scope())) == 2


def test_merge_collapses_duplicates(store, embedder):
    a = apply_ops(
        store,
        [BeliefOp(op=BeliefOpType.ADD, content="User lives in SF", scope_ids=_scope())],
        embedder,
    )
    b = apply_ops(
        store,
        [BeliefOp(op=BeliefOpType.ADD, content="User is based in San Francisco", scope_ids=_scope())],
        embedder,
    )
    id_a, id_b = a.created_ids[0], b.created_ids[0]

    result = apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.MERGE,
                target_ids=[id_a, id_b],
                content="User lives in San Francisco",
                scope_ids=_scope(),
            )
        ],
        embedder,
    )
    assert result.ok
    assert id_a in result.merged_ids and id_b in result.merged_ids
    active = store.list_by_scope(_scope())
    assert len(active) == 1
    assert active[0].content == "User lives in San Francisco"
    assert set(active[0].metadata.get("merged_from", [])) == {id_a, id_b}
    assert store.get(id_a).status == BeliefStatus.SUPERSEDED
    assert store.get(id_b).status == BeliefStatus.SUPERSEDED


def test_supersede_missing_target_records_error(store, embedder):
    result = apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.SUPERSEDE,
                target_id="bel_missing",
                content="new value",
            )
        ],
        embedder,
    )
    assert not result.ok
    assert any("not found" in e for e in result.errors)


def test_batch_continues_after_one_error(store, embedder):
    result = apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.SUPERSEDE,
                target_id="bel_missing",
                content="nope",
            ),
            BeliefOp(op=BeliefOpType.ADD, content="still added", scope_ids=_scope()),
        ],
        embedder,
    )
    assert result.errors
    assert len(result.created_ids) == 1
    assert store.list_by_scope(_scope())[0].content == "still added"
