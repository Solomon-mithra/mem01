"""Read-path search — no LLM."""

from __future__ import annotations

import pytest

from mem01.embeddings.fake import FakeEmbedder
from mem01.read.search import search_beliefs
from mem01.store.memory_store import InMemoryBeliefStore
from mem01.types import Belief, BeliefStatus, ScopeIds
from mem01.write.apply_ops import apply_ops
from mem01.types import BeliefOp, BeliefOpType


@pytest.fixture
def store() -> InMemoryBeliefStore:
    return InMemoryBeliefStore()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimensions=16)


def test_search_returns_empty_for_blank_query(store, embedder):
    assert search_beliefs(store, embedder, "  ", ScopeIds(user_id="u1")) == []


def test_search_orders_by_similarity(store, embedder):
    # Hand-seed so order is controlled (FakeEmbedder is hash-based, not semantic)
    close = Belief(content="User lives in San Francisco", scope_ids=ScopeIds(user_id="u1"))
    far = Belief(content="User prefers dark mode", scope_ids=ScopeIds(user_id="u1"))
    store.upsert(close)
    store.upsert(far)
    store.save_embedding(close.id, [1.0, 0.0, 0.0, 0.0] + [0.0] * 12)
    store.save_embedding(far.id, [0.0, 1.0, 0.0, 0.0] + [0.0] * 12)

    # Embedder for query is bypassed by using a store vector aligned to close —
    # we still go through search_beliefs which embeds query; monkeypatch embed
    class FixedEmbedder:
        dimensions = 16

        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0] + [0.0] * 12

    results = search_beliefs(
        store,
        FixedEmbedder(),
        "where do I live?",
        ScopeIds(user_id="u1"),
        k=2,
    )
    assert len(results) == 2
    assert results[0].belief.id == close.id
    assert results[0].score > results[1].score


def test_search_isolates_users(store, embedder):
    apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.ADD,
                content="u1 secret fact alpha",
                scope_ids=ScopeIds(user_id="u1"),
            )
        ],
        embedder,
    )
    apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.ADD,
                content="u2 other fact beta",
                scope_ids=ScopeIds(user_id="u2"),
            )
        ],
        embedder,
    )
    results = search_beliefs(
        store,
        embedder,
        "secret fact",
        ScopeIds(user_id="u1"),
        k=10,
    )
    assert all(r.belief.scope_ids.user_id == "u1" for r in results)
    assert all(r.belief.scope_ids.user_id != "u2" for r in results)


def test_search_skips_superseded_by_default(store, embedder):
    add = apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.ADD,
                content="User lives in New York",
                scope_ids=ScopeIds(user_id="u1"),
            )
        ],
        embedder,
    )
    old_id = add.created_ids[0]
    apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.SUPERSEDE,
                target_id=old_id,
                content="User lives in San Francisco",
                scope_ids=ScopeIds(user_id="u1"),
            )
        ],
        embedder,
    )
    results = search_beliefs(
        store,
        embedder,
        "where does the user live",
        ScopeIds(user_id="u1"),
        k=10,
    )
    ids = {r.belief.id for r in results}
    assert old_id not in ids
    assert all(r.belief.status == BeliefStatus.ACTIVE for r in results)


def test_search_respects_k(store, embedder):
    for i in range(5):
        apply_ops(
            store,
            [
                BeliefOp(
                    op=BeliefOpType.ADD,
                    content=f"fact number {i} unique token{i}",
                    scope_ids=ScopeIds(user_id="u1"),
                )
            ],
            embedder,
        )
    results = search_beliefs(
        store,
        embedder,
        "fact number",
        ScopeIds(user_id="u1"),
        k=2,
    )
    assert len(results) == 2
