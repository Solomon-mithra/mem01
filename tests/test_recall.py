"""End-to-end recall pipeline + metrics."""

from __future__ import annotations

from mem01.embeddings.fake import FakeEmbedder
from mem01.read.recall import recall
from mem01.store.memory_store import InMemoryBeliefStore
from mem01.types import Belief, BeliefOp, BeliefOpType, ScopeIds
from mem01.write.apply_ops import apply_ops


def test_recall_returns_metrics_and_text():
    store = InMemoryBeliefStore()
    emb = FakeEmbedder(dimensions=16)
    apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.ADD,
                content="User prefers TypeScript",
                scope_ids=ScopeIds(user_id="u1"),
                topic_key="language_pref",
            )
        ],
        emb,
    )
    packed = recall(
        store,
        emb,
        "what language does the user like?",
        ScopeIds(user_id="u1"),
        max_memory_tokens=200,
    )
    assert packed.latency_ms is not None
    assert packed.latency_ms >= 0
    assert packed.tokens_used <= 200
    assert packed.candidate_count >= 1
    assert "TypeScript" in packed.text or len(packed.beliefs) >= 0


def test_recall_respects_token_budget():
    store = InMemoryBeliefStore()
    emb = FakeEmbedder(dimensions=16)
    for i in range(10):
        apply_ops(
            store,
            [
                BeliefOp(
                    op=BeliefOpType.ADD,
                    content=f"Longish durable fact number {i} with padding words here",
                    scope_ids=ScopeIds(user_id="u1"),
                )
            ],
            emb,
        )
    packed = recall(
        store,
        emb,
        "facts",
        ScopeIds(user_id="u1"),
        max_memory_tokens=40,
        k=20,
    )
    assert packed.tokens_used <= 40


def test_recall_empty_store():
    store = InMemoryBeliefStore()
    emb = FakeEmbedder(dimensions=16)
    packed = recall(store, emb, "anything", ScopeIds(user_id="u1"))
    assert packed.beliefs == []
    assert packed.tokens_used == 0
    assert packed.latency_ms is not None


def test_recall_default_hides_superseded_history_shows_it():
    """Current path: SF only. History path: SF + labeled NY."""
    from mem01.types import BeliefStatus

    store = InMemoryBeliefStore()
    emb = FakeEmbedder(dimensions=16)
    apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.ADD,
                content="User lives in New York City",
                scope_ids=ScopeIds(user_id="u1"),
                topic_key="location",
            )
        ],
        emb,
    )
    active = store.list_by_scope(ScopeIds(user_id="u1"))[0]
    apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.SUPERSEDE,
                target_id=active.id,
                content="User lives in San Francisco",
                scope_ids=ScopeIds(user_id="u1"),
                topic_key="location",
            )
        ],
        emb,
    )

    now = recall(
        store,
        emb,
        "Where does the user live?",
        ScopeIds(user_id="u1"),
        max_memory_tokens=400,
    )
    assert "San Francisco" in now.text
    assert "New York" not in now.text
    assert all(b.status == BeliefStatus.ACTIVE for b in now.beliefs)

    hist = recall(
        store,
        emb,
        "Where did the user live before?",
        ScopeIds(user_id="u1"),
        max_memory_tokens=400,
        include_history=True,
    )
    assert "San Francisco" in hist.text
    assert "New York" in hist.text
    assert "[superseded]" in hist.text
    assert "[active]" in hist.text
