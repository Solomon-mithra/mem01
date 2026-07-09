"""Product-defining scenarios (PRODUCT.md §9) — better-than-mem0 gates."""

from __future__ import annotations

import json

from mem01.client import MemoryClient
from mem01.embeddings.fake import FakeEmbedder
from mem01.llm.fake import FakeLLM
from mem01.store.memory_store import InMemoryBeliefStore
from mem01.types import BeliefStatus, ScopeIds


def _client_with_responses(*payloads: list | str) -> MemoryClient:
    responses = [
        p if isinstance(p, str) else json.dumps(p) for p in payloads
    ]
    return MemoryClient(
        store=InMemoryBeliefStore(),
        embedder=FakeEmbedder(dimensions=16),
        llm=FakeLLM(responses),
        default_user_id="u1",
    )


def test_preference_flip_dark_to_light():
    client = _client_with_responses(
        [{"op": "ADD", "content": "User prefers dark mode", "topic_key": "theme"}],
        [
            {
                "op": "SUPERSEDE",
                "target_id": "WILL_PATCH",
                "content": "User prefers light mode",
                "topic_key": "theme",
            }
        ],
    )
    r1 = client.remember([{"role": "user", "content": "dark mode"}], user_id="u1")
    old = r1.apply.created_ids[0]
    client.llm = FakeLLM(
        json.dumps(
            [
                {
                    "op": "SUPERSEDE",
                    "target_id": old,
                    "content": "User prefers light mode",
                    "topic_key": "theme",
                }
            ]
        )
    )
    client.remember([{"role": "user", "content": "actually light mode"}], user_id="u1")
    active = client.store.list_by_scope(ScopeIds(user_id="u1"))
    assert len(active) == 1
    assert "light" in active[0].content.lower()
    assert client.get(old).status == BeliefStatus.SUPERSEDED


def test_location_ny_to_sf():
    client = _client_with_responses(
        [{"op": "ADD", "content": "User lives in New York", "topic_key": "location"}]
    )
    r1 = client.remember([{"role": "user", "content": "NY"}], user_id="u1")
    old = r1.apply.created_ids[0]
    client.llm = FakeLLM(
        json.dumps(
            [
                {
                    "op": "SUPERSEDE",
                    "target_id": old,
                    "content": "User lives in San Francisco",
                    "topic_key": "location",
                }
            ]
        )
    )
    client.remember([{"role": "user", "content": "moved to SF"}], user_id="u1")
    active = client.store.list_by_scope(ScopeIds(user_id="u1"))
    assert len(active) == 1
    assert "San Francisco" in active[0].content
    packed = client.recall("where do I live", user_id="u1")
    assert all("New York" not in b.content for b in packed.beliefs)


def test_explicit_correct():
    client = _client_with_responses(
        [{"op": "ADD", "content": "User name is Bob"}]
    )
    r = client.remember([{"role": "user", "content": "I'm Bob"}], user_id="u1")
    mid = r.apply.created_ids[0]
    client.correct(mid, "User name is Alice")
    active = client.store.list_by_scope(ScopeIds(user_id="u1"))
    assert len(active) == 1
    assert "Alice" in active[0].content
    assert client.get(mid).status == BeliefStatus.SUPERSEDED


def test_forget_removes_from_active_recall():
    client = _client_with_responses(
        [{"op": "ADD", "content": "User SSN is secret-should-not-keep"}]
    )
    r = client.remember([{"role": "user", "content": "ssn"}], user_id="u1")
    mid = r.apply.created_ids[0]
    client.forget(mid, reason="user forgot")
    assert client.store.list_by_scope(ScopeIds(user_id="u1")) == []
    packed = client.recall("ssn secret", user_id="u1")
    assert packed.beliefs == []


def test_scope_isolation_projects():
    client = _client_with_responses(
        [{"op": "ADD", "content": "Project A uses pnpm"}],
        [{"op": "ADD", "content": "Project B uses npm"}],
    )
    client.remember(
        [{"role": "user", "content": "pnpm"}],
        user_id="u1",
        project_id="proj-a",
    )
    client.remember(
        [{"role": "user", "content": "npm"}],
        user_id="u1",
        project_id="proj-b",
    )
    a = client.store.list_by_scope(ScopeIds(user_id="u1", project_id="proj-a"))
    b = client.store.list_by_scope(ScopeIds(user_id="u1", project_id="proj-b"))
    assert len(a) == 1 and "pnpm" in a[0].content
    assert len(b) == 1 and "npm" in b[0].content


def test_token_budget_hard_cap():
    ops = [
        {
            "op": "ADD",
            "content": f"Durable fact entry {i} with additional wording for bulk",
        }
        for i in range(12)
    ]
    # one remember with many ADDs
    client = _client_with_responses(ops)
    client.remember([{"role": "user", "content": "many facts"}], user_id="u1")
    packed = client.recall("fact", user_id="u1", max_memory_tokens=100)
    assert packed.tokens_used <= 100


def test_expired_valid_to_excluded_from_recall():
    from datetime import timedelta

    from mem01.types import Belief, utc_now
    from mem01.write.apply_ops import apply_ops
    from mem01.types import BeliefOp, BeliefOpType

    store = InMemoryBeliefStore()
    emb = FakeEmbedder(dimensions=16)
    client = MemoryClient(
        store=store,
        embedder=emb,
        llm=FakeLLM("[]"),
        default_user_id="u1",
    )
    # Add via apply then expire
    res = apply_ops(
        store,
        [
            BeliefOp(
                op=BeliefOpType.ADD,
                content="Temporary visa fact",
                scope_ids=ScopeIds(user_id="u1"),
            )
        ],
        emb,
    )
    bid = res.created_ids[0]
    b = store.get(bid)
    assert b is not None
    expired = b.model_copy(update={"valid_to": utc_now() - timedelta(hours=1)})
    store.upsert(expired)

    packed = client.recall("visa", user_id="u1")
    assert all(x.id != bid for x in packed.beliefs)
