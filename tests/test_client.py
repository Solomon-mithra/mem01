"""MemoryClient public API."""

from __future__ import annotations

import json

from mem01.client import MemoryClient
from mem01.embeddings.fake import FakeEmbedder
from mem01.llm.fake import FakeLLM
from mem01.store.memory_store import InMemoryBeliefStore
from mem01.types import BeliefStatus, ScopeIds


def _client(llm_payload) -> MemoryClient:
    if not isinstance(llm_payload, str):
        llm_payload = json.dumps(llm_payload)
    return MemoryClient(
        store=InMemoryBeliefStore(),
        embedder=FakeEmbedder(dimensions=16),
        llm=FakeLLM(llm_payload),
        default_user_id="u1",
    )


def test_remember_and_recall():
    client = _client(
        [
            {
                "op": "ADD",
                "content": "User prefers TypeScript",
                "confidence": 0.9,
                "topic_key": "language_pref",
            }
        ]
    )
    rem = client.remember(
        [{"role": "user", "content": "I prefer TypeScript"}],
        user_id="u1",
    )
    assert rem.llm_calls == 1
    assert rem.apply.ok
    assert len(rem.apply.created_ids) == 1

    packed = client.recall("language preference", user_id="u1")
    assert packed.latency_ms is not None
    assert packed.tokens_used >= 0
    # may or may not match depending on fake embed; at least one active in store
    active = client.store.list_by_scope(ScopeIds(user_id="u1"))
    assert len(active) == 1


def test_ny_to_sf_supersede_then_recall_only_sf():
    # First remember NY
    client = MemoryClient(
        store=InMemoryBeliefStore(),
        embedder=FakeEmbedder(dimensions=16),
        llm=FakeLLM(
            [
                json.dumps(
                    [
                        {
                            "op": "ADD",
                            "content": "User lives in New York",
                            "topic_key": "location",
                        }
                    ]
                ),
                json.dumps(
                    [
                        {
                            "op": "SUPERSEDE",
                            "target_id": "PLACEHOLDER",
                            "content": "User lives in San Francisco",
                            "topic_key": "location",
                        }
                    ]
                ),
            ]
        ),
        default_user_id="u1",
    )
    r1 = client.remember(
        [{"role": "user", "content": "I live in New York"}],
        user_id="u1",
    )
    old_id = r1.apply.created_ids[0]

    # Fix second scripted response with real id
    client.llm = FakeLLM(
        json.dumps(
            [
                {
                    "op": "SUPERSEDE",
                    "target_id": old_id,
                    "content": "User lives in San Francisco",
                    "topic_key": "location",
                }
            ]
        )
    )
    r2 = client.remember(
        [{"role": "user", "content": "I moved to San Francisco"}],
        user_id="u1",
    )
    assert r2.apply.ok
    assert old_id in r2.apply.superseded_ids

    active = client.store.list_by_scope(ScopeIds(user_id="u1"))
    assert len(active) == 1
    assert "San Francisco" in active[0].content
    assert client.get(old_id).status == BeliefStatus.SUPERSEDED

    packed = client.recall("where do I live?", user_id="u1", max_memory_tokens=200)
    # Active-only search: SF should be the only candidate content in store
    assert all("New York" not in b.content for b in packed.beliefs)


def test_correct_and_forget():
    client = _client(
        [{"op": "ADD", "content": "User likes coffee", "topic_key": "drink"}]
    )
    rem = client.remember(
        [{"role": "user", "content": "I like coffee"}],
        user_id="u1",
    )
    mid = rem.apply.created_ids[0]

    corr = client.correct(mid, "User likes tea")
    assert corr.ok
    assert mid in corr.superseded_ids
    active = client.store.list_by_scope(ScopeIds(user_id="u1"))
    assert len(active) == 1
    assert "tea" in active[0].content

    new_id = corr.created_ids[0]
    forgot = client.forget(new_id, reason="user request")
    assert new_id in forgot.invalidated_ids
    assert client.store.list_by_scope(ScopeIds(user_id="u1")) == []


def test_remember_reports_latency():
    client = _client([{"op": "ADD", "content": "x"}])
    rem = client.remember([{"role": "user", "content": "x"}], user_id="u1")
    assert rem.latency_ms >= 0
