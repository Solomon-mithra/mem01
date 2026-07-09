"""extract_ops with FakeLLM — no API keys required."""

from __future__ import annotations

import json

import pytest

from mem01.llm.fake import FakeLLM
from mem01.types import Belief, BeliefOpType, ScopeIds
from mem01.write.apply_ops import apply_ops
from mem01.write.extractor import extract_ops
from mem01.embeddings.fake import FakeEmbedder
from mem01.store.memory_store import InMemoryBeliefStore


def test_extract_add_ops_from_scripted_llm():
    payload = [
        {
            "op": "ADD",
            "content": "User prefers TypeScript",
            "confidence": 0.9,
            "topic_key": "language_pref",
        }
    ]
    llm = FakeLLM(json.dumps(payload))
    ops = extract_ops(
        [{"role": "user", "content": "I prefer TypeScript for all projects."}],
        llm=llm,
        scope_ids=ScopeIds(user_id="u1"),
    )
    assert len(ops) == 1
    assert ops[0].op == BeliefOpType.ADD
    assert ops[0].content == "User prefers TypeScript"
    assert ops[0].scope_ids.user_id == "u1"
    assert ops[0].topic_key == "language_pref"
    assert llm.call_count == 1


def test_extract_supersede_with_existing_belief_context():
    existing = Belief(
        id="bel_oldloc",
        content="User lives in New York",
        scope_ids=ScopeIds(user_id="u1"),
        metadata={"topic_key": "location"},
    )
    payload = [
        {
            "op": "SUPERSEDE",
            "target_id": "bel_oldloc",
            "content": "User lives in San Francisco",
            "topic_key": "location",
            "confidence": 0.95,
        }
    ]
    llm = FakeLLM(json.dumps(payload))
    ops = extract_ops(
        [{"role": "user", "content": "I moved to San Francisco last month."}],
        llm=llm,
        existing_beliefs=[existing],
        scope_ids=ScopeIds(user_id="u1"),
    )
    assert ops[0].op == BeliefOpType.SUPERSEDE
    assert ops[0].target_id == "bel_oldloc"

    # Full write pipeline: extract → apply
    store = InMemoryBeliefStore()
    store.upsert(existing)
    from mem01.types import BeliefStatus

    # need embedding for completeness though apply supersede creates new
    emb = FakeEmbedder()
    store.save_embedding(existing.id, emb.embed(existing.content))
    result = apply_ops(store, ops, emb)
    assert result.ok
    active = store.list_by_scope(ScopeIds(user_id="u1"))
    assert len(active) == 1
    assert "San Francisco" in active[0].content
    assert store.get("bel_oldloc").status == BeliefStatus.SUPERSEDED


def test_extract_handles_markdown_fence():
    inner = [{"op": "ADD", "content": "User name is Alex"}]
    llm = FakeLLM("```json\n" + json.dumps(inner) + "\n```")
    ops = extract_ops(
        [{"role": "user", "content": "My name is Alex"}],
        llm=llm,
        scope_ids=ScopeIds(user_id="u1"),
    )
    assert len(ops) == 1
    assert ops[0].content == "User name is Alex"


def test_extract_empty_array():
    llm = FakeLLM("[]")
    ops = extract_ops(
        [{"role": "user", "content": "hey"}],
        llm=llm,
        scope_ids=ScopeIds(user_id="u1"),
    )
    assert ops == []


def test_extract_drops_invalid_ops_keeps_valid():
    payload = [
        {"op": "ADD", "content": "good fact"},
        {"op": "SUPERSEDE"},  # missing fields — invalid
        {"op": "ADD", "content": "another good"},
    ]
    llm = FakeLLM(json.dumps(payload))
    ops = extract_ops(
        [{"role": "user", "content": "stuff"}],
        llm=llm,
        scope_ids=ScopeIds(user_id="u1"),
    )
    assert len(ops) == 2
    assert all(o.op == BeliefOpType.ADD for o in ops)


def test_extract_requires_conversation():
    llm = FakeLLM("[]")
    with pytest.raises(ValueError, match="at least one"):
        extract_ops([], llm=llm)


def test_fake_llm_records_messages():
    llm = FakeLLM("[]")
    extract_ops(
        [{"role": "user", "content": "hello memory"}],
        llm=llm,
        scope_ids=ScopeIds(user_id="u1"),
    )
    assert llm.call_count == 1
    roles = [m.role for m in llm.calls[0]]
    assert roles[0] == "system"
    assert any(m.content == "hello memory" for m in llm.calls[0])
