"""HTTP API smoke tests with in-memory store."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

# Force memory store before app imports create_belief_store
os.environ["MEM01_STORE"] = "memory"
os.environ.pop("DATABASE_URL", None)
os.environ.pop("MEM01_DATABASE_URL", None)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import mem01.api.app as app_module
import mem01.runtime as runtime_module
from mem01.api.app import app
from mem01.runtime import OpenAIRuntimeSettings
from mem01.types import Belief, BeliefSource, ScopeIds


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MEM01_STORE", "memory")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    expected_model = getattr(app.state.client.llm, "model", "fake")
    assert r.json()["extraction_model"] == expected_model


def test_remember_recall_roundtrip(client):
    # FakeLLM returns [] by default when no OPENAI key — still 200
    r = client.post(
        "/v1/remember",
        json={
            "user_id": "u1",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body

    r2 = client.post(
        "/v1/recall",
        json={"user_id": "u1", "query": "hello", "max_memory_tokens": 100},
    )
    assert r2.status_code == 200
    assert "text" in r2.json()


def test_history_endpoint(client):
    r = client.post(
        "/v1/history",
        json={"user_id": "u_hist", "limit": 10},
    )
    assert r.status_code == 200
    body = r.json()
    assert "beliefs" in body
    assert body["count"] == len(body["beliefs"])


def test_recall_accepts_include_history(client):
    r = client.post(
        "/v1/recall",
        json={
            "user_id": "u1",
            "query": "where before",
            "include_history": True,
            "max_memory_tokens": 100,
        },
    )
    assert r.status_code == 200
    assert r.json()["include_history"] is True


def test_recall_beliefs_include_source_and_timestamps(client):
    created_at = datetime(2026, 7, 13, 10, 30, tzinfo=UTC)
    updated_at = datetime(2026, 7, 15, 9, 45, tzinfo=UTC)
    belief = Belief(
        id="bel_provenance",
        content="User lives in San Francisco.",
        scope_ids=ScopeIds(user_id="u_provenance"),
        source=BeliefSource.EXTRACTION,
        created_at=created_at,
        updated_at=updated_at,
    )
    memory_client = app.state.client
    memory_client.store.upsert(belief)
    memory_client.store.save_embedding(
        belief.id, memory_client.embedder.embed(belief.content)
    )

    response = client.post(
        "/v1/recall",
        json={
            "user_id": "u_provenance",
            "query": "Where does the user live?",
            "max_memory_tokens": 100,
        },
    )

    assert response.status_code == 200
    recalled = response.json()["beliefs"][0]
    assert recalled["source"] == "extraction"
    assert recalled["created_at"] == created_at.isoformat()
    assert recalled["updated_at"] == updated_at.isoformat()


@pytest.mark.parametrize(
    ("model", "base_url", "expected_model", "expected_base_url"),
    [
        (None, None, "gpt-5.6-sol", "https://api.openai.com/v1"),
        (
            "gpt-5.6-sol",
            "https://gateway.example/v1",
            "gpt-5.6-sol",
            "https://api.openai.com/v1",
        ),
    ],
)
def test_build_client_configures_openai_compatible_extraction_model(
    monkeypatch,
    model,
    base_url,
    expected_model,
    expected_base_url,
):
    expected_client = object()
    captured = {}

    def fake_builder():
        settings = OpenAIRuntimeSettings.from_env()
        captured.update(model=settings.llm_model, base_url=settings.base_url)
        return expected_client

    monkeypatch.setattr(app_module, "load_env", lambda: None)
    monkeypatch.setattr(runtime_module, "load_env", lambda: [])
    monkeypatch.setattr(app_module, "build_openai_memory_client", fake_builder)
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    if model is None:
        monkeypatch.delenv("MEM01_LLM_MODEL", raising=False)
    else:
        monkeypatch.setenv("MEM01_LLM_MODEL", model)
    if base_url is None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    client = app_module._build_client()

    assert client is expected_client
    assert captured == {
        "model": expected_model,
        "base_url": expected_base_url,
    }
