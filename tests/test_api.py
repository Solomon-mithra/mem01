"""HTTP API smoke tests with in-memory store."""

from __future__ import annotations

import os

import pytest

# Force memory store before app imports create_belief_store
os.environ["MEM01_STORE"] = "memory"
os.environ.pop("DATABASE_URL", None)
os.environ.pop("MEM01_DATABASE_URL", None)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mem01.api.app import app


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
