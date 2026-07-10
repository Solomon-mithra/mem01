"""HTTP API for self-hosted mem01 (remember / recall / correct / forget).

Run via Docker Compose or:
  uvicorn mem01.api.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mem01.client import MemoryClient
from mem01.env import load_env
from mem01.store.factory import create_belief_store


class RememberBody(BaseModel):
    messages: list[dict[str, str]]
    user_id: str
    project_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None


class RecallBody(BaseModel):
    query: str
    user_id: str
    project_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    max_memory_tokens: int = Field(default=800, ge=0)
    k: int = Field(default=20, ge=1, le=200)
    # False = current truth only; True = include superseded/invalidated (labeled)
    include_history: bool = False


class HistoryBody(BaseModel):
    user_id: str
    project_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    include_invalidated: bool = True
    limit: int = Field(default=100, ge=1, le=500)


class CorrectBody(BaseModel):
    memory_id: str
    new_value: str
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)


class ForgetBody(BaseModel):
    memory_id: str
    reason: str | None = None


def _build_client() -> MemoryClient:
    load_env()
    store = create_belief_store()
    # Prefer real OpenAI when key present; else Fake for health-only deploys
    import os

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        from mem01.embeddings.openai_embedder import OpenAIEmbedder
        from mem01.llm.openai_compat import OpenAICompatLLM

        embedder = OpenAIEmbedder()
        llm = OpenAICompatLLM()
    else:
        from mem01.embeddings.fake import FakeEmbedder
        from mem01.llm.fake import FakeLLM

        dim = int(os.environ.get("MEM01_EMBEDDING_DIM", "1536"))
        embedder = FakeEmbedder(dimensions=dim)
        llm = FakeLLM("[]")
    return MemoryClient(store=store, embedder=embedder, llm=llm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = _build_client()
    yield
    close = getattr(app.state.client.store, "close", None)
    if callable(close):
        close()


app = FastAPI(
    title="mem01",
    description="Belief-based agent memory (self-hosted)",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mem01"}


@app.post("/v1/remember")
def remember(body: RememberBody) -> dict[str, Any]:
    client: MemoryClient = app.state.client
    try:
        result = client.remember(
            body.messages,
            user_id=body.user_id,
            project_id=body.project_id,
            session_id=body.session_id,
            agent_id=body.agent_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "ok": result.apply.ok,
        "created_ids": result.apply.created_ids,
        "updated_ids": result.apply.updated_ids,
        "superseded_ids": result.apply.superseded_ids,
        "invalidated_ids": result.apply.invalidated_ids,
        "errors": result.apply.errors,
        "llm_calls": result.llm_calls,
        "latency_ms": result.latency_ms,
    }


@app.post("/v1/recall")
def recall(body: RecallBody) -> dict[str, Any]:
    client: MemoryClient = app.state.client
    try:
        packed = client.recall(
            body.query,
            user_id=body.user_id,
            project_id=body.project_id,
            session_id=body.session_id,
            agent_id=body.agent_id,
            max_memory_tokens=body.max_memory_tokens,
            k=body.k,
            include_history=body.include_history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "text": packed.text,
        "tokens_used": packed.tokens_used,
        "max_memory_tokens": packed.max_memory_tokens,
        "candidate_count": packed.candidate_count,
        "latency_ms": packed.latency_ms,
        "include_history": body.include_history,
        "beliefs": [
            {
                "id": b.id,
                "content": b.content,
                "status": b.status.value,
                "supersedes_id": b.supersedes_id,
                "valid_from": b.valid_from.isoformat() if b.valid_from else None,
                "valid_to": b.valid_to.isoformat() if b.valid_to else None,
            }
            for b in packed.beliefs
        ],
    }


@app.post("/v1/history")
def history(body: HistoryBody) -> dict[str, Any]:
    """Full belief timeline for a scope — audit / “let me see previous facts”."""
    client: MemoryClient = app.state.client
    try:
        beliefs = client.history(
            user_id=body.user_id,
            project_id=body.project_id,
            session_id=body.session_id,
            agent_id=body.agent_id,
            include_invalidated=body.include_invalidated,
            limit=body.limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "count": len(beliefs),
        "beliefs": [
            {
                "id": b.id,
                "content": b.content,
                "status": b.status.value,
                "confidence": b.confidence,
                "supersedes_id": b.supersedes_id,
                "source": b.source.value,
                "valid_from": b.valid_from.isoformat() if b.valid_from else None,
                "valid_to": b.valid_to.isoformat() if b.valid_to else None,
                "created_at": b.created_at.isoformat(),
                "updated_at": b.updated_at.isoformat(),
            }
            for b in beliefs
        ],
    }


@app.post("/v1/correct")
def correct(body: CorrectBody) -> dict[str, Any]:
    client: MemoryClient = app.state.client
    try:
        result = client.correct(
            body.memory_id, body.new_value, confidence=body.confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "ok": result.ok,
        "created_ids": result.created_ids,
        "superseded_ids": result.superseded_ids,
        "errors": result.errors,
    }


@app.post("/v1/forget")
def forget(body: ForgetBody) -> dict[str, Any]:
    client: MemoryClient = app.state.client
    try:
        result = client.forget(body.memory_id, reason=body.reason)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "ok": result.ok,
        "invalidated_ids": result.invalidated_ids,
        "errors": result.errors,
    }
