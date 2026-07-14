"""OpenAI-only runtime construction for production memory clients."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from mem01.client import MemoryClient
from mem01.embeddings.openai_embedder import OpenAIEmbedder
from mem01.env import load_env
from mem01.llm.openai_compat import OpenAICompatLLM
from mem01.store.postgres_store import PostgresBeliefStore

_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_LLM_MODEL = "gpt-5.6-sol"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_EMBEDDING_DIMENSIONS = 1536


@dataclass(frozen=True, slots=True)
class OpenAIRuntimeSettings:
    """Immutable configuration for the production OpenAI memory runtime."""

    api_key: str = field(repr=False)
    database_url: str = field(repr=False)
    llm_model: str = _DEFAULT_LLM_MODEL
    embedding_model: str = _DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = _DEFAULT_EMBEDDING_DIMENSIONS
    base_url: str = _OPENAI_BASE_URL

    def __post_init__(self) -> None:
        for field_name in (
            "api_key",
            "database_url",
            "llm_model",
            "embedding_model",
            "base_url",
        ):
            value = getattr(self, field_name).strip()
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI runtime")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required for the OpenAI runtime")
        if not self.llm_model:
            raise ValueError("llm_model must not be empty")
        if not self.embedding_model:
            raise ValueError("embedding_model must not be empty")
        if self.embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be >= 1")
        if not self.base_url:
            raise ValueError("base_url must not be empty")

    @classmethod
    def from_env(cls) -> OpenAIRuntimeSettings:
        """Load the required secrets and supported model overrides from the environment."""
        load_env()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        database_url = os.environ.get("DATABASE_URL", "").strip()
        llm_model = os.environ.get("MEM01_LLM_MODEL", _DEFAULT_LLM_MODEL).strip()
        embedding_model = os.environ.get(
            "MEM01_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL
        ).strip()
        raw_dimensions = os.environ.get(
            "MEM01_EMBEDDING_DIM", str(_DEFAULT_EMBEDDING_DIMENSIONS)
        ).strip()

        return cls(
            api_key=api_key,
            database_url=database_url,
            llm_model=llm_model or _DEFAULT_LLM_MODEL,
            embedding_model=embedding_model or _DEFAULT_EMBEDDING_MODEL,
            embedding_dimensions=(
                int(raw_dimensions) if raw_dimensions else _DEFAULT_EMBEDDING_DIMENSIONS
            ),
            base_url=_OPENAI_BASE_URL,
        )


def build_openai_memory_client(
    *, settings: OpenAIRuntimeSettings | None = None
) -> MemoryClient:
    """Construct the production Postgres/OpenAI memory client without fake fallbacks."""
    resolved = settings or OpenAIRuntimeSettings.from_env()
    store = PostgresBeliefStore(
        resolved.database_url,
        embedding_dim=resolved.embedding_dimensions,
    )
    embedder = OpenAIEmbedder(
        api_key=resolved.api_key,
        model=resolved.embedding_model,
        base_url=resolved.base_url,
        dimensions=resolved.embedding_dimensions,
    )
    llm = OpenAICompatLLM(
        api_key=resolved.api_key,
        model=resolved.llm_model,
        base_url=resolved.base_url,
    )
    return MemoryClient(store=store, embedder=embedder, llm=llm)
