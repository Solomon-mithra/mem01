"""OpenAI runtime configuration and construction."""

from __future__ import annotations

import logging

import pytest

import mem01.runtime as runtime
from mem01.runtime import OpenAIRuntimeSettings, build_openai_memory_client


@pytest.fixture(autouse=True)
def isolated_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "load_env", lambda: [])
    for name in (
        "OPENAI_API_KEY",
        "DATABASE_URL",
        "OPENAI_BASE_URL",
        "MEM01_LLM_MODEL",
        "MEM01_EMBEDDING_MODEL",
        "MEM01_EMBEDDING_DIM",
    ):
        monkeypatch.delenv(name, raising=False)


def test_from_env_requires_openai_api_key_without_exposing_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = "postgresql://private-user:private-password@db.example/mem01"
    monkeypatch.setenv("DATABASE_URL", database_url)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY") as exc_info:
        OpenAIRuntimeSettings.from_env()

    assert database_url not in str(exc_info.value)


def test_from_env_requires_database_url_without_exposing_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "sk-private-runtime-test"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)

    with pytest.raises(RuntimeError, match="DATABASE_URL") as exc_info:
        OpenAIRuntimeSettings.from_env()

    assert api_key not in str(exc_info.value)


def test_from_env_uses_openai_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "  sk-test  ")
    monkeypatch.setenv("DATABASE_URL", "  postgresql://db/mem01  ")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")

    settings = OpenAIRuntimeSettings.from_env()

    assert settings.api_key == "sk-test"
    assert settings.database_url == "postgresql://db/mem01"
    assert settings.llm_model == "gpt-5.6-sol"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.base_url == "https://api.openai.com/v1"


def test_from_env_accepts_mem01_model_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DATABASE_URL", "postgresql://db/mem01")
    monkeypatch.setenv("MEM01_LLM_MODEL", "  gpt-custom  ")
    monkeypatch.setenv("MEM01_EMBEDDING_MODEL", "  embedding-custom  ")
    monkeypatch.setenv("MEM01_EMBEDDING_DIM", "1024")

    settings = OpenAIRuntimeSettings.from_env()

    assert settings.llm_model == "gpt-custom"
    assert settings.embedding_model == "embedding-custom"
    assert settings.embedding_dimensions == 1024


def test_settings_repr_redacts_credentials_but_keeps_non_secret_configuration() -> None:
    api_key = "sk-repr-must-not-leak"
    database_url = "postgresql://private-user:private-password@db.example/mem01"
    settings = OpenAIRuntimeSettings(
        api_key=api_key,
        database_url=database_url,
        llm_model="gpt-5.6-sol",
        embedding_model="text-embedding-3-small",
    )

    rendered = repr(settings)

    assert api_key not in rendered
    assert database_url not in rendered
    assert "private-user" not in rendered
    assert "private-password" not in rendered
    assert "gpt-5.6-sol" in rendered
    assert "text-embedding-3-small" in rendered


def test_explicit_settings_build_requested_openai_runtime_without_logging_secrets(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    constructed: dict[str, dict[str, object]] = {}

    class RecordingStore:
        def __init__(self, dsn: str, *, embedding_dim: int) -> None:
            constructed["store"] = {"dsn": dsn, "embedding_dim": embedding_dim}

    class RecordingEmbedder:
        def __init__(self, **kwargs: object) -> None:
            constructed["embedder"] = kwargs
            self.model = str(kwargs["model"])

    class RecordingLLM:
        def __init__(self, **kwargs: object) -> None:
            constructed["llm"] = kwargs
            self.model = str(kwargs["model"])

    monkeypatch.setattr(runtime, "PostgresBeliefStore", RecordingStore)
    monkeypatch.setattr(runtime, "OpenAIEmbedder", RecordingEmbedder)
    monkeypatch.setattr(runtime, "OpenAICompatLLM", RecordingLLM)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment")
    monkeypatch.setenv("DATABASE_URL", "postgresql://environment")
    settings = OpenAIRuntimeSettings(
        api_key="sk-explicit",
        database_url="postgresql://explicit",
        llm_model="gpt-5.6-sol",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1024,
        base_url="https://api.openai.com/v1",
    )

    with caplog.at_level(logging.DEBUG):
        client = build_openai_memory_client(settings=settings)

    assert client.llm.model == "gpt-5.6-sol"
    assert client.embedder.model == "text-embedding-3-small"
    assert constructed["store"] == {
        "dsn": "postgresql://explicit",
        "embedding_dim": 1024,
    }
    assert constructed["llm"]["api_key"] == "sk-explicit"
    assert constructed["embedder"]["api_key"] == "sk-explicit"
    assert "sk-explicit" not in caplog.text
    assert "postgresql://explicit" not in caplog.text
