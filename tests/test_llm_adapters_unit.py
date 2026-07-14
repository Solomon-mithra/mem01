"""Unit tests for optional LLM adapters (no network)."""

from __future__ import annotations

import json

import pytest

from mem01.llm.anthropic import AnthropicLLM
from mem01.llm.base import ChatMessage
from mem01.llm.openai_compat import OpenAICompatLLM


def test_openai_compat_requires_key():
    client = OpenAICompatLLM(api_key="")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        client.complete([])  # type: ignore[arg-type]


def test_openai_compat_defaults_to_build_week_model() -> None:
    client = OpenAICompatLLM(api_key="")

    assert client.model == "gpt-5.6-sol"


def test_anthropic_requires_key():
    client = AnthropicLLM(api_key="")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        client.complete([ChatMessage(role="user", content="hi")])


def test_openai_compat_omits_unsupported_temperature_for_gpt_5_6_sol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"[]"}}]}'

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatLLM(api_key="test-placeholder", model="gpt-5.6-sol")

    assert client.complete([ChatMessage(role="user", content="Remember NYC")]) == "[]"
    assert captured["body"] == {
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "Remember NYC"}],
    }
