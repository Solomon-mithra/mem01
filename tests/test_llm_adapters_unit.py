"""Unit tests for optional LLM adapters (no network)."""

from __future__ import annotations

import pytest

from mem01.llm.anthropic import AnthropicLLM
from mem01.llm.openai_compat import OpenAICompatLLM


def test_openai_compat_requires_key():
    client = OpenAICompatLLM(api_key="")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        client.complete([])  # type: ignore[arg-type]


def test_anthropic_requires_key():
    client = AnthropicLLM(api_key="")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        from mem01.llm.base import ChatMessage

        client.complete([ChatMessage(role="user", content="hi")])
