"""LLMClient protocol — provider-agnostic chat completion.

Why not hardcode OpenAI or Claude:
- Extractor only needs text out for JSON ops
- OpenAI-compatible, Anthropic Messages, LiteLLM, local models all fit
  behind the same complete() method
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> str:
        """Return assistant text (extractor will parse JSON from this)."""
        ...
