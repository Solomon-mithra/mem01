"""Scripted LLM for tests — no API keys, no network."""

from __future__ import annotations

from mem01.llm.base import ChatMessage


class FakeLLM:
    """Returns a fixed response, or the next item from a queue of responses."""

    def __init__(self, response: str | list[str] = "[]") -> None:
        if isinstance(response, str):
            self._responses = [response]
        else:
            self._responses = list(response)
        if not self._responses:
            self._responses = ["[]"]
        self._index = 0
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(list(messages))
        # Last scripted response repeats if we run out (stable for retries)
        idx = min(self._index, len(self._responses) - 1)
        self._index += 1
        return self._responses[idx]

    @property
    def call_count(self) -> int:
        return len(self.calls)
