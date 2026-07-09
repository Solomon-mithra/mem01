"""Optional Anthropic Claude client (native Messages API — not OpenAI-shaped).

Requires ANTHROPIC_API_KEY or api_key=. Uses stdlib urllib; `anthropic` package optional.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from mem01.llm.base import ChatMessage


class AnthropicLLM:
    """Claude via https://api.anthropic.com/v1/messages."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        base_url: str = "https://api.anthropic.com",
        max_tokens: int = 2048,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.anthropic_version = anthropic_version

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "AnthropicLLM requires api_key or ANTHROPIC_API_KEY."
            )

        system_parts: list[str] = []
        api_messages: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                # Anthropic only allows user/assistant in messages list
                role = m.role if m.role in ("user", "assistant") else "user"
                api_messages.append({"role": role, "content": m.content})

        if not api_messages:
            raise ValueError("AnthropicLLM.complete needs at least one non-system message")

        body: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        url = f"{self.base_url}/v1/messages"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API error {e.code}: {detail}") from e

        try:
            blocks = payload["content"]
            texts = [b["text"] for b in blocks if b.get("type") == "text"]
            if not texts:
                raise KeyError("no text blocks")
            return "\n".join(texts)
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"unexpected Anthropic response: {payload!r}") from e
