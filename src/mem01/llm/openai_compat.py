"""Optional OpenAI-compatible chat client (OpenAI, many proxies, some hosts).

Not required for Claude — use anthropic.py or a router (e.g. LiteLLM) instead.
Only imported when you construct OpenAICompatLLM; core mem01 does not need openai.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from mem01.llm.base import ChatMessage


class OpenAICompatLLM:
    """Minimal HTTP client for POST /v1/chat/completions.

    Uses stdlib urllib so `openai` package is optional.
    Set base_url to any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-5.6-sol",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        if api_key is None:
            try:
                from mem01.env import load_env

                load_env()
            except Exception:
                pass
            self.api_key = os.environ.get("OPENAI_API_KEY") or ""
        else:
            # Explicit "" means "no key" (tests); do not fall back to env
            self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "OpenAICompatLLM requires api_key or OPENAI_API_KEY. "
                "For Claude use AnthropicLLM (or another LLMClient adapter)."
            )
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if self.model != "gpt-5.6-sol":
            body["temperature"] = temperature
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible API error {e.code}: {detail}") from e

        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected OpenAI-compatible response: {payload!r}") from e
