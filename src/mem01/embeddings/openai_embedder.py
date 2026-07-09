"""OpenAI embeddings — mem0-style default embedder for real use.

Uses POST /v1/embeddings (stdlib urllib). Requires OPENAI_API_KEY.
Not free (usage-based); typically very cheap at small scale.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenAIEmbedder:
    """text-embedding-3-small by default (same class of model mem0 uses)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.model = model
        self.base_url = base_url.rstrip("/")
        # text-embedding-3-small default dim is 1536; optional API matryoshka trim
        self._dimensions = dimensions
        self._resolved_dim: int | None = dimensions

    @property
    def dimensions(self) -> int:
        if self._resolved_dim is not None:
            return self._resolved_dim
        # Known default until first call
        return 1536

    def embed(self, text: str) -> list[float]:
        if not self.api_key:
            raise RuntimeError("OpenAIEmbedder requires api_key or OPENAI_API_KEY")
        body: dict = {
            "model": self.model,
            "input": text,
        }
        if self._dimensions is not None:
            body["dimensions"] = self._dimensions

        url = f"{self.base_url}/embeddings"
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
            raise RuntimeError(f"OpenAI embeddings error {e.code}: {detail}") from e

        try:
            vec = list(payload["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected embeddings response: {payload!r}") from e

        self._resolved_dim = len(vec)
        return vec
