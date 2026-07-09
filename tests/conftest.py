"""Shared pytest fixtures.

We keep fakes (store, embedder, LLM) here as the core grows so every test
can run without API keys or network.
"""

from __future__ import annotations

# Fixtures will land here in later tasks (InMemoryBeliefStore, FakeEmbedder, …).
