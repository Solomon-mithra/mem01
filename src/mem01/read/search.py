"""Semantic search over beliefs — first stage of the hot read path.

Why this module:
- recall needs candidates before conflict filter and token packing
- Embed query once, then store.similarity_search (vector + scope + status)
- No LLM: keeps latency and $ off every agent turn (mem0-class constraint)
"""

from __future__ import annotations

from mem01.embeddings.base import Embedder
from mem01.store.base import BeliefStore, ScopeFilter
from mem01.types import BeliefStatus, ScoredBelief


def search_beliefs(
    store: BeliefStore,
    embedder: Embedder,
    query: str,
    scope_filter: ScopeFilter,
    *,
    k: int = 20,
    statuses: set[BeliefStatus] | None = None,
) -> list[ScoredBelief]:
    """Embed *query* and return top-k scored beliefs in scope.

    Default statuses = active only (store default). Pass a larger k than the
    final pack size so later stages have room to drop conflicts.
    """
    if not query or not query.strip():
        return []
    if k <= 0:
        return []

    vector = embedder.embed(query.strip())
    return store.similarity_search(
        vector,
        scope_filter,
        k=k,
        statuses=statuses,
    )
