"""Full hot-path recall: search → conflict → rank → pack (0 LLM).

This is the read SLA surface — every agent turn can call this.
"""

from __future__ import annotations

from mem01.embeddings.base import Embedder
from mem01.metrics import timer
from mem01.read.conflict import filter_conflicts
from mem01.read.pack import pack_beliefs
from mem01.read.rank import rank_candidates
from mem01.read.search import search_beliefs
from mem01.store.base import BeliefStore, ScopeFilter
from mem01.types import BeliefStatus, PackedMemory


def recall(
    store: BeliefStore,
    embedder: Embedder,
    query: str,
    scope_filter: ScopeFilter,
    *,
    max_memory_tokens: int = 800,
    k: int = 20,
    statuses: set[BeliefStatus] | None = None,
) -> PackedMemory:
    """Retrieve a budgeted, conflict-safe memory block for *query*.

    Never calls an LLM. Returns PackedMemory with tokens_used, latency_ms,
    and candidate_count for product gates.
    """
    with timer() as t:
        candidates = search_beliefs(
            store,
            embedder,
            query,
            scope_filter,
            k=k,
            statuses=statuses,
        )
        candidate_count = len(candidates)
        safe = filter_conflicts(candidates)
        ranked = rank_candidates(safe)
        packed = pack_beliefs(
            ranked,
            max_memory_tokens=max_memory_tokens,
            candidate_count=candidate_count,
        )

    return packed.model_copy(update={"latency_ms": t.latency_ms})
