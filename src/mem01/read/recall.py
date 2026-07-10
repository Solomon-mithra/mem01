"""Full hot-path recall: search → conflict → rank → pack (0 LLM).

This is the read SLA surface — every agent turn can call this.
"""

from __future__ import annotations

from mem01.embeddings.base import Embedder
from mem01.metrics import timer
from mem01.read.conflict import HistoryMode, filter_conflicts
from mem01.read.pack import pack_beliefs
from mem01.read.rank import rank_candidates
from mem01.read.search import search_beliefs
from mem01.store.base import BeliefStore, ScopeFilter
from mem01.types import BeliefStatus, PackedMemory

_HISTORY_STATUSES = {
    BeliefStatus.ACTIVE,
    BeliefStatus.SUPERSEDED,
    BeliefStatus.INVALIDATED,
}


def recall(
    store: BeliefStore,
    embedder: Embedder,
    query: str,
    scope_filter: ScopeFilter,
    *,
    max_memory_tokens: int = 800,
    k: int = 20,
    statuses: set[BeliefStatus] | None = None,
    include_history: bool = False,
) -> PackedMemory:
    """Retrieve a budgeted, conflict-safe memory block for *query*.

    Never calls an LLM. Returns PackedMemory with tokens_used, latency_ms,
    and candidate_count for product gates.

    *include_history*: also search superseded/invalidated beliefs and label them
    in the packed text (for temporal questions / audit). Default False so the
    agent prompt stays single-truth for “what is true now?”.
    """
    mode: HistoryMode = "history" if include_history else "current"
    if statuses is None and include_history:
        statuses = set(_HISTORY_STATUSES)

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
        safe = filter_conflicts(candidates, mode=mode)
        ranked = rank_candidates(safe)
        packed = pack_beliefs(
            ranked,
            max_memory_tokens=max_memory_tokens,
            candidate_count=candidate_count,
            show_status=include_history,
        )

    return packed.model_copy(update={"latency_ms": t.latency_ms})
