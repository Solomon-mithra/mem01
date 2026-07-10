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


# ── Multi-signal retrieval ──────────────────────────────────────────────────
# Vector similarity misses exact names, titles, and numbers when the query
# phrasing doesn't embed near the belief ("what did she research?" vs a belief
# about adoption agencies). A cheap lexical pass over in-scope beliefs plus
# rank fusion recovers those without any LLM or extra network calls.

import re as _re

_WORD_RE = _re.compile(r"[A-Za-z0-9']+")

_STOPWORDS = frozenset(
    "a an the and or but if then is are was were be been being do does did "
    "doing have has had having i you he she it we they them his her their my "
    "your our what when where who whom which why how to of in on for with at "
    "by from as that this these those not no nor so too very can will just "
    "about into over after before between during does user".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens; shared by lexical search and MMR."""
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def lexical_search_beliefs(
    store: BeliefStore,
    query: str,
    scope_filter: ScopeFilter,
    *,
    k: int = 20,
    statuses: set[BeliefStatus] | None = None,
) -> list[ScoredBelief]:
    """Keyword + entity scoring over in-scope beliefs. No LLM, no embeddings.

    Entity terms (capitalized words in the query, e.g. names and titles)
    count double: an exact name match is a stronger signal than a common word.
    """
    raw_terms = _WORD_RE.findall(query or "")
    if not raw_terms:
        return []
    entity_terms = {t.lower() for t in raw_terms if t[:1].isupper()}
    terms = {t.lower() for t in raw_terms} - _STOPWORDS
    if not terms:
        return []

    scored: list[ScoredBelief] = []
    for belief in store.list_by_scope(scope_filter, statuses=statuses):
        belief_tokens = set(tokenize(belief.content))
        hits = terms & belief_tokens
        if not hits:
            continue
        weight = sum(2.0 if t in entity_terms else 1.0 for t in hits)
        scored.append(ScoredBelief(belief=belief, score=weight / (len(terms) + 1)))
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:k]


def fuse_candidates(
    vector_hits: list[ScoredBelief],
    lexical_hits: list[ScoredBelief],
    *,
    k: int = 20,
    rrf_k: int = 60,
) -> list[ScoredBelief]:
    """Reciprocal-rank fusion of the two result lists, normalized to [0, 1].

    RRF is rank-based, so the incompatible score scales of cosine similarity
    and lexical overlap never need calibrating against each other. Scores are
    normalized so downstream ranking still sees a similarity-like [0, 1].
    """
    fused: dict[str, list] = {}
    for hits in (vector_hits, lexical_hits):
        for rank, sb in enumerate(hits):
            entry = fused.setdefault(sb.belief.id, [sb.belief, 0.0])
            entry[1] += 1.0 / (rrf_k + rank + 1)
    if not fused:
        return []
    top = max(score for _, score in fused.values())
    out = [
        ScoredBelief(belief=belief, score=score / top)
        for belief, score in fused.values()
    ]
    out.sort(key=lambda s: s.score, reverse=True)
    return out[:k]
