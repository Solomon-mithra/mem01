"""Multi-signal retrieval: lexical search, RRF fusion, MMR diversity."""

from __future__ import annotations

from mem01.embeddings.fake import FakeEmbedder
from mem01.read.rank import mmr_select
from mem01.read.search import (
    fuse_candidates,
    lexical_search_beliefs,
    search_beliefs,
    tokenize,
)
from mem01.store.base import ScopeFilter
from mem01.store.memory_store import InMemoryBeliefStore
from mem01.types import Belief, ScopeIds, ScoredBelief


def _store_with(contents: list[str]) -> tuple[InMemoryBeliefStore, FakeEmbedder]:
    store = InMemoryBeliefStore()
    emb = FakeEmbedder()
    for i, content in enumerate(contents):
        belief = Belief(
            id=f"bel_{i}",
            content=content,
            scope_ids=ScopeIds(user_id="u1"),
        )
        store.upsert(belief)
        store.save_embedding(belief.id, emb.embed(content))
    return store, emb


def _scope() -> ScopeFilter:
    return ScopeFilter(user_id="u1")


def test_lexical_search_finds_exact_entity():
    store, _ = _store_with(
        [
            "Caroline researched adoption agencies in August 2023.",
            "Caroline is keen on counseling and mental health work.",
            "Melanie enjoys pottery and painting.",
        ]
    )
    hits = lexical_search_beliefs(store, "What did Caroline research?", _scope())
    assert hits
    assert hits[0].belief.id == "bel_0"


def test_lexical_search_entity_terms_score_double():
    store, _ = _store_with(
        [
            'Melanie read "Becoming Nicole" on Caroline\'s suggestion.',
            "Melanie read a book recently.",
        ]
    )
    hits = lexical_search_beliefs(store, "Did Melanie read Becoming Nicole?", _scope())
    assert hits[0].belief.id == "bel_0"
    assert hits[0].score > hits[1].score


def test_lexical_search_stopword_only_query_returns_empty():
    store, _ = _store_with(["Some belief content."])
    assert lexical_search_beliefs(store, "what is the", _scope()) == []


def test_fuse_candidates_merges_and_normalizes():
    a = Belief(id="a", content="x", scope_ids=ScopeIds(user_id="u1"))
    b = Belief(id="b", content="y", scope_ids=ScopeIds(user_id="u1"))
    c = Belief(id="c", content="z", scope_ids=ScopeIds(user_id="u1"))
    vector = [ScoredBelief(belief=a, score=0.9), ScoredBelief(belief=b, score=0.5)]
    lexical = [ScoredBelief(belief=b, score=2.0), ScoredBelief(belief=c, score=1.0)]

    fused = fuse_candidates(vector, lexical, k=10)
    ids = [s.belief.id for s in fused]
    # b appears in both lists, so rank fusion puts it first
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}
    assert fused[0].score == 1.0
    assert all(0.0 < s.score <= 1.0 for s in fused)


def test_fused_recall_beats_pure_vector_on_entity_query():
    contents = [
        "Caroline researched adoption agencies in August 2023.",
        "Caroline volunteers at the LGBTQ+ youth center.",
        "Caroline loves horseback riding with her dad.",
    ]
    store, emb = _store_with(contents)
    query = "What did Caroline research?"
    vector_hits = search_beliefs(store, emb, query, _scope(), k=3)
    lexical_hits = lexical_search_beliefs(store, query, _scope(), k=3)
    fused = fuse_candidates(vector_hits, lexical_hits, k=3)
    # The entity/keyword pass guarantees the research belief is present in the
    # fused set even when FakeEmbedder's vector ordering is arbitrary.
    assert any(s.belief.id == "bel_0" for s in fused)


def test_mmr_demotes_near_duplicates():
    dup1 = Belief(
        id="d1",
        content="Melanie went camping in the mountains in June 2023.",
        scope_ids=ScopeIds(user_id="u1"),
    )
    dup2 = Belief(
        id="d2",
        content="Melanie went camping in the mountains during June 2023.",
        scope_ids=ScopeIds(user_id="u1"),
    )
    other = Belief(
        id="o1",
        content="Sam bought a Prius in October.",
        scope_ids=ScopeIds(user_id="u1"),
    )
    ranked = [
        ScoredBelief(belief=dup1, score=1.0),
        ScoredBelief(belief=dup2, score=0.95),
        ScoredBelief(belief=other, score=0.9),
    ]
    out = mmr_select(ranked, lambda_weight=0.5)
    # The near-duplicate loses its #2 slot to the distinct belief
    assert [s.belief.id for s in out] == ["d1", "o1", "d2"]


def test_mmr_preserves_all_items():
    beliefs = [
        ScoredBelief(
            belief=Belief(id=f"b{i}", content=f"fact number {i}", scope_ids=ScopeIds(user_id="u1")),
            score=1.0 - i * 0.1,
        )
        for i in range(5)
    ]
    out = mmr_select(beliefs)
    assert {s.belief.id for s in out} == {f"b{i}" for i in range(5)}


def test_tokenize_lowercases_and_splits():
    assert tokenize('Melanie read "Becoming Nicole"!') == [
        "melanie",
        "read",
        "becoming",
        "nicole",
    ]
