"""Postgres + pgvector store — skipped unless DATABASE_URL is set."""

from __future__ import annotations

import os

import pytest

from mem01.env import load_env
from mem01.types import Belief, BeliefStatus, ScopeIds, utc_now

load_env()

DSN = (
    os.environ.get("MEM01_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
).strip()

pytestmark = pytest.mark.postgres

requires_pg = pytest.mark.skipif(
    not DSN or not (DSN.startswith("postgres://") or DSN.startswith("postgresql://")),
    reason="Set DATABASE_URL=postgresql://... with pgvector to run",
)


@pytest.fixture
def store():
    from mem01.store.postgres_store import PostgresBeliefStore
    import psycopg
    from psycopg.rows import dict_row

    # Reset schema so embedding_dim=4 is not blocked by a prior vector(N) column
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS embeddings CASCADE")
            cur.execute("DROP TABLE IF EXISTS beliefs CASCADE")
        conn.commit()

    s = PostgresBeliefStore(DSN, embedding_dim=4)
    yield s
    s.close()


@requires_pg
def test_upsert_get_roundtrip(store):
    b = Belief(
        content="User prefers TypeScript",
        scope_ids=ScopeIds(user_id="u1"),
    )
    store.upsert(b)
    got = store.get(b.id)
    assert got is not None
    assert got.content == b.content
    assert got.scope_ids.user_id == "u1"


@requires_pg
def test_similarity_search_order(store):
    close = Belief(content="lives in SF", scope_ids=ScopeIds(user_id="u1"))
    far = Belief(content="likes pizza", scope_ids=ScopeIds(user_id="u1"))
    store.upsert(close)
    store.upsert(far)
    store.save_embedding(close.id, [1.0, 0.0, 0.0, 0.0])
    store.save_embedding(far.id, [0.0, 1.0, 0.0, 0.0])
    results = store.similarity_search(
        [1.0, 0.0, 0.0, 0.0],
        ScopeIds(user_id="u1"),
        k=2,
    )
    assert len(results) >= 1
    assert results[0].belief.id == close.id
    assert results[0].score > results[1].score


@requires_pg
def test_scope_isolation(store):
    store.upsert(Belief(content="mine", scope_ids=ScopeIds(user_id="u1")))
    store.upsert(Belief(content="theirs", scope_ids=ScopeIds(user_id="u2")))
    listed = store.list_by_scope(ScopeIds(user_id="u1"))
    assert len(listed) == 1
    assert listed[0].content == "mine"


@requires_pg
def test_set_status_supersede(store):
    b = Belief(content="old", scope_ids=ScopeIds(user_id="u1"))
    store.upsert(b)
    store.set_status(b.id, BeliefStatus.SUPERSEDED, valid_to=utc_now())
    assert store.list_by_scope(ScopeIds(user_id="u1")) == []
    assert store.get(b.id).status == BeliefStatus.SUPERSEDED


@requires_pg
def test_create_belief_store_factory():
    from mem01.store.factory import create_belief_store
    from mem01.store.postgres_store import PostgresBeliefStore

    s = create_belief_store(DSN, embedding_dim=4)
    assert isinstance(s, PostgresBeliefStore)
    s.close()
