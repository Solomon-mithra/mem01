"""Postgres + pgvector store — skipped unless DATABASE_URL is set."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any

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


class RecordingCursor:
    def __init__(self, connection: RecordingConnection) -> None:
        self.connection = connection

    def __enter__(self) -> RecordingCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, query: str, params: object = None) -> None:
        self.connection.queries.append((query, params))

    def fetchone(self) -> dict[str, Any] | None:
        if not self.connection.fetchone_results:
            return None
        return self.connection.fetchone_results.pop(0)

    def fetchall(self) -> list[dict[str, Any]]:
        return []

    @property
    def rowcount(self) -> int:
        return self.connection.rowcount


class RecordingConnection:
    def __init__(
        self,
        *,
        fetchone_results: list[dict[str, Any] | None] | None = None,
        rowcount: int = 0,
    ) -> None:
        self.queries: list[tuple[str, object]] = []
        self.fetchone_results = list(fetchone_results or [])
        self.rowcount = rowcount
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> RecordingCursor:
        return RecordingCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class RecordingConnectionContext(AbstractContextManager[RecordingConnection]):
    def __init__(self, connection: RecordingConnection) -> None:
        self.connection = connection

    def __enter__(self) -> RecordingConnection:
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        return None


class RecordingPool:
    instances: list[RecordingPool] = []

    def __init__(
        self,
        conninfo: str,
        *,
        kwargs: dict[str, object],
        configure,
        open: bool,
    ) -> None:
        self.conninfo = conninfo
        self.kwargs = kwargs
        self.connection_calls = 0
        self.close_calls = 0
        self.connections: list[RecordingConnection] = []
        self._next_fetchone_results: list[list[dict[str, Any] | None]] = []
        self._next_rowcounts: list[int] = []
        self.configured_connection = RecordingConnection()
        configure(self.configured_connection)
        self.open = open
        self.instances.append(self)

    def queue_fetchone_results(self, *results: dict[str, Any] | None) -> None:
        self._next_fetchone_results.append(list(results))

    def queue_rowcount(self, rowcount: int) -> None:
        self._next_rowcounts.append(rowcount)

    def connection(self) -> RecordingConnectionContext:
        self.connection_calls += 1
        results = self._next_fetchone_results.pop(0) if self._next_fetchone_results else []
        rowcount = self._next_rowcounts.pop(0) if self._next_rowcounts else 0
        connection = RecordingConnection(fetchone_results=results, rowcount=rowcount)
        self.connections.append(connection)
        return RecordingConnectionContext(connection)

    def close(self) -> None:
        self.close_calls += 1


def test_store_reuses_one_pool_and_acquires_connections_for_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mem01.store.postgres_store as postgres_store

    registered: list[RecordingConnection] = []
    row_factory = object()
    RecordingPool.instances.clear()
    monkeypatch.setattr(
        postgres_store,
        "_require_psycopg",
        lambda: (RecordingPool, row_factory, registered.append),
    )

    store = postgres_store.PostgresBeliefStore("postgresql://test", embedding_dim=4)
    pool = RecordingPool.instances[0]
    calls_after_migration = pool.connection_calls

    assert len(RecordingPool.instances) == 1
    assert pool.conninfo == "postgresql://test"
    assert pool.kwargs["row_factory"] is row_factory
    assert pool.kwargs["autocommit"] is False
    assert registered == [pool.configured_connection]

    assert store.get("missing") is None
    assert store.list_by_scope(ScopeIds(user_id="u1")) == []
    assert pool.connection_calls == calls_after_migration + 2
    assert pool.connections[-2] is not pool.connections[-1]
    assert pool.connections[-2].commits == 1
    assert pool.connections[-1].commits == 1


def test_store_rolls_back_failed_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    import mem01.store.postgres_store as postgres_store

    RecordingPool.instances.clear()
    monkeypatch.setattr(
        postgres_store,
        "_require_psycopg",
        lambda: (RecordingPool, object(), lambda connection: None),
    )
    store = postgres_store.PostgresBeliefStore("postgresql://test", embedding_dim=4)
    pool = RecordingPool.instances[0]

    with pytest.raises(KeyError, match="missing"):
        store.save_embedding("missing", [1.0, 0.0, 0.0, 0.0])

    failed_connection = pool.connections[-1]
    assert failed_connection.commits == 0
    assert failed_connection.rollbacks == 1


def test_store_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    import mem01.store.postgres_store as postgres_store

    RecordingPool.instances.clear()
    monkeypatch.setattr(
        postgres_store,
        "_require_psycopg",
        lambda: (RecordingPool, object(), lambda connection: None),
    )
    store = postgres_store.PostgresBeliefStore("postgresql://test", embedding_dim=4)
    pool = RecordingPool.instances[0]

    store.close()
    store.close()

    assert pool.close_calls == 1


def _belief_row(
    *,
    status: str = "superseded",
    valid_to: datetime | None = None,
    updated_at: datetime,
) -> dict[str, Any]:
    created_at = datetime(2026, 7, 1, tzinfo=UTC)
    return {
        "id": "bel-status",
        "content": "Unrelated content must survive",
        "status": status,
        "confidence": 0.73,
        "valid_from": created_at,
        "valid_to": valid_to,
        "supersedes_id": "bel-older",
        "scope": "user",
        "user_id": "u1",
        "session_id": "s1",
        "agent_id": "a1",
        "project_id": "p1",
        "source": "extraction",
        "created_at": created_at,
        "updated_at": updated_at,
        "metadata": {"unrelated": "preserved"},
    }


def _store_with_recording_pool():
    import mem01.store.postgres_store as postgres_store

    pool = RecordingPool(
        "postgresql://test",
        kwargs={"row_factory": object(), "autocommit": False},
        configure=lambda connection: None,
        open=True,
    )
    store = object.__new__(postgres_store.PostgresBeliefStore)
    store._pool = pool
    store.embedding_dim = 4
    return postgres_store, store, pool


def test_set_status_updates_only_status_and_updated_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres_store, store, pool = _store_with_recording_pool()
    updated_at = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    pool.queue_fetchone_results(_belief_row(updated_at=updated_at))
    monkeypatch.setattr(postgres_store, "utc_now", lambda: updated_at)

    result = store.set_status("bel-status", BeliefStatus.SUPERSEDED)

    connection = pool.connections[-1]
    assert len(connection.queries) == 1
    query, params = connection.queries[0]
    assert " ".join(query.split()) == (
        "UPDATE beliefs SET status = %s, updated_at = %s "
        "WHERE id = %s RETURNING *"
    )
    assert params == ("superseded", updated_at, "bel-status")
    assert result is not None
    assert result.content == "Unrelated content must survive"
    assert result.metadata == {"unrelated": "preserved"}
    assert result.status == BeliefStatus.SUPERSEDED


def test_set_status_updates_valid_to_only_when_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres_store, store, pool = _store_with_recording_pool()
    updated_at = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    valid_to = datetime(2026, 7, 13, 11, 59, tzinfo=UTC)
    pool.queue_fetchone_results(
        _belief_row(updated_at=updated_at, valid_to=valid_to)
    )
    monkeypatch.setattr(postgres_store, "utc_now", lambda: updated_at)

    result = store.set_status(
        "bel-status",
        BeliefStatus.SUPERSEDED,
        valid_to=valid_to,
    )

    query, params = pool.connections[-1].queries[0]
    assert " ".join(query.split()) == (
        "UPDATE beliefs SET status = %s, updated_at = %s, valid_to = %s "
        "WHERE id = %s RETURNING *"
    )
    assert params == ("superseded", updated_at, valid_to, "bel-status")
    assert result is not None
    assert result.valid_to == valid_to


def test_set_status_returns_none_when_belief_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres_store, store, pool = _store_with_recording_pool()
    updated_at = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)
    pool.queue_fetchone_results(None)
    monkeypatch.setattr(postgres_store, "utc_now", lambda: updated_at)

    result = store.set_status("missing", BeliefStatus.INVALIDATED)

    query, params = pool.connections[-1].queries[0]
    assert " ".join(query.split()) == (
        "UPDATE beliefs SET status = %s, updated_at = %s "
        "WHERE id = %s RETURNING *"
    )
    assert params == ("invalidated", updated_at, "missing")
    assert result is None


def test_delete_by_user_uses_scoped_delete_and_returns_rowcount() -> None:
    _, store, pool = _store_with_recording_pool()
    pool.queue_rowcount(3)

    deleted = store.delete_by_user("u1")

    connection = pool.connections[-1]
    assert len(connection.queries) == 1
    query, params = connection.queries[0]
    assert " ".join(query.split()) == "DELETE FROM beliefs WHERE user_id = %s"
    assert params == ("u1",)
    assert deleted == 3


@pytest.mark.parametrize("user_id", ["", "   "])
def test_delete_by_user_rejects_blank_user_id(user_id: str) -> None:
    _, store, pool = _store_with_recording_pool()
    connection_count = len(pool.connections)

    with pytest.raises(ValueError, match="user_id"):
        store.delete_by_user(user_id)

    assert len(pool.connections) == connection_count


def test_get_embedding_converts_pgvector_driver_value() -> None:
    class DriverVector:
        def to_list(self) -> list[float]:
            return [0.25, 0.75]

    _, store, pool = _store_with_recording_pool()
    pool.queue_fetchone_results({"embedding": DriverVector()})

    assert store.get_embedding("belief-1") == [0.25, 0.75]


def test_conditional_supersede_locks_owned_active_target_before_inserts() -> None:
    _, store, pool = _store_with_recording_pool()
    old_row = _belief_row(
        status="active",
        updated_at=datetime(2026, 7, 13, 12, 0, tzinfo=UTC),
    )
    pool.queue_fetchone_results(old_row)
    replacement = Belief(
        id="bel-replacement",
        content="new value",
        scope_ids=ScopeIds(user_id="u1"),
    )

    changed = store.supersede_if_owned(
        "bel-status",
        replacement,
        [1.0, 0.0, 0.0, 0.0],
        expected_user_id="u1",
        superseded_at=datetime(2026, 7, 13, 12, 1, tzinfo=UTC),
    )

    assert changed is True
    connection = pool.connections[-1]
    normalized = [" ".join(query.split()) for query, _ in connection.queries]
    assert len(pool.connections) == 1
    assert "FOR UPDATE" in normalized[0]
    assert "user_id = %s" in normalized[0]
    assert "status = %s" in normalized[0]
    assert normalized[1].startswith("INSERT INTO beliefs")
    assert normalized[2].startswith("INSERT INTO embeddings")
    assert normalized[3].startswith("UPDATE beliefs SET status")


def test_conditional_supersede_wrong_user_writes_nothing() -> None:
    _, store, pool = _store_with_recording_pool()
    pool.queue_fetchone_results(None)
    replacement = Belief(
        id="bel-replacement",
        content="new value",
        scope_ids=ScopeIds(user_id="attacker"),
    )

    changed = store.supersede_if_owned(
        "bel-victim",
        replacement,
        [1.0, 0.0, 0.0, 0.0],
        expected_user_id="attacker",
        superseded_at=datetime(2026, 7, 13, 12, 1, tzinfo=UTC),
    )

    assert changed is False
    assert len(pool.connections[-1].queries) == 1


def test_conditional_invalidate_uses_one_owned_active_update() -> None:
    _, store, pool = _store_with_recording_pool()
    pool.queue_fetchone_results({"id": "bel-status"})

    changed = store.invalidate_if_owned(
        "bel-status",
        expected_user_id="u1",
        reason="user request",
        invalidated_at=datetime(2026, 7, 13, 12, 1, tzinfo=UTC),
    )

    assert changed is True
    connection = pool.connections[-1]
    assert len(connection.queries) == 1
    query, params = connection.queries[0]
    normalized = " ".join(query.split())
    assert "WHERE id = %s AND user_id = %s AND status = %s" in normalized
    assert "RETURNING id" in normalized
    assert params[-3:] == ("bel-status", "u1", "active")


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
