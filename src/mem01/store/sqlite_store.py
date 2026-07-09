"""SQLite BeliefStore — durable local persistence, zero infra.

Why SQLite for v1:
- File-backed dogfooding without Docker/Postgres
- Same BeliefStore protocol as InMemoryBeliefStore
- Embeddings stored as JSON arrays; cosine in Python is fine for small/medium N
  (swap to pgvector later without changing apply_ops / recall)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from mem01.store.memory_store import cosine_similarity
from mem01.types import (
    Belief,
    BeliefSource,
    BeliefStatus,
    ScopeIds,
    ScopeKind,
    ScoredBelief,
    utc_now,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    supersedes_id TEXT,
    scope TEXT NOT NULL,
    user_id TEXT,
    session_id TEXT,
    agent_id TEXT,
    project_id TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS embeddings (
    belief_id TEXT PRIMARY KEY REFERENCES beliefs(id) ON DELETE CASCADE,
    vector TEXT NOT NULL,
    dim INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_beliefs_user ON beliefs(user_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_project ON beliefs(project_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status);
"""


def _dt_to_str(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _str_to_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    # fromisoformat handles offsets; ensure Z → +00:00 if needed
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _row_to_belief(row: sqlite3.Row) -> Belief:
    return Belief(
        id=row["id"],
        content=row["content"],
        status=BeliefStatus(row["status"]),
        confidence=row["confidence"],
        valid_from=_str_to_dt(row["valid_from"]),
        valid_to=_str_to_dt(row["valid_to"]),
        supersedes_id=row["supersedes_id"],
        scope=ScopeKind(row["scope"]),
        scope_ids=ScopeIds(
            user_id=row["user_id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            project_id=row["project_id"],
        ),
        source=BeliefSource(row["source"]),
        created_at=_str_to_dt(row["created_at"]) or utc_now(),
        updated_at=_str_to_dt(row["updated_at"]) or utc_now(),
        metadata=json.loads(row["metadata"] or "{}"),
    )


class SqliteBeliefStore:
    """File or in-memory SQLite implementation of BeliefStore."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        # check_same_thread=False allows use from test runners / simple apps;
        # a production multi-threaded server should use a connection pool later.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _migrate(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteBeliefStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def get(self, belief_id: str) -> Belief | None:
        cur = self._conn.execute("SELECT * FROM beliefs WHERE id = ?", (belief_id,))
        row = cur.fetchone()
        return _row_to_belief(row) if row else None

    def upsert(self, belief: Belief) -> None:
        self._conn.execute(
            """
            INSERT INTO beliefs (
                id, content, status, confidence, valid_from, valid_to,
                supersedes_id, scope, user_id, session_id, agent_id, project_id,
                source, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content = excluded.content,
                status = excluded.status,
                confidence = excluded.confidence,
                valid_from = excluded.valid_from,
                valid_to = excluded.valid_to,
                supersedes_id = excluded.supersedes_id,
                scope = excluded.scope,
                user_id = excluded.user_id,
                session_id = excluded.session_id,
                agent_id = excluded.agent_id,
                project_id = excluded.project_id,
                source = excluded.source,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata
            """,
            (
                belief.id,
                belief.content,
                belief.status.value,
                belief.confidence,
                _dt_to_str(belief.valid_from),
                _dt_to_str(belief.valid_to),
                belief.supersedes_id,
                belief.scope.value,
                belief.scope_ids.user_id,
                belief.scope_ids.session_id,
                belief.scope_ids.agent_id,
                belief.scope_ids.project_id,
                belief.source.value,
                _dt_to_str(belief.created_at),
                _dt_to_str(belief.updated_at),
                json.dumps(belief.metadata),
            ),
        )
        self._conn.commit()

    def set_status(
        self,
        belief_id: str,
        status: BeliefStatus,
        *,
        valid_to: datetime | None = None,
    ) -> Belief | None:
        belief = self.get(belief_id)
        if belief is None:
            return None
        updates: dict[str, Any] = {
            "status": status,
            "updated_at": utc_now(),
        }
        if valid_to is not None:
            updates["valid_to"] = valid_to
        updated = belief.model_copy(update=updates)
        self.upsert(updated)
        return self.get(belief_id)

    def list_by_scope(
        self,
        scope_filter: ScopeIds,
        *,
        statuses: set[BeliefStatus] | None = None,
    ) -> list[Belief]:
        wanted = statuses if statuses is not None else {BeliefStatus.ACTIVE}
        # Pull candidates with SQL on set fields, then exact ScopeIds.matches
        # for consistent semantics with InMemoryBeliefStore.
        clauses: list[str] = []
        params: list[Any] = []

        status_values = [s.value for s in wanted]
        placeholders = ",".join("?" * len(status_values))
        clauses.append(f"status IN ({placeholders})")
        params.extend(status_values)

        for col, val in (
            ("user_id", scope_filter.user_id),
            ("session_id", scope_filter.session_id),
            ("agent_id", scope_filter.agent_id),
            ("project_id", scope_filter.project_id),
        ):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)

        sql = "SELECT * FROM beliefs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        cur = self._conn.execute(sql, params)
        out: list[Belief] = []
        for row in cur.fetchall():
            belief = _row_to_belief(row)
            if scope_filter.matches(belief.scope_ids):
                out.append(belief)
        return out

    def save_embedding(self, belief_id: str, vector: list[float]) -> None:
        if self.get(belief_id) is None:
            raise KeyError(f"cannot embed unknown belief_id={belief_id!r}")
        payload = json.dumps(vector)
        self._conn.execute(
            """
            INSERT INTO embeddings (belief_id, vector, dim)
            VALUES (?, ?, ?)
            ON CONFLICT(belief_id) DO UPDATE SET
                vector = excluded.vector,
                dim = excluded.dim
            """,
            (belief_id, payload, len(vector)),
        )
        self._conn.commit()

    def get_embedding(self, belief_id: str) -> list[float] | None:
        cur = self._conn.execute(
            "SELECT vector FROM embeddings WHERE belief_id = ?",
            (belief_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return list(json.loads(row["vector"]))

    def similarity_search(
        self,
        vector: list[float],
        scope_filter: ScopeIds,
        *,
        k: int = 20,
        statuses: set[BeliefStatus] | None = None,
    ) -> list[ScoredBelief]:
        if k <= 0:
            return []
        candidates = self.list_by_scope(scope_filter, statuses=statuses)
        scored: list[ScoredBelief] = []
        for belief in candidates:
            emb = self.get_embedding(belief.id)
            if emb is None:
                continue
            score = cosine_similarity(vector, emb)
            scored.append(ScoredBelief(belief=belief, score=score))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]
