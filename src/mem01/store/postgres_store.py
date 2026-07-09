"""Postgres + pgvector BeliefStore — production self-hosted backend.

Use when others run mem01 on their server (or Neon).
Same BeliefStore protocol as InMemory (tests) — swap via DATABASE_URL.

Requires:
  pip install 'mem01[postgres]'
  Postgres with the pgvector extension (Neon supports it)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from mem01.types import (
    Belief,
    BeliefSource,
    BeliefStatus,
    ScopeIds,
    ScopeKind,
    ScoredBelief,
    utc_now,
)


def _require_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row
        from pgvector.psycopg import register_vector
    except ImportError as e:
        raise ImportError(
            "Postgres store requires: pip install 'mem01[postgres]' "
            "(psycopg[binary] + pgvector)"
        ) from e
    return psycopg, dict_row, register_vector


def _row_to_belief(row: dict[str, Any]) -> Belief:
    meta = row.get("metadata")
    if isinstance(meta, str):
        meta = json.loads(meta or "{}")
    elif meta is None:
        meta = {}
    return Belief(
        id=row["id"],
        content=row["content"],
        status=BeliefStatus(row["status"]),
        confidence=float(row["confidence"]),
        valid_from=row.get("valid_from"),
        valid_to=row.get("valid_to"),
        supersedes_id=row.get("supersedes_id"),
        scope=ScopeKind(row["scope"]),
        scope_ids=ScopeIds(
            user_id=row.get("user_id"),
            session_id=row.get("session_id"),
            agent_id=row.get("agent_id"),
            project_id=row.get("project_id"),
        ),
        source=BeliefSource(row["source"]),
        created_at=row.get("created_at") or utc_now(),
        updated_at=row.get("updated_at") or utc_now(),
        metadata=dict(meta),
    )


class PostgresBeliefStore:
    """Postgres + pgvector implementation of BeliefStore.

    Args:
        dsn: e.g. postgresql://user:pass@localhost:5432/mem01
             Neon: postgresql://...@ep-xxx.neon.tech/neondb?sslmode=require
        embedding_dim: vector size (1536 = OpenAI text-embedding-3-small default)
    """

    def __init__(self, dsn: str, *, embedding_dim: int = 1536) -> None:
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        self.dsn = dsn
        self.embedding_dim = embedding_dim
        psycopg, dict_row, register_vector = _require_psycopg()
        self._psycopg = psycopg
        self._dict_row = dict_row
        self._register_vector = register_vector
        self._conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=False)
        # Extension must exist before register_vector looks up the type OID
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.commit()
        register_vector(self._conn)
        self._migrate()

    def _migrate(self) -> None:
        dim = self.embedding_dim
        with self._conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS beliefs (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    valid_from TIMESTAMPTZ,
                    valid_to TIMESTAMPTZ,
                    supersedes_id TEXT,
                    scope TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    agent_id TEXT,
                    project_id TEXT,
                    source TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            # Rebuild embeddings if vector size does not match this process
            cur.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod) AS typ
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE c.relname = 'embeddings'
                  AND n.nspname = 'public'
                  AND a.attname = 'embedding'
                  AND NOT a.attisdropped
                """
            )
            typ_row = cur.fetchone()
            expected = f"vector({dim})"
            if typ_row and typ_row.get("typ"):
                actual = str(typ_row["typ"]).replace(" ", "")
                if actual != expected:
                    cur.execute("DROP TABLE embeddings CASCADE")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    belief_id TEXT PRIMARY KEY
                        REFERENCES beliefs(id) ON DELETE CASCADE,
                    embedding vector({dim}) NOT NULL,
                    dim INTEGER NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_beliefs_user ON beliefs(user_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_beliefs_project ON beliefs(project_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status)"
            )
        self._conn.commit()

        # Optional ANN index (pgvector HNSW) — ignore if unsupported
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
                    ON embeddings
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PostgresBeliefStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def get(self, belief_id: str) -> Belief | None:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM beliefs WHERE id = %s", (belief_id,))
            row = cur.fetchone()
            return _row_to_belief(row) if row else None

    def upsert(self, belief: Belief) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO beliefs (
                    id, content, status, confidence, valid_from, valid_to,
                    supersedes_id, scope, user_id, session_id, agent_id, project_id,
                    source, created_at, updated_at, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s::jsonb
                )
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    status = EXCLUDED.status,
                    confidence = EXCLUDED.confidence,
                    valid_from = EXCLUDED.valid_from,
                    valid_to = EXCLUDED.valid_to,
                    supersedes_id = EXCLUDED.supersedes_id,
                    scope = EXCLUDED.scope,
                    user_id = EXCLUDED.user_id,
                    session_id = EXCLUDED.session_id,
                    agent_id = EXCLUDED.agent_id,
                    project_id = EXCLUDED.project_id,
                    source = EXCLUDED.source,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata
                """,
                (
                    belief.id,
                    belief.content,
                    belief.status.value,
                    belief.confidence,
                    belief.valid_from,
                    belief.valid_to,
                    belief.supersedes_id,
                    belief.scope.value,
                    belief.scope_ids.user_id,
                    belief.scope_ids.session_id,
                    belief.scope_ids.agent_id,
                    belief.scope_ids.project_id,
                    belief.source.value,
                    belief.created_at,
                    belief.updated_at,
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
        self.upsert(belief.model_copy(update=updates))
        return self.get(belief_id)

    def list_by_scope(
        self,
        scope_filter: ScopeIds,
        *,
        statuses: set[BeliefStatus] | None = None,
    ) -> list[Belief]:
        wanted = statuses if statuses is not None else {BeliefStatus.ACTIVE}
        status_values = [s.value for s in wanted]
        clauses = ["status = ANY(%s)"]
        params: list[Any] = [status_values]
        for col, val in (
            ("user_id", scope_filter.user_id),
            ("session_id", scope_filter.session_id),
            ("agent_id", scope_filter.agent_id),
            ("project_id", scope_filter.project_id),
        ):
            if val is not None:
                clauses.append(f"{col} = %s")
                params.append(val)
        sql = "SELECT * FROM beliefs WHERE " + " AND ".join(clauses)
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            out: list[Belief] = []
            for row in cur.fetchall():
                belief = _row_to_belief(row)
                if scope_filter.matches(belief.scope_ids):
                    out.append(belief)
            return out

    def save_embedding(self, belief_id: str, vector: list[float]) -> None:
        if self.get(belief_id) is None:
            raise KeyError(f"cannot embed unknown belief_id={belief_id!r}")
        if len(vector) != self.embedding_dim:
            raise ValueError(
                f"embedding dim {len(vector)} != store dim {self.embedding_dim}. "
                "Match embedder (text-embedding-3-small → 1536) or set embedding_dim=."
            )
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO embeddings (belief_id, embedding, dim)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (belief_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        dim = EXCLUDED.dim
                    """,
                    (belief_id, vector, len(vector)),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get_embedding(self, belief_id: str) -> list[float] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT embedding FROM embeddings WHERE belief_id = %s",
                (belief_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return list(row["embedding"])

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
        if len(vector) != self.embedding_dim:
            raise ValueError(
                f"query dim {len(vector)} != store dim {self.embedding_dim}"
            )
        wanted = statuses if statuses is not None else {BeliefStatus.ACTIVE}
        status_values = [s.value for s in wanted]
        clauses = ["b.status = ANY(%s)"]
        where_params: list[Any] = [status_values]
        for col, val in (
            ("b.user_id", scope_filter.user_id),
            ("b.session_id", scope_filter.session_id),
            ("b.agent_id", scope_filter.agent_id),
            ("b.project_id", scope_filter.project_id),
        ):
            if val is not None:
                clauses.append(f"{col} = %s")
                where_params.append(val)

        sql = f"""
            SELECT b.*,
                   (1 - (e.embedding <=> %s::vector)) AS score
            FROM beliefs b
            INNER JOIN embeddings e ON e.belief_id = b.id
            WHERE {" AND ".join(clauses)}
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        """
        params = [vector, *where_params, vector, k]
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            results: list[ScoredBelief] = []
            for row in cur.fetchall():
                score = float(row["score"])
                data = {k: v for k, v in row.items() if k != "score"}
                belief = _row_to_belief(data)
                if scope_filter.matches(belief.scope_ids):
                    results.append(ScoredBelief(belief=belief, score=score))
            return results
