"""MemoryClient — public facade (mem0-class four verbs).

remember / recall / correct / forget wire write + read pipelines.
Users and later HTTP/MCP should depend on this, not internal modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from mem01.embeddings.base import Embedder
from mem01.llm.base import LLMClient
from mem01.metrics import timer
from mem01.read.recall import recall as recall_pipeline
from mem01.store.base import BeliefStore
from mem01.types import (
    Belief,
    BeliefOp,
    BeliefOpType,
    BeliefSource,
    BeliefStatus,
    PackedMemory,
    ScopeIds,
    ScopeKind,
)
from mem01.write.apply_ops import ApplyResult, apply_ops
from mem01.write.extractor import extract_ops


@dataclass
class RememberResult:
    """Outcome of remember()."""

    apply: ApplyResult
    ops: list[BeliefOp]
    llm_calls: int
    latency_ms: float
    existing_considered: int = 0


@dataclass
class MemoryClient:
    """High-level memory API.

    Defaults match product direction: inject real OpenAI (or other) LLM/embedder
    for real use; tests can pass scripted/fake implementations.
    """

    store: BeliefStore
    embedder: Embedder
    llm: LLMClient
    default_user_id: str | None = None
    # How many existing actives to show the extractor for SUPERSEDE decisions
    existing_for_extract_k: int = 15

    def _scope_ids(
        self,
        user_id: str | None = None,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> ScopeIds:
        uid = user_id if user_id is not None else self.default_user_id
        return ScopeIds(
            user_id=uid,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
        )

    def remember(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        scope: ScopeKind = ScopeKind.USER,
    ) -> RememberResult:
        """Extract belief ops from messages (1 LLM call) and apply them."""
        scope_ids = self._scope_ids(
            user_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
        )
        with timer() as t:
            existing = self._existing_snippets(scope_ids)
            ops = extract_ops(
                messages,
                llm=self.llm,
                existing_beliefs=existing,
                scope=scope,
                scope_ids=scope_ids,
            )
            apply_result = apply_ops(
                self.store,
                ops,
                self.embedder,
                default_source=BeliefSource.EXTRACTION,
                expected_user_id=scope_ids.user_id,
            )
        return RememberResult(
            apply=apply_result,
            ops=ops,
            llm_calls=1,
            latency_ms=t.latency_ms,
            existing_considered=len(existing),
        )

    def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        max_memory_tokens: int = 800,
        k: int = 20,
        include_history: bool = False,
    ) -> PackedMemory:
        """Hot-path retrieve — 0 LLM calls.

        Default: active beliefs only (current truth).
        include_history=True: also return superseded/invalidated, labeled in text
        (use for “before SF?” / medical audit style questions).
        """
        scope_ids = self._scope_ids(
            user_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
        )
        return recall_pipeline(
            self.store,
            self.embedder,
            query,
            scope_ids,
            max_memory_tokens=max_memory_tokens,
            k=k,
            include_history=include_history,
        )

    def history(
        self,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        include_invalidated: bool = True,
        limit: int = 100,
    ) -> list[Belief]:
        """Chronological belief timeline for a scope (audit / “let me see”).

        Includes active + superseded (+ invalidated by default). Newest first.
        Does not call an LLM or run vector search.
        """
        scope_ids = self._scope_ids(
            user_id,
            project_id=project_id,
            session_id=session_id,
            agent_id=agent_id,
        )
        statuses = {
            BeliefStatus.ACTIVE,
            BeliefStatus.SUPERSEDED,
        }
        if include_invalidated:
            statuses.add(BeliefStatus.INVALIDATED)
        beliefs = self.store.list_by_scope(scope_ids, statuses=statuses)
        beliefs.sort(key=lambda b: b.updated_at, reverse=True)
        return beliefs[: max(1, limit)]

    def correct(
        self,
        memory_id: str,
        new_value: str,
        *,
        user_id: str | None = None,
        confidence: float = 0.95,
    ) -> ApplyResult:
        """Human/agent fix: SUPERSEDE by id (no LLM)."""
        op = BeliefOp(
            op=BeliefOpType.SUPERSEDE,
            target_id=memory_id,
            content=new_value,
            confidence=confidence,
        )
        return apply_ops(
            self.store,
            [op],
            self.embedder,
            default_source=BeliefSource.CORRECTION,
            expected_user_id=self._scope_ids(user_id).user_id,
        )

    def forget(
        self,
        memory_id: str,
        *,
        user_id: str | None = None,
        reason: str | None = None,
    ) -> ApplyResult:
        """Invalidate a belief by id (no LLM)."""
        op = BeliefOp(
            op=BeliefOpType.INVALIDATE,
            target_id=memory_id,
            reason=reason,
        )
        return apply_ops(
            self.store,
            [op],
            self.embedder,
            default_source=BeliefSource.FORGET,
            expected_user_id=self._scope_ids(user_id).user_id,
        )

    def clear_user(self, *, user_id: str | None = None) -> int:
        """Hard-delete all stored beliefs for one user."""
        resolved_user_id = user_id if user_id is not None else self.default_user_id
        if resolved_user_id is None or not resolved_user_id.strip():
            raise ValueError("user_id must be non-empty")
        return self.store.delete_by_user(resolved_user_id)

    def get(self, memory_id: str) -> Belief | None:
        return self.store.get(memory_id)

    def _existing_snippets(self, scope_ids: ScopeIds) -> list[Belief]:
        """Active beliefs for extractor context (SUPERSEDE targets)."""
        active = self.store.list_by_scope(
            scope_ids,
            statuses={BeliefStatus.ACTIVE},
        )
        # Prefer recently updated
        active.sort(key=lambda b: b.updated_at, reverse=True)
        return active[: self.existing_for_extract_k]
