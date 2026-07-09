"""Apply BeliefOps to a BeliefStore — the write-path brain.

Why this module exists:
- Product wedge vs mem0-style ADD-only: SUPERSEDE / INVALIDATE / MERGE are real
- Fully deterministic: no LLM here. Extractor (later) produces ops; we execute them.
- Testable: unit tests prove "moved to SF" without paying for tokens

Flow for each op is small and explicit so lifecycle is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mem01.embeddings.base import Embedder
from mem01.ids import new_belief_id
from mem01.store.base import BeliefStore
from mem01.types import (
    Belief,
    BeliefOp,
    BeliefOpType,
    BeliefSource,
    BeliefStatus,
    ScopeIds,
    utc_now,
)


@dataclass
class ApplyResult:
    """What changed after applying a batch of ops."""

    created_ids: list[str] = field(default_factory=list)
    updated_ids: list[str] = field(default_factory=list)
    superseded_ids: list[str] = field(default_factory=list)
    invalidated_ids: list[str] = field(default_factory=list)
    merged_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def apply_ops(
    store: BeliefStore,
    ops: list[BeliefOp],
    embedder: Embedder,
    *,
    default_source: BeliefSource = BeliefSource.EXTRACTION,
) -> ApplyResult:
    """Execute ops in order against *store*. Embeds any new/changed content."""
    result = ApplyResult()
    for op in ops:
        try:
            if op.op == BeliefOpType.ADD:
                _apply_add(store, op, embedder, default_source, result)
            elif op.op == BeliefOpType.UPDATE:
                _apply_update(store, op, embedder, result)
            elif op.op == BeliefOpType.SUPERSEDE:
                _apply_supersede(store, op, embedder, default_source, result)
            elif op.op == BeliefOpType.INVALIDATE:
                _apply_invalidate(store, op, result)
            elif op.op == BeliefOpType.MERGE:
                _apply_merge(store, op, embedder, default_source, result)
            else:
                result.errors.append(f"unknown op type: {op.op!r}")
        except Exception as exc:  # keep batch resilient; one bad op shouldn't abort all
            result.errors.append(f"{op.op.value}: {exc}")
    return result


def _embed_and_save(store: BeliefStore, embedder: Embedder, belief_id: str, content: str) -> None:
    store.save_embedding(belief_id, embedder.embed(content))


def _metadata_with_topic(op: BeliefOp) -> dict:
    meta = dict(op.metadata)
    if op.topic_key:
        meta["topic_key"] = op.topic_key
    return meta


def _apply_add(
    store: BeliefStore,
    op: BeliefOp,
    embedder: Embedder,
    source: BeliefSource,
    result: ApplyResult,
) -> None:
    assert op.content is not None
    now = utc_now()
    belief = Belief(
        id=new_belief_id(),
        content=op.content.strip(),
        status=BeliefStatus.ACTIVE,
        confidence=op.confidence if op.confidence is not None else 0.7,
        valid_from=now,
        valid_to=None,
        scope=op.scope,
        scope_ids=op.scope_ids,
        source=source,
        created_at=now,
        updated_at=now,
        metadata=_metadata_with_topic(op),
    )
    store.upsert(belief)
    _embed_and_save(store, embedder, belief.id, belief.content)
    result.created_ids.append(belief.id)


def _apply_update(
    store: BeliefStore,
    op: BeliefOp,
    embedder: Embedder,
    result: ApplyResult,
) -> None:
    if not op.target_id:
        raise ValueError("UPDATE requires target_id")
    existing = store.get(op.target_id)
    if existing is None:
        raise KeyError(f"UPDATE target not found: {op.target_id}")
    if existing.status != BeliefStatus.ACTIVE:
        raise ValueError(f"UPDATE target is not active: {op.target_id} ({existing.status})")

    updates: dict = {"updated_at": utc_now()}
    if op.content is not None:
        updates["content"] = op.content.strip()
    if op.confidence is not None:
        updates["confidence"] = op.confidence
    if op.topic_key or op.metadata:
        meta = dict(existing.metadata)
        meta.update(op.metadata)
        if op.topic_key:
            meta["topic_key"] = op.topic_key
        updates["metadata"] = meta

    updated = existing.model_copy(update=updates)
    store.upsert(updated)
    if op.content is not None:
        _embed_and_save(store, embedder, updated.id, updated.content)
    result.updated_ids.append(updated.id)


def _apply_supersede(
    store: BeliefStore,
    op: BeliefOp,
    embedder: Embedder,
    source: BeliefSource,
    result: ApplyResult,
) -> None:
    """New active belief replaces an old one (core of anti-staleness)."""
    if not op.target_id or not op.content:
        raise ValueError("SUPERSEDE requires target_id and content")
    old = store.get(op.target_id)
    if old is None:
        raise KeyError(f"SUPERSEDE target not found: {op.target_id}")

    now = utc_now()
    # Prefer new belief's scope from op; fall back to old for continuity
    scope_ids = op.scope_ids if _scope_ids_any(op.scope_ids) else old.scope_ids
    scope = op.scope if _scope_ids_any(op.scope_ids) else old.scope

    new_belief = Belief(
        id=new_belief_id(),
        content=op.content.strip(),
        status=BeliefStatus.ACTIVE,
        confidence=op.confidence if op.confidence is not None else max(old.confidence, 0.7),
        valid_from=now,
        valid_to=None,
        supersedes_id=old.id,
        scope=scope,
        scope_ids=scope_ids,
        source=source,
        created_at=now,
        updated_at=now,
        metadata=_metadata_with_topic(op) or dict(old.metadata),
    )
    store.upsert(new_belief)
    _embed_and_save(store, embedder, new_belief.id, new_belief.content)

    store.set_status(old.id, BeliefStatus.SUPERSEDED, valid_to=now)
    result.created_ids.append(new_belief.id)
    result.superseded_ids.append(old.id)


def _apply_invalidate(
    store: BeliefStore,
    op: BeliefOp,
    result: ApplyResult,
) -> None:
    if not op.target_id:
        raise ValueError("INVALIDATE requires target_id")
    existing = store.get(op.target_id)
    if existing is None:
        raise KeyError(f"INVALIDATE target not found: {op.target_id}")
    now = utc_now()
    store.set_status(op.target_id, BeliefStatus.INVALIDATED, valid_to=now)
    # Optional reason in metadata
    if op.reason:
        updated = store.get(op.target_id)
        if updated is not None:
            meta = dict(updated.metadata)
            meta["invalidate_reason"] = op.reason
            store.upsert(updated.model_copy(update={"metadata": meta, "updated_at": now}))
    result.invalidated_ids.append(op.target_id)


def _apply_merge(
    store: BeliefStore,
    op: BeliefOp,
    embedder: Embedder,
    source: BeliefSource,
    result: ApplyResult,
) -> None:
    """Collapse multiple beliefs into one canonical active belief."""
    ids: list[str] = list(op.target_ids)
    if op.target_id and op.target_id not in ids:
        ids.insert(0, op.target_id)
    # unique preserve order
    seen: set[str] = set()
    unique_ids: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique_ids.append(i)
    if len(unique_ids) < 2:
        raise ValueError("MERGE requires at least two belief ids")

    beliefs: list[Belief] = []
    for bid in unique_ids:
        b = store.get(bid)
        if b is None:
            raise KeyError(f"MERGE target not found: {bid}")
        beliefs.append(b)

    now = utc_now()
    content = (op.content or beliefs[0].content).strip()
    # Keep highest confidence; scope from first / op
    confidence = op.confidence if op.confidence is not None else max(b.confidence for b in beliefs)
    scope_ids = op.scope_ids if _scope_ids_any(op.scope_ids) else beliefs[0].scope_ids
    scope = op.scope if _scope_ids_any(op.scope_ids) else beliefs[0].scope

    canonical = Belief(
        id=new_belief_id(),
        content=content,
        status=BeliefStatus.ACTIVE,
        confidence=confidence,
        valid_from=now,
        supersedes_id=beliefs[0].id,
        scope=scope,
        scope_ids=scope_ids,
        source=BeliefSource.MERGE if source == BeliefSource.EXTRACTION else source,
        created_at=now,
        updated_at=now,
        metadata={
            **_metadata_with_topic(op),
            "merged_from": unique_ids,
        },
    )
    store.upsert(canonical)
    _embed_and_save(store, embedder, canonical.id, canonical.content)

    for b in beliefs:
        store.set_status(b.id, BeliefStatus.SUPERSEDED, valid_to=now)
        result.superseded_ids.append(b.id)
        result.merged_ids.append(b.id)

    result.created_ids.append(canonical.id)


def _scope_ids_any(scope_ids: ScopeIds) -> bool:
    return any(
        getattr(scope_ids, name) is not None
        for name in ("user_id", "session_id", "agent_id", "project_id")
    )
