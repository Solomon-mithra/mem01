"""Core domain types for mem01.

This module is the product model in code (see PRODUCT.md §4):

- A **Belief** is a claim about the world with lifecycle (not a raw chat chunk).
- A **BeliefOp** is a proposed write (ADD / SUPERSEDE / …). The extractor emits
  ops; apply_ops turns them into store mutations. Keeping ops as data lets us
  unit-test writes without an LLM.
- **Scope** bounds who shares the belief (user / project / agent / session).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from mem01.ids import new_belief_id


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp (avoid naive datetimes)."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums — closed sets so we never store free-typo statuses like "Active"
# ---------------------------------------------------------------------------


class BeliefStatus(str, Enum):
    """Lifecycle of a stored belief.

    active      → eligible for default recall
    superseded  → replaced by a newer belief (kept for history)
    invalidated → explicitly wrong / forgotten
    archived    → retained but out of normal active set (e.g. after sleep)
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    ARCHIVED = "archived"


class BeliefOpType(str, Enum):
    """Write operations. This is the language of the write pipeline."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"
    INVALIDATE = "INVALIDATE"
    MERGE = "MERGE"


class ScopeKind(str, Enum):
    """Primary scope axis for a belief (what kind of boundary it lives on)."""

    USER = "user"
    SESSION = "session"
    AGENT = "agent"
    PROJECT = "project"


class BeliefSource(str, Enum):
    """Provenance: where the belief came from."""

    EXTRACTION = "extraction"  # remember() / chat extract
    CORRECTION = "correction"  # correct()
    FORGET = "forget"  # forget() path metadata
    MERGE = "merge"  # consolidation
    MANUAL = "manual"  # explicit API / tool inject
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Scope identifiers — multi-tool sharing needs explicit ids
# ---------------------------------------------------------------------------


class ScopeIds(BaseModel):
    """Concrete ids for filtering (any subset may be set depending on scope)."""

    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    project_id: str | None = None

    def matches(self, other: ScopeIds) -> bool:
        """True if every non-None field on *self* equals the same field on *other*.

        Used when listing/searching: a filter with only user_id set matches
        beliefs that share that user_id (project may differ unless also set).
        """
        for name in ("user_id", "session_id", "agent_id", "project_id"):
            wanted = getattr(self, name)
            if wanted is None:
                continue
            if getattr(other, name) != wanted:
                return False
        return True


# ---------------------------------------------------------------------------
# Belief — one row in the memory store
# ---------------------------------------------------------------------------


class Belief(BaseModel):
    """A durable belief with status, validity, and scope."""

    id: str = Field(default_factory=new_belief_id)
    content: str = Field(min_length=1)
    status: BeliefStatus = BeliefStatus.ACTIVE
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    # Light temporal model (not a full knowledge graph)
    valid_from: datetime | None = None
    valid_to: datetime | None = None  # None => still current (if active)

    supersedes_id: str | None = None  # prior belief this one replaces

    scope: ScopeKind = ScopeKind.USER
    scope_ids: ScopeIds = Field(default_factory=ScopeIds)

    source: BeliefSource = BeliefSource.UNKNOWN
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Freeform: topic_key, entity hints, pins, etc.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("content must not be blank")
        return stripped

    def is_current(self, at: datetime | None = None) -> bool:
        """Whether this belief is active and within its validity window."""
        if self.status != BeliefStatus.ACTIVE:
            return False
        when = at or utc_now()
        if self.valid_from is not None and when < self.valid_from:
            return False
        if self.valid_to is not None and when >= self.valid_to:
            return False
        return True


# ---------------------------------------------------------------------------
# BeliefOp — one proposed change from extractor / correct / forget
# ---------------------------------------------------------------------------


class BeliefOp(BaseModel):
    """A single write instruction. apply_ops() executes these against the store.

    Field usage by op type (enforced lightly; apply_ops is the strict enforcer):
    - ADD:        content required
    - UPDATE:     target_id + content and/or confidence
    - SUPERSEDE:  target_id (old) + content (new)
    - INVALIDATE: target_id
    - MERGE:      target_ids (sources) + content (canonical) optional target_id keep
    """

    op: BeliefOpType
    content: str | None = None
    target_id: str | None = None
    target_ids: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    scope: ScopeKind = ScopeKind.USER
    scope_ids: ScopeIds = Field(default_factory=ScopeIds)
    reason: str | None = None
    # Optional extractor hint so read-side conflict filter can prefer one topic
    topic_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_op_shape(self) -> BeliefOp:
        if self.op == BeliefOpType.ADD:
            if not self.content or not self.content.strip():
                raise ValueError("ADD requires non-empty content")
        elif self.op == BeliefOpType.UPDATE:
            if not self.target_id:
                raise ValueError("UPDATE requires target_id")
            if self.content is None and self.confidence is None:
                raise ValueError("UPDATE requires content and/or confidence")
        elif self.op == BeliefOpType.SUPERSEDE:
            if not self.target_id:
                raise ValueError("SUPERSEDE requires target_id (belief being replaced)")
            if not self.content or not self.content.strip():
                raise ValueError("SUPERSEDE requires non-empty content for the new belief")
        elif self.op == BeliefOpType.INVALIDATE:
            if not self.target_id:
                raise ValueError("INVALIDATE requires target_id")
        elif self.op == BeliefOpType.MERGE:
            if len(self.target_ids) < 2 and not (
                self.target_id and self.target_ids
            ):
                # Allow target_id + target_ids, or >=2 target_ids
                ids = list(self.target_ids)
                if self.target_id:
                    ids = [self.target_id, *ids]
                if len(set(ids)) < 2:
                    raise ValueError("MERGE requires at least two belief ids to merge")
        return self


# ---------------------------------------------------------------------------
# Shared result shapes (used later by apply / recall — defined early for stability)
# ---------------------------------------------------------------------------


class ScoredBelief(BaseModel):
    """A belief plus a retrieval score (similarity / rank)."""

    belief: Belief
    score: float


class PackedMemory(BaseModel):
    """Budgeted recall output ready to inject into an agent prompt."""

    beliefs: list[Belief]
    text: str
    tokens_used: int
    max_memory_tokens: int
    candidate_count: int = 0
    latency_ms: float | None = None
