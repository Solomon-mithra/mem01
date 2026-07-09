"""Tests for domain types — the product model in code."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from mem01.ids import new_belief_id
from mem01.types import (
    Belief,
    BeliefOp,
    BeliefOpType,
    BeliefSource,
    BeliefStatus,
    ScopeIds,
    ScopeKind,
    utc_now,
)


def test_new_belief_id_is_unique_and_prefixed():
    a, b = new_belief_id(), new_belief_id()
    assert a != b
    assert a.startswith("bel_")
    assert b.startswith("bel_")


def test_belief_defaults_to_active_with_id():
    b = Belief(content="User prefers TypeScript", scope_ids=ScopeIds(user_id="u1"))
    assert b.status == BeliefStatus.ACTIVE
    assert b.id.startswith("bel_")
    assert b.confidence == 0.7
    assert b.scope == ScopeKind.USER
    assert b.scope_ids.user_id == "u1"
    assert b.valid_to is None


def test_belief_rejects_blank_content():
    with pytest.raises(ValidationError):
        Belief(content="   ")


def test_belief_strips_content():
    b = Belief(content="  lives in SF  ")
    assert b.content == "lives in SF"


def test_belief_is_current_respects_status_and_validity():
    now = utc_now()
    active = Belief(content="ok", valid_from=now - timedelta(days=1))
    assert active.is_current(now) is True

    superseded = Belief(content="old", status=BeliefStatus.SUPERSEDED)
    assert superseded.is_current(now) is False

    expired = Belief(
        content="was true",
        valid_to=now - timedelta(seconds=1),
    )
    assert expired.is_current(now) is False

    future = Belief(
        content="not yet",
        valid_from=now + timedelta(days=1),
    )
    assert future.is_current(now) is False


def test_scope_ids_matches_partial_filter():
    belief_scope = ScopeIds(user_id="u1", project_id="p1")
    filter_user_only = ScopeIds(user_id="u1")
    filter_other_user = ScopeIds(user_id="u2")
    filter_user_and_project = ScopeIds(user_id="u1", project_id="p1")
    filter_wrong_project = ScopeIds(user_id="u1", project_id="p2")

    assert filter_user_only.matches(belief_scope) is True
    assert filter_other_user.matches(belief_scope) is False
    assert filter_user_and_project.matches(belief_scope) is True
    assert filter_wrong_project.matches(belief_scope) is False


def test_add_op_requires_content():
    op = BeliefOp(
        op=BeliefOpType.ADD,
        content="User lives in SF",
        scope_ids=ScopeIds(user_id="u1"),
    )
    assert op.op == BeliefOpType.ADD

    with pytest.raises(ValidationError):
        BeliefOp(op=BeliefOpType.ADD, content=None)


def test_supersede_op_requires_target_and_content():
    op = BeliefOp(
        op=BeliefOpType.SUPERSEDE,
        target_id="bel_old",
        content="User lives in SF",
    )
    assert op.target_id == "bel_old"

    with pytest.raises(ValidationError):
        BeliefOp(op=BeliefOpType.SUPERSEDE, content="only content")

    with pytest.raises(ValidationError):
        BeliefOp(op=BeliefOpType.SUPERSEDE, target_id="bel_old")


def test_invalidate_requires_target():
    op = BeliefOp(op=BeliefOpType.INVALIDATE, target_id="bel_x", reason="user forgot")
    assert op.op == BeliefOpType.INVALIDATE

    with pytest.raises(ValidationError):
        BeliefOp(op=BeliefOpType.INVALIDATE)


def test_update_requires_target_and_change():
    op = BeliefOp(
        op=BeliefOpType.UPDATE,
        target_id="bel_x",
        confidence=0.9,
    )
    assert op.confidence == 0.9

    with pytest.raises(ValidationError):
        BeliefOp(op=BeliefOpType.UPDATE, target_id="bel_x")


def test_merge_requires_at_least_two_ids():
    op = BeliefOp(
        op=BeliefOpType.MERGE,
        target_ids=["bel_a", "bel_b"],
        content="canonical merged belief",
    )
    assert len(op.target_ids) == 2

    with pytest.raises(ValidationError):
        BeliefOp(op=BeliefOpType.MERGE, target_ids=["bel_only_one"])


def test_belief_source_and_status_are_enums():
    b = Belief(
        content="x",
        source=BeliefSource.EXTRACTION,
        status=BeliefStatus.ACTIVE,
    )
    assert b.source == BeliefSource.EXTRACTION
    # JSON-friendly string values
    assert b.status.value == "active"
