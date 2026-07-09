"""mem01 — belief-based agent memory.

Public API grows as we implement the client.
See PRODUCT.md for product intent and IMPLEMENTATION_PLAN.md for build order.
"""

from mem01.ids import new_belief_id
from mem01.types import (
    Belief,
    BeliefOp,
    BeliefOpType,
    BeliefSource,
    BeliefStatus,
    PackedMemory,
    ScopeIds,
    ScopeKind,
    ScoredBelief,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Belief",
    "BeliefOp",
    "BeliefOpType",
    "BeliefSource",
    "BeliefStatus",
    "PackedMemory",
    "ScopeIds",
    "ScopeKind",
    "ScoredBelief",
    "new_belief_id",
]
