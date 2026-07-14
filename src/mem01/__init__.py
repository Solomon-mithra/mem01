"""mem01 — belief-based agent memory.

See PRODUCT.md for product intent and IMPLEMENTATION_PLAN.md for build order.
"""

from mem01.client import MemoryClient, RememberResult
from mem01.ids import new_belief_id
from mem01.runtime import OpenAIRuntimeSettings, build_openai_memory_client
from mem01.store import (
    BeliefStore,
    InMemoryBeliefStore,
    ScopeFilter,
    create_belief_store,
)
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
from mem01.write.apply_ops import ApplyResult

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Belief",
    "ApplyResult",
    "BeliefOp",
    "BeliefOpType",
    "BeliefSource",
    "BeliefStatus",
    "BeliefStore",
    "InMemoryBeliefStore",
    "MemoryClient",
    "OpenAIRuntimeSettings",
    "PackedMemory",
    "RememberResult",
    "ScopeFilter",
    "ScopeIds",
    "ScopeKind",
    "ScoredBelief",
    "create_belief_store",
    "build_openai_memory_client",
    "new_belief_id",
]
