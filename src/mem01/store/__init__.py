"""Persistence adapters. Core logic depends only on BeliefStore."""

from mem01.store.base import BeliefStore, ScopeFilter
from mem01.store.memory_store import InMemoryBeliefStore

__all__ = [
    "BeliefStore",
    "InMemoryBeliefStore",
    "ScopeFilter",
]
