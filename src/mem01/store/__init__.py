"""Persistence adapters. Core logic depends only on BeliefStore."""

from mem01.store.base import BeliefStore, ScopeFilter
from mem01.store.factory import create_belief_store
from mem01.store.memory_store import InMemoryBeliefStore

__all__ = [
    "BeliefStore",
    "InMemoryBeliefStore",
    "ScopeFilter",
    "create_belief_store",
]


def __getattr__(name: str):
    if name == "PostgresBeliefStore":
        from mem01.store.postgres_store import PostgresBeliefStore

        return PostgresBeliefStore
    raise AttributeError(name)
