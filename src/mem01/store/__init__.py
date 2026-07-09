"""Persistence adapters. Core logic depends only on BeliefStore."""

from mem01.store.base import BeliefStore, ScopeFilter
from mem01.store.factory import create_belief_store
from mem01.store.memory_store import InMemoryBeliefStore
from mem01.store.sqlite_store import SqliteBeliefStore

__all__ = [
    "BeliefStore",
    "InMemoryBeliefStore",
    "ScopeFilter",
    "SqliteBeliefStore",
    "create_belief_store",
]


def __getattr__(name: str):
    # Lazy import so core package works without psycopg installed
    if name == "PostgresBeliefStore":
        from mem01.store.postgres_store import PostgresBeliefStore

        return PostgresBeliefStore
    raise AttributeError(name)
