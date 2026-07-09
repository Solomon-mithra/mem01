"""Lightweight in-process metrics for cost/latency gates.

Why:
- Product success bar needs tokens_used, latency, llm_calls — not accuracy alone
- No external deps (Prometheus later); attach numbers to recall/remember results
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class TimerResult:
    latency_ms: float


@contextmanager
def timer() -> Iterator[TimerResult]:
    """Context manager that records wall-clock latency in milliseconds."""
    result = TimerResult(latency_ms=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.latency_ms = (time.perf_counter() - start) * 1000.0


@dataclass
class Counter:
    """Simple named counters (llm_calls, etc.)."""

    values: dict[str, float] = field(default_factory=dict)

    def incr(self, name: str, amount: float = 1.0) -> None:
        self.values[name] = self.values.get(name, 0.0) + amount

    def get(self, name: str, default: float = 0.0) -> float:
        return self.values.get(name, default)
