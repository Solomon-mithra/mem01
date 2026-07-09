"""Token estimation for memory packing.

Why approximate tokens:
- Product requires max_memory_tokens as a first-class budget
- Exact tiktoken is optional; chars/4 is good enough for v1 gating
- Can swap to tiktoken later without changing pack() call sites
"""

from __future__ import annotations

import math


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars per token for English-ish text)."""
    if not text:
        return 0
    # ceil so we never under-count into the budget
    return max(1, math.ceil(len(text) / 4))
