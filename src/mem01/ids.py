"""Stable identifiers for beliefs.

Why a helper (not raw uuid in every call site):
- One place to change id format later (ULID, prefixed ids, …)
- Call sites stay readable: `new_belief_id()` instead of uuid boilerplate
"""

from __future__ import annotations

import uuid


def new_belief_id() -> str:
    """Return a new unique belief id (UUID4 hex with bel_ prefix)."""
    return f"bel_{uuid.uuid4().hex}"
