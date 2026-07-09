"""Budgeted packing — fit memories under max_memory_tokens.

Why first-class budgets:
- Product constraint: same or better quality at mem0-class token cost
- Unlimited top-k is how memory becomes expensive and slow for the agent LLM
"""

from __future__ import annotations

from mem01.tokens import estimate_tokens
from mem01.types import Belief, PackedMemory, ScoredBelief


def format_belief_line(belief: Belief) -> str:
    """One line injected into the agent prompt for a belief."""
    return f"- {belief.content}"


def pack_beliefs(
    candidates: list[ScoredBelief],
    *,
    max_memory_tokens: int = 800,
    candidate_count: int | None = None,
) -> PackedMemory:
    """Greedy pack highest-score candidates until the token budget is hit.

    Assumes *candidates* are already conflict-filtered and preferably ranked
    highest-first. If not ranked, we sort by score descending here.
    """
    if max_memory_tokens <= 0:
        return PackedMemory(
            beliefs=[],
            text="",
            tokens_used=0,
            max_memory_tokens=max_memory_tokens,
            candidate_count=candidate_count if candidate_count is not None else len(candidates),
        )

    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    chosen: list[Belief] = []
    lines: list[str] = []
    tokens_used = 0

    for c in ordered:
        line = format_belief_line(c.belief)
        # +1 for newline between lines
        extra = estimate_tokens(line if not lines else "\n" + line)
        if tokens_used + extra > max_memory_tokens:
            continue  # try smaller later items? v1 skip; greedy by score only
        # Strict: if this one doesn't fit, skip it (don't stop entirely —
        # a shorter lower-score line might still fit)
        lines.append(line)
        chosen.append(c.belief)
        tokens_used += extra

    text = "\n".join(lines)
    # Recompute exact estimate on final text for honest metrics
    tokens_used = estimate_tokens(text) if text else 0

    return PackedMemory(
        beliefs=chosen,
        text=text,
        tokens_used=tokens_used,
        max_memory_tokens=max_memory_tokens,
        candidate_count=candidate_count if candidate_count is not None else len(candidates),
    )
