"""Extract BeliefOps from conversation — the only default LLM use on the write path.

Why extract to ops (not raw embeddings of chat):
- Ops are structured: SUPERSEDE needs target_id; ADD needs clean content
- apply_ops stays deterministic and unit-testable
- One LLM call per remember() batch (cost rule)

Provider-agnostic: any LLMClient (Claude, OpenAI-compat, FakeLLM, …).
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from mem01.llm.base import ChatMessage, LLMClient
from mem01.types import Belief, BeliefOp, BeliefOpType, ScopeIds, ScopeKind

EXTRACTOR_SYSTEM_PROMPT = """You are the write-side memory extractor for mem01.
Convert new conversation into a JSON array of belief operations.

Allowed ops:
- ADD: {{"op":"ADD","content":"...","confidence":0.0-1.0,"topic_key":"optional","reason":"optional"}}
- UPDATE: {{"op":"UPDATE","target_id":"bel_...","content":"...","confidence":0.0-1.0}}
- SUPERSEDE: {{"op":"SUPERSEDE","target_id":"bel_...","content":"new truth","confidence":0.0-1.0,"topic_key":"optional"}}
  Use when a new fact replaces an old one (e.g. moved cities, preference flip).
- INVALIDATE: {{"op":"INVALIDATE","target_id":"bel_...","reason":"optional"}}
  Use when the user says something was wrong or should be forgotten.
- MERGE: {{"op":"MERGE","target_ids":["bel_a","bel_b"],"content":"canonical"}}

Rules:
1. Only durable facts, preferences, and identity — not greetings or transient chatter.
2. If an existing belief is listed and the user updates it, SUPERSEDE (with target_id) instead of ADD.
3. Prefer topic_key for recurring topics (location, job, name, preference_*).
4. If nothing durable, return [].
5. Output ONLY a JSON array — no markdown fences, no commentary.
6. Preserve concrete specifics verbatim: names of people, places, organizations,
   titles of books/films/songs, numbers, and dates. Never replace a specific
   ("moved from Sweden") with an abstraction ("moved from her home country").
7. Notable one-time events count as durable facts: trips, purchases, meetings,
   performances, milestones. Record them with their date when stated or inferable.
8. SUPERSEDE is only for CURRENT-STATE facts that changed (moved cities, new job,
   preference flip). Repeated activities are distinct episodes, NOT updates: a second
   camping trip, another painting, another beach visit each get their own ADD with
   their own date. Never SUPERSEDE, UPDATE, or MERGE one dated event with another.
9. Every event belief must state its absolute date or timeframe in the content
   ("...on 8 May 2023", "...in early June 2023") when the session date is known.
10. Capture the specific details people share about their experiences, possessions,
   and creations: what an object or gift means to them, what a sign or artwork said
   or depicted, exact titles, years, durations, and counts. Prefer several small
   precise beliefs over one broad summary belief.

Existing active beliefs (may be empty):
{existing}
"""


def _format_existing(existing: list[Belief]) -> str:
    if not existing:
        return "(none)"
    lines: list[str] = []
    for b in existing:
        topic = b.metadata.get("topic_key", "")
        topic_s = f" topic_key={topic}" if topic else ""
        lines.append(f"- id={b.id}{topic_s} confidence={b.confidence:.2f}: {b.content}")
    return "\n".join(lines)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if fence:
        return fence.group(1).strip()
    return text


def _parse_ops_json(raw: str) -> list[dict[str, Any]]:
    text = _strip_code_fence(raw)
    data = json.loads(text)
    if data is None:
        return []
    if isinstance(data, dict) and "ops" in data:
        data = data["ops"]
    if not isinstance(data, list):
        raise ValueError("extractor output must be a JSON array of ops")
    return data


def extract_ops(
    messages: list[dict[str, str]] | list[ChatMessage],
    *,
    llm: LLMClient,
    existing_beliefs: list[Belief] | None = None,
    scope: ScopeKind = ScopeKind.USER,
    scope_ids: ScopeIds | None = None,
    temperature: float = 0.0,
) -> list[BeliefOp]:
    """Call LLM once; parse and validate BeliefOps; attach default scope_ids."""
    existing_beliefs = existing_beliefs or []
    scope_ids = scope_ids or ScopeIds()

    chat: list[ChatMessage] = [
        ChatMessage(
            role="system",
            content=EXTRACTOR_SYSTEM_PROMPT.format(
                existing=_format_existing(existing_beliefs)
            ),
        )
    ]
    for m in messages:
        if isinstance(m, ChatMessage):
            if m.role == "system":
                # already have system; fold extra system into user note
                chat.append(ChatMessage(role="user", content=f"[context] {m.content}"))
            else:
                chat.append(m)
        else:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role not in ("system", "user", "assistant"):
                role = "user"
            if role == "system":
                chat.append(ChatMessage(role="user", content=f"[context] {content}"))
            else:
                chat.append(ChatMessage(role=role, content=content))  # type: ignore[arg-type]

    # Ensure at least one user/assistant message beyond system
    if len(chat) < 2:
        raise ValueError("extract_ops requires at least one conversation message")

    raw = llm.complete(chat, temperature=temperature)
    try:
        items = _parse_ops_json(raw)
    except (json.JSONDecodeError, ValueError):
        # One retry: LLMs occasionally emit malformed JSON (e.g. "key=value"
        # instead of "key":"value"); a corrective resample usually fixes it.
        retry_chat = chat + [
            ChatMessage(role="assistant", content=raw),
            ChatMessage(
                role="user",
                content=(
                    "That output was not valid JSON. Re-emit the belief operations "
                    "as a valid JSON array only — every key and value quoted and "
                    "separated by a colon, no markdown fences, no commentary."
                ),
            ),
        ]
        raw = llm.complete(retry_chat, temperature=temperature)
        try:
            items = _parse_ops_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"failed to parse extractor JSON after retry: {e}\nraw={raw!r}"
            ) from e

    ops: list[BeliefOp] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Inject default scope if model omitted it
        payload = dict(item)
        payload.setdefault("scope", scope.value)
        if "scope_ids" not in payload:
            payload["scope_ids"] = scope_ids.model_dump()
        else:
            # merge defaults under model-provided ids
            merged = scope_ids.model_dump()
            merged.update({k: v for k, v in payload["scope_ids"].items() if v is not None})
            payload["scope_ids"] = merged
        # Normalize op string
        if "op" in payload and isinstance(payload["op"], str):
            payload["op"] = payload["op"].upper()
        try:
            ops.append(BeliefOp.model_validate(payload))
        except ValidationError:
            # Drop invalid op rather than failing the whole batch
            continue
    return ops
