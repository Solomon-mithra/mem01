#!/usr/bin/env python3
"""Minimal mem01 walkthrough: remember → recall → correct → forget.

Default: scripted LLM + hash embedder (no API keys).

Optional real providers (mem0-style OpenAI defaults):
  export OPENAI_API_KEY=...
  python examples/basic_usage.py --openai

Claude extract + OpenAI embeddings (Claude has no embed API):
  export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
  python examples/basic_usage.py --claude
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow finding the package from src/ when not installed; deps still need a venv.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from mem01 import InMemoryBeliefStore, MemoryClient, SqliteBeliefStore
    from mem01.embeddings.fake import FakeEmbedder
    from mem01.llm.fake import FakeLLM
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", None) or str(exc)
    print(
        "Missing dependency while importing mem01 "
        f"({missing}).\n\n"
        "Use the project virtualenv (macOS system python3 has no pydantic):\n\n"
        "  cd mem01\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -e \".[dev]\"\n"
        "  python3 examples/basic_usage.py\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


def _scripted_client(db: str | None) -> MemoryClient:
    """Fully offline client with predetermined extract outputs."""
    store = SqliteBeliefStore(db) if db else InMemoryBeliefStore()
    # Scripted conversation: ADD location, then SUPERSEDE is handled after first id known
    llm = FakeLLM(
        json.dumps(
            [
                {
                    "op": "ADD",
                    "content": "User lives in New York",
                    "topic_key": "location",
                    "confidence": 0.9,
                }
            ]
        )
    )
    return MemoryClient(
        store=store,
        embedder=FakeEmbedder(dimensions=32),
        llm=llm,
        default_user_id="demo-user",
    )


def _openai_client(db: str | None) -> MemoryClient:
    from mem01.embeddings.openai_embedder import OpenAIEmbedder
    from mem01.llm.openai_compat import OpenAICompatLLM

    store = SqliteBeliefStore(db) if db else InMemoryBeliefStore()
    return MemoryClient(
        store=store,
        embedder=OpenAIEmbedder(),
        llm=OpenAICompatLLM(),
        default_user_id="demo-user",
    )


def _claude_client(db: str | None) -> MemoryClient:
    from mem01.embeddings.openai_embedder import OpenAIEmbedder
    from mem01.llm.anthropic import AnthropicLLM

    store = SqliteBeliefStore(db) if db else InMemoryBeliefStore()
    return MemoryClient(
        store=store,
        embedder=OpenAIEmbedder(),  # Claude has no embeddings; OpenAI for vectors
        llm=AnthropicLLM(),
        default_user_id="demo-user",
    )


def run_demo(client: MemoryClient, *, live: bool) -> None:
    user = "demo-user"
    print("=== mem01 basic usage ===\n")

    print("1) remember — first fact (lives in NY)")
    r1 = client.remember(
        [{"role": "user", "content": "I live in New York."}],
        user_id=user,
    )
    print(f"   llm_calls={r1.llm_calls} created={r1.apply.created_ids} ok={r1.apply.ok}")
    if r1.apply.errors:
        print(f"   errors={r1.apply.errors}")

    if not r1.apply.created_ids:
        print("   No beliefs created; stopping.")
        return

    old_id = r1.apply.created_ids[0]

    if not live:
        # Script the supersede with the real id from step 1
        client.llm = FakeLLM(
            json.dumps(
                [
                    {
                        "op": "SUPERSEDE",
                        "target_id": old_id,
                        "content": "User lives in San Francisco",
                        "topic_key": "location",
                        "confidence": 0.95,
                    }
                ]
            )
        )

    print("\n2) remember — update (moved to SF)")
    r2 = client.remember(
        [{"role": "user", "content": "I moved to San Francisco last month."}],
        user_id=user,
    )
    print(
        f"   created={r2.apply.created_ids} superseded={r2.apply.superseded_ids} "
        f"ok={r2.apply.ok}"
    )

    print("\n3) recall — where does the user live?")
    packed = client.recall(
        "where does the user live?",
        user_id=user,
        max_memory_tokens=200,
    )
    print(f"   tokens_used={packed.tokens_used} latency_ms={packed.latency_ms:.2f}")
    print(f"   memory block:\n{packed.text or '(empty)'}")

    active_id = packed.beliefs[0].id if packed.beliefs else (
        r2.apply.created_ids[0] if r2.apply.created_ids else old_id
    )

    print("\n4) correct — explicit fix")
    corr = client.correct(active_id, "User lives in Oakland")
    print(f"   superseded={corr.superseded_ids} created={corr.created_ids}")

    packed2 = client.recall("home city", user_id=user, max_memory_tokens=200)
    print(f"   after correct:\n{packed2.text or '(empty)'}")

    final_id = corr.created_ids[0] if corr.created_ids else active_id
    print("\n5) forget — invalidate")
    forgot = client.forget(final_id, reason="demo cleanup")
    print(f"   invalidated={forgot.invalidated_ids}")
    packed3 = client.recall("home city", user_id=user)
    print(f"   after forget: {packed3.text or '(empty)'}")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="mem01 basic usage example")
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI LLM + OpenAI embeddings (needs OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Use Claude LLM + OpenAI embeddings (ANTHROPIC_API_KEY + OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Optional SQLite path (default: in-memory store)",
    )
    args = parser.parse_args()

    if args.openai and args.claude:
        parser.error("choose at most one of --openai / --claude")

    live = bool(args.openai or args.claude)
    if args.openai:
        client = _openai_client(args.db)
    elif args.claude:
        client = _claude_client(args.db)
    else:
        client = _scripted_client(args.db)

    try:
        run_demo(client, live=live)
    finally:
        close = getattr(client.store, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
