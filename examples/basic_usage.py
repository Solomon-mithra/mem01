#!/usr/bin/env python3
"""Minimal mem01 walkthrough: remember → recall → correct → forget.

Default: uses DATABASE_URL (Postgres). Start stack with:
  docker compose up -d

Offline unit-style (no Postgres): --memory
With OpenAI: --openai  (needs OPENAI_API_KEY + DATABASE_URL)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from mem01.env import load_env

    load_env()
    from mem01 import MemoryClient, create_belief_store
    from mem01.embeddings.fake import FakeEmbedder
    from mem01.llm.fake import FakeLLM
    from mem01.store.memory_store import InMemoryBeliefStore
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", None) or str(exc)
    print(
        "Missing dependency while importing mem01 "
        f"({missing}).\n\n"
        "  cd mem01 && source .venv/bin/activate\n"
        "  pip install -e \".[dev,postgres,api]\"\n"
        "  docker compose up -d\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


def _scripted_client(*, memory: bool) -> MemoryClient:
    if memory:
        store = InMemoryBeliefStore()
        dim = 32
    else:
        store = create_belief_store()
        dim = int(os.environ.get("MEM01_EMBEDDING_DIM", "1536"))
        # FakeEmbedder dim must match store if Postgres already migrated
        if hasattr(store, "embedding_dim"):
            dim = store.embedding_dim
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
        embedder=FakeEmbedder(dimensions=dim),
        llm=llm,
        default_user_id="demo-user",
    )


def _openai_client() -> MemoryClient:
    from mem01.embeddings.openai_embedder import OpenAIEmbedder
    from mem01.llm.openai_compat import OpenAICompatLLM

    return MemoryClient(
        store=create_belief_store(),
        embedder=OpenAIEmbedder(),
        llm=OpenAICompatLLM(),
        default_user_id="demo-user",
    )


def _claude_client() -> MemoryClient:
    from mem01.embeddings.openai_embedder import OpenAIEmbedder
    from mem01.llm.anthropic import AnthropicLLM

    return MemoryClient(
        store=create_belief_store(),
        embedder=OpenAIEmbedder(),
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
        "--memory",
        action="store_true",
        help="In-memory store only (no Postgres; for offline logic demo)",
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI LLM + embeddings (needs OPENAI_API_KEY + DATABASE_URL)",
    )
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Claude LLM + OpenAI embeddings (needs keys + DATABASE_URL)",
    )
    args = parser.parse_args()

    if args.openai and args.claude:
        parser.error("choose at most one of --openai / --claude")
    if args.memory and (args.openai or args.claude):
        parser.error("--memory is for offline fakes only")

    live = bool(args.openai or args.claude)
    if args.openai:
        client = _openai_client()
    elif args.claude:
        client = _claude_client()
    else:
        client = _scripted_client(memory=args.memory)

    try:
        run_demo(client, live=live)
    finally:
        close = getattr(client.store, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
