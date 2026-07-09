# mem01

Belief-based agent memory: **correct under evolution, cheap on recall**.

> Remembers what matters, forgets what’s wrong, and stays cheap and fast.

## Docs

| File | What it is |
|------|------------|
| [PRODUCT.md](./PRODUCT.md) | Product design (what & why) |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | Build plan (how) |

## Status

**Core milestone (Tasks 0–11):** `MemoryClient` + write/read pipelines + product suite.  
Next: examples, HTTP/MCP, OpenAI-default providers (mem0-style), consolidation.

## Setup

```bash
cd mem01
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Quick usage (core API)

```python
from mem01 import MemoryClient, InMemoryBeliefStore
from mem01.embeddings.fake import FakeEmbedder  # tests; use OpenAI embedder for real
from mem01.llm.fake import FakeLLM              # tests; use OpenAI/Claude for real

client = MemoryClient(
    store=InMemoryBeliefStore(),
    embedder=FakeEmbedder(),
    llm=FakeLLM('[{"op":"ADD","content":"User prefers TypeScript"}]'),
    default_user_id="u1",
)

client.remember([{"role": "user", "content": "I prefer TypeScript"}], user_id="u1")
packed = client.recall("language preference", user_id="u1", max_memory_tokens=800)
print(packed.text, packed.tokens_used, packed.latency_ms)
```

Real dogfood (later wiring): same client with OpenAI LLM + OpenAI embeddings (~$5 is fine).

## Design in one paragraph

Writes may use an LLM once to turn chat into belief ops (`ADD` / `SUPERSEDE` / …).  
Reads never call an LLM: vector search → conflict filter → token budget pack.  
That split keeps quality high without blowing latency or cost every agent turn.
