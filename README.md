# mem01

Belief-based agent memory: **correct under evolution, cheap on recall**.

> Remembers what matters, forgets what’s wrong, and stays cheap and fast.

## Docs

| File | What it is |
|------|------------|
| [PRODUCT.md](./PRODUCT.md) | Product design (what & why) |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | Build plan (how) |

## Status

**Core + example (Tasks 0–12):** `MemoryClient`, pipelines, product suite, `examples/basic_usage.py`.  
Next: HTTP/MCP, more providers, consolidation.

## Setup

```bash
cd mem01
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## API key + database (`.env`)

```bash
cp .env.example .env
# edit .env — set OPENAI_API_KEY=sk-...
```

| File | Purpose |
|------|---------|
| `mem01/.env` | **Your secrets** (gitignored) |
| `mem01/.env.example` | Template only — safe to commit |

### Store backends

| Backend | When |
|---------|------|
| **SQLite** (default if no URL) | Local dev |
| **Postgres + pgvector** | Production self-hosted (and **Neon**) |

```bash
# Local prod-like DB
docker compose up -d
# .env:
# DATABASE_URL=postgresql://mem01:mem01@localhost:5433/mem01
pip install -e '.[postgres,dev]'
```

```python
from mem01 import MemoryClient, create_belief_store
from mem01.embeddings.openai_embedder import OpenAIEmbedder
from mem01.llm.openai_compat import OpenAICompatLLM

store = create_belief_store()  # reads DATABASE_URL → Postgres, else SQLite
client = MemoryClient(store=store, embedder=OpenAIEmbedder(), llm=OpenAICompatLLM())
```

Same `DATABASE_URL` works with **Neon** (hosted Postgres + pgvector).

`load_env()` also checks `open-source/.env` and the current working directory.

## Try the example

```bash
# Activate the project venv first (macOS usually has python3, not python)
source .venv/bin/activate   # after: python3 -m venv .venv && pip install -e ".[dev]"

# Offline (no API keys) — scripted extract for a full walkthrough
python3 examples/basic_usage.py

# Optional SQLite file
python3 examples/basic_usage.py --db ./demo.db

# Real OpenAI LLM + embeddings (needs OPENAI_API_KEY; usage-based, not free)
export OPENAI_API_KEY=sk-...
python3 examples/basic_usage.py --openai

# Claude extract + OpenAI embeddings
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
python3 examples/basic_usage.py --claude
```

## Quick usage (core API)

```python
from mem01 import MemoryClient, InMemoryBeliefStore
from mem01.embeddings.openai_embedder import OpenAIEmbedder
from mem01.llm.openai_compat import OpenAICompatLLM

client = MemoryClient(
    store=InMemoryBeliefStore(),
    embedder=OpenAIEmbedder(),      # OPENAI_API_KEY
    llm=OpenAICompatLLM(),          # same key; or AnthropicLLM for Claude
    default_user_id="u1",
)

client.remember([{"role": "user", "content": "I prefer TypeScript"}], user_id="u1")
packed = client.recall("language preference", user_id="u1", max_memory_tokens=800)
print(packed.text, packed.tokens_used, packed.latency_ms)
```

For offline tests, pass scripted LLM/embedder fakes instead of OpenAI.

## Design in one paragraph

Writes may use an LLM once to turn chat into belief ops (`ADD` / `SUPERSEDE` / …).  
Reads never call an LLM: vector search → conflict filter → token budget pack.  
That split keeps quality high without blowing latency or cost every agent turn.
