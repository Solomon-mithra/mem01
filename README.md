# mem01

Belief-based agent memory: **correct under evolution, cheap on recall**.

> Remembers what matters, forgets what’s wrong, and stays cheap and fast.

## Docs

| File | What it is |
|------|------------|
| [PRODUCT.md](./PRODUCT.md) | Product design (what & why) |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | Build plan (how) |

## Status

**Phase 0 — scaffolding.** Core engine not shipped yet.

## Setup

```bash
cd mem01
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Design in one paragraph

Writes may use an LLM once to turn chat into belief ops (`ADD` / `SUPERSEDE` / …).  
Reads never call an LLM: vector search → conflict filter → token budget pack.  
That split keeps quality high without blowing latency or cost every agent turn.
