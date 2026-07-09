# mem01

Belief-based agent memory: **correct under evolution, cheap on recall**.

> Remembers what matters, forgets what’s wrong, and stays cheap and fast.

**Deploy shape:** self-hosted on your server (Postgres + pgvector). Not multi-tenant SaaS yet. Neon works as the DB later with the same `DATABASE_URL`.

## Docs

| File | What it is |
|------|------------|
| [PRODUCT.md](./PRODUCT.md) | Product design |
| [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) | Build plan |

## Quick start (Docker = dev & prod-like)

```bash
cd mem01
cp .env.example .env
# set OPENAI_API_KEY in .env

docker compose up -d --build
```

| Service | URL |
|---------|-----|
| **mem01 API** | http://localhost:8080 |
| Health | http://localhost:8080/health |
| OpenAPI docs | http://localhost:8080/docs |
| Postgres | `localhost:5433` (user/pass/db: `mem01`) |

```bash
# recall example (after some remember calls)
curl -s http://localhost:8080/v1/recall \
  -H 'Content-Type: application/json' \
  -d '{"query":"where do I live?","user_id":"u1"}'
```

Stop:

```bash
docker compose down
```

## Local Python (same Postgres)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai]"
# .env must include DATABASE_URL=postgresql://mem01:mem01@localhost:5433/mem01
# and docker compose up -d postgres   # or full stack

pytest
```

```python
from mem01 import MemoryClient, create_belief_store
from mem01.embeddings.openai_embedder import OpenAIEmbedder
from mem01.llm.openai_compat import OpenAICompatLLM

store = create_belief_store()  # requires DATABASE_URL → Postgres+pgvector
client = MemoryClient(store=store, embedder=OpenAIEmbedder(), llm=OpenAICompatLLM())
```

Offline logic demo only (no DB): `python examples/basic_usage.py --memory`

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/remember` | Extract + store beliefs |
| POST | `/v1/recall` | Budgeted conflict-safe recall |
| POST | `/v1/correct` | SUPERSEDE by id |
| POST | `/v1/forget` | INVALIDATE by id |
| GET | `/health` | Liveness |

## Design

Writes may use one LLM call to turn chat into belief ops (`ADD` / `SUPERSEDE` / …).  
Reads never call an LLM: vector search → conflict filter → token budget pack.
