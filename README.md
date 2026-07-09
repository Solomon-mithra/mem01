# mem01

Self-hosted long-term memory for AI agents.

mem01 extracts durable facts from conversation, stores them as **beliefs** with an explicit lifecycle (`ADD` / `UPDATE` / `SUPERSEDE` / `INVALIDATE` / `MERGE`), and retrieves a token-budgeted, conflict-filtered context block on each turn. Writes may call an LLM once per batch; reads do not call an LLM.

| | |
|--|--|
| **Deploy** | Self-hosted (your infrastructure) |
| **Store** | PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) |
| **Interfaces** | HTTP API, Python SDK |
| **Requirements** | Docker (recommended), OpenAI-compatible API key for extraction/embeddings |

Product intent and constraints: [PRODUCT.md](./PRODUCT.md).

---

## Architecture

```
Agents / apps
     │  HTTP or Python SDK
     ▼
┌──────────────────┐      ┌─────────────────────────┐
│  mem01 API       │─────▶│  PostgreSQL + pgvector  │
│  (FastAPI)       │      │  beliefs + embeddings   │
└──────────────────┘      └─────────────────────────┘
```

| Path | Behavior |
|------|----------|
| **Write** (`remember`) | Messages → LLM extraction → belief ops → embed → Postgres |
| **Read** (`recall`) | Embed query → vector search + scope filters → conflict rules → token packer → context string |

Default model stack (configurable): OpenAI chat for extraction, `text-embedding-3-small` (1536-d) for vectors. Anthropic is supported for extraction; embeddings still need an embedding provider.

---

## Quick start

### 1. Configure

```bash
cp .env.example .env
```

Set at least:

```bash
OPENAI_API_KEY=sk-...
```

Docker Compose sets `DATABASE_URL` for the API container. Host-side Python tools should use:

```bash
DATABASE_URL=postgresql://mem01:mem01@localhost:5433/mem01
```

### 2. Run the stack

```bash
docker compose up -d --build
```

| Service | Address |
|---------|---------|
| API | http://localhost:8080 |
| OpenAPI | http://localhost:8080/docs |
| Health | http://localhost:8080/health |
| PostgreSQL | `localhost:5433` (user / password / db: `mem01`) |

### 3. Call the API

```bash
curl -s http://localhost:8080/v1/remember \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "user_1",
    "messages": [{"role": "user", "content": "I live in San Francisco."}]
  }'

curl -s http://localhost:8080/v1/recall \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "user_1",
    "query": "Where does the user live?",
    "max_memory_tokens": 800
  }'
```

Stop:

```bash
docker compose down
```

---

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Process liveness |
| `POST` | `/v1/remember` | Ingest messages; extract and apply belief operations |
| `POST` | `/v1/recall` | Retrieve a budgeted memory block for a query |
| `POST` | `/v1/correct` | Supersede a belief by id with a corrected value |
| `POST` | `/v1/forget` | Invalidate a belief by id |

### Request shapes

**`POST /v1/remember`**

```json
{
  "user_id": "user_1",
  "project_id": "optional",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**`POST /v1/recall`**

```json
{
  "user_id": "user_1",
  "query": "Where does the user live?",
  "max_memory_tokens": 800,
  "k": 20
}
```

**`POST /v1/correct`**

```json
{
  "memory_id": "bel_...",
  "new_value": "User lives in Oakland",
  "confidence": 0.95
}
```

**`POST /v1/forget`**

```json
{
  "memory_id": "bel_...",
  "reason": "optional"
}
```

Interactive schema: http://localhost:8080/docs

---

## Python SDK

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,openai]"
```

Ensure Postgres is running (`docker compose up -d` or your own instance) and `DATABASE_URL` is set in `.env`.

```python
from mem01 import MemoryClient, create_belief_store
from mem01.embeddings.openai_embedder import OpenAIEmbedder
from mem01.llm.openai_compat import OpenAICompatLLM

store = create_belief_store()  # requires DATABASE_URL → Postgres + pgvector
client = MemoryClient(
    store=store,
    embedder=OpenAIEmbedder(),
    llm=OpenAICompatLLM(),
)

client.remember(
    [{"role": "user", "content": "I prefer TypeScript."}],
    user_id="user_1",
)
block = client.recall("language preference", user_id="user_1", max_memory_tokens=800)
print(block.text, block.tokens_used, block.latency_ms)
```

| Method | Description |
|--------|-------------|
| `remember(messages, user_id=...)` | Extract ops and persist |
| `recall(query, user_id=..., max_memory_tokens=800)` | Retrieve packed context |
| `correct(memory_id, new_value)` | Supersede by id |
| `forget(memory_id)` | Invalidate by id |

---

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For real extract/embed | OpenAI (or compatible) API key |
| `DATABASE_URL` | For API / SDK | `postgresql://...` (Docker host: port **5433**) |
| `MEM01_EMBEDDING_DIM` | No (default `1536`) | Must match embedding model dimensions |
| `OPENAI_BASE_URL` | No | Custom OpenAI-compatible base URL |
| `MEM01_LLM_MODEL` | No | Extraction model name |
| `MEM01_EMBED_MODEL` | No | Embedding model name |

**Neon:** use a Neon connection string as `DATABASE_URL` (include `sslmode=require`). No application code changes.

---

## Belief model (summary)

Stored units are beliefs, not raw chat chunks. Operations:

| Op | Effect |
|----|--------|
| `ADD` | Insert active belief |
| `UPDATE` | Revise content/confidence in place |
| `SUPERSEDE` | New active belief; previous marked superseded |
| `INVALIDATE` | Soft-delete (excluded from default recall) |
| `MERGE` | Collapse duplicates into one canonical belief |

Scopes: `user`, `project`, `agent`, `session`. Default sharing is user- and project-level.

Full design: [PRODUCT.md](./PRODUCT.md).

---

## Development

```bash
# Stack
docker compose up -d --build

# Tests (unit tests use an in-process store; Postgres tests need DATABASE_URL)
source .venv/bin/activate
pip install -e ".[dev,openai]"
pytest

# Postgres integration tests
export DATABASE_URL=postgresql://mem01:mem01@localhost:5433/mem01
pytest tests/test_postgres_store.py
```

Repository layout:

| Path | Role |
|------|------|
| `src/mem01/` | Library and API |
| `src/mem01/api/app.py` | FastAPI application |
| `src/mem01/store/postgres_store.py` | Postgres + pgvector backend |
| `docker-compose.yml` | API + database |
| `examples/basic_usage.py` | CLI walkthrough |

---

## Status

| Component | Status |
|-----------|--------|
| Belief store + ops | Implemented |
| Write path (extract → apply) | Implemented |
| Read path (search → conflict → pack) | Implemented |
| HTTP API + Docker Compose | Implemented |
| MCP server | Not yet |
| Background consolidation | Not yet |
| Multi-tenant hosted SaaS | Out of scope for v1 |

---

## License

See repository license file when published.
