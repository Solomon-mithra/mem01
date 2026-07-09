# mem01 — Product Design

**One-stop product brief.**  
Status: **core engine + self-hosted stack live** (beliefs, write/read pipelines, Postgres+pgvector, Docker Compose, HTTP API). MCP and cold consolidation still to ship.  
Positioning: a general agent memory layer that is a **better product than mem0** on correctness under evolution, under hard **cost / tokens / latency** constraints.  
Deploy shape (v1): **self-hosted** — others run mem01 on *their* server (not multi-tenant SaaS yet).

---

## 1. What we are building

**mem01** is a long-term memory layer for AI agents.

It does what mem0 does — extract durable information, store it, retrieve it across sessions — but treats memory as a set of **beliefs that evolve**, not a bag of facts that only grows.

### One-line pitch

> Remembers what matters, forgets what’s wrong, and stays cheap and fast.

### Product thesis

> A general memory layer like mem0, with a real belief model (add / update / supersede / invalidate), light time validity, and aggressive budgeted retrieval — so agents stay correct longer while matching or beating mem0 on tokens and latency.

### Who runs it

| Mode | What it means |
|------|----------------|
| **v1 Self-hosted** | Customer (or you) runs `docker compose` (or equivalent) on their infra; plugs agents into the HTTP API / Python SDK |
| **Later SaaS** | You host multi-tenant mem01 cloud — **not** the current milestone |

---

## 2. Goals and non-goals

### Goals

| Goal | Meaning |
|------|---------|
| Better product than mem0 | Fewer wrong, stale, or contradictory memories in the prompt |
| Cost discipline | Few LLM calls; cheap storage and query path |
| Token discipline | Same or better quality at ≤ mem0-class context size |
| Latency discipline | Fast hot-path recall; no “wait for graph to finish” |
| Long-horizon quality | Memory does not rot after months of use |
| mem0-class simplicity | Simple API + easy self-host packaging |
| Prod-shaped dev | Dev uses the same stack shape as deploy (Postgres + API), not a toy DB |

### Non-goals (v1)

- Beating every public benchmark as the primary success metric (benchmarks are evidence)
- Full temporal knowledge graph (Zep / Graphiti) as the core path
- Multi-tenant SaaS platform first (auth, billing, org isolation) — only when we push SaaS
- Agent self-edit as the default write path (extra LLM round-trips)
- Local-only “file DB” as the story — local **is** Docker Postgres + mem01 API

### Success bar vs mem0

| Axis | Win condition |
|------|----------------|
| Quality | ≥ mem0 on standard sets; **clearly better** on conflict / staleness |
| Tokens | ≤ mem0 average memory tokens per turn |
| Latency | ≤ mem0 p50/p95 recall (or within a tight band, e.g. +10% p95 max) |
| Cost | Fewer or equal LLM calls per write; transparent $ per 1k turns |

Benchmarks (LoCoMo, LongMemEval, etc.) are **evidence**, not the product.

---

## 3. Primary wedge

We do **not** compete as “mem0 with a higher LoCoMo score.”  
We compete as **belief management + efficient retrieval**.

| Role | Choice |
|------|--------|
| **Primary** | Belief management — conflict, supersede, invalidate, confidence |
| **Co-primary** | Light temporal validity (`valid_from` / `valid_to`, not full graph) |
| **Always-on systems** | Budgeted token packing + offline consolidation (“sleep”) |
| **Hard constraints** | Cost, tokens, latency as release gates |
| **Packaging** | HTTP API + Python SDK; Docker Compose; MCP later |
| **Optional GTM** | Coding agents as first showcase (Claude / Cursor / Grok sharing project memory) |

### Why this wedge

mem0 is strong at: easy API, fact extraction, token-efficient vector recall.  
It is weak at: contradictions, staleness, evolution of facts over time, quality decay at scale.

Heavy graphs can improve multi-hop IQ but often **lose on tokens and latency**. That violates our product constraints. So v1 is smarter structure **without** blocking on a knowledge graph.

---

## 4. Memory model

### Memory is beliefs, not a dump of chat

Each stored item is a **belief** with lifecycle and provenance.

### Belief record (conceptual schema)

| Field | Purpose |
|-------|---------|
| `id` | Stable identifier |
| `content` | Canonical natural-language (or structured) belief |
| `status` | `active` \| `superseded` \| `invalidated` \| `archived` |
| `confidence` | 0–1 or ordinal; used in ranking |
| `valid_from` | When this became true (nullable) |
| `valid_to` | When this stopped being true (null = still current) |
| `supersedes_id` | Prior belief this replaces |
| `scope` | `user` \| `session` \| `agent` \| `project` |
| `scope_ids` | e.g. `user_id`, `project_id`, `agent_id`, `session_id` |
| `source` | Where it came from (messages, tool, human correct) |
| `created_at` / `updated_at` | Audit |
| `embedding` | For vector retrieval (active beliefs preferred) |
| `metadata` | Freeform tags, entity hints, etc. |

### Write operations (not just ADD)

| Op | Meaning |
|----|---------|
| `ADD` | New belief; no conflict with actives |
| `UPDATE` | Same belief, refined wording or confidence |
| `SUPERSEDE` | New belief replaces an old one (old → `superseded`, `valid_to` set) |
| `INVALIDATE` | Mark wrong / no longer trusted without a replacement |
| `MERGE` | Collapse duplicates into one canonical belief |

### Scopes

| Scope | Shared across tools? | Example |
|-------|----------------------|---------|
| User | Yes (if same user id) | Prefers TypeScript; lives in SF |
| Project | Yes (for that project) | Monorepo uses pnpm + Vitest |
| Agent | No — per agent/tool | Tool-specific working notes |
| Session | No — one chat | Current-task scratch |

Default product behavior: share **user + project**; keep **session** ephemeral; **agent** optional isolation so tools don’t pollute each other.

### Light temporal (explicitly not full Zep)

**Do:** validity windows, supersede chains, prefer current beliefs; optional history when the query is temporal.

**Don’t (v1):** full bi-temporal enterprise graph, multi-hop path search on every request, async graph materialization before answers work.

---

## 5. Architecture

### Runtime stack (dev = prod-shaped)

```
┌─────────────────────────────────────────────────────┐
│  docker compose                                     │
│  ┌─────────────────┐     ┌────────────────────────┐ │
│  │  mem01 API      │────▶│  Postgres + pgvector   │ │
│  │  :8080          │     │  :5433 (host)          │ │
│  │  FastAPI        │     │  beliefs + embeddings  │ │
│  └─────────────────┘     └────────────────────────┘ │
└─────────────────────────────────────────────────────┘
         ▲
         │ HTTP / Python SDK
    customer agents / apps
```

| Component | Role |
|-----------|------|
| **mem01 API** | `remember` / `recall` / `correct` / `forget` |
| **Postgres + pgvector** | Production store; vector cosine search in DB |
| **Neon** | Optional hosted Postgres — same `DATABASE_URL`, same code |

### Pipelines

```
WRITE (async-friendly)                 READ (hot — must be fast)
──────────────────────                 ─────────────────────────
messages / events                      query + scopes
        │                                       │
        ▼                                       ▼
 extract + classify                    vector + filters + recency
 (≤ 1 LLM per batch)                   (NO LLM by default)
        │                                       │
        ▼                                       ▼
 belief ops:                           conflict filter (rules)
 ADD / SUPERSEDE /                     token packer (budget)
 INVALIDATE / MERGE                             │
        │                                       ▼
        ▼                              compact memory block
 Postgres beliefs +                    (mem0-sized or smaller)
  embeddings (pgvector)


COLD PATH (minutes–hours — not on the request path)  [planned]
───────────────────────────────────────────────────
 cluster duplicates → MERGE
 decay low-value / unused
 promote stable beliefs
 archive superseded
 (LLM only in batches, off peak)
```

### Hard product rules

1. **Default retrieve = 0 LLM calls** — vector + metadata/SQL + deterministic conflict rules + packer.
2. **Write path ≤ 1 LLM call** per batch of turns (batch when possible).
3. **Token budget is a first-class API argument** (e.g. `max_memory_tokens=800`).
4. **Supersede is structured at write time**, not “hope embeddings are close.”
5. **Graph is optional later**, only for queries that fail vector; never blocks hot path.
6. **Always measure:** quality, tokens injected, p95 latency, LLM calls per write.
7. **Deploy uses Postgres+pgvector** — not a separate “toy” store for development.

If a feature breaks (1) or (2), it is cold-path or advanced — not core.

---

## 6. API surface

### Python SDK (`MemoryClient`)

```text
remember(messages, user_id, project_id?, ...)
  → extract + belief ops → store

recall(query, user_id, project_id?, max_memory_tokens, ...)
  → budgeted, conflict-safe memory block

correct(memory_id, new_value)
  → SUPERSEDE by id

forget(memory_id)
  → INVALIDATE by id
```

`create_belief_store()` reads `DATABASE_URL` (required for real runs).

### HTTP API (shipped)

Base URL (Docker): `http://localhost:8080`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| POST | `/v1/remember` | Extract + store beliefs |
| POST | `/v1/recall` | Budgeted conflict-safe recall |
| POST | `/v1/correct` | SUPERSEDE by id |
| POST | `/v1/forget` | INVALIDATE by id |
| GET | `/docs` | OpenAPI UI |

### Packaging

| Interface | Status | Purpose |
|-----------|--------|---------|
| **Docker Compose** | **Shipped** | Default self-host: Postgres + mem01 API |
| **HTTP API** | **Shipped** | Language-agnostic plug-in for agents |
| **Python SDK** | **Shipped** | In-process `MemoryClient` |
| **MCP server** | Planned | Claude / Cursor shared store |
| **Proxy hooks** | Later | Inject memory on every LLM call |

Shared memory across Grok / Claude / Cursor is **not automatic**. It works only when each tool is wired to mem01 with the same identity (user / project).

### How to run (canonical)

```bash
cp .env.example .env   # OPENAI_API_KEY, DATABASE_URL if calling API from host tools
docker compose up -d --build
# API: http://localhost:8080
# DB:  localhost:5433 → mem01/mem01/mem01
```

Neon: set `DATABASE_URL` to the Neon Postgres URL (`?sslmode=require`); same application image/code.

---

## 7. Cost, tokens, latency strategy

| Concern | Strategy |
|---------|----------|
| Latency | Sync hot path; no wait on graph/consolidation |
| Tokens | Hard budget; rank by relevance × recency × confidence × status |
| Cost | No LLM on read; batch writes; cold-path LLM only |
| Quality at scale | Offline consolidation reduces junk → faster, smaller retrieves later |

Conflict filtering and packing should **reduce** tokens vs dumping top-k contradictory hits — a product win, not a tax.

---

## 8. How this compares

| System | Philosophy | mem01 stance |
|--------|------------|--------------|
| **mem0** | Extract facts → vector store; simple & efficient | Same simplicity; **stronger belief lifecycle** |
| **Zep / Graphiti** | Temporal knowledge graph | Borrow *light* time ideas; **not** full graph core |
| **Letta** | Agent edits memory blocks | Optional later; not default write path (cost/latency) |
| **Plain RAG** | Chunk history, embed, top-k | We store **beliefs**, not raw chat dumps |

---

## 9. Evaluation and release gates

Ship only when the scorecard is honest:

| Axis | What we track |
|------|----------------|
| Quality | Standard memory benchmarks; clear wins on conflict / staleness behavior |
| Tokens | Avg memory tokens injected per turn |
| Latency | p50 / p95 end-to-end recall |
| Cost | LLM calls per write; estimated $ per 1k turns |

---

## 10. What we deferred (and why)

| Deferred | Why |
|----------|-----|
| Heavy multi-hop graph | Tokens, latency, operational cost; crowded space |
| Multi-tenant SaaS | Needed only when *we* host for many customers |
| Agent self-edit as default | Extra LLM tool calls on hot path |
| MCP server | Next packaging layer after HTTP |
| Cold consolidation (“sleep”) jobs | Online path ships first; hygiene is phase 2 |
| Pure leaderboard chasing | Pulls architecture toward expensive designs |

---

## 11. Build status (phases)

| Phase | Content | Status |
|-------|---------|--------|
| 1 | Belief model + ops + scopes | **Done** |
| 2 | Write path (extract → apply_ops) | **Done** |
| 3 | Hot recall (search → conflict → rank → pack) | **Done** |
| 4 | `MemoryClient` + product conflict suite | **Done** |
| 5 | Postgres + pgvector store | **Done** |
| 6 | HTTP API + Docker Compose (API + DB) | **Done** |
| 7 | Prod-shaped dev stack (same as deploy) | **Done** |
| 8 | Cold consolidation | Planned |
| 9 | MCP server | Planned |
| 10 | Optional: light graph, coding-agent templates | Later |
| 11 | Public benchmarks when ready | Evidence |

---

## 12. Open decisions (remaining)

- [x] Primary store: **Postgres + pgvector** (Docker / self-hosted / Neon)
- [x] Deploy: **docker compose** runs mem01 + Postgres
- [ ] Extraction model routing (cheap vs strong for writes)
- [ ] Exact ranking formula tuning for the packer
- [ ] Multi-tenant / auth **only if** SaaS later
- [ ] Name/branding beyond working title **mem01**
- [ ] First vertical demo (general chat vs coding-agent shared memory)

---

## 13. Glossary

| Term | Meaning |
|------|---------|
| **Belief** | A stored memory unit with status and validity, not a raw chat chunk |
| **Hot path** | Code that runs on every remember/recall request |
| **Cold path** | Background consolidation (“sleep”) |
| **Supersede** | New belief replaces an old one cleanly |
| **Token budget** | Max tokens of memory allowed into the model context |
| **Scope** | Boundary of who/what a belief applies to (user, project, …) |
| **Self-hosted** | Customer runs mem01 + DB on their infrastructure |

---

## 14. Summary

**mem01** = mem0’s job (durable agent memory, simple API), done as a **belief system** with **time-aware supersession**, **conflict-safe budgeted retrieval**, and **offline hygiene** (planned) — so quality goes up without costing more tokens or latency.

**Ship shape today:** Docker Compose → **Postgres+pgvector + mem01 HTTP API**; plug agents in; swap DB to Neon via `DATABASE_URL` when wanted. Multi-tenant SaaS is a later product, not the current architecture.

| We optimize for | We refuse |
|-----------------|-----------|
| Correctness under evolution | Graph-first complexity on the hot path |
| Cost / tokens / latency | Accuracy-only leaderboard wins |
| Self-hosted, prod-shaped stack | Separate toy stack for development |
| Product people will run for months | Demo-only SOTA tables |

---

*This document is the source of truth for product intent. Implementation lives under `mem01/` (code, Docker, tests).*
