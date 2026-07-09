# mem01 — Product Design

**One-stop product brief.**  
Status: design locked from discussion (2026-07-08) · Implementation: not started  
Positioning: a general agent memory layer that is a **better product than mem0**, under hard **cost / tokens / latency** constraints.

---

## 1. What we are building

**mem01** is a long-term memory layer for AI agents.

It does what mem0 does — extract durable information, store it, retrieve it across sessions — but treats memory as a set of **beliefs that evolve**, not a bag of facts that only grows.

### One-line pitch

> Remembers what matters, forgets what’s wrong, and stays cheap and fast.

### Product thesis

> A general memory layer like mem0, with a real belief model (add / update / supersede / invalidate), light time validity, and aggressive budgeted retrieval — so agents stay correct longer while matching or beating mem0 on tokens and latency.

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
| mem0-class simplicity | Simple API + MCP packaging |

### Non-goals (v1)

- Beating every public benchmark as the primary success metric
- Full temporal knowledge graph (Zep / Graphiti territory) as the core path
- Multi-tenant org / enterprise platform first
- Agent self-edit as the default write path (extra LLM round-trips)
- Local-only as the sole story (local mode can come later)

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
| **Packaging** | mem0-like API + MCP |
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

Hot path (every turn) stays fast and cheap. Cold path improves quality over time.

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
 store beliefs with                    (mem0-sized or smaller)
  status, validity, source


COLD PATH (minutes–hours — not on the request path)
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

If a feature breaks (1) or (2), it is cold-path or advanced — not core.

---

## 6. API surface (mem0-class simplicity)

Conceptual interface (names may change in implementation):

```text
remember(content | messages, user_id, project_id?, ...)
  → extract + belief ops → store

recall(query, user_id, project_id?, max_memory_tokens, ...)
  → budgeted, conflict-safe memory block

correct(memory_id, new_value)
  → human/agent fix → SUPERSEDE

forget(memory_id | query)
  → INVALIDATE
```

### Packaging

| Interface | Purpose |
|-----------|---------|
| HTTP / SDK API | Drop-in memory layer (mem0-like DX) |
| MCP server | Claude, Cursor, and other MCP agents share the same store |
| Optional proxy hooks | Later: inject memory on LLM calls |

Shared memory across Grok / Claude / Cursor is **not automatic**. It works only when each tool is wired to mem01 with the same identity (user / project). That is intentional packaging, not magic.

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
| Quality | Standard memory benchmarks + **internal conflict/staleness suite** |
| Tokens | Avg memory tokens injected per turn |
| Latency | p50 / p95 end-to-end recall |
| Cost | LLM calls per write; estimated $ per 1k turns |

### Internal suite (product-defining)

Cases mem0-class systems often fail:

- Preference changes (dark mode → light mode)
- Location / job / relationship updates (supersede, not duplicate)
- Explicit corrections (“forget that / that was wrong”)
- Multi-session identity with same user
- Budget stress (strict `max_memory_tokens` still returns non-contradictory set)

---

## 10. What we deferred (and why)

| Deferred | Why |
|----------|-----|
| Heavy multi-hop graph | Tokens, latency, operational cost; crowded space |
| Org multi-agent platform | Auth, tenancy, audit dominate before memory is better |
| Agent self-edit as default | Extra LLM tool calls on hot path |
| Local-first as sole mode | Valuable feature later; not the core “better mem0” claim |
| Pure leaderboard chasing | Pulls architecture toward expensive designs |

Unlimited build time means we do the **correctness engine** properly — not that we build every memory paradigm at once.

---

## 11. Suggested build phases (guidance only)

Development time is not the bottleneck; order is about risk and product focus.

1. **Belief store + ops** — schema, ADD/SUPERSEDE/INVALIDATE/MERGE, scopes  
2. **Write path** — extraction that emits belief ops, not only new facts  
3. **Hot recall** — vector + filters + conflict rules + token packer (0 LLM)  
4. **API + MCP** — remember / recall / correct / forget  
5. **Cold consolidation** — merge, decay, archive jobs  
6. **Eval harness** — quality + tokens + latency gates vs mem0 baseline  
7. **Optional** — light graph mode, agent-editable tools, local deploy, coding-agent templates  

---

## 12. Open decisions (not yet locked)

These do not change the product thesis; they are implementation choices:

- [ ] Primary store (e.g. Postgres + pgvector vs other)
- [ ] Extraction model routing (cheap model vs strong model for writes)
- [ ] Exact ranking formula for the packer
- [ ] Multi-tenant / auth model if hosted
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

---

## 14. Summary

**mem01** = mem0’s job (durable agent memory, simple API), done as a **belief system** with **time-aware supersession**, **conflict-safe budgeted retrieval**, and **offline hygiene** — so quality goes up without costing more tokens or latency.

| We optimize for | We refuse |
|-----------------|-----------|
| Correctness under evolution | Graph-first complexity on the hot path |
| Cost / tokens / latency | Accuracy-only leaderboard wins |
| Product people will run for months | Demo-only SOTA tables |

---

*This document is the source of truth for product intent. Implementation plans and code should live under `mem01/` alongside this file.*
