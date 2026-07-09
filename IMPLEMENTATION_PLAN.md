# mem01 Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox (`- [ ]`) syntax. Prefer teaching the *why* while shipping working software.  
> **For you (human):** This is the build map. Each phase says **what**, **why**, and **what you should understand** — not line-by-line syntax.

**Goal:** Ship a working mem01 memory engine: belief store + write ops + hot-path recall (0 LLM) + budgeted packing + simple API, with cost/tokens/latency as gates.

**Architecture:** Layered library first (no cloud required). Core is a belief model with lifecycle ops. Writes may call an LLM once per batch to extract ops. Reads are pure retrieval + rules + packing. Background consolidation comes after the hot path works.

**Tech stack (v1):**
- **Language:** Python 3.11+
- **Package layout:** `mem01/` library under this folder (installable locally)
- **Store:** SQLite + sqlite-vec *or* pure SQLite with embeddings in a blob table for day-one simplicity; abstract behind a `BeliefStore` interface so we can swap Postgres+pgvector later
- **Embeddings:** pluggable (`openai` / local sentence-transformers) via an `Embedder` interface
- **LLM (write path only):** pluggable chat client (OpenAI-compatible) for extraction
- **API later:** FastAPI (HTTP) + MCP server — after core is solid
- **Tests:** pytest
- **Metrics:** simple counters/timers in-process first (tokens injected, latency ms, llm_calls)

**Product source of truth:** [`PRODUCT.md`](./PRODUCT.md)

---

## How to read this plan

| Symbol | Meaning |
|--------|---------|
| **Why** | Product or engineering reason this exists |
| **You should know** | Concept worth understanding (not syntax) |
| **We create X because** | Why a module/function exists as a unit |
| **Out of scope here** | Explicitly deferred so we don’t sprawl |

We build **bottom-up**: data model → store → ops → write pipeline → read pipeline → public API → packaging → consolidation → eval.  
Each layer only depends on layers below it. That way you can test beliefs without an LLM, and test recall without FastAPI.

---

## Big picture: data flow

```
                    ┌─────────────────────────────────────┐
   remember() ───►  │  Write pipeline                     │
                    │  extract (LLM) → belief ops → store │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │  Belief store   │
                          │  + embeddings   │
                          └────────┬────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
   recall()  ───►   │  Read pipeline (NO LLM)             │
                    │  search → conflict filter → pack    │
                    └─────────────────────────────────────┘
```

**Why two pipelines?**  
Write can be a bit slower and may use an LLM (quality of *what* we store).  
Read must be fast and cheap (every agent turn). Mixing them makes latency and $ blow up — that is the mem0-class product constraint.

---

## File map (what each file is for)

All paths relative to `open-source/mem01/`.

```
mem01/                          # this product folder
├── PRODUCT.md                  # product truth (exists)
├── IMPLEMENTATION_PLAN.md      # this plan
├── README.md                   # how to install & try (created later)
├── pyproject.toml              # package + deps
├── src/mem01/
│   ├── __init__.py             # public exports: MemoryClient, types
│   ├── types.py                # Belief, Scope, BeliefOp, enums
│   ├── ids.py                  # id generation helpers
│   ├── store/
│   │   ├── base.py             # BeliefStore protocol/interface
│   │   ├── sqlite_store.py     # SQLite implementation
│   │   └── memory_store.py     # in-memory store for tests
│   ├── embeddings/
│   │   ├── base.py             # Embedder protocol
│   │   ├── fake.py             # deterministic fake for tests
│   │   └── openai_embedder.py  # real embeddings (optional)
│   ├── llm/
│   │   ├── base.py             # LLMClient protocol
│   │   ├── fake.py             # scripted replies for tests
│   │   └── openai_compat.py    # real OpenAI-compatible client
│   ├── write/
│   │   ├── extractor.py        # messages → list[BeliefOp]
│   │   └── apply_ops.py        # apply ops to store (the “brain” of writes)
│   ├── read/
│   │   ├── search.py           # vector + scope + status filters
│   │   ├── conflict.py         # drop superseded/invalid; resolve clashes
│   │   ├── rank.py             # score candidates
│   │   └── pack.py             # fit under max_memory_tokens
│   ├── tokens.py               # token counting (approx OK for v1)
│   ├── metrics.py              # latency, token, llm_call counters
│   ├── client.py               # MemoryClient: remember/recall/correct/forget
│   └── consolidate/
│       └── sleep.py            # offline merge/decay (phase 2)
├── tests/
│   ├── conftest.py
│   ├── test_types_store.py
│   ├── test_apply_ops.py
│   ├── test_recall_pipeline.py
│   ├── test_extractor.py
│   ├── test_client.py
│   └── test_conflict_suite.py  # product-defining scenarios
└── examples/
    └── basic_usage.py
```

### Why these boundaries?

| Module | Responsibility | Why separate |
|--------|----------------|--------------|
| `types.py` | Shared shapes | One language for store, write, read — no dict soup |
| `store/*` | Persistence only | Swap SQLite → Postgres without rewriting recall |
| `embeddings/*` | Vectorize text | Tests use FakeEmbedder; prod uses real model |
| `llm/*` | Chat completions | Only write path; never imported by read pipeline |
| `write/*` | Extract + apply | “What should we believe now?” |
| `read/*` | Search + filter + pack | “What do we inject into the prompt?” |
| `client.py` | Public facade | Users see 4 methods, not 12 modules |
| `consolidate/*` | Background hygiene | Cold path; must not sit on hot path |
| `metrics.py` | Cost/latency observability | Product gates are unmeasurable without this |

---

# Phase 0 — Scaffolding

## Task 0: Project skeleton

**Why:** Empty repo structure so every later task has a home. Also locks Python packaging so `import mem01` works in tests.

**You should know:**  
- **src layout** (`src/mem01`) keeps package imports clean and avoids “tests import random scripts.”  
- **Interface-first** (protocols for store/embedder/LLM) is how we keep cost/latency testable without paid APIs.

**Files:**
- Create: `pyproject.toml`
- Create: `src/mem01/__init__.py`
- Create: `tests/conftest.py`
- Create: `README.md` (minimal: name + “see PRODUCT.md”)

- [ ] **Step 0.1 — Create package metadata**

`pyproject.toml` should define:
- package name `mem01`
- python ≥ 3.11
- deps: `pydantic` (or dataclasses — pick **pydantic** for clear validation of BeliefOp payloads), `pytest`, `httpx` later
- optional extras: `openai`, `mcp` later

- [ ] **Step 0.2 — Install editable + prove import**

```bash
cd open-source/mem01
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -c "import mem01; print('ok')"
```

- [ ] **Step 0.3 — Commit**

```bash
git init   # if mem01 is its own repo; otherwise commit inside open-source if tracked
git add pyproject.toml src/mem01 tests README.md
git commit -m "chore: scaffold mem01 package"
```

---

# Phase 1 — Belief model (the product core)

## Task 1: Types — Belief, Scope, Ops

**Why:** Everything else is applying, storing, or retrieving these types. If the model is wrong, the product is wrong. This is the implementation of PRODUCT.md §4.

**You should know:**
- **Belief vs chat message:** A belief is a *claim about the world* with lifecycle. A message is raw conversation. We store beliefs.
- **Status machine:** `active → superseded | invalidated | archived`. Reads default to `active` only.
- **Ops are the write language:** The extractor does not “insert rows”; it emits `ADD` / `SUPERSEDE` / … so the store logic stays deterministic and testable without an LLM.

**We create:**
| Symbol | Why |
|--------|-----|
| `BeliefStatus` enum | Prevents typos like `"Active"` vs `"active"` |
| `BeliefOpType` enum | Closed set of write operations from the product |
| `Scope` / `ScopeIds` | Multi-tool sharing needs explicit boundaries |
| `Belief` model | One row’s shape in memory |
| `BeliefOp` model | One proposed change from extractor or `correct()` |
| `new_belief_id()` | Stable ids for supersede links |

**Files:**
- Create: `src/mem01/types.py`
- Create: `src/mem01/ids.py`
- Create: `tests/test_types_store.py` (types section)

- [ ] **Step 1.1 — Write types**

Implement (fields aligned with PRODUCT.md):

```python
# Conceptual — exact pydantic/dataclass form in code
class BeliefStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    ARCHIVED = "archived"

class BeliefOpType(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"
    INVALIDATE = "INVALIDATE"
    MERGE = "MERGE"

class ScopeKind(str, Enum):
    USER = "user"
    SESSION = "session"
    AGENT = "agent"
    PROJECT = "project"

# Belief: id, content, status, confidence, valid_from, valid_to,
#         supersedes_id, scope, scope_ids, source, timestamps, metadata
# BeliefOp: op, content?, target_id?, confidence?, scope, scope_ids, reason?
```

- [ ] **Step 1.2 — Tests that models validate required fields**

- [ ] **Step 1.3 — Commit** `feat: belief and op types`

---

## Task 2: Store interface + in-memory store

**Why:** Persistence is a *detail*. Behavior (ops, recall) must be testable in RAM in milliseconds. Interface = “ports and adapters”: core logic does not care if bytes live in SQLite or Postgres.

**You should know:**
- **Repository pattern:** `BeliefStore` is the only thing allowed to read/write beliefs. Write/read modules never open SQL directly.
- **Why in-memory first:** Proves the product logic before fighting SQL vector extensions.

**We create:**
| Function / type | Why |
|-----------------|-----|
| `BeliefStore` protocol | `upsert`, `get`, `list_by_scope`, `search_vector`, `mark_status`, … |
| `InMemoryBeliefStore` | Fast tests; reference behavior |
| `save_embedding(id, vector)` / `search(query_vector, …)` | Vectors live beside beliefs but are a separate concern |

**Files:**
- Create: `src/mem01/store/base.py`
- Create: `src/mem01/store/memory_store.py`
- Modify: `tests/test_types_store.py`

- [ ] **Step 2.1 — Define protocol methods** (minimum):

```text
get(belief_id) -> Belief | None
upsert(belief: Belief) -> None
set_status(belief_id, status, valid_to=None) -> None
list_active(scope_filter) -> list[Belief]
save_embedding(belief_id, vector: list[float]) -> None
similarity_search(vector, scope_filter, k, statuses={active}) -> list[ScoredBelief]
```

- [ ] **Step 2.2 — Implement InMemoryBeliefStore** (cosine similarity is fine)

- [ ] **Step 2.3 — Tests:** insert, get, status change, similarity order

- [ ] **Step 2.4 — Commit** `feat: in-memory belief store`

---

## Task 3: SQLite store

**Why:** Real durability for demos and local use without Docker. Postgres+pgvector can replace this later behind the same protocol.

**You should know:**
- **Why SQLite for v1:** Zero ops cost, file-backed, good enough for single-user and early multi-session demos. Matches local real use and low latency.
- **Embeddings in SQLite:** Store as JSON blob or binary; compute cosine in Python for small N. When N is huge, swap to pgvector — **same interface**.

**Files:**
- Create: `src/mem01/store/sqlite_store.py`
- Create: `tests/test_sqlite_store.py`

- [ ] **Step 3.1 — Schema migration on open** (`beliefs` table + `embeddings` table)

- [ ] **Step 3.2 — Implement BeliefStore methods**

- [ ] **Step 3.3 — Same behavioral tests as in-memory** (parametrize both backends if clean)

- [ ] **Step 3.4 — Commit** `feat: sqlite belief store`

---

# Phase 2 — Write path (belief ops engine)

## Task 4: `apply_ops` — the heart of “better than mem0”

**Why:** This is the product wedge in code. mem0-like systems often only ADD. We interpret SUPERSEDE / INVALIDATE / MERGE so the database reflects *current truth*.

**You should know:**
- **Deterministic apply:** Given ops + store state → new state. No LLM here. That means we can unit-test “moved to SF” without paying for tokens.
- **Supersede chain:** New belief `active`, old belief `superseded` + `valid_to=now` + optional link `supersedes_id`.
- **Idempotency:** Applying the same logical correction twice should not create garbage (best-effort: prefer UPDATE/MERGE when content matches).

**We create:**
| Function | Why |
|----------|-----|
| `apply_ops(store, ops, embedder) -> ApplyResult` | Single entry for all write mutations |
| `_apply_add` | Create belief + embed |
| `_apply_supersede` | Link old→new, flip statuses |
| `_apply_invalidate` | Soft-delete for “that was wrong” |
| `_apply_update` | Same id, new content/confidence, re-embed |
| `_apply_merge` | Collapse duplicates into canonical |

**Files:**
- Create: `src/mem01/write/apply_ops.py`
- Create: `src/mem01/embeddings/base.py`
- Create: `src/mem01/embeddings/fake.py`
- Create: `tests/test_apply_ops.py`

- [ ] **Step 4.1 — FakeEmbedder** (e.g. hash text → fixed-dim vector) so tests never call the network

- [ ] **Step 4.2 — Tests first (product scenarios):**
  1. ADD creates active belief  
  2. SUPERSEDE deactivates old, activates new  
  3. INVALIDATE hides belief from active list  
  4. Two ADDs with same scope stay two beliefs (extractor’s job to merge later)

- [ ] **Step 4.3 — Implement apply_ops**

- [ ] **Step 4.4 — Commit** `feat: apply belief ops`

**Out of scope here:** Fancy automatic conflict detection between two unrelated ADDs (that’s extractor + later consolidate). Apply only executes what it’s told.

---

## Task 5: Extractor (LLM → ops)

**Why:** Humans and agents speak in chat. The system needs structured ops. This is the **only** hot-ish path allowed to call an LLM (and we batch).

**You should know:**
- **Structured output:** Ask the model for JSON list of ops, not free prose. Validate into `BeliefOp` models; discard invalid.
- **Why not embed raw messages as memory:** Messages are noisy. Beliefs are compressible and budget-friendly.
- **Cost rule:** One LLM call per `remember()` batch, not per sentence.
- **Prompt contract:** System prompt teaches ADD vs SUPERSEDE vs INVALIDATE with few-shot examples (NY→SF, preference flip, “forget that”).

**We create:**
| Function / type | Why |
|-----------------|-----|
| `LLMClient.complete(messages) -> str` | Swap real/fake LLM |
| `extract_ops(messages, existing_snippets, llm) -> list[BeliefOp]` | Chat → ops |
| `FakeLLM` | Return fixture JSON in tests |

**Files:**
- Create: `src/mem01/llm/base.py`
- Create: `src/mem01/llm/fake.py`
- Create: `src/mem01/llm/openai_compat.py`
- Create: `src/mem01/write/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 5.1 — Protocol + FakeLLM**

- [ ] **Step 5.2 — extract_ops with pydantic validation**

- [ ] **Step 5.3 — Unit tests with FakeLLM scripted outputs** (no API key required)

- [ ] **Step 5.4 — Optional live test marked `@pytest.mark.integration`** if `OPENAI_API_KEY` set

- [ ] **Step 5.5 — Commit** `feat: llm belief extractor`

**Design note for you:**  
We pass **small existing_snippets** (top related active beliefs) into the extractor so the model can emit SUPERSEDE with `target_id` instead of blind ADD. That is how conflict is prevented *at write time* — cheaper than fixing at read forever.

---

# Phase 3 — Read path (fast, 0 LLM)

## Task 6: Search

**Why:** Find candidate beliefs by meaning + scope. This is classic vector memory — not our wedge alone, but required plumbing.

**You should know:**
- **Filter before rank:** Always constrain by `user_id` / `project_id` and `status=active` (unless historical query). Wrong-scope recall is a product bug, not a model bug.
- **k larger than final pack:** Retrieve e.g. 20–50, pack down to token budget. Conflict filter needs a pool.

**We create:**
| Function | Why |
|----------|-----|
| `search_beliefs(store, embedder, query, scope_filter, k) -> list[ScoredBelief]` | Embed query + similarity_search |

**Files:**
- Create: `src/mem01/read/search.py`
- Create: `tests/test_recall_pipeline.py` (search section)

- [ ] **Step 6.1 — Implement + test scope isolation** (user A never sees user B)

- [ ] **Step 6.2 — Commit** `feat: belief search`

---

## Task 7: Conflict filter

**Why:** Even with good writes, the store may have near-duplicates or a bad ADD that slipped through. Read path must not inject two opposite active beliefs when we can detect them.

**You should know:**
- **Prefer structured truth:** If one belief supersedes another, old is already non-active — filter is trivial. Hard case is two actives that *semantically* clash without a link.
- **v1 pragmatic approach:**  
  1. Drop non-active always.  
  2. Drop expired (`valid_to < now`).  
  3. Optional: if two candidates share a metadata `topic_key` (set by extractor), keep higher confidence / newer.  
  4. Full NLI contradiction model = **later** (cost/latency risk).

**We create:**
| Function | Why |
|----------|-----|
| `filter_conflicts(candidates) -> list[ScoredBelief]` | Deterministic, no LLM |

**Files:**
- Create: `src/mem01/read/conflict.py`
- Modify: `tests/test_recall_pipeline.py`

- [ ] **Step 7.1 — Tests for superseded/expired exclusion**

- [ ] **Step 7.2 — Tests for topic_key preference**

- [ ] **Step 7.3 — Implement**

- [ ] **Step 7.4 — Commit** `feat: conflict-safe candidate filter`

---

## Task 8: Rank + token pack

**Why:** Product constraint: **token budget is first-class**. Unlimited top-k is how memory becomes expensive and slow.

**You should know:**
- **Ranking multiplies signals:** similarity × recency × confidence (and boost pins later). Pure cosine often promotes stale near-matches.
- **Packing is a knapsack:** Greedy by score until `max_memory_tokens` is hit. Approximate tokens with `len(text)//4` or `tiktoken` if we add the dep.
- **Why packing reduces latency too:** Smaller prompt → faster LLM downstream (agent side), even if mem01 recall itself is similar.

**We create:**
| Function | Why |
|----------|-----|
| `score(candidate, now) -> float` | Single ranking function we can tune |
| `pack(candidates, max_memory_tokens) -> PackedMemory` | Enforce budget |
| `estimate_tokens(text) -> int` | Shared helper |

**Files:**
- Create: `src/mem01/tokens.py`
- Create: `src/mem01/read/rank.py`
- Create: `src/mem01/read/pack.py`
- Modify: `tests/test_recall_pipeline.py`

- [ ] **Step 8.1 — Tests: pack never exceeds budget; higher scores preferred**

- [ ] **Step 8.2 — Implement rank + pack**

- [ ] **Step 8.3 — Commit** `feat: budgeted memory packing`

---

## Task 9: Wire `recall` pipeline + metrics

**Why:** One function that composes search → conflict → rank → pack and records metrics. This is the hot path SLA surface.

**We create:**
| Function | Why |
|----------|-----|
| `recall(...)` in read or client | Stable entrypoint |
| `Metrics.time("recall_ms")` | Latency gate |
| `Metrics.observe("memory_tokens", n)` | Token gate |

**Files:**
- Create: `src/mem01/metrics.py`
- Modify: `src/mem01/read/*` or `client.py`
- Modify: tests

- [ ] **Step 9.1 — End-to-end recall test with FakeEmbedder + preloaded beliefs**

- [ ] **Step 9.2 — Assert metrics fields present on result** (`tokens_used`, `latency_ms`, `candidate_count`)

- [ ] **Step 9.3 — Commit** `feat: recall pipeline with metrics`

---

# Phase 4 — Public client (the product API)

## Task 10: `MemoryClient`

**Why:** Users (and future HTTP/MCP) should not assemble pipelines. Four verbs match PRODUCT.md.

**You should know:**
- **Facade pattern:** Client holds store, embedder, llm, default scopes.
- **`correct` / `forget`:** Human or agent fixes without re-running full chat extraction — critical for trust.
- **`remember` flow:** extract_ops → apply_ops → return created/updated ids + `llm_calls=1`.

**We create:**
| Method | Why |
|--------|-----|
| `remember(messages, user_id, project_id=None, …)` | Ingest |
| `recall(query, user_id, max_memory_tokens=800, …)` | Retrieve |
| `correct(memory_id, new_value)` | SUPERSEDE by id |
| `forget(memory_id)` | INVALIDATE by id |

**Files:**
- Create: `src/mem01/client.py`
- Create: `tests/test_client.py`
- Modify: `src/mem01/__init__.py` exports

- [ ] **Step 10.1 — Client tests with FakeLLM + FakeEmbedder + InMemory store**

- [ ] **Step 10.2 — Scenario: remember “lives in NY” then remember “moved to SF” → recall “where do I live?” returns SF only** (FakeLLM returns proper SUPERSEDE ops)

- [ ] **Step 10.3 — Commit** `feat: MemoryClient public API`

---

## Task 11: Product conflict suite

**Why:** This is our **definition of better product** — not LoCoMo yet. Codify PRODUCT.md §9 internal suite.

**Files:**
- Create: `tests/test_conflict_suite.py`

Scenarios (each with FakeLLM scripts + assertions):

| # | Scenario | Expect |
|---|----------|--------|
| 1 | Preference flip dark→light | Only light active in recall |
| 2 | Location NY→SF | Only SF; NY superseded |
| 3 | Explicit correct() | New content active; old superseded |
| 4 | forget() | Not in active recall |
| 5 | Scope isolation | project A ≠ project B |
| 6 | Token budget 100 | `tokens_used ≤ 100` |
| 7 | Expired valid_to | Excluded from default recall |

- [ ] **Step 11.1 — Implement suite; all green**

- [ ] **Step 11.2 — Commit** `test: product conflict and budget suite`

---

## Task 12: Example script + README

**Why:** Dogfood path. You should run one script and *feel* the product.

**Files:**
- Create: `examples/basic_usage.py`
- Modify: `README.md`

- [ ] **Step 12.1 — Example with fakes (default) and optional real keys**

- [ ] **Step 12.2 — Commit** `docs: basic usage example`

---

# Phase 5 — Packaging for agents (after core works)

## Task 13: HTTP API (optional but recommended)

**Why:** Language-agnostic access; path to multi-tool sharing without MCP yet.

**You should know:** Thin layer over `MemoryClient` — **no business logic in routes**.

**Files:**
- Create: `src/mem01/api/app.py` (FastAPI)
- Create: `tests/test_api.py`

Endpoints sketch:
- `POST /v1/remember`
- `POST /v1/recall`
- `POST /v1/correct`
- `POST /v1/forget`

- [ ] **Step 13.1 — Implement + test with TestClient**

- [ ] **Step 13.2 — Commit** `feat: HTTP API`

---

## Task 14: MCP server

**Why:** Claude / Cursor can call tools against the same store → shared user/project memory (PRODUCT packaging goal).

**You should know:** MCP tools are just another facade over the same four methods. Identity (`user_id`, `project_id`) must be explicit in tool args or config — **shared memory is not magic**.

**Files:**
- Create: `src/mem01/mcp_server.py`
- Create: docs snippet in README for Claude/Cursor config

- [ ] **Step 14.1 — Tools: remember, recall, correct, forget**

- [ ] **Step 14.2 — Manual smoke with one MCP client**

- [ ] **Step 14.3 — Commit** `feat: MCP server`

---

# Phase 6 — Cold path (“sleep”)

## Task 15: Consolidation job

**Why:** Quality at months of use without slowing every turn. Offline MERGE of near-duplicates, archive old superseded, decay unused low-confidence beliefs.

**You should know:**
- Runs on a schedule or CLI: `python -m mem01.consolidate --db ./mem01.db`
- May use LLM in **batches** for merge decisions — never on `recall()`
- Improves future token use by shrinking the active set

**Files:**
- Create: `src/mem01/consolidate/sleep.py`
- Create: `tests/test_consolidate.py`

- [ ] **Step 15.1 — Near-duplicate detection via embedding distance within scope**

- [ ] **Step 15.2 — MERGE apply_ops + archive old superseded beyond retention**

- [ ] **Step 15.3 — Commit** `feat: offline consolidation`

---

# Phase 7 — Eval gates (honest scorecard)

## Task 16: Benchmark harness (lightweight)

**Why:** PRODUCT success bar: quality + tokens + latency. Start with **internal suite + timing**, not full LoCoMo on day one.

**Files:**
- Create: `evals/run_internal.py`
- Create: `evals/README.md`

Report JSON:
```json
{
  "conflict_suite_pass": true,
  "avg_recall_ms": 0,
  "avg_memory_tokens": 0,
  "llm_calls_per_remember": 1
}
```

- [ ] **Step 16.1 — Script runs suite and prints scorecard**

- [ ] **Step 16.2 — Later: optional LoCoMo adapter** (separate task when ready)

- [ ] **Step 16.3 — Commit** `feat: internal eval scorecard`

---

# Dependency graph (build order)

```
Task 0 scaffold
    → Task 1 types
        → Task 2 memory store
            → Task 3 sqlite store
            → Task 4 apply_ops (+ fake embedder)
                → Task 5 extractor (+ fake llm)
            → Task 6 search
                → Task 7 conflict
                    → Task 8 rank/pack
                        → Task 9 recall metrics
                            → Task 10 client
                                → Task 11 conflict suite
                                → Task 12 examples
                                    → Task 13 HTTP
                                    → Task 14 MCP
                                    → Task 15 sleep
                                    → Task 16 evals
```

**Vertical slice milestone (first “it works” moment):**  
After **Task 10–11**, you have a real product core: remember/recall/correct/forget with conflict suite green — without HTTP/MCP yet.

---

# Hard rules while implementing (do not violate)

1. **`read/` must not import `llm/`** — enforced by code review / later lint.
2. **Default `recall` = 0 LLM calls.**
3. **`remember` ≤ 1 LLM call** per invocation (batch messages inside).
4. **Every `recall` result exposes `tokens_used` and `latency_ms`.**
5. **No graph database in v1.**
6. **Tests must pass without API keys** (fakes default).

---

# What you will learn by the end (curriculum map)

| Phase | Concept you will own |
|-------|----------------------|
| 1–2 | Belief model & repository pattern |
| 4 | Deterministic state machines for memory writes |
| 5 | Structured LLM extraction (ops, not prose) |
| 6–8 | Hybrid retrieval: vector + filters + budget |
| 9–10 | Product facade & observability |
| 11 | Spec-as-tests (quality you can regress) |
| 13–14 | Distribution (HTTP / MCP) without forking logic |
| 15 | Online vs offline systems design |
| 16 | Multi-objective evaluation (not accuracy-only) |

---

# Open implementation choices (locked defaults)

| Choice | Default for v1 | Why |
|--------|----------------|-----|
| Validation library | Pydantic v2 | Clear op validation from LLM JSON |
| Store | SQLite file + in-memory for tests | Zero infra |
| Embeddings | Fake in tests; OpenAI-compatible in prod | Pluggable |
| Token estimate | `tiktoken` if easy, else chars/4 | Budget must exist even if approximate |
| HTTP | FastAPI | Simple, typed |
| Async | Sync first | Latency still fine; less complexity |

---

# Spec coverage checklist

| PRODUCT.md item | Tasks |
|-----------------|-------|
| Belief schema + statuses | 1–3 |
| ADD/UPDATE/SUPERSEDE/INVALIDATE/MERGE | 4–5 |
| Light temporal valid_from/to | 1, 4, 7 |
| Scopes user/session/agent/project | 1–2, 6, 11 |
| Hot path 0 LLM | 6–9 (rule + structure) |
| Write ≤ 1 LLM | 5, 10 |
| Token budget API | 8–10 |
| correct / forget | 10 |
| Metrics tokens + latency | 9–10, 16 |
| MCP packaging | 14 |
| Offline consolidation | 15 |
| Conflict suite | 11 |
| Non-goal: heavy graph | — explicitly omitted |

---

# Execution handoff

Plan saved to: **`mem01/IMPLEMENTATION_PLAN.md`**

**When we implement, two styles:**

1. **Walkthrough mode (recommended for you)** — We do one task at a time in this session. Before each task I explain *what/why*, then we write code/tests. You see the system grow.
2. **Faster agent mode** — Subagents execute tasks; I still summarize what was built and why after each phase.

**Suggested first milestone:** Tasks **0 → 11** (core product + conflict suite). HTTP/MCP after you can try the client for real.

---

*Next action when you say go: start Task 0 (scaffold) in walkthrough mode, and pause after each task for questions.*
