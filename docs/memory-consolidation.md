# codesurgeon memory consolidation

> **Keep this doc up to date whenever consolidation logic or parameters change.**
> The pipeline lives in `crates/cs-core/src/engine.rs::consolidate_observations` and
> `crates/cs-core/src/memory.rs::compress_observations`.

---

## Overview

codesurgeon keeps a persistent observation store (SQLite) that records two
kinds of entries:

- **Agent-attested memory** — what the agent decided was worth saving via the
  explicit `save_observation` tool. Kinds: `manual`, `insight`.
- **Auto-observations** — `(query, top pivot FQNs)` tuples captured on every
  `run_pipeline` / `get_context_capsule` call. **Opt-in** since #72 (see
  [Auto-observations are opt-in](#auto-observations-are-opt-in) below). Kinds:
  `auto`, `passive`, `file_thrash`, `dead_end`.

Two complementary passes run at startup to keep the store tidy:

```
Startup
  ├── prune_expired()          hard-delete rows past their expires_at
  ├── compress_observations()  per-symbol summary compression  (no embedder needed)
  └── load_embedder()
       └── consolidate_observations()  cross-symbol semantic deduplication  [embeddings build only]
```

---

## Pass 1 — expiry pruning

`MemoryStore::prune_expired` deletes every observation whose `expires_at` is in the past.
This runs unconditionally on every startup, before any other pass.

Default TTLs by kind:

| Kind | Default TTL |
|------|-------------|
| `auto`, `passive`, `file_thrash`, `dead_end` | 7 days (configurable via `auto_ttl_days`) |
| `summary`, `consolidated` | 90 days |
| `manual`, `insight` | none (never expires unless `manual_ttl_days` is set) |

TTLs can be overridden in a workspace-level `codesurgeon.toml`:

```toml
[memory]
auto_ttl_days = 14
manual_ttl_days = 365
```

---

## Pass 2 — per-symbol compression

`MemoryStore::compress_observations` runs next, before the embedder is ready.
It does not need embeddings — it operates purely on symbol FQNs.

**Trigger:** any symbol with ≥ 3 non-expired, non-summary observations.

**Algorithm:**
1. Query all FQNs that exceed the threshold (`fqns_needing_compression`).
2. For each qualifying FQN, take all non-summary observations.
3. Create one `Summary` observation whose content is:
   `[summary of N observations] <most recent observation's content>`
4. Expire (soft-delete) the originals.

`Summary` entries carry a 90-day TTL and are excluded from future compression passes,
so the store converges rather than re-compressing indefinitely.

**When it fires:** only for symbols with multiple saved observations — typically symbols
that have been revisited across many sessions.

---

## Pass 3 — semantic consolidation (embeddings build only)

`CoreEngine::consolidate_observations` runs after `load_embedder` succeeds. It is a no-op
in builds without the `embeddings` feature flag.

**Candidates:** all non-expired `auto` and `passive` observations across all symbols and
sessions. `manual` and `insight` observations are never touched.

**Algorithm:**

```
1. Embed all candidate observation contents in one batch (NomicEmbedTextV15Q, 768-dim)
2. Greedy clustering:
   for each unassigned observation i:
     start cluster {i}
     for each unassigned observation j > i:
       if cosine_similarity(embed[i], embed[j]) >= 0.92:
         add j to cluster
   if cluster.size >= 2: emit cluster
3. For each cluster:
   a. Merge content with merge_cluster_content()
   b. Insert one Consolidated observation (90-day TTL)
   c. Expire all original cluster members
```

**Similarity threshold:** 0.92 (cosine). Tuned to collapse near-duplicate auto-observations
(e.g. the same file re-indexed after minor edits) without merging unrelated entries.

**Content merging (`merge_cluster_content`):**

Auto-observations have a fixed format:
```
Agent queried: "task description" — pivots: fqn1, fqn2
```

The merger:
- Extracts unique query phrases across all cluster members.
- Extracts unique pivot FQNs across all cluster members.
- Produces:
  ```
  [consolidated from N observations] Queries: "q1", "q2" — pivots: fqn1, fqn2
  ```

Observations that don't match the auto format are treated as opaque strings and
deduplicated by exact content.

---

## Auto-observations are opt-in

`run_pipeline` and `get_context_capsule` used to record an `auto` observation
on every call with the content:

```
Agent queried: "task description" — pivots: fqn1, fqn2, fqn3
```

Pass 3 (semantic consolidation) then merged related entries into
`[consolidated from N observations] Queries: X — pivots: Y` rows that
re-surfaced in future capsules under the "Session memory" heading.

**The record side has no success signal.** A run whose pivots missed the fix
site got recorded identically to one that led to the correct diff. Repeated
failures on the same query class therefore cemented the wrong pivots as
"canonical memory" and reinforced the agent's wrong direction on subsequent
attempts — observed on sympy-21379 in the SWE-bench harness, where three
stacked consolidated memories reading *"pivots: symbols, exp, interval.exp"*
actively steered the agent away from `Mod.eval` (the real fix site).

**Default since #72: off.** `EngineConfig::auto_observations = false` skips
the record call sites; nothing is written for `run_pipeline` /
`get_context_capsule` queries. Pass 3 still runs, but with no new auto rows
to consolidate it becomes a no-op on most workspaces.

**Explicit `save_observation` is unaffected** — it writes `manual` / `insight`
kinds, which are never touched by the consolidator and never dropped on
startup. Agents that want cross-session memory should call `save_observation`
deliberately once they know the outcome was good.

**Opt back in** (restores pre-#72 behaviour) via `.codesurgeon/config.toml`:

```toml
[observability]
auto_observations = true
```

This is the right choice if you want the consolidator to learn from query
patterns and your workflow self-corrects quickly enough that bad patterns
don't accumulate. For benchmarks and any workflow where failed runs are
common, leave it off.

---

## Observation kinds reference

| Kind | Written by | Merged by consolidation | Merged by compression | Default TTL |
|------|-----------|------------------------|----------------------|-------------|
| `auto` | engine (passive, on query) | yes | yes | 7d |
| `passive` | engine (file watcher, on `reindex_file`) | yes | yes | 7d |
| `file_thrash` | engine | yes | yes | 7d |
| `dead_end` | engine | yes | yes | 7d |
| `manual` | `save_observation` tool | **no** | yes (per-symbol) | none |
| `insight` | `save_observation` tool | **no** | **no** | none |
| `summary` | compression pass | **no** | **no** | 90d |
| `consolidated` | consolidation pass | **no** | **no** | 90d |

**`passive` observations** carry an additional `change_category` field (one of
`new_symbol`, `deleted_symbol`, `signature_change`, `body_change`, `dependency_added`)
set by the AST diff in `reindex_file`. See `docs/change-categories.md` for details.

---

## Capsule memory injection

When a context capsule is assembled (`run_pipeline`, `get_context_capsule`,
`get_diff_capsule`), observations are injected into the capsule's memory section.
The selection uses semantic relevance rather than plain recency:

**`engine.rs::ranked_observations`:**

1. Fetch a pool of `limit * 3` (min 30) recent non-expired observations.
2. Embed all observation contents in one batch alongside the query (embeddings build only).
3. Score each observation by cosine similarity to the query vector.
4. Drop any observation below `OBSERVATION_MIN_SIMILARITY` (0.3) — these are considered
   topically unrelated and must not consume capsule budget.
5. Return the top `limit` survivors sorted by descending similarity.

Falls back to plain recency order when the embedder is unavailable.

For `get_diff_capsule`, which has no single query string, a synthetic query is constructed
from the names of the changed symbols and used for ranking.

Memory entries are then passed to `build_capsule`, which fits them into a 15% token budget
reserve in order, stopping when that sub-budget is exhausted.

---

## Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Compression threshold | 3 observations per FQN | `memory.rs::compress_observations` |
| Consolidation similarity threshold | 0.92 cosine | `engine.rs::consolidate_observations` |
| Minimum cluster size | 2 | `engine.rs::consolidate_observations` |
| Capsule memory pool size | `limit * 3` (min 30) | `engine.rs::ranked_observations` |
| Capsule memory limit (`get_context_capsule` / `run_pipeline`) | 20 | `engine.rs::build_context_capsule` |
| Capsule memory limit (`get_diff_capsule`) | 10 | `engine.rs::get_diff_capsule` |
| Capsule memory min similarity | 0.3 cosine | `engine.rs::OBSERVATION_MIN_SIMILARITY` |
| Capsule memory budget | 15% of total token budget | `capsule.rs::build_capsule` |
| Auto/passive/thrash/dead-end TTL | 7 days | `memory.rs::ObservationKind::default_ttl_days` |
| Summary/consolidated TTL | 90 days | `memory.rs::ObservationKind::default_ttl_days` |
| Embedding model | NomicEmbedTextV15Q (768-dim) | `embedder.rs::Embedder::new` |
| Auto-observation recording (default since #72) | off | `engine.rs::EngineConfig::auto_observations` |
| Auto-observation TOML override | `[observability] auto_observations = true` | `memory.rs::IndexingConfig::load_from_toml` |
