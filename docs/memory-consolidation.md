# codesurgeon memory consolidation

> **Keep this doc up to date whenever consolidation logic or parameters change.**
> The pipeline lives in `crates/cs-core/src/engine.rs::consolidate_observations` and
> `crates/cs-core/src/memory.rs::compress_observations`.

---

## Overview

codesurgeon keeps a persistent observation store (SQLite) that accumulates entries as agents
query the codebase. Without housekeeping this grows unboundedly, degrades retrieval quality,
and wastes tokens when injected into capsules.

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

## Observation kinds reference

| Kind | Written by | Merged by consolidation | Merged by compression | Default TTL |
|------|-----------|------------------------|----------------------|-------------|
| `auto` | engine (passive, on query) | yes | yes | 7d |
| `passive` | engine (file watcher) | yes | yes | 7d |
| `file_thrash` | engine | yes | yes | 7d |
| `dead_end` | engine | yes | yes | 7d |
| `manual` | `save_observation` tool | **no** | yes (per-symbol) | none |
| `insight` | `save_observation` tool | **no** | **no** | none |
| `summary` | compression pass | **no** | **no** | 90d |
| `consolidated` | consolidation pass | **no** | **no** | 90d |

---

## Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Compression threshold | 3 observations per FQN | `memory.rs::compress_observations` |
| Consolidation similarity threshold | 0.92 cosine | `engine.rs::consolidate_observations` |
| Minimum cluster size | 2 | `engine.rs::consolidate_observations` |
| Auto/passive/thrash/dead-end TTL | 7 days | `memory.rs::ObservationKind::default_ttl_days` |
| Summary/consolidated TTL | 90 days | `memory.rs::ObservationKind::default_ttl_days` |
| Embedding model | NomicEmbedTextV15Q (768-dim) | `embedder.rs::Embedder::new` |
