# codesurgeon ranking pipeline

> **Keep this doc up to date whenever ranking logic or parameters change.**
> The pipeline lives in `crates/cs-core/src/engine.rs::build_context_capsule` and
> `crates/cs-core/src/search.rs::rerank_by_query_proximity`.

---

## Overview

Ranking runs in five stages every time `run_pipeline` or `get_context_capsule` is called:

```
Stage 1: Candidate Retrieval
  ├── BM25 (Tantivy)           top-50 lexical matches
  ├── Graph neighbor expansion top-25 1-hop neighbors of BM25 seeds, by centrality
  ├── Explicit anchors         up to 20 exact symbol-name matches from the query
  └── Semantic (flat scan)     top-25 semantic nearest neighbors  [embeddings build only]
       └── RRF merge ────────  fused candidate pool

Stage 2: Structural injection   high-centrality hub types  [Structural intent only]

Stage 3: Proximity rerank       term-in-name/sig/body boosts, file path penalties,
                                kind boosts for Structural/Explore

Stage 4: Centrality + semantic blend
         ├── centrality boost (code symbols)
         ├── centrality bypass (Markdown symbols)
         ├── embeddings blend (metal/embeddings build only)
         └── Structural re-sort + coordinator bonus  [Structural intent only]

Stage 5: Pivot selection and adjacents
```

---

## Stage 1 — Candidate Retrieval (`engine.rs:build_context_capsule`)

Three independent retrievers run in parallel; their ranked lists are merged with
**Reciprocal Rank Fusion (RRF)**. A candidate appearing in multiple lists is promoted;
unique candidates from any single source are preserved.

### 1a. BM25 (`search.rs:SearchIndex::search`)

- Tantivy full-text search over: `name`, `fqn`, `signature`, `docstring`, `body`
- Returns top-50 candidates by BM25 score
- Body is indexed as-is for callables; for **type definitions** the body is replaced with
  a preview (first ~400 chars) so property declarations are searchable

**Why 50?** Wide enough to catch symbols whose names don't lexically match the query but
whose body/docstring does (e.g. `PDFLibrary` matching "documents lists categories" via
its `@Published var documents` property).

### 1b. Graph neighbor expansion (`engine.rs:graph_candidates`)

- Takes the BM25 seed IDs and walks 1 hop in both directions (callers + callees)
- Excludes seeds already in the BM25 pool
- Ranks neighbors by `centrality_score`, caps at 25

**Why graph neighbors?** BM25 finds lexically matching symbols; their neighbors are
contextually related even when their names don't match the query. A `login_handler` that
calls a `verify_token` found by BM25 is relevant even if "verify" doesn't appear in its
own name. These candidates would otherwise only surface as adjacents, never as pivots.

**Why 1 hop?** 2-hop expansion explodes combinatorially on dense call graphs (a single
utility function called by 200 places would flood the pool with noise).

### 1c. Explicit anchors (`engine.rs:anchor_candidates`, `anchors.rs`)

- Extracts identifier-shaped tokens from the query using three sources:
  1. **Code-block API calls** — `xr.where(...)`, `parse_latex(...)` inside fenced code blocks
  2. **Import statements** — `from sympy.parsing.latex import parse_latex`
  3. **Prose identifiers** — snake_case or CamelCase tokens in the problem statement (stop-list filtered)
- For each extracted name, looks up matching symbols in two stages:
  1. **Exact name** in SQLite (`symbols_by_exact_name`) — strongest signal, score 1.0.
  2. **Name-field BM25** via Tantivy (`search_name`) — fallback when exact lookup
     returns nothing. Tantivy's default tokenizer splits on `_`, so
     `needs_extensions` matches `verify_needs_extensions` naturally. Scored
     slightly lower (0.9) so exact matches rank above tokenised matches when
     both fire. Restricting to the `name` field avoids the signal dilution that
     caused the full-pipeline BM25 to lose the target to body-heavy prose.
- Dotted calls (`xr.where`) use their last segment (`where`).
- Up to `ANCHOR_ROWS_PER_NAME = 5` hits per name per stage; the merged list is
  capped at `ANCHOR_CANDIDATES = 20`.
- Order reflects extraction priority (code blocks first, then prose) and, within
  each name, exact matches before name-BM25 matches.

**Why precision, not recall:** BM25 already handles fuzzy retrieval. Anchors exist to
short-circuit the "task literally names the target function" case where BM25 tokenises
the symbol name (`needs_extensions` → `needs` + `extensions`) and scores each subword
independently. If a token doesn't match a symbol name exactly, the anchor source drops
it — the other retrievers pick up fuzzy matches.

**Why this beats higher BM25 scoring on symbol name:** boosting name matches in Tantivy
would help prose-mentioned names but not code-snippet API calls (`xr.where(...)` tokenises
to `xr`, `where`). The anchor source treats code blocks structurally — the identifier before
`(` is a call target, regardless of surrounding BM25-relevant noise.

**When this is empty:** queries with no extractable identifiers (pure prose bug reports)
produce zero anchors, and the RRF blend degrades gracefully to its prior behaviour.

### 1d. Semantic retrieval (`engine.rs:ann_candidates`) — embeddings build only

- Embeds the query using NomicEmbedTextV15Q (768-dim, L2-normalised)
- Runs a parallel flat cosine scan over the in-memory embedding cache (rayon)
- Returns top-25 nearest neighbors by cosine similarity

The embedding cache is loaded lazily on the first semantic query (`cache_once` in
`CoreEngine`), not at startup. After every reindex pass, `refresh_embedding_cache`
updates the cache directly and marks the once as done so the lazy-init path is skipped.

**Why a flat scan instead of HNSW?** The previous implementation used an HNSW index
(`instant-distance`) persisted to `hnsw.bin`. The index was exact-enough for practical
use but added complexity: background rebuild threads, a binary persistence format, a
stale-count invalidation check, and two extra dependencies (`instant-distance`, `bincode`).
A rayon parallel scan over the mmap'd embedding cache is exact (100% recall), handles
~1M vectors in <100 ms, and has zero warm-up cost — sufficient for any local codebase.

**Why semantic retrieval at Stage 1, not just re-ranking?** Previously, semantic scoring
only re-ranked the BM25 pool (Stage 4). Symbols with high semantic similarity but low
lexical overlap with the query — the hardest cases — never made it into the pool.
Semantic retrieval surfaces these symbols before any reranking occurs.

### 1e. RRF merge (`engine.rs:rrf_merge`)

```
RRF(candidate) = Σ  1 / (60 + rank_i + 1)
                 i
```

One term per retriever list `i` in which the candidate appears.
`k = 60` is the standard default from the original RRF paper (Cormack et al. 2009).
Candidates present in all four lists (strong lexical + graph + semantic + anchor signal)
receive the highest fused scores.

---

## Stage 2 — Structural intent injection (`engine.rs:inject_structural_candidates`)

Triggered only for `SearchIntent::Structural` queries (keywords: "coordinator", "central",
"manager", "architecture", "orchestrat", "hub", "entry point", "main class", "controller").

Injects the top `max_pivots * 2` (default: 16) type definitions ranked by
**`family_in_degree_score`** into the candidate pool, skipping any already present.

**Why this still exists after Stage 1 graph expansion?** Graph neighbor expansion only
reaches symbols adjacent to BM25 hits — it can't surface a hub type whose name has zero
lexical overlap with the query AND which no BM25 hit directly calls. Structural injection
is a global top-K sweep that bypasses both BM25 and graph topology entirely.

**Why `family_in_degree_score` not `in_degree_score`:**
Call edges go to *methods* (`PDFLibrary::addDocument`), not to the class type itself.
A data model like `PDFDocumentWrapper` (in-degree=100, referenced in every signature)
would beat the actual coordinator `PDFLibrary` (class in-degree=11) on raw in-degree.
`family_in_degree_score` aggregates the class's own in-degree plus all its methods'
in-degrees, giving `PDFLibrary` a family score of ~141 vs `PDFDocumentWrapper`'s ~100.

Formula (`graph.rs:family_in_degree_score`):
```
total = class.in_degree + sum(method.in_degree for methods with name prefix "ClassName::")
score = total / (total + 5)   # sigmoid, k=5
```

Injected candidates enter the pool with score `family_in_degree_score * 5.0`.

---

## Stage 3 — Proximity rerank (`search.rs:rerank_by_query_proximity`)

Multiplies each candidate's score by a boost factor `b` (starts at 1.0):

| Signal | Boost |
|--------|-------|
| Query term appears in symbol name | +2.0 per term |
| Query term appears in signature | +1.0 per term |
| Query term appears in body | +0.3 per term |
| File path contains "test"/"spec"/"mock"/"uitest" | × 0.25 |
| Filename starts with check-/run-/setup/generate/gen-/build-/deploy- | × 0.30 |
| Structural/Explore intent + type definition (`is_type_definition()`) | × 2.5 |
| Structural/Explore intent + impl block | × 1.5 |
| Structural/Explore intent + callable (non-test) | × 0.6 |
| Symbol language is Markdown | × 1.5 |

**Why test penalty:** Test files have high BM25 scores because they contain the exact method
names and domain terms they exercise. Without the penalty they flood the top-10 for almost
every query.

**Why body term boost:** Markdown section bodies contain prose that matches conceptual
queries but those terms don't appear in headings/signatures. Without body credit, BM25
score alone isn't enough to beat code symbols with matching names.

**Why Markdown × 1.5:** Documentation is preferred over code for conceptual queries.
Combined with the centrality bypass in Stage 4, this ensures docs compete on content
rather than graph connectivity.

---

## Stage 4 — Centrality boost + semantic blend (`engine.rs:apply_centrality_and_semantics`)

### 4a. Centrality boost

```
# Code symbols:
centrality = centrality_score(id)          # (in*2 + out) / (in*2 + out + 15)
final = reranked_score * (1 + centrality * 3)

# Markdown symbols — centrality bypass:
final = reranked_score * 2.5
```

`centrality_score` uses the symbol's own in+out degree (not family), appropriate for the
general boost since it applies to all symbol kinds equally.

**Why markdown bypasses centrality:** Markdown symbols have no graph edges (docs are never
imported or called), so `centrality_score` is always 0. Without the bypass, a markdown
section that BM25-matches well would score flat while code symbols with even moderate
centrality get amplified. The fixed 2.5 multiplier puts a well-matching doc section on
equal footing with a moderately-central code symbol.

### 4b. Embeddings blend — embeddings build only

```
# Only for candidates with a semantic score > 0:
final = centrality_boosted * 0.5 + cosine_similarity * 0.5
```

Semantic scores are computed **only for candidates already in the pool** (not all N
symbols). Cosine similarity is cheap here: the pool is ~100 candidates at most, and
vectors are L2-normalised so it reduces to a dot product.

**Why re-score here when ANN already retrieved semantic candidates?**
ANN retrieval (Stage 1c) surfaces semantically relevant candidates. This blend
re-ranks them alongside BM25 and graph candidates using a fresh per-candidate cosine
score, combining the graph-boosted lexical signal with the semantic signal in a single
final score.

**Embedding text for markdown:** Section bodies use a 1000-char preview (vs 500 for code
type definitions) so that prose content — not just heading text — is semantically indexed.

### 4c. Structural re-sort (`engine.rs:apply_structural_resort`)

For `SearchIntent::Structural`, after scoring, re-sort so type definitions rank first using
**`family_in_degree_score`** as the dominant signal:

```
type_score = family_in_degree_score(id) * 20.0 + accumulated_score * 0.05
```

The 20:0.05 ratio ensures in-degree dominates over BM25 body-match. A file that merely
*mentions* the query terms in its body cannot beat the actual hub type.

After re-sorting types, non-type symbols (callables, imports) are appended — they become
adjacents/skeletons rather than pivots.

### 4d. Coordinator bonus (`engine.rs:apply_structural_resort`)

Within the structural re-sort, an extra bonus fires for types that **declare** BM25-matched
types as member variables:

```
seed_names = [type names from BM25 results whose name overlaps with 4-char query stems]
for each type in type_scored (non-test):
    owned = count(seed_names declared as ": TypeName" or ": [TypeName]" in body)
    if owned >= 2:
        score += owned * 5.0
```

**Why this is needed:** Even with family in-degree, data models can outrank the coordinator
because they're referenced in more signatures. The coordinator is the class that **owns**
those models as properties. `PDFLibrary` declares `var documents: [PDFDocumentWrapper]`,
`var readingLists: [ReadingList]`, `var categories: [ReadingCategory]` — no other class
does this for all three simultaneously.

Seeds are restricted to BM25-matched types (not centrality-injected or graph-expanded)
to prevent ORM models from matching when the query targets app-layer types, which would
give a false bonus to `PersistenceController`.

---

## Stage 5 — Pivot selection and adjacents (`engine.rs:build_context_capsule`)

- Top `max_pivots` (default: 8) from the scored list become **pivots** (full body shown)
- Adjacents depend on intent:
  - `Debug`: dependencies of pivots (callees — follow error paths down)
  - `Refactor`: blast radius (callers up to depth 5)
  - `Structural`: direct dependents, test files excluded
  - Others: dependents of pivots
- Adjacents capped at `max_adjacent` (default: 20), shown as skeletons

---

## Key discoveries and design decisions

### Why call edges were broken before 2026-03-21

`extract_call_edges` built a `name_to_ids` map keyed on full symbol names
(`Type::method`), but the call-site scanner only extracts simple identifiers (stops at
`:`). So every method call resolved to nothing — only free functions (name has no `::`)
ever matched. Fix: also index by simple name (portion after last `::`).

Edge counts before/after on pdfreader: **1,855 → 12,641**.

### Why `in_degree_score` on types is misleading

Call edges go to methods, not to the class type. Data models (PDFDocumentWrapper,
in-degree=100) are referenced in every function signature, giving them higher class-level
in-degree than the actual coordinator (PDFLibrary, in-degree=11). `family_in_degree_score`
corrects this by aggregating method in-degrees under the parent class.

### Why `all_simple_paths` caused timeouts

With dense call graphs, `all_simple_paths` with depth=20 explores exponentially many paths
before yielding the first one. Replaced with BFS in `graph.rs:find_path` — O(V+E),
finds shortest path, safe on any graph density.

### Why the coordinator bonus exists

Pure in-degree (even family) cannot distinguish the class that *orchestrates* domain models
from the models themselves. The property-ownership check (`var x: [DomainModel]`) uniquely
identifies the coordinator. Requires `>= 2` owned seed types to avoid false positives.

---

## Parameters summary

| Parameter | Value | Location |
|-----------|-------|----------|
| BM25 pool size | 50 | `ranking.rs:BM25_POOL_SIZE` |
| Graph neighbor candidates | 25 | `ranking.rs:GRAPH_CANDIDATES` |
| Semantic retrieval candidates | 25 | `ranking.rs:ANN_CANDIDATES` |
| Explicit anchor candidates | 20 | `ranking.rs:ANCHOR_CANDIDATES` |
| Anchor rows per distinct name | 5 | `ranking.rs:ANCHOR_ROWS_PER_NAME` |
| RRF k constant | 60 | `ranking.rs:RRF_K` |
| Structural injection cap | `max_pivots * 2` = 16 | `engine.rs:inject_structural_candidates` |
| Injected candidate score | `family_in_degree * 5.0` | `engine.rs:inject_structural_candidates` |
| Centrality boost multiplier | 3.0 | `engine.rs:apply_centrality_and_semantics` |
| Markdown centrality bypass | × 2.5 fixed | `engine.rs:apply_centrality_and_semantics` |
| Embedding blend weight | 0.5 / 0.5 | `engine.rs:apply_centrality_and_semantics` |
| Markdown embedding body preview | 1000 chars | `engine.rs` |
| Structural re-sort: in-degree weight | 20.0 | `engine.rs:apply_structural_resort` |
| Structural re-sort: BM25 weight | 0.05 | `engine.rs:apply_structural_resort` |
| Coordinator bonus threshold | owned >= 2 | `engine.rs:apply_structural_resort` |
| Coordinator bonus per owned type | 5.0 | `engine.rs:apply_structural_resort` |
| Test file penalty | × 0.25 | `search.rs:rerank_by_query_proximity` |
| Utility script penalty | × 0.30 | `search.rs:rerank_by_query_proximity` |
| Type definition boost (Structural) | × 2.5 | `search.rs:rerank_by_query_proximity` |
| Callable penalty (Structural) | × 0.6 | `search.rs:rerank_by_query_proximity` |
| Markdown language boost (rerank) | × 1.5 | `search.rs:rerank_by_query_proximity` |
| Body term match boost | +0.3 per term | `search.rs:rerank_by_query_proximity` |
| max_pivots | 8 | `engine.rs` |
| max_adjacent | 20 | `engine.rs` |
| max_blast_radius_depth | 5 | `engine.rs` |
| family_in_degree k | 5 | `graph.rs` |
| centrality_score k | 15 | `graph.rs` |
| Stub score weight | × 0.3 | `ranking.rs:STUB_SCORE_WEIGHT` |
