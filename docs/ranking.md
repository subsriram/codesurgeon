# codesurgeon ranking pipeline

> **Keep this doc up to date whenever ranking logic or parameters change.**
> The pipeline lives in `crates/cs-core/src/engine.rs::build_context_capsule` and
> `crates/cs-core/src/search.rs::rerank_by_query_proximity`.

---

## Overview

Ranking runs in four stages every time `run_pipeline` or `get_context_capsule` is called:

```
BM25 (Tantivy)  →  structural injection  →  rerank + centrality boost  →  top-N pivots
                                              + embeddings blend (metal build)
```

---

## Stage 1 — BM25 candidate pool (`search.rs:89`)

- Tantivy full-text search over: `name`, `fqn`, `signature`, `docstring`, `body`
- Returns top-50 candidates by BM25 score
- Body is indexed as-is for callables; for **type definitions** the body is replaced with
  a preview (first ~400 chars) at index time so the class's property declarations are
  searchable rather than the full implementation

**Why 50?** Wide enough to catch symbols whose names don't lexically match the query but
whose body/docstring does (e.g. `PDFLibrary` matching "documents lists categories" via
its `@Published var documents` property).

---

## Stage 2 — Structural intent injection (`engine.rs:707`)

Triggered only for `SearchIntent::Structural` queries (keywords: "coordinator", "central",
"manager", "architecture", "orchestrat", "hub", "entry point", "main class", "controller").

Injects the top `max_pivots * 2` (default: 16) type definitions ranked by
**`family_in_degree_score`** into the candidate pool, skipping any already present from BM25.

**Why injection is needed:** BM25 cannot surface `PDFLibrary` for the query
"central state coordinator" if the class name has no lexical overlap with the query terms.
Injection bypasses BM25 entirely for these candidates.

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

**Cap at `max_pivots * 2` (not max_pivots):** The top-N by family in-degree skews toward
data models. The coordinator (true answer) often sits at rank 10–20. Capping at 2×
max_pivots gives it room to enter.

---

## Stage 3 — Rerank + centrality boost (`engine.rs:722`, `search.rs:135`)

### 3a. Query-proximity rerank (`search.rs:rerank_by_query_proximity`)

Multiplies each BM25 score by a boost factor `b` (starts at 1.0):

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

**Why test penalty on scores not adjacents (originally):** Test files have high BM25 scores
because they contain the exact method names and domain terms they exercise. Without the
penalty they flood the top-10 for almost every query.

**Why body term boost:** Markdown section bodies contain prose that matches conceptual
queries ("BM25 rerank centrality") but those terms don't appear in headings/signatures.
Without body credit, BM25 score alone isn't enough to beat code symbols with matching names.

**Why Markdown × 1.5:** Documentation is preferred over code for conceptual queries.
Combined with the centrality bypass below, this ensures docs compete on content rather
than graph connectivity.

### 3b. Centrality boost + embeddings blend (`engine.rs:760`)

```
# Code symbols:
centrality = centrality_score(id)          # (in*2 + out) / (in*2 + out + 15)
bm25_final = bm25_reranked * (1 + centrality * 3)

# Markdown symbols — centrality bypass:
bm25_final = bm25_reranked * 2.5

# metal/embeddings build only:
if semantic_score > 0:
    final = bm25_final * 0.5 + semantic_cosine * 0.5
else:
    final = bm25_final
```

`centrality_score` uses the symbol's own in+out degree (not family), appropriate for the
general boost since it applies to all symbol kinds equally.

**Why markdown bypasses centrality:** Markdown symbols have no graph edges (docs are never
imported or called), so `centrality_score` is always 0. Without the bypass, a markdown
section that BM25-matches well would score flat while `Symbol` or `CoreEngine` (centrality
≈ 0.5) get ×2.5 amplification. The fixed 2.5 multiplier puts a well-matching doc section
on equal footing with a moderately-central code symbol.

**Embedding text for markdown:** Section bodies use a 1000-char preview (vs 500 for code
type definitions) so that prose content — not just heading text — is semantically indexed.

### 3c. Structural re-sort (`engine.rs:785`)

For `SearchIntent::Structural`, after scoring, re-sort so type definitions rank first using
**`family_in_degree_score`** as the dominant signal:

```
type_score = family_in_degree_score(id) * 20.0 + accumulated_score * 0.05
```

The 20:0.05 ratio ensures in-degree dominates over BM25 body-match. A file that merely
*mentions* the query terms in its body cannot beat the actual hub type.

After re-sorting types, non-type symbols (callables, imports) are appended — they become
adjacents/skeletons rather than pivots.

### 3d. Coordinator bonus (`engine.rs:803`)

Within the structural re-sort, an extra bonus fires for types that **declare** BM25-matched
types as member variables (property ownership, not just body references):

```
seed_names = [type names from BM25 results whose name overlaps with 4-char query stems]
for each type in type_scored (non-test):
    owned = count(seed_names declared as ": TypeName" or ": [TypeName]" in body)
    if owned >= 2:
        score += owned * 5.0
```

**Why this is needed:** Even with family in-degree, data models (PDFDocumentWrapper,
ReadingList, ReadingCategory) can outrank the coordinator because they're referenced
in more signatures. The coordinator is the class that **owns** those models as properties.
`PDFLibrary` declares `var documents: [PDFDocumentWrapper]`, `var readingLists: [ReadingList]`,
`var categories: [ReadingCategory]` — no other class does this for all three simultaneously.

Seeds are restricted to BM25-matched types (not centrality-injected) to prevent SwiftData
ORM models (DocumentModel, CategoryModel) from matching when the query targets app-layer
types (PDFDocumentWrapper, ReadingCategory), which would give a false bonus to
`PersistenceController`.

---

## Stage 4 — Pivot selection and adjacents (`engine.rs:891`)

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
| BM25 pool size | 50 | `engine.rs:697` |
| Structural injection cap | `max_pivots * 2` = 16 | `engine.rs:717` |
| Injected candidate score | `family_in_degree * 5.0` | `engine.rs:719` |
| Centrality boost multiplier | 3.0 | `engine.rs:764` |
| Embedding blend weight | 0.5 / 0.5 | `engine.rs:767` |
| Structural re-sort: in-degree weight | 20.0 | `engine.rs:800` |
| Structural re-sort: BM25 weight | 0.05 | `engine.rs:800` |
| Coordinator bonus threshold | owned >= 2 | `engine.rs:861` |
| Coordinator bonus per owned type | 5.0 | `engine.rs:862` |
| Test file penalty | × 0.25 | `search.rs:173` |
| Utility script penalty | × 0.30 | `search.rs:185` |
| Type definition boost (Structural) | × 2.5 | `search.rs:192` |
| Callable penalty (Structural) | × 0.6 | `search.rs:196` |
| Markdown language boost (rerank) | × 1.5 | `search.rs` |
| Body term match boost | +0.3 per term | `search.rs` |
| Markdown centrality bypass | × 2.5 fixed | `engine.rs` |
| Markdown embedding body preview | 1000 chars | `engine.rs` |
| max_pivots | 8 | `engine.rs:44` |
| max_adjacent | 20 | `engine.rs:46` |
| max_blast_radius_depth | 5 | `engine.rs:47` |
| family_in_degree k | 5 | `graph.rs` |
| in_degree k | 5 | `graph.rs` |
| centrality_score k | 15 | `graph.rs` |
