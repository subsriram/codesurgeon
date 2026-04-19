# Design (stub): Query-History-Aware Ranking

> **Status**: stub — no implementation yet.
> **Target**: `crates/cs-core/src/engine.rs::build_context_capsule`,
> `crates/cs-core/src/memory.rs`, `crates/cs-core/src/db.rs` (`query_log`,
> `observations`).
> **Related**: `docs/explicit-symbol-anchors.md` (v1.6 anchors handle
> identifier-bearing prose; this doc handles the orthogonal case where the
> task wording has no identifiers).

## Motivation

v1.6 file-diversity pinning fixes the centrality-bias case for queries that
contain explicit identifiers (snake_case / CamelCase / dotted calls). It does
nothing for queries that are pure prose.

Concrete failure case from the SWE-bench v1.6 investigation
(`target/swebench/with/sympy__sympy-21612/transcript.jsonl`):

- Task prose: `"fix latex parsing of fractions yielding wrong expression
  due to missing brackets in denominator"`.
- Anchors extracted: **0** (no identifier-shaped tokens).
- Resulting capsule: 1 marginally-relevant pivot
  (`sympy/parsing/latex/__init__.py::parse_latex`, the public dispatcher) +
  17 unrelated `evalf`/`dict`/`sqrt` skeletons.
- Actual bug site (`_parse_latex_antlr.py::convert_frac`) was nowhere in
  the capsule.
- Agent burned 27 follow-up Grep/Read calls plus 3 wasted pytest attempts
  navigating to the bug site manually. Wall: 893s. Cost: $1.98.

**But the session memory at the bottom of the same capsule already knew the
right answer:**

```
[consolidated from 2 observations] Queries: "fix parse_latex nested
fractions in denominator yielding wrong expression" — pivots: …
sympy/parsing/latex/_parse_latex_antlr.py::parse_latex,
sympy/parsing/latex/__init__.py::parse_latex, …
```

Memory persists as text but is not fed back into ranking. Past pivots from
semantically similar past queries are the strongest available prior on a
prose-only query. We should use them.

## Two implementations to consider

### A. Cheap — memory-FQN anchors as a 5th RRF source

Extract symbol FQNs mentioned in the entries returned by
`ranked_observations()`. Resolve each FQN via `graph.find_by_fqn`. Inject the
resolved IDs as a 5th list in the existing RRF fusion at a low `RRF_K`
(comparable to explicit anchors but slightly weaker, because memory can be
stale).

- Pros: ~1–2 day patch. Drops cleanly into the existing fusion. No new
  retrieval path. Inherits v1.6's pinning if the FQN appears as a literal
  in the current task, but works orthogonally otherwise.
- Cons: only fires when the current query is *already* close enough to the
  past one for `ranked_observations` to surface it. Misses paraphrased tasks
  unless the embedder generalises well.

### B. Better — semantic match against past query texts

Index past task strings (from `observations.context_query` and / or
`query_log.task`) into a separate semantic index. At query time, retrieve
the top-K most similar past queries; for each, take the pivot FQNs that
that past run produced; inject those as the 5th RRF source.

- Pros: handles paraphrased prose. The "pivot of past run that asked the
  same question" is the strongest possible prior.
- Cons: ~1 week. Needs a second embedding store (or reuse symbol embeddings
  on query strings — possibly viable). Needs the consolidation pass to
  preserve a stable mapping from `query → pivots[]` (already largely there,
  just needs schema).

Recommend A first as a low-risk experiment, then B if A shows positive
signal on the SWE-bench prose-only tasks.

## Risks

1. **Memory rot.** Code moves; FQNs rename. Validate `graph.find_by_fqn`
   resolves before injecting, and silently drop stale entries. Surface the
   stale-rate in `index_status` so we can see when memory drifts.
2. **Lock-in / loop closure.** Pivots boosted by memory get re-saved as
   pivots in the new observation, reinforcing themselves on every future
   call. Mitigations:
   - Down-weight memory-derived pivots when computing the *new* observation
     (only persist pivots that came from BM25/ANN/graph/anchor sources, not
     from the memory channel itself).
   - Or only persist pivots in observations that the agent actually *acted
     on* (file in the diff, file Read with body returned, file Edit'd).
3. **Cross-pollution between unrelated tasks.** A past observation tagged
   to one subsystem leaks into queries about a different subsystem because
   of incidental token overlap. Mitigations:
   - Filter memory entries by language match if the current query has a
     language hint.
   - Bias the memory-channel `RRF_K` upward (weaker than current anchors)
     to prevent over-promotion.
4. **Cold-start.** Empty memory means the channel contributes nothing —
   should be a no-op, not a regression. Verify with an integration test
   on a fresh tempdir workspace.

## Validation plan

- New integration test: seed a workspace, run two `run_pipeline` calls
  with literal task strings, then a third call with paraphrased prose.
  Assert the third call's pivots include the symbols pinned by the first
  two.
- SWE-bench prose-only regression set: identify the subset of the 6
  pilot tasks whose prompts have ≤1 identifier-shaped tokens. sympy-21612
  is the canonical example. Re-run with this feature and measure
  cost/walltime delta.
- A/B against v1.6 baseline. Pure win required on tasks with anchors
  (where the channel should be a no-op or small additive lift); meaningful
  cost reduction required on prose-only tasks.

## Out of scope

- Cross-workspace memory sharing.
- Persistence of *unsuccessful* observations as negative signal.
- Re-ranking based on embedded similarity between the current task and
  past task strings *as a standalone scoring channel* (B subsumes this).
