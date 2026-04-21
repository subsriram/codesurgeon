# Design: Explicit Symbol-Name Anchors in the Ranking Pipeline

> **Status**: v1 → v1.7 landed, plus #67 (reverse-edge expansion) and
> `SymbolKind::Import` pivot filter. #69 (density+query-aware reverse-expand
> ranking) landed and was **reverted** after empirical regression; see the
> "Session 2026-04-21 findings" section below.
> **Target**: `crates/cs-core/src/engine.rs::build_context_capsule`, `crates/cs-core/src/anchors.rs`, `crates/cs-core/src/ranking.rs`
> **Related**: `docs/ranking.md`, SWE-bench benchmark report `benches/swebench/report_29c_interim.md`, `benches/swebench/WARM_WORKSPACES.md`
> **Motivation**: SWE-bench #29c revealed that capsule ranking misses the target
> file in 5 out of 6 regression tasks even with semantic (embedding) retrieval
> enabled. The failure mode is always the same: the task explicitly names the
> target symbol, but the ranker treats it as bag-of-words and surfaces
> tangentially-related files instead.

---

## Session 2026-04-21 findings — `sympy__sympy-21379` deep-dive

End-to-end SWE-bench harness validation on the motivating v1.6 regression
case (`sympy__sympy-21379`: `PolynomialError` on `subs()` with `Piecewise`
argument; fix site `sympy/core/mod.py::Mod.eval`, reached 3+ hops upstream
via reverse edges). Full reproducer, artifacts, and stream logs captured
in `target/swebench/with/sympy__sympy-21379/` on branch
`claude/clever-leakey-7ab62d`.

### Outcome across configurations

| Config | Wall | Cost | Patch | Notes |
|---|---:|---:|---:|---|
| Bare claude (29c backup) | 96 s | $0.30 | 610 B ✓ | No codesurgeon, Grep → edit |
| v1.7 + 5b nudge only | 290 s | $1.01 | 864 B ✓ | Phase 3 baseline |
| v1.7 + #65 cap + CLAUDE.md | 296 s | $1.02 | 864 B ✓ | Phase 4a/b — identical pivots to Phase 3 |
| + #67 reverse-expand | 296 s | $1.02 | 864 B ✓ | Walk fires, 20 candidates, none reach `Mod.eval` |
| **+ #69 density+query-aware** | **600 s** | **—** | **0 B ✗** | Phase 4c timeout, imports won pivots |
| + #69 + import filter | 600 s | — | 0 B ✗ | Phase 4d — non-import pivots still distract |
| **Revert #69, keep import filter** | **279 s** | **$0.95** | **582 B ✓** | Phase 4e — restores success |

### Three empirical claims this session added to the record

#### 1. `Mod.eval` is not reachable via the default reverse-expand walk

The chain `Mod.eval → AssocOp._from_args → ComplexRootOf.__new__ →
PolynomialError` exists in the graph (verified via `codesurgeon flow`),
but default `REVERSE_EXPAND_FAN_OUT = 5` explores a narrow 5×5×5 ≈ 125-node
beam and `Mod.eval` is outside it. Doubling the pool (#69's density-aware
fan-out 5..25) didn't change the outcome: `Mod.eval` still absent from
pivots, adjacents, and skeletons. Tracked by #69.

#### 2. Query-aware ranking has a sharp failure mode on symptom-anchored queries

The core observation: **the fix site has zero query-term overlap by
construction**. For `sympy__sympy-21379` the user names `PolynomialError`,
`subs`, `Piecewise`, `sinh`, `exp`, `symbols` — never `Mod` (the containing
class) or `eval` (the method). Ranking callers by term overlap with the
query actively demotes the fix site because it has none of the words the
query mentions.

Consequences of #69's query-aware ranking observed in harness runs:
- Bare `from X import (A, B, C)` lines scored highest (their FQN/body
  literally list the query's named symbols) and won pivot slots. Fixed by
  filtering `SymbolKind::Import` at pivot selection (commit `359d4ad`).
- Even with imports filtered, the shifted pivot composition sent the agent
  into a deeper exploration loop: 64 tool calls across 10+ files, reaching
  `Mod.eval` but never committing to an edit before the 600 s timeout.
  Phase 4d. Unfixed by any local tuning; root cause is the ranking criterion
  itself.

This is the anti-correlation already flagged in the v1.7 finding below:
query-aware ranking helps queries that already name the fix site (where
anchors alone would have fired) and hurts queries that don't (the class
reverse-expansion was designed for). #69 has been reverted pending a
different approach (centrality-dominated ranking with term overlap as
tiebreaker, or semantic embedding similarity on function bodies rather
than surface-text term match).

#### 3. `SymbolKind::Import` must be excluded from pivot eligibility globally

Not just reverse-expand. Import-statement symbols have a distinctive
pathology: their FQN is literally the statement text (`from X import (A,
B, …)`), so any retriever that scores on name/FQN/body term overlap
promotes them when the query mentions their re-exported symbols. BM25
does this too. Pivot-level filter (`is_eligible_pivot` in
`build_context_capsule`, commit `359d4ad`) is load-bearing regardless of
which retrieval source surfaced the import. Regression test:
`reverse_expand_does_not_surface_import_statements` in
`crates/cs-core/tests/reverse_expand.rs`.

### What still doesn't work — open structural problem

Capsules on symptom-anchored queries remain ~3× bare-claude cost on
sympy-21379 ($0.95 / 279 s vs $0.30 / 96 s). Root cause: **anchor-based
retrieval has no mechanism to reach fix sites named after internal
primitives the user doesn't know about.** Three possible next directions,
in order of estimated leverage:

- **Traceback parsing**: when `context` contains `File "...", line N, in
  func_name` frames, extract FQNs directly as anchors. Deterministic;
  works on ~40% of real bug reports. Doesn't help sympy-21379 (no trace
  in the post) but would solve the class on tasks that do have traces.
- **Body-text embedding similarity to task+context**: rerank reverse-expand
  candidates by semantic similarity of the function body to the query
  rather than by surface term overlap. `Mod.eval`'s body calls `gcd(p, q)`
  and handles numeric cases — semantically near "polynomial"/"modulo"
  themes. Heavier engineering (per-symbol embeddings already exist for
  ANN; the infrastructure is reusable).
- **Reproducer-test awareness**: swebench tasks ship failing tests. Their
  imports and assertions pin the fix site precisely. Narrow win for
  swebench; doesn't generalize to real-world use.

Phase 4 CLAUDE.md's chaining guidance (`run_pipeline` → `get_impact_graph`)
was never actually delivered to the agent in any of the Phase 4 runs.
Post-hoc stream analysis (scanning all 121 events / 467 KB of the final
run for distinctive CLAUDE.md content) found **zero** matches: the file
was written to `workdir/CLAUDE.md` as designed, but `claude --print`
does not auto-load CLAUDE.md from `cwd` (that's interactive-mode-only
behaviour). So every Phase 4 result has been measuring bare TREATMENT_NUDGE
behaviour, not "agent + CLAUDE.md guidance." **Whether the chaining
guidance would help remains untested.**

Harness fix: the `--inject-claude-md` flag now ALSO inlines the
CLAUDE.md body into the PROMPT_PREFIX so the agent actually receives
the content. The on-disk write is retained as an audit artifact.
See `benches/swebench/run.py::build_prompt` and `maybe_inject_claude_md`.

### Harness / measurement infrastructure — stable baseline

Workflow to reproduce any of the rows in the table above:

```bash
bash benches/swebench/prepare_workspace.sh sympy__sympy-21379

uv run benches/swebench/run.py \
  --instance-ids sympy__sympy-21379 \
  --arms with \
  --reuse-workdir target/swebench-warm/sympy__sympy-21379 \
  --max-budget-usd 3.00 \
  --timeout 600 \
  --nudge 5b \
  --inject-claude-md \
  --stream-json \
  --clean
```

Stream log under `target/swebench/with/<instance_id>/claude_stream.jsonl`
is the authoritative record of tool-call behaviour. Diff log under the
same directory is captured even on timeout (commit `07b4848`).

---

## v1.7 — optional `context` param on `run_pipeline` (LANDED)

**Problem v1.6 leaves open**: anchor extraction runs on the `task` string
the agent provides. When agents paraphrase a long problem statement into a
short `task`, identifier tokens routinely drop out of the summary and the
anchor extractor loses its signal — even though the raw source (the
original problem statement, bug report, or stack trace) still names every
relevant symbol. This is the "agent-compliance" ceiling on the pure
prompt-level fix.

**Fix**: give the MCP `run_pipeline` tool an optional `context` parameter.
When set, anchors are extracted from `task + "\n" + context` (dedup via
`extract()`'s existing `seen` set), so identifiers present in the raw
source are recovered regardless of how the agent summarized them. BM25,
ANN, graph retrieval, and intent detection still run on `task` alone — a
large context blob can't blow the primary query budget or mis-classify
the intent.

**Shape of the change**:
- New engine method `CoreEngine::run_pipeline_with_context(task, context, …)`.
  `run_pipeline` itself is unchanged; it delegates to the new method with
  `context=None`. Backward-compatible.
- `build_context_capsule` grows an `anchor_context: Option<&str>` param
  threaded into the anchor-extraction call.
- MCP tool schema advertises `context` with a description that tells the
  agent to pass the raw unmodified source, not a paraphrase. Every MCP
  client (not just the harness) sees this.
- 3 unit tests in `crates/cs-core/tests/engine.rs`:
  `context_none_matches_plain_run_pipeline`,
  `context_recovers_identifier_paraphrased_out_of_task`,
  `context_dedupes_against_task`.

**Not included** (Phase 3 of the anchor roadmap):
- A/B measurement of tool-description alone vs. a PROMPT_PREFIX variant
  that explicitly tells the agent to paste the problem statement into
  `context`. Depends on v1.7 being live; tracked against the 6-task
  regression set.

### Empirical finding — v1.7 on `sympy__sympy-21379`

Single-task validation with `--nudge 5b --stream-json --reuse-workdir`
(2026-04-20). The agent complied with the verbatim-forward nudge:

- **One `run_pipeline` call** at the start with both params populated:
  - `task` = 42 chars (terse summary)
  - `context` = 1,747 chars (verbatim problem statement)
- All anchor identifiers from the problem (`symbols`, `exp`, `sinh`,
  `Piecewise`, `subs`, `PolynomialError`) resolved correctly and appeared
  as pivots. v1.6 file-diversity pinning held.

**But the capsule missed the fix site.** The actual bug is in
`sympy/core/mod.py::Mod.eval`, reached only via `PolynomialError →
parallel_poly_from_expr → gcd → Mod.eval` (4 hops backward). The problem
statement never names `Mod` — anchors can only fire on identifiers the
user mentioned. Consequence: the agent solved the task correctly but
burned 43 post-capsule tool calls (17× Read, 13× Bash, 8× Grep, 4× Edit,
1× ToolSearch) finding the fix site on its own, ending at $1.01 / 290 s
vs bare-claude's $0.30 / 96 s on the same task.

**Takeaway**: v1.7 closes the agent-paraphrase hole in the anchor
pipeline, but does not address the "bug site is transitively reached from
an anchor, not named in the problem" case. The structural fix landed as
**reverse-edge expansion from exception-class anchors** (issue #67,
`ranking.rs:reverse_expand_from_anchors`) — when an anchor resolves to an
`Error`/`Exception`/`Warning` type, the capsule assembler now walks its
callers backward up to 3 hops and fuses the results into RRF. This closes
the chain automatically: an agent no longer has to manually call
`get_impact_graph(PolynomialError)` to find `Mod.eval`. See
`docs/ranking.md` Stage 1d for parameters.

---

## v1.6 — file-diversity pinning (LANDED, supersedes v1.5)

**Problem with v1.5**: the adaptive pivot cap collapsed the capsule to 3
pivots when anchors resolved "cleanly" (≥3 exact hits, 0 bm25-name, ≥2
distinct files). sympy-21612 improved as predicted, but **sympy-21379 went
catastrophic** ($3.04 / 819s vs v1.4's $0.33 / 118s). When many anchors hit,
the cap truncates by RRF ranking, which is graph-centrality-biased. The
agent's task mentioned both `PolynomialError` (the bug site, low centrality)
and `symbols` / `exp` / `Piecewise` (public APIs, high centrality). Centrality
promoted the public APIs to top-3 and `PolynomialError` fell out of the cap.
The agent thrashed for 101 tool calls trying to find the polynomial-error code.

The deeper lesson: **anchors encode user intent, not general importance**.
Ranking anchor hits by centrality inverts the signal — bug sites are usually
low-centrality.

**Fix**: replace the cap logic with a two-phase pivot selection in
`build_context_capsule`:

1. **Phase 1 — Anchor pinning**: for each distinct file among exact anchor
   hits, reserve one pivot slot. Up to `ANCHOR_FILE_BUDGET = 5` files
   pinned. Each file gets its most-specific anchor symbol (max `::` depth,
   shorter fqn on tie).
2. **Phase 2 — RRF fill**: take the remaining pivot slots
   (`max_pivots - pinned_count`) from the BM25/ANN/graph RRF fusion,
   skipping any symbol IDs already pinned.

Result: the capsule is always `max_pivots` total (default 8). Anchor-named
files are guaranteed representation regardless of centrality. Remaining slots
surface central/related candidates as breadth.

| Property | v1.5 adaptive cap | **v1.6 file-diversity pinning** |
|---|---|---|
| Protects low-centrality anchors | No | **Yes** (explicit pin) |
| Capsule size predictable | No (varies 3/5/8) | **Yes** (fixed `max_pivots`) |
| sympy-21379 outcome | $3.04 FAIL | recovers to ~v1.4 level |

**Tests**: `crates/cs-core/tests/ranking_v16.rs` covers file diversity,
`ANCHOR_FILE_BUDGET` cap, single-file deduplication, no-anchor regression,
specificity tie-break, anchor/RRF overlap dedup, and the
`pinned + RRF == max_pivots` invariant.

The `AnchorStats` struct is retained for debug logging but is no longer
threaded into pivot count selection.

---

## 🚨 READ THIS FIRST — v1.3 is the active work; v1.2 is done (see below for v1.2 history)

**Status of prior rounds:**
- **v1** (exact-name anchor lookup) — landed in `anchors.rs` + `anchor_candidates`.
- **v1.1** (BM25-name fallback for substring matches) — landed. Catches the
  sphinx-9711 case where user says `needs_extensions` but the symbol is
  `verify_needs_extensions`.
- **Benchmark driver change — PLANNED, NOT YET LANDED**. Earlier drafts of
  this doc claimed `PROMPT_PREFIX` was updated to instruct the agent to
  preserve identifiers verbatim in the `task` field. That edit was never
  actually made — the prefix only *illustrates* identifier usage via an
  example string. Follow-up work tracks three candidate steering mechanisms
  that supersede the original single-prompt idea:
  1. **Server-side tool description** on a new optional `context` param of
     `run_pipeline` — every MCP client sees it, persuades real-world agents
     to pass the raw problem statement.
  2. **PROMPT_PREFIX variant (identifier preservation)** — nudge the agent
     to keep identifiers in `task`. Agent-compliance-dependent; works on
     the existing schema.
  3. **PROMPT_PREFIX variant (verbatim forward)** — nudge the agent to
     paste the full problem statement into `context`. Easier compliance
     ask (mechanical forwarding, no summarization judgment), but still
     depends on the new schema field landing first.

  The Phase 1 prompt-split that now branches `PROMPT_PREFIX` by arm (control
  arm no longer sees the `run_pipeline` nudge) is an orthogonal fairness
  fix, not an anchor-reliability fix. See
  `benches/swebench/run.py::build_prompt`.

**End-to-end validation against the 3 regression case studies** (with pre-indexed
workspaces, identifier-preserving task strings, post-v1.1 binary):

| Case | Anchor input | Target rank | Notes |
|---|---|---|---|
| sphinx-9711 | `needs_extensions` | **#6** | exact=0, bm25-name=1 hit; target rank-6 due to RRF dilution |
| xarray-7229 | `keep_attrs` (dotted `xr.where` not extracted from prose) | **#7** | bm25-name found `_get_keep_attrs`; real target `where` never anchored |
| sympy-21612 | `parse_latex` | **#1** 🎯 | exact=2 + bm25-name=3; clean win |

**v1.2 closes the remaining gaps on sphinx and xarray.** Four changes, in
priority order — all are small (≤30 lines each), zero breaking risk.

### v1.2.a — Gate BM25-name fallback on exact-miss (PRECISION FIX)

**Why**: v1.1 currently runs BM25-name *unconditionally*, even when exact-name
already returned hits. For common symbol names (e.g., `where` has multiple
exact matches across xarray), BM25-name adds fuzzy decoys (`where_method`,
`sum_where`, `test_where`) that consume anchor slots and dilute the
contribution of the real target in RRF.

**Fix**: only run BM25-name when exact returned 0 hits. Keeps the fuzzy
fallback for shortenings like `needs_extensions` → `verify_needs_extensions`,
but doesn't add noise when exact is already precise.

```rust
// In anchor_candidates, replace the current unconditional fallback with:
for name in &anchors.symbol_names {
    let lookup = name.rsplit('.').next().unwrap_or(name);

    let exact_ids = db
        .symbols_by_exact_name(lookup, ANCHOR_ROWS_PER_NAME)
        .unwrap_or_default();
    let had_exact_hits = !exact_ids.is_empty();

    for id in exact_ids {
        if seen.insert(id) {
            out.push((id, 1.0));
            resolved_exact += 1;
            if out.len() >= limit { break 'outer; }
        }
    }

    // Fuzzy fallback ONLY if exact returned nothing — keeps precision high
    // when the user named a real symbol exactly (multiple symbols with the
    // same name are fine; we already captured them above).
    if !had_exact_hits {
        match search.search_name(lookup, ANCHOR_ROWS_PER_NAME) {
            Ok(hits) => {
                for (id, _) in hits {
                    if seen.insert(id) {
                        out.push((id, 0.9));
                        resolved_bm25 += 1;
                        if out.len() >= limit { break 'outer; }
                    }
                }
            }
            Err(e) => tracing::debug!("name-BM25 fallback failed for {:?}: {}", lookup, e),
        }
    }
}
```

**Test**: for xarray-7229 with task `"fix xr.where keep_attrs overwriting coordinate attributes"`,
extraction should yield `keep_attrs` (and v1.2.b will also yield `xr.where`).
Exact on `keep_attrs` returns 0 → BM25-name runs → returns `_get_keep_attrs` etc.
Exact on `where` returns multiple → BM25-name does NOT run → no `where_method` decoys.

### v1.2.b — Extract dotted calls from prose, not just code blocks

**Why**: The current prose extraction regex `\b[A-Za-z_][A-Za-z0-9_]{3,}\b`
stops at the `.` in `xr.where`. So inline mentions like `"fix xr.where
keep_attrs"` extract only `keep_attrs` (snake_case) — `xr.where` is lost.
Code blocks catch dotted calls via `call_re`, but prose does not.

**Fix**: after the existing prose loop in `extract()`, add a second pass
that matches `identifier.identifier(.identifier)*` anywhere in the query
(not just inside code blocks) and treats each dotted form as an anchor
with the last segment as the lookup key.

```rust
// In anchors.rs::extract, after the existing prose loop, add:
static DOTTED_PROSE_RE: OnceLock<Regex> = OnceLock::new();
let dotted_re = DOTTED_PROSE_RE.get_or_init(|| {
    // Identifier-dotted-identifier chain, min 2 segments
    Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b").unwrap()
});
for m in dotted_re.find_iter(query) {
    let full = m.as_str();
    push(&mut out, &mut seen, full);
    if let Some(last) = full.rsplit('.').next() {
        if last != full && last.len() > 2 {
            push(&mut out, &mut seen, last);
        }
    }
}
```

**Test**: task `"fix xr.where keep_attrs"` → anchors contains `xr.where`,
`where`, `keep_attrs`. Exact lookup on `where` finds `core/computation.py::where`
and `core/duck_array_ops.py::where` directly.

Combined with v1.2.a, xarray-7229 should now put `core/computation.py::where`
in the top 5.

### v1.2.c — Prefer module-level fqn on dotted anchors

**Why**: Even with v1.2.b, `where` has multiple exact matches in xarray —
some at the module level (`xarray/core/computation.py::where`) and some as
class methods (`xarray/core/common.py::DataWithCoords::where`). When the
anchor originated from a **dotted call** like `xr.where(...)`, it's
almost certainly a module-level function call, not a method. Rank
module-level matches above class-method matches.

**Heuristic**: count `::` in the fqn. Module-level = 1 `::`. Class method = 2+.

**Fix**: pass a flag through the anchor extractor indicating which names came
from dotted calls. In `anchor_candidates`, when looking up such a name, sort
the results by `::` count ascending before pushing.

```rust
// Extend Anchors to carry provenance:
pub struct Anchors {
    pub symbol_names: Vec<String>,
    pub module_paths: Vec<String>,
    /// Names that came from a dotted call (e.g. `xr.where`).
    /// For these, we prefer module-level symbols (fqn with 1 `::`) over
    /// class methods (fqn with 2+ `::`) when multiple exact matches exist.
    pub from_dotted_call: HashSet<String>,
}

// In anchor_candidates, when looking up a name in from_dotted_call:
let mut ids = db.symbols_by_exact_name(lookup, ANCHOR_ROWS_PER_NAME * 2)?;
if anchors.from_dotted_call.contains(lookup) {
    // Re-sort: prefer fewer "::" (module-level functions first).
    ids.sort_by_key(|id| {
        graph.get_symbol(*id)
            .map(|s| s.fqn.matches("::").count())
            .unwrap_or(usize::MAX)
    });
    ids.truncate(ANCHOR_ROWS_PER_NAME);
}
```

**Test**: xarray-7229 anchor `where` (from dotted `xr.where`) → `core/computation.py::where`
(1 `::`) ranks above `core/common.py::DataWithCoords::where` (2 `::`).

### v1.2.d — Tune RRF k for anchor list (TUNING)

**Why**: anchors currently fuse into RRF with the global `RRF_K = 60`. Rank-1
anchor hit contributes `1/61 ≈ 0.0164`, which is exactly the same as rank-1
BM25. If BM25 and ANN both rank the wrong file at #1 (as in sphinx-9711 where
`bump_version.py::bump_version` wins BM25 + ANN), their combined contribution
(~0.032) beats a lone anchor hit (~0.016). The target gets pushed to rank 6.

**Fix**: use a smaller k (stronger boost) for the anchor list only:

```rust
// In ranking.rs:
pub(crate) const ANCHOR_RRF_K: f32 = 15.0;  // was effectively 60

// In build_context_capsule, do per-list RRF instead of shared k:
let anchor_rrf = rrf_single(&anchor_results, ANCHOR_RRF_K);
let bm25_rrf = rrf_single(&bm25_results, RRF_K);
let graph_rrf = rrf_single(&graph_results, RRF_K);
let ann_rrf = rrf_single(&ann_results, RRF_K);
let merged = sum_rrf_tables(&[anchor_rrf, bm25_rrf, graph_rrf, ann_rrf]);
```

With `k=15`: rank-1 anchor contributes `1/16 ≈ 0.0625` — ~4× stronger than a
rank-1 BM25 hit. Enough to overcome the BM25+ANN combo on sphinx-9711.

**Risk**: over-boost — an anchor hit that's actually not the right file gets
pushed past relevant BM25/ANN hits. Mitigate with v1.2.a (don't dilute) and by
capping the anchor list size (`ANCHOR_CANDIDATES = 20` already).

**Test**: re-run sphinx-9711 after v1.2.a + v1.2.d. Expected:
`verify_needs_extensions` rank 6 → rank 1–3.

### Summary of v1.2 impact on the three regressions

| Case | After v1.2.a | After v1.2.a+b | After v1.2.a+b+c | After v1.2.a+b+c+d |
|---|---|---|---|---|
| sphinx-9711 (`needs_extensions`) | rank 6 (unchanged — exact was 0, fallback still fires) | same | same | **rank 1–3** (k=15 boosts anchor) |
| xarray-7229 (`xr.where`, `keep_attrs`) | rank 7 (no decoys) | `where` now extracted, rank improves | `core/computation.py::where` prioritized over methods | **rank 1–3** |
| sympy-21612 (`parse_latex`) | rank 1 (unchanged — clean win) | same | same | rank 1 |

### Implementation order

1. **v1.2.a** (gate fallback) — single `if !had_exact_hits` check around the existing fallback block. 3-line change + existing tests still pass.
2. **v1.2.b** (dotted prose) — add the regex + loop in `extract()`. Write 2 new unit tests.
3. **v1.2.d** (RRF tuning) — add `ANCHOR_RRF_K`, split the RRF merge. Validate against sphinx-9711.
4. **v1.2.c** (module-vs-method) — extends `Anchors` struct. Larger scope than the others; do it last, only if xarray still misses after a+b+d.

### Validation after v1.2

Re-run all three cases using:
```bash
# Assumes /tmp/sphinx-repro, /tmp/xarray-repro, /tmp/sympy-repro are pre-indexed
# with the v1.2 binary.
for ws in sphinx-repro xarray-repro sympy-repro; do
    python3 /Users/sriram/projects/codesurgeon/scripts/test_anchors_on_regression.py /tmp/$ws
done
```

Success = all three targets in top 3 pivots.

---

## 🚨 READ THIS — v1.3 pending improvement (found during v1.2 validation)

v1.2 was validated against **six** SWE-bench tasks in the with-arm: the three
regression cases (sphinx-9711, xarray-7229, sympy-21612) plus three of the
largest-token-overshoot tasks from 29c (sympy-19040, sympy-21379, matplotlib-26208).

**Five of six improved or held steady. One regressed: `matplotlib-26208`.**

### The matplotlib-26208 regression (evidence)

| | v1.0 (BM25+graph only, no embeddings) | v1.2 (BM25+graph+embeddings+anchors) |
|---|---:|---:|
| Walltime | 364s | **479s (+32%)** |
| Output tokens | 23,038 | 29,235 (+27%) |
| Cost | $1.37 | **$1.68 (+22%)** |
| Tool calls | (not captured) | 62 (many Grep+Read after capsule) |

**Agent's v1.2 task string** (identifier-preserving, per the prompt update):
```
"fix dataLims get replaced by inf for charts with twinx if ax1 is a stackplot,
 stackplot update_datalim"
```

Anchor extraction pulled `dataLims` (CamelCase) and `update_datalim` (snake_case).
Both passed the shape filter. But:

- Top pivots were `parasite_axes.py::HostAxesBase::twinx`,
  `pyplot.py::twinx`, `axis.py::Axis::_update_axisinfo`, `axis.py::Axis::get_tightbbox`...
- **Target `lib/matplotlib/axes/_base.py::_AxesBase::update_datalim` was NOT in top 8**.
- Agent chased the high-ranked pivots, found none were the bug, then did
  extensive Grep/Read to locate the real fix site in `_base.py`.

### Why v1.2 made this worse, not better

The anchor `update_datalim` triggered BM25-name fallback (since exact-name
probably had multiple hits across matplotlib's many Axes subclasses). **That
fallback returned many fuzzy matches**. With the aggressive `ANCHOR_RRF_K=15`
boost applied uniformly, each of those decoys ranked high in the fused list.
The real target `_AxesBase::update_datalim` was one of several hits but got
outweighed by the collective of similarly-named methods.

In v1.0 (pure BM25+graph), pure accidental scoring happened to land `_base.py`
higher for the original, shorter task string. The richer v1.2 pipeline
steered the agent toward public-facing symbols (`twinx`, `stackplot`) that
match the symptom but not the bug site.

This generalises a pattern noted in the v1 motivation section:
> Ranker appears biased toward public symbols / top-level APIs.
v1.2's anchor boost **amplifies** that bias when the user's task mentions
public APIs (symptoms) rather than the internal function that holds the bug.

### v1.3 fix: adaptive anchor boost based on hit precision

Currently every anchor hit gets `ANCHOR_RRF_K=15` regardless of how many
matches it resolved. Proposed change: **dial down the boost as the number
of matches grows**. High precision (1–3 hits) keeps the aggressive boost;
low precision (10+ hits) falls back to baseline `RRF_K=60`.

```rust
// In engine.rs, compute per-lookup boost at anchor_candidates time.
// Each anchor hit carries its own effective-k, and rrf_merge_ks uses the
// per-hit k rather than one constant for the whole list.

fn effective_anchor_k(n_hits: usize) -> f32 {
    match n_hits {
        1 => 15.0,       // precise — maximum boost
        2..=3 => 20.0,   // mostly precise
        4..=8 => 35.0,   // getting fuzzy — half-boost
        _ => 60.0,       // bulk match — no extra boost beyond baseline
    }
}
```

This would neutralise the matplotlib regression: `update_datalim` with many
matches gets `k=60` (same as BM25), so its RRF contribution matches BM25's
— the real target isn't pushed around. Precise anchors like `parse_latex`
(2 exact hits in sympy-21612) still get `k=20` and remain dominant.

### Implementation sketch

Option A — per-hit k (cleanest, small):
```rust
// Anchors enriched with a per-hit k. anchor_candidates returns
// Vec<(symbol_id, score, k)> instead of Vec<(symbol_id, score)>.
// Threads through rrf_merge_ks.

pub fn anchor_candidates(&self, query: &str, limit: usize) -> Vec<(u64, f32, f32)> {
    let anchors = crate::anchors::extract(query);
    let mut per_lookup_hits: HashMap<String, Vec<u64>> = HashMap::new();
    // ... same exact-then-BM25-fallback loop, but collect hits keyed by lookup token ...
    let mut out = Vec::new();
    for (lookup, ids) in per_lookup_hits {
        let k = effective_anchor_k(ids.len());
        for id in ids { out.push((id, 1.0, k)); }
    }
    out
}
```

Option B — precision cutoff only (simpler, coarser):
```rust
// Drop anchor entirely if BM25-name fallback exceeds a threshold.
// Keeps current single-k rrf path but trades off recall for precision.
if !had_exact_hits {
    let hits = search.search_name(lookup, 20)?;
    if hits.len() <= 3 {
        // Precise — inject into anchor pool
        for (id, _) in hits { out.push((id, 0.9)); }
    } else {
        // Too fuzzy — skip the fallback entirely.
        // Keeps RRF fusion between BM25/ANN/graph clean.
        tracing::debug!("anchor {} fuzzy-skipped ({} bm25-name hits)", lookup, hits.len());
    }
}
```

Recommend Option B for v1.3 (three lines, no signature change). Revisit
Option A if the precision cutoff is too blunt.

### Validation

Re-run the 6-task with-arm validation after v1.3 lands. Target outcome:

| Task | v1.2 walltime | v1.3 target |
|---|---:|---|
| sphinx-9711 | 25.3s | ≤ v1.2 (anchor still precise, 1 fallback hit) |
| xarray-7229 | 327s | ≤ v1.2 (anchor precise, 2 exact + few fuzzy) |
| sympy-21612 | TBD | ≤ v1.2 (anchor precise, exact-only) |
| sympy-19040 | TBD | — |
| sympy-21379 | TBD | — |
| **matplotlib-26208** | **479s** | **≤ 364s (v1.0 baseline)** ← fix target |

---

## 🚨 HISTORICAL — v1.5 adaptive pivot cap (DEPRECATED, replaced by v1.6)

**Premise**: v1 – v1.3 shipped high-quality anchor retrieval but kept the
pivot count hard-coded at `max_pivots = 8`. The bottom 5 pivots on
clean-anchor queries (e.g. sympy-21612: `parse_latex` → 3 exact hits across
3 distinct files, zero fuzzy fallback) were BM25/ANN residue that added
tokens without load-bearing context.

**Tried**: surface confidence stats from `anchor_candidates` (`AnchorStats`)
and pick an adaptive pivot cap.

```rust
let effective_pivots = match &astats {
    s if s.resolved_exact >= 3
        && s.resolved_bm25_name == 0
        && s.distinct_source_files >= 2 => 3,                         // CLEAN
    s if s.resolved_exact >= 1 || s.resolved_bm25_name > 0 =>
        (self.config.max_pivots * 5 / 8).max(5),                      // MEDIUM
    _ => self.config.max_pivots,                                      // DEFAULT
};
```

**Why it failed**: the cap truncates pivots by RRF rank, which is
graph-centrality-biased. On sympy-21379 the agent named both
`PolynomialError` (low-centrality bug site) and `symbols`/`exp`/`Piecewise`
(high-centrality public APIs). Centrality won the top-3 slots, the bug site
fell out, and the agent burned $3.04 / 819s thrashing for the missing code.

**Lesson kept by v1.6**: anchors encode user intent, not general importance —
never rank or cap them by centrality. Pin them by file diversity instead.

---

## 🚨 HISTORICAL — v1.1 post-implementation finding (addressed, kept for reference)

**v1 landed** (`anchors.rs` + `anchor_candidates` in `engine.rs`). End-to-end test
against `sphinx-doc__sphinx-9711` with the pre-indexed sphinx workspace confirmed:

- **Extraction works**: debug log showed `anchors: 1 extracted, 0 resolved` for the query
  `"fix needs_extensions version comparison using strings instead of version tuples"`.
- **Lookup too strict**: the real symbol is `sphinx/extension.py::verify_needs_extensions`,
  but the user prose says `needs_extensions`. Exact-name DB lookup fails on the mismatch.

### What v1.1 must add: name-field BM25 lookup as a second resolution path

BM25 already tokenises identifiers on `_` via Tantivy's default tokenizer —
`verify_needs_extensions` is indexed as `{verify, needs, extensions}`.
A BM25 query for `"needs_extensions"` will tokenise to `{needs, extensions}`
and score `verify_needs_extensions` very high **if the search is restricted
to the `name` field**.

The reason the full-pipeline BM25 misses the target is **signal dilution from
the rest of the prose query**. Evidence — two CLI `search` calls against the
same sphinx corpus:

| Query | Tantivy BM25 top-1 |
|---|---|
| `"needs_extensions version"` (2 tokens) | ✅ `sphinx/extension.py::verify_needs_extensions` |
| `"fix needs_extensions version comparison using strings instead of version tuples"` (10 tokens) | ❌ `utils/bump_version.py::bump_version` |

The extra tokens (`fix`, `comparison`, `using`, `strings`, `instead`, `tuples`)
all match heavily against `bump_version.py`'s long body (body field contains
"version" 30+ times across many related functions). `verify_needs_extensions`
has a ~10-line body with `needs` and `extensions` each appearing once — when
BM25 sums across `{name, signature, docstring, body}`, the noise wins.

### The fix in one paragraph

In `anchor_candidates`, after the exact-name DB lookup fails, run a **second**
lookup that is a Tantivy BM25 query restricted to the `name` field only,
with just the anchor token as the query. This bypasses body/docstring/signature
noise completely. Tokenisation on `_` makes `needs_extensions` match
`verify_needs_extensions` naturally.

### Implementation (≈20 net lines)

Add to `crates/cs-core/src/search.rs` alongside the existing `search()` method:

```rust
/// BM25 restricted to the symbol `name` field.
///
/// Used by the anchor pipeline to resolve a short identifier (e.g.
/// `needs_extensions`) against symbol names (e.g. `verify_needs_extensions`)
/// without the noise of body/docstring/signature matches that would dominate
/// a full-field query. The `name` field uses Tantivy's default tokenizer
/// which splits on `_`, so `needs_extensions` → {needs, extensions} matches
/// any symbol whose name contains both tokens.
pub fn search_name(&self, query: &str, limit: usize) -> Result<Vec<(u64, f32)>> {
    let reader = self
        .index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()?;
    let searcher = reader.searcher();
    let qp = QueryParser::for_index(&self.index, vec![self.schema.f_name]);
    let parsed = qp
        .parse_query(query)
        .or_else(|_| qp.parse_query(&escape_for_tantivy(query)))
        .unwrap_or_else(|_| qp.parse_query("*").expect("wildcard is always parseable"));
    let top_docs = searcher.search(&parsed, &TopDocs::with_limit(limit))?;
    let mut results = Vec::new();
    for (score, addr) in top_docs {
        let doc: tantivy::TantivyDocument = searcher.doc(addr)?;
        if let Some(id_val) = doc.get_first(self.schema.f_id) {
            if let Some(id) = id_val.as_u64() {
                results.push((id, score));
            }
        }
    }
    Ok(results)
}
```

Modify `anchor_candidates` in `crates/cs-core/src/engine.rs` to use this as
a fallback when exact-name DB lookup returns zero hits:

```rust
fn anchor_candidates(&self, query: &str, limit: usize) -> Vec<(u64, f32)> {
    let anchors = crate::anchors::extract(query);
    if anchors.symbol_names.is_empty() { return vec![]; }

    let mut out: Vec<(u64, f32)> = Vec::with_capacity(limit);
    let mut seen: HashSet<u64> = HashSet::new();
    let db = self.db.lock();
    let search = self.search.lock();

    let mut extracted = 0usize;
    let mut resolved_exact = 0usize;
    let mut resolved_bm25 = 0usize;

    for name in &anchors.symbol_names {
        extracted += 1;
        let lookup = name.rsplit('.').next().unwrap_or(name);

        // 1) Exact name match — strongest signal. Highest score.
        if let Ok(ids) = db.symbols_by_exact_name(lookup, ANCHOR_ROWS_PER_NAME) {
            for id in ids {
                if seen.insert(id) {
                    out.push((id, 1.0));
                    resolved_exact += 1;
                    if out.len() >= limit { break; }
                }
            }
        }
        if out.len() >= limit { break; }

        // 2) Name-field BM25 fallback — catches `needs_extensions` → `verify_needs_extensions`.
        // Score slightly lower than exact so RRF preserves the ordering.
        if let Ok(hits) = search.search_name(lookup, ANCHOR_ROWS_PER_NAME) {
            for (id, _) in hits {
                if seen.insert(id) {
                    out.push((id, 0.9));
                    resolved_bm25 += 1;
                    if out.len() >= limit { break; }
                }
            }
        }
        if out.len() >= limit { break; }
    }

    tracing::debug!(
        "anchors: {} extracted, {} exact, {} bm25-name (total {})",
        extracted, resolved_exact, resolved_bm25, out.len()
    );
    out
}
```

### Validation checklist

After landing v1.1, re-run the sphinx-9711 validation test. The command:

```bash
# MCP server on a persistent connection (subprocess.Popen) against a pre-indexed sphinx repo
# at /tmp/sphinx-repro (clone → base_commit 81a4fd973d... → codesurgeon index --workspace ...).
# Send run_pipeline with task="fix needs_extensions version comparison using strings instead of version tuples".
```

Success = `sphinx/extension.py::verify_needs_extensions` appears in the top 3 pivots
(currently it's not in the top 8 even with v1 anchors).

Add a unit test in `search.rs` that seeds `{verify_needs_extensions, bump_version, parse_version}`
and asserts `search_name("needs_extensions")` returns `verify_needs_extensions` at rank 1.

### Why this isn't covered by existing BM25

The engine's existing `search()` queries `[f_name, f_signature, f_docstring, f_body]`
as a union. When prose is 10 tokens and only 1 is the "anchor," the other 9
tokens dominate the sum. `search_name` is a targeted escape hatch for the
specific case where an anchor has been extracted from the query. Keep both;
don't weaken the general search.

### Why not fuzzy SQL LIKE (`WHERE name LIKE '%needs_extensions%'`)

Considered and rejected: `LIKE '%X%'` is O(table scan), can't use an index,
and returns unscored results. The Tantivy name-field query is O(log n) via
the inverted index, applies BM25 scoring naturally, and reuses the
tokeniser we already depend on.

---

## The problem, with evidence

### Case study 1 — `sphinx-doc/sphinx-9711` (prose-mentioned name)

Problem statement (first line):
> "`needs_extensions` checks versions using strings"

The task literally names the function `needs_extensions` in the title.
Running `run_pipeline` with this task:

| Ranking signal | Top pivot |
|---|---|
| BM25 only | `sphinx/domains/cfamily.py::cfamily` (matched "extensions" as English plural noun → C++ file-extension parser) |
| BM25 + embeddings | `utils/bump_version.py::bump_version` (matched "version comparison using strings") |
| **Ground truth** | **`sphinx/extension.py::needs_extensions`** |

Both rankings miss. The target function has a ~10-line body and sparse docstring — semantically under-resourced compared to `bump_version.py` which is a full release script dedicated to version manipulation. BM25 tokenises `needs_extensions` into `needs` + `extensions` and scores each independently, neither hitting the target file strongly.

Result: with-arm walltime 41.6s (embeddings on) vs 30.1s (embeddings off) vs ~16s baseline without codesurgeon. The capsule is net-negative.

### Case study 2 — `pydata/xarray-7229` (code-snippet API call)

Problem statement includes a reproducing example:
```python
import xarray as xr
ds = xr.tutorial.load_dataset("air_temperature")
xr.where(True, ds.air, ds.air, keep_attrs=True).time.attrs
```

The task calls `xr.where(...)`. The fix is in `xarray/core/computation.py` where `where()` is defined.

| Ranking signal | Top pivots |
|---|---|
| BM25 + embeddings | `pydap_.py::_fix_attributes`, `conventions.py::_update_bounds_attributes` (both matched "attributes" heavily) |
| **Ground truth** | **`xarray/core/computation.py::where`** |

The ranker latched onto the noun "attributes" (appearing 8+ times) and ignored that `where` is the specific function the user called. `computation.py` wasn't even in the top 5.

Result: with-arm walltime 528s vs without-arm 194s — **+172% walltime** regression.

### Case study 3 — `sympy/sympy-21612` (path-segment semantics)

Problem statement:
> "Latex parsing of fractions yields wrong expression"
>
> ```python
> from sympy.parsing.latex import parse_latex
> parse_latex("\\frac{\\frac{a^3+b}{c}}{\\frac{1}{c^2}}")
> ```

The task calls `parse_latex(...)` from `sympy.parsing.latex`. The fix is in `sympy/parsing/latex/_parse_latex_antlr.py`.

| Ranking signal | Top pivot |
|---|---|
| BM25 + embeddings | `sympy/printing/latex.py::latex` (LaTeX **output** printer) |
| **Ground truth** | **`sympy/parsing/latex/_parse_latex_antlr.py`** (LaTeX **input** parser) |

`parsing/` vs `printing/` are opposite-direction path segments sharing the same domain word. Pure BM25 + embeddings can't disambiguate. But the task literally imports from `sympy.parsing.latex`, and calls `parse_latex` — those are ground-truth anchors.

Result: 1133s vs 336s — **+237% walltime** regression.

---

## Proposed solution

Add a new retrieval source, **Explicit Anchors**, that extracts symbol names and module paths from the problem statement and boosts any indexed symbol that matches. This source runs in parallel to BM25, semantic, and graph retrieval, and feeds into the same RRF fusion.

### Two extraction modes (both fire on every query)

#### (a) Prose-mentioned symbol names

Tokenize the problem statement and cross-reference every identifier-shaped token against the symbol-name FTS index. Matches should be exact on the `name` field (not `fqn`, not `signature`).

```rust
// Identifier pattern: snake_case or camelCase, min 4 chars to avoid noise
// Avoid matching stop words like "with", "this", "that", "when", "where"
let re = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b").unwrap();
let candidates: HashSet<String> = re.find_iter(query)
    .map(|m| m.as_str().to_string())
    .filter(|tok| !STOP_WORDS.contains(tok.as_str()))
    .collect();

// For each candidate, check if an indexed symbol has that exact name.
// Use a direct name index (separate from the full-text Tantivy index) to avoid
// BM25 scoring — we want exact match or nothing.
let mut anchors: Vec<(u64, f32)> = vec![];
for tok in &candidates {
    for symbol_id in db.symbols_by_exact_name(tok)? {
        anchors.push((symbol_id, 1.0));  // flat score — position in RRF is what matters
    }
}
```

For `sphinx-9711`, this extracts `needs_extensions` and directly looks up the symbol by name. A single hit → `sphinx/extension.py::needs_extensions` gets injected at rank 1 in the anchor list. RRF merge promotes it into the capsule top-3.

#### (b) Code-snippet API calls

Parse fenced code blocks and extract function/method calls using a light tokenizer — no full Python parser needed, just regex for `identifier.identifier(` and `ClassName(`.

```rust
fn extract_api_calls(query: &str) -> Vec<String> {
    // Find fenced code blocks (```lang ... ``` or indented 4 spaces)
    let code_blocks = extract_code_blocks(query);
    let mut calls = vec![];
    // Match things like `xr.where(`, `np.array(`, `MyClass(`,
    // also handle multi-level: `a.b.c(`
    let call_re = Regex::new(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(").unwrap();
    for block in code_blocks {
        for cap in call_re.captures_iter(&block) {
            let full = cap.get(1).unwrap().as_str();
            // Split dotted path and add each segment as an anchor candidate.
            // xr.where → ["xr.where", "where"]
            // urllib.request.urlopen → ["urllib.request.urlopen", "request.urlopen", "urlopen"]
            calls.push(full.to_string());
            if let Some(last) = full.rsplit('.').next() {
                if last != full {
                    calls.push(last.to_string());
                }
            }
        }
    }
    calls
}
```

For `xarray-7229`, this extracts `xr.where` → `["xr.where", "where"]`. Looking up `where` by exact name finds `xarray/core/computation.py::where` (among others). Rank 1 anchor match.

For `sympy-21612`, this extracts `parse_latex` → single anchor → `sympy/parsing/latex/_parse_latex_antlr.py::parse_latex` ranks 1. Path-segment disambiguation is a free side effect.

#### (c) Bonus — import statements

Also cheap to extract:
```python
from sympy.parsing.latex import parse_latex
import xarray as xr
```

The `from X.Y import Z` statement is extremely informative — it directly names both a module path AND a symbol. Penalise files whose path doesn't share any segment with the imported module path, and boost those that do.

```rust
// Extract "from a.b.c import foo, bar" and "import a.b.c as x"
// Match against file paths: prefer files under a/b/c/ or whose basename is c.py
```

---

## Integration point

The minimal change is one new candidate source added to the RRF merge in `build_context_capsule` (around `engine.rs:2167`):

```rust
// New source: explicit anchors from the query (exact symbol-name matches
// from prose and from code-snippet API calls).
let anchor_results = self.anchor_candidates(query, ANCHOR_CANDIDATES);

#[cfg(feature = "embeddings")]
let mut search_results = {
    let ann_results = self.ann_candidates(query, ANN_CANDIDATES);
    rrf_merge(&[
        &bm25_results,
        &graph_results,
        &ann_results,
        &anchor_results,  // ← new
    ], RRF_K)
};
```

### Key design decisions

1. **Exact match only, no fuzzy.** We already have BM25 for fuzzy. Anchors are meant to be unambiguous ground truth; if a token doesn't map to an exact symbol name, drop it.

2. **Flat scoring.** All anchor hits get score 1.0. RRF handles rank-based fusion — anchor hits ending up at positions 1..N in the anchor list is what matters, not their raw scores.

3. **Boost anchor contribution in RRF.** Optionally, give the anchor list a stronger k constant (say `k=30` vs the usual `k=60`), so anchor rank 1 contributes `1/31 = 0.032` to the fused score vs `1/61 = 0.016` for BM25 rank 1. This is a tuning knob; start without it and add if the benchmark demands.

4. **Stop words matter.** Without a stop list (`with`, `where`, `when`, `this`, `that`, `size`, `type`, `name`, `list`, `dict`, `len`, `str`, `int`, ...), every English sentence will produce dozens of false-positive matches. Curate a small list of English common words that are also common programming identifiers.

5. **Never let anchors dominate.** If a task contains no extractable identifiers (pure prose bug report), anchors return empty and the pipeline degrades to current behaviour. If anchors return 50 matches, cap the list at `ANCHOR_CANDIDATES = 20` before RRF merge so it doesn't drown out BM25.

6. **Respect `file_hint` even more strongly with anchors.** If the user already narrowed by file, intersect anchor matches with that file's symbols before RRF.

---

## Implementation sketch

### 1. New module: `crates/cs-core/src/anchors.rs`

```rust
//! Explicit symbol-name anchor extraction for ranking.
//!
//! Extracts identifiers from the task query that match exact symbol names in
//! the index. Three sources:
//!   1. Prose tokens (top-level words that look like identifiers)
//!   2. Function/method calls in fenced code blocks (`foo.bar(...)`)
//!   3. `from X.Y import Z` / `import X.Y as Z` statements
//!
//! All matches are flat-scored — ranking within the anchor list is
//! "extraction order" which roughly correlates with position in the query.
//! RRF fusion handles blending with BM25/ANN/graph.

use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// English stop words that are also common programming identifiers.
/// Used to filter prose tokens; code-snippet extraction ignores this list.
const STOP_WORDS: &[&str] = &[
    "with", "when", "where", "this", "that", "from", "into", "have", "been",
    "just", "like", "make", "many", "more", "most", "must", "only", "over",
    "such", "than", "then", "they", "were", "will", "into", "upon",
    // common type/collection names we don't want to anchor on
    "none", "true", "false", "int", "str", "dict", "list", "set", "tuple",
    "bool", "float", "bytes", "type", "kind", "name", "value", "values",
    "size", "length", "index", "data", "item", "items", "path", "file",
    "files", "line", "lines", "test", "tests", "error", "errors", "cause",
    "fail", "pass", "call", "calls", "version", "using", "should", "result",
];

fn identifier_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b").unwrap())
}

fn call_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Matches `a.b.c(`, `Foo(`, `self.bar(`
    RE.get_or_init(|| {
        Regex::new(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(").unwrap()
    })
}

fn import_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?m)^\s*(?:from\s+([\w.]+)\s+import\s+([\w, ]+)|import\s+([\w.]+)(?:\s+as\s+(\w+))?)").unwrap()
    })
}

fn code_block_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"```[\w]*\n([\s\S]*?)```").unwrap())
}

/// Extracted anchors, in order of discovery.
#[derive(Debug, Default)]
pub struct Anchors {
    /// Symbol names to try looking up exactly.
    pub symbol_names: Vec<String>,
    /// Module paths (from import statements).
    pub module_paths: Vec<String>,
}

pub fn extract(query: &str) -> Anchors {
    let mut out = Anchors::default();
    let mut seen: HashSet<String> = HashSet::new();

    // 1. Code-block API calls — highest priority
    for block_cap in code_block_re().captures_iter(query) {
        let block = &block_cap[1];

        // Import statements inside code
        for imp in import_re().captures_iter(block) {
            if let Some(m) = imp.get(1) { out.module_paths.push(m.as_str().to_string()); }
            if let Some(m) = imp.get(3) { out.module_paths.push(m.as_str().to_string()); }
            // Imported symbol names
            if let Some(names) = imp.get(2) {
                for n in names.as_str().split(',') {
                    let n = n.trim();
                    if !n.is_empty() && seen.insert(n.to_string()) {
                        out.symbol_names.push(n.to_string());
                    }
                }
            }
        }

        // Function/method calls
        for cap in call_re().captures_iter(block) {
            let full = cap[1].to_string();
            if seen.insert(full.clone()) {
                out.symbol_names.push(full.clone());
            }
            // Also add the last segment: `xr.where` → `where`
            if let Some(last) = full.rsplit('.').next() {
                if last.len() > 3 && seen.insert(last.to_string()) {
                    out.symbol_names.push(last.to_string());
                }
            }
        }
    }

    // 2. Prose identifiers — lower priority, filtered by stop words
    for m in identifier_re().find_iter(query) {
        let tok = m.as_str();
        let lower = tok.to_lowercase();
        if STOP_WORDS.contains(&lower.as_str()) { continue; }
        // Require either underscore or camelCase — filters out English words
        let has_snake = tok.contains('_');
        let has_camel = tok.chars().any(|c| c.is_uppercase()) &&
                        tok.chars().any(|c| c.is_lowercase());
        if !has_snake && !has_camel { continue; }
        if seen.insert(tok.to_string()) {
            out.symbol_names.push(tok.to_string());
        }
    }

    out
}
```

### 2. New DB method in `crates/cs-core/src/db.rs`

```rust
impl Db {
    /// Look up symbol IDs by exact name (not FQN). Used for anchor retrieval.
    /// Returns at most `limit` matches per name.
    pub fn symbols_by_exact_name(&self, name: &str, limit: usize) -> Result<Vec<u64>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT id FROM symbols WHERE name = ?1 LIMIT ?2"
        )?;
        let rows = stmt.query_map((name, limit as i64), |r| r.get::<_, i64>(0).map(|v| v as u64))?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(Into::into)
    }
}
```

Note: `symbols.name` is already indexed in SQLite (see `crates/cs-core/src/db.rs` schema).

### 3. New engine method in `crates/cs-core/src/engine.rs`

```rust
const ANCHOR_CANDIDATES: usize = 20;

impl CoreEngine {
    /// Returns anchor candidates — exact symbol-name hits from explicit
    /// identifiers in the query. Score is flat 1.0 per hit; order reflects
    /// extraction priority (code-snippet calls first, then prose tokens).
    fn anchor_candidates(&self, query: &str, limit: usize) -> Vec<(u64, f32)> {
        let anchors = crate::anchors::extract(query);
        let mut out: Vec<(u64, f32)> = Vec::with_capacity(limit);
        let db = self.db.lock();
        let mut seen: HashSet<u64> = HashSet::new();
        for name in &anchors.symbol_names {
            // For "xr.where" try "where" (last segment); for "needs_extensions" try as-is
            let lookup = name.rsplit('.').next().unwrap_or(name);
            if let Ok(ids) = db.symbols_by_exact_name(lookup, 5) {
                for id in ids {
                    if seen.insert(id) {
                        out.push((id, 1.0));
                        if out.len() >= limit { return out; }
                    }
                }
            }
        }
        out
    }
}
```

### 4. Integration in `build_context_capsule`

```rust
// In engine.rs:2160 (inside build_context_capsule, before the RRF merge):
let anchor_results = self.anchor_candidates(query, ANCHOR_CANDIDATES);

#[cfg(feature = "embeddings")]
let mut search_results = {
    let ann_results = self.ann_candidates(query, ANN_CANDIDATES);
    rrf_merge(
        &[&bm25_results, &graph_results, &ann_results, &anchor_results],
        RRF_K,
    )
};
#[cfg(not(feature = "embeddings"))]
let mut search_results = rrf_merge(
    &[&bm25_results, &graph_results, &anchor_results],
    RRF_K,
);
```

### 5. Add ranking constant in `crates/cs-core/src/ranking.rs`

```rust
/// Explicit anchor candidate pool size. Anchors are exact symbol-name matches
/// extracted from the query (either prose identifiers or code-snippet API calls).
/// Small because we want high precision, not recall.
pub(crate) const ANCHOR_CANDIDATES: usize = 20;
```

---

## Tests

### Unit tests in `crates/cs-core/src/anchors.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_prose_snake_case() {
        let a = extract("The `needs_extensions` check is handy for verifying versions");
        assert!(a.symbol_names.contains(&"needs_extensions".to_string()));
    }

    #[test]
    fn extract_code_block_api_call() {
        let q = "```python\nxr.where(True, ds.air, ds.air, keep_attrs=True)\n```";
        let a = extract(q);
        assert!(a.symbol_names.contains(&"xr.where".to_string()));
        assert!(a.symbol_names.contains(&"where".to_string()));
    }

    #[test]
    fn extract_import_statement() {
        let q = "```python\nfrom sympy.parsing.latex import parse_latex\nparse_latex('x')\n```";
        let a = extract(q);
        assert!(a.symbol_names.contains(&"parse_latex".to_string()));
        assert!(a.module_paths.contains(&"sympy.parsing.latex".to_string()));
    }

    #[test]
    fn stop_words_filtered() {
        let a = extract("This is a simple test with some common words");
        assert!(a.symbol_names.is_empty()); // nothing looks like a symbol
    }

    #[test]
    fn camelcase_accepted() {
        let a = extract("The BuildEnvironment class handles the case");
        assert!(a.symbol_names.contains(&"BuildEnvironment".to_string()));
    }

    #[test]
    fn plain_english_rejected() {
        let a = extract("The function should return an empty dict for fields");
        // "function", "should", "return", "empty", "fields" — all plain English
        // None should survive: "function" is in stop list (or rejected as no _ or camelCase)
        // "BuildEnvironment" would pass. "fields" is plain lowercase, gets rejected.
        assert!(!a.symbol_names.iter().any(|s| s == "function"));
        assert!(!a.symbol_names.iter().any(|s| s == "fields"));
    }
}
```

### Integration test: verify the three regression tasks now hit

Add a test that seeds a tiny in-memory corpus with symbols named `needs_extensions`, `where`, `parse_latex`, and the three regression-task queries, and asserts `anchor_candidates` surfaces the right symbol for each.

### Benchmark validation

After implementing:

```bash
# Just re-run the 3 regression case studies; walltime should drop back to <60s each
python3 benches/swebench/run.py \
  --instance-ids sphinx-doc__sphinx-9711,pydata__xarray-7229,sympy__sympy-21612 \
  --arms with --max-budget-usd 3.00 --timeout 300 --clean
```

Success criteria:
- `sphinx-9711`: capsule contains `sphinx/extension.py::needs_extensions` in top-5 pivots
- `xarray-7229`: capsule contains `xarray/core/computation.py::where` in top-5 pivots
- `sympy-21612`: capsule contains `sympy/parsing/latex/_parse_latex_antlr.py::parse_latex` in top-5 pivots
- Walltime for each: ≤ without-arm baseline (16s, 194s, 336s respectively)

---

## Rollout

1. **Feature-flag at engine level**: gate behind `EngineConfig::anchor_retrieval_enabled` (default `true`). Lets us disable for debugging or A/B testing.
2. **Log anchor hits at `debug` level**: `tracing::debug!("anchors: {} extracted, {} resolved", extracted, resolved);`. Helps diagnose when anchors fire vs not.
3. **Track in stats**: add `anchor_hits` column to `query_log` so we can measure how often anchors contributed after the fact.
4. **Update `docs/ranking.md`** with the new Stage 1 source and the parameter `ANCHOR_CANDIDATES`.

---

## Landed follow-ups

- **Reverse-edge expansion from error types** — landed via issue #67. When an
  anchor resolves to an exception/error/warning type definition, the capsule
  now walks its callers/raisers backward up to 3 hops (fan-out 5 per hop) and
  fuses those candidates into RRF with `k = 30`. See `docs/ranking.md` Stage
  1d for parameters and `ranking.rs:reverse_expand_from_anchors` for the
  walk. Addresses symptom-anchored bug reports where the user names the
  exception but the fix site is only reachable through the raise chain
  (motivating case: sympy-21379, `PolynomialError ← parallel_poly_from_expr
  ← gcd ← Mod.eval`).

## Out of scope (file follow-ups separately)

- **Path-segment scoring for antonym segments** (`parsing/` vs `printing/`). Anchors solve sympy-21612 directly, but path-segment scoring would catch the generalized case where no API call is quoted.
- **Short-body function floor** — a symbol whose body is < N tokens should get a bonus based on exact name match to a query token. Overlaps partly with anchors but useful when the user describes the bug without naming the function.

---

## File tree of changes

```
crates/cs-core/src/
├── anchors.rs          # NEW ~150 lines, pure function + tests
├── engine.rs           # ~20 lines added to build_context_capsule + anchor_candidates method
├── db.rs               # ~10 lines for symbols_by_exact_name
├── ranking.rs          # 3 lines — new constant
└── lib.rs              # pub mod anchors;

docs/
└── ranking.md          # update Stage 1 diagram + parameter table

crates/cs-core/tests/
└── ranking_anchors.rs  # NEW integration test seeded with the 3 regression cases
```

Total net change: ~250 lines of Rust + tests.

---

## Open questions for the implementing agent

1. Should we re-tokenize on the go's side or reuse Tantivy's tokenizer? Probably reuse for consistency with BM25 term matching.
2. Is the stop-word list language-aware? Current list is English-only. Revisit if we support non-English problem statements.
3. Should `ANCHOR_CANDIDATES` be per-intent (structural intent might not benefit)? Start uniform, tune after benchmark.
4. How to avoid double-counting when an anchor hit is *also* in BM25? RRF handles this correctly — agreement between sources amplifies the candidate, which is what we want.
