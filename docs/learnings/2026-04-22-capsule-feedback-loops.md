# Learnings — capsule feedback loops and pivot-stub leakage

**Date**: 2026-04-22
**Context**: SWE-bench sympy-21379 debug session that landed commits `8684552`
(reverse-expand semantic scoring), `d50d824` (auto-obs off + trivial-exception
filter), and `8037aaf` (BM25 tool descriptions).
**Related docs**: `docs/explicit-symbol-anchors.md` (§ Session 2026-04-22),
`docs/memory-consolidation.md` (§ Auto-observations are opt-in),
`docs/ranking.md` (§ Pivot eligibility filter),
`benches/swebench/WARM_WORKSPACES.md` (§ Observation table poisoned).

> Read this before tuning ranking or memory logic on the back of a single
> harness run. Two of these lessons were re-learnt the hard way because the
> earlier sessions didn't write them down.

---

## 1. The capsule is a *feedback* system, not a pure retrieval system

Every `run_pipeline` used to write `(query, top pivot FQNs)` back into the
observation store as an `auto` kind. The consolidator merged related entries
into `Consolidated` rows that re-appeared in future capsules' "Session
memory" section. The loop:

```
query → capsule → pivots chosen → auto-observation written → consolidated
  → future query retrieves Consolidated row → biases future pivot selection
  → …
```

The loop has **no success signal**. A run whose pivots missed the fix site
gets recorded identically to one that led to the correct diff. Three
consecutive failed runs on sympy-21379 cemented *"pivots: symbols, exp,
interval.exp"* as the canonical memory for that query — actively steering the
agent away from `Mod.eval` (the real fix site).

**Action taken**: default `auto_observations = false` (#72). Opt back in via
`[observability] auto_observations = true` for workflows where queries tend
to self-correct.

**Heuristic for future work**: any new auto-write channel feeding back into
retrieval must either (a) gate on an explicit success signal, or (b) default
to off and require opt-in. Don't rely on consolidation volume to smooth out
the noise — one repeated failure drowns out an occasional success.

---

## 2. BM25 + centrality promote stubs over behaviour when the query names
   a symbol that has both a stub and a body

Example: `class PolynomialError(BasePolynomialError): pass` in
`sympy/polys/polyerrors.py` is a 1-line stub. BM25 scores it highly on any
task that names `PolynomialError`, because both the FQN and the body contain
the term. But the body is a single declaration — useless as a pivot — and
the behaviour actually lives in the raisers (`gcd()` → `parallel_poly_from_expr`,
dozens of call sites).

Before the trivial-exception filter (#73), this took a pivot slot that
should have gone to a behaviour-carrying caller.

**Action taken**: `is_trivial_exception_pivot` filter in `ranking.rs`:

```rust
// kind is a type definition AND name ends Error/Exception/Warning
// AND body has ≤3 non-blank lines
```

Stubs stay eligible as reverse-expand seeds — we still want the walk — they
just can't occupy pivot slots on their own.

**Heuristic**: when a symbol's *name* matches but its *body* is below a
token-complexity floor, it's probably a stub. Prefer its neighbours (callers,
raisers) for the pivot slot. Applies beyond exceptions — empty trait impls,
type aliases, re-export `pub use` lines all have the same shape.

---

## 3. Removing misleading signal beats adding correct signal

The 2026-04-22 sympy-21379 success happened **even though `Mod.eval` was
still not in the capsule pivots**. The semantic reverse-expand (#69 v2) —
which was supposed to rank `Mod.eval` highly via body-text cosine — didn't
break it into the top 8. File-diversity pinning still held 5 of those slots
for anchor-named files.

What changed was the absence of misleading signal:

- No session memory pointing the agent the wrong way (auto-obs off).
- No 1-line exception stub wasting slot 8 (trivial-exception filter).

With that, the agent's own `grep -rn "gcd" sympy/core/` found the fix site.
Turn count dropped 38 → 28, cost dropped $1.06 → $0.62, stop reason flipped
`error_max_budget_usd` → `success`.

**Heuristic**: before proposing a new retriever or a new scoring term, audit
the existing capsule for **negative signal**. Stale memories, stub pivots,
and high-noise adjacent-skeleton bodies all take budget from the content the
agent actually needs. Removing them can be as valuable as adding correct
retrieval — and the change surface is smaller.

---

## 4. `n=1` on a stochastic bench is never a signal

Claude-code's exploration order varies run-to-run. Two consecutive runs on
the same workspace and same prompt can produce turn-counts that differ by
±30% purely from tool-order variance. In this session alone:

- Phase 4e claimed "$0.95, 279s baseline" from n=1 — turned out to be a
  lucky sample.
- 14:09 UTC run failed at $1.06 / 389s. 14:34 UTC run on the same workspace
  (with the fixes) succeeded at $0.62 / 203s. Fixes are real; the gap is
  inflated by variance. Expect tighter spread across a larger sample.

**Heuristic**: a single harness run is a **probe**, not evidence. Before
claiming a ranking change fixes (or regresses) a task, collect n≥3 samples
on the same config and report the spread. Budget-hit timeouts are especially
noisy because the cutoff is binary — a task 2 turns from success looks
identical on the outcome axis to a task 20 turns away.

---

## 5. Warm-workspace state is sticky — flag changes aren't retroactive

Disabling auto-observations via `EngineConfig::auto_observations = false`
stops *new* rows from landing. It does not delete existing ones. A warm
workspace that accumulated Auto / Consolidated rows under the old default
still serves them on the next query.

**Action taken**: harness docs now document the clean-up SQL:

```bash
sqlite3 target/swebench-warm/<iid>/.codesurgeon/index.db \
  "DELETE FROM observations WHERE kind IN ('auto', 'consolidated');"
```

**Heuristic**: any flag that gates a write-side effect needs a clean-up path
documented alongside the flag itself. Readers who opt into the new default
via config won't see the change until stale rows are wiped. Same issue would
apply to any future "disable auto-observation by intent" or "disable graph
neighbour expansion on ambiguous FQNs" — check for retroactive impact on
existing indexes.

---

## 6. Claude 2.1.117's MCP init reports `mcp_servers: []` on `--print`

Observed throughout this session — every run's `system/init` event on 2.1.117
reports zero MCP servers attached, even when the sidecar is pre-warmed and
the subprocess MCP is confirmed alive via `lsof`. Tools arrive via
ToolSearch on first call instead of being registered at init time.

Practical consequence: nudges that reference cs-codesurgeon tools by exact
name (e.g. 5b says `mcp__cs-codesurgeon__run_pipeline`) work because
ToolSearch accepts `select:<exact-name>` queries. Nudges that describe the
tool by capability ("use the context engine", "use the bug-fix helper")
depend on BM25 ranking of tool descriptions — hence the description rewrites
in commit `8037aaf`.

**Heuristic**: until 2.1.117's race is fixed upstream (or we downgrade), the
5b-style "name the tool verbatim" nudge is the load-bearing path. Don't
invest in capability-described nudges as the primary interface without
re-verifying ToolSearch surfaces the right tool on each query class.
