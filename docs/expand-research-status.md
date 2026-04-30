# Anchor expansion research — status & resumption guide

**Branch:** `claude/focused-haibt-ea358c` ([PR #93](https://github.com/subsriram/codesurgeon/pull/93))
**Tracking issues:** [#69](https://github.com/subsriram/codesurgeon/issues/69), [#95](https://github.com/subsriram/codesurgeon/issues/95), [#96](https://github.com/subsriram/codesurgeon/issues/96)
**Companion repo:** [cs-benchmark](https://github.com/subsriram/cs-benchmark) — diagnostic panel + per-task gold fix-sites
**Last updated:** before resuming, check git log + recent #96 comments for new probe results.

## TL;DR

13 commits of layered ranking work investigating whether walker-based anchor expansion (forward + reverse, BFS + best-first, depth-stratified RRF, per-seed RRF list split) moves recall on a stratified SWE-bench Verified panel. **As of the most recent probe (`0d5e627`), `none` is still the panel-recommended default.** The walker line is on hold pending the next probe.

The universally-useful subset of this branch landed on main as **[PR #97](https://github.com/subsriram/codesurgeon/pull/97)** — version stamping, `--json` CLI surface, `anchors` debug subcommand, stderr logging, Python traceback shortcut, and the foundational `leaf_name` column that fixes class-method lookup in `anchor_candidates` and `traceback_candidates`. **PR #97 doesn't depend on this branch in any way; it can be merged independently.**

## What's where

### On main (via PR #97)

| Component | Files | Why it's here |
|---|---|---|
| Version stamping | `crates/cs-core/build.rs`, `lib.rs`, both `main.rs` | Useful for any build identification |
| `--json` flags | `cs-cli/main.rs::Commands::{Context, Impact, Anchors}` | Tooling hooks; runs under any strategy |
| `anchors` debug subcommand | Same | Useful for any user debugging retrieval |
| stderr logging in cs-cli | `cs-cli/main.rs::main` | Bug fix: `--json` was being polluted by stdout debug |
| Python traceback shortcut | `anchors.rs::TracebackFrame`, `engine.rs::traceback_candidates`, `ranking.rs::TRACEBACK_RRF_K` | Runs in RRF regardless of walker; high-precision |
| `leaf_name` column + lookup | `db.rs::leaf_of_name` / `symbols_by_leaf_name` + migration | **Foundational** — without it, anchor and traceback paths systematically miss class methods |

### Stays on this branch (research)

| Layer | Files | Status |
|---|---|---|
| `CS_EXPAND_STRATEGY` env var | `ranking.rs::ExpandStrategy::from_env` | 8 variants: `none / v0 / v1a / v1b / v1ab / v2 / v3a / v3b` |
| `CS_EXPAND_DIRECTION` env var | `ranking.rs::ExpandDirection::from_env` | 4 modes: `auto / forward / reverse / both` |
| `CS_EXPAND_TOTAL_BUDGET` / `CS_EXPAND_GRAPH_BUDGET` | `ranking.rs::resolve_*_budget` | Walker output / graph-expansion caps; default 200 / 1000 |
| Walker variants | `ranking.rs::expand_*_directional` | BFS (v0/v1a/v1b/v1ab/v2) + best-first (v3a/v3b) |
| Density-aware fan-out | `ranking.rs::FanOutPolicy::DensityScaled` | v1a / v1ab |
| Direction routing | `ranking.rs::classify_direction` + `WalkDirection` | Per-anchor classifier (kind + fan-out ratio) |
| Seed promotion | `engine.rs` seed_candidate_ids loop | Uses `symbols_by_leaf_name` to catch class methods as walker seeds |
| Seed-as-pivot guarantee | `engine.rs` (anchor_results injection) | Seeds get ANCHOR_RRF_K precision-first treatment |
| Depth-aware RRF re-weight | `engine.rs` partition by score 0.45 | Shallow at `EXPAND_RRF_K = 30`; deep at `EXPAND_DEEP_RRF_K = 8` |
| Per-seed RRF list split | `engine.rs` `per_seed_deep: HashMap<u64, Vec<...>>` | Each seed's deep emissions get their own ranked queue |
| Per-seed depth_dist diagnostics | `ranking.rs::log_expand_stats` | Debug log per parent_seed |
| UCB exploration bonus (`v3b`) | `ranking.rs::expand_best_first_directional` | Per-root-seed UCB term added to priority |

## Empirical state (last full probe)

5 × 11 panel run on cs-benchmark, build `0d5e627a8956`:

- **`none` baseline recall:** ~20% (driven by astropy-7166's class-pivot ⊃ method-fix-site path; that's a cs-benchmark scorer rule, not engine work).
- **Best variant:** v3-family unlocks one cell that no other variant reaches (`sklearn-25102` — symptom_only/dense/2-3 hops).
- **Net:** v3 family doesn't beat `none` on the "any recall" metric across the panel.
- **Closed loops:**
  - Seed selection bug → fixed in `909047d` (leaf_name lookup catches class methods)
  - Per-seed depth_dist diagnostic → shipped in `cddf032`
  - `Axes::hist` not in pivots → fixed in `4c399d6` (seed-as-pivot inject)
  - Depth-3 chain not surviving fusion → addressed in `3e6c379` (depth-aware k=8) and `0d5e627` (per-seed list split)

The expected matplotlib-24177 cascade (chain symbols `fill → add_patch → _update_patch_limits` reaching final pivots) hasn't been verified on `0d5e627`. **The next step is the user's panel re-run on this build.**

## Bugs in this branch that PR #97 fixes (not yet ported back)

Two real bugs in this branch's anchor extraction that PR #97 caught and fixed:

1. **`anchor_candidates` uses `symbols_by_exact_name` (line ~2354)** — should use `symbols_by_leaf_name`. Class methods aren't in the anchor pool.
2. **`traceback_candidates` matches `sym.name != func` (line ~2307)** — should use `leaf_of_name(&sym.name)`. Tracebacks miss class methods.

The branch worked around this for the walker by adding a separate seed-promotion code path that uses `symbols_by_leaf_name`. The precision-first anchor / traceback paths weren't fixed. **Once PR #97 merges, rebase this branch on main to pick up the fixes** — see "Resuming" below.

## Open questions / next steps when resuming

In rough priority:

1. **Re-run the panel on `0d5e627`** to verify per-seed RRF list split actually moves matplotlib-24177 + django-16938 recall. If it does → engine work pays off; recommendation in #69 flips. If it doesn't → walker priority signal is the real bottleneck (next investigation).

2. **If panel still flat:** the priority signal underweights depth. Two candidate fixes, in increasing scope:
   - **Depth-continuation bonus** in best-first: boost the priority of children whose parent was just popped at high priority. Encourages chain-following over sibling-surveying. ~30 lines in `expand_best_first_directional`.
   - **Per-tree-node UCB (UCT)**: replace per-root-seed `subtree_visits` with per-node visit + value tracking. Real MCTS-shape exploration. Bigger change (~80 lines + harder testing).

3. **Cross-language validation:** the panel is Python-only. Walker behavior on Rust impl-method graphs, TS class methods, and Swift extensions hasn't been tested. The `leaf_name` infrastructure should already cover them (the `::` convention is shared), but RRF behavior under different graph densities is unverified.

4. **`v3a`/`v3b` produced identical pivots** on the matplotlib probe. UCB exploration isn't moving the queue. Either the bonus magnitude is too small (currently `c = 0.5`, vs UCB1's `c = √2`), or the per-root-seed granularity is too coarse for the queue's actual decision points. Tied to (2).

5. **Forward-reach metric on cs-benchmark** — user mentioned prototyping `fix_site_in_forward_reach` (mirrors `fix_site_in_impact` but for callee direction). If that lands in cs-benchmark, the panel can credit forward-shaped tasks where the chain is reachable-by-one-call from a pivot, even if not directly in pivots. Reduces engine-side pressure.

## Resuming this branch

```bash
cd ~/projects/codesurgeon
git fetch origin

# 1. Pick up the latest #96 probe results before doing anything.
gh issue view 96 --comments

# 2. If PR #97 merged while away: rebase this branch on main to pick up
#    the anchor_candidates + traceback_candidates leaf_name fixes.
git worktree add .claude/worktrees/focused-haibt-ea358c claude/focused-haibt-ea358c
cd .claude/worktrees/focused-haibt-ea358c
git rebase origin/main

# Expect conflicts in:
#   - crates/cs-core/src/db.rs       (PR #97 added leaf_name; this branch already has it)
#   - crates/cs-core/src/engine.rs   (PR #97 fixed traceback_candidates and anchor_candidates;
#                                     keep PR #97's `leaf_of_name(&sym.name)` and
#                                     `symbols_by_leaf_name(lookup)` calls)
#   - crates/cs-core/src/anchors.rs  (PR #97 added Serialize derive + traceback_frames;
#                                     this branch already has them — keep main's version)
#   - crates/cs-core/src/ranking.rs  (PR #97 added TRACEBACK_RRF_K; keep)
#   - crates/cs-core/build.rs        (identical between branches)
#   - crates/cs-cli/src/main.rs      (PR #97's --json + anchors + stderr — keep)
#   - crates/cs-mcp/src/main.rs      (PR #97's argv parser — keep)

# 3. Resolve conflicts by keeping PR #97's versions of universally-useful code,
#    and re-applying this branch's walker-specific additions on top.
#    Most conflicts are "both added the same thing" — `git checkout --theirs` for
#    most files, then verify the walker-specific additions are still there.

# 4. Pre-commit:
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# 5. Force-push:
git push --force-with-lease

# 6. Refresh the deployed binary in cs-benchmark for the next probe:
cargo build --release --features cs-core/metal -p cs-core -p cs-cli -p cs-mcp
cp target/release/codesurgeon target/release/codesurgeon-mcp \
   ~/projects/cs-benchmark/target/release/

# 7. Sanity smoke:
~/projects/cs-benchmark/target/release/codesurgeon --version
# Should report a fresh SHA, no `+dirty` marker.
```

## Resuming from cs-benchmark side

```bash
cd ~/projects/cs-benchmark
git pull

# Verify the panel + scorer + warm workspaces are still intact.
ls bench/reverse_expand_panel/panel/tasks/  # 20 task TOMLs
uv run bench/reverse_expand_panel/verify_fix_sites.py --strict

# Re-run the same matplotlib probe to confirm the fix-site lands or not:
CS_LOG=cs_core::ranking=debug \
CS_EXPAND_STRATEGY=v3a \
CS_EXPAND_DIRECTION=forward \
target/release/codesurgeon context "[Bug]: ax.hist density not auto-scaled when using histtype='step'" \
  --json > /tmp/cap.json 2> /tmp/log

grep -E "expand seeds|per-seed deep|emissions:" /tmp/log
jq '.pivots[].fqn, .skeletons[].fqn' /tmp/cap.json | grep -E "fill|add_patch|_update_patch"

# Then full panel:
CODESURGEON_BIN=$PWD/target/release/codesurgeon \
  uv run bench/reverse_expand_panel/run_panel.py \
    --run-id $(date +%Y%m%d-%H%M)-resume

uv run bench/reverse_expand_panel/report.py \
  target/reverse_expand_panel/<old>.jsonl \
  target/reverse_expand_panel/<new>.jsonl \
  --baseline none
```

## Decision tree on the next probe

| Outcome | What it means | Next action |
|---|---|---|
| matplotlib-24177 + django-16938 flip `fix_site_in_pivots: True` | Per-seed RRF works. Engine pays off. | Update [#69 recommendation](https://github.com/subsriram/codesurgeon/issues/69#issuecomment-4347417418) from "stay on `none`" to ship `v3*_both`. Land PR #93 or rebase + new PR. |
| Chain visible in `emissions:` log + per-seed depth_dist for the right seed, but `fix_site_in_pivots: False` | RRF fix landed but post-fusion rerank/centrality is dropping the chain. | Investigate `apply_centrality_and_semantics` and `rerank_by_query_proximity` next. Probably another small fix. |
| Chain still missing from `emissions:` log entirely | Walker priority signal is the bottleneck. | Implement depth-continuation bonus (option 2 in "Open questions"). |
| All variants produce identical results | Strategy gate isn't being read, or env var routing is broken. | Check `ExpandStrategy::from_env()` — also verify the deployed binary is the right SHA. |

## Key files to re-read when resuming

In rough order of relevance:

1. **`crates/cs-core/src/ranking.rs`** — all walker logic, `ExpandStrategy`, `WalkDirection`, `FanOutPolicy`, both walkers, `log_expand_stats`. ~1500 lines.
2. **`crates/cs-core/src/engine.rs`** — engine dispatch (`build_context_capsule` block from line ~2660), seed promotion, RRF fusion with per-seed splits.
3. **`crates/cs-core/src/anchors.rs`** — anchor extraction (already minimal; PR #97 covered the universal piece).
4. **[`docs/explicit-symbol-anchors.md`](explicit-symbol-anchors.md)** — historical context on why anchor extraction matters; predates this branch.
5. **[`docs/ranking.md`](ranking.md)** — official ranking pipeline doc; lists all the constants. Update when ranking changes ship.
6. **cs-benchmark `docs/reverse_expand_panel.md`** + `runbook.md` — panel design + how to run.

## Why the walker investment was valuable even if it doesn't ship

- Surfaced and fixed two real precision-first lookup bugs (now in PR #97): `anchor_candidates` and `traceback_candidates` were silently missing class methods.
- Built the cs-benchmark diagnostic panel — task curation, gold fix-site verification, per-task forward-reach metric. Reusable for any future ranking work.
- Established the (strategy × direction × budget) × (per-task category) heatmap as the right empirical frame. Future ranking changes have a measurement harness.
- Generated layered diagnostics (per-seed depth_dist, emission spot-check, fusion split log) that are useful even on `none` — they read out exactly what the engine is doing per query.

If the panel re-run still shows `none` winning, the walker work parks here as a documented research line. If a future ranking idea (semantic-driven priority? LLM-rerank? structural similarity?) needs forward expansion infrastructure, this branch is where you start.

## Session notes (workflow + gotchas worth preserving)

### How the work was driven

- Empirical-first. Every engine change was validated against a cs-benchmark probe (panel run or single-task `codesurgeon context` invocation with `CS_LOG=debug`) before deciding whether it shipped or got reverted. Match this style: don't ship ranking changes on hand-waved math; the math is consistently wrong about which direction recall moves.
- Issue threads (`#69`, `#95`, `#96`) carry the long-form reasoning. The commit messages are redundant with the issue comments by design — the issue comment IS the design doc, the commit is the execution.
- Diagnostic logs were added *before* algorithmic changes when possible. Pattern: ship the diagnostic, run the probe, read the diagnostic, decide the fix. Don't fix-then-add-tracing.

### Build / deploy mechanics

- This worktree is at `/Users/sriram/projects/codesurgeon/.claude/worktrees/focused-haibt-ea358c`.
- cs-benchmark is at `~/projects/cs-benchmark`. Binaries get copied to `~/projects/cs-benchmark/target/release/` for the panel to find them.
- `CLAUDE.md` says `cargo build --release --features metal`, but `metal` is on cs-core only — use `cargo build --release --features cs-core/metal -p cs-core -p cs-cli -p cs-mcp` to build everything with embeddings.
- `--version` shows the SHA. Always verify `~/projects/cs-benchmark/target/release/codesurgeon --version` matches the latest commit before running a panel; stale binaries silently misled multiple probes early in the session.
- `+dirty` marker means uncommitted changes in the worktree. Don't probe with a `+dirty` binary unless you're explicitly testing local code; results aren't reproducible from git history.

### Pre-commit checklist (worktree convention)

In this exact order — `cargo fmt --all` last, since it's the most common CI failure:

```bash
cargo build --release --features cs-core/metal -p cs-core -p cs-cli -p cs-mcp  # smoke
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo fmt --all
```

Gotchas that bit during the session:

- Clippy is strict on `Vec<(u64, f32, u64)>` style "complex types" — extract a type alias if it appears in 2+ places, otherwise wrap in a block.
- `into_iter().chain(other.into_iter())` triggers `useless_conversion`; second arg can be the bare `other`.
- `if let Ok(...) = row { acc.push(...) }` triggers `manual_flatten`; use `rows.flatten().collect()` instead.
- rusqlite `prepare()` borrow lives until the row iterator is consumed — don't try to chain `.filter_map().collect()` directly off `query_map()`; bind to a `let stmt = ...` then iterate.

### Tracing diagnostics — gotchas

- `tracing::debug!(target: "cs_core::ranking", ...)` overrides the target. **`tracing::enabled!(Level::DEBUG)` does NOT** — without an explicit `target:`, it checks the callsite's module path. So a guard like
  ```rust
  if tracing::enabled!(Level::DEBUG) {                   // checks cs_core::engine
      tracing::debug!(target: "cs_core::ranking", ...);  // never fires under cs_core::ranking=debug
  }
  ```
  is broken. Always pass `target:` to both. This was a real bug for `expand seeds:` for one cycle.
- The user's panel filter is `CS_LOG=cs_core::ranking=debug`. All expand-related debug logs route through `target: "cs_core::ranking"` for that filter to catch them. Engine logs that emit walker-related diagnostics use the `cs_core::ranking` target even though they're in `engine.rs`.
- cs-cli writes to stderr. cs-mcp writes to stderr (because stdout is JSON-RPC). cs-cli used to write to stdout, polluting `--json` output. Check `tracing_subscriber::fmt().with_writer(std::io::stderr)` is set in any new binary.

### Walker semantics — quick reference

- **Score encoding:** both BFS (`1.0 / (depth + 2)` from source-depth) and best-first (`1.0 / (next_depth + 1)`) emit the same numbers per depth. Depth 1 → 0.5, depth 2 → 0.333, depth 3 → 0.25. Threshold of 0.45 cleanly splits depth 1 from depth ≥ 2 in `expand_shallow` / `expand_deep` partition.
- **`parent_seed`** is the *root* seed of an emission's subtree, not the immediate parent. Inherited unchanged from `seed → child → grandchild`. Both walkers carry it now (best-first via `PqItem.parent_seed`, BFS via the `(id, depth, parent_seed)` queue tuple). Output type is `Vec<(u64, f32, u64)>` aliased as `ExpandEmission`.
- **Seed eligibility** has direction-specific gates:
  - Forward: `dependencies(id).len() <= EXPAND_SEED_FANOUT_LIMIT` (500). No kind gate.
  - Reverse: `is_reverse_expand_seed(sym)` (exception/error/warning class) AND `dependents(id).len() <= EXPAND_SEED_FANOUT_LIMIT`.
- **Seed promotion** uses `db.symbols_by_leaf_name(...)` not `symbols_by_exact_name(...)`. The former catches class methods; the latter does not. Bug was discovered after a probe showed only `pyplot::hist` becoming a seed for matplotlib's `hist` query.
- **UCB exploration (v3b)** is currently per-root-seed: one `subtree_visits` counter per seed. UCT (per-tree-node) was discussed but not implemented — open question whether it would actually help given empirical data.

### When something doesn't work as expected

In rough order of "which mistake have I made this time":

1. Stale binary in `~/projects/cs-benchmark/target/release/`. Check `--version` SHA matches the latest commit.
2. `+dirty` build. Commit before deploying.
3. Wrong CS_LOG filter — for full diagnostic visibility use `CS_LOG=cs_core::ranking=debug` (matches all expand-related logs); for everything use `CS_LOG=debug`.
4. Forgot `--features cs-core/metal` — embeddings disabled silently, v2 / semantic-related strategies degrade.
5. Tests pass, clippy passes, but logs aren't visible — check the `tracing::enabled!()` guard target matches the inner `debug!`'s explicit target.
6. The walker emits but pivots are missing the chain — that's the per-seed RRF / fusion question, not a walker bug. Check the emission spot-check log first.

### What surprised me during the session

- `none` winning the panel was unexpected. Walker investments were paying off in synthetic tests but not in panel recall. The empirical harness saved us from shipping a regression.
- The seed-selection bug (only one of two `hist` matches becoming a seed) was upstream of the walker entirely — `symbols_by_exact_name` was the broken layer. PR #97 fixes this on main *better* than any walker-side workaround.
- Per-seed depth_dist + emission spot-check together resolved a question the user asked at the start of the #96 thread: "why isn't `_update_patch_limits` in pivots?" Three rounds of "is this an engine bug or a panel-shape question?" eventually answered itself once the diagnostics were good enough.
- `claude --print` doesn't auto-load `cwd/CLAUDE.md` — that was the source of an early misdiagnosis on the cs-benchmark side ("CLAUDE.md guidance was ignored" → actually never delivered).

### People / repo conventions to remember

- The user is `subsriram` on GitHub. Issues, PRs, and commits all under that account.
- Co-author trailer on commits: `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.
- `gh` CLI: not on the default PATH on this machine — use `/opt/homebrew/bin/gh`.
- The user runs probes asynchronously and posts results back as issue comments. Watch for new comments on `#69`, `#95`, `#96` between sessions.
- Issue comments are the canonical reasoning record. Long debug-log dumps go in comments, not commits.
- cs-benchmark and codesurgeon are separate repos. PRs / commits go to their respective remotes (`subsriram/codesurgeon`, `subsriram/cs-benchmark`).
