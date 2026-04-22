# SWE-bench warm-workspace workflow

How to run the SWE-bench harness against persistently-indexed task workspaces
instead of cold-cloning + indexing on every run. Indexing is treated as a
**separate job** that runs once per (task, binary version); the harness is
only fired when the index is known ready.

## Why this exists

`benches/swebench/run.py` originally cloned each task's repo into a
`tempfile.TemporaryDirectory()` and ran `codesurgeon index` inline before
spawning `claude --print`. That works for the full 100-task pilot, but it's
wasteful for iterative development:

- Cold-cloning sympy on every iteration costs ~30-60 s of network.
- First-time indexing of a ~1500-file repo takes 5-15 min (more with
  embeddings).
- A schema bump or binary rebuild invalidates the index and forces a full
  re-parse. If this happens *inside* a harness run, the agent waits.
  Commit `19cd12e` made the MCP serve the warm index while re-indexing
  continues in the background — but the ranking the agent sees still
  reflects a partially-updated state, which confounds measurements.

The workflow below separates concerns:

| Step | Command | Typical cost |
|---|---|---|
| One-time: clone task repo | `git clone … && git checkout <base_commit>` | 30-60 s + disk |
| One-time per binary version: build the index | `codesurgeon index --workspace $WS` | 5-45 min |
| Each harness iteration | `run.py --reuse-workdir $WS …` | seconds of pre-claude overhead |

## Directory convention

Warm workspaces live under **`target/swebench-warm/`** inside the repo by
default. `target/` is already covered by the top-level `.gitignore`, so the
indexes never get committed accidentally. Each task gets its own directory
named after its `instance_id`:

```
<repo_root>/target/swebench-warm/
├── sympy__sympy-21379/          ← sympy clone at base_commit
│   ├── .codesurgeon/             ← codesurgeon index (SQLite + tantivy + embeddings)
│   ├── sympy/                    ← repo source
│   └── …
├── sphinx-doc__sphinx-9711/
├── pydata__xarray-7229/
└── …
```

The `.codesurgeon/` directory inside each workspace holds the warm index.
That's what the harness's `--reuse-workdir` flag points at.

**Why under `target/`?** The warm indexes are tied to the `codesurgeon`
binary under `target/release/` that wrote them. Co-locating them means:

- Already gitignored — zero risk of committing the 280 MB SQLite blob.
- `cargo clean` wipes both the binary and the indexes together. That's the
  correct invalidation: a rebuilt binary may have a different graph schema
  and the old indexes would need regeneration anyway.
- Each `git worktree` has its own `target/`, so warm indexes built by one
  worktree's binary can't be silently opened by another worktree's binary
  (avoiding schema-mismatch corruption).

**Overriding the location.** If you want cross-worktree sharing or a
persistent cache that survives `cargo clean`, set `$SWEBENCH_WARM_ROOT`:

```bash
# e.g. keep warm indexes under a user-owned cache dir
export SWEBENCH_WARM_ROOT=$HOME/.cache/codesurgeon/swebench-warm
```

Do this only if you're confident the binaries using the cache stay on the
same graph schema — otherwise you'll hit forced re-indexes every time you
switch worktrees.

## Preparing a warm workspace (one-time per task)

The helper script below clones + checks out + indexes in one shot. Save it
as `benches/swebench/prepare_workspace.sh` (also committed to the repo).

```bash
#!/usr/bin/env bash
# Usage: ./prepare_workspace.sh <instance_id> [workspace_root]
set -euo pipefail
instance_id="${1:?usage: $0 <instance_id> [workspace_root]}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
root="${2:-${SWEBENCH_WARM_ROOT:-$repo_root/target/swebench-warm}}"
tasks_json="$repo_root/benches/swebench/tasks.json"
cs_bin="$repo_root/target/release/codesurgeon"

# Extract repo + base_commit from tasks.json
eval "$(python3 - <<EOF
import json
t = next(x for x in json.loads(open("$tasks_json").read())["tasks"]
         if x["instance_id"] == "$instance_id")
print(f"repo_url=https://github.com/{t['repo']}.git")
print(f"base_commit={t['base_commit']}")
EOF
)"

mkdir -p "$root"
ws="$root/$instance_id"

if [ -d "$ws/.git" ]; then
  echo "[prepare] existing workspace at $ws — reusing clone"
  git -C "$ws" reset --hard "$base_commit"
  git -C "$ws" clean -fdx -e ".codesurgeon"
else
  echo "[prepare] cloning $repo_url @ $base_commit into $ws"
  git init --quiet "$ws"
  git -C "$ws" remote add origin "$repo_url"
  git -C "$ws" fetch --depth 1 origin "$base_commit" \
    || git -C "$ws" fetch --depth 50 origin
  git -C "$ws" checkout --quiet "$base_commit"
fi

echo "[prepare] indexing with $cs_bin"
CS_WORKSPACE="$ws" "$cs_bin" index --workspace "$ws"

echo "[prepare] done — warm workspace at $ws"
```

Run it once per task:

```bash
bash benches/swebench/prepare_workspace.sh sympy__sympy-21379
bash benches/swebench/prepare_workspace.sh sphinx-doc__sphinx-9711
# … one per task you plan to iterate on
```

Cost: one cold-parse pass (5-45 min depending on repo size + whether
embeddings are enabled), after which the `.codesurgeon/index.db` is durable.

## Verifying an index is ready

Before firing the harness, confirm the index is present and current:

```bash
# 1. Non-zero symbol count
./target/release/codesurgeon --workspace $SWEBENCH_WARM_ROOT/sympy__sympy-21379 status
# Expected: "Symbols : N" with N > 0

# 2. A quick incremental re-index returns fast (no "parsing N files" log)
./target/release/codesurgeon index --workspace $SWEBENCH_WARM_ROOT/sympy__sympy-21379
# Expected: finishes in seconds. If it says "graph schema bumped → re-indexing
# all files", your binary is newer than the last build — let it finish.

# 3. No stale MCP holding the PID lock
cat $SWEBENCH_WARM_ROOT/sympy__sympy-21379/.codesurgeon/mcp.pid 2>/dev/null
ps -p $(cat $SWEBENCH_WARM_ROOT/sympy__sympy-21379/.codesurgeon/mcp.pid 2>/dev/null) 2>/dev/null
# Expected: no process. If a zombie mcp.pid exists from a crashed run,
# rm -f $WS/.codesurgeon/mcp.pid before proceeding.
```

Only then fire the harness.

## Running the harness against a warm workspace

```bash
uv run benches/swebench/run.py \
  --instance-ids sympy__sympy-21379 \
  --arms with \
  --reuse-workdir "$WARM/sympy__sympy-21379" \
  --max-budget-usd 3.00 \
  --timeout 600 \
  --clean
```

What `--reuse-workdir` does ([run.py](run.py) ≈ L380-L420):

1. Skips cloning — uses the warm checkout directly.
2. `git reset --hard <base_commit>` + `git clean -fdx -e .codesurgeon` before
   each run, so prior agent edits and untracked files are cleared but the
   index survives.
3. Skips the inline `codesurgeon index` pre-step — trusts what's already on
   disk.
4. **Path resolution**: the argument is `.resolve()`'d to an absolute path
   inside `run.py` before anything else. Passing a relative path is fine
   from your shell, but internally it must be absolute because claude is
   spawned with `cwd=<workdir>` and would otherwise resolve the
   `--mcp-config` path twice, producing a double-nested non-existent path.
   This bit a rerun earlier in the history; the resolve is now automatic.

The `codesurgeon-mcp` child still calls `index_workspace()` on boot, but with
all file hashes matching it's a fast walk (~1–5 s) rather than a full
re-parse. Agent queries during that brief window are served from the warm
SQLite per commit `19cd12e`.

### Flags that shape a harness run

| Flag | Default | Effect |
|---|---|---|
| `--nudge {5b,5c}` | `5b` | Treatment-arm PROMPT_PREFIX variant. **5b** tells the agent to paste the raw problem statement into `context` on `run_pipeline`; **5c** doesn't mention `context` at all — relies purely on the MCP tool description. 5c is the baseline for measuring whether an in-prompt nudge buys anything beyond the server-side description. |
| `--inject-claude-md` | off | Drops a codesurgeon-tool-guidance `CLAUDE.md` into the workdir (treatment arm only). If the task repo already ships its own `CLAUDE.md` at `base_commit`, our content is prepended, not overwritten. Used to advertise the `run_pipeline → get_impact_graph` chaining workflow to the agent. |
| `--stream-json` | off | Switches `claude`'s output format to `stream-json --verbose`. Raw NDJSON of per-turn events is saved as `claude_stream.jsonl` instead of `claude.json`. Lets you inspect each tool call's args (e.g. confirm the agent populated a new MCP param like `context`). Slight cost overhead vs `json` mode; use when you specifically need per-turn visibility. |

The JSON summary fields in `results.jsonl` (tokens, cost, diff bytes) are
identical across `--stream-json` and plain `json` modes — the flag only
affects which artifact type gets saved alongside the numeric summary.
Downstream consumers that read `claude_json_path` must branch on the file
suffix: `.json` for single-object summaries, `.jsonl` for NDJSON streams.

### Typical A/B workflow for a single task

```bash
WS="$WARM/sympy__sympy-21379"

# Baseline (bare claude, no codesurgeon)
uv run benches/swebench/run.py --instance-ids sympy__sympy-21379 \
  --arms without --reuse-workdir "$WS" \
  --max-budget-usd 3.00 --timeout 600 --clean

# Treatment, prompt-only verbatim-forward (5b), no CLAUDE.md
uv run benches/swebench/run.py --instance-ids sympy__sympy-21379 \
  --arms with --reuse-workdir "$WS" --nudge 5b \
  --max-budget-usd 3.00 --timeout 600 --stream-json

# Treatment, 5b + CLAUDE.md (Phase 4)
uv run benches/swebench/run.py --instance-ids sympy__sympy-21379 \
  --arms with --reuse-workdir "$WS" --nudge 5b --inject-claude-md \
  --max-budget-usd 3.00 --timeout 600 --stream-json
```

Don't pass `--clean` between the last two — accumulate rows in
`results.jsonl` so they can be diffed side-by-side.

## Reference data — prior full-harness runs

- [`benches/swebench/pilot_results/results.jsonl`](pilot_results/results.jsonl)
  — 10-task pilot (#29b). 20 rows (10 × 2 arms).
- [`target/swebench/results_29c_backup.jsonl`](../../target/swebench/results_29c_backup.jsonl)
  — 83-task #29c run (completed of the 100 scheduled). 166 rows. Useful
  historical baseline; includes without-arm numbers for
  `sympy__sympy-21379` ($0.30 / 96 s) and sixteen other sympy tasks that
  repeatedly appear in regression work.

`cargo clean` wipes the 29c backup too, so if it matters to you, copy it
out to `~/` first.

## Invalidating after binary rebuilds

codesurgeon's graph schema is versioned. When you `cargo build` a binary
that bumped the version, every warm workspace's index becomes stale.
Symptom: starting any tool against a stale workspace logs

```
Graph schema version changed (expected N); forcing re-index
```

and triggers a full re-parse. To upgrade cleanly:

```bash
# Rebuild once
cargo build --release --features metal

# Then re-run prepare_workspace.sh for each warm workspace.
# This is an incremental operation — the clone stays, only the index
# re-writes (because the schema bumped, this pass will be a full re-parse).
for iid in sympy__sympy-21379 sphinx-doc__sphinx-9711 pydata__xarray-7229; do
  bash benches/swebench/prepare_workspace.sh "$iid" &
done
wait
```

Running the upgrades in parallel is safe — each workspace has its own
`.codesurgeon/` directory, so there's no lock contention across tasks.

## Multi-task harness runs

Once every warm workspace is ready, the harness can iterate over them
without re-cloning or re-indexing:

```bash
for iid in sympy__sympy-21379 sphinx-doc__sphinx-9711 pydata__xarray-7229; do
  uv run benches/swebench/run.py \
    --instance-ids "$iid" \
    --arms with \
    --reuse-workdir "$SWEBENCH_WARM_ROOT/$iid" \
    --max-budget-usd 3.00 \
    --timeout 600
done
```

Budget totals: `--clean` is deliberately omitted so `results.jsonl`
accumulates across tasks — analyze with `scripts/swebench_report.py`.

**Why not a single invocation of `run.py --instance-ids a,b,c`?**
`--reuse-workdir` currently takes a single path, not a map from
`instance_id → workspace`. For now, one harness invocation per task is the
cleanest wiring.

## Troubleshooting

> **If every harness run in a session starts failing with `mcp_servers: []`
> at init and the agent reporting `mcp__cs-codesurgeon__*` tools as
> "not available"**, jump straight to the *"Agent reports tool is not
> available in the deferred tools list"* section below — the cause is
> almost always a stale `claude mcp add*` registration poisoning Claude
> Code's global MCP state, **not** a harness / binary / workspace bug.

### `fatal: reference is not a tree: <base_commit>`
`prepare_workspace.sh` defaults to a shallow fetch. Some repos disallow
single-commit fetches. The script already falls back to `--depth 50`; if
that still fails, widen the fallback or clone full-depth manually.

### `Graph schema version changed (expected N); forcing re-index`
Expected on the first run after a `cargo build` that bumped schema. Let the
index job finish — subsequent runs will be fast.

### Harness hangs on first `run_pipeline` call
Shouldn't happen post-`19cd12e`, but if it does, symptoms point to a real
bug — capture a backtrace:

```bash
ps aux | grep codesurgeon-mcp | awk '{print $2}' | head -1 | xargs -I{} sample {} 5
```

### First MCP boot after a workspace move produces heavy WAL writes
If you moved a warm workspace from one location to another (e.g.
`/tmp/foo-repro` → `target/swebench-warm/foo`) **with a binary change in
between**, the first MCP boot against the new location may churn hundreds
of MB of WAL before settling. This is the schema-migration re-parse
catching up, not a hang — but it can look alarming in `ps aux`.

Symptoms:
- `codesurgeon-mcp` holding `index.db-wal` open, size growing into the
  hundreds of MB
- Sustained 400–600 % CPU for minutes
- Agent's first tool call eventually returns, but after a noticeable wait

Mitigation: after a move, run a standalone `codesurgeon index` against
the new location **before** firing the harness. The index pass will
complete the migration once; subsequent MCP boots will be fast. This is
what [`prepare_workspace.sh`](prepare_workspace.sh) does automatically
when called on an already-present clone.

### `claude --print` arg format mismatch
`--stream-json` requires `claude` >= 2.1.x (enforces `--verbose`
alongside `stream-json`). The harness uses whatever `claude` resolves on
`$PATH`; set `CLAUDE_BIN=/path/to/newer/claude` if your default is older.
Verify with `claude --version`.

### Warm workspace got polluted (uncommitted edits from a crashed run)
```bash
git -C $SWEBENCH_WARM_ROOT/<iid> reset --hard <base_commit>
git -C $SWEBENCH_WARM_ROOT/<iid> clean -fdx -e .codesurgeon
```
The harness does this automatically at the start of each `--reuse-workdir`
run, so this is only needed if a run aborted before that step executed.

### `.codesurgeon/mcp.pid` points at a dead process
```bash
rm -f $SWEBENCH_WARM_ROOT/<iid>/.codesurgeon/mcp.pid
```
No data loss — the pid file is a lock, not part of the index.

### Agent reports "tool is not available in the deferred tools list"

Symptom: every harness run times out / fails, the saved stream's init
event shows `mcp_servers: []` and zero `mcp__cs-codesurgeon__*` tools
advertised. Agent explains: *"The tool `mcp__cs-codesurgeon__run_pipeline`
is not available in this environment."*

This is **not** a harness bug. Claude Code's CLI maintains a global MCP
registry state (in `~/.claude.json`) that can be poisoned by stale /
conflicting `claude mcp add` or `claude mcp add-json` registrations made
outside the harness. When the registry is in a bad state, Claude Code's
`--print` mode stops advertising **any** MCP tools to the agent — even
from servers explicitly passed via `--mcp-config` with `--strict-mcp-config`.

Diagnostic: run
```bash
claude mcp list
```
and inspect the output. Every server entry that points at a stale path
(e.g. a binary you moved or rebuilt elsewhere), or overlaps with
`--mcp-config` server names, is a candidate for removal. Remove with:
```bash
claude mcp remove <name>
```
then kill any orphan processes and rerun the harness:
```bash
pkill -f codesurgeon-mcp
bash benches/swebench/prepare_workspace.sh <iid>   # re-warm index, clears mcp.pid
```

**Prevention**: don't use `claude mcp add*` to register per-harness MCP
servers. The harness uses `--mcp-config <ephemeral json>` +
`--strict-mcp-config` which is supposed to isolate, but the CLI's state
machine can still break if global registrations collide. Keep global
registrations for interactive use only; the harness flow should never
mutate `~/.claude.json`.

This was the root cause of a multi-hour debugging detour in the
2026-04-21 session: an `add-json cs-worktree ...` done during
investigation poisoned state for every subsequent `claude --print`
invocation — the exact symptom looked like "my binary stopped working"
(zombie MCPs, `mcp_servers: []` at init, agent timing out) but was
really "CLI state is poisoned." Evidence saved under
`target/swebench/with/sympy__sympy-21379/archive/2026-04-21T22-*` and
`archive/2026-04-22T*`.

### Index size feels too large
Full sympy at v1.7 with embeddings:
- `index.db` ≈ 280 MB
- `index.db-wal` ≈ up to 100 MB during writes, shrinks at quiescence
- `embeddings.bin` ≈ 50-150 MB
The `.db-wal` only grows during active writes; if it's persistently large
after indexing, a checkpoint didn't run — safe to remove when no writer
holds the db.

## Putting it together — end-to-end quickstart

```bash
# Default warm root = <repo_root>/target/swebench-warm/
# Override with: export SWEBENCH_WARM_ROOT=/some/other/path
WARM=${SWEBENCH_WARM_ROOT:-$(pwd)/target/swebench-warm}

# 1. Build the binary (once per cs-core change)
cargo build --release --features metal

# 2. Prepare the warm workspace (once per task, or after schema bump)
bash benches/swebench/prepare_workspace.sh sympy__sympy-21379

# 3. Verify
./target/release/codesurgeon --workspace \
  "$WARM/sympy__sympy-21379" status

# 4. Run the harness (as many iterations as you want without re-indexing)
uv run benches/swebench/run.py \
  --instance-ids sympy__sympy-21379 \
  --arms with \
  --reuse-workdir "$WARM/sympy__sympy-21379" \
  --max-budget-usd 3.00 \
  --timeout 600 \
  --nudge 5b \
  --clean
```

That's the whole loop. Step 2 is where the 5-45 min cost lives; steps 3-4
are iteration-speed after that.
