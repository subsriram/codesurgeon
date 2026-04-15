# SWE-bench Verified benchmark

End-to-end benchmark that measures whether attaching codesurgeon as an MCP
server lifts Claude Code's pass@1 on real GitHub issues, and by how much it
changes token consumption per task. Internal comparison only — bare Claude
Code vs Claude Code + codesurgeon on the same model, same machine, same day.

See `PLAN.md §B1` for the original spec, `#29` for the tracking meta-issue,
and `#29a` / `#29b` / `#29c` for the three-stage rollout.

## Layout

```
benches/swebench/
├── README.md              # this file
├── tasks.json             # 100 stratified Verified tasks (seed=17)
├── select_tasks.py        # regenerates tasks.json from HF API
├── mcp_with.json          # treatment-arm MCP config (template)
├── mcp_without.json       # control-arm MCP config (empty)
├── run.py                 # driver: clone repo → claude --print → capture diff
├── smoke.md               # #29a wiring smoke test report
└── report_pilot.md        # #29b pilot run report (landed after pilot completes)

scripts/
├── swebench_eval.py       # swebench harness wrapper (Docker-based eval)
├── swebench_pilot.sh      # one-shot detached pilot launcher (#29b)
└── swebench_report.py     # markdown renderer
```

## Prerequisites

- **uv** + Python 3.14 — scripts are self-installing via PEP 723 inline metadata
- **Docker** (Desktop or daemon) — required by `swebench.harness` for the eval step. Not required for `run.py --dry-run`.
- **`claude` on PATH** — Claude Code CLI v2.1+, authenticated via OAuth (`~/.claude.json`). Detached runs work; no TTY needed.
- **`target/release/codesurgeon-mcp`** — build with `cargo build --release --features metal`
- **Disk** — ~30 GB for swebench Docker image cache once all repos are pulled

## Running the #29b pilot (10 tasks × 2 arms)

Launch as a fully-detached background process so the run survives terminal
close and doesn't inflate the parent shell's quota:

```bash
cd /Users/sriram/projects/codesurgeon
nohup bash scripts/swebench_pilot.sh > /tmp/cs-swe-pilot.out 2>&1 < /dev/null &
disown
```

Then from any shell:

```bash
tail -f /tmp/cs-swe-pilot.out
```

Expected walltime: **1–3 hours** (10 tasks × 2 arms × ~5 min/task agent +
~30 s/task eval, at 4-way Docker parallelism). Expected synthetic cost:
**$5–$15** — but OAuth is subscription-billed, so the real bill is zero;
the cost figure is `usage × list_price` for directional comparison only.

### Kill any interactive Claude Code sessions first

The detached run reads and refreshes OAuth tokens in `~/.claude.json`.
A simultaneously-running interactive session can race on token refresh
writes. It's atomic-rename safe, but simpler to just close your editor's
Claude Code sessions before kicking off the pilot.

### Environment knobs (optional)

```bash
PILOT_TASKS=5 PILOT_BUDGET_USD=1.50 bash scripts/swebench_pilot.sh
```

| Var | Default | What it does |
|---|---|---|
| `PILOT_TASKS` | 10 | First N tasks from `tasks.json` |
| `PILOT_BUDGET_USD` | 3.00 | Per-task `--max-budget-usd` cap |
| `PILOT_TIMEOUT` | 900 | Per-task wallclock seconds |
| `PILOT_MODEL` | *(default)* | `--model` override (e.g. `sonnet`, `opus`) |
| `PILOT_MAX_WORKERS` | 4 | `swebench.harness` Docker parallelism |

## Phases

The pilot script runs three phases in sequence and bails on the first failure:

### Phase 1 — agent runs

`benches/swebench/run.py` clones each task's repo at `base_commit` into a
fresh tempdir, materializes the per-arm MCP config, spawns
`claude --print --output-format json --strict-mcp-config …` with the task's
`problem_statement` as the prompt, waits for completion (or `--timeout`),
captures the git diff + token stats, and appends one JSONL row per
(task, arm) to `target/swebench/results.jsonl`.

Artifacts per task:

```
target/swebench/<arm>/<instance_id>/
├── claude.json    # raw --output-format json (full usage data)
└── patch.diff     # captured unified diff
```

### Phase 2 — swebench harness eval

`scripts/swebench_eval.py` reads `results.jsonl`, builds a predictions
file in swebench format, and runs `python -m swebench.harness.run_evaluation`
once per arm. The harness spins up one Docker container per task, applies
the patch, runs the pinned test suite, and records pass/fail.

Verdicts are merged back into `target/swebench/<run-id>/results_evaluated.jsonl`
and the top-level `results.jsonl` is overwritten with the augmented copy.

### Phase 3 — render report

`scripts/swebench_report.py --pilot` reads the augmented `results.jsonl`
and prints a markdown report to `benches/swebench/report_pilot.md`:

- Headline table: pass@1 / avg tokens / avg cost / avg walltime × (bare, +codesurgeon, Δ)
- Per-repo breakdown: same columns grouped by repo
- Errors section: any runs with non-zero exit or harness failures

## Interpreting the result

From `#29b` go/no-go gate:

| Check | How to verify |
|---|---|
| Harness stable | `results.jsonl` has 20 rows, all with `exit_code == 0` |
| Directional signal | `+ codesurgeon` pass@1 ≥ bare pass@1 − 10pp |
| Per-task walltime | avg `walltime_s` ≤ 600 (10 minutes) |

All three green → open **#29c** and run the full 100-task version.
Any red → diagnose under `target/swebench/<run-id>/` and iterate in #29b.

## Resuming after failure

The agent runs write results incrementally; if Phase 1 dies after 6 of 10
tasks, `results.jsonl` has 12 rows (6 tasks × 2 arms). Re-running with
`--clean` wipes and restarts from scratch; omit `--clean` to preserve
existing rows and append new ones. The launch script always passes
`--clean` — if you want resume semantics, run `run.py` manually instead.

## Dry-run (no Docker, no OAuth, no cost)

Verify the wiring without touching anything expensive:

```bash
uv run --python 3.14 benches/swebench/run.py --tasks 3 --dry-run
uv run --python 3.14 scripts/swebench_report.py
```

This exercises the command-building, arm iteration, and results.jsonl
append paths without spawning claude or cloning repos.

## Regenerating tasks.json

```bash
uv run --python 3.14 benches/swebench/select_tasks.py --seed 17
```

Different seed gives a different 100-task stratified sample. The committed
`tasks.json` uses seed 17 — do not regenerate casually, since reruns against
different subsets aren't comparable.
