#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""SWE-bench Verified harness driver (issue #29a).

For each (task, arm) pair:

1. Shallow-clones the task repo at ``base_commit`` into a tempdir
2. Materializes a per-arm MCP config (substitutes @CODESURGEON_MCP_BIN@ and
   @CS_WORKSPACE@ in ``mcp_with.json`` / passes ``mcp_without.json`` as-is)
3. Runs ``claude --print --output-format json --strict-mcp-config …`` against
   the checkout with the task's ``problem_statement`` as the prompt
4. Captures: resolved diff (``git diff``), token usage (from JSON output),
   walltime, claude exit code, arm tag, instance id
5. Appends one JSONL record per (task, arm) to ``target/swebench/results.jsonl``

This script only produces agent patches. It does **not** evaluate them against
the swebench test harness — that is deferred to #29b via
``scripts/swebench_report.py`` which invokes ``swebench.harness.run_evaluation``
on the collected diffs.

Usage:
    uv run benches/swebench/run.py --dry-run              # wiring smoke test
    uv run benches/swebench/run.py --tasks 1              # run 1 task x 2 arms
    uv run benches/swebench/run.py --tasks 10 --arms with # just the treatment
    uv run benches/swebench/run.py                        # full 100 x 2

Environment:
    CODESURGEON_MCP_BIN  default: <repo>/target/release/codesurgeon-mcp
    CLAUDE_BIN           default: claude (resolved via PATH)
    CLAUDE_MODEL         default: unset (use Claude Code's default)
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_PATH = REPO_ROOT / "benches" / "swebench" / "tasks.json"
MCP_WITH_TEMPLATE = REPO_ROOT / "benches" / "swebench" / "mcp_with.json"
MCP_WITHOUT_PATH = REPO_ROOT / "benches" / "swebench" / "mcp_without.json"
RESULTS_DIR = REPO_ROOT / "target" / "swebench"
RESULTS_PATH = RESULTS_DIR / "results.jsonl"

DEFAULT_MCP_BIN = REPO_ROOT / "target" / "release" / "codesurgeon-mcp"
DEFAULT_CS_BIN = REPO_ROOT / "target" / "release" / "codesurgeon"
DEFAULT_CLAUDE_BIN = "claude"

# Hard safety caps. #29a is wiring-only; #29b raises these for the pilot run.
DEFAULT_MAX_BUDGET_USD = 1.00
DEFAULT_TASK_TIMEOUT_S = 900  # 15 minutes

# Instruction preamble injected before the task's problem_statement. Keeps the
# agent focused on producing a diff rather than asking clarifying questions.
#
# Split into a shared `PROMPT_BASE` (both arms see this) and an arm-specific
# `TREATMENT_NUDGE` (only the `with` arm, which actually has cs-codesurgeon
# mounted, is told about `run_pipeline`). Previously the nudge was shared,
# which instructed the control arm to call a tool that didn't exist under
# `--strict-mcp-config` — a confound that muddied the A/B.
PROMPT_BASE = """\
You are fixing a real GitHub issue in this repository. Read the problem
statement carefully, inspect the code, and make the minimal change needed to
fix the bug. Do not add new tests. Do not reformat unrelated code. When you
are confident the fix is complete, stop — your changes will be captured as a
git diff and evaluated automatically.
"""

# Phase 3 A/B variants. Selected per-run via `--nudge`.
#
# 5b — verbatim-forward: explicitly tells the agent to paste the raw
#      problem statement into the new `context` param.
# 5c — tool-description-only: no mention of `context` in the nudge at all
#      (relies purely on the MCP server's tool description, which still
#      advertises `context`). Used to measure whether tool-description
#      alone steers the agent without an in-prompt instruction.
TREATMENT_NUDGE_5B = """\

Before you start reading files, call `mcp__cs-codesurgeon__run_pipeline`
with two fields:
  - `task`: a short description of the work, e.g.
    task="fix PolynomialError on subs with Piecewise"
  - `context`: the ENTIRE problem statement below, pasted verbatim
    (copy-paste exactly — do not paraphrase, summarize, or omit code
    snippets, error messages, or identifiers)

`context` is what anchors the search on the function names, class names,
and API calls that appear in the raw source — identifiers you might
otherwise paraphrase out of `task`. The capsule typically returns in
under 200ms and replaces 5–10 exploratory tool calls. Only after you
have the capsule should you start opening files with Read.
"""
# NOTE (Phase 4g, 2026-04-21): a 134-char line advertising
# `get_impact_graph` / `get_skeleton` / `search_logic_flow` was added
# here and empirically regressed sympy-21379 from $0.95/279s/582B to
# $1.73/479s/887B. The agent still only used `run_pipeline` (same as
# baseline) but made 50% more exploratory Grep/Read/Bash calls. Prompt-
# level tool advertising monotonically increased cost across doses
# (0 chars, 134 chars, 2,781 chars from CLAUDE.md) without ever
# triggering chained tool use. Do not add tool-advertising prose here;
# the MCP `tools/list` init event already surfaces tool availability.

TREATMENT_NUDGE_5C = """\

Before you start reading files, call `mcp__cs-codesurgeon__run_pipeline`
with a short description of the task (e.g. `task="fix VLA diff bug in
io.fits.FITSDiff"`). It returns a budgeted capsule of the most relevant
symbols, files, and call-graph edges for the problem — use this to find
the right file to edit instead of doing Grep + Read exploration. The
capsule typically returns in under 200ms and replaces 5–10 exploratory
tool calls. Only after you have the capsule should you start opening
files with Read.
"""

# Phase 4h variants — leverage LLM inference to surface internal fix sites
# the user didn't name. 5b's verbatim-only context delegates anchor
# extraction to server-side regex; both 5e/5f put part of the extraction
# back on the model. Kept minimal (~50–200 char deltas) because prior dose-
# response showed prompt-level content monotonically increases cost across
# 0/134/2781-char experiments. Small deltas may still net positive if the
# model's inference adds signal the server regex can't.

TREATMENT_NUDGE_5E = """\

Before you start reading files, call `mcp__cs-codesurgeon__run_pipeline`
with two fields:
  - `task`: a short description of the work, e.g.
    task="fix PolynomialError on subs with Piecewise"
  - `context`: one symbol name or FQN per line — include both the
    identifiers named in the problem statement AND internal
    functions/classes/modules the error chain implies, even if the
    user didn't name them (e.g. likely fix-site methods).

The capsule surfaces these as anchored pivots. Only after you have
the capsule should you start opening files with Read.
"""

TREATMENT_NUDGE_5F = """\

Before you start reading files, call `mcp__cs-codesurgeon__run_pipeline`
with two fields:
  - `task`: a short description of the work, e.g.
    task="fix PolynomialError on subs with Piecewise"
  - `context`: the ENTIRE problem statement below, pasted verbatim
    (copy-paste exactly — do not paraphrase, summarize, or omit code
    snippets, error messages, or identifiers); then append any
    internal symbols or FQNs the error chain implies.

`context` is what anchors the search on the function names, class names,
and API calls that appear in the raw source — identifiers you might
otherwise paraphrase out of `task`. The capsule typically returns in
under 200ms and replaces 5–10 exploratory tool calls. Only after you
have the capsule should you start opening files with Read.
"""

TREATMENT_NUDGE_5G = """\

Before you start reading files, call `mcp__cs-codesurgeon__run_pipeline`
with two fields:
  - `task`: a short description of the work, e.g. task="fix PolynomialError on subs with Piecewise"
  - `context`: one symbol name or FQN per line — include both the identifiers named in the problem statement or in the error chain..
After receiving the capsule, you can investigate further with the following codesurgeon tools: `get_impact_graph` (callers/raisers of a symbol), `get_skeleton` (file API), `search_logic_flow` (trace A→B).
"""

TREATMENT_NUDGE_5H = """\

Before you start reading files, call mcp__cs-codesurgeon__run_pipeline
with two fields:
  - task: a short description of the work, e.g. task="fix PolynomialError on subs with Piecewise"
  - context: one symbol name or FQN per line — include the identifiers named in the problem statement or in the error chain..
After receiving the capsule, you can investigate further with the following: mcp__cs-codesurgeon__get_impact_graph (callers/raisers of a symbol),  mcp__cs-codesurgeon__get_skeleton (file API),  mcp__cs-codesurgeon__search_logic_flow (trace A→B).
"""

TREATMENT_NUDGE_5I = """\

First, call mcp__cs-codesurgeon__run_pipeline with two fields:
  - task: a short description of the work, e.g. task="fix PolynomialError on subs with Piecewise"
  - context:   symbol names or FQNs named in the problem statement or in the error chain.
With the returned capsule, you can investigate further with: mcp__cs-codesurgeon__get_impact_graph (callers/raisers of a symbol),  mcp__cs-codesurgeon__get_skeleton (file API),  mcp__cs-codesurgeon__search_logic_flow (trace A→B).
"""

TREATMENT_NUDGE_5J = """\

Before you start reading files, call mcp__cs-codesurgeon__run_pipeline with two fields:
1. task: a short description of the work, e.g. task="fix PolynomialError on subs with Piecewise"
2. context: one symbol name or FQN per line — include both the identifiers named in the problem statement AND internal functions/classes/modules the error chain implies that are likely to be fix-site methods, even if the user didn't name them.

After receiving the capsule, you can investigate further with the following cs-codesurgeon tools: get_impact_graph (callers/raisers of a symbol), get_skeleton (file API), search_logic_flow (trace A→B).
"""

TREATMENT_NUDGE_5K = """\

Before you start reading files, call mcp__cs-codesurgeon__run_pipeline with two fields:
1. task: a summary description of the work
2. context: one symbol name or FQN per line — include both the identifiers named in the problem statement AND internal functions/classes/modules on the error chain.

After receiving the capsule, use the following cs-codesurgeon tools: get_impact_graph (callers/raisers of a symbol), get_skeleton (file API), search_logic_flow (trace A→B).
"""

NUDGES: dict[str, str] = {
    "5b": TREATMENT_NUDGE_5B,
    "5c": TREATMENT_NUDGE_5C,
    "5e": TREATMENT_NUDGE_5E,
    "5f": TREATMENT_NUDGE_5F,
    "5g": TREATMENT_NUDGE_5G,
    "5h": TREATMENT_NUDGE_5H,
    "5i": TREATMENT_NUDGE_5I,
    "5j": TREATMENT_NUDGE_5J,
    "5k": TREATMENT_NUDGE_5K,
}

# Empirical results on `sympy__sympy-21379` — each variant, single run,
# same warm workspace, same binary (f1f8157 + #65 + import filter).
# Used to choose `5b` as the default; all other variants kept for
# reproducibility and future re-runs.
#
# IMPORTANT — measured on **claude 2.1.114 / 2.1.117**. The 2.1.119
# update changed `--print`-mode behaviour materially: a fresh n=1 5b
# run on the same workspace produced 81.8 s / $0.79 / 1067 B (3.4×
# faster wall, larger-but-still-canonical patch). Treat the table
# below as a **historical** reference, not a target. Any nudge-tuning
# decision made on its numbers needs to be re-measured on 2.1.119+
# before being acted on.
#
# | Variant | Prompt ch | Wall  | Cost   | Patch  | Notes
# |---------|----------:|------:|-------:|-------:|------
# | 5b      |       716 | 279 s | $0.95  | 582 B  | VERBATIM-ONLY; reference baseline (2.1.114-era)
# | 5g      |       529 | 256 s | $0.99  | 582 B  | grounded inference + short tool names; Pareto-comparable
# | 5i      |       482 | 368 s | $1.34  | 582 B  | tightest imperative + FQN tool names; correct, costlier
# | 5e      |       568 | 474 s | $1.60  | 597 B  | speculative inference; agent guessed `trigsimp` (wrong subtree)
# | (4g)    |       850 | 479 s | $1.73  | 887 B  | 5b + 134-char tool-advertising line (reverted)
# | 5f      |       787 | 600 s | TIMEOUT|   0 B  | verbatim + inferred; wrong-direction inference amplified
# | 5h      |       559 | 600 s | TIMEOUT|   0 B  | FQN tools + "include the" phrasing; agent hallucinated `ask_key`
# | (4f)    |     3,497 | 600 s | TIMEOUT|   0 B  | 5b + full 2,781-char CLAUDE.md (reverted)
#
# Key signals that survive noise (single-run, n=1 per variant) — **on
# 2.1.114/2.1.117**:
#  - 5b and 5g occupy the same success band (~$0.95-$0.99 / ~256-279 s)
#  - Every variant that promotes speculation OR adds tool advertising
#    >~130 chars degraded or timed out
#  - Agent NEVER called get_impact_graph / get_skeleton / search_logic_flow
#    across all 7 variants, regardless of tool-name format or framing
#
# Conclusion (also 2.1.114/2.1.117-era; **re-confirmed on 2.1.119**
# 2026-04-25): prompt-level workflow steering is closed as a lever on
# this task class. Further gains come from server-side capsule content
# (#69 v2: body-text embedding similarity, traceback parsing, etc.).
#
# 2026-04-25 re-validation: nudge `5g` (advertising get_impact_graph
# / get_skeleton / search_logic_flow) was run on 2.1.119 against the
# same warm sympy-21379 workspace. Result: agent made zero chained-
# MCP calls — same Bash + Read + Edit fallback as 2.1.117 era. Wall
# 132.1 s / $1.17 / 1037 B canonical fix; worse than 5b's 81.8 s /
# $0.79 / 1067 B from the same session. The 2.1.119 fix to MCP
# eager-loading (Cause 1) does NOT unlock chained MCP usage — the
# agent's bias toward Bash/Read/Edit is structural. 5b stays the
# default; the 5e/5f/5g/5h/5i/5j/5k variants are kept for archival
# only.

PROMPT_SUFFIX = """

Problem statement:
"""


# Phase 4: optional CLAUDE.md injected into the `with` arm's workdir to
# advertise the codesurgeon tool surface in the location Claude Code
# auto-loads from. Gated behind `--inject-claude-md` so it can be A/B'd
# independently of the PROMPT_PREFIX variants.
CODESURGEON_CLAUDE_MD = """\
# codesurgeon MCP tools — consult before exploring

This repository is indexed by codesurgeon. Before using Read or Grep to
explore the codebase, use these MCP tools to get targeted context:

| Tool | When to use |
|------|-------------|
| `mcp__cs-codesurgeon__run_pipeline` | **First call on any task**. Returns a budgeted capsule of relevant symbols with full source plus call-graph edges. Pass both `task` (short description) and `context` (raw problem statement, verbatim, unmodified). |
| `mcp__cs-codesurgeon__get_impact_graph` | Walks callers / importers / raisers of a symbol. Use when the first capsule named the symptom (an exception class, a public API) but didn't include the function that needs changing. See the chaining note below. |
| `mcp__cs-codesurgeon__get_skeleton` | File API surface without bodies. 70–90% fewer tokens than reading the full file. |
| `mcp__cs-codesurgeon__save_observation` | Persist an insight tied to a symbol for future sessions. |

## Why pass `context` on `run_pipeline`

`task` is the agent's summary; `context` is the raw source the summary was
derived from. Identifiers (function names, class names, dotted API calls)
that you might paraphrase out of `task` are recovered from `context` via
server-side symbol-anchor extraction. BM25, semantic search, and intent
detection still run on `task` alone, so `context` has no effect on query
budget — only on anchor resolution.

## Chain `run_pipeline` → `get_impact_graph` when the bug site is unnamed

Bug reports usually describe symptoms — *"I get `PolynomialError` when I call
`subs()` on a `Piecewise`"* — while the actual fix site is a function that
**raises** or **catches** the error, and that function is almost never named
in the report. `run_pipeline` anchors on the identifiers the user DID
mention (the exception class, the triggering API); the fix site won't be
in the capsule because nothing textually anchors to it.

**Workflow when the first capsule doesn't include an obviously-fixable
symbol**:

1. Pick the most specific identifier from the capsule — typically the
   exception class (e.g. `PolynomialError`) or the user-facing method
   (e.g. `Piecewise`).
2. Call `get_impact_graph` on that symbol's FQN. It returns every function
   that raises it, catches it, or transitively calls a raiser.
3. Scan the dependents for a function name that matches the symptom domain
   (for a `subs()`-related crash, look for `eval`, `doit`, `_eval_subs`, or
   the class named in the triggering call).
4. Open that function with Read — it will usually be the fix site.

This two-call chain typically replaces 10–20 exploratory Grep/Read calls.
If you skip it and default to Grep on the exception name, you'll walk the
same call graph manually and burn tokens.
"""


def maybe_inject_claude_md(workdir: Path, arm: str, inject: bool) -> Path | None:
    """Write (or prepend) codesurgeon guidance to `workdir/CLAUDE.md`.

    Only runs in the `with` arm when `inject` is True. If the task repo
    already ships a CLAUDE.md at `base_commit`, our content is prepended
    so the agent sees codesurgeon guidance first — we never silently
    overwrite upstream instructions.

    NOTE (2026-04-21): `claude --print` does NOT auto-load CLAUDE.md from
    `cwd` — that's an interactive-mode-only behavior. Writing this file
    to disk was a no-op for every prior run: the file sat in the workdir
    but the agent never saw its content. Verified empirically by scanning
    the entire stream for distinctive CLAUDE.md text (0 matches across
    121 events / 467 KB). `build_prompt` now ALSO inlines the same
    content into the treatment-arm prompt prefix when this flag is set,
    which is the mechanism that actually delivers the guidance.

    The on-disk write is retained for audit / debug / symmetry with how
    a human operator would use codesurgeon interactively.
    """
    if arm != "with" or not inject:
        return None
    path = workdir / "CLAUDE.md"
    existing = path.read_text() if path.exists() else ""
    if existing:
        path.write_text(
            CODESURGEON_CLAUDE_MD + "\n\n---\n\n## Upstream CLAUDE.md (preserved)\n\n" + existing
        )
    else:
        path.write_text(CODESURGEON_CLAUDE_MD)
    return path


def build_prompt(
    arm: str,
    problem_statement: str,
    nudge_variant: str = "5b",
) -> str:
    """Assemble the per-arm prompt.

    Control arm (`without`) gets only the bug-fix preamble — no mention of
    codesurgeon or `run_pipeline`, since the tool isn't available under
    `--strict-mcp-config` with an empty mcpServers map.

    Treatment arm (`with`) gets the preamble + one of the NUDGES keyed by
    `nudge_variant`. Default 5b (verbatim-forward of problem statement
    into `context`); 5c (tool-description-only) is used for A/B isolation.

    CLAUDE.md guidance for the treatment arm is delivered **on-disk only**
    via `maybe_inject_claude_md` (gated on `--inject-claude-md`). The
    in-prompt inline path was removed on 2026-04-25 because `claude
    --print` auto-loads workdir/CLAUDE.md on 2.1.119+ — keeping the
    inline as well would double-inject the same content. On 2.1.114 the
    inline was the only delivery path; if you need to support that
    legacy version, revive the inline branch (see git history) or pin
    `CLAUDE_BIN=2.1.114` and accept that on-disk CLAUDE.md will be
    ignored by claude.
    """
    parts = [PROMPT_BASE]
    if arm == "with":
        if nudge_variant not in NUDGES:
            raise ValueError(
                f"unknown nudge_variant {nudge_variant!r}; expected one of {list(NUDGES)}"
            )
        parts.append(NUDGES[nudge_variant])
    parts.append(PROMPT_SUFFIX)
    parts.append(problem_statement)
    return "".join(parts)


@dataclass
class RunResult:
    instance_id: str
    repo: str
    arm: str  # "with" | "without"
    exit_code: int
    walltime_s: float
    input_tokens: int | None
    output_tokens: int | None
    cache_creation_tokens: int | None
    cache_read_tokens: int | None
    total_cost_usd: float | None
    diff_bytes: int
    diff_path: str | None
    claude_json_path: str | None
    error: str | None


def load_tasks(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["tasks"]


def materialize_mcp_with(mcp_bin: Path, workspace: Path, tmp: Path) -> Path:
    """Render mcp_with.json with real paths substituted."""
    template = MCP_WITH_TEMPLATE.read_text()
    rendered = (
        template.replace("@CODESURGEON_MCP_BIN@", str(mcp_bin))
        .replace("@CS_WORKSPACE@", str(workspace))
    )
    # Validate we produced valid JSON.
    json.loads(rendered)
    out = tmp / "mcp_with.json"
    out.write_text(rendered)
    return out


def mcp_preflight(mcp_bin: Path, workspace: Path, timeout_s: int = 5) -> tuple[bool, str, list[str]]:
    """Verify ``codesurgeon-mcp`` is launchable and advertises its tools.

    Spawns the binary against a **disposable empty tempdir** (not the
    task workspace), sends NDJSON-framed ``initialize`` + ``tools/list``
    requests, reads the response, and **SIGKILLs** the server to prevent
    background indexing from leaving orphan processes that would
    contend for the real workspace's pid lock.

    Critical design choice — why the disposable workspace:
      A previous version pointed preflight at the actual task workspace
      and let the server exit naturally when stdin EOF'd. In practice:
        - The server's main stdio loop exits on EOF
        - The server's background-indexing thread does not — it keeps
          parsing + re-embedding the (~1,500-file sympy) workspace
        - Process remains alive at 100% CPU for minutes after
          subprocess.run returns
        - Subsequent ``claude --print`` launches a second MCP against
          the same workspace, which contends with the zombie over SQLite
          and falls into secondary mode where tools aren't eagerly
          advertised at init
        - Agent's init event shows ``mcp_servers: []`` — the exact
          failure preflight was supposed to prevent
      Preflighting against an empty tempdir means the indexer finds no
      source files, finishes immediately, and the subprocess exits
      cleanly. We belt-and-suspenders with ``Popen + communicate +
      kill`` to guarantee no zombie survives preflight.

    ``tools/list`` responses are workspace-independent — the server
    advertises the same tool set regardless of which workspace it
    points at — so the disposable tempdir is a valid test of binary
    health and schema availability.

    Returns ``(ok, message, tool_names)``. Fast-fail on timeout / spawn
    error / missing tools.
    """
    init_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "swebench-harness-preflight", "version": "0"},
        },
    })
    tools_req = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    payload = f"{init_req}\n{tools_req}\n"
    # `workspace` arg kept for API symmetry — we ignore it and use an
    # empty tempdir instead. The two lints below stop formatters from
    # collapsing the unused-arg guard.
    _ = workspace  # noqa: F841

    with tempfile.TemporaryDirectory(prefix="cs-preflight-") as td:
        proc = subprocess.Popen(
            [str(mcp_bin)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "CS_WORKSPACE": td},
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(input=payload, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            return False, f"preflight timed out after {timeout_s}s", []
        except FileNotFoundError:
            return False, f"mcp binary not found: {mcp_bin}", []
        finally:
            # Guarantee no orphan even on happy path — the server's
            # background threads may outlive stdio loop exit. We have
            # the NDJSON response we need; kill aggressively.
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

    # Parse NDJSON responses — look for the tools/list result (id=2).
    tools: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict) and obj.get("id") == 2 and "result" in obj:
            for t in obj["result"].get("tools", []) or []:
                name = t.get("name")
                if isinstance(name, str):
                    tools.append(name)

    if not tools:
        tail = (stderr or "")[-500:]
        return (
            False,
            f"no tools/list response from codesurgeon-mcp; stderr tail: {tail!r}",
            [],
        )
    if not any("run_pipeline" in t for t in tools):
        return (
            False,
            f"run_pipeline missing from tools/list (got {len(tools)}: {tools[:5]}...)",
            tools,
        )
    return True, f"verified {len(tools)} tools incl. run_pipeline", tools


def mcp_sidecar_start(
    mcp_bin: Path, workspace: Path, ready_timeout_s: int = 30
) -> tuple[subprocess.Popen | None, str]:
    """Spawn ``codesurgeon-mcp`` against ``workspace`` as a primary daemon,
    wait for its engine to be ready, then leave it running.

    When ``claude --print`` later spawns its own MCP via ``--mcp-config``,
    the sidecar already holds the PID lock (``.codesurgeon/mcp.pid``).
    Claude's process falls into the server's "secondary mode" branch in
    ``cs-mcp::main`` — secondary mode runs ``CoreEngine::new`` synchronously
    (``without_embedder()``, so it's fast) and populates ``cell`` before
    starting the stdio loop. That eliminates the race where the first
    ``tools/call run_pipeline`` arrives before the engine is ready and
    returns the ``⏳ Engine still initializing`` placeholder.

    TODO (post-2.1.119 update): the sidecar was added to dodge an
    initialize-vs-engine-ready race observed on 2.1.117. If 2.1.119
    serialises ``initialize`` against engine readiness internally (or
    if the race never manifested on 2.1.119 to begin with), the
    sidecar is dead weight — costs ~0.2-1.4 s per run and one extra
    PID-lock holder. To measure: run with the sidecar code path
    bypassed (early-return ``(None, "sidecar disabled")``) and confirm
    no ``⏳ Engine still initializing`` placeholders in the first
    ``tools/call run_pipeline`` result across n=3 runs. If clean,
    drop the sidecar or gate it behind ``--use-sidecar`` (default off
    on 2.1.119+).

    Returns ``(Popen, message)``. Returns ``(None, err_msg)`` if the
    sidecar failed to reach engine-ready within ``ready_timeout_s``; the
    caller should abort the run in that case. On success, the caller is
    responsible for killing the Popen in a ``finally`` block after
    ``claude --print`` returns.

    Why a primary-mode sidecar (not two secondaries):
      - Only the primary does background indexing. A schema-mismatched
        workspace still gets re-indexed in the background while claude's
        secondary serves warm from SQLite — behaviour unchanged from today.
      - Secondary mode is synchronous + embedder-skipping, so it's fast
        exactly when we need it (during the agent's first tool call).
    """
    init_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "swebench-harness-sidecar", "version": "0"},
        },
    })
    status_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "index_status", "arguments": {}},
    })

    proc = subprocess.Popen(
        [str(mcp_bin)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "CS_WORKSPACE": str(workspace)},
        text=True,
    )

    # Kick off the handshake. `initialize` always returns fast (doesn't
    # wait for engine). We then poll `tools/call index_status` — it
    # returns the "Engine still initializing" placeholder while the
    # engine is loading, and the real status once `cell` is populated.
    try:
        proc.stdin.write(init_req + "\n")
        proc.stdin.flush()
    except BrokenPipeError:
        proc.kill()
        proc.wait(timeout=5)
        return None, "sidecar: broken pipe writing initialize"

    # Drain the initialize response (we don't need its contents — just want
    # to move past it in the response stream).
    init_line = proc.stdout.readline()
    if not init_line:
        proc.kill()
        proc.wait(timeout=5)
        stderr_tail = proc.stderr.read()[-500:] if proc.stderr else ""
        return None, f"sidecar: no initialize response; stderr: {stderr_tail!r}"

    import time as _time

    start = _time.monotonic()
    poll_count = 0
    while _time.monotonic() - start < ready_timeout_s:
        poll_count += 1
        try:
            proc.stdin.write(status_req + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            break
        # We send a new id=2 each round but since the server's id doesn't
        # have to be unique from the client's side we just re-read the
        # next line and parse. If the response is the placeholder, keep
        # polling; if it's real, we're ready.
        line = proc.stdout.readline()
        if not line:
            break
        try:
            obj = json.loads(line.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        result = obj.get("result") or {}
        content = result.get("content") or []
        text = content[0].get("text", "") if content else ""
        if "Engine still initializing" in text:
            _time.sleep(0.2)
            continue
        # Real response — engine is ready.
        elapsed = _time.monotonic() - start
        return proc, f"sidecar ready after {elapsed:.1f}s, {poll_count} polls"

    # Timed out. Kill and surface the error.
    stderr_tail = ""
    if proc.stderr:
        try:
            stderr_tail = proc.stderr.read(2000) or ""
        except Exception:
            stderr_tail = ""
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return (
        None,
        f"sidecar: engine not ready within {ready_timeout_s}s; stderr tail: {stderr_tail[-500:]!r}",
    )


def mcp_sidecar_stop(proc: subprocess.Popen) -> None:
    """Kill the sidecar MCP started by ``mcp_sidecar_start``.

    Closes stdin (lets the stdio loop exit cleanly if it can) then
    SIGKILLs after a brief grace period to guarantee background threads
    don't keep the process alive.
    """
    try:
        if proc.stdin:
            proc.stdin.close()
    except Exception:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass


def git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def clone_task_repo(task: dict, dest: Path) -> None:
    """Shallow clone the task's repo at base_commit into dest.

    Uses ``git fetch --depth 1`` of the exact base_commit SHA to avoid
    pulling the full history. This is ~10x faster than a full clone.
    """
    repo_url = f"https://github.com/{task['repo']}.git"
    base_commit = task["base_commit"]

    dest.mkdir(parents=True, exist_ok=True)
    git(["init", "--quiet"], cwd=dest)
    git(["remote", "add", "origin", repo_url], cwd=dest)
    # Some providers don't allow single-commit fetches without unshallow;
    # fall back to a deeper fetch if needed.
    try:
        git(["fetch", "--depth", "1", "origin", base_commit], cwd=dest)
    except subprocess.CalledProcessError:
        git(["fetch", "--depth", "50", "origin"], cwd=dest)
    git(["checkout", "--quiet", base_commit], cwd=dest)


def build_claude_cmd(
    claude_bin: str,
    mcp_config: Path,
    workdir: Path,
    prompt: str,
    max_budget_usd: float,
    model: str | None,
    stream_json: bool = False,
) -> list[str]:
    """Assemble the ``claude --print`` command for one task run.

    When ``stream_json`` is True, the output format is switched to
    ``stream-json`` (NDJSON of per-turn events). The final line in the
    stream carries the same ``result`` / ``usage`` / ``total_cost_usd``
    fields as the single-object ``json`` format, so downstream token
    extraction still works. Use stream_json when you need per-tool-call
    visibility (e.g. to confirm an agent populated a new MCP param).
    """
    cmd = [
        claude_bin,
        "--print",
    ]
    if stream_json:
        # stream-json requires --verbose per Claude Code's CLI contract.
        cmd.extend(["--output-format", "stream-json", "--verbose"])
    else:
        cmd.extend(["--output-format", "json"])
    cmd.extend([
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_config),
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
        "--add-dir",
        str(workdir),
        "--max-budget-usd",
        f"{max_budget_usd:.2f}",
    ])
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    return cmd


def parse_claude_json(stdout: str, stream_json: bool = False) -> dict:
    """Extract the structured result from ``claude --print`` output.

    ``json`` mode: stdout is a single top-level object; parse and return.
    ``stream-json`` mode: stdout is NDJSON of per-turn events. The last
    line of type ``result`` carries the same summary fields; return that
    so downstream code sees the same shape as json mode.
    Defensively returns an empty dict on parse failure so the caller can
    degrade gracefully and still capture the diff.
    """
    if not stdout.strip():
        return {}
    if stream_json:
        # Walk lines in reverse to find the last "result" event.
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and obj.get("type") == "result":
                return obj
        return {}
    try:
        return json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return {}


def extract_token_stats(claude_json: dict) -> dict[str, int | float | None]:
    """Pull token and cost figures out of the parsed claude json.

    Schema varies by Claude Code version. Check several known field paths
    and fall back to None if missing — downstream consumers (the report
    script) degrade gracefully when fields are unavailable.
    """
    usage = claude_json.get("usage") or claude_json.get("token_usage") or {}
    cost = claude_json.get("total_cost_usd") or claude_json.get("cost_usd")

    return {
        "input_tokens": usage.get("input_tokens")
        or usage.get("prompt_tokens"),
        "output_tokens": usage.get("output_tokens")
        or usage.get("completion_tokens"),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_tokens": usage.get("cache_read_input_tokens"),
        "total_cost_usd": cost,
    }


def capture_diff(workdir: Path, exclude_claude_md: bool = False) -> str:
    """Return the uncommitted changes as a unified diff.

    Excludes ``.codesurgeon/`` because the pre-index step writes the
    symbol index there inside the task workdir, and we don't want that
    noise in the patch sent to the swebench harness. The harness's
    ``git apply`` chokes on binary blobs (index.db-shm, index.db-wal)
    and even if it didn't, a stray cache dir in the patched repo is
    poison for test stability.

    When ``exclude_claude_md`` is True (set by the harness when it
    injected a CLAUDE.md into the workdir via ``--inject-claude-md``),
    also excludes ``CLAUDE.md`` at the workdir root. Without this, a
    run that times out before the agent made any real edits still
    produces a non-empty diff (just the harness-written CLAUDE.md),
    which is a false positive. Note this also hides agent edits to an
    upstream-shipped CLAUDE.md, which swebench evaluation does not test
    for, so the trade-off is acceptable.

    The ``git add -N .`` makes untracked files visible to ``git diff``
    so new source files the agent creates show up; the pathspec
    exclusions keep harness-owned paths out regardless.
    """
    subprocess.run(
        ["git", "add", "-N", "."],
        cwd=workdir,
        check=False,
        capture_output=True,
    )
    pathspecs = [".", ":(exclude).codesurgeon"]
    if exclude_claude_md:
        pathspecs.append(":(exclude)CLAUDE.md")
    result = subprocess.run(
        ["git", "diff", "--no-color", "--", *pathspecs],
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout


def run_one(
    task: dict,
    arm: str,
    claude_bin: str,
    mcp_bin: Path,
    cs_bin: Path,
    max_budget_usd: float,
    timeout_s: int,
    model: str | None,
    dry_run: bool,
    results_dir: Path,
    inject_claude_md: bool = False,
    nudge_variant: str = "5b",
    reuse_workdir: Path | None = None,
    stream_json: bool = False,
) -> RunResult:
    """Run one (task, arm) pair and return the captured result.

    When `reuse_workdir` is set, skip clone + index and use that path as
    the workdir directly. Before each run we `git reset --hard
    <base_commit>` and `git clean -fdx -e .codesurgeon/` so the agent
    starts from the pristine base state without re-indexing. Meant for
    rapid iteration against a pre-indexed warm workspace.
    """
    print(f"  [{arm:7s}] {task['instance_id']}", file=sys.stderr, end="", flush=True)

    import contextlib

    # Either a real TemporaryDirectory (normal flow) or a null context
    # wrapping the reuse-workdir's parent (so the `tmp` Path still exists
    # for mcp_with.json materialization).
    _ctx: contextlib.AbstractContextManager[str] = (
        contextlib.nullcontext(str(reuse_workdir.parent))
        if reuse_workdir is not None
        else tempfile.TemporaryDirectory(prefix=f"cs-swe-{task['instance_id']}-{arm}-")
    )
    with _ctx as tmp_s:
        tmp = Path(tmp_s)
        workdir = reuse_workdir if reuse_workdir is not None else tmp / "repo"

        # 1. Materialize MCP config for this arm. For the treatment arm we
        # point CS_WORKSPACE at the per-task workdir (set below, after clone),
        # not at the codesurgeon repo — pointing it at REPO_ROOT would make
        # run_pipeline return capsules from codesurgeon's own source code,
        # which is what #29b's first pilot was silently doing.
        mcp_config: Path | None = None
        if arm == "without":
            mcp_config = MCP_WITHOUT_PATH

        # 2. Clone the task repo (skipped in dry-run and reuse-workdir).
        if dry_run:
            workdir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "--quiet"], cwd=workdir, check=True)
            (workdir / "README.md").write_text(f"# dry-run stub for {task['instance_id']}\n")
            subprocess.run(["git", "add", "."], cwd=workdir, check=True)
            subprocess.run(
                ["git", "-c", "user.email=bench@codesurgeon.dev", "-c", "user.name=bench", "commit", "-q", "-m", "stub"],
                cwd=workdir,
                check=True,
            )
        elif reuse_workdir is not None:
            # Reset the reuse workdir to `base_commit` and clean untracked
            # files from prior runs, but preserve `.codesurgeon/` so the
            # warm index carries over.
            try:
                subprocess.run(
                    ["git", "-C", str(workdir), "reset", "--hard", task["base_commit"]],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["git", "-C", str(workdir), "clean", "-fdx", "-e", ".codesurgeon"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"  FAIL (reuse-reset)", file=sys.stderr)
                return RunResult(
                    instance_id=task["instance_id"],
                    repo=task["repo"],
                    arm=arm,
                    exit_code=-1,
                    walltime_s=0.0,
                    input_tokens=None,
                    output_tokens=None,
                    cache_creation_tokens=None,
                    cache_read_tokens=None,
                    total_cost_usd=None,
                    diff_bytes=0,
                    diff_path=None,
                    claude_json_path=None,
                    error=f"reuse-reset failed: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}",
                )
        else:
            try:
                clone_task_repo(task, workdir)
            except subprocess.CalledProcessError as e:
                print(f"  FAIL (clone)", file=sys.stderr)
                return RunResult(
                    instance_id=task["instance_id"],
                    repo=task["repo"],
                    arm=arm,
                    exit_code=-1,
                    walltime_s=0.0,
                    input_tokens=None,
                    output_tokens=None,
                    cache_creation_tokens=None,
                    cache_read_tokens=None,
                    total_cost_usd=None,
                    diff_bytes=0,
                    diff_path=None,
                    claude_json_path=None,
                    error=f"clone failed: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}",
                )

        # 2a. Treatment arm — pre-index the task repo and render the MCP
        # config to point CS_WORKSPACE at the workdir. Pre-indexing is
        # synchronous and fast (sub-second for repos up to ~2000 files);
        # without it the first run_pipeline call blocks on cold indexing
        # and hits the task timeout.
        #
        # Skipped when `reuse_workdir` is set — the `.codesurgeon/` dir
        # inside the reuse path is assumed to be an up-to-date index from
        # a prior run.
        if arm == "with" and not dry_run and reuse_workdir is not None:
            mcp_config = materialize_mcp_with(mcp_bin, workdir, tmp)
        elif arm == "with" and not dry_run:
            t_idx = time.monotonic()
            idx_proc = subprocess.run(
                [str(cs_bin), "index", "--workspace", str(workdir)],
                env={**os.environ, "CS_WORKSPACE": str(workdir)},
                capture_output=True,
                text=True,
                timeout=600,
            )
            idx_wall = time.monotonic() - t_idx
            if idx_proc.returncode != 0:
                print(f"  INDEX-FAIL wall={idx_wall:.1f}s", file=sys.stderr)
                return RunResult(
                    instance_id=task["instance_id"],
                    repo=task["repo"],
                    arm=arm,
                    exit_code=-3,
                    walltime_s=idx_wall,
                    input_tokens=None,
                    output_tokens=None,
                    cache_creation_tokens=None,
                    cache_read_tokens=None,
                    total_cost_usd=None,
                    diff_bytes=0,
                    diff_path=None,
                    claude_json_path=None,
                    error=f"index failed: {idx_proc.stderr[-500:]}",
                )
            mcp_config = materialize_mcp_with(mcp_bin, workdir, tmp)
        elif arm == "with" and dry_run:
            mcp_config = materialize_mcp_with(mcp_bin, workdir, tmp)

        assert mcp_config is not None

        # 2a.5. Preflight — verify codesurgeon-mcp comes up and advertises
        # its tools BEFORE we spawn claude --print. Without this, runs
        # where MCP fails to come up in time end up silently running the
        # agent with zero cs-codesurgeon tool access (agent's init event
        # has `mcp_servers: []`), invalidating the whole measurement.
        # Observed 1 in 6 saved streams in the 2026-04-21 session.
        #
        # Only in the treatment arm with a live MCP — skip for control
        # and dry-run. Takes < 2s when the MCP starts cleanly.
        if arm == "with" and not dry_run:
            ok, msg, tools = mcp_preflight(mcp_bin, workdir)
            if not ok:
                print(f"  MCP-PREFLIGHT-FAIL: {msg}", file=sys.stderr)
                return RunResult(
                    instance_id=task["instance_id"],
                    repo=task["repo"],
                    arm=arm,
                    exit_code=-4,
                    walltime_s=0.0,
                    input_tokens=None,
                    output_tokens=None,
                    cache_creation_tokens=None,
                    cache_read_tokens=None,
                    total_cost_usd=None,
                    diff_bytes=0,
                    diff_path=None,
                    claude_json_path=None,
                    error=f"mcp preflight failed: {msg}",
                )

        # 2a.6. MCP sidecar — spawn codesurgeon-mcp as a primary daemon
        # against the real workspace BEFORE claude --print starts. The
        # sidecar holds the pid lock and its engine is fully warmed when
        # claude --print spawns its own MCP. Claude's MCP detects the
        # lock and takes the secondary-mode path (synchronous engine
        # init, no background indexing), so `cell` is populated before
        # `run_stdio_loop` accepts the first `tools/call`. This bypasses
        # the race where the agent's first `run_pipeline` could arrive
        # before `CoreEngine::new` finished and receive the
        # "⏳ Engine still initializing" placeholder.
        #
        # Treatment arm only. Killed in the `finally` block regardless
        # of how the claude invocation exits.
        sidecar: subprocess.Popen | None = None
        if arm == "with" and not dry_run:
            sidecar, sidecar_msg = mcp_sidecar_start(mcp_bin, workdir)
            if sidecar is None:
                print(f"  MCP-SIDECAR-FAIL: {sidecar_msg}", file=sys.stderr)
                return RunResult(
                    instance_id=task["instance_id"],
                    repo=task["repo"],
                    arm=arm,
                    exit_code=-5,
                    walltime_s=0.0,
                    input_tokens=None,
                    output_tokens=None,
                    cache_creation_tokens=None,
                    cache_read_tokens=None,
                    total_cost_usd=None,
                    diff_bytes=0,
                    diff_path=None,
                    claude_json_path=None,
                    error=f"mcp sidecar failed: {sidecar_msg}",
                )
            print(f"  [sidecar: {sidecar_msg}]", file=sys.stderr, end="", flush=True)

        # 2b. Phase 4 — optionally seed `workdir/CLAUDE.md` with codesurgeon
        # tool guidance so the child Claude Code session auto-loads it at
        # startup. Gated by `inject_claude_md`. Treatment arm only.
        injected_claude_md = maybe_inject_claude_md(workdir, arm, inject_claude_md)

        # 3. Build prompt and spawn claude. Prompt branches on arm — the
        # control arm does not see the codesurgeon nudge (see build_prompt).
        prompt = build_prompt(
            arm,
            task["problem_statement"],
            nudge_variant=nudge_variant,
        )
        cmd = build_claude_cmd(
            claude_bin, mcp_config, workdir, prompt, max_budget_usd, model,
            stream_json=stream_json,
        )

        if injected_claude_md:
            print(
                f"  (injected CLAUDE.md: {injected_claude_md.relative_to(workdir)})",
                file=sys.stderr,
                end="",
                flush=True,
            )

        if dry_run:
            # Don't actually spawn — just verify the command shape.
            print(f"  DRY-RUN cmd-len={len(cmd)}", file=sys.stderr)
            return RunResult(
                instance_id=task["instance_id"],
                repo=task["repo"],
                arm=arm,
                exit_code=0,
                walltime_s=0.0,
                input_tokens=None,
                output_tokens=None,
                cache_creation_tokens=None,
                cache_read_tokens=None,
                total_cost_usd=None,
                diff_bytes=0,
                diff_path=None,
                claude_json_path=None,
                error=None,
            )

        t0 = time.monotonic()
        try:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    cwd=workdir,
                )
                exit_code = proc.returncode
                stdout = proc.stdout
                stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
                err_msg = None if exit_code == 0 else f"exit={exit_code}; stderr-tail={stderr_tail}"
            except subprocess.TimeoutExpired as e:
                # Preserve whatever was captured before the kill so the partial
                # stream (stream-json mode) can be inspected post-mortem —
                # otherwise a timed-out run is a total black box.
                #
                # `subprocess.TimeoutExpired.stdout` / `.stderr` are **bytes**
                # on Python 3.14, regardless of `text=True` on `run()`. (A
                # completed `proc.stdout` under `text=True` is `str`; the
                # exception attrs are not decoded by the same path.) Decode
                # defensively so downstream code that expects `str` (e.g.
                # `Path.write_text`) doesn't crash.
                def _decode(x: object) -> str:
                    if x is None:
                        return ""
                    if isinstance(x, bytes):
                        return x.decode("utf-8", errors="replace")
                    return x  # type: ignore[return-value]

                exit_code = -2
                stdout = _decode(e.stdout)
                partial_stderr = _decode(e.stderr)
                stderr_tail = partial_stderr[-2000:]
                err_msg = (
                    f"timeout after {timeout_s}s "
                    f"(captured {len(stdout)} B stdout, {len(partial_stderr)} B stderr)"
                )
        finally:
            # Always reap the sidecar MCP, even on timeout / exception.
            # Leaking the sidecar would leave orphan codesurgeon-mcp
            # processes holding the pid lock, poisoning subsequent runs.
            if sidecar is not None:
                mcp_sidecar_stop(sidecar)
                sidecar = None
        walltime = time.monotonic() - t0

        # 4. Capture artifacts (diff + raw claude json) into results_dir for
        # downstream evaluation. Named by (arm, instance_id).
        #
        # Every artifact is written twice:
        #   - `claude.json` / `claude_stream.jsonl` / `patch.diff` — the
        #     canonical "latest run" names used by downstream tooling and
        #     by report scripts that don't care about history.
        #   - `archive/<isotime>_*` — permanent per-run copies so
        #     subsequent runs don't overwrite them. Preserves the stream
        #     for every configuration tried on the same task.
        artifact_base = results_dir / arm / task["instance_id"]
        artifact_base.mkdir(parents=True, exist_ok=True)
        archive_dir = artifact_base / "archive"
        archive_dir.mkdir(exist_ok=True)
        run_ts = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "-")

        claude_json_path: Path | None = None
        if stdout:
            # In stream-json mode we keep the raw NDJSON (one event per
            # line) under `claude_stream.jsonl` so per-turn tool calls can
            # be inspected later. In plain json mode, the single-object
            # summary goes to `claude.json`. The field in RunResult keeps
            # the plain name so downstream consumers don't branch.
            artifact_name = "claude_stream.jsonl" if stream_json else "claude.json"
            claude_json_path = artifact_base / artifact_name
            claude_json_path.write_text(stdout)
            # Archive a timestamped copy.
            shutil.copy2(claude_json_path, archive_dir / f"{run_ts}_{artifact_name}")

        diff = capture_diff(workdir, exclude_claude_md=injected_claude_md is not None)
        diff_path: Path | None = None
        if diff:
            diff_path = artifact_base / "patch.diff"
            diff_path.write_text(diff)
            shutil.copy2(diff_path, archive_dir / f"{run_ts}_patch.diff")

        # 5. Parse token stats.
        claude_json = parse_claude_json(stdout, stream_json=stream_json) if stdout else {}
        tokens = extract_token_stats(claude_json)

        result = RunResult(
            instance_id=task["instance_id"],
            repo=task["repo"],
            arm=arm,
            exit_code=exit_code,
            walltime_s=walltime,
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
            cache_creation_tokens=tokens["cache_creation_tokens"],
            cache_read_tokens=tokens["cache_read_tokens"],
            total_cost_usd=tokens["total_cost_usd"],
            diff_bytes=len(diff.encode("utf-8")),
            diff_path=str(diff_path.relative_to(REPO_ROOT)) if diff_path else None,
            claude_json_path=str(claude_json_path.relative_to(REPO_ROOT)) if claude_json_path else None,
            error=err_msg,
        )

        status = "OK" if exit_code == 0 else "FAIL"
        print(
            f"  {status} wall={walltime:.1f}s diff={result.diff_bytes}B tok={tokens['input_tokens']}/{tokens['output_tokens']}",
            file=sys.stderr,
        )
        return result


def append_result(path: Path, result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=None, help="run only first N tasks")
    parser.add_argument(
        "--instance-ids",
        default=None,
        help="comma-separated instance_ids to run (overrides --tasks ordering, preserves list order)",
    )
    parser.add_argument(
        "--arms",
        default="with,without",
        help="comma-separated arms to run (default: with,without)",
    )
    parser.add_argument("--dry-run", action="store_true", help="skip git clone + claude invocation")
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=DEFAULT_MAX_BUDGET_USD,
        help=f"per-task claude budget cap (default: {DEFAULT_MAX_BUDGET_USD})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TASK_TIMEOUT_S,
        help=f"per-task wallclock timeout in seconds (default: {DEFAULT_TASK_TIMEOUT_S})",
    )
    parser.add_argument("--model", default=os.environ.get("CLAUDE_MODEL"), help="override claude model")
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help=f"results.jsonl output path (default: {RESULTS_PATH.relative_to(REPO_ROOT)})",
    )
    parser.add_argument("--clean", action="store_true", help="truncate results.jsonl before running")
    parser.add_argument(
        "--inject-claude-md",
        action="store_true",
        help=(
            "Phase 4: deliver codesurgeon tool guidance to the agent "
            "(treatment arm only). Inlines the CLAUDE.md body into the "
            "prompt prefix AND writes workdir/CLAUDE.md as an audit "
            "artifact. `claude --print` does not auto-load CLAUDE.md; "
            "the prompt-inline path is what actually reaches the agent."
        ),
    )
    parser.add_argument(
        "--nudge",
        choices=sorted(NUDGES.keys()),
        default="5b",
        help="treatment-arm PROMPT_PREFIX variant: 5b = verbatim-forward of context (default), 5c = tool-description-only",
    )
    parser.add_argument(
        "--reuse-workdir",
        type=Path,
        default=None,
        help="path to a pre-indexed checkout to reuse (skip clone + index; `git reset --hard <base_commit>` between runs, preserve .codesurgeon/). Only valid with a single instance_id.",
    )
    parser.add_argument(
        "--stream-json",
        action="store_true",
        help="use claude --output-format stream-json --verbose, save raw NDJSON as claude_stream.jsonl (per-turn tool-call visibility; useful for confirming a new MCP param was populated)",
    )
    args = parser.parse_args()

    if args.reuse_workdir is not None:
        if not args.reuse_workdir.is_dir():
            print(f"--reuse-workdir not a directory: {args.reuse_workdir}", file=sys.stderr)
            return 2
        if not (args.reuse_workdir / ".codesurgeon").is_dir():
            print(
                f"--reuse-workdir missing .codesurgeon/ (is it indexed?): {args.reuse_workdir}",
                file=sys.stderr,
            )
            return 2
        # Resolve to absolute so mcp_with.json lands at an absolute path.
        # Otherwise claude --print spawned with cwd=workdir would try to
        # resolve the relative --mcp-config path against the workdir,
        # producing a double-nested path that doesn't exist.
        args.reuse_workdir = args.reuse_workdir.resolve()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("with", "without"):
            print(f"unknown arm: {a}", file=sys.stderr)
            return 2

    claude_bin = os.environ.get("CLAUDE_BIN", DEFAULT_CLAUDE_BIN)
    if shutil.which(claude_bin) is None and not args.dry_run:
        print(f"claude binary not found: {claude_bin}", file=sys.stderr)
        return 2

    mcp_bin = Path(os.environ.get("CODESURGEON_MCP_BIN", str(DEFAULT_MCP_BIN)))
    if "with" in arms and not args.dry_run and not mcp_bin.exists():
        print(f"codesurgeon-mcp binary not found: {mcp_bin}", file=sys.stderr)
        print("  build it: cargo build --release --features metal", file=sys.stderr)
        return 2

    cs_bin = Path(os.environ.get("CODESURGEON_BIN", str(DEFAULT_CS_BIN)))
    if "with" in arms and not args.dry_run and not cs_bin.exists():
        print(f"codesurgeon CLI binary not found: {cs_bin}", file=sys.stderr)
        print("  build it: cargo build --release --features metal", file=sys.stderr)
        return 2

    tasks = load_tasks(TASKS_PATH)
    if args.instance_ids:
        wanted = [s.strip() for s in args.instance_ids.split(",") if s.strip()]
        by_id = {t["instance_id"]: t for t in tasks}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            print(f"unknown instance_ids: {missing}", file=sys.stderr)
            return 2
        tasks = [by_id[i] for i in wanted]
    elif args.tasks is not None:
        tasks = tasks[: args.tasks]

    print(
        f"running {len(tasks)} tasks × {len(arms)} arms ({'dry-run' if args.dry_run else 'live'}) → {args.results.relative_to(REPO_ROOT) if args.results.is_relative_to(REPO_ROOT) else args.results}",
        file=sys.stderr,
    )

    if args.clean and args.results.exists():
        args.results.unlink()

    results_dir = args.results.parent

    for task in tasks:
        for arm in arms:
            result = run_one(
                task=task,
                arm=arm,
                claude_bin=claude_bin,
                mcp_bin=mcp_bin,
                cs_bin=cs_bin,
                max_budget_usd=args.max_budget_usd,
                timeout_s=args.timeout,
                model=args.model,
                dry_run=args.dry_run,
                results_dir=results_dir,
                inject_claude_md=args.inject_claude_md,
                nudge_variant=args.nudge,
                reuse_workdir=args.reuse_workdir,
                stream_json=args.stream_json,
            )
            append_result(args.results, result)

    print(f"done → {args.results}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
