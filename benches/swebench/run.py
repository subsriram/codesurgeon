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

NUDGES: dict[str, str] = {"5b": TREATMENT_NUDGE_5B, "5c": TREATMENT_NUDGE_5C}

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


def build_prompt(arm: str, problem_statement: str, nudge_variant: str = "5b") -> str:
    """Assemble the per-arm prompt.

    Control arm (`without`) gets only the bug-fix preamble — no mention of
    codesurgeon or `run_pipeline`, since the tool isn't available under
    `--strict-mcp-config` with an empty mcpServers map.

    Treatment arm (`with`) gets the preamble + one of the NUDGES keyed by
    `nudge_variant`. Default 5b (verbatim-forward of problem statement
    into `context`); 5c (tool-description-only) is used for A/B isolation.
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

        # 2b. Phase 4 — optionally seed `workdir/CLAUDE.md` with codesurgeon
        # tool guidance so the child Claude Code session auto-loads it at
        # startup. Gated by `inject_claude_md`. Treatment arm only.
        injected_claude_md = maybe_inject_claude_md(workdir, arm, inject_claude_md)

        # 3. Build prompt and spawn claude. Prompt branches on arm — the
        # control arm does not see the codesurgeon nudge (see build_prompt).
        prompt = build_prompt(arm, task["problem_statement"], nudge_variant=nudge_variant)
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
        walltime = time.monotonic() - t0

        # 4. Capture artifacts (diff + raw claude json) into results_dir for
        # downstream evaluation. Named by (arm, instance_id).
        artifact_base = results_dir / arm / task["instance_id"]
        artifact_base.mkdir(parents=True, exist_ok=True)

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

        diff = capture_diff(workdir, exclude_claude_md=injected_claude_md is not None)
        diff_path: Path | None = None
        if diff:
            diff_path = artifact_base / "patch.diff"
            diff_path.write_text(diff)

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
        help="Phase 4: write codesurgeon tool guidance to workdir/CLAUDE.md (with arm only)",
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
