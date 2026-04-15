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
DEFAULT_CLAUDE_BIN = "claude"

# Hard safety caps. #29a is wiring-only; #29b raises these for the pilot run.
DEFAULT_MAX_BUDGET_USD = 1.00
DEFAULT_TASK_TIMEOUT_S = 900  # 15 minutes

# Instruction prefix injected before the task's problem_statement. Keeps the
# agent focused on producing a diff rather than asking clarifying questions.
PROMPT_PREFIX = """\
You are fixing a real GitHub issue in this repository. Read the problem
statement carefully, inspect the code, and make the minimal change needed to
fix the bug. Do not add new tests. Do not reformat unrelated code. When you
are confident the fix is complete, stop — your changes will be captured as a
git diff and evaluated automatically.

Problem statement:
"""


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
) -> list[str]:
    """Assemble the ``claude --print`` command for one task run."""
    cmd = [
        claude_bin,
        "--print",
        "--output-format",
        "json",
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
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    return cmd


def parse_claude_json(stdout: str) -> dict:
    """Extract the structured result from ``claude --print --output-format json``.

    Claude Code's json output is a single top-level object; defensively
    return an empty dict on parse failure so the caller can degrade
    gracefully and still capture the diff.
    """
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


def capture_diff(workdir: Path) -> str:
    """Return the uncommitted changes as a unified diff."""
    # ``git add -N`` ensures untracked-but-new files show in ``git diff``.
    subprocess.run(
        ["git", "add", "-N", "."],
        cwd=workdir,
        check=False,
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "diff", "--no-color"],
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
    parent_workspace: Path,
    max_budget_usd: float,
    timeout_s: int,
    model: str | None,
    dry_run: bool,
    results_dir: Path,
) -> RunResult:
    """Run one (task, arm) pair and return the captured result."""
    print(f"  [{arm:7s}] {task['instance_id']}", file=sys.stderr, end="", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"cs-swe-{task['instance_id']}-{arm}-") as tmp_s:
        tmp = Path(tmp_s)
        workdir = tmp / "repo"

        # 1. Materialize MCP config for this arm.
        if arm == "with":
            mcp_config = materialize_mcp_with(mcp_bin, parent_workspace, tmp)
        else:
            mcp_config = MCP_WITHOUT_PATH

        # 2. Clone the task repo (skipped in dry-run).
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

        # 3. Build prompt and spawn claude.
        prompt = PROMPT_PREFIX + task["problem_statement"]
        cmd = build_claude_cmd(claude_bin, mcp_config, workdir, prompt, max_budget_usd, model)

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
        except subprocess.TimeoutExpired:
            exit_code = -2
            stdout = ""
            err_msg = f"timeout after {timeout_s}s"
        walltime = time.monotonic() - t0

        # 4. Capture artifacts (diff + raw claude json) into results_dir for
        # downstream evaluation. Named by (arm, instance_id).
        artifact_base = results_dir / arm / task["instance_id"]
        artifact_base.mkdir(parents=True, exist_ok=True)

        claude_json_path: Path | None = None
        if stdout:
            claude_json_path = artifact_base / "claude.json"
            claude_json_path.write_text(stdout)

        diff = capture_diff(workdir)
        diff_path: Path | None = None
        if diff:
            diff_path = artifact_base / "patch.diff"
            diff_path.write_text(diff)

        # 5. Parse token stats.
        claude_json = parse_claude_json(stdout) if stdout else {}
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
    args = parser.parse_args()

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

    parent_workspace = REPO_ROOT  # codesurgeon indexes itself as the workspace

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
                parent_workspace=parent_workspace,
                max_budget_usd=args.max_budget_usd,
                timeout_s=args.timeout,
                model=args.model,
                dry_run=args.dry_run,
                results_dir=results_dir,
            )
            append_result(args.results, result)

    print(f"done → {args.results}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
