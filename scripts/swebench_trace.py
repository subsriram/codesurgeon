#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Capture tool-call traces for a set of SWE-bench tasks (issue #29b diagnostic).

Spawns ``claude --print --output-format stream-json`` with the codesurgeon
MCP config and ``CS_LOG=info,cs_mcp=debug`` in the server env, saving:

    target/swebench/trace/<instance_id>/
    ├── stream.ndjson     # full claude stream-json transcript (messages + tool uses)
    ├── mcp.stderr        # codesurgeon-mcp tracing output
    ├── claude.stderr     # claude CLI stderr
    ├── patch.diff        # captured git diff
    └── summary.json      # tallied tool use counts, capsule sizes, etc.

Usage:
    uv run scripts/swebench_trace.py \\
        --instance-ids astropy__astropy-14539,django__django-11163,\\
matplotlib__matplotlib-13989,psf__requests-1766

Only runs the **with** arm — the question we're answering is what
codesurgeon *served* and what the agent *did with it*, not comparison
against bare (which we already have from the pilot).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = REPO_ROOT / "benches" / "swebench" / "tasks.json"
MCP_BIN = REPO_ROOT / "target" / "release" / "codesurgeon-mcp"
CS_BIN = REPO_ROOT / "target" / "release" / "codesurgeon"
TRACE_ROOT = REPO_ROOT / "target" / "swebench" / "trace"

PROMPT_PREFIX = """\
You are fixing a real GitHub issue in this repository. Read the problem
statement carefully, inspect the code, and make the minimal change needed to
fix the bug. Do not add new tests. Do not reformat unrelated code. When you
are confident the fix is complete, stop — your changes will be captured as a
git diff and evaluated automatically.

Before you start reading files, call `mcp__cs-codesurgeon__run_pipeline`
with a short description of the task (e.g. `task="fix VLA diff bug in
io.fits.FITSDiff"`). It returns a budgeted capsule of the most relevant
symbols, files, and call-graph edges for the problem — use this to find
the right file to edit instead of doing Grep + Read exploration. The
capsule typically returns in under 200ms and replaces 5–10 exploratory
tool calls. Only after you have the capsule should you start opening
files with Read.

Problem statement:
"""


def load_tasks() -> dict[str, dict]:
    data = json.loads(TASKS_PATH.read_text())
    return {t["instance_id"]: t for t in data["tasks"]}


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def clone_task_repo(task: dict, dest: Path) -> None:
    repo_url = f"https://github.com/{task['repo']}.git"
    base_commit = task["base_commit"]
    dest.mkdir(parents=True, exist_ok=True)
    git(["init", "--quiet"], cwd=dest)
    git(["remote", "add", "origin", repo_url], cwd=dest)
    try:
        git(["fetch", "--depth", "1", "origin", base_commit], cwd=dest)
    except subprocess.CalledProcessError:
        git(["fetch", "--depth", "50", "origin"], cwd=dest)
    git(["checkout", "--quiet", base_commit], cwd=dest)


def materialize_mcp_config(tmp: Path, workspace: Path) -> Path:
    """Render mcp_with.json with CS_WORKSPACE pointing at the task repo.

    Critical: ``workspace`` must be the **cloned task repo** (e.g. the
    requests/ checkout), not the codesurgeon source tree. Pointing it at
    the codesurgeon repo makes run_pipeline return irrelevant capsules
    from cs-core/cs-mcp Rust symbols, which is what #29b's first trace
    was secretly doing.
    """
    cfg = {
        "mcpServers": {
            "cs-codesurgeon": {
                "command": str(MCP_BIN),
                "args": [],
                "env": {
                    "CS_WORKSPACE": str(workspace),
                    "CS_LOG": "info,cs_mcp=debug,cs_core=debug",
                },
            }
        }
    }
    out = tmp / "mcp_with.json"
    out.write_text(json.dumps(cfg, indent=2))
    return out


def run_one(iid: str, task: dict, timeout_s: int, max_budget_usd: float) -> dict:
    out_dir = TRACE_ROOT / iid
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {iid} ===", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"cs-swe-trace-{iid}-") as tmp_s:
        tmp = Path(tmp_s)
        workdir = tmp / "repo"
        print(f"  cloning {task['repo']}@{task['base_commit'][:8]} …", flush=True)
        clone_task_repo(task, workdir)

        # Pre-index the task repo synchronously so the first MCP run_pipeline
        # call doesn't block on cold indexing. This builds .codesurgeon/ in
        # the tempdir; the subsequent codesurgeon-mcp child reuses it.
        print(f"  pre-indexing with codesurgeon …", flush=True)
        t_idx = time.monotonic()
        idx_proc = subprocess.run(
            [str(CS_BIN), "index", "--workspace", str(workdir)],
            env={**os.environ, "CS_WORKSPACE": str(workdir)},
            capture_output=True,
            text=True,
            timeout=600,
        )
        idx_wall = time.monotonic() - t_idx
        (out_dir / "index.stdout").write_text(idx_proc.stdout)
        (out_dir / "index.stderr").write_text(idx_proc.stderr)
        print(f"    indexed in {idx_wall:.1f}s (exit={idx_proc.returncode})", flush=True)
        if idx_proc.returncode != 0:
            print(f"    index failed; stderr tail: {idx_proc.stderr[-300:]}", flush=True)

        mcp_config = materialize_mcp_config(tmp, workdir)
        prompt = PROMPT_PREFIX + task["problem_statement"]

        cmd = [
            "claude",
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",  # required with stream-json
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
            prompt,
        ]

        stream_path = out_dir / "stream.ndjson"
        stderr_path = out_dir / "claude.stderr"
        mcp_stderr_path = out_dir / "mcp.stderr"

        print(f"  spawning claude --output-format stream-json …", flush=True)
        t0 = time.monotonic()
        with stream_path.open("wb") as stream_f, stderr_path.open("wb") as stderr_f:
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=stream_f,
                    stderr=stderr_f,
                    timeout=timeout_s,
                    cwd=workdir,
                )
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                exit_code = -2
        walltime = time.monotonic() - t0
        print(f"  done exit={exit_code} wall={walltime:.1f}s", flush=True)

        # Capture git diff.
        subprocess.run(["git", "add", "-N", "."], cwd=workdir, check=False, capture_output=True)
        diff = subprocess.run(
            ["git", "diff", "--no-color"],
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        (out_dir / "patch.diff").write_text(diff)

        # codesurgeon-mcp's tracing goes to the spawned server's stderr, which
        # Claude Code buffers and surfaces via --debug or separately. In practice
        # the stderr we captured above is claude's own stderr including MCP
        # server stderr multiplexed in — split later if needed.
        # For now just note mcp_stderr_path exists as a hook.
        mcp_stderr_path.write_bytes(b"")  # placeholder

        # Tally tool uses from stream.
        summary = analyze_stream(stream_path, diff)
        summary["instance_id"] = iid
        summary["exit_code"] = exit_code
        summary["walltime_s"] = round(walltime, 2)
        summary["diff_bytes"] = len(diff.encode("utf-8"))
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def analyze_stream(stream_path: Path, diff: str) -> dict:
    """Parse claude stream-json output into a per-task summary.

    stream-json emits one JSON object per line. The shapes we care about:

    - ``{"type":"system","subtype":"init", ...}`` — session start
    - ``{"type":"assistant","message":{"content":[{"type":"tool_use", ...}]}}``
    - ``{"type":"user","message":{"content":[{"type":"tool_result", ...}]}}``
    - ``{"type":"result", ...}`` — final aggregate
    """
    tool_calls: list[dict] = []
    tool_results: dict[str, int] = {}  # tool_use_id -> result size bytes
    result_event = None
    turn_count = 0

    if not stream_path.exists():
        return {"error": "stream.ndjson missing"}

    with stream_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "assistant":
                turn_count += 1
                msg = ev.get("message", {})
                for block in msg.get("content", []):
                    if block.get("type") == "tool_use":
                        name = block.get("name", "?")
                        tool_calls.append(
                            {
                                "id": block.get("id"),
                                "name": name,
                                "input": block.get("input", {}),
                            }
                        )
            elif t == "user":
                msg = ev.get("message", {})
                for block in msg.get("content", []):
                    if block.get("type") == "tool_result":
                        tuid = block.get("tool_use_id")
                        content = block.get("content", "")
                        if isinstance(content, list):
                            content = json.dumps(content)
                        tool_results[tuid] = len(str(content).encode("utf-8"))
            elif t == "result":
                result_event = ev

    # Classify tool calls.
    name_counts = Counter(c["name"] for c in tool_calls)
    cs_calls = [c for c in tool_calls if c["name"].startswith("mcp__cs-codesurgeon__")]
    read_calls = [c for c in tool_calls if c["name"] == "Read"]
    grep_calls = [c for c in tool_calls if c["name"] == "Grep"]
    glob_calls = [c for c in tool_calls if c["name"] == "Glob"]
    edit_calls = [c for c in tool_calls if c["name"] in ("Edit", "Write", "MultiEdit")]

    # Per-call response size (bytes returned to the agent).
    def sizes(calls):
        return [tool_results.get(c["id"], 0) for c in calls]

    # Read targets — which files did the agent Read, and in what order?
    read_targets = [
        {"path": c["input"].get("file_path", "?"), "bytes": tool_results.get(c["id"], 0)}
        for c in read_calls
    ]

    cs_detail = []
    for c in cs_calls:
        cs_detail.append(
            {
                "tool": c["name"].removeprefix("mcp__cs-codesurgeon__"),
                "input": c["input"],
                "result_bytes": tool_results.get(c["id"], 0),
            }
        )

    usage = (result_event or {}).get("usage", {}) if result_event else {}
    total_input = (
        (usage.get("input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
    )
    return {
        "num_turns": turn_count,
        "tool_call_counts": dict(name_counts),
        "cs_codesurgeon_calls": cs_detail,
        "cs_call_count": len(cs_calls),
        "read_call_count": len(read_calls),
        "grep_call_count": len(grep_calls),
        "glob_call_count": len(glob_calls),
        "edit_call_count": len(edit_calls),
        "read_targets": read_targets,
        "tool_result_bytes": {
            "cs_codesurgeon_total": sum(sizes(cs_calls)),
            "read_total": sum(sizes(read_calls)),
            "grep_total": sum(sizes(grep_calls)),
        },
        "usage": {
            "input_tokens": usage.get("input_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_input": total_input,
        },
        "total_cost_usd": (result_event or {}).get("total_cost_usd"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-ids", required=True, help="comma-separated")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-budget-usd", type=float, default=3.00)
    args = parser.parse_args()

    tasks_by_id = load_tasks()
    wanted = [s.strip() for s in args.instance_ids.split(",") if s.strip()]
    missing = [i for i in wanted if i not in tasks_by_id]
    if missing:
        print(f"unknown instance_ids: {missing}", file=sys.stderr)
        return 2

    if not MCP_BIN.exists():
        print(f"codesurgeon-mcp missing: {MCP_BIN}", file=sys.stderr)
        return 2

    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"tracing {len(wanted)} tasks → {TRACE_ROOT.relative_to(REPO_ROOT)}/")

    summaries = []
    for iid in wanted:
        s = run_one(iid, tasks_by_id[iid], args.timeout, args.max_budget_usd)
        summaries.append(s)
        # Progress print.
        print(
            f"  → cs_calls={s.get('cs_call_count')} "
            f"reads={s.get('read_call_count')} "
            f"greps={s.get('grep_call_count')} "
            f"edits={s.get('edit_call_count')} "
            f"turns={s.get('num_turns')} "
            f"diff={s.get('diff_bytes')}B",
            flush=True,
        )

    # Top-level rollup.
    rollup_path = TRACE_ROOT / "rollup.json"
    rollup_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {rollup_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
