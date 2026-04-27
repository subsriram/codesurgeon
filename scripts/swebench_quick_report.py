#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Quick in-flight analysis of target/swebench/results.jsonl.

Shows: row counts per arm, status breakdown (ok/timeout/failed), per-repo
deltas (walltime, tokens, cost), and walltime win/loss counts — without
invoking the swebench harness. Useful during long runs.

Usage:
    uv run scripts/swebench_quick_report.py
    uv run scripts/swebench_quick_report.py --results path/to/file.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "target" / "swebench" / "results.jsonl"
TASKS_PATH = REPO_ROOT / "benches" / "swebench" / "tasks.json"


def total_tok(r: dict) -> int:
    return (
        (r.get("input_tokens") or 0)
        + (r.get("cache_creation_tokens") or 0)
        + (r.get("cache_read_tokens") or 0)
        + (r.get("output_tokens") or 0)
    )


def classify(r: dict) -> str:
    if r["exit_code"] == 0:
        return "ok"
    if r["exit_code"] == -2:
        return "timeout"
    wall = r.get("walltime_s", 0)
    out = r.get("output_tokens") or 0
    if wall < 10 and out == 0:
        return "auth_fail"
    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]

    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["instance_id"], {})[r["arm"]] = r

    print(f"=== TOTAL ROWS: {len(rows)} ===\n")

    # Status counts per arm
    for arm in ("with", "without"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        status_counts: dict[str, int] = defaultdict(int)
        for r in arm_rows:
            status_counts[classify(r)] += 1
        status = "  ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
        print(f"{arm:7s}: n={len(arm_rows)}  {status}")
    print()

    # Paired tasks (both arms have nonzero tokens — excludes auth fails / pre-task timeouts)
    paired = []
    for iid, d in by.items():
        if "with" in d and "without" in d:
            w, o = d["with"], d["without"]
            wt, ot = total_tok(w), total_tok(o)
            if wt > 0 and ot > 0:
                paired.append((iid, w, o, wt, ot))

    repo_data: dict[str, list] = defaultdict(list)
    for iid, w, o, wt, ot in paired:
        repo_data[w.get("repo", "unknown")].append((iid, w, o, wt, ot))

    avg = lambda vals: sum(vals) / len(vals) if vals else 0

    print(f"=== PER-REPO BREAKDOWN (n={len(paired)} paired tasks) ===")
    print(f"{'repo':30s} {'n':>3s} {'w_wall':>8s} {'o_wall':>8s} {'Δ wall':>8s} {'Δ tokens':>10s} {'Δ cost':>8s}")
    print("-" * 82)

    for repo in sorted(repo_data):
        pairs = repo_data[repo]
        n = len(pairs)
        aw = avg([w.get("walltime_s", 0) for _, w, _, _, _ in pairs])
        ao = avg([o.get("walltime_s", 0) for _, _, o, _, _ in pairs])
        at = avg([wt for _, _, _, wt, _ in pairs])
        aot = avg([ot for _, _, _, _, ot in pairs])
        ac = avg([w.get("total_cost_usd") or 0 for _, w, _, _, _ in pairs])
        aoc = avg([o.get("total_cost_usd") or 0 for _, _, o, _, _ in pairs])
        dw = ((aw - ao) / ao) * 100 if ao else 0
        dt = ((at - aot) / aot) * 100 if aot else 0
        dc = ((ac - aoc) / aoc) * 100 if aoc else 0
        print(f"{repo:30s} {n:>3d} {aw:>7.1f}s {ao:>7.1f}s {dw:>+7.1f}% {dt:>+9.1f}% {dc:>+7.1f}%")

    print("-" * 82)
    n = len(paired)
    aw = avg([w.get("walltime_s", 0) for _, w, _, _, _ in paired])
    ao = avg([o.get("walltime_s", 0) for _, _, o, _, _ in paired])
    at = avg([wt for _, _, _, wt, _ in paired])
    aot = avg([ot for _, _, _, _, ot in paired])
    ac = avg([w.get("total_cost_usd") or 0 for _, w, _, _, _ in paired])
    aoc = avg([o.get("total_cost_usd") or 0 for _, _, o, _, _ in paired])
    dw_all = ((aw - ao) / ao) * 100 if ao else 0
    dt_all = ((at - aot) / aot) * 100 if aot else 0
    dc_all = ((ac - aoc) / aoc) * 100 if aoc else 0
    print(f"{'OVERALL':30s} {n:>3d} {aw:>7.1f}s {ao:>7.1f}s {dw_all:>+7.1f}% {dt_all:>+9.1f}% {dc_all:>+7.1f}%")

    wins = losses = ties = 0
    for _, w, o, _, _ in paired:
        ww = w.get("walltime_s", 0)
        ow = o.get("walltime_s", 0)
        if ww < ow * 0.8:
            wins += 1
        elif ww > ow * 1.2:
            losses += 1
        else:
            ties += 1
    print(f"\nWalltime wins/losses/ties (20% threshold): with={wins}  without={losses}  similar={ties}")

    if TASKS_PATH.exists():
        done_pairs = set((r["instance_id"], r["arm"]) for r in rows)
        tasks = json.loads(TASKS_PATH.read_text())["tasks"]
        remaining = sum(1 for t in tasks for a in ("with", "without") if (t["instance_id"], a) not in done_pairs)
        print(f"\nProgress: {len(rows)}/{len(tasks)*2} rows, {remaining} remaining")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
