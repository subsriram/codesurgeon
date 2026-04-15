#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Select a stratified 100-task subset from SWE-bench Verified.

Fetches the SWE-bench_Verified dataset from HuggingFace's datasets-server
HTTP API (no dataset library dependency — stdlib only), stratifies across
repos, and writes ``benches/swebench/tasks.json``.

The committed ``tasks.json`` is the source of truth for reruns; this script
only needs to be executed when the task set is intentionally regenerated.

Usage:
    uv run benches/swebench/select_tasks.py [--seed N] [--count N]

Target repo distribution (for 100 tasks):
    astropy ~20  django ~20  sympy ~20  matplotlib ~15  requests ~15  other ~10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

DATASET = "princeton-nlp/SWE-bench_Verified"
CONFIG = "default"
SPLIT = "test"

# Per-repo cap. Keys use the SWE-bench ``repo`` field (``org/name`` form).
# Values sum to ≥ count so we have slack; the tail is filled from "other".
REPO_CAPS = {
    "astropy/astropy": 20,
    "django/django": 20,
    "sympy/sympy": 20,
    "matplotlib/matplotlib": 15,
    "psf/requests": 15,
}
OTHER_CAP = 10  # remaining slots filled proportionally from any other repo

OUTPUT_PATH = Path(__file__).parent / "tasks.json"


def fetch_rows(offset: int, length: int) -> list[dict]:
    """Fetch a slice of rows from the HF datasets-server API."""
    params = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [row["row"] for row in data.get("rows", [])]


def fetch_all() -> list[dict]:
    """Page through the dataset and return all rows."""
    rows: list[dict] = []
    offset = 0
    page = 100  # HF API caps responses at 100 rows/page
    while True:
        batch = fetch_rows(offset, page)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
        print(f"  fetched {len(rows)} rows…", file=sys.stderr)
    return rows


def stratify(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Group rows by repo, then sample up to REPO_CAPS per group.

    Remaining slots (count - sum(per-repo picks)) are drawn from "other"
    repos in round-robin order to keep the tail diverse.
    """
    rng = random.Random(seed)

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_repo[row["repo"]].append(row)

    picked: list[dict] = []
    for repo, cap in REPO_CAPS.items():
        pool = by_repo.get(repo, [])
        if not pool:
            print(f"  warn: repo {repo} has 0 rows in dataset", file=sys.stderr)
            continue
        k = min(cap, len(pool))
        picked.extend(rng.sample(pool, k))

    # Fill the "other" bucket from everything not already covered.
    covered = set(REPO_CAPS)
    other_pool: list[dict] = []
    for repo, pool in by_repo.items():
        if repo in covered:
            continue
        other_pool.extend(pool)
    rng.shuffle(other_pool)

    remaining = count - len(picked)
    if remaining > 0:
        picked.extend(other_pool[:remaining])

    return picked[:count]


def to_task_record(row: dict) -> dict:
    """Strip a dataset row down to the fields the harness needs.

    The full SWE-bench row is ~50 KB (includes patch + test_patch + hints);
    the harness only needs the minimum to drive an agent and evaluate its
    output. Keep FAIL_TO_PASS / PASS_TO_PASS so #29b can wire into the
    swebench evaluation harness without re-fetching.
    """
    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "environment_setup_commit": row.get("environment_setup_commit"),
        "problem_statement": row["problem_statement"],
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS"),
        "PASS_TO_PASS": row.get("PASS_TO_PASS"),
        "version": row.get("version"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17, help="random seed (default: 17)")
    parser.add_argument("--count", type=int, default=100, help="task count (default: 100)")
    args = parser.parse_args()

    print(f"fetching {DATASET} {CONFIG}/{SPLIT} via HF datasets-server…", file=sys.stderr)
    rows = fetch_all()
    print(f"  total rows: {len(rows)}", file=sys.stderr)

    picked = stratify(rows, args.count, args.seed)

    # Distribution report.
    dist: dict[str, int] = defaultdict(int)
    for p in picked:
        dist[p["repo"]] += 1
    print(f"  picked {len(picked)} tasks across {len(dist)} repos:", file=sys.stderr)
    for repo in sorted(dist, key=lambda r: -dist[r]):
        print(f"    {dist[repo]:3d}  {repo}", file=sys.stderr)

    out = {
        "source": DATASET,
        "split": SPLIT,
        "seed": args.seed,
        "count": len(picked),
        "tasks": [to_task_record(row) for row in picked],
    }
    OUTPUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH.relative_to(Path.cwd()) if OUTPUT_PATH.is_relative_to(Path.cwd()) else OUTPUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
