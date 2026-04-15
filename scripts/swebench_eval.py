#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "swebench>=3.0.0",
# ]
# ///
"""Evaluate captured SWE-bench patches via the official harness (issue #29b).

Reads ``target/swebench/results.jsonl`` (produced by
``benches/swebench/run.py``), builds a predictions file in the swebench
harness format, runs ``swebench.harness.run_evaluation`` per arm, and
merges the ``passed`` verdict and the path to the per-task eval report
back into results.jsonl.

The swebench harness spins up one Docker container per task instance to
apply the patch and run the pinned test suite. Docker must be running
before invoking this script.

Usage:
    uv run scripts/swebench_eval.py --run-id pilot-20260415-103000
    uv run scripts/swebench_eval.py --run-id pilot --arms with
    uv run scripts/swebench_eval.py --run-id pilot --max-workers 4

Layout produced under target/swebench/<run-id>/:
    predictions_with.json
    predictions_without.json
    eval_with/        ← swebench per-instance reports (created by harness)
    eval_without/
    results_evaluated.jsonl  ← augmented copy of results.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "target" / "swebench" / "results.jsonl"
DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
MODEL_NAME_TEMPLATE = "codesurgeon-{arm}"  # "with" | "without"


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group_by_arm(results: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        out[r["arm"]].append(r)
    return out


def read_diff(row: dict) -> str:
    """Load the captured diff for a row. Empty string if none was captured."""
    if not row.get("diff_path"):
        return ""
    p = REPO_ROOT / row["diff_path"]
    if not p.exists():
        return ""
    return p.read_text()


def write_predictions(rows: list[dict], arm: str, out_path: Path) -> int:
    """Write a swebench predictions.json for one arm.

    Schema (per swebench docs):
        [
          {
            "instance_id": "astropy__astropy-7166",
            "model_name_or_path": "codesurgeon-with",
            "model_patch": "<unified diff or empty string>"
          },
          ...
        ]

    Rows with no captured diff still get an empty-patch prediction so the
    harness records them as unresolved rather than silently dropping them.
    """
    preds = []
    for row in rows:
        preds.append(
            {
                "instance_id": row["instance_id"],
                "model_name_or_path": MODEL_NAME_TEMPLATE.format(arm=arm),
                "model_patch": read_diff(row),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preds, indent=2))
    return len(preds)


def check_docker() -> bool:
    """Return True if the Docker daemon is reachable."""
    try:
        r = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_harness(
    predictions_path: Path,
    run_id: str,
    max_workers: int,
    log_dir: Path,
) -> subprocess.CompletedProcess:
    """Invoke ``python -m swebench.harness.run_evaluation``.

    The harness writes per-instance reports under ``logs/run_evaluation/<run_id>/…``
    in the current working directory. We cd into ``log_dir`` so those logs
    land under ``target/swebench/<run-id>/`` rather than polluting the repo.
    """
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--predictions_path",
        str(predictions_path),
        "--run_id",
        run_id,
        "--dataset_name",
        DATASET_NAME,
        "--max_workers",
        str(max_workers),
        "--cache_level",
        "env",  # cache conda envs, not full instance images
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"  running: {' '.join(cmd[:3])} … (logs → {log_dir.relative_to(REPO_ROOT)})", file=sys.stderr)
    return subprocess.run(
        cmd,
        cwd=log_dir,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_harness_report(log_dir: Path, run_id: str, arm: str) -> dict[str, bool]:
    """Extract per-instance pass/fail from the harness output tree.

    After ``run_evaluation`` completes, a summary report file is written
    to ``<cwd>/<model_name>.<run_id>.json``. Each entry has a ``resolved``
    key. We fall back to scanning per-instance report.json files if the
    summary is absent.
    """
    verdicts: dict[str, bool] = {}
    model_name = MODEL_NAME_TEMPLATE.format(arm=arm)

    # Primary path — summary JSON next to the predictions file.
    summary = log_dir / f"{model_name}.{run_id}.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            for iid in data.get("resolved_ids", []) or data.get("resolved", []):
                verdicts[iid] = True
            for iid in data.get("unresolved_ids", []) or data.get("unresolved", []):
                verdicts[iid] = False
            for iid in data.get("error_ids", []) or data.get("errors", []):
                verdicts.setdefault(iid, False)
            if verdicts:
                return verdicts
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback — walk per-instance reports.
    reports_root = log_dir / "logs" / "run_evaluation" / run_id / model_name
    if reports_root.exists():
        for inst_dir in reports_root.iterdir():
            report_file = inst_dir / "report.json"
            if not report_file.exists():
                continue
            try:
                data = json.loads(report_file.read_text())
                # swebench per-instance report shape: {instance_id: {resolved: bool, ...}}
                for iid, info in data.items():
                    verdicts[iid] = bool(info.get("resolved", False))
            except (json.JSONDecodeError, KeyError):
                continue
    return verdicts


def merge_verdicts(
    results: list[dict],
    verdicts_by_arm: dict[str, dict[str, bool]],
    log_dir: Path,
) -> list[dict]:
    """Return a new results list with ``passed`` field added.

    Rows without a verdict keep ``passed=None`` so the report can
    distinguish "did not evaluate" from "evaluated and failed".
    """
    out = []
    for row in results:
        verdicts = verdicts_by_arm.get(row["arm"], {})
        passed = verdicts.get(row["instance_id"])
        augmented = dict(row)
        augmented["passed"] = passed
        augmented["eval_log_dir"] = str(log_dir.relative_to(REPO_ROOT))
        out.append(augmented)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help=f"results.jsonl path (default: {RESULTS_PATH.relative_to(REPO_ROOT)})",
    )
    parser.add_argument("--run-id", required=True, help="swebench harness run id (e.g. 'pilot-20260415')")
    parser.add_argument(
        "--arms",
        default="with,without",
        help="comma-separated arms to evaluate (default: with,without)",
    )
    parser.add_argument("--max-workers", type=int, default=4, help="parallel docker workers (default: 4)")
    parser.add_argument("--skip-docker-check", action="store_true", help="skip preflight docker ps")
    args = parser.parse_args()

    if not args.skip_docker_check and not check_docker():
        print("ERROR: docker daemon not reachable. Start Docker Desktop and retry.", file=sys.stderr)
        return 2

    results = load_results(args.results)
    if not results:
        print(f"no results at {args.results}", file=sys.stderr)
        return 1

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("with", "without"):
            print(f"unknown arm: {a}", file=sys.stderr)
            return 2

    by_arm = group_by_arm(results)
    log_dir = args.results.parent / args.run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    verdicts_by_arm: dict[str, dict[str, bool]] = {}
    for arm in arms:
        rows = by_arm.get(arm, [])
        if not rows:
            print(f"arm {arm}: no rows to evaluate", file=sys.stderr)
            continue

        print(f"arm {arm}: {len(rows)} tasks", file=sys.stderr)
        preds_path = log_dir / f"predictions_{arm}.json"
        n = write_predictions(rows, arm, preds_path)
        print(f"  wrote {preds_path.relative_to(REPO_ROOT)} ({n} predictions)", file=sys.stderr)

        proc = run_harness(
            predictions_path=preds_path,
            run_id=args.run_id,
            max_workers=args.max_workers,
            log_dir=log_dir,
        )
        # Capture harness stdout/stderr for debugging.
        (log_dir / f"harness_{arm}.stdout").write_text(proc.stdout)
        (log_dir / f"harness_{arm}.stderr").write_text(proc.stderr)
        if proc.returncode != 0:
            print(f"  WARN harness exit={proc.returncode}; continuing to parse report", file=sys.stderr)

        verdicts = parse_harness_report(log_dir, args.run_id, arm)
        verdicts_by_arm[arm] = verdicts
        print(
            f"  verdicts: {sum(1 for v in verdicts.values() if v)} passed / {sum(1 for v in verdicts.values() if not v)} failed "
            f"(of {len(rows)})",
            file=sys.stderr,
        )

    # Merge verdicts back into a new results_evaluated.jsonl.
    augmented = merge_verdicts(results, verdicts_by_arm, log_dir)
    out_path = log_dir / "results_evaluated.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in augmented) + "\n")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}", file=sys.stderr)

    # Also overwrite the top-level results.jsonl so swebench_report.py can
    # read the canonical location without --results juggling.
    shutil.copy(out_path, args.results)
    print(f"updated {args.results.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
