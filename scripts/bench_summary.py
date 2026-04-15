#!/usr/bin/env python3
"""Render bench output as markdown for PR comments.

Reads two inputs:
  - target/criterion/*/new/estimates.json — criterion indexing bench (#28)
  - target/token_savings.json             — token savings example (#27)

Each is diffed against a committed baseline:
  - benches/baseline.json                 — indexing medians
  - benches/token_baseline.json           — token savings per query

Usage:
    scripts/bench_summary.py [indexing_baseline] [token_baseline]

Prints a markdown summary to stdout. Exits 0 regardless of drift — this is
advisory only, not a CI gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CRITERION_DIR = Path("target/criterion")
TOKEN_SAVINGS_PATH = Path("target/token_savings.json")
BENCHMARKS = [
    "index_cold",
    "index_warm",
    "run_pipeline/fix retry logic",
    "run_pipeline/token budget assembly",
    "run_pipeline/how does BM25 search work",
]


def fmt_ns(ns: float) -> str:
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.1f} µs"
    return f"{ns:.0f} ns"


def load_estimate(name: str) -> dict[str, float] | None:
    path = CRITERION_DIR / name / "new" / "estimates.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {
        "median_ns": data["median"]["point_estimate"],
        "mean_ns": data["mean"]["point_estimate"],
    }


def load_baseline(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def delta_pct(current: float, baseline: float) -> str:
    if baseline == 0:
        return "—"
    pct = (current - baseline) / baseline * 100
    sign = "+" if pct >= 0 else ""
    arrow = "🔴" if pct > 10 else ("🟢" if pct < -10 else "")
    return f"{sign}{pct:.1f}% {arrow}".strip()


def render_indexing(baseline_path: Path) -> None:
    baseline = load_baseline(baseline_path)

    rows = []
    for name in BENCHMARKS:
        current = load_estimate(name)
        if current is None:
            rows.append((name, "—", "—", "missing"))
            continue
        median = fmt_ns(current["median_ns"])
        mean = fmt_ns(current["mean_ns"])
        if name in baseline:
            delta = delta_pct(current["median_ns"], baseline[name]["median_ns"])
        else:
            delta = "—"
        rows.append((name, median, mean, delta))

    print("### Indexing benchmark")
    print()
    print("| Benchmark | Median | Mean | Δ vs baseline |")
    print("|---|---:|---:|---:|")
    for name, median, mean, delta in rows:
        print(f"| `{name}` | {median} | {mean} | {delta} |")
    print()
    if not baseline:
        print("_No `benches/baseline.json` committed yet — showing raw numbers only._")
    else:
        print(
            "_Advisory only. Δ is % change in median vs committed baseline. "
            "🔴 > +10%, 🟢 < −10%._"
        )


def render_token_savings(baseline_path: Path) -> None:
    if not TOKEN_SAVINGS_PATH.exists():
        return
    current = json.loads(TOKEN_SAVINGS_PATH.read_text())
    baseline: dict = {}
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())

    print()
    print("### Token savings")
    print()
    print(
        f"Corpus: {current['workspace_tokens']:,} workspace tokens  ·  "
        f"{current['query_count']} queries  ·  "
        f"avg capsule **{current['avg_capsule_tokens']:.0f}** tokens  ·  "
        f"workspace savings **{current['savings_pct']:.1f}%**"
    )
    if baseline:
        base_avg = baseline.get("avg_capsule_tokens", 0.0)
        if base_avg > 0:
            pct = (current["avg_capsule_tokens"] - base_avg) / base_avg * 100
            arrow = "🔴" if pct > 10 else ("🟢" if pct < -10 else "")
            sign = "+" if pct >= 0 else ""
            print(f"Δ avg capsule vs baseline: **{sign}{pct:.1f}%** {arrow}".strip())
    print()
    print("<details><summary>Per-query capsule tokens</summary>")
    print()
    print("| Query | Current | Baseline | Δ |")
    print("|---|---:|---:|---:|")
    base_pq: dict = baseline.get("per_query", {}) if baseline else {}
    for q, tokens in current["per_query"].items():
        base = base_pq.get(q)
        if base:
            diff = tokens - base
            diff_sign = "+" if diff >= 0 else ""
            delta_str = f"{diff_sign}{diff}"
        else:
            delta_str = "—"
            base = "—"
        print(f"| {q} | {tokens} | {base} | {delta_str} |")
    print()
    print("</details>")


def main() -> int:
    indexing_baseline = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/baseline.json")
    )
    token_baseline = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benches/token_baseline.json")
    )
    render_indexing(indexing_baseline)
    render_token_savings(token_baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
