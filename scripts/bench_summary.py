#!/usr/bin/env python3
"""Render criterion output as a markdown table for PR comments.

Reads target/criterion/*/new/estimates.json produced by `cargo bench --bench
indexing`, and optionally diffs against a committed benches/baseline.json.

Usage:
    scripts/bench_summary.py [baseline.json]

Prints a markdown table to stdout. Exits 0 regardless of drift — this is
advisory only, not a CI gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CRITERION_DIR = Path("target/criterion")
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


def main() -> int:
    baseline_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benches/baseline.json")
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
        print("_Advisory only. Δ is % change in median vs committed baseline. 🔴 > +10%, 🟢 < −10%._")
    return 0


if __name__ == "__main__":
    sys.exit(main())
