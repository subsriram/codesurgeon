#!/usr/bin/env bash
# Launch the SWE-bench pilot run (issue #29b). Intended for detached execution:
#
#     cd /Users/sriram/projects/codesurgeon
#     nohup bash scripts/swebench_pilot.sh > /tmp/cs-swe-pilot.out 2>&1 < /dev/null &
#     disown
#     tail -f /tmp/cs-swe-pilot.out   # watch progress from any shell
#
# Runs 10 tasks × 2 arms (~1–3h walltime, ~$5–$15 synthetic cost) over OAuth.
# Refuses to start if Docker is unreachable, since the swebench harness
# evaluation step requires it and we don't want to burn agent quota on
# output that can't be scored.
#
# Environment overrides:
#   PILOT_TASKS        task count (default: 10)
#   PILOT_BUDGET_USD   per-task max-budget cap (default: 3.00)
#   PILOT_TIMEOUT      per-task wallclock seconds (default: 900)
#   PILOT_MODEL        override --model (default: Claude Code default)
#   PILOT_MAX_WORKERS  swebench harness parallelism (default: 4)
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

TASKS="${PILOT_TASKS:-10}"
BUDGET="${PILOT_BUDGET_USD:-3.00}"
TIMEOUT="${PILOT_TIMEOUT:-900}"
MODEL="${PILOT_MODEL:-}"
MAX_WORKERS="${PILOT_MAX_WORKERS:-4}"

RUN_ID="pilot-$(date +%Y%m%d-%H%M%S)"
LOG_DIR="target/swebench/${RUN_ID}"
mkdir -p "${LOG_DIR}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "swebench pilot run_id=${RUN_ID}"
log "  tasks=${TASKS} budget=\$${BUDGET} timeout=${TIMEOUT}s workers=${MAX_WORKERS}"
log "  logs → ${LOG_DIR}/"

# Preflight: Docker must be running for the eval step.
if ! docker ps >/dev/null 2>&1; then
  log "ERROR: docker daemon not reachable"
  log "  start Docker Desktop and re-run this script"
  log "  (nothing was consumed — agent runs not yet started)"
  exit 2
fi
log "preflight: docker OK"

# Preflight: codesurgeon-mcp binary must exist for the treatment arm.
if [[ ! -x "target/release/codesurgeon-mcp" ]]; then
  log "ERROR: target/release/codesurgeon-mcp missing"
  log "  build it: cargo build --release --features metal"
  exit 2
fi
log "preflight: codesurgeon-mcp OK"

# Preflight: claude binary on PATH.
if ! command -v claude >/dev/null 2>&1; then
  log "ERROR: claude not on PATH"
  exit 2
fi
log "preflight: claude OK ($(claude --version 2>/dev/null | head -1 || echo 'unknown'))"

log "preflight: warning — kill any interactive Claude Code sessions before"
log "  long runs, to avoid ~/.claude.json write contention during token refresh"

# Phase 1 — agent runs. Writes target/swebench/results.jsonl incrementally.
log "phase 1/3: agent runs (${TASKS} tasks × 2 arms)"
AGENT_ARGS=(
  --tasks "${TASKS}"
  --clean
  --max-budget-usd "${BUDGET}"
  --timeout "${TIMEOUT}"
)
if [[ -n "${MODEL}" ]]; then
  AGENT_ARGS+=(--model "${MODEL}")
fi

uv run --python 3.14 benches/swebench/run.py "${AGENT_ARGS[@]}" \
  > "${LOG_DIR}/run.log" 2>&1 || {
    rc=$?
    log "phase 1 exited with code ${rc}"
    log "  tail -50 ${LOG_DIR}/run.log:"
    tail -50 "${LOG_DIR}/run.log" | sed 's/^/    /'
    exit "${rc}"
}
log "phase 1 done — see ${LOG_DIR}/run.log"

RESULT_ROWS=$(wc -l < target/swebench/results.jsonl | tr -d ' ')
log "  wrote ${RESULT_ROWS} rows to target/swebench/results.jsonl"

# Phase 2 — swebench harness eval (Docker per task).
log "phase 2/3: swebench harness evaluation"
uv run --python 3.14 scripts/swebench_eval.py \
  --run-id "${RUN_ID}" \
  --max-workers "${MAX_WORKERS}" \
  > "${LOG_DIR}/eval.log" 2>&1 || {
    rc=$?
    log "phase 2 exited with code ${rc}"
    log "  tail -50 ${LOG_DIR}/eval.log:"
    tail -50 "${LOG_DIR}/eval.log" | sed 's/^/    /'
    exit "${rc}"
}
log "phase 2 done — see ${LOG_DIR}/eval.log"

# Phase 3 — render markdown report.
log "phase 3/3: render report"
uv run --python 3.14 scripts/swebench_report.py --pilot \
  > "benches/swebench/report_pilot.md" 2> "${LOG_DIR}/report.err" || {
    rc=$?
    log "phase 3 exited with code ${rc}"
    cat "${LOG_DIR}/report.err" | sed 's/^/    /'
    exit "${rc}"
}
log "phase 3 done — report at benches/swebench/report_pilot.md"

log ""
log "================  PILOT COMPLETE  ================"
log ""
log "next:"
log "  cat benches/swebench/report_pilot.md"
log "  evaluate #29b go/no-go gate:"
log "    - harness stable (zero infra errors)?"
log "    - pass@1 directional signal (with >= without - 10pp)?"
log "    - per-task walltime ≤ 10min avg?"
log "  if all three pass, open #29c"
log "  if not, diagnose in ${LOG_DIR}/ and iterate"
