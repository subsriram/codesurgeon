#!/usr/bin/env bash
# Prepare a warm-indexed SWE-bench task workspace.
#
# Usage:
#   prepare_workspace.sh <instance_id> [workspace_root]
#
#   instance_id      e.g. sympy__sympy-21379, sphinx-doc__sphinx-9711
#   workspace_root   (optional) parent dir for task workspaces
#                    defaults to $SWEBENCH_WARM_ROOT or
#                    $HOME/.cache/codesurgeon/swebench-warm
#
# Effect:
#   - Clones <repo> at <base_commit> into <workspace_root>/<instance_id>/
#     (or resets an existing clone to base_commit, preserving .codesurgeon/)
#   - Runs `codesurgeon index` against the workspace so the index is ready
#     for `run.py --reuse-workdir ...` to pick up.
#
# See benches/swebench/WARM_WORKSPACES.md for the larger context.
set -euo pipefail

instance_id="${1:?usage: $0 <instance_id> [workspace_root]}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
# Default to a worktree-scoped cache under target/ — gitignored, invalidated
# by `cargo clean` alongside the binary that built the index (which is
# what we want; schema mismatches can't silently bleed across worktrees).
# Override with $SWEBENCH_WARM_ROOT to share across worktrees at your own risk.
root="${2:-${SWEBENCH_WARM_ROOT:-$repo_root/target/swebench-warm}}"
tasks_json="$repo_root/benches/swebench/tasks.json"
cs_bin="$repo_root/target/release/codesurgeon"

if [ ! -x "$cs_bin" ]; then
    echo "error: codesurgeon binary not found at $cs_bin" >&2
    echo "       build first: cargo build --release --features metal" >&2
    exit 2
fi

# Extract repo + base_commit from tasks.json. Use python3 because tasks.json
# is big (~1.6 MB, 100 tasks) and we want string-safe lookups.
read -r repo_slug base_commit < <(
    python3 - <<EOF
import json, sys
tasks = json.loads(open("$tasks_json").read())["tasks"]
hits = [t for t in tasks if t["instance_id"] == "$instance_id"]
if not hits:
    sys.stderr.write(f"error: instance_id $instance_id not in tasks.json\n")
    sys.exit(2)
t = hits[0]
print(t["repo"], t["base_commit"])
EOF
)

repo_url="https://github.com/${repo_slug}.git"
mkdir -p "$root"
ws="$root/$instance_id"

if [ -d "$ws/.git" ]; then
    echo "[prepare] existing workspace at $ws — resetting to $base_commit"
    git -C "$ws" reset --hard "$base_commit"
    git -C "$ws" clean -fdx -e ".codesurgeon"
else
    echo "[prepare] cloning $repo_url @ ${base_commit:0:12} into $ws"
    mkdir -p "$ws"
    git -C "$ws" init --quiet
    git -C "$ws" remote add origin "$repo_url"
    # Prefer a single-commit fetch; some servers don't allow it.
    git -C "$ws" fetch --depth 1 origin "$base_commit" \
        || git -C "$ws" fetch --depth 50 origin
    git -C "$ws" checkout --quiet "$base_commit"
fi

# Kill any stale MCP holding the PID lock before we index.
pid_file="$ws/.codesurgeon/mcp.pid"
if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "[prepare] killing stale codesurgeon-mcp (pid $pid)"
        kill "$pid" || true
        sleep 1
    fi
    rm -f "$pid_file"
fi

echo "[prepare] indexing with $cs_bin"
CS_WORKSPACE="$ws" "$cs_bin" index --workspace "$ws"

echo "[prepare] done — warm workspace at $ws"
echo "[prepare] verify with:"
echo "  $cs_bin --workspace $ws status"
