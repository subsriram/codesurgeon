# SWE-bench harness smoke test (#29a)

Wiring-only verification of the harness scaffolding landed in #29a. Fires a
**canary prompt** (not a real SWE-bench task) against `claude --print` with
each per-arm MCP config and confirms:

1. `claude --print` spawns cleanly via OAuth from inside the parent Claude
   Code session
2. `--strict-mcp-config` isolates the two arms — codesurgeon is present in
   `with`, absent in `without`
3. `--output-format json` returns the structured result + `usage` +
   `total_cost_usd` fields the harness driver needs

## Setup

```bash
# Both arms run from a tempdir outside the codesurgeon repo to avoid the
# child Claude Code auto-loading the project CLAUDE.md, which lists every
# cs-codesurgeon tool and would cause the agent to hallucinate tool
# availability even in the control arm.
SMOKE_DIR=$(mktemp -d -t cs-swe-smoke)
cd "$SMOKE_DIR"
```

Canary prompt (identical in both arms):

> Call the tool literally named `mcp__cs-codesurgeon__index_status`. Report
> the result. If that tool is not available to you, respond with exactly
> the string `TOOL_NOT_AVAILABLE` and nothing else.

Common flags:

```
claude --print \
  --output-format json \
  --strict-mcp-config \
  --mcp-config <arm>.json \
  --permission-mode bypassPermissions \
  --no-session-persistence \
  --max-budget-usd 0.50 \
  --model sonnet
```

## Control arm — `mcp_without.json`

```json
{ "mcpServers": {} }
```

**Result:**

```
TOOL_NOT_AVAILABLE
```

| Field | Value |
|---|---|
| `num_turns` | 1 |
| `duration_ms` | 1,573 |
| `input_tokens` | 3 |
| `output_tokens` | 10 |
| `cache_creation_input_tokens` | 4,960 |
| `cache_read_input_tokens` | 23,228 |
| `total_cost_usd` | $0.0262 |
| `stop_reason` | `end_turn` |

The control arm makes **zero tool calls**, cannot see `cs-codesurgeon`, and
returns the sentinel string in a single turn. `--strict-mcp-config` with an
empty `mcpServers` object successfully blocks every globally-registered MCP
server in `~/.claude.json`.

## Treatment arm — `mcp_with.json` (materialized)

```json
{
  "mcpServers": {
    "cs-codesurgeon": {
      "command": "/Users/sriram/projects/codesurgeon/target/release/codesurgeon-mcp",
      "args": [],
      "env": { "CS_WORKSPACE": "/Users/sriram/projects/codesurgeon" }
    }
  }
}
```

**Result:**

```json
{"symbols": 1337, "edges": 15260, "files": 50}
```

| Field | Value |
|---|---|
| `num_turns` | 2 |
| `duration_ms` | 5,683 |
| `input_tokens` | 6 |
| `output_tokens` | 232 |
| `cache_creation_input_tokens` | 31,399 |
| `cache_read_input_tokens` | 31,133 |
| `total_cost_usd` | $0.1310 |
| `stop_reason` | `end_turn` |

The treatment arm spawns `codesurgeon-mcp` as a child process, invokes
`index_status` via the MCP protocol, and reports the real counts from the
codesurgeon workspace index. **End-to-end MCP plumbing is working.**

## Delta

| Arm | Turns | Walltime | Cost | Tool used |
|---|---:|---:|---:|---|
| control  | 1 | 1.6s | $0.0262 | none |
| treatment | 2 | 5.7s | $0.1310 | `mcp__cs-codesurgeon__index_status` |

The 2× cost difference on this canary is **not** representative of real
SWE-bench tasks — it's an artifact of the canary being trivially short. The
control arm fires one model call and returns a 10-token string; the
treatment arm does a tool call + post-tool-result synthesis. On real tasks
the codesurgeon capsule should reduce total tokens, not increase them, by
letting the agent avoid speculative file reads.

## What this proves

- ✅ `claude --print` spawns cleanly via OAuth from a parent Claude Code session
- ✅ `--strict-mcp-config` isolates per-arm MCP servers (no leak from `~/.claude.json`)
- ✅ `--mcp-config` accepts a tempfile path; the template substitution in
  `run.py` produces valid config JSON
- ✅ `codesurgeon-mcp` spawns as a child process under the spawned Claude
  Code and serves MCP tool calls
- ✅ `--output-format json` returns `result`, `usage.*`, `total_cost_usd`,
  `num_turns`, `duration_ms` — everything `run.py`'s `extract_token_stats`
  pulls out
- ✅ Running the child from a **non-repo working directory** is required
  to avoid CLAUDE.md auto-loading (which would cause the model to
  hallucinate tool availability). `run.py` already does this — each task
  gets a fresh tempdir that is also the claude cwd.

## What this does NOT prove (deferred to #29b)

- Running a **real SWE-bench task** end-to-end (cloning the task repo,
  feeding the problem statement, producing an applicable patch)
- That `benches/swebench/run.py`'s git-clone step works against all 100
  tasks in `tasks.json`
- That the `swebench` evaluation harness can apply the captured diff and
  run the pinned test suite to produce a pass/fail verdict
- Any statement about codesurgeon's *effect* on pass@1 or token cost

Those are #29b's job — the pilot run of 10 tasks × 2 arms.
