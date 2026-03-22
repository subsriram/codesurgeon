# codesurgeon — MCP context engine

## What this is
codesurgeon is a local-first dependency graph and session memory server for AI coding agents.
It parses your codebase into a symbol graph, then serves token-budgeted context capsules via MCP.

## MCP tools available

| Tool | When to use |
|------|-------------|
| `run_pipeline` | **Before any edit** — returns relevant context auto-tuned to your task |
| `get_context_capsule` | Lightweight context search for a specific query |
| `get_impact_graph` | **Before any refactor** — see what breaks |
| `get_skeleton` | Understand a file's API surface without reading the full body |
| `search_logic_flow` | Trace execution path between two functions |
| `index_status` | Check index health |
| `get_session_context` | See what was learned in previous sessions |
| `save_observation` | Save an insight so it persists across sessions |

## Recommended workflow

1. Before editing any file: `run_pipeline(task="<what you're about to do>")`
2. Before refactoring a function: `get_impact_graph(symbol_fqn="...")`
3. When navigating unfamiliar code: `get_skeleton(file_path="...")`
4. After solving a non-obvious problem: `save_observation(content="...", symbol_fqn="...")`

## Building

Default build uses embeddings (nomic-ai/nomic-embed-text-v1.5, 768-dim) with Apple Accelerate BLAS:

```bash
cd /Users/sriram/projects/codesurgeon
cargo build --release --features metal
```

Feature flags:
- `--features metal` — **default** — embeddings + Apple Accelerate BLAS (Apple Silicon)
- `--features embeddings` — embeddings, CPU only
- (no features) — no embeddings, BM25+graph only

Binaries produced:
- `target/release/codesurgeon-mcp` — the MCP server (add to Claude Code config)
- `target/release/codesurgeon` — the CLI

## Adding to Claude Code

Use `claude mcp add` (CLI v2.x stores servers in `~/.claude.json`, not `mcp_settings.json`):

```bash
claude mcp add --scope user \
  -e CS_WORKSPACE=/path/to/your/project \
  codesurgeon \
  /path/to/codesurgeon/target/release/codesurgeon-mcp
```

Then restart Claude Code — the server indexes in the background on first start.

## Testing MCP over JSON-RPC

Run the full protocol invariant test suite before any merge:

```bash
cargo test -p cs-mcp --test mcp_protocol
```

This covers: `jsonrpc` field presence, wire format mirroring, `resources` capability,
parallel connection handling, NDJSON round-trip, and more. All 11 tests run in under 1 second.

To drive the binary manually — the server mirrors the client's wire format:

```bash
# Content-Length input (Codex style) → Content-Length response
msg='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}'; \
printf "Content-Length: ${#msg}\r\n\r\n${msg}" \
  | CS_WORKSPACE=/path/to/workspace timeout 10 ./target/release/codesurgeon-mcp 2>/dev/null

# NDJSON input (Claude Code CLI style) → NDJSON response
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"claude-code","version":"0"}}}' \
  | CS_WORKSPACE=/path/to/workspace timeout 10 ./target/release/codesurgeon-mcp 2>/dev/null
```

## Invariants — do not break these

These have been broken by accident before. Treat them as hard constraints.
**All are verified by `cargo test -p cs-mcp --test mcp_protocol` — run this before every merge.**

### 1. `jsonrpc: "2.0"` field in every response
Every JSON-RPC response **must** include `"jsonrpc":"2.0"`. This field was accidentally
dropped during a refactor; clients hard-fail on responses that omit it.
See `Response` struct in `crates/cs-mcp/src/main.rs`.

### 2. Mirror the client's wire format
`codesurgeon-mcp` detects the incoming message format and responds in kind:

- **Content-Length input** → **Content-Length response** — required by Codex (spec-correct)
- **NDJSON input** → **NDJSON response** — required by Claude Code CLI (v2.1.81+)

Do not collapse these into a single output format. Codex drops the connection on bare NDJSON
responses; Claude Code CLI drops the connection on Content-Length responses.
See `transport::Format` and `write_message` in `crates/cs-mcp/src/transport.rs`.

### 3. `resources` capability + empty-list handlers
`initialize` must advertise `"resources": {}` in capabilities, and both `resources/list` and
`resources/templates/list` must return empty arrays (not `-32601 Method not found`).
Codex probes these methods unconditionally; a -32601 causes "MCP startup failed".
codesurgeon exposes **tools only** — the resource handlers are stubs for protocol compliance.

### 4. Secondary instances must not exit on PID lock conflict
When a second process tries to serve the same workspace (e.g. parallel Codex probes), it
must **not** call `exit(0)`. It must serve the connection read-only without background
indexing or the embedder. The old code exited on PID lock conflict, causing
"connection closed: initialize response" for the second connection.
See `acquire_pid_lock` path in `crates/cs-mcp/src/main.rs`.

---

## Ranking pipeline

The search/ranking logic is documented in `docs/ranking.md`.

> **Whenever you change ranking logic or parameters in `engine.rs`, `search.rs`, or
> `graph.rs`, update `docs/ranking.md` to reflect the change — including the parameters
> table at the bottom.**

## Language support

| Language | Parser | Notes |
|----------|--------|-------|
| Python | tree-sitter | Full AST |
| TypeScript / TSX | tree-sitter | Full AST |
| JavaScript / JSX | tree-sitter | Full AST |
| Shell (bash/zsh) | tree-sitter | Function extraction |
| HTML | tree-sitter | Script/style blocks |
| Rust | tree-sitter | Full AST incl. impl/trait |
| Swift | tree-sitter | Full AST — class/struct/enum/extension/protocol/func/method |
| SQL | tree-sitter (tree-sitter-sequel) | CREATE TABLE/VIEW/FUNCTION/INDEX/TYPE |
