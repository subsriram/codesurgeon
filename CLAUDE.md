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

Add to `~/.claude/mcp_settings.json` (note: **not** `settings.json` — Claude Code's schema rejects `mcpServers` there):

```json
{
  "mcpServers": {
    "codesurgeon": {
      "command": "/Users/sriram/projects/codesurgeon/target/release/codesurgeon-mcp",
      "args": [],
      "env": {
        "CS_WORKSPACE": "/Users/sriram/projects/codesurgeon"
      }
    }
  }
}
```

Then restart Claude Code — the server indexes in the background on first start.

## Testing MCP over JSON-RPC

You can test the MCP server directly without restarting Claude Code by piping raw JSON-RPC requests to the binary:

```bash
# index_status
printf '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"index_status","arguments":{}}}' \
  | CS_WORKSPACE=/path/to/workspace timeout 60 ./target/release/codesurgeon-mcp

# re-index a workspace
printf '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"run_pipeline","arguments":{"task":"reindex"}}}' \
  | CS_WORKSPACE=/path/to/workspace timeout 120 ./target/release/codesurgeon-mcp

# get_context_capsule query
printf '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_context_capsule","arguments":{"query":"central state coordinator for documents lists and categories","max_tokens":4000}}}' \
  | CS_WORKSPACE=/path/to/workspace timeout 60 ./target/release/codesurgeon-mcp
```

The server speaks JSON-RPC 2.0 over stdin/stdout. Wrap the response through `| python3 -m json.tool` for readable output.

## Invariants — do not break these

These have been broken by accident before. Treat them as hard constraints.

### 1. `jsonrpc: "2.0"` field in every response
Every JSON-RPC response **must** include `"jsonrpc":"2.0"`. This field was accidentally
dropped during a refactor; clients hard-fail on responses that omit it.
See `Response` struct in `crates/cs-mcp/src/main.rs`.

### 2. LSP-framed stdio transport (`Content-Length` headers)
`codesurgeon-mcp` supports two read modes and **always writes framed responses**:

- **Framed (LSP-style)** — `Content-Length: N\r\n\r\n{json}` — required by Codex
- **NDJSON fallback** — raw newline-terminated JSON — used by Claude Code

**Do not remove the Content-Length framing from writes.** Codex will silently drop the
connection if responses are bare NDJSON. The dual-read logic in the stdio loop must also
be preserved so Claude Code continues to work.

See `main()` in `crates/cs-mcp/src/main.rs`, and the smoke-test one-liner:

```bash
msg='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}'; \
printf "Content-Length: ${#msg}\r\n\r\n${msg}" \
  | CS_WORKSPACE=. timeout 10 ./target/release/codesurgeon-mcp 2>/dev/null
# expect: Content-Length: N\r\n\r\n{"jsonrpc":"2.0",...}
```

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
