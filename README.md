# codesurgeon

**Local-first codebase context engine for AI coding agents.**

codesurgeon parses your codebase into a symbol dependency graph, then serves token-budgeted context capsules to Claude (or any MCP-compatible agent) via the Model Context Protocol. Only the code that matters for your task is returned — full source for the most relevant symbols, signatures-only for adjacent ones.

## Benchmark vs baseline

| Metric | Baseline (full context) | codesurgeon |
|--------|------------------------|-------------|
| Tokens per query (30-file project) | ~30,000 | ~3,000 |
| Token reduction | — | **~90%** |
| Relevant pivots surfaced | all files | 8 symbols |
| Setup | none | `cargo install` + 1 config line |

> Token figures are from the leading competitor's published results. codesurgeon will report your actual per-workspace savings via `codesurgeon stats` once Phase 11 ships.

### SWE-bench Verified

The leading competitor in this space benchmarked against [SWE-bench Verified](https://swebench.com) — 100 real GitHub issues, same model and cost cap across all agents:

| Agent | Pass@1 | $/Task |
|-------|--------|--------|
| Leading competitor + Claude Code | **73%** | **$0.67** |
| OpenHands | 70% | $1.77 |
| Sonar Foundation | 70% | $1.98 |

Key insight from the per-repo breakdown: dependency-graph context (like codesurgeon's symbol graph) yields large gains on **import-heavy, interconnected codebases** (astropy: 80% vs 40% for alternatives) but smaller gains on **rendering-heavy or procedural code** (matplotlib: 43% vs 86%). codesurgeon is best suited for Rust, Python backend, and TypeScript projects with deep module graphs.

codesurgeon will run the same benchmark once Phase 8 (tool parity) and Phase 9 (session memory) are stable.

Compared to the leading alternative:

| | Leading alternative | codesurgeon |
|-|---------------------|-------------|
| Runtime | TypeScript/Node.js wrapper + Rust core | Pure Rust, single binary |
| Search | FTS5 + TF-IDF | BM25 (Tantivy) + graph centrality + optional embeddings |
| Embeddings | None | nomic-embed-text-v1.5 (768-dim, Apple Silicon Metal) |
| Call edges | caller → callee | caller → callee + args snippet |
| Graph extras | imports, calls | + trait impls, type-flow references |
| Diff support | No | `get_diff_capsule` for PR review |
| Session memory | No | Cross-session observations, stale detection |
| Languages | Python, JS/TS, Rust | + Swift, Shell, HTML, SQL |

## How it works

```
Your codebase
     │
     ▼
tree-sitter AST parsing (parallel, rayon)
     │
     ▼
Symbol graph (petgraph DAG)
  Nodes: functions, classes, structs, enums, methods
  Edges: Calls, Imports, Implements, Inherits, References
     │
     ▼
Query: "fix the retry logic in the HTTP client"
     │
     ├─ Intent detection → Debug
     ├─ BM25 search (Tantivy) → top-50 candidates
     ├─ Graph centrality boost → re-rank
     ├─ Semantic similarity blend (embeddings, optional)
     └─ Token-budgeted capsule assembly
           ├─ Pivots (8): full source of most relevant symbols
           └─ Adjacents (20): signatures only (70–90% smaller)
```

## Quick start

### 1. Build

```bash
git clone https://github.com/sriramk/codesurgeon
cd codesurgeon

# Apple Silicon (Metal embeddings, recommended)
cargo build --release --features metal

# CPU-only embeddings
cargo build --release --features embeddings

# No embeddings (BM25 + graph only)
cargo build --release
```

### 2. Add to Claude Code

Add to `~/.claude/mcp_settings.json`:

```json
{
  "mcpServers": {
    "cs-myproject": {
      "command": "/path/to/codesurgeon/target/release/codesurgeon-mcp",
      "args": [],
      "env": {
        "CS_WORKSPACE": "/path/to/your/project"
      }
    }
  }
}
```

Restart Claude Code — the server indexes your workspace in the background on first start.

### Multiple projects

```json
{
  "mcpServers": {
    "cs-frontend": {
      "command": "/path/to/codesurgeon-mcp",
      "env": { "CS_WORKSPACE": "/projects/frontend" }
    },
    "cs-backend": {
      "command": "/path/to/codesurgeon-mcp",
      "env": { "CS_WORKSPACE": "/projects/backend" }
    }
  }
}
```

Tools are namespaced: `cs-frontend__run_pipeline`, `cs-backend__run_pipeline`, etc.

### 3. Use in Claude Code

```
Before editing: run_pipeline(task="add retry logic to the HTTP client")
Before refactoring: get_impact_graph(symbol_fqn="src/http.rs::HttpClient::send")
Navigating code: get_skeleton(file_path="src/http.rs")
After solving hard problem: save_observation(content="retry uses exponential backoff")
```

## MCP tools

### Quick reference

| Tool | When to use |
|------|-------------|
| `run_pipeline` | **Before every edit** — primary tool |
| `get_context_capsule` | Lightweight search for a specific query |
| `get_impact_graph` | **Before any refactor** — see what breaks |
| `get_skeleton` | Understand a file's shape without reading bodies |
| `search_logic_flow` | Trace how A calls B |
| `get_diff_capsule` | Context for a PR or patch |
| `index_status` | Health check / Xcode MCP availability |
| `get_session_context` | Catch up at the start of a session |
| `save_observation` | Persist an insight across sessions |
| `generate_module_docs` | Write per-directory CLAUDE.md files |

### Tool reference

**`run_pipeline`** — your primary tool
Call this before every edit. Give it a plain-English task description. It auto-detects intent (debug / refactor / add / explore / structural), runs BM25 + graph centrality + semantic search, and returns a token-budgeted capsule: full source for the 8 most relevant symbols, signatures-only for up to 20 adjacent ones, plus anything remembered from previous sessions.
```
run_pipeline(task="fix the retry logic in the HTTP client")
```

---

**`get_context_capsule`** — lightweight search
Same search engine as `run_pipeline` but without intent routing or session memory. Use it when you have a specific query and don't need the full pipeline.
```
get_context_capsule(query="token budget assembly")
```

---

**`get_impact_graph`** — blast radius before a refactor
Call this before renaming or changing any function or type. Give it a fully-qualified symbol name and it returns all direct and transitive callers — everything that will break.
```
get_impact_graph(symbol_fqn="src/http.rs::HttpClient::send")
```

---

**`get_skeleton`** — file API surface
Returns all signatures and docstrings from a file with bodies stripped out. Typically 70–90% fewer tokens than the full file. Use it when you need to understand what a file exports without reading everything.
```
get_skeleton(file_path="src/engine.rs")
```

---

**`search_logic_flow`** — trace a path between two functions
Finds the shortest call-graph path from one symbol to another. Use it to understand how A eventually reaches B, or to debug unexpected call chains.
```
search_logic_flow(from_fqn="src/main.rs::handle_request", to_fqn="src/db.rs::query")
```

---

**`get_diff_capsule`** — context for a PR or patch
Paste in a `git diff` and it returns the changed symbols, their callers, and related test files — all token-budgeted. Designed for code review.
```
get_diff_capsule(diff="<paste unified diff here>")
```

---

**`index_status`** — health check
Returns symbol count, edge count, file count, session ID, and Xcode MCP availability. Call it to confirm the index is ready after startup, or to check whether re-indexing is still in progress.
```
index_status()
```

---

**`get_session_context`** — what was learned before
Returns the last ~50 observations saved across all sessions for this workspace. Use it at the start of a session to catch up on what was discovered previously.
```
get_session_context()
```

---

**`save_observation`** — persist an insight
Saves a note tied optionally to a specific symbol. Persists across sessions and is surfaced by `run_pipeline` in future sessions. Call it after solving something non-obvious.
```
save_observation(content="retry uses exponential backoff, max 3 attempts", symbol_fqn="src/http.rs::HttpClient::send")
```

---

**`generate_module_docs`** — write CLAUDE.md files per directory
Generates per-directory CLAUDE.md summaries from the symbol graph — types, functions, and (for Swift directories) Xcode MCP guidance. Pass `write_files=true` to write them to disk.
```
generate_module_docs(write_files=true)
```

---

### Decision guide

| Situation | Tool |
|-----------|------|
| About to edit anything | `run_pipeline` |
| About to rename / move / delete | `get_impact_graph` first |
| Unfamiliar file, need the shape | `get_skeleton` |
| Reviewing a PR | `get_diff_capsule` |
| How does A call B? | `search_logic_flow` |
| Is the index ready? | `index_status` |
| Starting a new session | `get_session_context` |
| Just solved something tricky | `save_observation` |
| Setting up a new project | `generate_module_docs` |

## CLI

```bash
codesurgeon index                          # Index (or re-index) workspace
codesurgeon status                         # Symbol/edge/file counts
codesurgeon search "retry logic"           # BM25 search
codesurgeon skeleton src/http.rs           # File skeleton
codesurgeon impact src/http.rs::send       # Blast radius
codesurgeon flow src/http.rs::send src/retry.rs::with_retry  # Logic flow
codesurgeon diff < my.patch               # Diff-aware capsule
codesurgeon docs                           # Generate per-module CLAUDE.md files
codesurgeon memory                         # List saved observations (shows IDs)
codesurgeon memory --delete <id>           # Delete an observation by ID
codesurgeon observe "insight text"         # Save a manual observation
codesurgeon observe "insight" --symbol src/http.rs::send  # Attach to a symbol
```

## Language support

| Language | Parser | Notes |
|----------|--------|-------|
| Rust | tree-sitter | Full AST incl. impl/trait |
| Python | tree-sitter | Full AST |
| TypeScript / TSX | tree-sitter | Full AST; optional resolved types via TS compiler shim |
| JavaScript / JSX | tree-sitter | Full AST; optional resolved types via TS compiler shim |
| Swift | tree-sitter + Xcode MCP (optional) | Full AST — class/struct/enum/extension/protocol/func/method; Xcode MCP adds resolved types |
| Shell (bash/zsh) | tree-sitter | Function extraction |
| HTML | tree-sitter | Script/style blocks |
| SQL | tree-sitter | CREATE TABLE/VIEW/FUNCTION/INDEX/TYPE |

## TypeScript / JavaScript enrichment — compiler shim

codesurgeon's tree-sitter pass gives you full TypeScript/JavaScript symbol structure. For resolved types (e.g. `Promise<User>` instead of `Promise<any>`), enable the optional compiler shim:

**Requirements:** `node` on PATH, `tsconfig.json` in your workspace root, and `typescript` in `node_modules` (or globally installed).

**Enable** by adding to `.codesurgeon/config.toml`:

```toml
[indexing]
ts_types = true
```

At index time codesurgeon invokes a bundled Node.js script (`ts-enricher.js`) that runs `ts.createProgram()` + `TypeChecker` over your workspace and annotates symbols with their resolved return/property types. The results are stored in the index as `resolved_type` and surfaced in context capsules.

**Incremental:** the shim only re-runs when `tsconfig.json` changes. Existing annotations are preserved across re-indexes.

**Plain JS:** the shim sets `allowJs: true`, so JSDoc-annotated JavaScript files are resolved correctly too.

**VS Code users:** if you have the [codesurgeon VS Code extension](https://marketplace.visualstudio.com/items?itemName=codesurgeon.codesurgeon) installed, the `submit_lsp_edges` tool pushed from the running TypeScript language server is the faster alternative — no subprocess spawning required. `ts_types` is the right choice for CI, Codex, or non-VS Code editors.

---

## Swift enrichment — Xcode MCP

codesurgeon's tree-sitter pass gives you full Swift symbol structure. For resolved types
and live build diagnostics, pair it with **Xcode MCP** (Xcode 26+):

```bash
# Enable in Xcode: Settings → Intelligence → Enable Model Context Protocol
xcrun mcpbridge install --claude-code
```

When Xcode MCP is configured, `run_pipeline` on Swift files will note its availability
and agents can call its tools for type-resolved details. When it's absent, `run_pipeline`
says so explicitly — the tree-sitter graph remains fully usable for semantic search,
impact analysis, and session memory. `index_status` always reports whether Xcode MCP
was detected.

For Xcode < 26 or SPM-only projects: [XcodeBuildMCP](https://github.com/cameroncooke/XcodeBuildMCP)
or [xcode-mcp-server](https://github.com/r-huijts/xcode-mcp-server).

## Privacy

codesurgeon is fully local-first. Your code never leaves your machine.

- **No network calls** — zero outbound connections during indexing, search, or capsule assembly
- **No telemetry** — no usage metrics are sent anywhere, ever
- **No cloud dependencies** — no API keys, no external services, no subscriptions
- **Local index** — the symbol graph lives in `.codesurgeon/index.db` alongside your project (gitignored by default)
- **On-device embeddings** — semantic search uses a local model (nomic-embed-text-v1.5) running entirely on your CPU or Apple Silicon GPU

The only binary that runs is `codesurgeon-mcp`, started by your MCP client (Claude Code / Codex) as a subprocess over stdio. It reads your source files, builds a local index, and responds to tool calls — nothing else.

## Troubleshooting

### MCP server not connecting (Claude Code / Codex)

**Symptom:** Tools like `run_pipeline` are not available, or Claude reports the MCP server failed to start.

1. Confirm `CS_WORKSPACE` points to an existing directory:
   ```bash
   echo $CS_WORKSPACE
   ls "$CS_WORKSPACE"
   ```
2. Run the binary directly to see startup errors:
   ```bash
   echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}' \
     | CS_WORKSPACE=/path/to/project timeout 5 ./target/release/codesurgeon-mcp
   ```
   A healthy server replies with an `initialize` response on stdout. Any error on stderr explains the failure.
3. Re-check the server entry in `~/.claude.json` (Claude Code CLI v2.x):
   ```bash
   claude mcp list
   ```

---

### Index not ready / stale results

**Symptom:** `index_status()` returns 0 symbols, or results look stale after editing files.

- The index builds in the background on first start. Give it a few seconds, then call `index_status()` again to confirm.
- To force a full re-index:
  ```bash
  CS_WORKSPACE=/path/to/project codesurgeon index
  ```
- Check how many files were indexed:
  ```bash
  CS_WORKSPACE=/path/to/project codesurgeon status
  ```

---

### Second instance running in read-only mode

**Symptom:** A second Claude Code window or parallel Codex probe connects but sees no results, or logs show "serving read-only".

This is expected behaviour. Only one `codesurgeon-mcp` instance per workspace runs background indexing (the first one to acquire the PID lock). Subsequent instances serve the existing index read-only — they won't write new embeddings or trigger re-indexing. This is intentional to avoid concurrent index writes.

If you want the new instance to become the primary, stop the existing one first:
```bash
kill $(cat /path/to/project/.codesurgeon/mcp.pid)
```

---

### Stale PID file after a crash

**Symptom:** After a hard kill or power loss, new instances unexpectedly enter read-only mode.

codesurgeon auto-detects this: on startup it reads the existing PID file and runs `kill -0 <pid>` to check whether that process is actually alive. If it's dead, the stale file is overwritten automatically and the new instance becomes primary. **No manual cleanup is needed in normal cases.**

If for some reason this fails (e.g. the PID was reused by a system process), delete the file manually:
```bash
rm /path/to/project/.codesurgeon/mcp.pid
```

---

### Binary not found after `cargo build`

**Symptom:** `codesurgeon` or `codesurgeon-mcp` command not found.

The binaries are placed in `target/release/`, not on your `PATH` automatically. Either use the full path:
```bash
/path/to/codesurgeon/target/release/codesurgeon-mcp
```
Or add the release directory to your PATH, or symlink:
```bash
ln -s /path/to/codesurgeon/target/release/codesurgeon /usr/local/bin/codesurgeon
```

## License

MIT
