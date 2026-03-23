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

Compared to [vexp](https://vexp.dev) (the project that inspired this):

| | vexp | codesurgeon |
|-|------|-------------|
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
```

## Language support

| Language | Parser | Notes |
|----------|--------|-------|
| Rust | tree-sitter | Full AST incl. impl/trait |
| Python | tree-sitter | Full AST |
| TypeScript / TSX | tree-sitter | Full AST |
| JavaScript / JSX | tree-sitter | Full AST |
| Swift | tree-sitter + Xcode MCP (optional) | Full AST — class/struct/enum/extension/protocol/func/method; Xcode MCP adds resolved types |
| Shell (bash/zsh) | tree-sitter | Function extraction |
| HTML | tree-sitter | Script/style blocks |
| SQL | tree-sitter | CREATE TABLE/VIEW/FUNCTION/INDEX/TYPE |

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

## License

MIT
