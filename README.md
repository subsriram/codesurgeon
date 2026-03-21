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

| Tool | Description |
|------|-------------|
| `run_pipeline` | **Primary tool.** Auto-detects intent, returns context + impact in one call |
| `get_context_capsule` | Lightweight context search bounded to token budget |
| `get_impact_graph` | Blast-radius: what breaks if this symbol changes |
| `get_skeleton` | File API surface — signatures + docstrings, no bodies |
| `search_logic_flow` | Trace execution path between two functions |
| `get_diff_capsule` | Context for a git diff — changed symbols + callers + tests |
| `index_status` | Health check: symbol/edge/file counts |
| `get_session_context` | Cross-session observations with stale flags |
| `save_observation` | Persist an insight linked to a symbol |

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
| Swift | tree-sitter | Full AST — class/struct/enum/extension/protocol/func/method |
| Shell (bash/zsh) | tree-sitter | Function extraction |
| HTML | tree-sitter | Script/style blocks |
| SQL | Regex fallback | CREATE TABLE/VIEW/FUNCTION |

## License

MIT
