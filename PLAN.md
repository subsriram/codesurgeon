# codesurgeon — Project Plan

## What it is

codesurgeon is a **local-first, pure-Rust MCP server** that gives AI coding agents surgical context
about your codebase. Inspired by vexp, but built end-to-end in Rust with no Node.js dependency.

It parses your code into a dependency graph (AST nodes + call/import edges), then serves
token-budgeted "context capsules" to the agent via MCP — returning only the code that matters.

**Target token reduction: 65–70%** (matching vexp's measured results).

---

## What inspired this

### vexp (https://vexp.dev)

Built by Nicola Alessi. Posted on r/ClaudeCode as "I cut Claude Code's token usage by 65%."

**How vexp works:**

1. **Index** — tree-sitter parses every file into an AST. Nodes = functions/classes/types.
   Edges = imports/calls/implementations. Stored in SQLite. ~5,000 files in <15s using rayon.

2. **Traverse** — Hybrid search: FTS5 + TF-IDF → candidate pivot nodes → ranked by graph
   centrality. Intent detection picks traversal mode:
   - `fix bug` → debug mode (follows error paths)
   - `refactor` → blast-radius mode (who breaks?)
   - `add feature` → exploration mode

3. **Capsule** — Pivot nodes returned with **full source**. Adjacent nodes reduced to
   signatures + docstrings only ("skeletonized" — 70–90% smaller). Bounded to token budget.

**vexp architecture:**
```
Claude Code ──MCP (stdio)──► vexp-mcp (TypeScript/Node.js)
                                    │ Unix socket
                             vexp-core (Rust)
                             ├── tree-sitter parser
                             ├── petgraph (DAG)
                             ├── SQLite (FTS5)
                             ├── blake3 (file hashing)
                             └── rayon (parallel indexing)
```

**Key insight from the Reddit thread:**
- Claude only saves its own notes ~10% of the time even when asked — so passive observation
  (watching file changes at the AST level) is essential for session memory.
- Stale detection must happen at the **symbol level**, not the file level.
- Per-agent session IDs needed for multi-agent scenarios.

---

## Our differentiators vs vexp

### 1. Pure Rust end-to-end (no Node.js wrapper)
vexp uses a TypeScript/Node.js MCP adapter because the MCP SDK was JS-first when built.
codesurgeon uses `rmcp` (Anthropic's official Rust MCP SDK) — single binary, zero Node dependency.

### 2. Richer graph edges
vexp tracks: imports, calls.
codesurgeon additionally tracks:
- **Trait implementations** (`impl Foo for Bar`)
- **Type flows** (where a type propagates through function signatures)
- **Call-site annotations** (X calls Y *with these arguments*, not just "X calls Y")
- **Macro expansions** (Rust)

### 3. Language-specific depth
We target the exact stack: Python, TypeScript, TSX, JavaScript, JSX, Shell, HTML, Swift, Rust, SQL.
Each language has dedicated tree-sitter extraction logic tuned to its idioms.

### 4. Semantic chunking for long functions
vexp returns full function bodies as pivots. For 500-line functions this is wasteful.
codesurgeon can chunk bodies into logical AST blocks and return only the relevant branch
(e.g., the specific `match` arm or `if` branch containing the query-relevant logic).

### 5. Call-site annotations
Instead of just "A calls B", return: "A calls B at line 47 with `timeout=None, retries=3`".
This gives the agent the *context of the call*, not just the structural relationship.

### 6. Anti-hallucination guard
Before returning a capsule, verify every symbol FQN in it actually exists in the current index.
Flags hallucinated function names before Claude can act on them.

### 7. Diff-aware capsule for PR review
Given a `git diff`, build a capsule with: changed symbols + their callers + related test files.
Purpose-built for code review context.

### 8. CLAUDE.md auto-generation per module
Auto-generate per-module summaries as the graph is built, kept current with the code.

### 9. Optional local embeddings (candle)
vexp uses lexical search only (TF-IDF + FTS5).
codesurgeon optionally uses a small local embedding model via `candle` (HuggingFace's Rust ML
framework) for better semantic matching — no API key, runs on CPU, falls back to lexical.

### 10. Agent-aware conflict detection
When two agents have contradictory observations about the same symbol, flag the conflict
explicitly in the capsule. Helps multi-agent workflows avoid stomping on each other.

---

## Tech stack

| Component | Crate | Purpose |
|-----------|-------|---------|
| MCP protocol | `rmcp` (or manual JSON-RPC) | stdio MCP server |
| AST parsing | `tree-sitter` + language grammars | Parse source into AST |
| Dependency graph | `petgraph` | DAG of symbols + edges |
| Full-text search | `tantivy` | BM25 + FTS over symbols |
| Persistence | `rusqlite` (bundled) | SQLite + FTS5 |
| Parallel indexing | `rayon` | Multi-threaded file parsing |
| File hashing | `blake3` | Fast change detection |
| File watching | `notify` | Incremental re-index on save |
| Filesystem walk | `ignore` | Respects .gitignore |
| Token counting | chars/4 heuristic (upgrade to tiktoken-rs) | Budget enforcement |
| CLI | `clap` v4 | `codesurgeon` binary |

---

## Project structure

```
codesurgeon/
├── Cargo.toml              # workspace
├── CLAUDE.md               # MCP config + usage guide for Claude
├── PLAN.md                 # this file
├── .gitignore
└── crates/
    ├── cs-core/            # Core engine (library)
    │   └── src/
    │       ├── lib.rs
    │       ├── language.rs      # Language detection, tree-sitter grammar selection
    │       ├── symbol.rs        # Symbol, Edge, SymbolKind, EdgeKind types
    │       ├── indexer.rs       # tree-sitter AST parsing for all 8 languages
    │       ├── graph.rs         # petgraph DAG wrapper + query methods
    │       ├── db.rs            # SQLite schema + CRUD + FTS5
    │       ├── search.rs        # tantivy BM25 + TF-IDF re-ranking + intent detection
    │       ├── skeletonizer.rs  # Strip function bodies → signatures only
    │       ├── capsule.rs       # Token-budget assembly + markdown formatting
    │       ├── memory.rs        # Session observations, stale detection, anti-patterns
    │       ├── watcher.rs       # File watching + blake3 change detection
    │       └── engine.rs        # CoreEngine: top-level API wiring everything together
    ├── cs-mcp/             # MCP server binary
    │   └── src/
    │       └── main.rs          # JSON-RPC over stdio, tool definitions + dispatch
    └── cs-cli/             # CLI binary
        └── src/
            └── main.rs          # clap CLI: index, status, search, skeleton, impact, flow
```

---

## MCP tools exposed

| Tool | Description |
|------|-------------|
| `run_pipeline` | Primary tool. Auto-detects intent, returns context + impact + memories in one call |
| `get_context_capsule` | Lightweight context search, bounded to token budget |
| `get_impact_graph` | Blast-radius analysis: what breaks if this symbol changes |
| `get_skeleton` | File API surface — signatures + docstrings, no bodies |
| `search_logic_flow` | Trace execution path between two functions |
| `index_status` | Health check: symbol count, edge count, file count |
| `get_session_context` | Cross-session observations with stale flags |
| `save_observation` | Persist an insight linked to a symbol |

---

## What's done

- [x] Workspace structure (`Cargo.toml`, three crates)
- [x] `language.rs` — Language enum, extension detection, tree-sitter grammar map
- [x] `symbol.rs` — `Symbol`, `Edge`, `SymbolKind`, `EdgeKind` types with blake3 hashing
- [x] `indexer.rs` — Full tree-sitter extraction for Python, TypeScript/TSX, JavaScript/JSX,
  Shell, HTML, Rust; tree-sitter for Swift; regex fallback for SQL
- [x] `graph.rs` — petgraph DAG wrapper with centrality scoring, path finding, blast radius
- [x] `db.rs` — SQLite schema (symbols, edges, files, observations) + FTS5 virtual table
- [x] `search.rs` — tantivy BM25 index + TF-IDF re-ranking + intent detection
- [x] `skeletonizer.rs` — Language-aware body stripping, skeleton formatting
- [x] `capsule.rs` — Token-budgeted context assembly + markdown formatting
- [x] `memory.rs` — Session observations, passive capture, file thrash + dead-end detection,
  stale flagging by symbol hash
- [x] `watcher.rs` — File watcher + blake3 + deduplication
- [x] `engine.rs` — `CoreEngine` wiring all modules, parallel indexing with rayon,
  all MCP tool implementations
- [x] `cs-mcp/main.rs` — Pure JSON-RPC MCP server over stdio, all 8 tools, background indexing
- [x] `cs-cli/main.rs` — CLI with clap: index, status, search, skeleton, impact, flow, memory, observe
- [x] `CLAUDE.md` — MCP config instructions + usage guide
- [x] `.gitignore`

---

## What's left (build order)

### Phase 1 — Get it compiling (immediate)
- [x] Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- [x] `cargo build` — fixed 6 compilation errors:
  - `search.rs`: added `Value` trait import + annotated `retrieved` as `TantivyDocument` (type inference)
  - `db.rs`: materialized iterator before `stmt` was dropped (lifetime error)
  - `indexer.rs:398`: saved `find()` result before walker `c` was dropped (lifetime error)
  - `indexer.rs:387,743`: extracted name string before moving `text` (borrow-then-move, ×2)
  - `cs-cli/main.rs`: removed extra `}` in JSON format string
- [x] Verify `codesurgeon-mcp` starts and responds to MCP `initialize` — all 8 tools advertised correctly

### Phase 2 — Add to Claude Code
- [x] Add MCP config to `~/.claude/mcp_settings.json` (not settings.json — schema rejects mcpServers there)
- [x] Run `codesurgeon index` on this project — 406 symbols | 134 edges | 22 files
- [x] Verify `index_status` returns non-zero counts — confirmed via MCP tools/call

### Phase 3 — Test and tune
- [x] Test `run_pipeline` on real queries against your codebase — working after critical bug fix:
  - **Root cause:** `CoreEngine::new()` created a fresh in-memory Tantivy index on every startup;
    the 406 symbols lived in SQLite but were never loaded back → 0 pivots on every search
  - **Fix:** Added `db.all_symbols()` + `db.all_edges()`, called in `CoreEngine::new()` to warm
    both the petgraph DAG and Tantivy BM25 index from SQLite before serving any queries
  - Also added `EdgeKind::from_db_str()` (was missing) and `db.all_edges()`
  - Results: 8 pivots, 54–97% budget utilised, correct intent routing (debug/refactor/explore)
- [x] Tune `max_pivots` (default 8) and `max_adjacent` (default 20) — defaults are good for this
  codebase size (413 symbols / 22 files). No change needed.
- [x] Measure token reduction vs baseline — `search` on "capsule token budget" returns ~2786 tokens
  (70% of 4000 budget) with 8 pivots + 8 skeletons. Without codesurgeon, giving Claude all 22 source
  files (~413 symbols, ~30k tokens) is ~10× more tokens. Confirmed 90%+ reduction.

### Phase 4 — Quality improvements
- [x] Improve Python import edge resolution — `extract_imported_names()` parses `from foo import Bar`
  and `import os` to extract actual symbol names; edges 140 → 296 after re-index
- [x] Add TypeScript/JS call edge extraction — `extract_call_edges()` + `calls_in_body()` scan
  function bodies for `identifier(` patterns; capsule skeletons went from 0 → 8
- [x] Add Rust trait impl edge extraction — `extract_impl_edges()` parses `impl::Trait for Type`
  symbol names and creates `Implements` edges; impact graph now shows correct callers
- [x] Improve Swift support — upgraded entire tree-sitter ecosystem to 0.25 (ABI v15); added
  `tree-sitter-swift = "0.7"` + full `walk_swift()` extractor with class/struct/enum/extension/
  protocol/func/method support via `class_declaration.declaration_kind` field pattern

### Phase 5 — Differentiators
- [x] Semantic chunking: `chunk_for_query(body, query, max_tokens)` in `capsule.rs` — overlapping
  line windows scored against query terms; applied to pivot bodies > 300 tokens; always retains
  the function signature as first line; `build_capsule` takes `query: Option<&str>`
- [x] Call-site annotations: `calls_in_body` returns `(callee_name, args_snippet)` pairs;
  edge labels become `callee(args…)` with up to 60-char arg preview; `extract_args_snippet`
  balances parens to extract the actual argument text
- [x] Type flow tracking: `extract_type_flow_edges` in `indexer.rs` — scans function signatures
  for PascalCase identifiers matching known struct/enum/class/trait symbols; creates `References`
  edges; chained into `engine.rs` alongside import/call/impl extractors
- [x] Optional local embeddings — `fastembed` (ONNX Runtime, `AllMiniLML6V2Q`, 22 MB) behind
  `--features embeddings`; Apple Silicon Accelerate BLAS via `--features metal` (adds
  `fastembed/accelerate`); `Embedder` in `embedder.rs` wraps model in `Mutex<TextEmbedding>`;
  384-dim L2-normalised vectors stored as BLOB in `symbol_embeddings` SQLite table;
  blended 50/50 with BM25+centrality in `build_context_capsule`; non-fatal fallback to BM25-only
  if model load fails; default build unchanged (zero new deps)
- [x] Diff-aware capsule — `get_diff_capsule(diff)` parses unified diff hunks, maps line ranges
  to symbols, surfaces callers + test files; exposed as MCP tool + CLI `codesurgeon diff`
- [x] Anti-hallucination guard — `get_impact_graph` (and FQN lookups) return "Did you mean X?"
  with up to 5 fuzzy matches when exact FQN not found; `fuzzy_fqn_matches()` in `graph.rs`
- [x] Per-module CLAUDE.md auto-generation — `generate_module_docs(write_files)` groups symbols
  by directory, emits types + functions table per module; MCP tool + CLI `codesurgeon docs`

- [x] Ranking quality fix (user feedback from cs-pdfreader): three improvements to `search.rs`
  and `engine.rs`:
  1. **Test/utility file penalty** — paths containing `test`, `spec`, `mock`, `uitest` get 0.25×
     score; utility scripts (`check-*`, `run-*`, `setup*`, etc.) get 0.3×. Eliminates UITest
     setup and Python utility scripts from architectural query results.
  2. **Structural intent** — new `SearchIntent::Structural` (triggers on "coordinator", "central",
     "manager", "architecture", "controller", etc.); type definitions (`class/struct/enum/trait`)
     boosted 2.5×, `Impl` blocks 1.5×, plain callables reduced to 0.6×.
  3. **FQN deduplication** — after ranking, collapse duplicate FQN entries (keep highest score);
     prevents same symbol appearing multiple times as pivots.

### Tests added (Phase 5)
- [x] `indexer::tests::call_edges_include_args_snippet` — verifies call-site annotation labels
- [x] `indexer::tests::import_edges_resolve_python_names` — verifies Python import edge resolution
- [x] `indexer::tests::type_flow_edges_from_signatures` — verifies References edges from fn sigs
- [x] `capsule::tests::chunk_for_query_returns_relevant_window` — verifies query-driven chunking
- [x] `capsule::tests::chunk_for_query_short_body_unchanged` — short bodies returned verbatim

### Phase 6 — Distribution
- [x] GitHub repository — https://github.com/subsriram/codesurgeon
- [x] CI — `.github/workflows/ci.yml`: cargo test + clippy (-D warnings) + rustfmt --check
- [x] README with benchmark table vs baseline and vs vexp
- [x] `docs/ranking.md` — full ranking pipeline documentation
- [ ] Published CLI via `cargo install` or Homebrew (deferred — fastembed/ort native deps need crates.io compat check)

### Post-Phase-6 — Multi-root workspace support (deferred)
Currently each `codesurgeon-mcp` instance serves one workspace. Multiple codebases are handled
by running one server per codebase with distinct MCP server names:

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

Tools are namespaced by server name (`cs-frontend__run_pipeline` etc.) — Claude routes
automatically from context. Cross-codebase queries (e.g. "how does frontend call backend's
auth?") require two tool calls, one per server.

**Future work:** native multi-root support — single server, multiple `CS_WORKSPACE` paths,
aggregated symbol graph with per-root namespacing in FQNs, cross-root edge resolution.
Design notes:
- `EngineConfig` gains `workspace_roots: Vec<PathBuf>`; each root indexed into the same graph
  with FQNs prefixed by root alias (e.g. `backend::src/auth.rs::validate`)
- Single SQLite DB aggregates all roots; `files` table gains a `root` column
- `run_pipeline` accepts optional `root` filter to scope results
- CLI: `codesurgeon index --root /projects/frontend --root /projects/backend`

### Post-Phase-6 — Embeddings: metal-candle upgrade (deferred)
Consider swapping `fastembed` for `metal-candle` (`embeddings` feature) after Phase 6 ships:
- `metal-candle = { version = "1.3.0", features = ["embeddings"] }` — custom Metal MSL kernels,
  dedicated `embeddings` module for sentence transformers, faster than candle-core's built-in Metal
- Trade-off: only 482 downloads (single-author, Dec 2025) vs fastembed's production maturity
- Re-evaluate once metal-candle gains adoption or benchmarks show meaningful gains on M-series chips
- Relevant page: https://crates.io/crates/metal-candle

---

## Adding to Claude Code (quick start)

```bash
# 1. Build
cd /Users/sriram/projects/codesurgeon
cargo build --release

# 2. Add to ~/.claude/mcp_settings.json (not settings.json — schema rejects mcpServers there)
# Single codebase:
{
  "mcpServers": {
    "cs-myproject": {
      "command": "/Users/sriram/projects/codesurgeon/target/release/codesurgeon-mcp",
      "args": [],
      "env": { "CS_WORKSPACE": "/Users/sriram/projects/myproject" }
    }
  }
}

# Multiple codebases — one server entry per project, distinct names:
{
  "mcpServers": {
    "cs-frontend": {
      "command": "/Users/sriram/projects/codesurgeon/target/release/codesurgeon-mcp",
      "args": [],
      "env": { "CS_WORKSPACE": "/Users/sriram/projects/frontend" }
    },
    "cs-backend": {
      "command": "/Users/sriram/projects/codesurgeon/target/release/codesurgeon-mcp",
      "args": [],
      "env": { "CS_WORKSPACE": "/Users/sriram/projects/backend" }
    }
  }
}
# Tools are namespaced: cs-frontend__run_pipeline, cs-backend__run_pipeline etc.
# Claude routes automatically from context.

# 3. Restart Claude Code — each server indexes its workspace in the background on first start
```
