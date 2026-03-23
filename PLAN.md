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
- [x] Stale PID file after crash — `acquire_pid_lock` uses `kill -0` to detect dead processes and
  silently overwrites the stale file; no manual cleanup needed (invariant #4 in CLAUDE.md)
- [ ] Troubleshooting section in README (MCP not connecting, index not ready, second instance
  read-only mode, binary not on PATH after `cargo build`)
- [ ] Privacy statement in README — explicitly document: no network calls, no telemetry, no
  cloud dependencies; index lives in `.codesurgeon/` locally; embeddings run on-device
- [ ] Published CLI via `cargo install` or Homebrew (deferred — fastembed/ort native deps need crates.io compat check)

### Phase 7 — Language enrichment: type stubs, toolchain integration, library APIs

Goal: close the gap between what codesurgeon's tree-sitter pass can see and what agents actually
need — resolved types, macro-generated symbols, and third-party library APIs — without introducing
heavy runtime dependencies.

Enrichment runs as an **opt-in indexing-time pass** after the base tree-sitter index is built.
Results are stored in the existing SQLite schema (new `resolved_type`, `expanded_body` columns on
`symbols`; new `library` partition flag on `files`). The `content_hash` per symbol drives
incremental re-enrichment — only changed symbols are re-processed.

---

#### 7a — Tier 1: Index type stubs already on disk (all languages, near-zero effort)

No new tools required. Extend the indexer to treat these paths as a low-weight `library`
partition: indexed as skeletons only, never returned as pivots, lower ranking weight.
Fixes the most common agent failure mode: hallucinated library signatures.

| Language | Stub files to index |
|----------|---------------------|
| TypeScript / JS | `node_modules/@types/**/*.d.ts`, `node_modules/**/index.d.ts` |
| Python | `site-packages/**/*.pyi`, typeshed stubs (if pyright/mypy installed) |
| Swift | `.swiftinterface` files in Xcode toolchain + SPM package caches |
| Rust | `rustdoc --output-format json` (see 7b) covers this more completely |
| SQL | No stubs needed — schemas are self-describing |
| Shell | No type system — skip |
| HTML | Piggybacks on JS/TS stub indexing for inline scripts |

Implementation notes:
- Add `is_library: bool` column to `files` table; library symbols get ranking weight ×0.3
- Respect `.gitignore` but add explicit include rules for `node_modules/@types` and `site-packages`
- `EngineConfig` gains `index_stubs: bool` (default: true) and `stub_paths: Vec<PathBuf>` override

---

#### 7b — Tier 2: Rust toolchain enrichment (`cargo-expand` + `rustdoc` JSON)

Solves the two biggest Rust-specific blind spots: macro-generated symbols and resolved public types.

**`cargo-expand` — macro expansion**
- Run `cargo expand <module>` at index time for each Rust file
- Output is expanded Rust source — re-feed through the existing `walk_rust()` tree-sitter extractor
- Adds visibility into: `#[derive(Serialize, Debug, Clone)]` generated impls, `tokio::main` expansion,
  builder macros, proc macros
- Only re-run when the file's `content_hash` changes
- Requires `cargo-expand` installed (`cargo install cargo-expand`); skip gracefully if absent

**`rustdoc --output-format json` — resolved public API types**
- Run `cargo rustdoc -- --output-format json` once per crate at index time
- Deserialize with the `rustdoc-types` crate (native Rust, no subprocess parsing)
- Annotate existing symbols with `resolved_type` and trait impl lists from rustdoc output
- Covers: generic instantiations, associated types, full trait impl lists
- Gate behind `--features rustdoc-enrichment` to avoid mandatory `cargo rustdoc` on every workspace

Implementation notes:
- New `enricher.rs` in `cs-core/src/` — `RustEnricher` struct with `expand_macros()` and
  `annotate_from_rustdoc()` methods
- `Symbol` gains optional `resolved_type: Option<String>` and `expanded: bool` fields
- `engine.rs` runs enrichment pass after base indexing completes, async so MCP server stays responsive

---

#### 7c — Tier 2: TypeScript/JavaScript enrichment (`typescript` npm package)

> **Note:** For VS Code users, `submit_lsp_edges` (Phase 8c) is the preferred path —
> it uses the language server already running in the editor rather than spawning a
> separate subprocess. 7c remains the right approach for non-VS Code environments
> (CI, Codex, other editors) and is now priority #10 vs 8c at #7.

The `typescript` package is already present in most TS/JS projects as a dev dependency.
A small Node.js shim invoked at index time uses `ts.createProgram()` + `TypeChecker` to annotate
symbols with their resolved types — no new installs for the user.

```
codesurgeon indexer
  → detects tsconfig.json in workspace
  → spawns: node enrich-ts.js <workspace>          ← shim bundled with codesurgeon
  → ts.createProgram() over tsconfig.json
  → for each symbol: checker.getTypeAtLocation()
  → outputs NDJSON: { fqn, resolved_type, declaration_file, declaration_line }
  → codesurgeon annotates symbol graph
```

- Works for plain JS too (`allowJs: true` in shim's compiler options)
- JSDoc types in JS files resolved correctly
- `node_modules/@types/**/*.d.ts` resolution is automatic (TypeScript handles it)
- Skip gracefully if `node` not available or no `tsconfig.json` found
- Gate behind `--features ts-enrichment`

---

#### 7d — Tier 2: Python enrichment (`pyright --outputjson`)

> **Note:** For VS Code users, `submit_lsp_edges` (Phase 8c) covers Python via
> Pylance's call-hierarchy provider. 7d remains the fallback for non-VS Code
> environments. Now priority #8 in the queue.

Run `pyright --outputjson` at index time to annotate Python symbols with inferred types.
Lower priority than Tier 1 stub indexing (which covers library APIs already); adds value for
inferred types on user-defined code where annotations are absent.

- `pyright --outputjson` produces structured JSON with per-symbol type info and diagnostics
- Parse output and annotate matching symbols in the graph by file + line range
- Skip gracefully if `pyright` not on PATH
- Gate behind `--features pyright-enrichment`

---

#### 7e — Tier 3: Swift enrichment via Xcode MCP ✅

Apple ships a built-in MCP server in Xcode 26 (Settings → Intelligence → "Enable MCP").
Rather than codesurgeon reimplementing Swift type resolution, agents use Xcode MCP alongside
codesurgeon MCP: codesurgeon for semantic search + session memory, Xcode MCP for precise
Swift type and build information.

For non-Xcode Swift projects (SPM-only), fall back to `.swiftinterface` stub indexing (7a).
Community options if Xcode 26 unavailable:
- XcodeBuildMCP (https://github.com/cameroncooke/XcodeBuildMCP) — build/test/debug via MCP
- xcode-mcp-server (https://github.com/r-huijts/xcode-mcp-server) — project structure + SPM

**Implemented:**
- `detect_xcode_mcp()` — probes `xcrun --find mcpbridge` once at startup via `OnceLock`;
  result cached for the process lifetime
- `swift_enrichment_hint()` — two-path message: "Xcode MCP available, use it" vs
  "not found, tree-sitter only — here's how to fix it"
- `run_pipeline` — appends hint when any pivot or skeleton is a `.swift` file
- `index_status` — reports Xcode MCP availability as a status line
- `IndexStats.xcode_mcp_available: bool` — serialised in JSON output
- `CLAUDE.md` — agent-facing failover instructions (try Xcode MCP → fall back to
  tree-sitter with explicit caveat about missing resolved types)
- `README.md` — setup instructions + community alternatives for Xcode < 26

---

#### 7f — Shell and SQL: parser-level fixes (no external tools)

**Shell:** The current extractor captures function definitions only. The primary gap is
`source ./lib.sh` / `. ./util.sh` — file-level import edges that enable graph traversal across
shell scripts. Fix at the tree-sitter level in `walk_shell()` in `indexer.rs`. No external tool.

**SQL:** Schemas are already self-describing; no type enrichment needed. The gap is cross-schema
references and stored procedure call graphs (e.g. a procedure calling another procedure).
Extend `walk_sql()` to extract `CALL` and `EXEC` statements as `Calls` edges.

---

#### Build order within Phase 7

1. **7a** — stub indexing (highest ROI, contained change to indexer + db)
2. **7b** — `cargo-expand` (re-uses existing tree-sitter pass, additive)
3. **7b** — `rustdoc JSON` (new `rustdoc-types` dep, annotates existing symbols)
4. **7f** — shell `source` edges + SQL call edges (parser-level, self-contained)
5. **7c** — TypeScript shim (requires bundling a Node.js script)
6. **7d** — pyright (subprocess integration, lowest incremental value given 7a)
7. **7e** — Xcode MCP (documentation only)

---

### Phase 8 — vexp parity + tool improvements

Gaps identified by reviewing vexp.dev/docs. Split into quick wins (parameter additions to
existing tools) and new tools.

---

#### 8a — Quick wins: parameter additions to existing tools (Low effort)

Four small additions that close the most visible gaps with vexp. No new tools, no schema
changes — all are additive parameters with backward-compatible defaults.

**`observation` param on `run_pipeline`**
Auto-save an observation as part of the pipeline call, saving a round-trip.
```
run_pipeline(task="...", observation="discovered that X always retries 3 times")
```

**`include_tests` param on `run_pipeline` / `get_context_capsule`**
Currently test files are penalised 0.25× in ranking with no override. Add `include_tests: bool`
(default `false`) to let callers opt in when working on tests directly.

**`format` param on `get_impact_graph`**
Add `format: "list" | "tree" | "mermaid"` (default `"list"`). The Mermaid option outputs a
diagram that renders in Claude's markdown — useful for visualising blast radius at a glance.

**`max_paths` on `search_logic_flow`**
Currently returns only the shortest path. Add `max_paths: u32` (default `1`) to return
multiple parallel call chains — useful when there are several routes between A and B.

---

#### 8b — `search_memory` tool (Low-med effort)

A dedicated hybrid memory search tool, separate from `get_session_context`. vexp uses
text relevance + semantic similarity + recency + code-graph proximity. codesurgeon currently
only surfaces memories passively through `run_pipeline` or chronologically via
`get_session_context` — there is no way to directly query past observations.

```
search_memory(query="how does the retry backoff work", max_results=10)
```

Implementation: reuse the existing BM25 + embeddings stack already in `search.rs`, scoped
to the `observations` table rather than the symbol table. The memory store already has
`content` and `symbol_fqn` fields that are indexable.

---

#### 8c — `submit_lsp_edges` tool (Med effort)

The most architecturally interesting gap. vexp accepts type-resolved call edges submitted
from a VS Code Language Server extension, supplementing static analysis with precise type
information. This is the "thin LSP-client bridge" approach: rather than codesurgeon spawning
language servers (Phase 7b–7d), IDE users push edges *to* codesurgeon from the language
server that's already running in their editor.

```
submit_lsp_edges(edges=[
  {"caller": "src/main.rs::handle_request", "callee": "src/db.rs::Database::query"},
  ...
])
```

Edges are stored in the graph DB as `EdgeKind::LspResolved` (new variant), weighted higher
than tree-sitter-inferred `Calls` edges in `get_impact_graph` and `search_logic_flow`.

**Why this matters:** For VS Code users, this would replace the need for 7c (TS shim) and 7d
(pyright) entirely — the language server already running in the editor provides the resolved
edges without codesurgeon needing to spawn subprocesses. For non-VS Code users, 7c/7d remain
the fallback.

Implementation notes:
- New `EdgeKind::LspResolved` variant in `symbol.rs`
- `submit_lsp_edges` stores edges by FQN pair; tolerates unknown FQNs gracefully (skips, logs)
- Edges expire after configurable TTL (default: 24h) since LSP state can become stale
- A companion VS Code extension (separate repo) would wire `vscode.languages` call-hierarchy
  provider → codesurgeon MCP on file save

---

#### 8d — `workspace_setup` tool (Low effort, low priority)

Onboarding tool that detects the agent type, generates a `workspace.json` config template,
and returns setup instructions. Reduces friction for new users. Low priority vs. the above
since codesurgeon's `generate_module_docs` already covers the CLAUDE.md onboarding case.

---

### Phase 9 — Memory system improvements

Goal: close the gap between codesurgeon's basic observation store and vexp's more
sophisticated session memory. Ordered from highest to lowest value/effort ratio.

---

#### 9a — Auto-capture tool calls as observations (Low effort, high value)

Currently codesurgeon only passively captures file-change events. vexp records every
`run_pipeline` and `get_context_capsule` call as a compact observation (task + top pivot FQNs).
This builds a picture of what the agent has explored across sessions without any manual saves —
the most common case where session context is actually useful.

Implementation:
- After each `run_pipeline` / `get_context_capsule` call, auto-save a compact observation:
  `ObservationKind::Auto` with content = task description + top 3 pivot FQNs
- Gate behind a dedupe check: if an identical task was recorded in the last 30 minutes, skip
- No new schema changes — `ObservationKind::Auto` already exists or is a one-line addition

---

#### 9b — Session TTL + compression (Low-med effort, high value)

The observation store currently grows unbounded. vexp auto-compresses sessions after 2 hours
of inactivity into structural summaries and enforces:
- Auto-observations: expire after session compression
- Manual observations: persist permanently
- Sessions older than 90 days: fully deleted

Implementation:
- Add `compressed_at: Option<DateTime>` and `expires_at: Option<DateTime>` to the `observations`
  table
- Background task (runs on startup) compresses inactive sessions: extract key paths, FQNs, and
  terms into a single `ObservationKind::Summary` entry; mark auto-observations as expired
- Manual observations (`ObservationKind::Manual`) never get an `expires_at`
- Prune observations where `expires_at < now()` on each startup

---

#### 9c — L1 / L2 / L3 detail levels for `search_memory` (Low effort)

Build this into `search_memory` (Phase 8b) from the start rather than retrofitting later.
vexp surfaces results at three token levels:
- **L1** (~20 tokens): headline only — symbol name + one-line summary
- **L2** (~50 tokens): standard — includes linked symbol signature
- **L3** (~100 tokens): full observation content

The caller specifies the level; default is L2. Prevents memory results from eating the token
budget when a capsule already contains the relevant code.

---

#### 9d — Memory consolidation (Med effort)

Semantically similar auto-observations are merged into a single consolidated entry.
codesurgeon currently accumulates duplicates — e.g. 20 `run_pipeline` calls on the same
module produce 20 near-identical observations.

Implementation:
- On session compression (9b), cluster auto-observations by embedding cosine similarity
  (threshold ~0.92) using the existing embeddings stack
- Replace each cluster with a single `ObservationKind::Consolidated` entry whose content
  merges the unique terms across the cluster
- Manual observations never merge

---

#### 9e — Richer AST change categories (Med effort)

vexp's file watcher classifies changes into 6 specific categories:
`Added`, `Removed`, `Renamed`, `SignatureChanged`, `BodyChanged`, `VisibilityChanged`.
codesurgeon detects that a file changed and counts re-indexed symbols, but doesn't
classify the change type.

Implementation:
- In `reindex_file()`, compare the new symbol list against the previous DB snapshot:
  - symbol present in new but not old → `Added`
  - symbol present in old but not new → `Removed`
  - same `start_line`, different `name` → `Renamed`
  - same FQN, different `signature` → `SignatureChanged`
  - same FQN + signature, different `body` → `BodyChanged`
  - (visibility change requires language-specific parsing — defer)
- Store change category in the auto-captured observation (9a) for richer session context
- Surface in `get_diff_capsule` output

---

#### 9f — Project rules (High effort, lower priority)

When 3+ similar observations recur in the same scope, vexp auto-generates rule candidates
and injects them as standing conventions into capsule responses — e.g. "this codebase always
uses `anyhow::Result`" stops being a repeated observation and becomes a rule.

Implementation:
- After each session compression (9b), scan consolidated observations for recurring patterns
  by scope (directory or symbol namespace)
- Candidate rules require: 3+ similar observations, recency within 30 days, no contradicting
  observations
- Rules stored as `ObservationKind::Rule`; injected at the top of `format_capsule` output
  when the query scope matches
- Manual review step: rules start as `pending` and are promoted to `active` only after the
  agent (or user) confirms them via `save_observation(kind="rule", ...)`

---

## Benchmark plan

### B1 — SWE-bench Verified (external, end-to-end quality)

The gold-standard external validation for coding agents is
[SWE-bench Verified](https://swebench.com) — 100 real GitHub issues, same model and cost
cap across all agents. vexp's published results (Claude Opus 4.5, $3/task cap, 250 turns):

| Agent | Pass@1 | $/Task |
|-------|--------|--------|
| vexp + Claude Code | **73%** | **$0.67** |
| OpenHands | 70% | $1.77 |
| Sonar Foundation | 70% | $1.98 |

**Key insight from per-repo breakdown:**
- **astropy: 80% (vexp) vs 40% (competitors)** — import-heavy, interconnected dependencies;
  exactly the pattern a symbol dependency graph is built for
- **matplotlib: 43% (vexp) vs 86% (Sonar Foundation)** — rendering/procedural logic; less
  amenable to symbolic traversal, more about symbolic graph traversal

This maps onto codesurgeon's expected profile: strong on Rust (trait/impl graphs), Python
backend (deep import chains), TypeScript (module boundaries); weaker on procedural/numerical
code with few call-graph edges.

**When to run:** once Phase 8 (vexp tool parity) and Phase 9 (session memory) are stable.
The benchmark is open-source and runs in under 10 minutes.

**What to measure:**
- Pass@1 on the same 100-task subset with the same model (Claude Opus) and cost cap ($3/task)
- $/task with codesurgeon vs without (bare Claude Code, same task set)
- Per-repo breakdown to identify where the symbol graph helps and where it doesn't
- Comparison row against vexp's published numbers

**Setup:** `pip install swebench` → clone the 100-task subset → run Claude Code with
codesurgeon MCP enabled → record pass/fail and token cost per task.

---

### B2 — Token savings micro-benchmark (local, per-workspace)

Measures actual token reduction on real codebases without a full agent run.
Run this before and after ranking changes to catch regressions.

**Method:**
1. Take a fixed set of 20 representative queries against a known codebase (e.g. codesurgeon itself)
2. Record `total_tokens`, `candidate_file_tokens`, and `workspace_tokens` per query (from `query_log`)
3. Compute candidate-file savings %, workspace savings %, skeletonisation savings %
4. Compare against a baseline snapshot committed to `benches/token_baseline.json`

**Script:** `cargo bench --bench token_savings` — runs the 20 queries headlessly against
the codesurgeon workspace, writes results to `benches/token_savings_latest.json`, diffs
against baseline, fails CI if candidate-file savings drop below 70%.

**Queries to include (codesurgeon self-benchmark):**
- "fix the retry logic" → should pivot on engine.rs + capsule.rs
- "add a new language parser" → should pivot on parser/ + symbol.rs
- "token budget assembly" → should pivot on capsule.rs
- "how does BM25 search work" → should pivot on search.rs
- "session memory observations" → should pivot on memory.rs / observations table

---

### B3 — Index performance benchmark (local, throughput)

Measures indexing speed and query latency. Run in CI on every merge to catch performance
regressions.

**Metrics:**
- Index build time: files/sec and symbols/sec for a mid-size repo (codesurgeon itself ~30 files)
- Query latency: BM25 time, graph centrality time, embedding time, capsule assembly time
- Memory usage: peak RSS during index build and during query

**Script:** `cargo bench --bench indexing` — uses `criterion` for statistical latency
measurements. Baseline committed; CI fails if p95 query latency exceeds 500ms or indexing
throughput drops below 50 files/sec.

---

### B4 — Ranking quality spot-checks (manual, periodic)

Automated benchmarks can't easily measure "did the right symbol end up as a pivot?" Run
these manually after any change to `engine.rs`, `search.rs`, or `graph.rs`.

**Spot-check queries (codesurgeon self-index):**

| Query | Expected top pivot | Pass? |
|---|---|---|
| "token budget" | `capsule::build_capsule` | |
| "BM25 search" | `search::bm25_search` | |
| "graph centrality" | `graph::centrality_scores` | |
| "session memory stale" | `memory::mark_stale` | |
| "PID lock second instance" | `main::acquire_pid_lock` | |

Record results in `benches/ranking_spotchecks.md` after each manual run. If a spot-check
fails, investigate before merging — ranking regressions are hard to catch automatically.

---

### When to run each benchmark

| Benchmark | When |
|---|---|
| B1 SWE-bench | After Phase 8 + Phase 9 ship; re-run after major ranking changes |
| B2 Token savings | Every merge via CI; baseline update requires explicit approval |
| B3 Index performance | Every merge via CI; regression = build failure |
| B4 Ranking spot-checks | Manually after any change to `engine.rs`, `search.rs`, `graph.rs` |

---

## Priority queue — what to build next

Criteria: agent value (context quality improvement), breadth (languages/users affected),
effort, risk, dogfooding opportunity (can codesurgeon benefit on itself immediately).

| Priority | Item | Effort | Key reason |
|----------|------|--------|------------|
| ✅ done | 7e Xcode MCP | Zero | Free — guidance auto-injected by `generate_module_docs` |
| 1 | 8a Quick wins | Low | Four parameter additions; closes most visible vexp gaps immediately |
| 2 | 9a Tool call auto-capture | Low | Builds cross-session picture without manual saves; high value, tiny change |
| 3 | 7a Stub indexing | Low-med | Highest ROI on enrichment; fixes hallucinated library signatures |
| 4 | 7f Shell/SQL edges | Low | Quick win; contained tree-sitter changes; no new deps |
| 5 | 11a+b Secrets exclusion + `.codesurgeonignore` | Low | Security gap; prevents secrets leaking into capsules |
| 6 | 9b Session TTL + compression | Low-med | Prevents unbounded growth; lifecycle for auto vs manual observations |
| 7 | 7b `cargo-expand` | Med | Macro blind spot; codesurgeon dogfoods on itself immediately |
| 8 | 7b `rustdoc` JSON | Med | Resolved types for Rust; follows from `cargo-expand` work |
| 9 | 8b `search_memory` + 9c L1/L2/L3 | Low-med | Build together — detail levels should be in from the start |
| 10 | 8c `submit_lsp_edges` | Med | Smarter than 7c/7d for IDE users; edges pushed from running LSP |
| 11 | 9d Memory consolidation | Med | Deduplicates auto-observations; depends on 9b compression being in place |
| 12 | 7d pyright | Low-med | Fallback for non-IDE Python users after `submit_lsp_edges` lands |
| 13 | 9e Richer AST change categories | Med | Improves observation quality; depends on 9a auto-capture |
| 14 | 10a Manifest + 10c opt-out | Low | Incremental rebuild on clone; almost free given existing blake3/files table |
| 15 | 11c+d Local observability (stats CLI + `get_stats`) | Low | `query_log` table + `codesurgeon stats`; makes token savings concrete and visible |
| 16 | Phase 6 distribution | Med | `cargo install` / Homebrew; gate on product maturity |
| 17 | 7c TS compiler shim | Med | Lower priority — `submit_lsp_edges` covers VS Code TS users |
| 18 | 10b Git merge driver | Med | Union merge for manifest.json; most useful once distributed |
| 19 | Multi-root workspace | High | Wait until enrichment + memory solid; schema migration risk |
| 20 | 9f Project rules | High | Powerful but complex; needs 9b + 9d as foundation |
| 21 | 8d `workspace_setup` | Low | Nice to have; `generate_module_docs` already covers onboarding |
| ∞ | metal-candle upgrade | High risk | `fastembed` works; single-author crate; defer indefinitely |
| ∞ | MCP resources | Med | Browsable index URI scheme; useful for client UI; defer until tools mature |

---

**✅ done — 7e Xcode MCP** · Zero effort
Guidance auto-injected into Swift projects via `generate_module_docs`; `run_pipeline`
appends inline hint when Swift symbols appear. No code needed in target projects.

---

**#1 — 8a Quick wins (parameter additions)** · Low effort
Four backward-compatible additions in one sprint before anything else — highest ratio of
agent value to implementation cost in the entire backlog.

---

**#2 — 9a Tool call auto-capture** · Low effort
Four backward-compatible additions to existing tools that close the most visible gaps
with vexp in a single sprint: `observation` on `run_pipeline`, `include_tests` flag,
`format`/Mermaid on `get_impact_graph`, `max_paths` on `search_logic_flow`. No new
infrastructure, no schema changes.

---

Every `run_pipeline` and `get_context_capsule` call auto-saved as a compact observation
(task + top pivot FQNs). Builds cross-session exploration history without any manual saves.
Tiny change — one `save_observation` call at the end of each tool dispatch with a 30-minute
dedupe guard.

---

**#3 — 7a Stub indexing** · Low-med effort
Highest ROI of any remaining item. Indexes `.d.ts`, `.pyi`, and `.swiftinterface` files
already on disk — no new tools, no new deps. Fixes the most common agent failure mode
(hallucinated library signatures) across all supported languages simultaneously.
Foundational: 7c and 7d both add diminishing value once stubs are indexed.

---

**#4 — 7f Shell `source` edges + SQL `CALL` edges** · Low effort
Two self-contained changes to `indexer.rs`; no new crates, no schema changes, no subprocess
integration. `get_impact_graph` and `search_logic_flow` are currently broken for Shell and SQL
because cross-file/cross-procedure edges are missing. Low enough effort to ship in the same
sprint as 7a.

---

**#5 — 11a+b Secrets exclusion + `.codesurgeonignore`** · Low effort
Two small indexer changes that close a real security gap.

**11a — Secrets exclusion**: auto-skip files matching sensitive patterns during indexing.
Never index these as pivots or skeletons; log them as excluded at index time:
- `**/.env`, `**/.env.*`, `**/.env.local`, `**/.env.production`, `**/.env.development`
- `**/*secret*`, `**/*credential*`, `**/*password*`
- Files whose first 20 lines match common API key patterns (e.g. `[A-Z0-9]{20,}`)

Uses the `ignore` crate (already a transitive dep via tree-sitter tooling). Prevents secrets
leaking into capsule output handed to the agent — and by extension into Claude's context window.

**11b — `.codesurgeonignore`**: project-level exclusion file using gitignore syntax, layered
on top of `.gitignore`. Lets users exclude `fixtures/`, `testdata/`, `vendor/`, generated
files, or anything else they don't want in the index. Parsed by the same `ignore` crate pass.

---

**#6 — 9b Session TTL + compression** · Low-med effort
Prevents the observation store growing unbounded. Auto-compress sessions idle for 2+ hours
into structural summaries; expire auto-observations; delete sessions older than 90 days;
manual observations persist permanently. Needs to land before 9d (consolidation) since
compression is when merging happens.

---

**#7 — 7b `cargo-expand`** · Med effort
Solves the most painful Rust blind spot: macro-generated symbols (`#[derive(...)]`, `tokio::main`,
proc macros) are invisible to tree-sitter. Output is Rust source, so the existing `walk_rust()`
pass reuses with no new parsing logic. codesurgeon can dogfood the result on its own codebase
immediately — serde/tokio derives become visible in the graph.

---

**#8 — 7b `rustdoc` JSON** · Med effort
Natural follow-on once `enricher.rs` is in place from #4. `cargo rustdoc --output-format json`
gives resolved types and full trait impl lists; deserialise with the `rustdoc-types` crate
(native Rust, no subprocess parsing). Annotates existing symbols rather than adding new ones.

---

**#9 — 8b `search_memory` + 9c L1/L2/L3 detail levels** · Low-med effort
Build together — retrofitting detail levels after the fact is harder than starting with them.
`search_memory` reuses the existing BM25 + embeddings stack scoped to the `observations`
table. Results at three token levels: L1 (~20 tokens, headline only), L2 (~50 tokens,
standard + linked symbol signature), L3 (~100 tokens, full content). Caller specifies level;
default L2.

---

**#10 — 8c `submit_lsp_edges`** · Med effort
The smartest enrichment architecture: IDE users push type-resolved edges from the language
server already running in their editor, rather than codesurgeon spawning subprocesses.
For VS Code users this replaces 7c (TS shim) and 7d (pyright) entirely. New `EdgeKind::LspResolved`
variant, TTL-based expiry, graceful handling of unknown FQNs. Companion VS Code extension
needed (separate repo).

---

**#11 — 9d Memory consolidation** · Med effort
Cluster semantically similar auto-observations at session compression time using the existing
embeddings stack (cosine similarity ~0.92 threshold). Replace each cluster with a single
consolidated entry. Requires 9b (compression) to be in place first. Manual observations
never merge.

---

**#12 — 7d pyright** · Low-med effort
Fallback for Python users not running VS Code (where `submit_lsp_edges` isn't available).
Subprocess pattern established by #4/#5; mostly wiring. Lower value after 7a covers `.pyi`
stubs — only adds inferred types for unannotated user-defined code.

---

**#13 — 9e Richer AST change categories** · Med effort
Classify file watcher events into Added / Removed / Renamed / SignatureChanged / BodyChanged
by comparing new symbol list against the previous DB snapshot in `reindex_file()`. Enriches
auto-captured observations (9a) and `get_diff_capsule` output. Depends on 9a being in place
so there's something to enrich.

---

**#14 — 10a Manifest + 10c opt-out** · Low effort
Serialise the existing `files` table to `.codesurgeon/manifest.json` after each index pass.
On a fresh clone with no `index.db`, read the manifest and re-index only files whose hashes
differ — seconds instead of a full re-index. `CS_TRACK_MANIFEST=false` to opt out. Almost
entirely free given blake3 hashing and incremental re-indexing are already in place.

---

**#15 — 11c+d Local observability** · Low effort
Persist per-query metrics in a `query_log` SQLite table and expose them via CLI and MCP tool.
Makes codesurgeon's value concrete and measurable without any data leaving the machine.

**Schema — `query_log` table:**
```sql
CREATE TABLE query_log (
  id                    INTEGER PRIMARY KEY,
  timestamp             TEXT    NOT NULL,
  task                  TEXT,
  intent                TEXT,
  pivot_count           INTEGER,
  skeleton_count        INTEGER,
  total_tokens          INTEGER,
  budget_tokens         INTEGER,
  candidate_file_tokens INTEGER,   -- sum of full-file tokens for files with returned symbols
  workspace_tokens      INTEGER,   -- total workspace tokens (cached at index time)
  cost_usd              REAL,      -- total_tokens × configured $/token rate
  cost_saved_usd        REAL,      -- (candidate_file_tokens - total_tokens) × $/token
  latency_ms            INTEGER,
  bm25_ms               INTEGER,
  graph_ms              INTEGER,
  embed_ms              INTEGER,
  assembly_ms           INTEGER,
  memory_hit            INTEGER,   -- 1 if any memory appeared in capsule
  languages_hit         TEXT       -- JSON array e.g. ["rust","typescript"]
);
```

**Token savings baselines (all three tracked per query):**
- *Candidate-file* (headline): `(candidate_file_tokens - total_tokens) / candidate_file_tokens` — most honest; "files you'd have read to find this code"
- *Workspace*: `(workspace_tokens - total_tokens) / workspace_tokens` — matches README benchmark claim
- *Skeletonisation*: `(adjacent_full_tokens - adjacent_skeleton_tokens) / adjacent_full_tokens` — compression contribution only

**Cost tracking:** `cost_saved_usd = tokens_saved × rate`, where `rate` is configurable via
`CS_TOKEN_RATE_USD` (default: claude-sonnet-4 input pricing). Consistent with vexp's SWE-bench
finding that $0.67/task vs $1.98/task is the headline metric users remember — dollar figures
communicate value more clearly than token percentages.

**`codesurgeon stats` CLI output:**
```
── Cost savings (last 30 days) ─────────────────────────
  Total queries:        47
  Estimated cost saved: $0.55  (@ $3.00/Mtok, candidate-file baseline)
  Avg token reduction:  90.3%  ·  182,400 tokens saved

── Latency ─────────────────────────────────────────────
  Median:    180ms    p95: 420ms

── Intent breakdown ────────────────────────────────────
  debug 38%  ·  add 30%  ·  refactor 21%  ·  explore 11%

── Language distribution ───────────────────────────────
  Rust 62%  ·  TypeScript 28%  ·  SQL 8%  ·  Python 2%

── Session memory ──────────────────────────────────────
  Observations: 12  ·  Hit rate: 58%  ·  Stale: 2

── Index health ────────────────────────────────────────
  Symbols: 1,247  ·  Files: 89  ·  Parse errors: 0  ·  Freshness: 100%
```

**`get_stats` MCP tool (11d):** thin wrapper over `query_log` aggregates, identical output to
`codesurgeon stats`. Lets the agent surface stats in conversation: `get_stats(days=30)`.
Also feeds a one-liner into `get_session_context`: "12 queries this session · ~$0.14 saved · avg 180ms latency".

---

**#16 — Phase 6 distribution (`cargo install` / Homebrew)** · Med effort
Doesn't improve context quality — only discoverability and adoption friction. The blocker is
`fastembed`/`ort` native deps that need crates.io compat work. Worth tackling once the
enrichment story is solid enough to be worth distributing.

---

**#17 — 7c TypeScript compiler shim** · Med effort
Demoted from #7 to #10 because `submit_lsp_edges` covers the same gap for VS Code TS users
with better architecture. Remains useful as a standalone option for non-VS Code environments.
FQN alignment between tree-sitter and the TypeScript compiler is still the main risk.

---

**#18 — 10b Git merge driver** · Med effort
`codesurgeon merge-manifest` CLI subcommand + `.gitattributes` registration via
`codesurgeon git-setup`. Takes union of file hashes across base/ours/theirs versions.
Most valuable once distributed (#14) and teams are actively using the manifest.

---

**#19 — Multi-root workspace support** · High effort
High real-world value (most non-trivial projects span frontend + backend + shared libs), but
architecturally significant: schema migration (`root` column, FQN namespacing), PID lock
rethink, `EngineConfig` overhaul. Do this after enrichment is stable so the SQLite schema
isn't migrated twice.

Design is informed by vexp's multi-repo model (see deferred section below):
- `workspace.json` declares each repo with `alias`, `path`, `languages`, `role` (consumer/provider), and typed `cross_repo_links` (e.g. `"type": "openapi"` for API contracts)
- Consumer repos query provider APIs; typed links enable contract-driven cross-boundary resolution instead of heuristic name-matching
- All observations tagged with `repo_alias` so memory stays scoped when working across repos
- MCP tools accept optional `repos` parameter to scope or fan-out queries
- VS Code `.code-workspace` file auto-detection for zero-config multi-root
- Machine-level shared cross-repo registry (separate from per-repo indexes) for fast cross-root symbol lookup

---

**#20 — 9f Project rules** · High effort
When 3+ similar observations recur in the same scope, auto-generate rule candidates and
inject them as standing conventions into capsule responses. Requires 9b (compression) and
9d (consolidation) as foundations — rules are derived from consolidated observation clusters.
Rules start as `pending` and are promoted to `active` only after agent/user confirmation.

---

**#21 — 8d `workspace_setup`** · Low effort, low priority
Onboarding tool that generates config templates. `generate_module_docs` already covers the
CLAUDE.md onboarding case. Add this when distribution (#9) is done and new-user friction
becomes the main concern.

---

**∞ — metal-candle embeddings upgrade** · Defer indefinitely
`fastembed` works. metal-candle is a single-author crate with ~482 downloads (Dec 2025);
swapping would invalidate all existing user embeddings (full re-index required) for a
performance gain that only benefits Apple Silicon. Re-evaluate if it gains meaningful adoption.

---

### Phase 10 — Git integration: manifest-based incremental rebuild

Goal: make codesurgeon team- and multi-machine-friendly by tracking a lightweight manifest
in git. New clones and pulls rebuild only changed files rather than re-indexing from scratch.

The core infrastructure is already in place — `blake3` hashing on every file, a `files`
table in SQLite with paths and hashes, and incremental re-indexing logic that already skips
unchanged files. The manifest is effectively the `files` table serialised to JSON.

---

#### 10a — Manifest file: `.codesurgeon/manifest.json` (Low effort)

After each index pass, write a `manifest.json` alongside `index.db`:

```json
{
  "version": 1,
  "workspace": "/projects/myapp",
  "updated_at": "2026-03-23T17:00:00Z",
  "files": {
    "src/main.rs":    "a3f1c2d4...",
    "src/engine.rs":  "b7e9a1f2...",
    ...
  }
}
```

- Serialised from the `files` table after `index_workspace_inner()` completes
- `index.db` stays gitignored; `manifest.json` is git-tracked
- On startup, if `manifest.json` exists but `index.db` is absent or empty: read manifest,
  compare hashes against local files, re-index only changed files — incremental rebuild in
  seconds on a fresh clone
- `index_status` reports manifest age and file count

Implementation: new `write_manifest()` + `read_manifest()` in `engine.rs`; called at the
end of `index_workspace_inner()` and at the start of `CoreEngine::new()`.

---

#### 10b — Git merge driver for `manifest.json` (Med effort)

When two branches each update different source files, their `manifest.json` entries don't
conflict — the correct merge is the union of all file hashes (newest hash wins per file).
Without a merge driver, git treats this as a text conflict requiring manual resolution.

Register a custom merge driver via `.gitattributes`:

```
# .gitattributes
.codesurgeon/manifest.json merge=codesurgeon-manifest
```

And in `.gitconfig` (or the project-level `.git/config`):

```
[merge "codesurgeon-manifest"]
  name = codesurgeon manifest merge driver
  driver = codesurgeon merge-manifest %O %A %B %P
```

The `merge-manifest` CLI subcommand reads the three versions (base, ours, theirs), takes
the union of file entries (theirs wins on conflict for any given file), and writes the merged
result to `%A`. Exit 0 on success, 1 if it cannot resolve.

- New `codesurgeon merge-manifest` subcommand in `cs-cli`
- Only registers if the user runs `codesurgeon git-setup` (opt-in, not forced on install)
- `codesurgeon git-setup` also adds the `.gitattributes` entry if not already present

---

#### 10c — `vexp.autoCommitIndex` equivalent: `CS_TRACK_MANIFEST` (Low effort)

Environment variable (default: `true`) to opt out of manifest tracking — for users who
prefer to gitignore `.codesurgeon/` entirely. When `false`, `write_manifest()` is skipped
and the `.codesurgeon/` directory is not expected to be git-tracked.

---

#### Build order within Phase 10

1. **10a** — manifest write/read (highest ROI, almost free given existing infrastructure)
2. **10c** — opt-out env var (one-liner, add alongside 10a)
3. **10b** — git merge driver (separate sprint; most useful once distributed)

---

### Phase 11 — Privacy, security & local observability

Goal: close the security gap around secrets in the index, give users control over index scope,
and make codesurgeon's value measurable without any data leaving the machine.

---

#### 11a — Secrets exclusion (Low effort)

Auto-skip files matching sensitive patterns during the indexer's file walk. These files are
never indexed as pivots or skeletons, never appear in capsule output, and are logged as
excluded in `index_status`.

Patterns excluded by default:
- `.env`, `.env.*`, `.env.local`, `.env.production`, `.env.development`
- `*secret*`, `*credential*`, `*password*` (basename match)
- Files whose first 20 lines contain a bare API key pattern: `[A-Z0-9_]{20,}=`

Uses the `ignore` crate (already in the dependency tree) for glob matching.
A new `CS_ALLOW_SECRETS=1` env var opt-out for users who deliberately want `.env` files indexed
(e.g. non-sensitive development configs).

---

#### 11b — `.codesurgeonignore` (Low effort)

Project-level exclusion file using gitignore syntax, layered on top of `.gitignore`.
Parsed by the same `ignore` crate pass as 11a.

```
# .codesurgeonignore
fixtures/
testdata/
vendor/
**/*.generated.ts
migrations/
```

Useful for: large generated files that bloat the index, test fixtures that distort rankings,
vendored copies of third-party code that shouldn't be treated as project symbols.

---

#### 11c — Local observability: `query_log` + `codesurgeon stats` (Low effort)

Persist per-query metrics in a `query_log` SQLite table written at the end of every
`run_pipeline` and `get_context_capsule` call. All data stays local.

Schema:
```sql
CREATE TABLE query_log (
  id                    INTEGER PRIMARY KEY,
  timestamp             TEXT    NOT NULL,
  task                  TEXT,
  intent                TEXT,
  pivot_count           INTEGER,
  skeleton_count        INTEGER,
  total_tokens          INTEGER,
  budget_tokens         INTEGER,
  candidate_file_tokens INTEGER,  -- sum of full-file tokens for files with returned symbols
  workspace_tokens      INTEGER,  -- cached at index time
  cost_usd              REAL,     -- total_tokens × CS_TOKEN_RATE_USD
  cost_saved_usd        REAL,     -- (candidate_file_tokens - total_tokens) × CS_TOKEN_RATE_USD
  latency_ms            INTEGER,
  bm25_ms               INTEGER,
  graph_ms              INTEGER,
  embed_ms              INTEGER,
  assembly_ms           INTEGER,
  memory_hit            INTEGER,  -- 1 if any memory appeared in capsule
  languages_hit         TEXT      -- JSON array
);
```

`codesurgeon stats` CLI: reads from `query_log`, leads with **cost saved** (dollar figure)
rather than token % — consistent with vexp's SWE-bench finding that $/task is the metric
users internalise. Token % surfaced as secondary. Rate configurable via `CS_TOKEN_RATE_USD`
(default: claude-sonnet-4 input pricing).

Token savings baselines (all stored per query):
- **Candidate-file** (headline): files containing returned symbols — most honest
- **Workspace**: total workspace tokens — matches README claim, most dramatic
- **Skeletonisation**: adjacent full-body vs skeleton — compression contribution alone

---

#### 11d — `get_stats` MCP tool (Low effort)

Thin wrapper over `query_log` aggregates, returns the same summary as `codesurgeon stats`.
Lets the agent surface stats inline: `get_stats(days=30)`.

Also injects a one-liner into `get_session_context` output:
`"12 queries this session · ~$0.14 saved · avg 180ms latency"`

---

#### Build order within Phase 11

1. **11a + 11b** — secrets exclusion + ignore file (single indexer PR, no schema change)
2. **11c** — `query_log` schema + write path + `codesurgeon stats` CLI
3. **11d** — `get_stats` MCP tool (add after CLI is working and numbers look right)

---

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

**Future work:** native multi-root support — single server, multiple workspace roots,
aggregated symbol graph with per-root namespacing in FQNs, cross-root edge resolution.

Design informed by vexp's multi-repo model:

**Configuration — `workspace.json`**
```json
{
  "repos": [
    {
      "alias": "frontend",
      "path": "/projects/frontend",
      "languages": ["typescript", "tsx"],
      "role": "consumer",
      "cross_repo_links": [
        { "alias": "backend", "type": "openapi", "spec": "./openapi/backend.yaml" }
      ]
    },
    {
      "alias": "backend",
      "path": "/projects/backend",
      "languages": ["rust"],
      "role": "provider"
    }
  ]
}
```

- **Consumer/provider roles**: provider repos expose public APIs; consumer repos query them.
  Role tagging lets the ranker boost provider symbols when a consumer repo query crosses the boundary.
- **Typed `cross_repo_links`**: `"type": "openapi"` (or `grpc`, `graphql`, `ts-types`) enables
  contract-driven cross-boundary resolution — resolves to the actual schema definition rather
  than heuristic name-matching across repos.
- **`repos` parameter on MCP tools**: `run_pipeline(task="...", repos=["frontend","backend"])`
  fans out to both indexes and merges capsules; omitting it defaults to all roots.
- **`repo_alias` on observations**: every saved observation is tagged with its originating repo
  alias so `get_session_context` can filter memories to the relevant repo.
- **VS Code `.code-workspace` auto-detection**: if a `.code-workspace` file exists alongside
  `workspace.json`, parse its `folders` array for zero-config multi-root setup.
- **Machine-level cross-repo registry**: separate from per-repo SQLite indexes; holds cross-repo
  FQN → path mappings so a single lookup resolves symbols without loading all repo indexes.

**Core schema changes:**
- `EngineConfig` gains `workspace_roots: Vec<(alias: String, path: PathBuf, role: RepoRole)>`
- `files` table gains `repo_alias TEXT` column; FQNs prefixed: `backend::src/auth.rs::validate`
- `run_pipeline` accepts optional `repos: Vec<String>` to scope or fan-out queries
- CLI: `codesurgeon index --workspace workspace.json`

### Post-Phase-11 — MCP resources: browsable index URI scheme (deferred)

Currently codesurgeon stubs out `resources/list` with an empty array (protocol compliance,
invariant #3). A future resources implementation would make the index machine-readable and
enable client UI integrations beyond what tools expose.

Proposed URI scheme:
| Resource URI | Contents |
|---|---|
| `codesurgeon://metrics` | Live stats snapshot (subscribable for dashboard UIs) |
| `codesurgeon://symbols/{fqn}` | Full symbol record for a given FQN |
| `codesurgeon://files/{path}` | All symbols extracted from a file |
| `codesurgeon://observations` | All session memory entries |
| `codesurgeon://index/status` | Index health snapshot |

Requires: removing the empty-list stub, adding URI routing, designing the resource schema.
Non-trivial architecture change — defer until MCP client tooling (Claude Desktop, Codex) makes
meaningful use of resources beyond what tool calls already provide.

---

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
