# Changelog

All notable changes to codesurgeon are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-10

First stable release. codesurgeon is a local-first dependency graph and session
memory server for AI coding agents. It parses your codebase into a symbol graph,
then serves token-budgeted context capsules via MCP.

### Added

#### Core engine
- Tree-sitter parsing for Python, TypeScript/TSX, JavaScript/JSX, Rust, Swift,
  Shell (bash/zsh), HTML, SQL, and Markdown (.md/.mdx)
- Hybrid candidate retrieval: BM25 full-text search + graph neighbors + ANN
  semantic search, fused with Reciprocal Rank Fusion (RRF)
- Proximity-based reranking with file-distance scoring and role-aware multipliers
- Centrality-boosted ranking using in-degree/out-degree graph analysis
- Token-budgeted context capsules with pivot (full source) and skeleton (signatures only) output
- Call-graph edge extraction for all supported languages including shell/SQL
- File watcher for live incremental re-indexing

#### Embeddings (optional)
- Local semantic embeddings via fastembed (nomic-embed-text-v1.5, 768-dim)
- Apple Silicon Accelerate BLAS support (`--features metal`)
- Memory-mapped embedding store (`embeddings.bin`) — OS pages out unused vectors
- Lazy-load: embedding cache built on first semantic query, not at startup

#### Enrichment passes
- TypeScript compiler shim enrichment (`ts_types = true` in config)
- Pyright Python type enrichment (`python_pyright = true`)
- Rust `cargo-expand` macro enrichment (`rust_expand_macros = true`)
- Rust `rustdoc` JSON resolved-type enrichment (`rust_rustdoc_types = true`)
- Swift/Xcode MCP enrichment with graceful fallback
- LSP edge submission (`submit_lsp_edges` MCP tool) for IDE-resolved types

#### Session memory
- Auto-capture of `run_pipeline` / `get_context_capsule` calls as observations
- Session TTL with observation compression and staleness scoring
- Memory consolidation to deduplicate auto-observations
- Semantic ranking of capsule memories by relevance
- AST-aware change categories for richer observation context
- `search_memory` tool with L1/L2/L3 detail levels

#### MCP server (`codesurgeon-mcp`)
- Full MCP protocol support (JSON-RPC 2.0)
- Dual wire format: Content-Length (Codex) and NDJSON (Claude Code CLI)
- 13 MCP tools: `run_pipeline`, `get_context_capsule`, `get_impact_graph`,
  `get_skeleton`, `search_logic_flow`, `get_diff_capsule`, `get_stats`,
  `index_status`, `get_session_context`, `save_observation`, `search_memory`,
  `submit_lsp_edges`, `generate_module_docs`
- Orphaned process self-termination (parent liveness check)
- PID lock for single-writer, multi-reader concurrency
- Secondary instances serve read-only without loading the embedder

#### CLI (`codesurgeon`)
- Subcommands: `query`, `skeleton`, `impact`, `flow`, `diff`, `stats`,
  `session`, `memory`, `submit-lsp-edges`, `observe`
- Pipe-friendly: `git diff | codesurgeon diff -`

#### Configuration
- `.codesurgeon/config.toml` with sections for `[indexing]`, `[git]`, `[memory]`
- `.codesurgeonignore` for custom file exclusion patterns
- Secrets detection — files containing API key patterns are auto-excluded
- Auto-generated `.codesurgeon/.gitignore`

#### Observability
- `get_stats` MCP tool and `stats` CLI for query log analysis
- `generate_module_docs` for per-directory CLAUDE.md generation
- `manifest.json` tracking with `CS_TRACK_MANIFEST` opt-in

#### Testing
- 57+ tests across all crates
- MCP protocol invariant test suite (11 tests covering wire format, jsonrpc
  field presence, resource stubs, parallel connections, NDJSON round-trip)
- Corrupt SQLite recovery test
- Concurrent query stress tests
- Enrichment integration tests (TS, rustdoc, pyright)

### Fixed

- `jsonrpc: "2.0"` field preserved in all JSON-RPC responses (was accidentally
  dropped during early refactor)
- Content-Length framing for Codex compatibility (bare NDJSON caused connection drops)
- Parallel Codex probes no longer crash the server (secondary instances serve
  read-only instead of exiting)
- Orphaned MCP processes self-terminate after parent dies
- Stale files pruned from index on re-index
- Markdown symbols bypass centrality multiplier to avoid suppressing documentation

## [0.1.0] - 2026-03-15

Initial development release with core indexing, graph construction, and MCP server.
