# Changelog

All notable changes to codesurgeon are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Reverse-edge expansion from exception anchors** (#67). When the task names
  an exception/error/warning type, BFS walks **incoming** edges up to 3 hops
  to surface callers and raisers that BM25 + graph-forward expansion miss.
  Gated by `EngineConfig::reverse_expand_anchors` (default `true`). Per-hop
  fan-out is capped at 5; total candidates at 20. Six regression tests in
  `crates/cs-core/tests/reverse_expand.rs` cover the feature flag, generic-
  anchor suppression, and pivot eligibility. See `docs/ranking.md` §1d.
- **Body-text semantic similarity in reverse-expand** (#69 v2). When the
  `embeddings` feature is active, per-hop caller scoring blends
  `cos(query_embedding, caller_body_embedding)` into the selection beam so
  zero-overlap fix sites (e.g. sympy-21379's `Mod.eval`) surface by topical
  relevance instead of losing the slot to centrality. Weight = 2.0, calibrated
  so one lexical term match still outweighs a moderately related semantic hit.
  Four unit tests in `crates/cs-core/src/ranking.rs` cover the scorer
  branches. See `docs/ranking.md` §1d "Why body-text semantic similarity".
- **Python traceback frame extraction in anchors** (#69 v2, traceback half).
  `anchors::extract` now pulls function/method identifiers out of pasted
  Python tracebacks (`File "...", line N, in <name>`) as a new step 3,
  inserted between imports and prose. Frame-name identifiers bypass the
  snake/camel shape filter that prose tokens require, so plain lowercase
  names (`eval`, `apply`, `frobnicate`) become anchors when they appear in
  a stack frame. Synthetic frames (`<module>`, `<listcomp>`, `<genexpr>`,
  `<lambda>`), stop-words, and names <3 chars are filtered. Dotted frames
  (`Mod.eval`) push both the full chain (flagged `from_dotted_call`) and
  the tail. Six unit tests in `anchors.rs` plus an integration test in
  `tests/engine.rs` (`context_traceback_frame_surfaces_plain_lowercase_function`)
  that proves the engine-layer wiring routes a pasted traceback through
  the new extractor and surfaces the frame's function as a pivot when the
  prose shape filter alone would reject it. Pairs with the semantic
  reverse-expand above to cover both halves of the #69 v2 design — the
  ~40% of Python bug reports that include a traceback (anchor path) and
  the ~60% that don't (semantic reverse-expand path).
- **`run_pipeline` optional `context` parameter** (anchor-extraction v1.7).
  Callers can now pass an additional raw-text blob alongside `task` — typically
  the full problem statement, bug report, or stack trace the task was derived
  from. Anchor extraction runs on `task + context` with dedup by symbol name,
  so identifiers paraphrased out of a compact task string are recovered from
  the raw source. BM25, semantic search, graph retrieval, and intent detection
  still run on `task` alone, so `context` has no effect on query budget or
  intent classification. Backward-compatible: existing callers see identical
  behavior. New `CoreEngine::run_pipeline_with_context` entrypoint; MCP tool
  schema advertises `context` with a persuasive description so real-world
  agents (not just the SWE-bench harness) populate the field. CLI gains
  `codesurgeon context --context @path/to/file|-|<literal>`. Three unit tests
  in `crates/cs-core/tests/engine.rs`. See `docs/explicit-symbol-anchors.md`
  §v1.7 for design notes.
- **SWE-bench harness — per-arm prompt fairness**. `benches/swebench/run.py`
  now branches `PROMPT_PREFIX` by arm via `build_prompt(arm, problem_statement)`.
  The control (`without`) arm no longer receives the
  `mcp__cs-codesurgeon__run_pipeline` nudge, since the tool isn't available
  under `--strict-mcp-config` with an empty `mcpServers` map. Removes a
  long-standing confound from the A/B.
- **`get_impact_graph` response size cap** (#65). Hard-caps the number of
  dependents/dependencies serialized into the response so a query against a
  high-fan-in utility doesn't produce a multi-MB payload that blows the
  client's context window.
- **MCP tool descriptions rewritten for BM25 ranking**. Tool descriptions
  now include bug-fix / refactor / exception-handling keywords the agent's
  internal tool selector scores against, so `run_pipeline` gets picked for
  symptom-anchored tasks instead of a generic search tool.

### Fixed

- **Reverse-expand no longer surfaces `SymbolKind::Import` statements as
  pivots.** Re-export shims (`from err import DeepError`) were winning pivot
  slots because their FQN literally contained the query term. Filter applied
  in both `reverse_expand_from_anchors` and pivot eligibility. Regression
  test in `reverse_expand.rs`.
- **Trivial exception-class stubs excluded from pivot slots.** A 1-line
  `class FooError(Base): pass` has no body to show; before the filter, it
  would beat behaviour-carrying callers on BM25 when the task named the
  exception. Stubs remain available as reverse-expand *seeds*; they just
  can't occupy a pivot slot on their own. Regression tests cover both the
  stub exclusion and the preservation of non-trivial exception classes with
  real methods.
- **Auto-observation feedback loop disabled by default** (`auto_observations`
  config key). Writing every `run_pipeline` call back into the observation
  store poisoned session memory across runs: the consolidator merged query
  pivots into `Consolidated` rows that re-surfaced as hints in future
  capsules, biasing pivot selection toward prior choices regardless of their
  correctness. Opt-in for users who actively curate observations.

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
