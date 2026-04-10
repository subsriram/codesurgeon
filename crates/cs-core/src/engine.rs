use crate::capsule::{build_capsule, format_capsule, Capsule, MemoryEntry, DEFAULT_TOKEN_BUDGET};
use crate::db::Database;
use crate::diff::parse_diff_symbols;
#[cfg(feature = "embeddings")]
use crate::emb_store::EmbeddingStore;
#[cfg(feature = "embeddings")]
use crate::embedder::{cosine_similarity, Embedder};
use crate::graph::CodeGraph;
use crate::indexer::{
    extract_call_edges, extract_impl_edges, extract_import_edges, extract_shell_call_edges,
    extract_sql_ref_edges, extract_type_flow_edges, index_file,
};
use crate::language::Language;
use crate::macro_expand::run_macro_enrichment;
use crate::memory::{new_session_id, IndexingConfig, MemoryConfig, MemoryStore};
use crate::module_docs::{detect_xcode_mcp, swift_enrichment_hint};
use crate::pyright_enrich::run_pyright_enrichment;
use crate::ranking::BM25_POOL_SIZE;
use crate::ranking::{
    apply_structural_resort, dedup_by_fqn, graph_candidates, inject_structural_candidates,
    resolve_adjacents, rrf_merge, select_adjacents, CENTRALITY_BOOST, GRAPH_CANDIDATES,
    MARKDOWN_CENTRALITY_BYPASS, RRF_K, STUB_SCORE_WEIGHT,
};
#[cfg(feature = "embeddings")]
use crate::ranking::{ANN_CANDIDATES, BM25_BLEND_WEIGHT, SEMANTIC_BLEND_WEIGHT};
use crate::rustdoc_enrich::run_rustdoc_enrichment;
use crate::search::{SearchIndex, SearchIntent};
use crate::skeletonizer::skeletonize;
#[cfg(feature = "embeddings")]
use crate::symbol::SymbolKind;
use crate::symbol::{EdgeKind, LspEdge, Symbol};
use crate::ts_enrich::run_ts_enrichment;
use crate::watcher::{hash_content, ChangeKind};
use anyhow::Result;
use ignore::WalkBuilder;
#[cfg(feature = "embeddings")]
use parking_lot::Once;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(feature = "embeddings")]
use std::sync::OnceLock;
use std::time::Instant;

#[cfg(feature = "embeddings")]
fn utf8_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        s
    } else {
        let mut boundary = max_bytes;
        while !s.is_char_boundary(boundary) {
            boundary -= 1;
        }
        &s[..boundary]
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub workspace_root: PathBuf,
    pub db_path: PathBuf,
    pub default_token_budget: u32,
    pub max_pivots: usize,
    pub max_adjacent: usize,
    pub max_blast_radius_depth: u32,
    pub session_id: String,
    /// Whether to load the embedding model on startup.
    /// Set to false for secondary (read-only) instances to avoid loading the
    /// ~500 MB ONNX model when it won't be used for indexing or query embedding.
    pub load_embedder: bool,
    /// When true (default), index type stub files found in the workspace:
    /// `node_modules/@types/**/*.d.ts`, `site-packages/**/*.pyi`, and
    /// `.swiftinterface` files in SPM `.build/` directories.
    /// Gate with `[indexing] include_stubs = false` in `config.toml` to disable.
    pub index_stubs: bool,

    /// When true, run `cargo-expand` on Rust files that contain proc-macro or
    /// derive invocations and index the generated symbols.
    /// Set via `[indexing] rust_expand_macros = true` in `config.toml`.
    /// Default: false.
    pub rust_expand_macros: bool,

    /// When true, run `cargo +nightly doc --output-format json` and merge
    /// resolved types into the symbol index.
    /// Set via `[indexing] rust_rustdoc_types = true` in `config.toml`.
    /// Default: false.
    pub rust_rustdoc_types: bool,

    /// When true, run `pyright --outputjson` and merge resolved type
    /// annotations into existing Python symbols.
    /// Set via `[indexing] python_pyright = true` in `config.toml`.
    /// Default: false.
    pub python_pyright: bool,

    /// When true, invoke the bundled Node.js shim to resolve TypeScript/JavaScript
    /// symbol types via `ts.createProgram()` + `TypeChecker`.
    /// Set via `[indexing] ts_types = true` in `config.toml`.
    /// Default: false.
    pub ts_types: bool,

    /// When true, `manifest.json` is omitted from `.codesurgeon/.gitignore`
    /// so it can be committed and shared.
    /// Set via `CS_TRACK_MANIFEST=1` env var or `[git] track_manifest = true`
    /// in `config.toml`. Default: false.
    pub track_manifest: bool,
}

impl EngineConfig {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        let root: PathBuf = workspace_root.into();
        let db_path = root.join(".codesurgeon").join("index.db");
        let session_id = new_session_id();
        EngineConfig {
            workspace_root: root,
            db_path,
            default_token_budget: DEFAULT_TOKEN_BUDGET,
            max_pivots: 8,
            max_adjacent: 20,
            max_blast_radius_depth: 5,
            session_id,
            load_embedder: true,
            index_stubs: true,
            rust_expand_macros: false,
            rust_rustdoc_types: false,
            python_pyright: false,
            ts_types: false,
            track_manifest: false,
        }
    }

    pub fn without_embedder(mut self) -> Self {
        self.load_embedder = false;
        self
    }
}

// ── Manifest ──────────────────────────────────────────────────────────────────

/// On-disk manifest written to `.codesurgeon/manifest.json` after each full index.
/// Stores per-file blake3 hashes — enables incremental re-indexing and optional
/// git-tracking for shared fast-clone workflows.
#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    workspace: String,
    updated_at: String,
    files: HashMap<String, String>,
}

// ── Output types ──────────────────────────────────────────────────────────────

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub symbol_count: u64,
    pub edge_count: u64,
    pub lsp_edge_count: u64,
    pub file_count: u64,
    pub stub_symbol_count: u64,
    pub session_id: String,
    /// Whether Xcode 26+ MCP bridge (`xcrun mcpbridge`) was detected on this machine.
    /// When true, agents working on Swift files should prefer Xcode MCP for resolved
    /// types and live diagnostics; codesurgeon remains the fallback for semantic search
    /// and session memory.
    pub xcode_mcp_available: bool,
    /// Number of files recorded in the manifest, if present.
    pub manifest_file_count: Option<u64>,
    /// ISO-8601 timestamp from the manifest's `updated_at` field, if present.
    pub manifest_updated_at: Option<String>,
}

/// Return value of `get_session_context`.
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionContext {
    pub observations: Vec<crate::memory::Observation>,
    /// Percentage (0–100) of non-expired observations that are currently stale.
    /// High values indicate that significant code has changed since observations were recorded.
    pub staleness_score: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImpactResult {
    pub target_fqn: String,
    pub direct_dependents: Vec<SymbolRef>,
    pub transitive_dependents: Vec<SymbolRef>,
    pub total_affected: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SymbolRef {
    pub fqn: String,
    pub file_path: String,
    pub start_line: u32,
    pub kind: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SkeletonResult {
    pub file_path: String,
    pub symbols: Vec<SkeletonSymbol>,
    pub token_estimate: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SkeletonSymbol {
    pub fqn: String,
    pub kind: String,
    pub start_line: u32,
    pub skeleton: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FlowResult {
    pub from_fqn: String,
    pub to_fqn: String,
    pub path: Vec<SymbolRef>,
    pub found: bool,
}

// ── ANN index ─────────────────────────────────────────────────────────────────

// ── CoreEngine ────────────────────────────────────────────────────────────────

pub struct CoreEngine {
    config: EngineConfig,
    graph: Arc<RwLock<CodeGraph>>,
    db: Arc<Mutex<Database>>,
    search: Arc<Mutex<SearchIndex>>,
    memory: Arc<Mutex<MemoryStore>>,
    /// Set to true while index_workspace is running so callers can surface a
    /// "not ready" message rather than blocking or returning stale results.
    indexing: Arc<AtomicBool>,
    #[cfg(feature = "embeddings")]
    embedder: Arc<OnceLock<Embedder>>,
    /// Flat embedding store (mmap'd file when available, heap otherwise) — loaded after
    /// each index pass so run_pipeline never needs to hit SQLite for embedding lookups.
    /// OS-managed paging means pages that aren't accessed don't consume RSS.
    #[cfg(feature = "embeddings")]
    embedding_cache: Arc<RwLock<EmbeddingStore>>,
    /// Guards the first-time load of `embedding_cache` from SQLite.
    /// After the initial load, `refresh_embedding_cache` updates the cache directly.
    #[cfg(feature = "embeddings")]
    cache_once: Once,
}

impl CoreEngine {
    pub fn new(config: EngineConfig) -> Result<Self> {
        // Ensure .codesurgeon directory exists
        if let Some(parent) = config.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let db = Arc::new(Mutex::new(Database::open(&config.db_path)?));
        let graph = Arc::new(RwLock::new(CodeGraph::new()));
        let search = Arc::new(Mutex::new(SearchIndex::new()?));

        // Load optional configs from .codesurgeon/config.toml
        let config_path = config
            .workspace_root
            .join(".codesurgeon")
            .join("config.toml");
        let mem_config = MemoryConfig::load_from_toml(&config_path);
        let indexing_config = IndexingConfig::load_from_toml(&config_path);
        // Apply [indexing] / [git] settings onto EngineConfig.
        let mut config = config;
        if indexing_config.rust_expand_macros {
            config.rust_expand_macros = true;
        }
        if indexing_config.rust_rustdoc_types {
            config.rust_rustdoc_types = true;
        }
        if indexing_config.python_pyright {
            config.python_pyright = true;
        }
        if indexing_config.ts_types {
            config.ts_types = true;
        }
        if indexing_config.track_manifest {
            config.track_manifest = true;
        }

        // Write .codesurgeon/.gitignore if absent, excluding index.db always
        // and manifest.json unless track_manifest is enabled.
        let gitignore_path = config
            .workspace_root
            .join(".codesurgeon")
            .join(".gitignore");
        if !gitignore_path.exists() {
            let mut contents = "index.db\nembeddings.bin\n".to_string();
            if !config.track_manifest {
                contents.push_str("manifest.json\n");
            }
            let _ = std::fs::write(&gitignore_path, contents);
        }

        let memory = Arc::new(Mutex::new(
            MemoryStore::new(Arc::clone(&db), &config.session_id).with_config(mem_config),
        ));

        // Warm the in-memory graph and tantivy index from the persisted SQLite DB.
        // Without this, every fresh process starts with 0 pivots on every search.
        {
            let db_guard = db.lock();
            let mut graph_guard = graph.write();
            let mut search_guard = search.lock();

            let symbols = db_guard.all_symbols()?;
            for sym in &symbols {
                graph_guard.add_symbol(sym.clone());
                search_guard.index_symbol(sym)?;
            }

            let edges = db_guard.all_edges()?;
            for edge in &edges {
                graph_guard.add_edge(edge.from_id, edge.to_id, edge.kind.clone());
            }

            // Load LSP-pushed edges, resolving FQNs to IDs via the graph.
            let lsp_edges = db_guard.load_lsp_edges()?;
            let mut lsp_loaded = 0usize;
            for lsp in &lsp_edges {
                if let (Some(from_sym), Some(to_sym)) = (
                    graph_guard.find_by_fqn(&lsp.from_fqn),
                    graph_guard.find_by_fqn(&lsp.to_fqn),
                ) {
                    let from_id = from_sym.id;
                    let to_id = to_sym.id;
                    let kind = lsp_kind_to_edge_kind(&lsp.kind);
                    graph_guard.add_edge(from_id, to_id, kind);
                    lsp_loaded += 1;
                }
            }

            search_guard.commit()?;
            graph_guard.warm_caches();
            tracing::info!(
                "Warmed index: {} symbols, {} edges, {} LSP edges",
                symbols.len(),
                edges.len(),
                lsp_loaded
            );
        }

        // Prune expired observations and run compression pass on startup.
        {
            let mem = memory.lock();
            if let Ok(pruned) = mem.prune_expired() {
                if pruned > 0 {
                    tracing::info!("Pruned {} expired observation(s)", pruned);
                }
            }
            if let Ok(compressed) = mem.compress_observations() {
                if compressed > 0 {
                    tracing::info!("Compressed observations for {} symbol(s)", compressed);
                }
            }
        }

        // Embedder is loaded lazily via `load_embedder()` after the engine is made
        // available to the MCP stdio loop.  This lets BM25+graph queries proceed
        // immediately while the ~130 MB ONNX model loads in the background.
        #[cfg(feature = "embeddings")]
        let embedder = Arc::new(OnceLock::new());

        // Embedding cache and HNSW index are populated lazily on the first semantic query
        // (see `ensure_embedding_cache`). This avoids loading all embeddings into RAM at startup
        // when semantic search is never used (common for small/medium codebases).
        // On first semantic query, `ensure_embedding_cache` loads from embeddings.bin (mmap)
        // or falls back to SQLite.
        #[cfg(feature = "embeddings")]
        let embedding_cache = Arc::new(RwLock::new(EmbeddingStore::from_heap(vec![])));

        Ok(CoreEngine {
            config,
            graph,
            db,
            search,
            memory,
            indexing: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "embeddings")]
            embedder,
            #[cfg(feature = "embeddings")]
            embedding_cache,
            #[cfg(feature = "embeddings")]
            cache_once: Once::new(),
        })
    }

    /// Load the embedding model in the current thread.
    /// Call this from a background thread after the engine is available so queries
    /// can proceed with BM25+graph while the ~130 MB ONNX model downloads/loads.
    /// No-op when the `embeddings` feature is disabled or for read-only instances.
    #[cfg(feature = "embeddings")]
    pub fn load_embedder(&self) {
        if !self.config.load_embedder {
            tracing::info!("Embedder skipped (read-only instance)");
            return;
        }
        match Embedder::new() {
            Ok(e) => {
                let _ = self.embedder.set(e);
                tracing::info!("Embedder loaded (NomicEmbedTextV15Q, 768-dim)");
                match self.consolidate_observations() {
                    Ok(0) => {}
                    Ok(n) => tracing::info!("Consolidated {n} observation cluster(s)"),
                    Err(e) => tracing::warn!("Observation consolidation failed: {}", e),
                }
            }
            Err(e) => {
                tracing::warn!("Embedder unavailable, falling back to BM25: {}", e);
            }
        }
    }

    #[cfg(not(feature = "embeddings"))]
    pub fn load_embedder(&self) {}

    // ── Manifest ──────────────────────────────────────────────────────────────

    fn manifest_path(&self) -> PathBuf {
        self.config
            .workspace_root
            .join(".codesurgeon")
            .join("manifest.json")
    }

    /// Write `.codesurgeon/manifest.json` with the current files-table hashes.
    fn write_manifest(&self) -> Result<()> {
        let file_hashes = self.db.lock().all_file_hashes()?;
        let manifest = Manifest {
            version: 1,
            workspace: self.config.workspace_root.to_string_lossy().to_string(),
            updated_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            files: file_hashes,
        };
        let json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(self.manifest_path(), json)?;
        Ok(())
    }

    /// Read `.codesurgeon/manifest.json`. Returns `None` if absent or unparseable.
    fn read_manifest(&self) -> Option<Manifest> {
        let text = std::fs::read_to_string(self.manifest_path()).ok()?;
        serde_json::from_str(&text).ok()
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    /// Returns true while index_workspace is running.
    pub fn is_indexing(&self) -> bool {
        self.indexing.load(Ordering::Relaxed)
    }

    /// Walk the workspace and index all source files in parallel.
    pub fn index_workspace(&self) -> Result<IndexStats> {
        self.indexing.store(true, Ordering::Relaxed);
        let result = self.index_workspace_inner();
        self.indexing.store(false, Ordering::Relaxed);
        result
    }

    fn index_workspace_inner(&self) -> Result<IndexStats> {
        tracing::info!(
            "Indexing workspace: {}",
            self.config.workspace_root.display()
        );

        let files = self.collect_source_files()?;
        tracing::info!("Found {} source files", files.len());
        let stub_files = self.collect_stub_files();
        if !stub_files.is_empty() {
            tracing::info!("Found {} stub files", stub_files.len());
        }

        // Load baseline hashes for incremental skip:
        // - When DB has data: use the files table (handles re-index after git pull/checkout)
        // - When DB is empty: no baseline — full index required
        let baseline_hashes: HashMap<String, String> = {
            let db = self.db.lock();
            if db.file_count().unwrap_or(0) > 0 {
                db.all_file_hashes().unwrap_or_default()
            } else {
                HashMap::new()
            }
        };

        // ── Prune stale files ─────────────────────────────────────────────────
        // Remove symbols whose file no longer exists on disk (e.g. deleted
        // worktrees, git branch switches, manual deletions).
        {
            let on_disk: std::collections::HashSet<String> = files
                .iter()
                .chain(stub_files.iter())
                .filter_map(|p| {
                    p.strip_prefix(&self.config.workspace_root)
                        .ok()
                        .map(|r| r.to_string_lossy().to_string())
                })
                .collect();

            let db = self.db.lock();
            let tracked = db.all_file_paths().unwrap_or_default();
            let stale: Vec<&String> = tracked
                .iter()
                .filter(|p| !on_disk.contains(p.as_str()))
                .collect();
            if !stale.is_empty() {
                tracing::info!("Pruning {} stale file(s) from index", stale.len());
                let mut search = self.search.lock();
                let mut graph = self.graph.write();
                for rel in &stale {
                    let old_ids = db.symbol_ids_for_file(rel).unwrap_or_default();
                    let _ = db.delete_edges_for_symbols(&old_ids);
                    let _ = db.delete_embeddings_for_symbols(&old_ids);
                    let _ = db.delete_file_symbols(rel);
                    let _ = db.delete_lsp_edges_for_file(rel);
                    let _ = db.delete_file(rel);
                    search.delete_symbols(&old_ids);
                    graph.remove_file(rel);
                }
                let _ = search.commit();
                graph.warm_caches();
            }
        }

        // Parse files in parallel with rayon, skipping files whose hash matches baseline.
        let results: Vec<(PathBuf, String, Vec<Symbol>)> = files
            .par_iter()
            .filter_map(|path| {
                let content = std::fs::read_to_string(path).ok()?;
                // Compute hash first — cheap. Skip parse if file hasn't changed.
                let hash = hash_content(content.as_bytes());
                let rel = path
                    .strip_prefix(&self.config.workspace_root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string();
                if baseline_hashes
                    .get(&rel)
                    .map(|h| h == &hash)
                    .unwrap_or(false)
                {
                    return None; // unchanged — skip
                }
                let symbols = index_file(&self.config.workspace_root, path, &content)
                    .ok()
                    .unwrap_or_default();
                Some((path.clone(), content, symbols))
            })
            .collect();

        let skipped = files.len() - results.len();
        if skipped > 0 {
            tracing::info!(
                "Incremental: skipped {} unchanged file(s), re-indexing {}",
                skipped,
                results.len()
            );
        }

        // Pre-process parsed results into (rel_path, file_hash, symbols) tuples.
        // All of this is lock-free — results is already fully computed.
        let mut file_data: Vec<(String, String, Vec<Symbol>)> = Vec::new();
        let mut all_symbols: Vec<Symbol> = Vec::new();
        for (path, content, symbols) in &results {
            let rel = path
                .strip_prefix(&self.config.workspace_root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            let file_hash = hash_content(content.as_bytes());
            all_symbols.extend(symbols.iter().cloned());
            file_data.push((rel, file_hash, symbols.clone()));
        }

        // Build edges outside any lock — pure CPU work on already-owned data.
        let all_edges: Vec<_> = extract_import_edges(&all_symbols)
            .into_iter()
            .chain(extract_impl_edges(&all_symbols))
            .chain(extract_call_edges(&all_symbols))
            .chain(extract_type_flow_edges(&all_symbols))
            .chain(extract_shell_call_edges(&all_symbols))
            .chain(extract_sql_ref_edges(&all_symbols))
            .collect();

        // Flush everything to SQLite in a single transaction (brief db lock).
        // Batching into one transaction is 10–50x faster than autocommit per-row
        // and keeps the write lock held for a much shorter total duration.
        {
            let db = self.db.lock();
            db.begin_transaction()?;
            for (rel, file_hash, symbols) in &file_data {
                db.upsert_file(rel, file_hash)?;
                db.delete_file_symbols(rel)?;
                for sym in symbols {
                    if let Err(e) = db.mark_stale_by_symbol_hash(&sym.fqn, &sym.content_hash) {
                        tracing::warn!("Stale check error: {}", e);
                    }
                    db.upsert_symbol(sym)?;
                }
            }
            for edge in &all_edges {
                db.upsert_edge(edge)?;
            }
            db.commit_transaction()?;
        } // db lock released here — graph/search locks acquired separately below

        // Update in-memory graph and search index (no db lock held).
        {
            let mut graph = self.graph.write();
            let mut search = self.search.lock();
            for (rel, _, symbols) in &file_data {
                graph.remove_file(rel);
                for sym in symbols {
                    graph.add_symbol(sym.clone());
                    search.index_symbol(sym)?;
                }
            }
            for edge in &all_edges {
                graph.add_edge(edge.from_id, edge.to_id, edge.kind.clone());
            }
            search.commit()?;
            graph.warm_caches();
        } // graph + search locks released here

        // ── Macro expansion enrichment ────────────────────────────────────────
        // Run cargo-expand on Rust files with proc-macro/derive invocations and
        // add the generated symbols to the index.  Skipped when the feature is
        // disabled in config or when cargo-expand is not installed.
        if self.config.rust_expand_macros {
            let expanded_symbols = {
                let db = self.db.lock();
                run_macro_enrichment(&self.config.workspace_root, &file_data, &db)
            };
            if !expanded_symbols.is_empty() {
                tracing::info!(
                    "macro-expand: indexing {} expanded symbol(s)",
                    expanded_symbols.len()
                );
                {
                    let db = self.db.lock();
                    db.begin_transaction()?;
                    for sym in &expanded_symbols {
                        db.upsert_symbol(sym)?;
                    }
                    db.commit_transaction()?;
                }
                {
                    let mut graph = self.graph.write();
                    let mut search = self.search.lock();
                    for sym in &expanded_symbols {
                        graph.add_symbol(sym.clone());
                        search.index_symbol(sym)?;
                    }
                    search.commit()?;
                    graph.warm_caches();
                }
                all_symbols.extend(expanded_symbols);
            }
        }

        // ── rustdoc resolved-type enrichment ──────────────────────────────────
        // Merge resolved return-types and trait-impl lists from
        // `cargo +nightly doc --output-format json` into existing symbols.
        // Runs after the macro-expansion pass so expanded symbols are also
        // eligible for enrichment. Gated on Cargo.lock hash for incremental
        // skipping.
        if self.config.rust_rustdoc_types {
            let enriched_count = {
                let db = self.db.lock();
                run_rustdoc_enrichment(&self.config.workspace_root, &mut all_symbols, &db)
            };
            if enriched_count > 0 {
                tracing::info!(
                    "rustdoc-enrich: resolved types for {} symbol(s)",
                    enriched_count
                );
                // Flush updated resolved_type values back to SQLite.
                let db = self.db.lock();
                db.begin_transaction()?;
                for sym in all_symbols.iter().filter(|s| s.resolved_type.is_some()) {
                    db.upsert_symbol(sym)?;
                }
                db.commit_transaction()?;
            }
        }

        // ── Pyright Python type enrichment ───────────────────────────────────
        // Merge resolved return types and inferred type annotations from
        // `pyright --outputjson` into existing Python symbols.
        // Gated on Python file stats hash for incremental skipping.
        if self.config.python_pyright {
            let enriched_count = {
                let db = self.db.lock();
                run_pyright_enrichment(&self.config.workspace_root, &mut all_symbols, &db)
            };
            if enriched_count > 0 {
                tracing::info!(
                    "pyright-enrich: resolved types for {} symbol(s)",
                    enriched_count
                );
                // Flush updated resolved_type values back to SQLite.
                let db = self.db.lock();
                db.begin_transaction()?;
                for sym in all_symbols
                    .iter()
                    .filter(|s| s.source.as_deref() == Some("pyright"))
                {
                    db.upsert_symbol(sym)?;
                }
                db.commit_transaction()?;
            }
        }

        // ── TypeScript compiler resolved-type enrichment ──────────────────────
        // Invoke the bundled Node.js shim (`ts-enricher.js`) to resolve types
        // via `ts.createProgram()` + `TypeChecker` for TypeScript/JavaScript
        // symbols.  Gated on tsconfig.json hash for incremental skipping.
        if self.config.ts_types {
            let enriched_count = {
                let db = self.db.lock();
                run_ts_enrichment(&self.config.workspace_root, &mut all_symbols, &db)
            };
            if enriched_count > 0 {
                tracing::info!("ts-enrich: resolved types for {} symbol(s)", enriched_count);
                // Flush updated resolved_type values back to SQLite.
                let db = self.db.lock();
                db.begin_transaction()?;
                for sym in all_symbols.iter().filter(|s| s.resolved_type.is_some()) {
                    db.upsert_symbol(sym)?;
                }
                db.commit_transaction()?;
            }
        }

        // ── Stub file indexing ────────────────────────────────────────────────
        // Parse stub files in parallel, mark every symbol with is_stub=true,
        // then flush to SQLite and update the in-memory graph/search.
        // Edges are intentionally skipped for stub symbols — library internals
        // don't need to influence the project dependency graph.
        if !stub_files.is_empty() {
            let stub_results: Vec<(PathBuf, String, Vec<Symbol>)> = stub_files
                .par_iter()
                .filter_map(|path| {
                    let content = std::fs::read_to_string(path).ok()?;
                    let mut symbols = index_file(&self.config.workspace_root, path, &content)
                        .ok()
                        .unwrap_or_default();
                    for sym in &mut symbols {
                        sym.is_stub = true;
                    }
                    Some((path.clone(), content, symbols))
                })
                .collect();

            let mut stub_file_data: Vec<(String, String, Vec<Symbol>)> = Vec::new();
            for (path, content, symbols) in &stub_results {
                let rel = path
                    .strip_prefix(&self.config.workspace_root)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string();
                let file_hash = hash_content(content.as_bytes());
                stub_file_data.push((rel, file_hash, symbols.clone()));
            }

            {
                let db = self.db.lock();
                db.begin_transaction()?;
                for (rel, file_hash, symbols) in &stub_file_data {
                    db.upsert_file(rel, file_hash)?;
                    db.delete_file_symbols(rel)?;
                    for sym in symbols {
                        db.upsert_symbol(sym)?;
                    }
                }
                db.commit_transaction()?;
            }

            {
                let mut graph = self.graph.write();
                let mut search = self.search.lock();
                for (rel, _, symbols) in &stub_file_data {
                    graph.remove_file(rel);
                    for sym in symbols {
                        graph.add_symbol(sym.clone());
                        search.index_symbol(sym)?;
                    }
                }
                search.commit()?;
                graph.warm_caches();
            }

            let stub_sym_count: usize = stub_file_data.iter().map(|(_, _, s)| s.len()).sum();
            tracing::info!(
                "Indexed {} stub symbols from {} files",
                stub_sym_count,
                stub_file_data.len()
            );
        }

        // Embed symbols in batches of 64 (only when embeddings feature is enabled).
        // We embed the skeleton (signature + docstring) rather than the full body —
        // shorter text, lower noise, still captures what the symbol "is".
        // Runs after graph/search locks are released so queries can proceed in parallel.
        #[cfg(feature = "embeddings")]
        if let Some(emb) = self.embedder.get() {
            let skeletons: Vec<String> = all_symbols
                .iter()
                .map(|s| {
                    if s.signature.is_empty() {
                        s.name.clone()
                    } else if s.kind.is_type_definition() || s.kind == SymbolKind::Impl {
                        // For types: include body preview so property/field names are embedded.
                        // This allows semantic queries like "coordinator for documents and lists"
                        // to match a class whose signature is just "class PDFLibrary: ObservableObject"
                        // but whose body declares `@Published var documents`, `var lists`, etc.
                        let body_preview = utf8_truncate(&s.body, 500);
                        format!(
                            "{} {} {}",
                            s.signature,
                            s.docstring.as_deref().unwrap_or(""),
                            body_preview
                        )
                    } else if s.language == Language::Markdown {
                        // For markdown sections, embed the full section body so paragraph content
                        // is semantically searchable, not just the heading text.
                        let body_preview = utf8_truncate(&s.body, 1000);
                        format!("{} {}", s.signature, body_preview)
                    } else {
                        format!("{} {}", s.signature, s.docstring.as_deref().unwrap_or(""))
                    }
                })
                .collect();

            {
                let db = self.db.lock();
                db.begin_transaction()?;
                for (chunk_syms, chunk_texts) in all_symbols.chunks(64).zip(skeletons.chunks(64)) {
                    let refs: Vec<&str> = chunk_texts.iter().map(|s| s.as_str()).collect();
                    match emb.embed_batch(&refs) {
                        Ok(vecs) => {
                            for (sym, vec) in chunk_syms.iter().zip(vecs) {
                                if let Err(e) = db.upsert_embedding(sym.id, &vec) {
                                    tracing::warn!("embedding store error: {}", e);
                                }
                            }
                        }
                        Err(e) => tracing::warn!("embed_batch error: {}", e),
                    }
                }
                db.commit_transaction()?;
            }
            tracing::info!("Embeddings stored for {} symbols", all_symbols.len());
            self.refresh_embedding_cache();
        }

        // Write manifest after a successful index pass.
        if let Err(e) = self.write_manifest() {
            tracing::warn!("Failed to write manifest: {}", e);
        }

        self.index_stats()
    }

    /// Re-index a single file (called by the file watcher on change).
    pub fn reindex_file(&self, path: &Path, kind: ChangeKind) -> Result<()> {
        let rel = path
            .strip_prefix(&self.config.workspace_root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        tracing::debug!("Re-indexing file: {} ({:?})", rel, kind);

        // Phase 1: Snapshot old symbols + remove stale db rows (same db lock).
        let old_symbols = {
            let db = self.db.lock();
            let old_ids = db.symbol_ids_for_file(&rel)?;
            let snap = db.all_symbols_for_file(&rel)?;
            // Clean up edges and embeddings referencing these symbols.
            db.delete_edges_for_symbols(&old_ids)?;
            db.delete_embeddings_for_symbols(&old_ids)?;
            db.delete_file_symbols(&rel)?;
            // LSP edges from this file are now stale; the IDE hook will re-submit after save.
            db.delete_lsp_edges_for_file(&rel)?;
            snap
        };
        // Phase 2: Remove from in-memory graph (brief, independent graph lock).
        {
            self.graph.write().remove_file(&rel);
        }

        if kind == ChangeKind::Removed {
            // Phase 2b: Also purge from Tantivy and the files table.
            let old_ids: Vec<u64> = old_symbols.iter().map(|s| s.id).collect();
            {
                let mut search = self.search.lock();
                search.delete_symbols(&old_ids);
                search.commit()?;
            }
            {
                let db = self.db.lock();
                db.delete_file(&rel)?;
            }
            #[cfg(feature = "embeddings")]
            {
                self.refresh_embedding_cache();
            }
            return Ok(());
        }

        // Phase 3: Parse — no locks held.
        let content = std::fs::read_to_string(path)?;
        let file_hash = hash_content(content.as_bytes());
        let symbols = index_file(&self.config.workspace_root, path, &content)?;

        // Classify the diff between old and new symbol lists.
        let changes = {
            use crate::memory::SymbolChange;
            use crate::symbol::SymbolKind;
            use std::collections::HashMap;

            let old_map: HashMap<&str, &crate::symbol::Symbol> =
                old_symbols.iter().map(|s| (s.fqn.as_str(), s)).collect();
            let new_map: HashMap<&str, &crate::symbol::Symbol> =
                symbols.iter().map(|s| (s.fqn.as_str(), s)).collect();

            let mut changes: Vec<SymbolChange> = Vec::new();

            for sym in &symbols {
                match old_map.get(sym.fqn.as_str()) {
                    None => {
                        let cat = if sym.kind == SymbolKind::Import {
                            "dependency_added"
                        } else {
                            "new_symbol"
                        };
                        changes.push(SymbolChange::new(&sym.fqn, cat));
                    }
                    Some(old) => {
                        if old.signature != sym.signature {
                            changes.push(SymbolChange::new(&sym.fqn, "signature_change"));
                        } else if old.body != sym.body {
                            changes.push(SymbolChange::new(&sym.fqn, "body_change"));
                        }
                    }
                }
            }
            for old in &old_symbols {
                if !new_map.contains_key(old.fqn.as_str()) {
                    changes.push(SymbolChange::new(&old.fqn, "deleted_symbol"));
                }
            }
            changes
        };

        // Phase 4: Write new rows to SQLite in one transaction (brief db lock).
        // db lock is acquired fresh here — no overlap with graph/search locks.
        {
            let db = self.db.lock();
            db.begin_transaction()?;
            db.upsert_file(&rel, &file_hash)?;
            for sym in &symbols {
                db.mark_stale_by_symbol_hash(&sym.fqn, &sym.content_hash)?;
                db.upsert_symbol(sym)?;
            }
            db.commit_transaction()?;
        }

        // Phase 5: Update in-memory graph and search (no db lock held).
        {
            let mut graph = self.graph.write();
            let mut search = self.search.lock();
            for sym in &symbols {
                graph.add_symbol(sym.clone());
                search.index_symbol(sym)?;
            }
            search.commit()?;
            graph.warm_caches();
        }

        // Phase 6: Notify memory of the change (brief, independent memory lock).
        {
            let mut mem = self.memory.lock();
            let _ = mem.record_file_edit(&rel, &changes);
        }

        // Phase 7: Re-embed new symbols and refresh cache (brief db lock, no other locks held).
        #[cfg(feature = "embeddings")]
        if let Some(emb) = self.embedder.get() {
            let skeletons: Vec<String> = symbols
                .iter()
                .map(|s| {
                    if s.signature.is_empty() {
                        s.name.clone()
                    } else if s.kind.is_type_definition() || s.kind == SymbolKind::Impl {
                        let body_preview = utf8_truncate(&s.body, 500);
                        format!(
                            "{} {} {}",
                            s.signature,
                            s.docstring.as_deref().unwrap_or(""),
                            body_preview
                        )
                    } else if s.language == Language::Markdown {
                        let body_preview = utf8_truncate(&s.body, 1000);
                        format!("{} {}", s.signature, body_preview)
                    } else {
                        format!("{} {}", s.signature, s.docstring.as_deref().unwrap_or(""))
                    }
                })
                .collect();
            {
                let db = self.db.lock();
                db.begin_transaction()?;
                for (chunk_syms, chunk_texts) in symbols.chunks(64).zip(skeletons.chunks(64)) {
                    let refs: Vec<&str> = chunk_texts.iter().map(|s| s.as_str()).collect();
                    match emb.embed_batch(&refs) {
                        Ok(vecs) => {
                            for (sym, vec) in chunk_syms.iter().zip(vecs) {
                                if let Err(e) = db.upsert_embedding(sym.id, &vec) {
                                    tracing::warn!("embedding store error: {}", e);
                                }
                            }
                        }
                        Err(e) => tracing::warn!("embed_batch error: {}", e),
                    }
                }
                db.commit_transaction()?;
            }
            self.refresh_embedding_cache();
        }

        Ok(())
    }

    // ── MCP Tool implementations ──────────────────────────────────────────────

    /// Primary tool: auto-detects intent, returns context + impact + memories.
    pub fn run_pipeline(
        &self,
        task: &str,
        budget: Option<u32>,
        language: Option<&str>,
        file_hint: Option<&str>,
    ) -> Result<String> {
        let t0 = Instant::now();
        let budget = budget.unwrap_or(self.config.default_token_budget);
        let intent = SearchIntent::detect(task);

        tracing::debug!("run_pipeline: intent={:?}, task={}", intent, task);

        let capsule =
            self.build_context_capsule(task, budget, &intent, language, file_hint, None, None)?;
        let latency_ms = t0.elapsed().as_millis() as u64;
        let mut out = format_capsule(&capsule);

        // Append Swift enrichment hint when Swift symbols appear in results.
        // Points agents toward Xcode MCP if available, or documents the fallback
        // so they don't assume the tree-sitter results are complete.
        let has_swift = capsule
            .pivots
            .iter()
            .any(|p| p.file_path.ends_with(".swift"))
            || capsule
                .skeletons
                .iter()
                .any(|s| s.file_path.ends_with(".swift"));
        if has_swift {
            out.push_str(&swift_enrichment_hint(detect_xcode_mcp()));
        }

        // Auto-capture this tool call as an observation for cross-session memory.
        let pivot_fqns: Vec<String> = capsule.pivots.iter().map(|p| p.fqn.clone()).collect();
        if let Err(e) = self
            .memory
            .lock()
            .record_auto_observation(task, &pivot_fqns)
        {
            tracing::warn!("auto-observation failed: {}", e);
        }

        // Log query metrics.
        let unique_files: HashSet<&str> = capsule
            .pivots
            .iter()
            .map(|p| p.file_path.as_str())
            .collect();
        let candidate_file_tokens: u64 = {
            let db = self.db.lock();
            unique_files
                .iter()
                .filter_map(|fp| db.all_symbols_for_file(fp).ok())
                .flatten()
                .map(|s| s.token_estimate() as u64)
                .sum()
        };
        let mut langs: Vec<String> = capsule
            .pivots
            .iter()
            .map(|p| {
                std::path::Path::new(&p.file_path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        langs.sort();
        let languages_hit = langs.join(",");
        let timestamp = chrono::Utc::now().to_rfc3339();
        if let Err(e) = self.db.lock().log_query(&crate::db::QueryLogEntry {
            timestamp: &timestamp,
            task,
            intent: intent.as_str(),
            pivot_count: capsule.stats.pivot_count,
            total_tokens: capsule.stats.total_tokens,
            candidate_file_tokens,
            latency_ms,
            languages_hit: &languages_hit,
        }) {
            tracing::warn!("query log failed: {}", e);
        }

        Ok(out)
    }

    /// Get context capsule for a query.
    pub fn get_context_capsule(
        &self,
        query: &str,
        budget: Option<u32>,
        max_results: Option<usize>,
        min_score: Option<f32>,
    ) -> Result<String> {
        let budget = budget.unwrap_or(self.config.default_token_budget);
        let intent = SearchIntent::detect(query);
        let capsule =
            self.build_context_capsule(query, budget, &intent, None, None, max_results, min_score)?;

        // Auto-capture this tool call as an observation for cross-session memory.
        let pivot_fqns: Vec<String> = capsule.pivots.iter().map(|p| p.fqn.clone()).collect();
        if let Err(e) = self
            .memory
            .lock()
            .record_auto_observation(query, &pivot_fqns)
        {
            tracing::warn!("auto-observation failed: {}", e);
        }

        Ok(format_capsule(&capsule))
    }

    /// Return a formatted stats report over the last `days` days of run_pipeline calls.
    pub fn get_stats(&self, days: Option<u32>) -> Result<String> {
        let days = days.unwrap_or(30);
        let db = self.db.lock();
        let rows = db.query_log_rows(days)?;

        if rows.is_empty() {
            return Ok(format!(
                "No run_pipeline calls recorded in the last {} days.",
                days
            ));
        }

        let total_queries = rows.len();
        let total_tokens: u64 = rows.iter().map(|r| r.total_tokens).sum();
        let total_candidate: u64 = rows.iter().map(|r| r.candidate_file_tokens).sum();
        let savings_pct = if total_candidate > 0 {
            (total_candidate.saturating_sub(total_tokens)) as f64 / total_candidate as f64 * 100.0
        } else {
            0.0
        };
        let tokens_saved = total_candidate.saturating_sub(total_tokens);
        // claude-sonnet-4 input pricing: $3 / 1M tokens
        let cost_saved = tokens_saved as f64 * 3.0 / 1_000_000.0;

        // Latency percentiles
        let mut latencies: Vec<u64> = rows.iter().map(|r| r.latency_ms).collect();
        latencies.sort_unstable();
        let median = latencies[latencies.len() / 2];
        let p95 = latencies[(latencies.len() * 95) / 100];

        // Intent breakdown
        let mut intent_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        for row in &rows {
            *intent_counts.entry(row.intent.as_str()).or_insert(0) += 1;
        }
        let mut intent_vec: Vec<(&str, usize)> = intent_counts.into_iter().collect();
        intent_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let intent_line = intent_vec
            .iter()
            .map(|(k, v)| format!("{} {:.0}%", k, *v as f64 / total_queries as f64 * 100.0))
            .collect::<Vec<_>>()
            .join("  ·  ");

        // Language distribution
        let mut lang_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for row in &rows {
            for lang in row.languages_hit.split(',') {
                let lang = lang.trim();
                if !lang.is_empty() {
                    *lang_counts.entry(lang.to_string()).or_insert(0) += 1;
                }
            }
        }
        let lang_total: usize = lang_counts.values().sum();
        let mut lang_vec: Vec<(String, usize)> = lang_counts.into_iter().collect();
        lang_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let lang_line = if lang_total > 0 {
            lang_vec
                .iter()
                .map(|(k, v)| format!("{} {:.0}%", k, *v as f64 / lang_total as f64 * 100.0))
                .collect::<Vec<_>>()
                .join("  ·  ")
        } else {
            String::from("—")
        };

        // Workspace savings (on-demand)
        let workspace_tokens = db.workspace_token_estimate().unwrap_or(0);
        let ws_avg_tokens = if total_queries > 0 {
            total_tokens / total_queries as u64
        } else {
            0
        };
        let ws_savings_pct = if workspace_tokens > 0 {
            (workspace_tokens.saturating_sub(ws_avg_tokens)) as f64 / workspace_tokens as f64
                * 100.0
        } else {
            0.0
        };

        Ok(format!(
            "── Query stats (last {} days) {}\n\
             \x20 Total queries:        {}\n\
             \x20 Token savings:        {:.1}%  (candidate-file baseline)\n\
             \x20 Workspace savings:    {:.1}%  (avg capsule vs full workspace)\n\
             \x20 Estimated cost saved: ${:.2}  (@ claude-sonnet-4 pricing)\n\
             \n\
             ── Latency {}\n\
             \x20 Median: {}ms    p95: {}ms\n\
             \n\
             ── Intent breakdown {}\n\
             \x20 {}\n\
             \n\
             ── Language distribution {}\n\
             \x20 {}\n",
            days,
            "─".repeat(38usize.saturating_sub(format!("last {} days", days).len())),
            total_queries,
            savings_pct,
            ws_savings_pct,
            cost_saved,
            "─".repeat(49),
            median,
            p95,
            "─".repeat(44),
            intent_line,
            "─".repeat(43),
            lang_line,
        ))
    }

    /// Get impact graph: what breaks if `symbol_fqn` changes?
    /// `max_depth` overrides the config blast-radius depth.
    /// `include_tests` (default true) — set false to exclude test files from results.
    pub fn get_impact_graph(
        &self,
        symbol_fqn: &str,
        max_depth: Option<u32>,
        include_tests: bool,
    ) -> Result<ImpactResult> {
        let graph = self.graph.read();

        let target = graph.find_by_fqn(symbol_fqn).ok_or_else(|| {
            // Anti-hallucination: suggest similar FQNs when exact match fails
            let suggestions = graph.fuzzy_fqn_matches(symbol_fqn, 5);
            if suggestions.is_empty() {
                anyhow::anyhow!("Symbol not found: `{}`", symbol_fqn)
            } else {
                let list = suggestions
                    .iter()
                    .map(|s| format!("  - {}", s.fqn))
                    .collect::<Vec<_>>()
                    .join("\n");
                anyhow::anyhow!(
                    "Symbol not found: `{}`\n\nDid you mean one of these?\n{}",
                    symbol_fqn,
                    list
                )
            }
        })?;

        let target_id = target.id;
        let depth = max_depth.unwrap_or(self.config.max_blast_radius_depth);

        let is_test = |s: &SymbolRef| -> bool {
            let p = &s.file_path;
            p.contains("/test")
                || p.contains("_test.")
                || p.contains("test_")
                || p.contains("/spec")
                || p.contains("_spec.")
        };

        let direct: Vec<SymbolRef> = graph
            .dependents(target_id)
            .into_iter()
            .map(sym_ref)
            .filter(|s| include_tests || !is_test(s))
            .collect();

        let transitive: Vec<SymbolRef> = graph
            .blast_radius(target_id, depth)
            .into_iter()
            .map(sym_ref)
            .filter(|s| include_tests || !is_test(s))
            .collect();

        let total = direct.len() + transitive.len();

        Ok(ImpactResult {
            target_fqn: symbol_fqn.to_string(),
            direct_dependents: direct,
            transitive_dependents: transitive,
            total_affected: total,
        })
    }

    /// Get skeleton of a file: all signatures without bodies.
    /// `max_depth` limits nesting depth: 1 = top-level only, 2 = top-level + methods, etc.
    /// Depth is measured by counting `::` separators after the file prefix in the FQN.
    pub fn get_skeleton(&self, file_path: &str, max_depth: Option<u32>) -> Result<SkeletonResult> {
        let graph = self.graph.read();
        let symbols = graph.file_symbols(file_path);

        let mut total_tokens = 0u32;
        let skeleton_syms: Vec<SkeletonSymbol> = symbols
            .iter()
            .filter(|sym| {
                if let Some(depth) = max_depth {
                    let sym_depth = sym
                        .fqn
                        .split_once("::")
                        .map(|(_, rest)| rest.matches("::").count() as u32 + 1)
                        .unwrap_or(1);
                    sym_depth <= depth
                } else {
                    true
                }
            })
            .map(|sym| {
                let skel = skeletonize(sym);
                let tokens = (skel.len() / 4) as u32;
                total_tokens += tokens;
                SkeletonSymbol {
                    fqn: sym.fqn.clone(),
                    kind: sym.kind.to_string(),
                    start_line: sym.start_line,
                    skeleton: skel,
                }
            })
            .collect();

        Ok(SkeletonResult {
            file_path: file_path.to_string(),
            symbols: skeleton_syms,
            token_estimate: total_tokens,
        })
    }

    /// Trace execution path between two symbols.
    pub fn search_logic_flow(&self, from_fqn: &str, to_fqn: &str) -> Result<FlowResult> {
        let graph = self.graph.read();

        let from_sym = graph.find_by_fqn(from_fqn);
        let to_sym = graph.find_by_fqn(to_fqn);

        let (from_id, to_id) = match (from_sym, to_sym) {
            (Some(f), Some(t)) => (f.id, t.id),
            _ => {
                return Ok(FlowResult {
                    from_fqn: from_fqn.to_string(),
                    to_fqn: to_fqn.to_string(),
                    path: vec![],
                    found: false,
                });
            }
        };

        let path_ids = graph.find_path(from_id, to_id);
        let found = !path_ids.is_empty();

        let path: Vec<SymbolRef> = path_ids
            .iter()
            .filter_map(|&id| graph.get_symbol(id))
            .map(sym_ref)
            .collect();

        Ok(FlowResult {
            from_fqn: from_fqn.to_string(),
            to_fqn: to_fqn.to_string(),
            path,
            found,
        })
    }

    /// Accept LSP-resolved edges pushed by an IDE extension or Claude Code hook.
    ///
    /// Each edge is stored in the `lsp_edges` table and immediately reflected in the
    /// in-memory graph. Edges whose `from_fqn` refers to an unknown symbol are skipped
    /// (the IDE may have sent them before the index was fully built; it can re-submit).
    /// Edges are invalidated automatically when the source file is re-indexed.
    pub fn submit_lsp_edges(&self, edges: &[LspEdge]) -> Result<String> {
        let mut accepted = 0usize;
        let mut skipped = 0usize;

        let db = self.db.lock();
        let mut graph = self.graph.write();

        for lsp in edges {
            // Derive source_file from the from_fqn (everything before the first "::").
            let source_file = lsp.from_fqn.split("::").next().unwrap_or(&lsp.from_fqn);

            // Resolve both FQNs to symbol IDs in the current graph.
            let from_id = graph.find_by_fqn(&lsp.from_fqn).map(|s| s.id);
            let to_id = graph.find_by_fqn(&lsp.to_fqn).map(|s| s.id);

            match (from_id, to_id) {
                (Some(fid), Some(tid)) => {
                    db.upsert_lsp_edge(
                        &lsp.from_fqn,
                        &lsp.to_fqn,
                        &lsp.kind,
                        lsp.resolved_type.as_deref(),
                        source_file,
                    )?;
                    graph.add_edge(fid, tid, lsp_kind_to_edge_kind(&lsp.kind));
                    accepted += 1;
                }
                _ => {
                    skipped += 1;
                }
            }
        }

        if accepted > 0 {
            graph.warm_caches();
        }

        Ok(format!(
            "{} edge(s) accepted, {} skipped (unknown symbol FQN).",
            accepted, skipped
        ))
    }

    /// Number of vectors currently in the in-memory embedding cache.
    /// Returns 0 until the first semantic query triggers lazy initialisation.
    /// Exposed for testing; not part of the stable public API.
    #[cfg(feature = "embeddings")]
    pub fn embedding_cache_len(&self) -> usize {
        self.embedding_cache.read().len()
    }

    /// Index statistics and health.
    pub fn index_stats(&self) -> Result<IndexStats> {
        let db = self.db.lock();
        let manifest = self.read_manifest();
        Ok(IndexStats {
            symbol_count: db.symbol_count()?,
            edge_count: db.edge_count()?,
            lsp_edge_count: db.lsp_edge_count()?,
            file_count: db.file_count()?,
            stub_symbol_count: db.stub_symbol_count()?,
            session_id: self.config.session_id.clone(),
            xcode_mcp_available: detect_xcode_mcp(),
            manifest_file_count: manifest.as_ref().map(|m| m.files.len() as u64),
            manifest_updated_at: manifest.map(|m| m.updated_at),
        })
    }

    /// Get session observations (cross-session memory) with a staleness score.
    pub fn get_session_context(&self) -> Result<SessionContext> {
        let mem = self.memory.lock();
        let observations = mem.get_recent_observations(50)?;
        let staleness_score = mem.staleness_score()?;
        Ok(SessionContext {
            observations,
            staleness_score,
        })
    }

    /// Look up a symbol by FQN and return its signature + first 20 lines of body.
    /// Returns `None` if the symbol is not in the graph.
    pub fn get_symbol_snippet(&self, fqn: &str) -> Option<(String, String)> {
        let graph = self.graph.read();
        graph.find_by_fqn(fqn).map(|sym| {
            let body_preview = sym.body.lines().take(20).collect::<Vec<_>>().join("\n  ");
            (sym.signature.clone(), body_preview)
        })
    }

    /// Keyword search over saved observations. Returns up to `max_results` matches
    /// ranked by term overlap then recency.
    pub fn search_memory(
        &self,
        query: &str,
        max_results: Option<usize>,
    ) -> Result<Vec<crate::memory::Observation>> {
        let limit = max_results.unwrap_or(10);
        self.memory.lock().search_observations(query, limit)
    }

    /// Delete an observation by ID. Returns true if found and deleted.
    pub fn delete_observation(&self, id: &str) -> Result<bool> {
        self.memory.lock().delete(id)
    }

    /// Save a manual observation.
    pub fn save_observation(&self, content: &str, symbol_fqn: Option<&str>) -> Result<()> {
        let graph = self.graph.read();

        // Resolve symbol hash if an FQN was provided
        let symbol_hash = symbol_fqn
            .and_then(|fqn| graph.find_by_fqn(fqn))
            .map(|sym| sym.content_hash.clone());

        let mem = self.memory.lock();
        mem.save(content, symbol_fqn, symbol_hash.as_deref())
    }

    /// Semantic consolidation pass: clusters auto/passive observations by embedding cosine
    /// similarity (threshold 0.92) and merges each cluster of ≥ 2 into a single
    /// `Consolidated` entry whose content captures the unique queries and pivot FQNs.
    /// No-op when the embedder is not loaded.
    /// Returns the number of clusters merged.
    #[cfg(feature = "embeddings")]
    pub fn consolidate_observations(&self) -> Result<usize> {
        use crate::embedder::cosine_similarity;
        const SIMILARITY_THRESHOLD: f32 = 0.92;
        const MIN_CLUSTER_SIZE: usize = 2;

        let Some(emb) = self.embedder.get() else {
            return Ok(0);
        };

        let candidates = self.db.lock().get_consolidation_candidates()?;
        if candidates.len() < MIN_CLUSTER_SIZE {
            return Ok(0);
        }

        let texts: Vec<&str> = candidates.iter().map(|o| o.content.as_str()).collect();
        let embeddings = match emb.embed_batch(&texts) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("consolidation embed failed: {}", e);
                return Ok(0);
            }
        };

        // Greedy clustering: for each unvisited observation, start a new cluster and pull
        // in all remaining unvisited observations whose similarity exceeds the threshold.
        let n = candidates.len();
        let mut assigned = vec![false; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for i in 0..n {
            if assigned[i] {
                continue;
            }
            let mut cluster = vec![i];
            assigned[i] = true;
            for j in (i + 1)..n {
                if !assigned[j]
                    && cosine_similarity(&embeddings[i], &embeddings[j]) >= SIMILARITY_THRESHOLD
                {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }
            if cluster.len() >= MIN_CLUSTER_SIZE {
                clusters.push(cluster);
            }
        }

        let mem = self.memory.lock();
        let mem_config = mem.config().clone();
        let session_id = &self.config.session_id;

        for cluster in &clusters {
            let cluster_obs: Vec<&crate::memory::Observation> =
                cluster.iter().map(|&i| &candidates[i]).collect();

            let merged = merge_cluster_content(&cluster_obs);
            let consolidated = crate::memory::Observation::new(
                session_id,
                &merged,
                None,
                None,
                crate::memory::ObservationKind::Consolidated,
            )
            .with_ttl(&mem_config);

            let ids_to_expire: Vec<String> = cluster_obs.iter().map(|o| o.id.clone()).collect();
            let db = self.db.lock();
            db.insert_observation(&consolidated)?;
            db.expire_observations(&ids_to_expire)?;
        }

        Ok(clusters.len())
    }

    #[cfg(not(feature = "embeddings"))]
    pub fn consolidate_observations(&self) -> Result<usize> {
        Ok(0)
    }

    /// Fetch observations ranked by semantic relevance to `query`.
    ///
    /// Pulls a pool of `limit * 3` recent observations from the memory store, embeds them
    /// alongside the query, and returns up to `limit` entries sorted by descending cosine
    /// similarity. Entries scoring below [`OBSERVATION_MIN_SIMILARITY`] are excluded entirely
    /// so irrelevant observations never consume capsule budget.
    ///
    /// Falls back to plain recency order when the embedder is unavailable (e.g. the process
    /// was started without the `embeddings` feature, or the model hasn't been loaded yet).
    #[cfg(feature = "embeddings")]
    fn ranked_observations(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        // Fetch a larger pool so we can select the most relevant subset.
        let pool_size = (limit * 3).max(30);
        let raw = self.memory.lock().get_recent_observations(pool_size)?;
        if raw.is_empty() {
            return Ok(Vec::new());
        }

        let ranked_indices = if let Some(emb) = self.embedder.get() {
            if let Ok(query_vec) = emb.embed_one(query) {
                let texts: Vec<&str> = raw.iter().map(|o| o.content.as_str()).collect();
                if let Ok(obs_vecs) = emb.embed_batch(&texts) {
                    Some(rank_by_similarity(&query_vec, &obs_vecs, limit))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let entries: Vec<MemoryEntry> = if let Some(indices) = ranked_indices {
            indices
                .into_iter()
                .map(|i| &raw[i])
                .map(|obs| MemoryEntry {
                    content: obs.content.clone(),
                    symbol_fqn: obs.symbol_fqn.clone(),
                    is_stale: obs.is_stale,
                    created_at: obs.created_at.clone(),
                })
                .collect()
        } else {
            // Fallback: recency order
            raw.into_iter()
                .take(limit)
                .map(|obs| MemoryEntry {
                    content: obs.content,
                    symbol_fqn: obs.symbol_fqn,
                    is_stale: obs.is_stale,
                    created_at: obs.created_at,
                })
                .collect()
        };

        Ok(entries)
    }

    #[cfg(not(feature = "embeddings"))]
    fn ranked_observations(&self, _query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let raw = self.memory.lock().get_recent_observations(limit)?;
        Ok(raw
            .into_iter()
            .map(|obs| MemoryEntry {
                content: obs.content,
                symbol_fqn: obs.symbol_fqn,
                is_stale: obs.is_stale,
                created_at: obs.created_at,
            })
            .collect())
    }

    /// Diff-aware capsule: parse a git diff and return context for changed symbols.
    /// Identifies changed functions/methods, their callers, and any related test files.
    pub fn get_diff_capsule(&self, diff: &str, budget: Option<u32>) -> Result<String> {
        let budget = budget.unwrap_or(self.config.default_token_budget);
        let changed = parse_diff_symbols(diff);

        if changed.is_empty() {
            return Ok("No changed symbols detected in diff.".to_string());
        }

        let graph = self.graph.read();

        // Resolve changed symbol names/ranges → Symbol IDs
        let mut pivot_ids: Vec<u64> = Vec::new();
        let mut adjacent_ids: Vec<u64> = Vec::new();

        for (file, start, end) in &changed {
            let syms = graph.symbols_in_range(file, *start, *end);
            for sym in syms {
                if !pivot_ids.contains(&sym.id) {
                    pivot_ids.push(sym.id);
                }
                // Also include direct dependents (callers that will be affected)
                for dep in graph.dependents(sym.id) {
                    if !pivot_ids.contains(&dep.id) && !adjacent_ids.contains(&dep.id) {
                        adjacent_ids.push(dep.id);
                    }
                }
            }
        }

        // Surface test files referencing the changed symbols
        let test_ids: Vec<u64> = graph
            .all_symbols()
            .filter(|s| {
                let p = s.file_path.to_lowercase();
                (p.contains("test") || p.contains("spec")) && !pivot_ids.contains(&s.id)
            })
            .filter(|s| {
                // Test file references a changed symbol by name
                changed.iter().any(|(_, _, _)| {
                    pivot_ids.iter().any(|&id| {
                        graph
                            .get_symbol(id)
                            .map(|sym| s.body.contains(&sym.name))
                            .unwrap_or(false)
                    })
                })
            })
            .map(|s| s.id)
            .take(5)
            .collect();

        adjacent_ids.extend(test_ids);
        adjacent_ids.dedup();

        let pivot_syms: Vec<&Symbol> = pivot_ids
            .iter()
            .filter_map(|id| graph.get_symbol(*id))
            .collect();
        let adjacent_syms: Vec<&Symbol> = adjacent_ids
            .iter()
            .filter_map(|id| graph.get_symbol(*id))
            .collect();

        // Build a synthetic query from changed symbol names for memory ranking.
        let diff_query: String = pivot_syms
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let memory_entries = self.ranked_observations(&diff_query, 10)?;

        let capsule = build_capsule(pivot_syms, adjacent_syms, memory_entries, budget, None);
        let mut out = format!(
            "## Diff context capsule\n> {} changed symbol(s) detected\n\n",
            pivot_ids.len()
        );
        out.push_str(&format_capsule(&capsule));
        Ok(out)
    }

    /// Auto-generate per-directory CLAUDE.md summaries from the symbol graph.
    /// Returns the generated markdown. If `write_files` is true, also writes
    /// CLAUDE.md files into each directory.
    pub fn generate_module_docs(&self, write_files: bool) -> Result<String> {
        let graph = self.graph.read();

        // Group non-import symbols by directory
        let mut by_dir: std::collections::BTreeMap<String, Vec<&Symbol>> =
            std::collections::BTreeMap::new();

        for sym in graph.all_symbols() {
            if sym.kind == crate::symbol::SymbolKind::Import {
                continue;
            }
            let dir = std::path::Path::new(&sym.file_path)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|| ".".to_string());
            by_dir.entry(dir).or_default().push(sym);
        }

        // Check once whether this workspace has any Swift files at all.
        let workspace_has_swift = graph.all_symbols().any(|s| s.language == Language::Swift);
        let xcode_available = if workspace_has_swift {
            detect_xcode_mcp()
        } else {
            false
        };

        let mut all_docs = String::new();

        // Prepend a workspace-level Swift note to the combined output so agents reading
        // the full generate_module_docs result see it regardless of which directory they
        // navigate to first.
        if workspace_has_swift {
            all_docs.push_str("## Swift enrichment\n\n");
            all_docs.push_str(swift_enrichment_hint(xcode_available).trim_start_matches('\n'));
            all_docs.push_str("\n\n---\n\n");
        }

        for (dir, symbols) in &by_dir {
            if symbols.len() < 3 {
                continue; // Skip tiny directories
            }

            // Check whether this specific directory contains Swift files.
            let dir_has_swift = symbols.iter().any(|s| s.language == Language::Swift);

            // Group by kind for the summary
            let mut fns: Vec<&&Symbol> = symbols.iter().filter(|s| s.kind.is_callable()).collect();
            let mut types: Vec<&&Symbol> = symbols
                .iter()
                .filter(|s| s.kind.is_type_definition())
                .collect();

            fns.sort_by_key(|s| &s.name);
            types.sort_by_key(|s| &s.name);

            let mut doc = format!(
                "# {}\n\n",
                if dir.is_empty() || dir == "." {
                    "root"
                } else {
                    dir
                }
            );
            doc.push_str(&format!(
                "> {} symbols ({} functions/methods, {} types)\n\n",
                symbols.len(),
                fns.len(),
                types.len()
            ));

            if !types.is_empty() {
                doc.push_str("## Types\n\n");
                for sym in &types {
                    let doc_line = sym
                        .docstring
                        .as_deref()
                        .map(|d| format!(" — {}", d.lines().next().unwrap_or("").trim()))
                        .unwrap_or_default();
                    doc.push_str(&format!(
                        "- **`{}`** (`{}`){}\n",
                        sym.name, sym.kind, doc_line
                    ));
                }
                doc.push('\n');
            }

            if !fns.is_empty() {
                doc.push_str("## Functions / Methods\n\n");
                for sym in &fns {
                    let doc_line = sym
                        .docstring
                        .as_deref()
                        .map(|d| format!(" — {}", d.lines().next().unwrap_or("").trim()))
                        .unwrap_or_default();
                    doc.push_str(&format!(
                        "- **`{}`** @ `{}:{}`{}\n",
                        sym.name, sym.file_path, sym.start_line, doc_line
                    ));
                }
                doc.push('\n');
            }

            // Append Swift enrichment note to per-directory docs that contain Swift files.
            // This is the primary channel for session-start context in other projects —
            // agents read their project's CLAUDE.md before any tool calls, so the failover
            // instructions need to live here, not just in run_pipeline hints.
            if dir_has_swift {
                doc.push_str("## Swift enrichment\n\n");
                doc.push_str(swift_enrichment_hint(xcode_available).trim_start_matches('\n'));
                doc.push('\n');
            }

            if write_files {
                let claude_md_path = self.config.workspace_root.join(dir).join("CLAUDE.md");
                if let Some(parent) = claude_md_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&claude_md_path, &doc)?;
                tracing::info!("Wrote {}", claude_md_path.display());
            }

            all_docs.push_str(&doc);
            all_docs.push_str("---\n\n");
        }

        if all_docs.is_empty() {
            return Ok("No modules with enough symbols to document.".to_string());
        }

        Ok(all_docs)
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    /// Stage 1.5: parallel flat cosine scan over the embedding cache for the top-K nearest symbols.
    /// Returns `(symbol_id, cosine_similarity)` pairs, sorted descending by similarity.
    ///
    /// A rayon parallel scan handles up to ~1M vectors in <100ms, which covers any local
    /// codebase. Results are exact (not approximate), so recall is always 100%.
    #[cfg(feature = "embeddings")]
    fn ann_candidates(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        use rayon::prelude::*;
        self.ensure_embedding_cache();
        let Some(emb) = self.embedder.get() else {
            return vec![];
        };
        let query_vec = match emb.embed_one(query) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("ANN query embed failed: {}", e);
                return vec![];
            }
        };
        let cache = self.embedding_cache.read();
        if cache.is_empty() {
            return vec![];
        }
        let pairs: Vec<(u64, &[f32])> = cache.iter().collect();
        let mut results: Vec<(u64, f32)> = pairs
            .par_iter()
            .map(|(id, v)| (*id, cosine_similarity(&query_vec, v)))
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Ensure the embedding cache has been loaded at least once.
    /// Uses `cache_once` so the load happens exactly once across all concurrent callers;
    /// subsequent calls are a cheap atomic check. Tries the mmap'd embeddings.bin
    /// before falling back to SQLite.
    /// `refresh_embedding_cache` bypasses this and writes directly — it also marks
    /// `cache_once` as done to suppress any future redundant lazy loads.
    #[cfg(feature = "embeddings")]
    fn ensure_embedding_cache(&self) {
        let db = &self.db;
        let embedding_cache = &self.embedding_cache;
        let emb_path = self
            .config
            .workspace_root
            .join(".codesurgeon")
            .join("embeddings.bin");
        self.cache_once.call_once(|| {
            // Try mmap'd file first (O(1), OS-managed paging); fall back to SQLite.
            let store = if let Some(s) = EmbeddingStore::open(&emb_path) {
                tracing::info!("Lazy-loaded {} embeddings from mmap file", s.len());
                s
            } else {
                match db.lock().all_embeddings() {
                    Ok(embs) => EmbeddingStore::from_heap(embs),
                    Err(e) => {
                        tracing::warn!("lazy embedding cache load failed: {}", e);
                        return;
                    }
                }
            };
            if store.is_empty() {
                return;
            }
            tracing::info!("Embedding cache ready ({} vectors)", store.len());
            *embedding_cache.write() = store;
        });
    }

    /// Reload all embeddings from SQLite, write to embeddings.bin (mmap), and swap in the
    /// new store. Called after every index pass. Marks `cache_once` as done so the next
    /// query skips the lazy-init path.
    #[cfg(feature = "embeddings")]
    fn refresh_embedding_cache(&self) {
        let emb_path = self
            .config
            .workspace_root
            .join(".codesurgeon")
            .join("embeddings.bin");

        match self.db.lock().all_embeddings() {
            Ok(embs) => {
                let store_count = embs.len();
                if store_count == 0 {
                    return;
                }
                // Write the flat binary file and swap in the mmap store.
                let new_store = match EmbeddingStore::write_and_open(&emb_path, &embs) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!("Failed to write embeddings.bin: {}", e);
                        EmbeddingStore::from_heap(embs)
                    }
                };
                *self.embedding_cache.write() = new_store;
                // Mark as initialized so ensure_embedding_cache is a no-op hereafter.
                self.cache_once.call_once(|| {});
                tracing::info!("Embedding cache refreshed ({} vectors)", store_count);
            }
            Err(e) => tracing::warn!("Failed to refresh embedding cache: {}", e),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_context_capsule(
        &self,
        query: &str,
        budget: u32,
        intent: &SearchIntent,
        language: Option<&str>,
        file_hint: Option<&str>,
        max_results: Option<usize>,
        min_score: Option<f32>,
    ) -> Result<Capsule> {
        // ── Stage 1: Candidate Retrieval (BM25 + graph neighbors + ANN) ──────────
        let bm25_results = self.search.lock().search(query, BM25_POOL_SIZE)?;

        // Track original BM25 IDs (used for coordinator bonus in structural re-sort).
        let bm25_ids: std::collections::HashSet<u64> =
            bm25_results.iter().map(|(id, _)| *id).collect();

        // Graph neighbor expansion: 1-hop neighbors of BM25 seeds, ranked by centrality.
        let graph_results = {
            let graph = self.graph.read();
            graph_candidates(&graph, &bm25_ids, GRAPH_CANDIDATES)
        };

        // ANN semantic retrieval + RRF fusion across all three sources.
        #[cfg(feature = "embeddings")]
        let mut search_results = {
            let ann_results = self.ann_candidates(query, ANN_CANDIDATES);
            rrf_merge(&[&bm25_results, &graph_results, &ann_results], RRF_K)
        };
        #[cfg(not(feature = "embeddings"))]
        let mut search_results = rrf_merge(&[&bm25_results, &graph_results], RRF_K);

        let graph = self.graph.read();

        // 2. Inject high-centrality types for Structural queries (BM25 can't surface them)
        if *intent == SearchIntent::Structural {
            inject_structural_candidates(&graph, &mut search_results, self.config.max_pivots);
        }

        // 3. Re-rank by query proximity + centrality + optional semantic similarity
        let symbols_for_rerank: Vec<&Symbol> = search_results
            .iter()
            .filter_map(|(id, _)| graph.get_symbol(*id))
            .collect();
        let reranked = SearchIndex::rerank_by_query_proximity(
            search_results,
            &symbols_for_rerank,
            query,
            intent,
        );

        let mut scored = self.apply_centrality_and_semantics(&graph, reranked, query);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 4. For Structural intent: re-sort by in-degree + coordinator bonus
        let scored = if *intent == SearchIntent::Structural {
            apply_structural_resort(&graph, scored, &bm25_ids, query)
        } else {
            scored
        };

        // 5. Deduplicate by FQN — keep the highest-scored entry per unique FQN.
        let mut scored = dedup_by_fqn(&graph, scored);

        // 5.5a Apply stub score penalty: library stubs rank at ×0.3 relative to project symbols.
        // Re-sort after applying penalty so pivots are selected from the adjusted order.
        let has_stubs = scored
            .iter()
            .any(|(id, _)| graph.get_symbol(*id).map(|s| s.is_stub).unwrap_or(false));
        if has_stubs {
            for (id, score) in scored.iter_mut() {
                if graph.get_symbol(*id).map(|s| s.is_stub).unwrap_or(false) {
                    *score *= STUB_SCORE_WEIGHT;
                }
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // 5.5 Apply optional post-filters (language, file_hint, min_score)
        if language.is_some() || file_hint.is_some() || min_score.is_some() {
            scored.retain(|(id, score)| {
                if let Some(min) = min_score {
                    if *score < min {
                        return false;
                    }
                }
                if let Some(sym) = graph.get_symbol(*id) {
                    if let Some(lang) = language {
                        if !sym.language.as_str().eq_ignore_ascii_case(lang) {
                            return false;
                        }
                    }
                    if let Some(hint) = file_hint {
                        if !sym.file_path.contains(hint) {
                            return false;
                        }
                    }
                }
                true
            });
        }
        let max_pivots = max_results.unwrap_or(self.config.max_pivots);

        // 6. Select pivots and adjacents
        // Stubs are excluded from pivots — they are skeleton-only references.
        let pivot_ids: Vec<u64> = scored
            .iter()
            .filter(|(id, _)| !graph.get_symbol(*id).map(|s| s.is_stub).unwrap_or(false))
            .take(max_pivots)
            .map(|(id, _)| *id)
            .collect();
        let adjacent_ids = select_adjacents(&graph, &pivot_ids, intent, self.config.max_adjacent);

        // 7. Resolve IDs → Symbols with filtering
        let filter_adjacents = matches!(intent, SearchIntent::Structural | SearchIntent::Explore);
        let pivot_syms: Vec<&Symbol> = pivot_ids
            .iter()
            .filter_map(|id| graph.get_symbol(*id))
            .collect();
        let adjacent_syms = resolve_adjacents(&graph, &adjacent_ids, filter_adjacents);

        // 8. Fetch semantically relevant memories and assemble capsule
        let memory_entries = self.ranked_observations(query, 20)?;

        Ok(build_capsule(
            pivot_syms,
            adjacent_syms,
            memory_entries,
            budget,
            Some(query),
        ))
    }

    /// Apply centrality boost and optionally blend semantic similarity scores.
    /// Final = BM25_centrality * 0.5 + semantic_cosine * 0.5 (when embedder present).
    fn apply_centrality_and_semantics(
        &self,
        graph: &CodeGraph,
        reranked: Vec<(u64, f32)>,
        query: &str,
    ) -> Vec<(u64, f32)> {
        #[cfg(feature = "embeddings")]
        self.ensure_embedding_cache();
        #[cfg(feature = "embeddings")]
        let semantic_scores: std::collections::HashMap<u64, f32> =
            if let Some(emb) = self.embedder.get() {
                match emb.embed_one(query) {
                    Ok(query_vec) => {
                        let candidate_ids: std::collections::HashSet<u64> =
                            reranked.iter().map(|(id, _)| *id).collect();
                        let cache = self.embedding_cache.read();
                        cache
                            .iter()
                            .filter(|(id, _)| candidate_ids.contains(id))
                            .map(|(id, vec)| (id, cosine_similarity(&query_vec, vec)))
                            .collect()
                    }
                    Err(e) => {
                        tracing::warn!("query embed failed: {}", e);
                        std::collections::HashMap::new()
                    }
                }
            } else {
                std::collections::HashMap::new()
            };
        // Suppress unused variable warning when embeddings feature is off.
        #[cfg(not(feature = "embeddings"))]
        let _ = query;

        reranked
            .into_iter()
            .map(|(id, score)| {
                let is_markdown = graph
                    .get_symbol(id)
                    .map(|s| s.language == Language::Markdown)
                    .unwrap_or(false);
                // Markdown symbols have no graph edges so centrality is always 0.
                // Apply a fixed documentation boost instead of the centrality multiplier.
                let centrality = graph.centrality_score(id);
                let bm25_score = if is_markdown {
                    score * MARKDOWN_CENTRALITY_BYPASS
                } else {
                    score * (1.0 + centrality * CENTRALITY_BOOST)
                };
                #[cfg(feature = "embeddings")]
                let final_score = {
                    let sem = semantic_scores.get(&id).copied().unwrap_or(0.0);
                    if sem > 0.0 {
                        bm25_score * BM25_BLEND_WEIGHT + sem * SEMANTIC_BLEND_WEIGHT
                    } else {
                        bm25_score
                    }
                };
                #[cfg(not(feature = "embeddings"))]
                let final_score = bm25_score;
                (id, final_score)
            })
            .collect()
    }

    /// Walk a directory for files with specific extensions, ignoring all ignore rules.
    /// Used to scan stub directories that are typically excluded by `.gitignore`.
    fn walk_stub_dir(dir: &Path, extensions: &[&str]) -> Vec<PathBuf> {
        WalkBuilder::new(dir)
            .hidden(false)
            .git_ignore(false)
            .git_global(false)
            .git_exclude(false)
            .build()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .filter(|e| {
                let ext = e
                    .path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x.to_lowercase())
                    .unwrap_or_default();
                extensions.contains(&ext.as_str())
            })
            .map(|e| e.into_path())
            .collect()
    }

    /// Collect library stub files from well-known locations within the workspace:
    /// - TypeScript: `node_modules/@types/**/*.ts` (`.d.ts` files)
    /// - Python: `site-packages/**/*.pyi` inside common virtual-env directories
    /// - Swift: `**/*.swiftinterface` inside `.build/` (SPM package cache)
    ///
    /// Returns an empty list when `config.index_stubs` is false.
    fn collect_stub_files(&self) -> Vec<PathBuf> {
        if !self.config.index_stubs {
            return Vec::new();
        }
        let root = &self.config.workspace_root;
        let mut files: Vec<PathBuf> = Vec::new();

        // ── TypeScript: node_modules/@types/**/*.ts ────────────────────────────
        let types_dir = root.join("node_modules").join("@types");
        if types_dir.is_dir() {
            files.extend(Self::walk_stub_dir(&types_dir, &["ts"]));
        }

        // ── Python: site-packages/**/*.pyi ────────────────────────────────────
        // Search common virtual-environment root names for site-packages dirs.
        for venv in &["venv", ".venv", "env", ".env", ".tox"] {
            let venv_dir = root.join(venv);
            if !venv_dir.is_dir() {
                continue;
            }
            // Walk venv (gitignore disabled) and collect any site-packages dirs.
            let site_pkg_dirs: Vec<PathBuf> = WalkBuilder::new(&venv_dir)
                .hidden(false)
                .git_ignore(false)
                .git_global(false)
                .git_exclude(false)
                .max_depth(Some(6))
                .build()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_type().map(|t| t.is_dir()).unwrap_or(false)
                        && e.file_name() == "site-packages"
                })
                .map(|e| e.into_path())
                .collect();
            for dir in site_pkg_dirs {
                files.extend(Self::walk_stub_dir(&dir, &["pyi"]));
            }
        }

        // ── Swift: .swiftinterface in SPM .build/ cache ───────────────────────
        let build_dir = root.join(".build");
        if build_dir.is_dir() {
            files.extend(Self::walk_stub_dir(&build_dir, &["swiftinterface"]));
        }

        tracing::debug!("collect_stub_files: found {} stub files", files.len());
        files
    }

    fn collect_source_files(&self) -> Result<Vec<PathBuf>> {
        let walker = WalkBuilder::new(&self.config.workspace_root)
            .hidden(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .add_custom_ignore_filename(".codesurgeonignore")
            .build();

        let files: Vec<PathBuf> = walker
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if !entry.file_type()?.is_file() {
                    return None;
                }
                let path = entry.into_path();
                if is_sensitive_file(&path) {
                    return None;
                }
                // Filter by extension
                let ext = path.extension()?.to_str()?.to_lowercase();
                if matches!(
                    ext.as_str(),
                    "py" | "ts"
                        | "tsx"
                        | "js"
                        | "jsx"
                        | "mjs"
                        | "sh"
                        | "bash"
                        | "html"
                        | "htm"
                        | "rs"
                        | "swift"
                        | "sql"
                        | "md"
                        | "mdx"
                ) {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        Ok(files)
    }

    pub fn workspace_root(&self) -> &Path {
        &self.config.workspace_root
    }

    pub fn session_id(&self) -> &str {
        &self.config.session_id
    }
}

// ── Secrets exclusion (11a) ───────────────────────────────────────────────────

/// Returns true if the file's name matches a well-known sensitive-file pattern
/// and should never be indexed, regardless of other settings.
fn is_sensitive_filename(path: &Path) -> bool {
    let name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n.to_lowercase(),
        None => return false,
    };

    // .env, .env.local, .env.production, etc.
    if name.starts_with(".env") {
        return true;
    }
    // foo.env
    if name.ends_with(".env") {
        return true;
    }
    // secret / credential / password anywhere in the name
    if name.contains("secret") || name.contains("credential") || name.contains("password") {
        return true;
    }
    // Certificate / key material
    if matches!(
        path.extension().and_then(|e| e.to_str()).unwrap_or(""),
        "pem" | "key" | "p12" | "pfx"
    ) {
        return true;
    }

    false
}

/// Heuristically scans the first 4 KB of a file for common API key literal patterns.
/// Returns true if a high-confidence pattern is found.
fn file_contains_api_key(path: &Path) -> bool {
    use std::io::Read;
    let mut buf = [0u8; 4096];
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let n = f.read(&mut buf).unwrap_or(0);
    let content = match std::str::from_utf8(&buf[..n]) {
        Ok(s) => s,
        // Binary file — not a source file we'd index anyway, but be safe
        Err(_) => return false,
    };

    // (prefix, minimum length of the token after the prefix)
    const PREFIXES: &[(&str, usize)] = &[
        ("AKIA", 12),       // AWS access key ID
        ("sk-", 24),        // OpenAI-style secret key
        ("ghp_", 16),       // GitHub personal access token
        ("github_pat_", 8), // GitHub fine-grained PAT
        ("glpat-", 16),     // GitLab personal access token
        ("xoxb-", 8),       // Slack bot token
        ("xoxp-", 8),       // Slack user token
    ];

    for (prefix, min_after) in PREFIXES {
        let mut search = content;
        while let Some(pos) = search.find(prefix) {
            let after = &search[pos + prefix.len()..];
            let token_len = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '/'))
                .count();
            if token_len >= *min_after {
                return true;
            }
            // Advance past this occurrence
            search = &search[pos + prefix.len()..];
        }
    }

    false
}

/// Combined check: excludes the file if its name matches sensitive patterns
/// OR if its first 4 KB contains a high-confidence API key literal.
fn is_sensitive_file(path: &Path) -> bool {
    if is_sensitive_filename(path) {
        return true;
    }
    if file_contains_api_key(path) {
        tracing::debug!(path = %path.display(), "excluding file: API key pattern detected");
        return true;
    }
    false
}

/// Map the string kind accepted by `submit_lsp_edges` to an `EdgeKind`.
/// "extends" is an alias for "inherits"; anything unrecognised → `References`.
fn lsp_kind_to_edge_kind(kind: &str) -> EdgeKind {
    match kind {
        "calls" => EdgeKind::Calls,
        "imports" => EdgeKind::Imports,
        "implements" => EdgeKind::Implements,
        "extends" | "inherits" => EdgeKind::Inherits,
        _ => EdgeKind::References,
    }
}

fn sym_ref(s: &Symbol) -> SymbolRef {
    SymbolRef {
        fqn: s.fqn.clone(),
        file_path: s.file_path.clone(),
        start_line: s.start_line,
        kind: s.kind.to_string(),
    }
}

/// Merge a cluster of observations into a single content string.
/// Extracts unique query phrases and unique pivot FQNs from auto-observation content.
/// Falls back to a generic summary for non-standard content.
#[cfg(any(feature = "embeddings", test))]
fn merge_cluster_content(obs: &[&crate::memory::Observation]) -> String {
    let mut queries: Vec<String> = Vec::new();
    let mut pivots: Vec<String> = Vec::new();

    for o in obs {
        // Auto-observations have the format: Agent queried: "task" — pivots: fqn1, fqn2
        if let Some(rest) = o.content.strip_prefix("Agent queried: \"") {
            if let Some(q_end) = rest.find("\" — pivots: ") {
                let query = rest[..q_end].to_string();
                let pivot_part = &rest[q_end + "\" — pivots: ".len()..];
                if !queries.contains(&query) {
                    queries.push(query);
                }
                for p in pivot_part.split(", ") {
                    let p = p.to_string();
                    if !p.is_empty() && !pivots.contains(&p) {
                        pivots.push(p);
                    }
                }
                continue;
            }
        }
        // Fallback: treat entire content as a unique phrase
        if !queries.contains(&o.content) {
            queries.push(o.content.clone());
        }
    }

    let count = obs.len();
    if queries.is_empty() {
        return format!("[consolidated from {count} observations]");
    }
    let query_str = queries
        .iter()
        .map(|q| format!("\"{q}\""))
        .collect::<Vec<_>>()
        .join(", ");
    if pivots.is_empty() {
        format!("[consolidated from {count} observations] Queries: {query_str}")
    } else {
        format!(
            "[consolidated from {count} observations] Queries: {query_str} — pivots: {}",
            pivots.join(", ")
        )
    }
}

/// Minimum cosine similarity an observation must have to appear in a context capsule.
/// Observations below this threshold are considered topically unrelated to the query
/// and are excluded before the budget pass, so they can't crowd out relevant content.
#[cfg(feature = "embeddings")]
const OBSERVATION_MIN_SIMILARITY: f32 = 0.3;

/// Score `obs_vecs` against `query_vec` by cosine similarity, drop entries below
/// [`OBSERVATION_MIN_SIMILARITY`], sort descending, and return the indices of the
/// top `limit` survivors.
///
/// Returns indices into the original `obs_vecs` slice so the caller can look up the
/// corresponding [`Observation`] without copying. The returned vec may be shorter than
/// `limit` when fewer than `limit` observations pass the similarity threshold.
#[cfg(feature = "embeddings")]
fn rank_by_similarity(query_vec: &[f32], obs_vecs: &[Vec<f32>], limit: usize) -> Vec<usize> {
    let mut pairs: Vec<(usize, f32)> = obs_vecs
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_similarity(query_vec, v)))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs
        .into_iter()
        .filter(|(_, score)| *score >= OBSERVATION_MIN_SIMILARITY)
        .take(limit)
        .map(|(i, _)| i)
        .collect()
}

#[cfg(test)]
mod secrets_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── is_sensitive_filename ─────────────────────────────────────────────────

    #[test]
    fn dotenv_blocked() {
        assert!(is_sensitive_filename(Path::new(".env")));
        assert!(is_sensitive_filename(Path::new(".env.local")));
        assert!(is_sensitive_filename(Path::new(".env.production")));
        assert!(is_sensitive_filename(Path::new("dir/.env.test")));
    }

    #[test]
    fn dotenv_extension_blocked() {
        assert!(is_sensitive_filename(Path::new("config.env")));
        assert!(is_sensitive_filename(Path::new("prod.env")));
    }

    #[test]
    fn secret_credential_password_in_name_blocked() {
        assert!(is_sensitive_filename(Path::new("my_secret.py")));
        assert!(is_sensitive_filename(Path::new("db_credentials.json")));
        assert!(is_sensitive_filename(Path::new("user_passwords.sql")));
        assert!(is_sensitive_filename(Path::new("SECRET_KEY.txt")));
    }

    #[test]
    fn certificate_key_extensions_blocked() {
        assert!(is_sensitive_filename(Path::new("server.pem")));
        assert!(is_sensitive_filename(Path::new("id_rsa.key")));
        assert!(is_sensitive_filename(Path::new("keystore.p12")));
        assert!(is_sensitive_filename(Path::new("cert.pfx")));
    }

    #[test]
    fn normal_source_files_allowed() {
        assert!(!is_sensitive_filename(Path::new("main.rs")));
        assert!(!is_sensitive_filename(Path::new("config.toml")));
        assert!(!is_sensitive_filename(Path::new("README.md")));
        assert!(!is_sensitive_filename(Path::new("settings.py")));
        assert!(!is_sensitive_filename(Path::new("environment.ts")));
    }

    // ── file_contains_api_key ─────────────────────────────────────────────────

    fn tmp_with(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn aws_key_detected() {
        let f = tmp_with("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE123\n");
        assert!(file_contains_api_key(f.path()));
    }

    #[test]
    fn openai_key_detected() {
        let f = tmp_with("OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789\n");
        assert!(file_contains_api_key(f.path()));
    }

    #[test]
    fn github_pat_detected() {
        let f = tmp_with("token = ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ012345\n");
        assert!(file_contains_api_key(f.path()));
    }

    #[test]
    fn slack_bot_token_detected() {
        let f = tmp_with("SLACK_TOKEN=xoxb-12345678-abcdefghij\n");
        assert!(file_contains_api_key(f.path()));
    }

    #[test]
    fn clean_source_file_not_flagged() {
        let f = tmp_with("fn main() {\n    println!(\"Hello, world!\");\n}\n");
        assert!(!file_contains_api_key(f.path()));
    }

    #[test]
    fn short_prefix_match_not_flagged() {
        // "sk-" with only 3 chars after — below the min_after=24 threshold
        let f = tmp_with("let x = \"sk-abc\";\n");
        assert!(!file_contains_api_key(f.path()));
    }
}

#[cfg(test)]
mod consolidation_tests {
    use super::*;
    use crate::memory::{Observation, ObservationKind};

    fn auto_obs(content: &str) -> Observation {
        Observation::new("session-test", content, None, None, ObservationKind::Auto)
    }

    /// Standard auto-observation format: unique queries and pivots are extracted.
    #[test]
    fn merge_standard_auto_format() {
        let a = auto_obs("Agent queried: \"fix auth\" — pivots: mod::auth, mod::user");
        let b = auto_obs("Agent queried: \"fix login\" — pivots: mod::auth, mod::token");
        let merged = merge_cluster_content(&[&a, &b]);

        assert!(
            merged.starts_with("[consolidated from 2 observations]"),
            "unexpected prefix: {merged}"
        );
        assert!(merged.contains("\"fix auth\""), "missing query: {merged}");
        assert!(merged.contains("\"fix login\""), "missing query: {merged}");
        assert!(
            merged.contains("mod::auth"),
            "missing shared pivot: {merged}"
        );
        assert!(
            merged.contains("mod::token"),
            "missing unique pivot: {merged}"
        );
        // mod::auth must appear only once (deduplication)
        assert_eq!(
            merged.matches("mod::auth").count(),
            1,
            "pivot mod::auth should appear exactly once: {merged}"
        );
    }

    /// Identical queries in the cluster are deduplicated.
    #[test]
    fn merge_deduplicates_identical_queries() {
        let a = auto_obs("Agent queried: \"refactor engine\" — pivots: engine::run");
        let b = auto_obs("Agent queried: \"refactor engine\" — pivots: engine::run");
        let merged = merge_cluster_content(&[&a, &b]);

        assert_eq!(
            merged.matches("\"refactor engine\"").count(),
            1,
            "duplicate query should appear only once: {merged}"
        );
        assert_eq!(
            merged.matches("engine::run").count(),
            1,
            "duplicate pivot should appear only once: {merged}"
        );
    }

    /// Non-standard content falls back to listing unique phrases verbatim.
    #[test]
    fn merge_fallback_for_non_standard_content() {
        let a = auto_obs("Some free-form note A");
        let b = auto_obs("Some free-form note B");
        let merged = merge_cluster_content(&[&a, &b]);

        assert!(
            merged.starts_with("[consolidated from 2 observations]"),
            "unexpected prefix: {merged}"
        );
        assert!(
            merged.contains("Some free-form note A"),
            "missing content A: {merged}"
        );
        assert!(
            merged.contains("Some free-form note B"),
            "missing content B: {merged}"
        );
    }

    /// Cluster of size 1 should produce a valid (if trivial) merged string.
    #[test]
    fn merge_single_item_cluster() {
        let a = auto_obs("Agent queried: \"search\" — pivots: mod::search");
        let merged = merge_cluster_content(&[&a]);
        assert!(
            merged.starts_with("[consolidated from 1 observation"),
            "unexpected prefix: {merged}"
        );
    }
}

#[cfg(all(test, feature = "embeddings"))]
mod rank_by_similarity_tests {
    use super::*;

    /// Helper: build a simple unit vector with a 1.0 at position `idx` (rest zero).
    fn unit_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[idx] = 1.0;
        v
    }

    #[test]
    fn returns_indices_sorted_by_descending_similarity() {
        let query = unit_vec(4, 0); // [1, 0, 0, 0]
        let obs = vec![
            unit_vec(4, 2),           // idx 0: orthogonal → sim 0.0
            unit_vec(4, 0),           // idx 1: identical → sim 1.0
            vec![0.5, 0.5, 0.5, 0.5], // idx 2: partial → sim 0.5
        ];
        let result = rank_by_similarity(&query, &obs, 10);
        assert_eq!(
            result,
            vec![1, 2],
            "should return idx 1 (best) then idx 2, skip idx 0 (below threshold)"
        );
    }

    #[test]
    fn filters_below_min_similarity() {
        let query = unit_vec(4, 0);
        // All observations are orthogonal to query → cosine sim = 0.0
        let obs = vec![unit_vec(4, 1), unit_vec(4, 2), unit_vec(4, 3)];
        let result = rank_by_similarity(&query, &obs, 10);
        assert!(result.is_empty(), "all observations should be filtered out");
    }

    #[test]
    fn respects_limit() {
        let query = unit_vec(4, 0);
        let obs = vec![
            vec![0.9, 0.1, 0.0, 0.0], // high sim
            vec![0.8, 0.2, 0.0, 0.0], // high sim
            vec![0.7, 0.3, 0.0, 0.0], // high sim
        ];
        let result = rank_by_similarity(&query, &obs, 2);
        assert_eq!(result.len(), 2, "should only return 2 results");
    }

    #[test]
    fn empty_observations_returns_empty() {
        let query = unit_vec(4, 0);
        let result = rank_by_similarity(&query, &[], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn borderline_similarity_included() {
        // cosine_similarity([1,0,0,0], [0.35, 0.9367, 0, 0]) ≈ 0.35 (above 0.3 threshold)
        let query = unit_vec(4, 0);
        let s = (0.35f32 * 0.35 + 0.9367f32 * 0.9367).sqrt();
        let obs = vec![vec![0.35 / s, 0.9367 / s, 0.0, 0.0]];
        let sim = cosine_similarity(&query, &obs[0]);
        assert!(
            sim >= OBSERVATION_MIN_SIMILARITY,
            "sim {sim} should be >= 0.3"
        );
        let result = rank_by_similarity(&query, &obs, 10);
        assert_eq!(result.len(), 1, "borderline similarity should be included");
    }
}
