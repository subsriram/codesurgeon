use crate::capsule::{build_capsule, format_capsule, Capsule, MemoryEntry, DEFAULT_TOKEN_BUDGET};
use crate::db::Database;
#[cfg(feature = "embeddings")]
use crate::embedder::{cosine_similarity, Embedder};
use crate::graph::CodeGraph;
use crate::indexer::{
    extract_call_edges, extract_impl_edges, extract_import_edges, extract_type_flow_edges,
    index_file,
};
use crate::language::Language;
use crate::memory::{new_session_id, MemoryStore};
use crate::search::{SearchIndex, SearchIntent};
use crate::skeletonizer::skeletonize;
use crate::symbol::{Symbol, SymbolKind};
use crate::watcher::{hash_content, ChangeKind};
use anyhow::Result;
use ignore::WalkBuilder;
use rayon::prelude::*;

// ── Ranking constants ────────────────────────────────────────────────────────
// See docs/ranking.md for rationale. Update both when tuning.

/// BM25 candidate pool size passed to Tantivy.
const BM25_POOL_SIZE: usize = 50;
/// Structural injection: score multiplier for injected hub types.
const STRUCTURAL_INJECTION_SCORE: f32 = 5.0;
/// Centrality boost multiplier applied to BM25 score.
const CENTRALITY_BOOST: f32 = 3.0;
/// Fixed boost for markdown symbols (bypasses centrality which is always 0).
const MARKDOWN_CENTRALITY_BYPASS: f32 = 2.5;
/// Weight of BM25+centrality in the final blend (when embeddings are available).
#[cfg(feature = "embeddings")]
const BM25_BLEND_WEIGHT: f32 = 0.5;
/// Weight of semantic cosine similarity in the final blend.
#[cfg(feature = "embeddings")]
const SEMANTIC_BLEND_WEIGHT: f32 = 0.5;
/// Structural re-sort: in-degree weight (dominant signal).
const STRUCTURAL_INDEGREE_WEIGHT: f32 = 20.0;
/// Structural re-sort: accumulated BM25 weight (tiebreaker).
const STRUCTURAL_BM25_WEIGHT: f32 = 0.05;
/// Coordinator bonus per owned seed type.
const COORDINATOR_BONUS_PER_TYPE: f32 = 5.0;
/// Minimum owned seed types required to trigger coordinator bonus.
const COORDINATOR_MIN_OWNED: usize = 2;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
        }
    }

    pub fn without_embedder(mut self) -> Self {
        self.load_embedder = false;
        self
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub symbol_count: u64,
    pub edge_count: u64,
    pub file_count: u64,
    pub session_id: String,
    /// Whether Xcode 26+ MCP bridge (`xcrun mcpbridge`) was detected on this machine.
    /// When true, agents working on Swift files should prefer Xcode MCP for resolved
    /// types and live diagnostics; codesurgeon remains the fallback for semantic search
    /// and session memory.
    pub xcode_mcp_available: bool,
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
    embedder: Option<Embedder>,
    /// In-memory cache of all symbol embeddings — loaded after each index pass so
    /// run_pipeline never needs to hit SQLite for embedding lookups.
    #[cfg(feature = "embeddings")]
    embedding_cache: Arc<RwLock<Vec<(u64, Vec<f32>)>>>,
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
        let memory = Arc::new(Mutex::new(MemoryStore::new(
            Arc::clone(&db),
            &config.session_id,
        )));

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

            search_guard.commit()?;
            graph_guard.warm_caches();
            tracing::info!(
                "Warmed index: {} symbols, {} edges",
                symbols.len(),
                edges.len()
            );
        }

        // Attempt to load the embedding model (only compiled in with --features embeddings).
        // Skipped for secondary (read-only) instances — they don't compute new embeddings
        // and loading the ~500 MB ONNX model would waste ~1-2 GB of RAM per probe process.
        // Falls back to BM25-only search when None.
        #[cfg(feature = "embeddings")]
        let embedder = if config.load_embedder {
            match Embedder::new() {
                Ok(e) => {
                    tracing::info!("Embedder loaded (NomicEmbedTextV15Q, 768-dim)");
                    Some(e)
                }
                Err(e) => {
                    tracing::warn!("Embedder unavailable, falling back to BM25: {}", e);
                    None
                }
            }
        } else {
            tracing::info!("Embedder skipped (read-only instance)");
            None
        };

        // Warm the embedding cache from any previously stored embeddings.
        #[cfg(feature = "embeddings")]
        let embedding_cache = {
            let cached = db.lock().all_embeddings().unwrap_or_default();
            Arc::new(RwLock::new(cached))
        };

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
        })
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

        // Parse files in parallel with rayon
        let results: Vec<(PathBuf, String, Vec<Symbol>)> = files
            .par_iter()
            .filter_map(|path| {
                let content = std::fs::read_to_string(path).ok()?;
                let symbols = index_file(&self.config.workspace_root, path, &content)
                    .ok()
                    .unwrap_or_default();
                Some((path.clone(), content, symbols))
            })
            .collect();

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

        // Embed symbols in batches of 64 (only when embeddings feature is enabled).
        // We embed the skeleton (signature + docstring) rather than the full body —
        // shorter text, lower noise, still captures what the symbol "is".
        // Runs after graph/search locks are released so queries can proceed in parallel.
        #[cfg(feature = "embeddings")]
        if let Some(emb) = &self.embedder {
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

        // Phase 1: Remove stale db rows (brief, independent db lock).
        {
            self.db.lock().delete_file_symbols(&rel)?;
        }
        // Phase 2: Remove from in-memory graph (brief, independent graph lock).
        {
            self.graph.write().remove_file(&rel);
        }

        if kind == ChangeKind::Removed {
            return Ok(());
        }

        // Phase 3: Parse — no locks held.
        let content = std::fs::read_to_string(path)?;
        let file_hash = hash_content(content.as_bytes());
        let symbols = index_file(&self.config.workspace_root, path, &content)?;

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
            let change_summary = format!("{} symbol(s) re-indexed", symbols.len());
            let _ = mem.record_file_edit(&rel, &change_summary);
        }

        // Phase 7: Re-embed new symbols and refresh cache (brief db lock, no other locks held).
        #[cfg(feature = "embeddings")]
        if let Some(emb) = &self.embedder {
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
    pub fn run_pipeline(&self, task: &str, budget: Option<u32>) -> Result<String> {
        let budget = budget.unwrap_or(self.config.default_token_budget);
        let intent = SearchIntent::detect(task);

        tracing::debug!("run_pipeline: intent={:?}, task={}", intent, task);

        let capsule = self.build_context_capsule(task, budget, &intent)?;
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

        Ok(out)
    }

    /// Get context capsule for a query.
    pub fn get_context_capsule(&self, query: &str, budget: Option<u32>) -> Result<String> {
        let budget = budget.unwrap_or(self.config.default_token_budget);
        let intent = SearchIntent::detect(query);
        let capsule = self.build_context_capsule(query, budget, &intent)?;
        Ok(format_capsule(&capsule))
    }

    /// Get impact graph: what breaks if `symbol_fqn` changes?
    pub fn get_impact_graph(&self, symbol_fqn: &str) -> Result<ImpactResult> {
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

        let direct: Vec<SymbolRef> = graph
            .dependents(target_id)
            .into_iter()
            .map(sym_ref)
            .collect();

        let transitive: Vec<SymbolRef> = graph
            .blast_radius(target_id, self.config.max_blast_radius_depth)
            .into_iter()
            .map(sym_ref)
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
    pub fn get_skeleton(&self, file_path: &str) -> Result<SkeletonResult> {
        let graph = self.graph.read();
        let symbols = graph.file_symbols(file_path);

        let mut total_tokens = 0u32;
        let skeleton_syms: Vec<SkeletonSymbol> = symbols
            .iter()
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

    /// Index statistics and health.
    pub fn index_stats(&self) -> Result<IndexStats> {
        let db = self.db.lock();
        Ok(IndexStats {
            symbol_count: db.symbol_count()?,
            edge_count: db.edge_count()?,
            file_count: db.file_count()?,
            session_id: self.config.session_id.clone(),
            xcode_mcp_available: detect_xcode_mcp(),
        })
    }

    /// Get session observations (cross-session memory).
    pub fn get_session_context(&self) -> Result<Vec<crate::memory::Observation>> {
        let mem = self.memory.lock();
        mem.get_recent_observations(50)
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

        let raw_memories = self.memory.lock().get_recent_observations(10)?;
        let memory_entries: Vec<MemoryEntry> = raw_memories
            .into_iter()
            .map(|obs| MemoryEntry {
                content: obs.content,
                symbol_fqn: obs.symbol_fqn,
                is_stale: obs.is_stale,
                created_at: obs.created_at,
            })
            .collect();

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

    /// Reload all embeddings from SQLite into the in-memory cache.
    /// Called after every index pass so queries never need to hit the db for vectors.
    #[cfg(feature = "embeddings")]
    fn refresh_embedding_cache(&self) {
        match self.db.lock().all_embeddings() {
            Ok(embs) => *self.embedding_cache.write() = embs,
            Err(e) => tracing::warn!("Failed to refresh embedding cache: {}", e),
        }
    }

    fn build_context_capsule(
        &self,
        query: &str,
        budget: u32,
        intent: &SearchIntent,
    ) -> Result<Capsule> {
        // 1. Search for candidate symbols
        let mut search_results = self.search.lock().search(query, BM25_POOL_SIZE)?;
        let graph = self.graph.read();

        // Track original BM25 IDs before injection (used for coordinator bonus).
        let bm25_ids: std::collections::HashSet<u64> =
            search_results.iter().map(|(id, _)| *id).collect();

        // 2. Inject high-centrality types for Structural queries (BM25 can't surface them)
        if *intent == SearchIntent::Structural {
            Self::inject_structural_candidates(
                &graph,
                &mut search_results,
                self.config.max_pivots,
            );
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
            Self::apply_structural_resort(&graph, scored, &bm25_ids, query)
        } else {
            scored
        };

        // 5. Deduplicate by FQN — keep the highest-scored entry per unique FQN.
        let scored = Self::dedup_by_fqn(&graph, scored);

        // 6. Select pivots and adjacents
        let pivot_ids: Vec<u64> = scored
            .iter()
            .take(self.config.max_pivots)
            .map(|(id, _)| *id)
            .collect();
        let adjacent_ids =
            Self::select_adjacents(&graph, &pivot_ids, intent, self.config.max_adjacent);

        // 7. Resolve IDs → Symbols with filtering
        let filter_adjacents = matches!(intent, SearchIntent::Structural | SearchIntent::Explore);
        let pivot_syms: Vec<&Symbol> = pivot_ids
            .iter()
            .filter_map(|id| graph.get_symbol(*id))
            .collect();
        let adjacent_syms = Self::resolve_adjacents(&graph, &adjacent_ids, filter_adjacents);

        // 8. Fetch memories and assemble capsule
        let raw_memories = self.memory.lock().get_recent_observations(20)?;
        let memory_entries: Vec<MemoryEntry> = raw_memories
            .into_iter()
            .map(|obs| MemoryEntry {
                content: obs.content,
                symbol_fqn: obs.symbol_fqn,
                is_stale: obs.is_stale,
                created_at: obs.created_at,
            })
            .collect();

        Ok(build_capsule(
            pivot_syms,
            adjacent_syms,
            memory_entries,
            budget,
            Some(query),
        ))
    }

    /// Augment the BM25 candidate pool with top hub types ranked by family in-degree.
    /// BM25 cannot surface types whose names don't lexically match the query.
    fn inject_structural_candidates(
        graph: &CodeGraph,
        search_results: &mut Vec<(u64, f32)>,
        max_pivots: usize,
    ) {
        let mut by_in_degree: Vec<(u64, f32)> = graph
            .all_symbols()
            .filter(|s| s.kind.is_type_definition() || s.kind == SymbolKind::Module)
            .map(|s| (s.id, graph.family_in_degree_score(s.id)))
            .collect();
        by_in_degree.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (id, c_in) in by_in_degree.into_iter().take(max_pivots * 2) {
            if !search_results.iter().any(|(sid, _)| *sid == id) {
                search_results.push((id, c_in * STRUCTURAL_INJECTION_SCORE));
            }
        }
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
        let semantic_scores: std::collections::HashMap<u64, f32> =
            if let Some(emb) = &self.embedder {
                match emb.embed_one(query) {
                    Ok(query_vec) => {
                        let cache = self.embedding_cache.read();
                        cache
                            .iter()
                            .map(|(id, vec)| (*id, cosine_similarity(&query_vec, vec)))
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

    /// For Structural intent: re-sort so type definitions ranked by in-degree come first,
    /// with a coordinator bonus for types that declare BM25-matched types as properties.
    fn apply_structural_resort(
        graph: &CodeGraph,
        scored: Vec<(u64, f32)>,
        bm25_ids: &std::collections::HashSet<u64>,
        query: &str,
    ) -> Vec<(u64, f32)> {
        let is_hub_type = |id: u64| {
            graph
                .get_symbol(id)
                .map(|s| {
                    s.kind.is_type_definition()
                        || s.kind == SymbolKind::Impl
                        || s.kind == SymbolKind::Module
                })
                .unwrap_or(false)
        };

        let mut type_scored: Vec<(u64, f32)> = scored
            .iter()
            .filter(|(id, _)| is_hub_type(*id))
            .map(|(id, accumulated)| {
                let c_in = graph.family_in_degree_score(*id);
                (*id, c_in * STRUCTURAL_INDEGREE_WEIGHT + accumulated * STRUCTURAL_BM25_WEIGHT)
            })
            .collect();

        // Coordinator bonus: find the type that DECLARES BM25-matched types as member
        // variables. Seeds = BM25-matched types whose names overlap with 4-char query stems.
        let query_stems: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() > 3)
            .map(|t| t[..t.len().min(4)].to_lowercase())
            .collect();
        let seed_names: Vec<String> = type_scored
            .iter()
            .filter_map(|(id, _)| {
                if bm25_ids.contains(id) {
                    graph.get_symbol(*id).map(|s| s.name.clone())
                } else {
                    None
                }
            })
            .filter(|n| {
                n.len() > 4 && {
                    let nl = n.to_lowercase();
                    query_stems.iter().any(|stem| nl.contains(stem.as_str()))
                }
            })
            .collect();

        if seed_names.len() >= 2 {
            for (id, score) in &mut type_scored {
                if let Some(sym) = graph.get_symbol(*id) {
                    let path_lower = sym.file_path.to_lowercase();
                    if path_lower.contains("test")
                        || path_lower.contains("spec")
                        || path_lower.contains("mock")
                    {
                        continue;
                    }
                    let owned = seed_names
                        .iter()
                        .filter(|name| {
                            *name != &sym.name
                                && (sym.body.contains(&format!(": {}", name))
                                    || sym.body.contains(&format!(": [{}]", name))
                                    || sym.body.contains(&format!(": {}?", name))
                                    || sym.body.contains(&format!("[{}]", name)))
                        })
                        .count();
                    if owned >= COORDINATOR_MIN_OWNED {
                        *score += owned as f32 * COORDINATOR_BONUS_PER_TYPE;
                    }
                }
            }
        }

        type_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let others: Vec<(u64, f32)> = scored
            .into_iter()
            .filter(|(id, _)| !is_hub_type(*id))
            .collect();
        type_scored.into_iter().chain(others).collect()
    }

    /// Deduplicate by FQN — keep the highest-scored entry per unique FQN.
    fn dedup_by_fqn(graph: &CodeGraph, scored: Vec<(u64, f32)>) -> Vec<(u64, f32)> {
        let mut seen_fqns = std::collections::HashSet::new();
        scored
            .into_iter()
            .filter(|(id, _)| {
                graph
                    .get_symbol(*id)
                    .map(|sym| seen_fqns.insert(sym.fqn.clone()))
                    .unwrap_or(true)
            })
            .collect()
    }

    /// Select adjacent symbol IDs based on the search intent.
    fn select_adjacents(
        graph: &CodeGraph,
        pivot_ids: &[u64],
        intent: &SearchIntent,
        max_adjacent: usize,
    ) -> Vec<u64> {
        let raw: Vec<u64> = match intent {
            SearchIntent::Debug => pivot_ids
                .iter()
                .flat_map(|&id| {
                    graph
                        .dependencies(id)
                        .iter()
                        .map(|s| s.id)
                        .collect::<Vec<_>>()
                })
                .filter(|id| !pivot_ids.contains(id))
                .take(max_adjacent)
                .collect(),
            SearchIntent::Refactor => pivot_ids
                .iter()
                .flat_map(|&id| {
                    graph
                        .blast_radius(id, 2)
                        .iter()
                        .map(|s| s.id)
                        .collect::<Vec<_>>()
                })
                .filter(|id| !pivot_ids.contains(id))
                .take(max_adjacent)
                .collect(),
            _ => pivot_ids
                .iter()
                .flat_map(|&id| {
                    let mut adj: Vec<u64> =
                        graph.dependencies(id).iter().map(|s| s.id).collect();
                    adj.extend(graph.dependents(id).iter().map(|s| s.id));
                    adj
                })
                .filter(|id| !pivot_ids.contains(id))
                .take(max_adjacent)
                .collect(),
        };
        // Deduplicate (same symbol may be reachable from multiple pivots).
        let mut seen = std::collections::HashSet::new();
        raw.into_iter().filter(|id| seen.insert(*id)).collect()
    }

    /// Resolve adjacent IDs to symbols, filtering test files and capping per-file counts.
    fn resolve_adjacents<'a>(
        graph: &'a CodeGraph,
        adjacent_ids: &[u64],
        filter_test_files: bool,
    ) -> Vec<&'a Symbol> {
        let mut file_counts: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::new();
        adjacent_ids
            .iter()
            .filter_map(|id| graph.get_symbol(*id))
            .filter(|sym| {
                if filter_test_files {
                    let p = sym.file_path.to_lowercase();
                    !p.contains("test") && !p.contains("spec") && !p.contains("mock")
                } else {
                    true
                }
            })
            .filter(|sym| {
                let count = file_counts.entry(sym.file_path.as_str()).or_insert(0);
                *count += 1;
                *count <= 2 // max 2 symbols per file in adjacents
            })
            .collect()
    }

    fn collect_source_files(&self) -> Result<Vec<PathBuf>> {
        let walker = WalkBuilder::new(&self.config.workspace_root)
            .hidden(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .build();

        let files: Vec<PathBuf> = walker
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if !entry.file_type()?.is_file() {
                    return None;
                }
                let path = entry.into_path();
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

/// Parse a unified diff and return (file_path, start_line, end_line) for each changed hunk.
fn parse_diff_symbols(diff: &str) -> Vec<(String, u32, u32)> {
    let mut result = Vec::new();
    let mut current_file = String::new();
    let mut hunk_start = 0u32;
    let mut hunk_end = 0u32;

    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            // Flush previous hunk
            if !current_file.is_empty() && hunk_end >= hunk_start {
                result.push((current_file.clone(), hunk_start, hunk_end));
            }
            current_file = rest.trim().to_string();
            hunk_start = 0;
            hunk_end = 0;
        } else if line.starts_with("@@ ") {
            // Flush previous hunk for this file
            if !current_file.is_empty() && hunk_end >= hunk_start && hunk_start > 0 {
                result.push((current_file.clone(), hunk_start, hunk_end));
            }
            // Parse "@@ -old_start,old_len +new_start,new_len @@"
            // We care about the new file's line range (+new_start,new_len)
            if let Some((start, len)) = parse_hunk_header(line) {
                hunk_start = start;
                hunk_end = start + len.saturating_sub(1);
            }
        }
    }

    // Flush last hunk
    if !current_file.is_empty() && hunk_end >= hunk_start && hunk_start > 0 {
        result.push((current_file, hunk_start, hunk_end));
    }

    result
}

fn parse_hunk_header(line: &str) -> Option<(u32, u32)> {
    // "@@ -a,b +c,d @@" — extract c and d
    let plus_part = line.split('+').nth(1)?;
    let range_part = plus_part.split(' ').next()?;
    let mut parts = range_part.splitn(2, ',');
    let start: u32 = parts.next()?.parse().ok()?;
    let len: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    Some((start, len))
}

fn sym_ref(s: &Symbol) -> SymbolRef {
    SymbolRef {
        fqn: s.fqn.clone(),
        file_path: s.file_path.clone(),
        start_line: s.start_line,
        kind: s.kind.to_string(),
    }
}

/// Probe for Xcode 26+ MCP bridge availability. Result is cached after the first call
/// so repeated `run_pipeline` or `index_status` calls pay the subprocess cost only once.
fn detect_xcode_mcp() -> bool {
    use std::sync::OnceLock;
    static XCODE_MCP: OnceLock<bool> = OnceLock::new();
    *XCODE_MCP.get_or_init(|| {
        std::process::Command::new("xcrun")
            .args(["--find", "mcpbridge"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Human-readable hint appended to capsule output when Swift symbols are present.
/// Tells the agent which enrichment path is available and what the fallback is,
/// so it never silently operates on incomplete type information.
fn swift_enrichment_hint(xcode_mcp_available: bool) -> String {
    if xcode_mcp_available {
        "\n> **Swift symbols detected.** \
         Xcode MCP is available — call its tools for resolved types and live build diagnostics. \
         codesurgeon results reflect tree-sitter parsing and remain available for semantic search \
         and session memory.\n"
            .to_string()
    } else {
        "\n> **Swift symbols detected.** \
         Xcode MCP was not found — results are based on tree-sitter parsing only (no resolved types, \
         no macro-expanded symbols). \
         To enable full Swift enrichment: install Xcode 26+ and turn on \
         Settings → Intelligence → Enable Model Context Protocol, \
         then wire it up with `xcrun mcpbridge`. \
         codesurgeon's graph is still usable for semantic search and session memory.\n"
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn test_engine(dir: &TempDir) -> CoreEngine {
        let config = EngineConfig::new(dir.path()).without_embedder();
        CoreEngine::new(config).expect("engine init failed")
    }

    /// Parallel calls to `run_pipeline` must not deadlock or panic.
    /// This guards against lock-ordering bugs between graph/search/db.
    #[test]
    fn parallel_queries_do_not_deadlock() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Arc::new(test_engine(&dir));

        let handles: Vec<_> = (0..8)
            .map(|i| {
                let e = Arc::clone(&engine);
                std::thread::spawn(move || {
                    let query = format!("query number {}", i);
                    // Empty workspace — just must not panic/deadlock.
                    let _ = e.run_pipeline(&query, Some(500));
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    /// Concurrent `reindex_file` calls for the same file must not corrupt the
    /// index or deadlock. Each call should complete without panicking.
    #[test]
    fn concurrent_reindex_same_file() {
        let dir = tempfile::tempdir().unwrap();

        // Write a small Rust file into the workspace.
        let file_path = dir.path().join("lib.rs");
        std::fs::write(
            &file_path,
            "pub fn foo() {}\npub fn bar() {}\n",
        )
        .unwrap();

        let engine = Arc::new(test_engine(&dir));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let e = Arc::clone(&engine);
                let p = file_path.clone();
                std::thread::spawn(move || {
                    e.reindex_file(&p, crate::watcher::ChangeKind::Modified)
                        .expect("reindex_file failed");
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        // After concurrent reindexing, a query must still succeed.
        let _ = engine.run_pipeline("foo", Some(500));
    }

    /// Queries issued while indexing is flagged as in-progress must succeed
    /// (possibly with partial results) and must not panic.
    #[test]
    fn query_during_indexing_does_not_panic() {
        let dir = tempfile::tempdir().unwrap();

        // Write a few files so there's something to index.
        std::fs::write(dir.path().join("a.py"), "def alpha(): pass\n").unwrap();
        std::fs::write(dir.path().join("b.py"), "def beta(): pass\n").unwrap();

        let engine = Arc::new(test_engine(&dir));

        // Spawn an indexer thread.
        let e_idx = Arc::clone(&engine);
        let indexer = std::thread::spawn(move || {
            e_idx.index_workspace().expect("index_workspace failed");
        });

        // Fire queries from multiple threads while indexing may still be running.
        let query_handles: Vec<_> = (0..4)
            .map(|_| {
                let e = Arc::clone(&engine);
                std::thread::spawn(move || {
                    let _ = e.run_pipeline("alpha", Some(500));
                })
            })
            .collect();

        indexer.join().expect("indexer thread panicked");
        for h in query_handles {
            h.join().expect("query thread panicked");
        }
    }
}
