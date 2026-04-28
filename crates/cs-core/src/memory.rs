use crate::db::Database;
use anyhow::Result;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

// ── ObservationKind ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObservationKind {
    /// Manually saved by the agent or user
    Manual,
    /// Passively captured from file changes
    Passive,
    /// Anti-pattern: the agent went in circles
    DeadEnd,
    /// Anti-pattern: same file modified 4+ times rapidly
    FileThrash,
    /// Agent-generated insight about a symbol
    Insight,
    /// Automatically captured from run_pipeline / get_context_capsule calls
    Auto,
    /// Compressed summary created by the compression pass; replaces 3+ originals
    Summary,
    /// Semantically consolidated entry — merges 2+ similar auto-observations by embedding similarity
    Consolidated,
}

impl ObservationKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ObservationKind::Manual => "manual",
            ObservationKind::Passive => "passive",
            ObservationKind::DeadEnd => "dead_end",
            ObservationKind::FileThrash => "file_thrash",
            ObservationKind::Insight => "insight",
            ObservationKind::Auto => "auto",
            ObservationKind::Summary => "summary",
            ObservationKind::Consolidated => "consolidated",
        }
    }

    pub fn parse_kind(s: &str) -> Self {
        match s {
            "passive" => ObservationKind::Passive,
            "dead_end" => ObservationKind::DeadEnd,
            "file_thrash" => ObservationKind::FileThrash,
            "insight" => ObservationKind::Insight,
            "auto" => ObservationKind::Auto,
            "summary" => ObservationKind::Summary,
            "consolidated" => ObservationKind::Consolidated,
            _ => ObservationKind::Manual,
        }
    }

    /// Default TTL in days for each kind.
    /// Returns `None` for kinds that never expire (Manual, Insight).
    pub fn default_ttl_days(&self) -> Option<i64> {
        match self {
            ObservationKind::Auto => Some(7),
            ObservationKind::Passive => Some(7),
            ObservationKind::FileThrash => Some(7),
            ObservationKind::DeadEnd => Some(7),
            ObservationKind::Summary => Some(90),
            ObservationKind::Consolidated => Some(90),
            ObservationKind::Manual => None,
            ObservationKind::Insight => None,
        }
    }
}

// ── IndexingConfig ─────────────────────────────────────────────────────────────

/// Indexing-time enrichment configuration.
/// Loaded from `.codesurgeon/config.toml` under `[indexing]` if present.
///
/// Example config.toml:
/// ```toml
/// [indexing]
/// rust_expand_macros  = true
/// rust_rustdoc_types  = true
/// python_pyright      = true
/// ts_types            = true
///
/// [git]
/// track_manifest = true
/// ```
#[derive(Debug, Clone, Default)]
pub struct IndexingConfig {
    /// When true, run `cargo-expand` on Rust files that contain proc-macro or
    /// derive invocations and add the generated symbols to the index.
    /// Requires `cargo-expand` to be installed; skipped gracefully if absent.
    /// Default: false.
    pub rust_expand_macros: bool,

    /// When true, run `cargo +nightly doc --output-format json` and merge
    /// resolved return types and trait-impl lists into existing symbols.
    /// Requires a nightly Rust toolchain; skipped gracefully if absent.
    /// Re-run is gated on `Cargo.lock` content hash — skipped when unchanged.
    /// Default: false.
    pub rust_rustdoc_types: bool,

    /// When true, run `pyright --outputjson` and merge resolved type
    /// annotations into existing Python symbols.
    /// Requires `pyright` to be installed (npm install -g pyright); skipped
    /// gracefully if absent.
    /// Re-run is gated on a hash of Python file stats — skipped when unchanged.
    /// Default: false.
    pub python_pyright: bool,

    /// When true, invoke the bundled Node.js shim (`ts-enricher.js`) to run
    /// `ts.createProgram()` + `TypeChecker` over the workspace and annotate
    /// TypeScript/JavaScript symbols with their resolved types.
    ///
    /// Requires `node` on PATH and a `tsconfig.json` in the workspace root.
    /// The `typescript` package is loaded from `node_modules/typescript` first,
    /// then falls back to any globally installed copy.  Skipped gracefully when
    /// either prerequisite is absent.
    ///
    /// Re-run is gated on `tsconfig.json` content hash — skipped when unchanged.
    /// Default: false.
    pub ts_types: bool,

    /// When true, omit `manifest.json` from `.codesurgeon/.gitignore` so it
    /// can be committed and shared across clones.
    /// Set via `CS_TRACK_MANIFEST=1` env var or `[git] track_manifest = true`
    /// in `config.toml`. Default: false (manifest.json is gitignored).
    pub track_manifest: bool,

    /// Token budget per context capsule.
    /// Set via `[context] max_tokens = 8000` in `config.toml`. Default: None (use engine default).
    pub max_tokens: Option<u32>,

    /// Skeleton detail level: "minimal", "standard", or "detailed".
    /// Set via `[context] skeleton_detail = "standard"` in `config.toml`. Default: None.
    pub skeleton_detail: Option<String>,

    /// USD cost per token for savings display in `get_stats`.
    /// Set via `[observability] token_rate_usd = 0.000003` in `config.toml`. Default: None.
    pub token_rate_usd: Option<f64>,

    /// When true, every `run_pipeline` / `get_context_capsule` call auto-records
    /// the `query → top-pivots` tuple as an `Auto` observation. The consolidator
    /// later merges similar entries into `[consolidated from N observations]` memories
    /// that surface in future capsules.
    ///
    /// Default: **false**. The record-side has no success signal — a query
    /// whose capsule returned the wrong pivots is recorded identically to one
    /// that led to a correct fix, so repeated failures cement the wrong
    /// pivots as "canonical" memory and poison future runs (regression
    /// observed on sympy-21379 in the SWE-bench harness).
    ///
    /// Set `[observability] auto_observations = true` in `config.toml` to
    /// restore the pre-#72 behaviour. Explicit `save_observation` calls are
    /// unaffected — they remain the agent-attested memory path.
    pub auto_observations: bool,

    /// Percentile of the raw-degree distribution used to derive the centrality
    /// smoothing constant `k` (see `CodeGraph::warm_caches`).
    /// Set via `[ranking] centrality_k_percentile = 0.5` in `config.toml`.
    /// Default: None (engine uses 0.5 = median).
    pub centrality_k_percentile: Option<f32>,

    /// Explicit override for the centrality smoothing constant. When set,
    /// bypasses percentile-based derivation and pins `k` to this value.
    /// Set via `[ranking] centrality_k = 15.0` in `config.toml`. Default: None.
    pub centrality_k_override: Option<f32>,
}

impl IndexingConfig {
    /// Load from a `config.toml` file. Missing file or section → defaults.
    pub fn load_from_toml(path: &std::path::Path) -> Self {
        let mut cfg = IndexingConfig::default();
        let Ok(text) = std::fs::read_to_string(path) else {
            return cfg;
        };
        let Ok(table) = text.parse::<toml::Table>() else {
            return cfg;
        };
        if let Some(indexing) = table.get("indexing").and_then(|v| v.as_table()) {
            if let Some(v) = indexing.get("rust_expand_macros").and_then(|v| v.as_bool()) {
                cfg.rust_expand_macros = v;
            }
            if let Some(v) = indexing.get("rust_rustdoc_types").and_then(|v| v.as_bool()) {
                cfg.rust_rustdoc_types = v;
            }
            if let Some(v) = indexing.get("python_pyright").and_then(|v| v.as_bool()) {
                cfg.python_pyright = v;
            }
            if let Some(v) = indexing.get("ts_types").and_then(|v| v.as_bool()) {
                cfg.ts_types = v;
            }
        }
        if let Some(git) = table.get("git").and_then(|v| v.as_table()) {
            if let Some(v) = git.get("track_manifest").and_then(|v| v.as_bool()) {
                cfg.track_manifest = v;
            }
        }
        if let Some(context) = table.get("context").and_then(|v| v.as_table()) {
            if let Some(v) = context.get("max_tokens").and_then(|v| v.as_integer()) {
                cfg.max_tokens = Some(v.max(100) as u32);
            }
            if let Some(v) = context.get("skeleton_detail").and_then(|v| v.as_str()) {
                cfg.skeleton_detail = Some(v.to_string());
            }
        }
        if let Some(obs) = table.get("observability").and_then(|v| v.as_table()) {
            if let Some(v) = obs.get("token_rate_usd").and_then(|v| v.as_float()) {
                cfg.token_rate_usd = Some(v);
            }
            if let Some(v) = obs.get("auto_observations").and_then(|v| v.as_bool()) {
                cfg.auto_observations = v;
            }
        }
        if let Some(ranking) = table.get("ranking").and_then(|v| v.as_table()) {
            if let Some(v) = ranking
                .get("centrality_k_percentile")
                .and_then(|v| v.as_float())
            {
                cfg.centrality_k_percentile = Some(v as f32);
            }
            if let Some(v) = ranking.get("centrality_k").and_then(|v| v.as_float()) {
                cfg.centrality_k_override = Some(v as f32);
            } else if let Some(v) = ranking.get("centrality_k").and_then(|v| v.as_integer()) {
                cfg.centrality_k_override = Some(v as f32);
            }
        }
        // CS_TRACK_MANIFEST env var overrides config.toml
        if std::env::var("CS_TRACK_MANIFEST").as_deref() == Ok("1") {
            cfg.track_manifest = true;
        }
        cfg
    }

    /// Load user-level config from `~/.config/codesurgeon/config.toml`, then
    /// overlay workspace-level config on top. Workspace settings take precedence.
    pub fn load_with_user_fallback(workspace_config: &std::path::Path) -> Self {
        // Start with user-level config as base (lower precedence).
        let user_config = dirs_or_home().join("config.toml");
        let mut cfg = if user_config.exists() {
            Self::load_from_toml(&user_config)
        } else {
            Self::default()
        };

        // Overlay workspace config (higher precedence).
        if workspace_config.exists() {
            let ws = Self::load_from_toml(workspace_config);
            if ws.rust_expand_macros {
                cfg.rust_expand_macros = true;
            }
            if ws.rust_rustdoc_types {
                cfg.rust_rustdoc_types = true;
            }
            if ws.python_pyright {
                cfg.python_pyright = true;
            }
            if ws.ts_types {
                cfg.ts_types = true;
            }
            if ws.track_manifest {
                cfg.track_manifest = true;
            }
            if ws.max_tokens.is_some() {
                cfg.max_tokens = ws.max_tokens;
            }
            if ws.skeleton_detail.is_some() {
                cfg.skeleton_detail = ws.skeleton_detail;
            }
            if ws.token_rate_usd.is_some() {
                cfg.token_rate_usd = ws.token_rate_usd;
            }
            if ws.auto_observations {
                cfg.auto_observations = true;
            }
            if ws.centrality_k_percentile.is_some() {
                cfg.centrality_k_percentile = ws.centrality_k_percentile;
            }
            if ws.centrality_k_override.is_some() {
                cfg.centrality_k_override = ws.centrality_k_override;
            }
        }

        // CS_TRACK_MANIFEST env var overrides everything
        if std::env::var("CS_TRACK_MANIFEST").as_deref() == Ok("1") {
            cfg.track_manifest = true;
        }
        cfg
    }
}

/// Return `~/.config/codesurgeon/` (or a fallback).
fn dirs_or_home() -> std::path::PathBuf {
    if let Some(config_dir) = dirs_path() {
        config_dir.join("codesurgeon")
    } else {
        std::path::PathBuf::from(".config/codesurgeon")
    }
}

#[cfg(not(target_os = "windows"))]
fn dirs_path() -> Option<std::path::PathBuf> {
    std::env::var("XDG_CONFIG_HOME")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| std::path::PathBuf::from(h).join(".config"))
        })
}

#[cfg(target_os = "windows")]
fn dirs_path() -> Option<std::path::PathBuf> {
    std::env::var("APPDATA").ok().map(std::path::PathBuf::from)
}

// ── MemoryConfig ───────────────────────────────────────────────────────────────

/// TTL configuration for the observation store.
/// Loaded from `.codesurgeon/config.toml` under `[memory]` if present,
/// otherwise uses the built-in defaults.
///
/// Example config.toml:
/// ```toml
/// [memory]
/// auto_ttl_days    = 7
/// manual_ttl_days  = 90
/// ```
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// TTL for auto/passive/file_thrash/dead_end observations (days). Default: 7.
    pub auto_ttl_days: i64,
    /// TTL for manual/insight observations (days). `None` = never expire. Default: None.
    pub manual_ttl_days: Option<i64>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            auto_ttl_days: 7,
            manual_ttl_days: None,
        }
    }
}

impl MemoryConfig {
    /// Load from a `config.toml` file at the given path.
    /// If the file is missing or the `[memory]` section is absent, returns defaults.
    pub fn load_from_toml(path: &std::path::Path) -> Self {
        let mut cfg = MemoryConfig::default();
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return cfg,
            Err(e) => {
                tracing::warn!("Failed to read {}: {}", path.display(), e);
                return cfg;
            }
        };
        let table = match text.parse::<toml::Table>() {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    "Malformed TOML in {}: {} — using default memory config",
                    path.display(),
                    e
                );
                return cfg;
            }
        };
        if let Some(memory) = table.get("memory").and_then(|v| v.as_table()) {
            if let Some(v) = memory.get("auto_ttl_days").and_then(|v| v.as_integer()) {
                cfg.auto_ttl_days = v.max(1);
            }
            if let Some(v) = memory.get("manual_ttl_days").and_then(|v| v.as_integer()) {
                cfg.manual_ttl_days = Some(v.max(1));
            }
        }
        cfg
    }

    /// Compute `expires_at` for an observation of the given kind using this config.
    pub fn expires_at(&self, kind: &ObservationKind) -> Option<String> {
        let days = match kind {
            ObservationKind::Auto
            | ObservationKind::Passive
            | ObservationKind::FileThrash
            | ObservationKind::DeadEnd => Some(self.auto_ttl_days),
            ObservationKind::Manual | ObservationKind::Insight => self.manual_ttl_days,
            // Summaries and consolidated entries get the manual TTL (or 90 days if manual never expires)
            ObservationKind::Summary | ObservationKind::Consolidated => {
                self.manual_ttl_days.or(Some(90))
            }
        }?;
        let ts = chrono::Utc::now() + chrono::Duration::days(days);
        Some(ts.to_rfc3339())
    }
}

// ── Observation ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub id: String,
    pub session_id: String,
    /// Which agent wrote this (claude-code, cursor, etc.)
    pub agent_id: Option<String>,
    pub content: String,
    /// FQN of the symbol this observation is about
    pub symbol_fqn: Option<String>,
    /// Hash of the symbol body at time of observation.
    /// Used to detect staleness when code changes.
    pub symbol_hash: Option<String>,
    pub created_at: String,
    pub is_stale: bool,
    pub kind: ObservationKind,
    /// RFC-3339 timestamp after which this observation is considered expired.
    /// `None` means it never expires automatically (manual/insight with no TTL set).
    pub expires_at: Option<String>,
    /// AST-level change category for file-change observations.
    /// One of: `new_symbol`, `deleted_symbol`, `signature_change`, `body_change`, `dependency_added`.
    /// `None` for non-file-change observations (manual, auto, consolidated, etc.).
    pub change_category: Option<String>,
}

impl Observation {
    pub fn new(
        session_id: &str,
        content: &str,
        symbol_fqn: Option<&str>,
        symbol_hash: Option<&str>,
        kind: ObservationKind,
    ) -> Self {
        let expires_at = kind
            .default_ttl_days()
            .map(|days| (chrono::Utc::now() + chrono::Duration::days(days)).to_rfc3339());
        Observation {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            agent_id: None,
            content: content.to_string(),
            symbol_fqn: symbol_fqn.map(str::to_string),
            symbol_hash: symbol_hash.map(str::to_string),
            created_at: chrono::Utc::now().to_rfc3339(),
            is_stale: false,
            kind,
            expires_at,
            change_category: None,
        }
    }

    pub fn with_change_category(mut self, cat: &str) -> Self {
        self.change_category = Some(cat.to_string());
        self
    }

    /// Override the default `expires_at` with a value derived from `MemoryConfig`.
    pub fn with_ttl(mut self, config: &MemoryConfig) -> Self {
        self.expires_at = config.expires_at(&self.kind);
        self
    }

    pub fn with_agent(mut self, agent_id: &str) -> Self {
        self.agent_id = Some(agent_id.to_string());
        self
    }
}

// ── Symbol change classification ──────────────────────────────────────────────

/// One classified change detected during `reindex_file`.
#[derive(Debug, Clone)]
pub struct SymbolChange {
    pub fqn: String,
    pub category: &'static str,
}

impl SymbolChange {
    pub fn new(fqn: impl Into<String>, category: &'static str) -> Self {
        SymbolChange {
            fqn: fqn.into(),
            category,
        }
    }
}

// ── Anti-pattern tracking ──────────────────────────────────────────────────────

/// Tracks file edit frequency within a session window to detect file thrashing.
struct FileThrashTracker {
    /// file_path → list of timestamps
    edits: std::collections::HashMap<String, Vec<chrono::DateTime<chrono::Utc>>>,
    /// If a file is modified >= this many times in `window_secs`, flag it
    threshold: usize,
    window_secs: i64,
}

impl FileThrashTracker {
    fn new() -> Self {
        FileThrashTracker {
            edits: std::collections::HashMap::new(),
            threshold: 4,
            window_secs: 300, // 5 minutes
        }
    }

    /// Record an edit and return true if a thrash pattern is detected.
    fn record_edit(&mut self, file_path: &str) -> bool {
        let now = chrono::Utc::now();
        let edits = self.edits.entry(file_path.to_string()).or_default();

        // Prune old entries outside the window
        let cutoff = now - chrono::Duration::seconds(self.window_secs);
        edits.retain(|&t| t > cutoff);
        edits.push(now);

        edits.len() >= self.threshold
    }
}

/// Tracks symbol exploration to detect dead-end patterns
/// (something added then immediately removed in the same session).
struct DeadEndTracker {
    /// symbol_fqn → (added_at, removed_at)
    #[allow(clippy::type_complexity)]
    events: std::collections::HashMap<
        String,
        (
            Option<chrono::DateTime<chrono::Utc>>,
            Option<chrono::DateTime<chrono::Utc>>,
        ),
    >,
}

impl DeadEndTracker {
    fn new() -> Self {
        DeadEndTracker {
            events: std::collections::HashMap::new(),
        }
    }

    fn record_add(&mut self, fqn: &str) {
        self.events.entry(fqn.to_string()).or_default().0 = Some(chrono::Utc::now());
    }

    fn record_remove(&mut self, fqn: &str) -> bool {
        let entry = self.events.entry(fqn.to_string()).or_default();
        if entry.0.is_some() {
            entry.1 = Some(chrono::Utc::now());
            return true; // added and then removed → dead end
        }
        false
    }
}

// ── MemoryStore ────────────────────────────────────────────────────────────────

pub struct MemoryStore {
    db: Arc<Mutex<Database>>,
    session_id: String,
    agent_id: Option<String>,
    thrash_tracker: FileThrashTracker,
    dead_end_tracker: DeadEndTracker,
    config: MemoryConfig,
}

impl MemoryStore {
    pub fn new(db: Arc<Mutex<Database>>, session_id: &str) -> Self {
        MemoryStore {
            db,
            session_id: session_id.to_string(),
            agent_id: None,
            thrash_tracker: FileThrashTracker::new(),
            dead_end_tracker: DeadEndTracker::new(),
            config: MemoryConfig::default(),
        }
    }

    pub fn with_agent(mut self, agent_id: &str) -> Self {
        self.agent_id = Some(agent_id.to_string());
        self
    }

    pub fn with_config(mut self, config: MemoryConfig) -> Self {
        self.config = config;
        self
    }

    pub fn config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Save a manual observation (from agent or user).
    pub fn save(
        &self,
        content: &str,
        symbol_fqn: Option<&str>,
        symbol_hash: Option<&str>,
    ) -> Result<()> {
        let mut obs = Observation::new(
            &self.session_id,
            content,
            symbol_fqn,
            symbol_hash,
            ObservationKind::Manual,
        )
        .with_ttl(&self.config);
        if let Some(ref aid) = self.agent_id {
            obs = obs.with_agent(aid);
        }
        self.db.lock().insert_observation(&obs)?;
        Ok(())
    }

    /// Record a file edit passively. Auto-generates an observation if thrashing detected.
    ///
    /// `changes` is the classified diff produced by `reindex_file`. When non-empty the
    /// observation content is a structured breakdown; when empty a generic fallback is used.
    pub fn record_file_edit(&mut self, file_path: &str, changes: &[SymbolChange]) -> Result<()> {
        let (content, change_category) = if changes.is_empty() {
            (format!("File changed: {file_path}"), None)
        } else {
            // Priority order for the single top-level category tag.
            const PRIORITY: &[&str] = &[
                "new_symbol",
                "deleted_symbol",
                "signature_change",
                "dependency_added",
                "body_change",
            ];

            // Group FQNs by category.
            let mut by_cat: std::collections::HashMap<&str, Vec<&str>> =
                std::collections::HashMap::new();
            for sc in changes {
                by_cat.entry(sc.category).or_default().push(&sc.fqn);
            }

            // Build a human-readable summary, categories in priority order.
            let mut parts: Vec<String> = Vec::new();
            for &cat in PRIORITY {
                if let Some(fqns) = by_cat.get(cat) {
                    let listed = if fqns.len() <= 3 {
                        fqns.join(", ")
                    } else {
                        format!("{}, … ({} total)", fqns[..2].join(", "), fqns.len())
                    };
                    parts.push(format!("{cat}: {listed}"));
                }
            }

            let top_cat = PRIORITY
                .iter()
                .find(|&&c| by_cat.contains_key(c))
                .copied()
                .unwrap_or("body_change");

            (
                format!("File changed: {file_path} — {}", parts.join("; ")),
                Some(top_cat),
            )
        };

        let mut obs = Observation::new(
            &self.session_id,
            &content,
            None,
            None,
            ObservationKind::Passive,
        )
        .with_ttl(&self.config);
        if let Some(cat) = change_category {
            obs = obs.with_change_category(cat);
        }
        self.db.lock().insert_observation(&obs)?;

        // Check for file thrashing
        if self.thrash_tracker.record_edit(file_path) {
            let thrash_obs = Observation::new(
                &self.session_id,
                &format!(
                    "⚠️ File thrash detected: `{}` has been modified {} times in the last 5 minutes. \
                     Consider breaking the change into smaller steps.",
                    file_path, self.thrash_tracker.threshold
                ),
                None,
                None,
                ObservationKind::FileThrash,
            )
            .with_ttl(&self.config);
            self.db.lock().insert_observation(&thrash_obs)?;
            tracing::warn!("File thrash detected: {}", file_path);
        }

        Ok(())
    }

    /// Record that a symbol was added (for dead-end detection).
    pub fn record_symbol_add(&mut self, fqn: &str) {
        self.dead_end_tracker.record_add(fqn);
    }

    /// Record that a symbol was removed. If it was added this session → dead end.
    pub fn record_symbol_remove(&mut self, fqn: &str) -> Result<()> {
        if self.dead_end_tracker.record_remove(fqn) {
            let obs = Observation::new(
                &self.session_id,
                &format!(
                    "⚠️ Dead-end detected: `{}` was added and then removed in the same session. \
                     Previous approach may have been incorrect.",
                    fqn
                ),
                Some(fqn),
                None,
                ObservationKind::DeadEnd,
            )
            .with_ttl(&self.config);
            self.db.lock().insert_observation(&obs)?;
            tracing::warn!("Dead-end exploration detected for: {}", fqn);
        }
        Ok(())
    }

    /// Auto-capture a tool call as an observation.
    /// Skips when `pivot_fqns` is empty or an identical task was recorded within 30 minutes.
    pub fn record_auto_observation(&self, task: &str, pivot_fqns: &[String]) -> Result<()> {
        if pivot_fqns.is_empty() {
            return Ok(());
        }

        let top_fqns = pivot_fqns
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let content = format!("Agent queried: \"{task}\" — pivots: {top_fqns}");

        // Deduplicate: skip if same task recorded in last 30 minutes
        if self.db.lock().has_recent_auto_observation(task, 30)? {
            return Ok(());
        }

        let mut obs = Observation::new(
            &self.session_id,
            &content,
            None,
            None,
            ObservationKind::Auto,
        )
        .with_ttl(&self.config);
        if let Some(ref aid) = self.agent_id {
            obs = obs.with_agent(aid);
        }
        self.db.lock().insert_observation(&obs)?;
        Ok(())
    }

    /// Delete an observation by ID. Returns true if a row was deleted.
    pub fn delete(&self, id: &str) -> Result<bool> {
        self.db.lock().delete_observation(id)
    }

    /// Mark observations stale when their linked symbol's code has changed.
    pub fn check_and_mark_stale(&self, symbol_fqn: &str, new_hash: &str) -> Result<u64> {
        self.db
            .lock()
            .mark_stale_by_symbol_hash(symbol_fqn, new_hash)
    }

    /// Delete all observations whose TTL has elapsed.
    pub fn prune_expired(&self) -> Result<u64> {
        self.db.lock().prune_expired_observations()
    }

    /// Compression pass: for each symbol with ≥ 3 non-expired, non-summary observations,
    /// create one `Summary` entry (keeping the most recent wording) and expire the originals.
    /// Returns the number of symbols compressed.
    pub fn compress_observations(&self) -> Result<usize> {
        const COMPRESSION_THRESHOLD: usize = 3;
        let fqns = self
            .db
            .lock()
            .fqns_needing_compression(COMPRESSION_THRESHOLD)?;
        let count = fqns.len();
        for fqn in &fqns {
            let obs = self.db.lock().get_observations_for_fqn(fqn)?;
            // Keep only non-summary entries for compression
            let to_compress: Vec<_> = obs
                .iter()
                .filter(|o| !matches!(o.kind, ObservationKind::Summary))
                .collect();
            if to_compress.len() < COMPRESSION_THRESHOLD {
                continue;
            }
            // Use the most recent observation's content as the summary wording
            let most_recent = to_compress.last().expect("checked len");
            let summary_content = format!(
                "[summary of {} observations] {}",
                to_compress.len(),
                most_recent.content
            );
            let summary = Observation::new(
                &self.session_id,
                &summary_content,
                Some(fqn),
                most_recent.symbol_hash.as_deref(),
                ObservationKind::Summary,
            )
            .with_ttl(&self.config);

            let ids_to_expire: Vec<String> = to_compress.iter().map(|o| o.id.clone()).collect();
            {
                let db = self.db.lock();
                db.insert_observation(&summary)?;
                db.expire_observations(&ids_to_expire)?;
            }
        }
        Ok(count)
    }

    /// Retrieve observations for the current session.
    pub fn get_session_observations(&self) -> Result<Vec<Observation>> {
        self.db.lock().get_session_observations(&self.session_id)
    }

    /// Retrieve recent observations across all sessions (for cross-session context).
    pub fn get_recent_observations(&self, limit: usize) -> Result<Vec<Observation>> {
        self.db.lock().get_recent_observations(limit)
    }

    /// Keyword search over observation content and symbol FQN.
    pub fn search_observations(&self, query: &str, max_results: usize) -> Result<Vec<Observation>> {
        self.db.lock().search_observations(query, max_results)
    }

    /// Returns the fraction (0.0–100.0) of non-expired observations that are stale.
    pub fn staleness_score(&self) -> Result<f32> {
        let (stale, total) = self.db.lock().observation_staleness_counts()?;
        if total == 0 {
            return Ok(0.0);
        }
        Ok(stale as f32 / total as f32 * 100.0)
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

#[cfg(test)]
mod obs_kind_tests {
    use super::*;

    #[test]
    fn consolidated_as_str() {
        assert_eq!(ObservationKind::Consolidated.as_str(), "consolidated");
    }

    #[test]
    fn consolidated_parse_kind_roundtrip() {
        assert!(matches!(
            ObservationKind::parse_kind("consolidated"),
            ObservationKind::Consolidated
        ));
    }

    #[test]
    fn consolidated_default_ttl_is_90_days() {
        assert_eq!(ObservationKind::Consolidated.default_ttl_days(), Some(90));
    }

    #[test]
    fn consolidated_expires_at_is_set_on_new() {
        let obs = Observation::new("s", "content", None, None, ObservationKind::Consolidated);
        assert!(
            obs.expires_at.is_some(),
            "Consolidated must have an expires_at"
        );
    }

    /// expires_at() on MemoryConfig should assign a non-None TTL for Consolidated.
    #[test]
    fn memory_config_expires_at_for_consolidated() {
        let cfg = MemoryConfig::default();
        assert!(
            cfg.expires_at(&ObservationKind::Consolidated).is_some(),
            "MemoryConfig::expires_at must return Some for Consolidated"
        );
    }
}

#[cfg(test)]
mod config_load_tests {
    use super::*;

    #[test]
    fn load_from_toml_missing_file_returns_defaults() {
        let cfg = MemoryConfig::load_from_toml(std::path::Path::new("/nonexistent/config.toml"));
        assert_eq!(cfg.auto_ttl_days, 7);
        assert!(cfg.manual_ttl_days.is_none());
    }

    #[test]
    fn load_from_toml_valid_config_applies_settings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[memory]\nauto_ttl_days = 14\nmanual_ttl_days = 30\n",
        )
        .unwrap();
        let cfg = MemoryConfig::load_from_toml(&path);
        assert_eq!(cfg.auto_ttl_days, 14);
        assert_eq!(cfg.manual_ttl_days, Some(30));
    }

    #[test]
    fn load_from_toml_malformed_returns_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[memory\n  this is not valid toml").unwrap();
        let cfg = MemoryConfig::load_from_toml(&path);
        // Should return defaults, not panic.
        assert_eq!(cfg.auto_ttl_days, 7);
    }

    #[test]
    fn load_from_toml_missing_memory_section_returns_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[indexing]\nts_types = true\n").unwrap();
        let cfg = MemoryConfig::load_from_toml(&path);
        assert_eq!(cfg.auto_ttl_days, 7);
        assert!(cfg.manual_ttl_days.is_none());
    }
}

#[cfg(test)]
mod indexing_config_tests {
    use super::*;

    #[test]
    fn context_section_parsed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[context]\nmax_tokens = 8000\nskeleton_detail = \"detailed\"\n",
        )
        .unwrap();
        let cfg = IndexingConfig::load_from_toml(&path);
        assert_eq!(cfg.max_tokens, Some(8000));
        assert_eq!(cfg.skeleton_detail.as_deref(), Some("detailed"));
    }

    #[test]
    fn observability_section_parsed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[observability]\ntoken_rate_usd = 0.00001\n").unwrap();
        let cfg = IndexingConfig::load_from_toml(&path);
        assert!((cfg.token_rate_usd.unwrap() - 0.00001).abs() < 1e-10);
    }

    #[test]
    fn missing_context_section_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[indexing]\nts_types = true\n").unwrap();
        let cfg = IndexingConfig::load_from_toml(&path);
        assert!(cfg.max_tokens.is_none());
        assert!(cfg.skeleton_detail.is_none());
        assert!(cfg.token_rate_usd.is_none());
    }

    #[test]
    fn user_fallback_workspace_takes_precedence() {
        // This test only verifies the merge logic — actual user config path
        // is not written to avoid polluting the real home directory.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[context]\nmax_tokens = 6000\nskeleton_detail = \"minimal\"\n",
        )
        .unwrap();
        let cfg = IndexingConfig::load_with_user_fallback(&path);
        assert_eq!(cfg.max_tokens, Some(6000));
        assert_eq!(cfg.skeleton_detail.as_deref(), Some("minimal"));
    }
}

/// Generate a new session ID.
pub fn new_session_id() -> String {
    format!(
        "session-{}-{}",
        chrono::Utc::now().format("%Y%m%d-%H%M%S"),
        &Uuid::new_v4().to_string()[..8]
    )
}
