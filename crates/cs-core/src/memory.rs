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
        }
        cfg
    }
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
        let Ok(text) = std::fs::read_to_string(path) else {
            return cfg;
        };
        let Ok(table) = text.parse::<toml::Table>() else {
            return cfg;
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
            // Summaries get the manual TTL (or 90 days if manual never expires)
            ObservationKind::Summary => self.manual_ttl_days.or(Some(90)),
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
        }
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
    pub fn record_file_edit(&mut self, file_path: &str, change_summary: &str) -> Result<()> {
        // Passive observation about the change
        let obs = Observation::new(
            &self.session_id,
            &format!("File changed: {} — {}", file_path, change_summary),
            None,
            None,
            ObservationKind::Passive,
        )
        .with_ttl(&self.config);
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

/// Generate a new session ID.
pub fn new_session_id() -> String {
    format!(
        "session-{}-{}",
        chrono::Utc::now().format("%Y%m%d-%H%M%S"),
        &Uuid::new_v4().to_string()[..8]
    )
}
