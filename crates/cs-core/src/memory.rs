use crate::db::Database;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

// ── Observation ───────────────────────────────────────────────────────────────

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
}

impl ObservationKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ObservationKind::Manual => "manual",
            ObservationKind::Passive => "passive",
            ObservationKind::DeadEnd => "dead_end",
            ObservationKind::FileThrash => "file_thrash",
            ObservationKind::Insight => "insight",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "passive" => ObservationKind::Passive,
            "dead_end" => ObservationKind::DeadEnd,
            "file_thrash" => ObservationKind::FileThrash,
            "insight" => ObservationKind::Insight,
            _ => ObservationKind::Manual,
        }
    }
}

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
}

impl Observation {
    pub fn new(
        session_id: &str,
        content: &str,
        symbol_fqn: Option<&str>,
        symbol_hash: Option<&str>,
        kind: ObservationKind,
    ) -> Self {
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
        }
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
}

impl MemoryStore {
    pub fn new(db: Arc<Mutex<Database>>, session_id: &str) -> Self {
        MemoryStore {
            db,
            session_id: session_id.to_string(),
            agent_id: None,
            thrash_tracker: FileThrashTracker::new(),
            dead_end_tracker: DeadEndTracker::new(),
        }
    }

    pub fn with_agent(mut self, agent_id: &str) -> Self {
        self.agent_id = Some(agent_id.to_string());
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
        );
        if let Some(ref aid) = self.agent_id {
            obs = obs.with_agent(aid);
        }
        self.db.lock().unwrap().insert_observation(&obs)?;
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
        );
        self.db.lock().unwrap().insert_observation(&obs)?;

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
            );
            self.db.lock().unwrap().insert_observation(&thrash_obs)?;
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
            );
            self.db.lock().unwrap().insert_observation(&obs)?;
            tracing::warn!("Dead-end exploration detected for: {}", fqn);
        }
        Ok(())
    }

    /// Mark observations stale when their linked symbol's code has changed.
    pub fn check_and_mark_stale(&self, symbol_fqn: &str, new_hash: &str) -> Result<u64> {
        self.db
            .lock()
            .unwrap()
            .mark_stale_by_symbol_hash(symbol_fqn, new_hash)
    }

    /// Retrieve observations for the current session.
    pub fn get_session_observations(&self) -> Result<Vec<Observation>> {
        self.db
            .lock()
            .unwrap()
            .get_session_observations(&self.session_id)
    }

    /// Retrieve recent observations across all sessions (for cross-session context).
    pub fn get_recent_observations(&self, limit: usize) -> Result<Vec<Observation>> {
        self.db.lock().unwrap().get_recent_observations(limit)
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
