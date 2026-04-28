use crate::language::Language;
use crate::memory::Observation;
use crate::symbol::{Edge, EdgeKind, Symbol, SymbolKind};
use anyhow::Result;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::Path;

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; \
             PRAGMA synchronous=NORMAL; \
             PRAGMA busy_timeout=5000; \
             PRAGMA cache_size=-8192;",
        )?;
        let db = Database { conn };
        db.create_schema()?;
        Ok(db)
    }

    fn create_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS symbols (
                id            INTEGER PRIMARY KEY,
                fqn           TEXT NOT NULL,
                name          TEXT NOT NULL,
                kind          TEXT NOT NULL,
                file_path     TEXT NOT NULL,
                start_line    INTEGER NOT NULL,
                end_line      INTEGER NOT NULL,
                signature     TEXT NOT NULL,
                docstring     TEXT,
                body          TEXT NOT NULL,
                language      TEXT NOT NULL,
                content_hash  TEXT NOT NULL,
                is_stub       INTEGER NOT NULL DEFAULT 0,
                source        TEXT,
                resolved_type TEXT
            );

            CREATE TABLE IF NOT EXISTS macro_expand_cache (
                file_path    TEXT PRIMARY KEY,
                source_hash  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_fqn  ON symbols(fqn);

            -- FTS5 for fast text search across names and signatures
            CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                name, fqn, signature, docstring, content='symbols', content_rowid='id'
            );

            CREATE TABLE IF NOT EXISTS edges (
                from_id  INTEGER NOT NULL,
                to_id    INTEGER NOT NULL,
                kind     TEXT NOT NULL,
                label    TEXT,
                PRIMARY KEY (from_id, to_id, kind)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_id);

            CREATE TABLE IF NOT EXISTS files (
                path          TEXT PRIMARY KEY,
                content_hash  TEXT NOT NULL,
                indexed_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id            TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                agent_id      TEXT,
                content       TEXT NOT NULL,
                symbol_fqn    TEXT,
                symbol_hash   TEXT,
                created_at    TEXT NOT NULL,
                is_stale      INTEGER NOT NULL DEFAULT 0,
                kind          TEXT NOT NULL DEFAULT 'manual',
                expires_at    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_obs_session  ON observations(session_id);
            CREATE INDEX IF NOT EXISTS idx_obs_symbol   ON observations(symbol_fqn);

            -- Optional: stored embedding vectors (packed f32 LE bytes, 384-dim)
            -- Populated only when built with --features embeddings
            CREATE TABLE IF NOT EXISTS symbol_embeddings (
                symbol_id  INTEGER PRIMARY KEY,
                embedding  BLOB NOT NULL
            );
        "#,
        )?;
        // Idempotent migrations — fail silently with "duplicate column name" on fresh databases.
        let _ = self.conn.execute(
            "ALTER TABLE symbols ADD COLUMN is_stub INTEGER NOT NULL DEFAULT 0",
            [],
        );
        let _ = self
            .conn
            .execute("ALTER TABLE symbols ADD COLUMN source TEXT", []);
        let _ = self
            .conn
            .execute("ALTER TABLE symbols ADD COLUMN resolved_type TEXT", []);
        // `leaf_name` — last `::`-segment of the qualified `name`. Issue #96:
        // class methods are stored with `name = "Class::method"` so a
        // direct lookup by leaf identifier (e.g. `name = "method"`) misses
        // them. The leaf column lets `symbols_by_leaf_name` find them
        // with a single indexed lookup.
        let _ = self
            .conn
            .execute("ALTER TABLE symbols ADD COLUMN leaf_name TEXT", []);
        // Index whether the column was just added or already existed.
        let _ = self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_symbols_leaf_name ON symbols(leaf_name);",
        );
        // Populate leaf_name for rows that pre-date this migration.
        // SQLite has no native `rsplit`, so compute in Rust and write back.
        // One-shot at startup; idempotent because we only touch NULL rows.
        let pending: Vec<(i64, String)> = {
            let mut stmt = self
                .conn
                .prepare("SELECT id, name FROM symbols WHERE leaf_name IS NULL")?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
            rows.flatten().collect()
        };
        if !pending.is_empty() {
            self.conn.execute_batch("BEGIN")?;
            {
                let mut update = self
                    .conn
                    .prepare("UPDATE symbols SET leaf_name = ?1 WHERE id = ?2")?;
                for (id, name) in &pending {
                    let leaf = leaf_of_name(name);
                    let _ = update.execute(params![leaf, id]);
                }
            }
            self.conn.execute_batch("COMMIT")?;
        }
        let _ = self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS macro_expand_cache \
             (file_path TEXT PRIMARY KEY, source_hash TEXT NOT NULL);",
        );
        let _ = self
            .conn
            .execute("ALTER TABLE observations ADD COLUMN expires_at TEXT", []);
        let _ = self.conn.execute(
            "ALTER TABLE observations ADD COLUMN change_category TEXT",
            [],
        );
        // Index may already exist on new databases; ignore error.
        let _ = self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_obs_expires ON observations(expires_at);",
        );
        // LSP edges — pushed by IDE extensions or Claude Code hooks.
        let _ = self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS lsp_edges ( \
                 from_fqn      TEXT NOT NULL, \
                 to_fqn        TEXT NOT NULL, \
                 kind          TEXT NOT NULL, \
                 resolved_type TEXT, \
                 source_file   TEXT NOT NULL, \
                 PRIMARY KEY (from_fqn, to_fqn, kind) \
             ); \
             CREATE INDEX IF NOT EXISTS idx_lsp_edges_source ON lsp_edges(source_file);",
        );
        // Query log — per-call metrics for run_pipeline.
        let _ = self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS query_log ( \
                 id                    INTEGER PRIMARY KEY, \
                 timestamp             TEXT NOT NULL, \
                 task                  TEXT NOT NULL, \
                 intent                TEXT NOT NULL, \
                 pivot_count           INTEGER NOT NULL, \
                 total_tokens          INTEGER NOT NULL, \
                 candidate_file_tokens INTEGER NOT NULL, \
                 latency_ms            INTEGER NOT NULL, \
                 languages_hit         TEXT NOT NULL \
             ); \
             CREATE INDEX IF NOT EXISTS idx_qlog_ts ON query_log(timestamp);",
        );
        Ok(())
    }

    // ── Transactions ──────────────────────────────────────────────────────────

    pub fn begin_transaction(&self) -> Result<()> {
        self.conn.execute_batch("BEGIN")?;
        Ok(())
    }

    pub fn commit_transaction(&self) -> Result<()> {
        self.conn.execute_batch("COMMIT")?;
        Ok(())
    }

    // ── Symbols ───────────────────────────────────────────────────────────────

    pub fn upsert_symbol(&self, sym: &Symbol) -> Result<()> {
        self.conn.execute(
            r#"INSERT OR REPLACE INTO symbols
               (id, fqn, name, kind, file_path, start_line, end_line,
                signature, docstring, body, language, content_hash, is_stub, source,
                resolved_type, leaf_name)
               VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)"#,
            params![
                sym.id as i64,
                sym.fqn,
                sym.name,
                sym.kind.to_db_str(),
                sym.file_path,
                sym.start_line,
                sym.end_line,
                sym.signature,
                sym.docstring,
                sym.body,
                sym.language.as_str(),
                sym.content_hash,
                sym.is_stub as i64,
                sym.source,
                sym.resolved_type,
                leaf_of_name(&sym.name),
            ],
        )?;
        // Keep FTS in sync
        self.conn.execute(
            r#"INSERT OR REPLACE INTO symbols_fts(rowid, name, fqn, signature, docstring)
               VALUES (?1, ?2, ?3, ?4, ?5)"#,
            params![
                sym.id as i64,
                sym.name,
                sym.fqn,
                sym.signature,
                sym.docstring.as_deref().unwrap_or(""),
            ],
        )?;
        Ok(())
    }

    pub fn delete_file_symbols(&self, file_path: &str) -> Result<()> {
        // Remove from FTS first
        let ids: Vec<i64> = {
            let mut stmt = self
                .conn
                .prepare("SELECT id FROM symbols WHERE file_path = ?1")?;
            let ids = stmt
                .query_map(params![file_path], |row| row.get(0))?
                .filter_map(|r| r.ok())
                .collect();
            ids
        };
        for id in &ids {
            self.conn
                .execute("DELETE FROM symbols_fts WHERE rowid = ?1", params![id])?;
        }
        self.conn.execute(
            "DELETE FROM symbols WHERE file_path = ?1",
            params![file_path],
        )?;
        Ok(())
    }

    /// Return all symbol IDs belonging to a file (for cleanup in Tantivy/embeddings).
    pub fn symbol_ids_for_file(&self, file_path: &str) -> Result<Vec<u64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM symbols WHERE file_path = ?1")?;
        let ids = stmt
            .query_map(params![file_path], |row| row.get::<_, i64>(0))?
            .filter_map(|r| r.ok())
            .map(|id| id as u64)
            .collect();
        Ok(ids)
    }

    /// Look up symbol IDs by exact name (not FQN). Used for explicit-anchor
    /// retrieval where we want precise symbol-name matches rather than BM25
    /// tokenized scoring. Returns at most `limit` rows.
    pub fn symbols_by_exact_name(&self, name: &str, limit: usize) -> Result<Vec<u64>> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT id FROM symbols WHERE name = ?1 LIMIT ?2")?;
        let ids = stmt
            .query_map(params![name, limit as i64], |row| row.get::<_, i64>(0))?
            .filter_map(|r| r.ok())
            .map(|id| id as u64)
            .collect();
        Ok(ids)
    }

    /// Match by the *leaf* of the qualified `name` — the segment after the
    /// last `::`. Class methods are indexed with `name = "Class::method"`
    /// so `symbols_by_exact_name("method")` misses them; this lookup
    /// catches both module-level functions (where leaf == name) and
    /// class methods (where leaf is the trailing segment). Issue #96.
    pub fn symbols_by_leaf_name(&self, leaf: &str, limit: usize) -> Result<Vec<u64>> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT id FROM symbols WHERE leaf_name = ?1 LIMIT ?2")?;
        let ids = stmt
            .query_map(params![leaf, limit as i64], |row| row.get::<_, i64>(0))?
            .filter_map(|r| r.ok())
            .map(|id| id as u64)
            .collect();
        Ok(ids)
    }

    /// Delete all edges referencing any of the given symbol IDs.
    ///
    /// Uses a single batched DELETE with `IN (...)` instead of one statement per
    /// ID — on a file with 200 symbols this is ~200x fewer round-trips.
    pub fn delete_edges_for_symbols(&self, symbol_ids: &[u64]) -> Result<()> {
        if symbol_ids.is_empty() {
            return Ok(());
        }
        // SQLite allows up to 999 host parameters by default; chunk to stay safe.
        for chunk in symbol_ids.chunks(500) {
            let placeholders: String = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "DELETE FROM edges WHERE from_id IN ({p}) OR to_id IN ({p})",
                p = placeholders
            );
            let params: Vec<rusqlite::types::Value> = chunk
                .iter()
                .map(|id| rusqlite::types::Value::Integer(*id as i64))
                .collect();
            // Duplicate params for both IN clauses.
            let mut all_params: Vec<rusqlite::types::Value> = params.clone();
            all_params.extend(params);
            self.conn
                .execute(&sql, rusqlite::params_from_iter(all_params))?;
        }
        Ok(())
    }

    /// Delete embeddings for the given symbol IDs.
    ///
    /// Batched into a single `DELETE ... WHERE symbol_id IN (...)` per chunk.
    pub fn delete_embeddings_for_symbols(&self, symbol_ids: &[u64]) -> Result<()> {
        if symbol_ids.is_empty() {
            return Ok(());
        }
        for chunk in symbol_ids.chunks(500) {
            let placeholders: String = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "DELETE FROM symbol_embeddings WHERE symbol_id IN ({})",
                placeholders
            );
            let params: Vec<rusqlite::types::Value> = chunk
                .iter()
                .map(|id| rusqlite::types::Value::Integer(*id as i64))
                .collect();
            self.conn
                .execute(&sql, rusqlite::params_from_iter(params))?;
        }
        Ok(())
    }

    /// Remove a file entry from the files table.
    pub fn delete_file(&self, file_path: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM files WHERE path = ?1", params![file_path])?;
        Ok(())
    }

    /// Return all file paths tracked in the files table.
    pub fn all_file_paths(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files")?;
        let paths = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(paths)
    }

    pub fn get_symbol(&self, id: u64) -> Result<Option<Symbol>> {
        let mut stmt = self.conn.prepare(
            "SELECT id,fqn,name,kind,file_path,start_line,end_line,\
             signature,docstring,body,language,content_hash,is_stub,source,resolved_type \
             FROM symbols WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id as i64], row_to_symbol)?;
        Ok(rows.next().transpose()?)
    }

    /// FTS5 full-text search. Returns (symbol_id, rank) pairs.
    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<(u64, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT rowid, rank FROM symbols_fts WHERE symbols_fts MATCH ?1 \
             ORDER BY rank LIMIT ?2",
        )?;
        let results = stmt
            .query_map(params![query, limit as i64], |row| {
                Ok((row.get::<_, i64>(0)? as u64, row.get::<_, f64>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    pub fn symbol_count(&self) -> Result<u64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM symbols", [], |r| r.get::<_, i64>(0))?
            as u64)
    }

    pub fn stub_symbol_count(&self) -> Result<u64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM symbols WHERE is_stub = 1", [], |r| {
                r.get::<_, i64>(0)
            })? as u64)
    }

    // ── Edges ─────────────────────────────────────────────────────────────────

    pub fn upsert_edge(&self, edge: &Edge) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (from_id, to_id, kind, label) VALUES (?1,?2,?3,?4)",
            params![
                edge.from_id as i64,
                edge.to_id as i64,
                edge.kind.to_db_str(),
                edge.label.as_deref(),
            ],
        )?;
        Ok(())
    }

    pub fn edge_count(&self) -> Result<u64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get::<_, i64>(0))? as u64)
    }

    // ── LSP edges ─────────────────────────────────────────────────────────────

    pub fn upsert_lsp_edge(
        &self,
        from_fqn: &str,
        to_fqn: &str,
        kind: &str,
        resolved_type: Option<&str>,
        source_file: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO lsp_edges \
             (from_fqn, to_fqn, kind, resolved_type, source_file) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![from_fqn, to_fqn, kind, resolved_type, source_file],
        )?;
        Ok(())
    }

    pub fn delete_lsp_edges_for_file(&self, source_file: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM lsp_edges WHERE source_file = ?1",
            params![source_file],
        )?;
        Ok(())
    }

    pub fn load_lsp_edges(&self) -> Result<Vec<crate::symbol::LspEdge>> {
        let mut stmt = self
            .conn
            .prepare("SELECT from_fqn, to_fqn, kind, resolved_type FROM lsp_edges")?;
        let rows = stmt.query_map([], |r| {
            Ok(crate::symbol::LspEdge {
                from_fqn: r.get(0)?,
                to_fqn: r.get(1)?,
                kind: r.get(2)?,
                resolved_type: r.get(3)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn lsp_edge_count(&self) -> Result<u64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM lsp_edges", [], |r| r.get::<_, i64>(0))?
            as u64)
    }

    // ── Files ─────────────────────────────────────────────────────────────────

    pub fn upsert_file(&self, path: &str, content_hash: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO files (path, content_hash, indexed_at) \
             VALUES (?1, ?2, datetime('now'))",
            params![path, content_hash],
        )?;
        Ok(())
    }

    pub fn get_file_hash(&self, path: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT content_hash FROM files WHERE path = ?1")?;
        let mut rows = stmt.query_map(params![path], |r| r.get::<_, String>(0))?;
        Ok(rows.next().transpose()?)
    }

    pub fn file_count(&self) -> Result<u64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get::<_, i64>(0))? as u64)
    }

    /// Return all (path, content_hash) pairs from the files table.
    pub fn all_file_hashes(&self) -> Result<HashMap<String, String>> {
        let mut stmt = self.conn.prepare("SELECT path, content_hash FROM files")?;
        let map = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(map)
    }

    // ── Embeddings ────────────────────────────────────────────────────────────

    /// Store a 384-dim embedding for a symbol (packed as little-endian f32 bytes).
    pub fn upsert_embedding(&self, symbol_id: u64, embedding: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT OR REPLACE INTO symbol_embeddings (symbol_id, embedding) VALUES (?1, ?2)",
            params![symbol_id as i64, bytes],
        )?;
        Ok(())
    }

    /// Load all stored embeddings. Returns `(symbol_id, embedding_vec)` pairs.
    pub fn all_embeddings(&self) -> Result<Vec<(u64, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT symbol_id, embedding FROM symbol_embeddings")?;
        let rows: Vec<(u64, Vec<f32>)> = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let bytes: Vec<u8> = row.get(1)?;
                Ok((id as u64, bytes))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, bytes)| {
                let floats = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                (id, floats)
            })
            .collect();
        Ok(rows)
    }

    // ── Observations ──────────────────────────────────────────────────────────

    pub fn insert_observation(&self, obs: &Observation) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO observations \
             (id, session_id, agent_id, content, symbol_fqn, symbol_hash, created_at, is_stale, kind, expires_at, change_category) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![
                obs.id,
                obs.session_id,
                obs.agent_id,
                obs.content,
                obs.symbol_fqn,
                obs.symbol_hash,
                obs.created_at,
                obs.is_stale as i64,
                obs.kind.as_str(),
                obs.expires_at,
                obs.change_category,
            ],
        )?;
        Ok(())
    }

    pub fn get_session_observations(&self, session_id: &str) -> Result<Vec<Observation>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, agent_id, content, symbol_fqn, symbol_hash, \
             created_at, is_stale, kind, expires_at, change_category \
             FROM observations \
             WHERE session_id = ?1 \
               AND (expires_at IS NULL OR datetime(expires_at) > datetime('now')) \
             ORDER BY created_at ASC",
        )?;
        let results = stmt
            .query_map(params![session_id], row_to_observation)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    pub fn get_recent_observations(&self, limit: usize) -> Result<Vec<Observation>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, agent_id, content, symbol_fqn, symbol_hash, \
             created_at, is_stale, kind, expires_at, change_category \
             FROM observations \
             WHERE (expires_at IS NULL OR datetime(expires_at) > datetime('now')) \
             ORDER BY created_at DESC LIMIT ?1",
        )?;
        let results = stmt
            .query_map(params![limit as i64], row_to_observation)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Returns true if an auto-observation with the same task was recorded within `window_mins`.
    pub fn has_recent_auto_observation(&self, task: &str, window_mins: i64) -> Result<bool> {
        let cutoff = chrono::Utc::now() - chrono::Duration::minutes(window_mins);
        let pattern = format!("Agent queried: \"{task}\" —%");
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM observations \
             WHERE kind = 'auto' AND content LIKE ?1 AND created_at > ?2",
            params![pattern, cutoff.to_rfc3339()],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    pub fn delete_observation(&self, id: &str) -> Result<bool> {
        let count = self
            .conn
            .execute("DELETE FROM observations WHERE id = ?1", params![id])?;
        Ok(count > 0)
    }

    /// Mark observations stale when the code they refer to has changed.
    pub fn mark_stale_by_symbol_hash(&self, symbol_fqn: &str, new_hash: &str) -> Result<u64> {
        let count = self.conn.execute(
            "UPDATE observations SET is_stale = 1 \
             WHERE symbol_fqn = ?1 AND symbol_hash != ?2 AND is_stale = 0",
            params![symbol_fqn, new_hash],
        )? as u64;
        Ok(count)
    }

    /// Delete all observations whose `expires_at` is in the past.
    pub fn prune_expired_observations(&self) -> Result<u64> {
        let count = self.conn.execute(
            "DELETE FROM observations WHERE expires_at IS NOT NULL AND datetime(expires_at) <= datetime('now')",
            [],
        )? as u64;
        Ok(count)
    }

    /// Return all non-expired observations for a specific symbol FQN, oldest first.
    pub fn get_observations_for_fqn(&self, fqn: &str) -> Result<Vec<Observation>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, agent_id, content, symbol_fqn, symbol_hash, \
             created_at, is_stale, kind, expires_at, change_category \
             FROM observations \
             WHERE symbol_fqn = ?1 \
               AND (expires_at IS NULL OR datetime(expires_at) > datetime('now')) \
             ORDER BY created_at ASC",
        )?;
        let results = stmt
            .query_map(params![fqn], row_to_observation)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Immediately expire a set of observations by setting their `expires_at` to now.
    /// Used by the compression pass to retire originals after creating a Summary.
    pub fn expire_observations(&self, ids: &[String]) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        for id in ids {
            self.conn.execute(
                "UPDATE observations SET expires_at = ?1 WHERE id = ?2",
                params![now, id],
            )?;
        }
        Ok(())
    }

    /// Returns all distinct symbol FQNs that have ≥ `threshold` non-expired,
    /// non-summary/consolidated observations. Used to find candidates for compression.
    pub fn fqns_needing_compression(&self, threshold: usize) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT symbol_fqn, COUNT(*) as cnt \
             FROM observations \
             WHERE symbol_fqn IS NOT NULL \
               AND kind NOT IN ('summary', 'consolidated') \
               AND (expires_at IS NULL OR datetime(expires_at) > datetime('now')) \
             GROUP BY symbol_fqn \
             HAVING cnt >= ?1",
        )?;
        let results = stmt
            .query_map(params![threshold as i64], |row| row.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Returns all non-expired auto/passive observations that are not already summary or
    /// consolidated. These are the candidates for semantic consolidation by the engine.
    pub fn get_consolidation_candidates(&self) -> Result<Vec<Observation>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, agent_id, content, symbol_fqn, symbol_hash, \
             created_at, is_stale, kind, expires_at, change_category \
             FROM observations \
             WHERE kind IN ('auto', 'passive') \
               AND (expires_at IS NULL OR datetime(expires_at) > datetime('now')) \
             ORDER BY created_at ASC",
        )?;
        let results = stmt
            .query_map([], row_to_observation)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Multi-term keyword search over observation content and symbol_fqn.
    /// Splits `query` on whitespace, requires at least one term to match (OR logic),
    /// and ranks results by how many terms match (descending), then recency.
    pub fn search_observations(&self, query: &str, limit: usize) -> Result<Vec<Observation>> {
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|t| format!("%{}%", t.to_lowercase()))
            .collect();
        if terms.is_empty() {
            return Ok(vec![]);
        }

        // Build a score expression: one point per matching term.
        let score_expr: String = terms
            .iter()
            .enumerate()
            .map(|(i, _)| {
                format!(
                    "(CASE WHEN lower(content) LIKE ?{i} OR lower(coalesce(symbol_fqn,'')) LIKE ?{i} THEN 1 ELSE 0 END)",
                    i = i + 1
                )
            })
            .collect::<Vec<_>>()
            .join(" + ");

        // WHERE: at least one term must match.
        let where_clause: String = terms
            .iter()
            .enumerate()
            .map(|(i, _)| {
                format!(
                    "lower(content) LIKE ?{i} OR lower(coalesce(symbol_fqn,'')) LIKE ?{i}",
                    i = i + 1
                )
            })
            .collect::<Vec<_>>()
            .join(" OR ");

        let sql = format!(
            "SELECT id, session_id, agent_id, content, symbol_fqn, symbol_hash, \
             created_at, is_stale, kind, expires_at, change_category, ({score}) as _score \
             FROM observations \
             WHERE ({where_clause}) \
               AND (expires_at IS NULL OR datetime(expires_at) > datetime('now')) \
             ORDER BY _score DESC, created_at DESC \
             LIMIT {limit}",
            score = score_expr,
            where_clause = where_clause,
            limit = limit,
        );

        let mut stmt = self.conn.prepare(&sql)?;
        // row_to_observation uses named column access; _score column is ignored automatically.
        let params_vec: Vec<&dyn rusqlite::types::ToSql> = terms
            .iter()
            .map(|t| t as &dyn rusqlite::types::ToSql)
            .collect();
        let results = stmt
            .query_map(params_vec.as_slice(), row_to_observation)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    /// Returns (stale_count, total_count) for non-expired observations.
    pub fn observation_staleness_counts(&self) -> Result<(u64, u64)> {
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM observations \
             WHERE (expires_at IS NULL OR datetime(expires_at) > datetime('now'))",
            [],
            |row| row.get(0),
        )?;
        let stale: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM observations \
             WHERE is_stale = 1 \
               AND (expires_at IS NULL OR datetime(expires_at) > datetime('now'))",
            [],
            |row| row.get(0),
        )?;
        Ok((stale as u64, total as u64))
    }

    /// Load every symbol from the database (used to warm the in-memory graph + search index on startup).
    pub fn all_symbols(&self) -> Result<Vec<Symbol>> {
        let mut stmt = self.conn.prepare(
            "SELECT id,fqn,name,kind,file_path,start_line,end_line,\
             signature,docstring,body,language,content_hash,is_stub,source,resolved_type \
             FROM symbols",
        )?;
        let results = stmt
            .query_map([], row_to_symbol)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    pub fn all_edges(&self) -> Result<Vec<crate::symbol::Edge>> {
        let mut stmt = self
            .conn
            .prepare("SELECT from_id, to_id, kind FROM edges")?;
        let results = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)? as u64,
                    row.get::<_, i64>(1)? as u64,
                    row.get::<_, String>(2)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .map(|(from_id, to_id, kind)| crate::symbol::Edge {
                from_id,
                to_id,
                kind: crate::symbol::EdgeKind::from_db_str(&kind),
                label: None,
            })
            .collect();
        Ok(results)
    }

    pub fn all_symbols_for_file(&self, file_path: &str) -> Result<Vec<Symbol>> {
        let mut stmt = self.conn.prepare(
            "SELECT id,fqn,name,kind,file_path,start_line,end_line,\
             signature,docstring,body,language,content_hash,is_stub,source,resolved_type \
             FROM symbols WHERE file_path = ?1",
        )?;
        let results = stmt
            .query_map(params![file_path], row_to_symbol)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(results)
    }

    // ── Macro expansion cache ─────────────────────────────────────────────────

    /// Return the source_hash stored for `file_path` in the macro-expand cache,
    /// or `None` if this file has never been expanded.
    pub fn get_macro_expand_hash(&self, file_path: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT source_hash FROM macro_expand_cache WHERE file_path = ?1")?;
        let mut rows = stmt.query_map(params![file_path], |row| row.get(0))?;
        Ok(rows.next().transpose()?)
    }

    /// Update (or insert) the cache entry for `file_path`.
    pub fn set_macro_expand_hash(&self, file_path: &str, source_hash: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO macro_expand_cache (file_path, source_hash) VALUES (?1, ?2)",
            params![file_path, source_hash],
        )?;
        Ok(())
    }

    // ── Query log ─────────────────────────────────────────────────────────────

    pub fn log_query(&self, entry: &QueryLogEntry) -> Result<()> {
        self.conn.execute(
            "INSERT INTO query_log \
             (timestamp, task, intent, pivot_count, total_tokens, \
              candidate_file_tokens, latency_ms, languages_hit) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                entry.timestamp,
                entry.task,
                entry.intent,
                entry.pivot_count as i64,
                entry.total_tokens as i64,
                entry.candidate_file_tokens as i64,
                entry.latency_ms as i64,
                entry.languages_hit,
            ],
        )?;
        Ok(())
    }

    /// Aggregate stats from the query_log for the last `days` days.
    /// Returns rows of (intent, count, total_tokens_sum, candidate_file_tokens_sum, latency_ms).
    pub fn query_log_rows(&self, days: u32) -> Result<Vec<QueryLogRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT intent, pivot_count, total_tokens, candidate_file_tokens, latency_ms, languages_hit \
             FROM query_log \
             WHERE datetime(timestamp) >= datetime('now', ?1) \
             ORDER BY timestamp ASC",
        )?;
        let since = format!("-{} days", days);
        let rows = stmt
            .query_map(params![since], |row| {
                Ok(QueryLogRow {
                    intent: row.get(0)?,
                    pivot_count: row.get::<_, i64>(1)? as usize,
                    total_tokens: row.get::<_, i64>(2)? as u64,
                    candidate_file_tokens: row.get::<_, i64>(3)? as u64,
                    latency_ms: row.get::<_, i64>(4)? as u64,
                    languages_hit: row.get(5)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Total token estimate across all indexed symbols (rough workspace baseline).
    pub fn workspace_token_estimate(&self) -> Result<u64> {
        let bytes: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(LENGTH(body)), 0) FROM symbols",
            [],
            |r| r.get(0),
        )?;
        Ok((bytes as u64) / 4)
    }
}

/// Input record for `Database::log_query`.
pub struct QueryLogEntry<'a> {
    pub timestamp: &'a str,
    pub task: &'a str,
    pub intent: &'a str,
    pub pivot_count: usize,
    pub total_tokens: u32,
    pub candidate_file_tokens: u64,
    pub latency_ms: u64,
    pub languages_hit: &'a str,
}

pub struct QueryLogRow {
    pub intent: String,
    pub pivot_count: usize,
    pub total_tokens: u64,
    pub candidate_file_tokens: u64,
    pub latency_ms: u64,
    pub languages_hit: String,
}

// ── Row mappers ───────────────────────────────────────────────────────────────

fn row_to_symbol(row: &rusqlite::Row) -> rusqlite::Result<Symbol> {
    use crate::symbol::Symbol;
    Ok(Symbol {
        id: row.get::<_, i64>(0)? as u64,
        fqn: row.get(1)?,
        name: row.get(2)?,
        kind: SymbolKind::from_db_str(&row.get::<_, String>(3)?),
        file_path: row.get(4)?,
        start_line: row.get(5)?,
        end_line: row.get(6)?,
        signature: row.get(7)?,
        docstring: row.get(8)?,
        body: row.get(9)?,
        language: Language::from_db_str(&row.get::<_, String>(10)?),
        content_hash: row.get(11)?,
        is_stub: row.get::<_, i64>(12)? != 0,
        source: row.get(13)?,
        resolved_type: row.get(14)?,
    })
}

fn row_to_observation(row: &rusqlite::Row) -> rusqlite::Result<Observation> {
    Ok(Observation {
        id: row.get("id")?,
        session_id: row.get("session_id")?,
        agent_id: row.get("agent_id")?,
        content: row.get("content")?,
        symbol_fqn: row.get("symbol_fqn")?,
        symbol_hash: row.get("symbol_hash")?,
        created_at: row.get("created_at")?,
        is_stale: row.get::<_, i64>("is_stale")? != 0,
        kind: crate::memory::ObservationKind::parse_kind(&row.get::<_, String>("kind")?),
        expires_at: row.get("expires_at")?,
        change_category: row.get("change_category")?,
    })
}

// ── DB serialization helpers on types ─────────────────────────────────────────

impl SymbolKind {
    pub fn to_db_str(&self) -> &'static str {
        match self {
            SymbolKind::Function => "function",
            SymbolKind::AsyncFunction => "async_function",
            SymbolKind::Class => "class",
            SymbolKind::Method => "method",
            SymbolKind::AsyncMethod => "async_method",
            SymbolKind::Interface => "interface",
            SymbolKind::TypeAlias => "type_alias",
            SymbolKind::Enum => "enum",
            SymbolKind::Struct => "struct",
            SymbolKind::Trait => "trait",
            SymbolKind::Impl => "impl",
            SymbolKind::Constant => "constant",
            SymbolKind::Variable => "variable",
            SymbolKind::Import => "import",
            SymbolKind::Module => "module",
            SymbolKind::Macro => "macro",
            SymbolKind::ScriptBlock => "script_block",
            SymbolKind::StyleBlock => "style_block",
        }
    }

    pub fn from_db_str(s: &str) -> Self {
        match s {
            "function" => SymbolKind::Function,
            "async_function" => SymbolKind::AsyncFunction,
            "class" => SymbolKind::Class,
            "method" => SymbolKind::Method,
            "async_method" => SymbolKind::AsyncMethod,
            "interface" => SymbolKind::Interface,
            "type_alias" => SymbolKind::TypeAlias,
            "enum" => SymbolKind::Enum,
            "struct" => SymbolKind::Struct,
            "trait" => SymbolKind::Trait,
            "impl" => SymbolKind::Impl,
            "constant" => SymbolKind::Constant,
            "variable" => SymbolKind::Variable,
            "import" => SymbolKind::Import,
            "module" => SymbolKind::Module,
            "macro" => SymbolKind::Macro,
            "script_block" => SymbolKind::ScriptBlock,
            "style_block" => SymbolKind::StyleBlock,
            _ => SymbolKind::Variable,
        }
    }
}

impl EdgeKind {
    pub fn to_db_str(&self) -> &'static str {
        match self {
            EdgeKind::Imports => "imports",
            EdgeKind::Calls => "calls",
            EdgeKind::Implements => "implements",
            EdgeKind::Inherits => "inherits",
            EdgeKind::References => "references",
            EdgeKind::DefinedIn => "defined_in",
        }
    }

    pub fn from_db_str(s: &str) -> Self {
        match s {
            "imports" => EdgeKind::Imports,
            "calls" => EdgeKind::Calls,
            "implements" => EdgeKind::Implements,
            "inherits" => EdgeKind::Inherits,
            "defined_in" => EdgeKind::DefinedIn,
            _ => EdgeKind::References,
        }
    }
}

impl Language {
    pub fn from_db_str(s: &str) -> Self {
        match s {
            "python" => Language::Python,
            "typescript" => Language::TypeScript,
            "tsx" => Language::Tsx,
            "javascript" => Language::JavaScript,
            "jsx" => Language::Jsx,
            "shell" => Language::Shell,
            "html" => Language::Html,
            "rust" => Language::Rust,
            "swift" => Language::Swift,
            "sql" => Language::Sql,
            _ => Language::JavaScript,
        }
    }
}

/// Compute the leaf segment of a qualified `name`. Symbols stored with
/// `name = "Class::method"` (the indexer convention for class methods)
/// resolve to leaf `"method"`. Symbols stored with `name = "function"`
/// (top-level functions) resolve to themselves. Used by both
/// `symbols_by_leaf_name` and the `leaf_name` column populated at
/// `upsert_symbol` time. Issue #96.
fn leaf_of_name(name: &str) -> &str {
    name.rsplit("::").next().unwrap_or(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_temp_db() -> (Database, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dir.path().join("test.db")).unwrap();
        (db, dir)
    }

    #[test]
    fn leaf_of_name_extracts_trailing_segment() {
        // Top-level functions: leaf == name.
        assert_eq!(leaf_of_name("hist"), "hist");
        // Class methods: stored as `Class::method`, leaf is the method name.
        assert_eq!(leaf_of_name("Axes::hist"), "hist");
        // Nested: deepest segment wins.
        assert_eq!(leaf_of_name("Outer::Inner::method"), "method");
        // Defensive: empty name returns empty.
        assert_eq!(leaf_of_name(""), "");
    }

    #[test]
    fn leaf_name_migration_backfills_existing_rows() {
        // Issue #96: existing `.codesurgeon` workspaces have rows with
        // `leaf_name = NULL` from before the column was added. The
        // schema setup runs a one-shot UPDATE to populate them. Verify
        // by simulating an "old" DB: open once to create the schema,
        // null out the leaf_name column for an existing row, close,
        // reopen — the migration should re-populate.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        // First open: insert a method symbol, leaf_name gets populated
        // by `upsert_symbol`.
        {
            let db = Database::open(&db_path).unwrap();
            let method = Symbol::new(
                "lib/bar.py",
                "Axes::hist",
                SymbolKind::Method,
                5,
                20,
                "def hist(self): ...".to_string(),
                None,
                "def hist(self): pass".to_string(),
                Language::Python,
            );
            db.upsert_symbol(&method).unwrap();
        }

        // Simulate the "pre-migration" state by nulling the leaf_name.
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute("UPDATE symbols SET leaf_name = NULL", [])
                .unwrap();
        }

        // Reopen — `create_schema` should backfill leaf_name.
        let db = Database::open(&db_path).unwrap();
        let by_leaf = db.symbols_by_leaf_name("hist", 10).unwrap();
        assert_eq!(
            by_leaf.len(),
            1,
            "migration should have backfilled leaf_name = 'hist' for the method"
        );
    }

    #[test]
    fn symbols_by_leaf_name_finds_class_methods() {
        // Issue #96: `Axes::hist` is stored with `name = "Axes::hist"` so
        // `WHERE name = "hist"` (the existing exact-name lookup) misses
        // it. The leaf-name lookup must catch both top-level functions
        // (where leaf == name) and class methods (where leaf is the
        // trailing `::`-segment).
        let (db, _dir) = open_temp_db();
        let top_level = Symbol::new(
            "lib/foo.py",
            "hist",
            SymbolKind::Function,
            1,
            10,
            "def hist(): ...".to_string(),
            None,
            "def hist(): pass".to_string(),
            Language::Python,
        );
        let method = Symbol::new(
            "lib/bar.py",
            "Axes::hist",
            SymbolKind::Method,
            5,
            20,
            "def hist(self): ...".to_string(),
            None,
            "def hist(self): pass".to_string(),
            Language::Python,
        );
        db.upsert_symbol(&top_level).unwrap();
        db.upsert_symbol(&method).unwrap();

        // Old behaviour: `name`-only lookup misses the method.
        let by_name = db.symbols_by_exact_name("hist", 10).unwrap();
        assert_eq!(by_name.len(), 1, "exact-name should miss Class::method");
        assert_eq!(by_name[0], top_level.id);

        // New behaviour: leaf-name lookup catches both.
        let by_leaf = db.symbols_by_leaf_name("hist", 10).unwrap();
        assert_eq!(
            by_leaf.len(),
            2,
            "leaf-name should catch top-level + method"
        );
        assert!(by_leaf.contains(&top_level.id));
        assert!(by_leaf.contains(&method.id));
    }

    /// `log_query` must not error, and `query_log_rows` must return the inserted row.
    #[test]
    fn log_query_round_trip() {
        let (db, _dir) = open_temp_db();
        db.log_query(&QueryLogEntry {
            timestamp: "2026-01-01T00:00:00Z",
            task: "fix auth bug",
            intent: "debug",
            pivot_count: 3,
            total_tokens: 400,
            candidate_file_tokens: 2000,
            latency_ms: 120,
            languages_hit: "rs",
        })
        .unwrap();

        let rows = db.query_log_rows(365).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].intent, "debug");
        assert_eq!(rows[0].pivot_count, 3);
        assert_eq!(rows[0].total_tokens, 400);
        assert_eq!(rows[0].candidate_file_tokens, 2000);
        assert_eq!(rows[0].latency_ms, 120);
        assert_eq!(rows[0].languages_hit, "rs");
    }

    /// `query_log_rows(0)` must return no rows regardless of what was logged.
    #[test]
    fn query_log_rows_zero_days_returns_empty() {
        let (db, _dir) = open_temp_db();
        db.log_query(&QueryLogEntry {
            timestamp: "2026-01-01T00:00:00Z",
            task: "task",
            intent: "general",
            pivot_count: 1,
            total_tokens: 100,
            candidate_file_tokens: 500,
            latency_ms: 50,
            languages_hit: "rs",
        })
        .unwrap();
        let rows = db.query_log_rows(0).unwrap();
        assert!(rows.is_empty(), "expected no rows for 0-day window");
    }

    /// Multiple rows logged within the window must all be returned.
    #[test]
    fn query_log_rows_returns_all_within_window() {
        let (db, _dir) = open_temp_db();
        let now = chrono::Utc::now().to_rfc3339();
        for i in 0..5u64 {
            db.log_query(&QueryLogEntry {
                timestamp: &now,
                task: "task",
                intent: "add",
                pivot_count: 1,
                total_tokens: 100 + i as u32,
                candidate_file_tokens: 500,
                latency_ms: 50 + i,
                languages_hit: "ts",
            })
            .unwrap();
        }
        let rows = db.query_log_rows(30).unwrap();
        assert_eq!(rows.len(), 5);
    }

    /// `workspace_token_estimate` returns 0 when no symbols are indexed.
    #[test]
    fn workspace_token_estimate_empty_db_returns_zero() {
        let (db, _dir) = open_temp_db();
        assert_eq!(db.workspace_token_estimate().unwrap(), 0);
    }

    /// Deleting edges for an empty list must be a no-op (not an error).
    #[test]
    fn delete_edges_for_symbols_empty_list_noop() {
        let (db, _dir) = open_temp_db();
        assert!(db.delete_edges_for_symbols(&[]).is_ok());
    }

    /// Deleting embeddings for an empty list must be a no-op (not an error).
    #[test]
    fn delete_embeddings_for_symbols_empty_list_noop() {
        let (db, _dir) = open_temp_db();
        assert!(db.delete_embeddings_for_symbols(&[]).is_ok());
    }

    /// Batched delete with > 500 IDs must chunk correctly without SQL errors.
    #[test]
    fn delete_edges_for_symbols_batches_over_500() {
        let (db, _dir) = open_temp_db();
        // Build a list of 600 fake IDs (most won't match any rows, but the SQL must not error).
        let ids: Vec<u64> = (1..=600).collect();
        assert!(db.delete_edges_for_symbols(&ids).is_ok());
        assert!(db.delete_embeddings_for_symbols(&ids).is_ok());
    }
}
