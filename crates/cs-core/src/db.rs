use crate::language::Language;
use crate::memory::Observation;
use crate::symbol::{Edge, EdgeKind, Symbol, SymbolKind};
use anyhow::Result;
use rusqlite::{params, Connection};
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
             PRAGMA cache_size=-65536;",
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
        let _ = self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS macro_expand_cache \
             (file_path TEXT PRIMARY KEY, source_hash TEXT NOT NULL);",
        );
        let _ = self
            .conn
            .execute("ALTER TABLE observations ADD COLUMN expires_at TEXT", []);
        // Index may already exist on new databases; ignore error.
        let _ = self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_obs_expires ON observations(expires_at);",
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
                resolved_type)
               VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)"#,
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
             (id, session_id, agent_id, content, symbol_fqn, symbol_hash, created_at, is_stale, kind, expires_at) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
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
            ],
        )?;
        Ok(())
    }

    pub fn get_session_observations(&self, session_id: &str) -> Result<Vec<Observation>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, agent_id, content, symbol_fqn, symbol_hash, \
             created_at, is_stale, kind, expires_at \
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
             created_at, is_stale, kind, expires_at \
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
             created_at, is_stale, kind, expires_at \
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
    /// non-summary observations. Used to find candidates for compression.
    pub fn fqns_needing_compression(&self, threshold: usize) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT symbol_fqn, COUNT(*) as cnt \
             FROM observations \
             WHERE symbol_fqn IS NOT NULL \
               AND kind != 'summary' \
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
             created_at, is_stale, kind, expires_at, ({score}) as _score \
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
        let params_vec: Vec<&dyn rusqlite::types::ToSql> =
            terms.iter().map(|t| t as &dyn rusqlite::types::ToSql).collect();
        let results = stmt
            .query_map(params_vec.as_slice(), |row| {
                // row_to_observation reads columns 0..9; column 10 is _score (ignored)
                row_to_observation(row)
            })?
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
        id: row.get(0)?,
        session_id: row.get(1)?,
        agent_id: row.get(2)?,
        content: row.get(3)?,
        symbol_fqn: row.get(4)?,
        symbol_hash: row.get(5)?,
        created_at: row.get(6)?,
        is_stale: row.get::<_, i64>(7)? != 0,
        kind: crate::memory::ObservationKind::parse_kind(&row.get::<_, String>(8)?),
        expires_at: row.get(9)?,
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
