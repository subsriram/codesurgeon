//! codesurgeon MCP server
//!
//! Implements the Model Context Protocol (JSON-RPC 2.0) over stdin/stdout.
//! Add to Claude Code's MCP config:
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "codesurgeon": {
//!       "command": "codesurgeon-mcp",
//!       "args": [],
//!       "env": { "CS_WORKSPACE": "/path/to/your/project" }
//!     }
//!   }
//! }
//! ```

use anyhow::Result;
use cs_core::{engine::EngineConfig, watcher::FileWatcher, CoreEngine};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Shared engine state: `None` = still initializing, `Some` = ready.
type EngineCell = Arc<OnceLock<Arc<CoreEngine>>>;

mod transport;

// ── JSON-RPC types ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Request {
    #[allow(dead_code)]
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct Response {
    jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i32,
    message: String,
}

impl Response {
    fn ok(id: Option<Value>, result: Value) -> Self {
        Response {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn err(id: Option<Value>, code: i32, message: impl Into<String>) -> Self {
        Response {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
            }),
        }
    }
}

// ── Tool definitions ──────────────────────────────────────────────────────────

fn tool_list() -> Value {
    json!({
        "tools": [
            {
                "name": "run_pipeline",
                "description": "Primary tool. Single-call pipeline: hybrid search + graph traversal + session memory. Auto-detects intent from your task description (debug/refactor/add/explore). Returns compressed context with full source for pivot symbols. Use this for most tasks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Describe what you want to do, e.g. 'fix JWT validation bug' or 'refactor UserService'"
                        },
                        "budget_tokens": {
                            "type": "integer",
                            "description": "Max tokens to include in the capsule (default: 4000)",
                            "default": 4000
                        }
                    },
                    "required": ["task"]
                }
            },
            {
                "name": "get_context_capsule",
                "description": "Lightweight context search. Returns only the code relevant to your query, bounded to token budget. Pivot symbols in full, adjacent symbols as skeletons.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What you're looking for, e.g. 'authentication middleware' or 'database connection pool'"
                        },
                        "budget_tokens": {
                            "type": "integer",
                            "description": "Max tokens (default: 4000)",
                            "default": 4000
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_impact_graph",
                "description": "Show every caller, importer, and dependent that would break if this symbol changes. Use before any refactor to understand blast radius.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol_fqn": {
                            "type": "string",
                            "description": "Fully-qualified name, e.g. 'src/auth/service.py::AuthService::validate_token'"
                        }
                    },
                    "required": ["symbol_fqn"]
                }
            },
            {
                "name": "get_skeleton",
                "description": "File structure without implementation bodies. Shows signatures, docstrings, return types only. 70-90% token reduction. Use to understand a file's API surface.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Relative path to the file, e.g. 'src/auth/service.py'"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "search_logic_flow",
                "description": "Trace the execution path between two functions. Debug flow issues without reading every file in between.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "from_fqn": {
                            "type": "string",
                            "description": "FQN of the starting function"
                        },
                        "to_fqn": {
                            "type": "string",
                            "description": "FQN of the target function"
                        }
                    },
                    "required": ["from_fqn", "to_fqn"]
                }
            },
            {
                "name": "index_status",
                "description": "Health check and statistics: symbol count, edge count, file count, session ID.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_session_context",
                "description": "Returns observations from current and previous sessions. Shows what was explored, decided, and learned. Stale observations are flagged.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "save_observation",
                "description": "Persist an insight, decision, or note about the codebase. Optionally link to a symbol so it gets auto-flagged stale when that code changes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The observation to save"
                        },
                        "symbol_fqn": {
                            "type": "string",
                            "description": "Optional FQN of the symbol this observation is about"
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "get_diff_capsule",
                "description": "Given a git diff, return a context capsule focused on changed symbols, their callers, and related test files. Use before reviewing or merging a PR.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "diff": {
                            "type": "string",
                            "description": "Output of `git diff` or `git diff HEAD~1`"
                        },
                        "budget_tokens": {
                            "type": "integer",
                            "description": "Max tokens (default: 4000)",
                            "default": 4000
                        }
                    },
                    "required": ["diff"]
                }
            },
            {
                "name": "generate_module_docs",
                "description": "Auto-generate CLAUDE.md summaries for each directory in the codebase. Returns the generated documentation. Pass write_files=true to write CLAUDE.md files to disk.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "write_files": {
                            "type": "boolean",
                            "description": "If true, write CLAUDE.md files into each directory (default: false — preview only)",
                            "default": false
                        }
                    }
                }
            }
        ]
    })
}

// ── PID-file lock ─────────────────────────────────────────────────────────────

/// Attempt to acquire a per-workspace PID lock at `<workspace>/.codesurgeon/mcp.pid`.
///
/// Returns `Ok(())` if this process should become the server.
/// Returns `Err` with a human-readable message if another live instance is already running,
/// so the caller can exit cleanly rather than accumulating duplicate processes.
fn acquire_pid_lock(workspace: &Path) -> Result<PathBuf> {
    let pid_path = workspace.join(".codesurgeon").join("mcp.pid");
    std::fs::create_dir_all(pid_path.parent().unwrap())?;

    if let Ok(existing) = std::fs::read_to_string(&pid_path) {
        if let Ok(existing_pid) = existing.trim().parse::<u32>() {
            // `kill -0 <pid>` exits 0 if the process exists, non-zero otherwise.
            let alive = std::process::Command::new("kill")
                .args(["-0", &existing_pid.to_string()])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if alive && existing_pid != std::process::id() {
                anyhow::bail!(
                    "Another codesurgeon-mcp is already serving this workspace (PID {}). \
                     Kill it first or remove {}.",
                    existing_pid,
                    pid_path.display()
                );
            }
        }
    }

    std::fs::write(&pid_path, std::process::id().to_string())?;
    Ok(pid_path)
}

fn release_pid_lock(pid_path: &Path) {
    let _ = std::fs::remove_file(pid_path);
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Log to stderr so it doesn't pollute the MCP stdio channel
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(std::env::var("CS_LOG").unwrap_or_else(|_| "warn".to_string()))
        .init();

    let workspace = resolve_workspace();
    tracing::info!(
        "Starting codesurgeon-mcp for workspace: {}",
        workspace.display()
    );

    // Try to become the exclusive background-indexing instance for this workspace.
    // If another instance is already running we do NOT exit — we still serve this
    // connection (e.g. a parallel Codex probe) — we just skip background work to
    // avoid duplicate index writes.
    let _pid_path = match acquire_pid_lock(&workspace) {
        Ok(p) => {
            // Build the engine in the background so the stdio loop (and the
            // `initialize` handshake) start immediately.  Claude Code and Codex
            // both time out if the first response takes more than a few seconds;
            // loading the ONNX embedding model can take 10-15 s on first run.
            let cell: EngineCell = Arc::new(OnceLock::new());
            let cell_bg = Arc::clone(&cell);
            let workspace_bg = workspace.clone();

            tokio::task::spawn_blocking(move || {
                let config = EngineConfig::new(&workspace_bg);
                let engine = match CoreEngine::new(config) {
                    Ok(e) => Arc::new(e),
                    Err(e) => {
                        tracing::error!("CoreEngine init failed: {}", e);
                        return;
                    }
                };

                // Kick off background indexing (primary instance only)
                let e2 = Arc::clone(&engine);
                std::thread::spawn(move || match e2.index_workspace() {
                    Ok(stats) => tracing::info!(
                        "Index complete: {} symbols, {} edges, {} files",
                        stats.symbol_count,
                        stats.edge_count,
                        stats.file_count
                    ),
                    Err(e) => tracing::error!("Indexing failed: {}", e),
                });

                // Watch for file changes and re-index incrementally (primary instance only)
                let e3 = Arc::clone(&engine);
                std::thread::spawn(move || {
                    let watcher = match FileWatcher::new(&workspace_bg) {
                        Ok(w) => w,
                        Err(e) => {
                            tracing::error!("Failed to start file watcher: {}", e);
                            return;
                        }
                    };
                    tracing::info!("File watcher started for {}", workspace_bg.display());
                    loop {
                        for event in watcher.poll(Duration::from_millis(500)) {
                            if let Err(e) = e3.reindex_file(&event.path, event.kind) {
                                tracing::warn!("reindex_file failed for {:?}: {}", event.path, e);
                            }
                        }
                    }
                });

                // Make the engine available to the stdio loop.
                let _ = cell_bg.set(engine);
                tracing::info!("Engine ready");
            });

            run_stdio_loop(cell).await;
            release_pid_lock(&p);
            return Ok(());
        }
        Err(e) => {
            // Another instance owns the index — serve this connection read-only
            // (no background indexing or file watching) so parallel probes still work.
            tracing::warn!("PID lock held by another instance ({}); serving read-only", e);
        }
    };

    // Secondary instance (no PID lock): serve read-only, no background tasks.
    // Skip the embedder — secondary instances don't compute new embeddings and
    // loading the ONNX model wastes ~1-2 GB of RAM per short-lived probe process.
    // No embedder means init is fast, so blocking here is fine.
    let config = EngineConfig::new(&workspace).without_embedder();
    let engine = Arc::new(CoreEngine::new(config)?);
    let cell: EngineCell = Arc::new(OnceLock::new());
    let _ = cell.set(engine);
    run_stdio_loop(cell).await;
    Ok(())
}

// ── stdio loop ────────────────────────────────────────────────────────────────

/// Drives the MCP JSON-RPC session on stdin/stdout.
///
/// Supports two wire formats so the same binary works with Claude Code and Codex:
///   • LSP-framed  — `Content-Length: N\r\n\r\n{json}` — required by Codex (spec-correct)
///   • NDJSON      — raw newline-terminated JSON        — accepted by Claude Code
///
/// Detection is per-message: if the first non-empty line starts with
/// `Content-Length:` the message is read using LSP framing; otherwise the line
/// itself is the JSON body.  Responses are **always** written with Content-Length
/// framing because Codex requires it and Claude Code accepts it per the MCP spec.
async fn run_stdio_loop(cell: EngineCell) {
    let mut stdin_reader = std::io::BufReader::new(std::io::stdin());
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    loop {
        let (message, fmt) = match transport::read_message(&mut stdin_reader) {
            Ok(Some(m)) => m,
            Ok(None) => break,  // EOF — client closed the connection
            Err(e) => { tracing::error!("stdin read error: {}", e); break; }
        };

        if message.is_empty() {
            continue;
        }

        let response = handle_message(cell.get(), &message).await;

        if let Some(resp) = response {
            let json = match serde_json::to_string(&resp) {
                Ok(j) => j,
                Err(e) => { tracing::error!("Failed to serialize response: {}", e); continue; }
            };
            if let Err(e) = transport::write_message(&mut out, &json, fmt) {
                tracing::error!("stdout write error: {}", e);
                break;
            }
        }
    }
}

async fn handle_message(engine: Option<&Arc<CoreEngine>>, line: &str) -> Option<Response> {
    let req: Request = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("Failed to parse message: {} — {}", e, line);
            return Some(Response::err(None, -32700, format!("Parse error: {}", e)));
        }
    };

    tracing::debug!("← {}", req.method);

    match req.method.as_str() {
        "initialize" => {
            // Echo back the client's requested protocol version so any version
            // of the Claude client (old or new) sees a match and doesn't reject us.
            let protocol_version = req
                .params
                .get("protocolVersion")
                .and_then(|v| v.as_str())
                .unwrap_or("2024-11-05");
            Some(Response::ok(
                req.id,
                json!({
                    "protocolVersion": protocol_version,
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    },
                    "serverInfo": {
                        "name": "codesurgeon",
                        "version": env!("CARGO_PKG_VERSION")
                    },
                    "instructions": "codesurgeon provides graph-based context from your codebase. \
                        Call run_pipeline before editing code. \
                        Call get_impact_graph before refactoring. \
                        Call save_observation to persist insights across sessions."
                }),
            ))
        }

        "notifications/initialized" => {
            // Notification — no response
            None
        }

        // Resources — we expose no resources but must respond so Codex probes succeed.
        "resources/list" => Some(Response::ok(req.id, json!({ "resources": [] }))),
        "resources/templates/list" => Some(Response::ok(req.id, json!({ "resourceTemplates": [] }))),

        // Prompts — we expose no prompts; return empty list so newer clients don't disconnect.
        "prompts/list" => Some(Response::ok(req.id, json!({ "prompts": [] }))),

        "tools/list" => Some(Response::ok(req.id, tool_list())),

        "tools/call" => {
            let name = req
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = req.params.get("arguments").cloned().unwrap_or(json!({}));

            let Some(engine) = engine else {
                return Some(Response::ok(
                    req.id,
                    json!({ "content": [{ "type": "text", "text":
                        "⏳ Engine still initializing (loading index + embedding model). \
                         Retry in a few seconds or call `index_status` to check." }] }),
                ));
            };

            let result = dispatch_tool(engine, name, &args).await;

            match result {
                Ok(text) => Some(Response::ok(
                    req.id,
                    json!({ "content": [{ "type": "text", "text": text }] }),
                )),
                Err(e) => {
                    tracing::error!("Tool error ({}): {}", name, e);
                    Some(Response::err(req.id, -32603, format!("Tool error: {}", e)))
                }
            }
        }

        "ping" => Some(Response::ok(req.id, json!({}))),

        other => {
            tracing::warn!("Unknown method: {}", other);
            Some(Response::err(
                req.id,
                -32601,
                format!("Method not found: {}", other),
            ))
        }
    }
}

/// Tools that require a populated index to return useful results.
const INDEX_DEPENDENT_TOOLS: &[&str] = &[
    "run_pipeline",
    "get_context_capsule",
    "get_impact_graph",
    "get_skeleton",
    "search_logic_flow",
    "get_diff_capsule",
    "generate_module_docs",
];

async fn dispatch_tool(engine: &Arc<CoreEngine>, name: &str, args: &Value) -> Result<String> {
    // Block index-dependent tools only when the index is genuinely empty (first-ever
    // run with no persisted data). When a warm index exists in SQLite we serve from it
    // immediately — re-indexing runs in the background and results stay usable.
    if INDEX_DEPENDENT_TOOLS.contains(&name) && engine.is_indexing() {
        let stats = engine.index_stats().unwrap_or_default();
        if stats.symbol_count == 0 {
            return Ok(
                "⏳ Index build in progress — no symbols yet. \
                 Retry in a few seconds or call `index_status` to monitor."
                    .to_string(),
            );
        }
        // Warm index available: fall through and serve results.
        // Re-indexing is finishing in the background; output reflects last-known state.
        tracing::debug!(
            "Serving from warm index ({} symbols) while re-index runs in background",
            stats.symbol_count
        );
    }

    // Clone the Arc and move into blocking thread so we don't block the async runtime
    let engine = Arc::clone(engine);
    let name = name.to_string();
    let args = args.clone();

    tokio::task::spawn_blocking(move || match name.as_str() {
        "run_pipeline" => {
            let task = string_arg(&args, "task")?;
            let budget = args
                .get("budget_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            engine.run_pipeline(&task, budget)
        }

        "get_context_capsule" => {
            let query = string_arg(&args, "query")?;
            let budget = args
                .get("budget_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            engine.get_context_capsule(&query, budget)
        }

        "get_impact_graph" => {
            let fqn = string_arg(&args, "symbol_fqn")?;
            let result = engine.get_impact_graph(&fqn)?;
            Ok(serde_json::to_string_pretty(&result)?)
        }

        "get_skeleton" => {
            let file_path = string_arg(&args, "file_path")?;
            let result = engine.get_skeleton(&file_path)?;
            let mut out = format!(
                "## Skeleton: {}\n({} symbols, ~{} tokens)\n\n",
                result.file_path,
                result.symbols.len(),
                result.token_estimate
            );
            for sym in &result.symbols {
                out.push_str(&format!(
                    "### `{}` ({}) @ line {}\n```\n{}\n```\n\n",
                    sym.fqn, sym.kind, sym.start_line, sym.skeleton
                ));
            }
            Ok(out)
        }

        "search_logic_flow" => {
            let from = string_arg(&args, "from_fqn")?;
            let to = string_arg(&args, "to_fqn")?;
            let result = engine.search_logic_flow(&from, &to)?;
            if result.found {
                let path_str: Vec<String> = result
                    .path
                    .iter()
                    .map(|s| format!("`{}` ({})", s.fqn, s.file_path))
                    .collect();
                Ok(format!(
                    "## Logic flow: {} → {}\n\n{}\n",
                    from,
                    to,
                    path_str.join("\n  ↓\n")
                ))
            } else {
                Ok(format!("No direct path found from `{}` to `{}`.", from, to))
            }
        }

        "index_status" => {
            let indexing = engine.is_indexing();
            let stats = engine.index_stats()?;
            let xcode_line = if stats.xcode_mcp_available {
                "- Swift enrichment: Xcode MCP available (Xcode 26+) — use it for resolved types and diagnostics\n"
            } else {
                "- Swift enrichment: Xcode MCP not detected — tree-sitter only (see README for setup)\n"
            };
            Ok(format!(
                "## codesurgeon index status\n\
                 - Indexing: {}\n\
                 - Symbols: {}\n\
                 - Edges: {}\n\
                 - Files: {}\n\
                 - Session: {}\n\
                 {}",
                if indexing { "in progress" } else { "ready" },
                stats.symbol_count,
                stats.edge_count,
                stats.file_count,
                stats.session_id,
                xcode_line,
            ))
        }

        "get_session_context" => {
            let observations = engine.get_session_context()?;
            if observations.is_empty() {
                return Ok("No session observations yet.".to_string());
            }
            let mut out = "## Session memory\n\n".to_string();
            for obs in &observations {
                let stale = if obs.is_stale { " ⚠️ [stale]" } else { "" };
                let sym = obs
                    .symbol_fqn
                    .as_deref()
                    .map(|f| format!(" (re: `{}`)", f))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "- [{}]{}{}: {}\n",
                    obs.created_at, sym, stale, obs.content
                ));
            }
            Ok(out)
        }

        "save_observation" => {
            let content = string_arg(&args, "content")?;
            let symbol_fqn = args.get("symbol_fqn").and_then(|v| v.as_str());
            engine.save_observation(&content, symbol_fqn)?;
            Ok("Observation saved.".to_string())
        }

        "get_diff_capsule" => {
            let diff = string_arg(&args, "diff")?;
            let budget = args
                .get("budget_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            engine.get_diff_capsule(&diff, budget)
        }

        "generate_module_docs" => {
            let write_files = args
                .get("write_files")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            engine.generate_module_docs(write_files)
        }

        other => Err(anyhow::anyhow!("Unknown tool: {}", other)),
    })
    .await?
}

fn string_arg(args: &Value, key: &str) -> Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("Missing required argument: {}", key))
}

fn resolve_workspace() -> PathBuf {
    // 1. Explicit env var
    if let Ok(ws) = std::env::var("CS_WORKSPACE") {
        return PathBuf::from(ws);
    }
    // 2. Claude Code sets CLAUDE_CODE_WORKSPACE
    if let Ok(ws) = std::env::var("CLAUDE_CODE_WORKSPACE") {
        return PathBuf::from(ws);
    }
    // 3. Current directory
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}
