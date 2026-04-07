//! Pyright JSON type-enrichment pass.
//!
//! Runs `pyright --outputjson` against the workspace and merges
//! resolved type annotations (return types from explicit annotations and
//! inferred types from pyright diagnostics) into existing Python symbols.
//!
//! Enabled by `[indexing] python_pyright = true` in `.codesurgeon/config.toml`.
//! Default: false.
//!
//! Graceful skip when:
//! - no Python files exist in the workspace
//! - `pyright` is not installed (a hint is logged, indexing continues)
//! - the Python-files hash is unchanged since last run (incremental skip)
//! - `pyright` subprocess fails for any reason (logged at WARN)
//!
//! Incremental: re-run gated on a hash of Python `.py` file stats
//! (path + size + mtime) across the workspace. Skipped when unchanged.
//!
//! What gets stored in `resolved_type`:
//! - Functions/methods: the return type string, e.g. `"str"`, `"List[int]"`,
//!   `"Optional[Dict[str, Any]]"`.  Sourced first from the explicit `-> T`
//!   annotation already captured in `signature`, then from any `information`-
//!   level pyright diagnostic at the same file/line.
//! - Classes: comma-separated base-class names (already in `signature`, mirrored
//!   here for consistency with the rustdoc enrichment convention).

use crate::db::Database;
use crate::language::Language;
use crate::symbol::{Symbol, SymbolKind};
use crate::watcher::hash_content;
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the pyright enrichment pass.
///
/// Mutates symbols in `all_symbols` in-place, setting `resolved_type` and
/// `source = "pyright"` on matched Python symbols.
/// Returns the number of symbols enriched.
///
/// Silently returns 0 when:
/// - no Python symbols are present in `all_symbols`
/// - `pyright` is not installed
/// - the Python-files hash matches the cached value (incremental skip)
/// - `pyright` subprocess fails (logged at WARN)
pub fn run_pyright_enrichment(
    workspace_root: &Path,
    all_symbols: &mut [Symbol],
    db: &Database,
) -> usize {
    // Gate 1: must have Python symbols to enrich
    if !all_symbols.iter().any(|s| s.language == Language::Python) {
        return 0;
    }

    // Gate 2: pyright must be installed
    if !pyright_available() {
        tracing::info!(
            "pyright not found — skipping Python type enrichment. \
             Install with: npm install -g pyright"
        );
        return 0;
    }

    // Gate 3: incremental — skip if Python file stats haven't changed
    let py_hash = python_files_hash(workspace_root);
    match db.get_macro_expand_hash("__pyright__") {
        Ok(Some(cached)) if cached == py_hash => {
            tracing::debug!("pyright-enrich cache hit (Python files unchanged)");
            return 0;
        }
        _ => {}
    }

    // Run pyright and collect its JSON output
    let pyright_json = match run_pyright(workspace_root) {
        Ok(json) => json,
        Err(e) => {
            tracing::warn!("pyright-enrich: pyright failed: {}", e);
            return 0;
        }
    };

    // Parse any type information from the pyright JSON diagnostics
    let diag_map = parse_pyright_diagnostics(&pyright_json, workspace_root);

    // Merge into symbols: explicit annotations first, then pyright diagnostics
    let count = merge_pyright_types(all_symbols, &diag_map);

    // Update incremental cache
    if let Err(e) = db.set_macro_expand_hash("__pyright__", &py_hash) {
        tracing::warn!("pyright-enrich cache write failed: {}", e);
    }

    count
}

// ── Detection helpers ─────────────────────────────────────────────────────────

static PYRIGHT_AVAILABLE: OnceLock<bool> = OnceLock::new();

fn pyright_available() -> bool {
    *PYRIGHT_AVAILABLE.get_or_init(|| {
        Command::new("pyright")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

// ── Incremental hash ─────────────────────────────────────────────────────────

/// Compute a quick hash over all `.py` file stats (path + size + mtime) in the
/// workspace.  Does NOT read file content — just `stat` calls.
fn python_files_hash(workspace_root: &Path) -> String {
    let mut parts: Vec<String> = Vec::new();
    collect_py_stats(workspace_root, &mut parts);
    parts.sort();
    hash_content(parts.join("\n").as_bytes())
}

fn collect_py_stats(dir: &Path, parts: &mut Vec<String>) {
    // Note: uses std::fs::read_dir rather than ignore::WalkBuilder intentionally —
    // we're only stat-ing files for the incremental hash, not indexing them,
    // so .gitignore compliance is not required here.
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Skip hidden dirs and common virtual-env / cache directories
            if name.starts_with('.')
                || name == "node_modules"
                || name == "__pycache__"
                || name == ".venv"
                || name == "venv"
                || name == "site-packages"
            {
                continue;
            }
            collect_py_stats(&path, parts);
        } else if path.extension().map(|e| e == "py").unwrap_or(false) {
            if let Ok(meta) = std::fs::metadata(&path) {
                let size = meta.len();
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                parts.push(format!("{}:{}:{}", path.display(), size, mtime));
            }
        }
    }
}

// ── pyright subprocess ────────────────────────────────────────────────────────

/// Invoke `pyright --outputjson` from `workspace_root`.
///
/// pyright exits with:
/// - 0 when there are no type errors
/// - 1 when type errors are found (output is still valid JSON)
/// - ≥2 on configuration / fatal errors
///
/// We treat exit codes 0 and 1 as success and parse the output.
fn run_pyright(workspace_root: &Path) -> Result<String> {
    let output = Command::new("pyright")
        .arg("--outputjson")
        .current_dir(workspace_root)
        .output()?;

    let exit_code = output.status.code().unwrap_or(2);
    if exit_code >= 2 {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("pyright exited {}: {}", exit_code, stderr.trim());
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

// ── Diagnostic JSON parsing ───────────────────────────────────────────────────

/// Type information extracted from a pyright diagnostic message.
#[derive(Debug)]
struct DiagTypeInfo {
    /// Resolved type string extracted from the message, e.g. `"str"`.
    resolved_type: String,
}

/// Parse the pyright `--outputjson` output and return a map of
/// `(relative_file_path, 1-based_line)` → `DiagTypeInfo`.
///
/// We look for `information`-severity diagnostics whose messages match
/// patterns like `Type of "foo" is "(x: int) -> str"`.  These are emitted
/// when pyright infers a type and the caller has enabled the relevant rule.
fn parse_pyright_diagnostics(
    json_str: &str,
    workspace_root: &Path,
) -> HashMap<(String, u32), DiagTypeInfo> {
    let mut map = HashMap::new();

    let Ok(root) = serde_json::from_str::<serde_json::Value>(json_str) else {
        return map;
    };

    let Some(diags) = root.get("generalDiagnostics").and_then(|v| v.as_array()) else {
        return map;
    };

    for diag in diags {
        let severity = diag.get("severity").and_then(|v| v.as_str()).unwrap_or("");
        // Only information-level diagnostics carry inferred-type messages
        if severity != "information" {
            continue;
        }

        let file = diag.get("file").and_then(|v| v.as_str()).unwrap_or("");
        let message = diag.get("message").and_then(|v| v.as_str()).unwrap_or("");
        // pyright reports 0-based lines; we store 1-based to match Symbol::start_line
        let line = diag
            .pointer("/range/start/line")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
            + 1;

        let rel_file = make_relative(file, workspace_root);

        if let Some(info) = extract_type_from_message(message) {
            map.insert((rel_file, line), info);
        }
    }

    map
}

/// Make an absolute path relative to `workspace_root`.  Falls back to the
/// original string when the path does not start with the workspace root.
fn make_relative(file: &str, workspace_root: &Path) -> String {
    PathBuf::from(file)
        .strip_prefix(workspace_root)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| file.to_string())
}

/// Try to extract a type string from a pyright diagnostic message.
///
/// Recognised patterns:
/// - `Type of "foo" is "(x: int) -> str"` → `"(x: int) -> str"`
/// - `Type of "foo" is "str"`             → `"str"`
fn extract_type_from_message(message: &str) -> Option<DiagTypeInfo> {
    // Look for: ... is "<type>"
    let is_marker = "\" is \"";
    let is_pos = message.find(is_marker)?;
    let type_start = is_pos + is_marker.len();
    let closing = message[type_start..].find('"')?;
    let type_str = message[type_start..type_start + closing].trim();
    if type_str.is_empty() {
        return None;
    }
    Some(DiagTypeInfo {
        resolved_type: type_str.to_string(),
    })
}

// ── Signature parsing ─────────────────────────────────────────────────────────

/// Extract the return type from a Python function signature string.
///
/// Examples:
/// - `"def foo(x: int) -> str:"` → `Some("str")`
/// - `"def bar() -> Optional[Dict[str, Any]]:"` → `Some("Optional[Dict[str, Any]]")`
/// - `"def baz():"` → `None`
pub fn extract_return_type_from_sig(sig: &str) -> Option<String> {
    // rfind handles signatures with default args that contain "->", e.g.:
    //   def f(x: Callable[[], int] = lambda: 0) -> str:
    let arrow = sig.rfind(" -> ")?;
    let after = sig[arrow + 4..].trim_end_matches(':').trim();
    if after.is_empty() {
        return None;
    }
    Some(after.to_string())
}

/// Extract base-class names from a Python class signature string.
///
/// Example: `"class MyView(APIView, LogMixin):"` → `Some("APIView, LogMixin")`
fn extract_bases_from_sig(sig: &str) -> Option<String> {
    let open = sig.find('(')?;
    let close = sig.rfind(')')?;
    if close <= open {
        return None;
    }
    let bases = sig[open + 1..close].trim();
    if bases.is_empty() {
        None
    } else {
        Some(bases.to_string())
    }
}

// ── Merge pass ────────────────────────────────────────────────────────────────

/// Merge pyright type information into `symbols`.
///
/// For each Python function/method symbol:
/// 1. Try to extract the return type from the existing `signature` field
///    (explicit `-> T` annotation captured by tree-sitter).
/// 2. Fall back to a matching `information`-level pyright diagnostic at the
///    same file + start-line.
///
/// For each Python class symbol:
/// - Extract base-class names from the `signature` field.
///
/// Sets `resolved_type` and `source = "pyright"`.  Skips symbols that
/// already have a `resolved_type`.
///
/// Returns the number of symbols enriched.
fn merge_pyright_types(
    all_symbols: &mut [Symbol],
    diag_map: &HashMap<(String, u32), DiagTypeInfo>,
) -> usize {
    let mut count = 0;

    for sym in all_symbols.iter_mut() {
        if sym.language != Language::Python {
            continue;
        }
        // Already enriched (e.g. by a previous run restored from the DB)
        if sym.resolved_type.is_some() {
            continue;
        }

        let resolved: Option<String> = match sym.kind {
            SymbolKind::Function
            | SymbolKind::Method
            | SymbolKind::AsyncFunction
            | SymbolKind::AsyncMethod => {
                // 1. Explicit annotation in signature
                let from_sig = extract_return_type_from_sig(&sym.signature);
                // 2. Pyright information diagnostic at this location
                let from_diag = || {
                    diag_map
                        .get(&(sym.file_path.clone(), sym.start_line))
                        .map(|info| info.resolved_type.clone())
                };
                from_sig.or_else(from_diag)
            }
            SymbolKind::Class => extract_bases_from_sig(&sym.signature),
            _ => None,
        };

        if let Some(rt) = resolved {
            sym.resolved_type = Some(rt);
            sym.source = Some("pyright".to_string());
            count += 1;
        }
    }

    count
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_return_type_simple() {
        assert_eq!(
            extract_return_type_from_sig("def foo(x: int) -> str:"),
            Some("str".to_string())
        );
    }

    #[test]
    fn extract_return_type_generic() {
        assert_eq!(
            extract_return_type_from_sig("def bar() -> Optional[Dict[str, Any]]:"),
            Some("Optional[Dict[str, Any]]".to_string())
        );
    }

    #[test]
    fn extract_return_type_none_annotation() {
        assert_eq!(
            extract_return_type_from_sig("def baz() -> None:"),
            Some("None".to_string())
        );
    }

    #[test]
    fn extract_return_type_missing() {
        assert_eq!(extract_return_type_from_sig("def baz():"), None);
    }

    #[test]
    fn extract_return_type_callable_default() {
        // Signature contains "->" inside a default value; rfind picks the last one
        assert_eq!(
            extract_return_type_from_sig("def f(cb: Callable[[], int] = lambda: 0) -> str:"),
            Some("str".to_string())
        );
    }

    #[test]
    fn extract_bases_with_parents() {
        assert_eq!(
            extract_bases_from_sig("class MyView(APIView, LogMixin):"),
            Some("APIView, LogMixin".to_string())
        );
    }

    #[test]
    fn extract_bases_no_parents() {
        assert_eq!(extract_bases_from_sig("class Bare:"), None);
    }

    #[test]
    fn extract_type_from_message_basic() {
        let msg = r#"Type of "foo" is "str""#;
        let info = extract_type_from_message(msg).expect("should parse");
        assert_eq!(info.resolved_type, "str");
    }

    #[test]
    fn extract_type_from_message_callable() {
        let msg = r#"Type of "process" is "(items: list[str]) -> bool""#;
        let info = extract_type_from_message(msg).expect("should parse");
        assert_eq!(info.resolved_type, "(items: list[str]) -> bool");
    }

    #[test]
    fn extract_type_from_message_no_match() {
        assert!(extract_type_from_message("Return type is partially unknown").is_none());
    }

    #[test]
    fn merge_enriches_annotated_function() {
        use crate::symbol::Symbol;
        let mut sym = Symbol::new(
            "src/app.py",
            "compute",
            SymbolKind::Function,
            1,
            3,
            "def compute(x: int) -> bool:".to_string(),
            None,
            "def compute(x: int) -> bool:\n    return x > 0".to_string(),
            Language::Python,
        );
        let diag_map = HashMap::new();
        let count = merge_pyright_types(std::slice::from_mut(&mut sym), &diag_map);
        assert_eq!(count, 1);
        assert_eq!(sym.resolved_type.as_deref(), Some("bool"));
        assert_eq!(sym.source.as_deref(), Some("pyright"));
    }

    #[test]
    fn merge_skips_unannotated_without_diag() {
        use crate::symbol::Symbol;
        let mut sym = Symbol::new(
            "src/app.py",
            "compute",
            SymbolKind::Function,
            1,
            3,
            "def compute(x):".to_string(),
            None,
            "def compute(x):\n    return x".to_string(),
            Language::Python,
        );
        let diag_map = HashMap::new();
        let count = merge_pyright_types(std::slice::from_mut(&mut sym), &diag_map);
        assert_eq!(count, 0);
        assert!(sym.resolved_type.is_none());
    }

    #[test]
    fn merge_skips_already_enriched_symbol() {
        use crate::symbol::Symbol;
        // A symbol that already has resolved_type set (e.g. from a previous run restored
        // from the DB) must not be mutated again — source must also be left intact.
        let mut sym = Symbol::new(
            "src/app.py",
            "compute",
            SymbolKind::Function,
            1,
            3,
            "def compute(x: int) -> bool:".to_string(),
            None,
            "def compute(x: int) -> bool:\n    return x > 0".to_string(),
            Language::Python,
        );
        sym.resolved_type = Some("bool".to_string());
        sym.source = Some("pyright".to_string());

        let diag_map = HashMap::new();
        let count = merge_pyright_types(std::slice::from_mut(&mut sym), &diag_map);
        assert_eq!(count, 0, "already-enriched symbol must not be re-enriched");
        assert_eq!(sym.resolved_type.as_deref(), Some("bool"));
        assert_eq!(sym.source.as_deref(), Some("pyright"));
    }

    #[test]
    fn merge_uses_diag_for_inferred_type() {
        use crate::symbol::Symbol;
        let mut sym = Symbol::new(
            "src/app.py",
            "compute",
            SymbolKind::Function,
            5,
            8,
            "def compute(x):".to_string(),
            None,
            "def compute(x):\n    return x + 1".to_string(),
            Language::Python,
        );
        let mut diag_map = HashMap::new();
        diag_map.insert(
            ("src/app.py".to_string(), 5),
            DiagTypeInfo {
                resolved_type: "int".to_string(),
            },
        );
        let count = merge_pyright_types(std::slice::from_mut(&mut sym), &diag_map);
        assert_eq!(count, 1);
        assert_eq!(sym.resolved_type.as_deref(), Some("int"));
    }

    #[test]
    fn merge_enriches_class_with_bases() {
        use crate::symbol::Symbol;
        let mut sym = Symbol::new(
            "src/app.py",
            "MyView",
            SymbolKind::Class,
            1,
            10,
            "class MyView(APIView, LogMixin):".to_string(),
            None,
            "class MyView(APIView, LogMixin):\n    pass".to_string(),
            Language::Python,
        );
        let diag_map = HashMap::new();
        let count = merge_pyright_types(std::slice::from_mut(&mut sym), &diag_map);
        assert_eq!(count, 1);
        assert_eq!(sym.resolved_type.as_deref(), Some("APIView, LogMixin"));
    }

    #[test]
    fn run_pyright_enrichment_skips_without_python_symbols() {
        use crate::db::Database;
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("index.db");
        let db = Database::open(&db_path).expect("db open");
        // No Python symbols → must return 0 without touching pyright
        let count = run_pyright_enrichment(dir.path(), &mut [], &db);
        assert_eq!(count, 0);
    }

    // ── parse_pyright_diagnostics ─────────────────────────────────────────────

    #[test]
    fn parse_diagnostics_extracts_information_entries() {
        let workspace = tempfile::tempdir().unwrap();
        let abs_path = workspace.path().join("src/app.py");
        let json = serde_json::json!({
            "generalDiagnostics": [
                {
                    "severity": "information",
                    "file": abs_path.to_str().unwrap(),
                    "message": "Type of \"compute\" is \"str\"",
                    "range": { "start": { "line": 4, "character": 0 } }
                }
            ]
        })
        .to_string();
        let map = parse_pyright_diagnostics(&json, workspace.path());
        assert_eq!(map.len(), 1);
        let key = ("src/app.py".to_string(), 5); // 0-based → 1-based
        assert_eq!(map[&key].resolved_type, "str");
    }

    #[test]
    fn parse_diagnostics_ignores_non_information_severity() {
        let workspace = tempfile::tempdir().unwrap();
        let abs_path = workspace.path().join("src/app.py");
        let json = serde_json::json!({
            "generalDiagnostics": [
                {
                    "severity": "error",
                    "file": abs_path.to_str().unwrap(),
                    "message": "Type of \"x\" is \"str\"",
                    "range": { "start": { "line": 0, "character": 0 } }
                },
                {
                    "severity": "warning",
                    "file": abs_path.to_str().unwrap(),
                    "message": "Type of \"y\" is \"int\"",
                    "range": { "start": { "line": 1, "character": 0 } }
                }
            ]
        })
        .to_string();
        let map = parse_pyright_diagnostics(&json, workspace.path());
        assert!(
            map.is_empty(),
            "error/warning diagnostics should be ignored"
        );
    }

    #[test]
    fn parse_diagnostics_handles_malformed_json() {
        let workspace = tempfile::tempdir().unwrap();
        let map = parse_pyright_diagnostics("not json at all", workspace.path());
        assert!(map.is_empty());
    }

    #[test]
    fn parse_diagnostics_handles_empty_diagnostics_array() {
        let workspace = tempfile::tempdir().unwrap();
        let json = r#"{"generalDiagnostics": []}"#;
        let map = parse_pyright_diagnostics(json, workspace.path());
        assert!(map.is_empty());
    }

    // ── python_files_hash ─────────────────────────────────────────────────────

    #[test]
    fn python_files_hash_changes_when_file_added() {
        let dir = tempfile::tempdir().unwrap();
        let h1 = python_files_hash(dir.path());
        std::fs::write(dir.path().join("mod.py"), "def f(): pass").unwrap();
        let h2 = python_files_hash(dir.path());
        assert_ne!(h1, h2, "hash must change after adding a .py file");
    }

    #[test]
    fn python_files_hash_ignores_non_python_files() {
        let dir = tempfile::tempdir().unwrap();
        let h1 = python_files_hash(dir.path());
        std::fs::write(dir.path().join("README.md"), "hello").unwrap();
        let h2 = python_files_hash(dir.path());
        assert_eq!(h1, h2, "hash must not change for non-.py files");
    }

    #[test]
    fn python_files_hash_skips_pycache_dir() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("__pycache__");
        std::fs::create_dir(&cache).unwrap();
        std::fs::write(cache.join("mod.cpython-311.pyc"), "bytecode").unwrap();
        // Also write a .py file in __pycache__ (shouldn't happen in practice but
        // ensures the directory skip is applied before checking extensions)
        std::fs::write(cache.join("hidden.py"), "").unwrap();
        let h1 = python_files_hash(dir.path());
        // Modifying content inside __pycache__ must not affect the hash
        std::fs::write(cache.join("hidden.py"), "changed").unwrap();
        let h2 = python_files_hash(dir.path());
        assert_eq!(h1, h2, "__pycache__ contents must be ignored");
    }

    // ── incremental cache ─────────────────────────────────────────────────────

    #[test]
    fn enrichment_cache_hit_returns_zero() {
        use crate::db::Database;
        use crate::symbol::Symbol;
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("index.db");
        let db = Database::open(&db_path).expect("db open");

        // Write a Python file so the hash is non-trivial
        std::fs::write(dir.path().join("app.py"), "def f(): pass").unwrap();
        let py_hash = python_files_hash(dir.path());

        // Prime the cache with the current hash
        db.set_macro_expand_hash("__pyright__", &py_hash).unwrap();

        // Build a Python symbol so Gate 1 passes
        let mut sym = Symbol::new(
            "app.py",
            "f",
            SymbolKind::Function,
            1,
            1,
            "def f():".to_string(),
            None,
            "def f(): pass".to_string(),
            Language::Python,
        );

        // Even with pyright not installed, the cache hit gate fires first →
        // returns 0 without attempting to call pyright.
        let count = run_pyright_enrichment(dir.path(), std::slice::from_mut(&mut sym), &db);
        assert_eq!(count, 0, "cache hit should short-circuit to 0");
        assert!(
            sym.resolved_type.is_none(),
            "symbol must not be mutated on cache hit"
        );
    }
}
