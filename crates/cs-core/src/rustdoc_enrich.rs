//! rustdoc JSON resolved-type enrichment pass.
//!
//! Runs `cargo +nightly doc --output-format json --no-deps` against the
//! workspace and merges resolved return-types and trait-impl lists into
//! existing symbols.
//!
//! Enabled by `[indexing] rust_rustdoc_types = true` in
//! `.codesurgeon/config.toml`. Skipped gracefully when:
//! - no nightly Rust toolchain is available
//! - `Cargo.toml` is absent from the workspace root
//! - the `cargo doc` invocation fails for any reason
//!
//! Incremental: re-run is gated on `Cargo.lock` content hash. When the lock
//! file hasn't changed the enrichment pass is skipped entirely and existing
//! `resolved_type` values are preserved in the DB.

use crate::db::Database;
use crate::symbol::Symbol;
use crate::watcher::hash_content;
use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the rustdoc enrichment pass.
///
/// Mutates symbols in `all_symbols` in-place by setting `resolved_type` and
/// `source = "rustdoc"` on matched entries.  Returns the number of symbols
/// enriched so the caller can log/stat.
///
/// Silently returns 0 when:
/// - `Cargo.toml` is absent from `workspace_root`
/// - nightly Rust is not installed
/// - `Cargo.lock` hash is unchanged since last run (incremental skip)
/// - `cargo doc` subprocess fails (logged at WARN)
pub fn run_rustdoc_enrichment(
    workspace_root: &Path,
    all_symbols: &mut [Symbol],
    db: &Database,
) -> usize {
    // Gate 1: workspace must have a Cargo.toml
    if !workspace_root.join("Cargo.toml").exists() {
        return 0;
    }

    // Gate 2: nightly toolchain must be available
    if !nightly_available() {
        tracing::info!(
            "nightly Rust not found — skipping rustdoc enrichment. \
             Install with: rustup toolchain install nightly"
        );
        return 0;
    }

    // Gate 3: incremental — skip if Cargo.lock hasn't changed
    let lock_hash = cargo_lock_hash(workspace_root);
    match db.get_macro_expand_hash("__rustdoc__") {
        Ok(Some(cached)) if cached == lock_hash => {
            tracing::debug!("rustdoc-enrich cache hit (Cargo.lock unchanged)");
            return 0;
        }
        _ => {}
    }

    // Run cargo doc and get the JSON file path
    let json_path = match run_cargo_doc(workspace_root) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("rustdoc-enrich: cargo doc failed: {}", e);
            return 0;
        }
    };

    // Parse the rustdoc JSON
    let resolved_map = match parse_rustdoc_json(&json_path) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                "rustdoc-enrich: failed to parse {}: {}",
                json_path.display(),
                e
            );
            return 0;
        }
    };

    // Merge into symbols
    let count = merge_resolved_types(all_symbols, &resolved_map);

    // Update cache
    if let Err(e) = db.set_macro_expand_hash("__rustdoc__", &lock_hash) {
        tracing::warn!("rustdoc-enrich cache write failed: {}", e);
    }

    count
}

// ── Detection helpers ─────────────────────────────────────────────────────────

static NIGHTLY_AVAILABLE: OnceLock<bool> = OnceLock::new();

fn nightly_available() -> bool {
    *NIGHTLY_AVAILABLE.get_or_init(|| {
        Command::new("cargo")
            .args(["+nightly", "--version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Compute a hash of `Cargo.lock` for incremental gating.
/// Returns a placeholder string if the file is absent or unreadable.
fn cargo_lock_hash(workspace_root: &Path) -> String {
    let lock_path = workspace_root.join("Cargo.lock");
    std::fs::read(lock_path)
        .map(|b| hash_content(&b))
        .unwrap_or_else(|_| "no_cargo_lock".to_string())
}

// ── cargo doc subprocess ──────────────────────────────────────────────────────

/// Run `cargo +nightly doc --output-format json --no-deps` and return the
/// path to the generated JSON file.
fn run_cargo_doc(workspace_root: &Path) -> Result<PathBuf> {
    let output = Command::new("cargo")
        .args([
            "+nightly",
            "doc",
            "--output-format",
            "json",
            "--no-deps",
            "--quiet",
        ])
        .env("RUSTDOCFLAGS", "-Zunstable-options --output-format json")
        .env("CARGO_TERM_COLOR", "never")
        .current_dir(workspace_root)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "cargo +nightly doc exited {}: {}",
            output.status,
            stderr.trim()
        );
    }

    // Find the produced JSON file under target/doc/<crate>.json
    let doc_dir = workspace_root.join("target").join("doc");
    let json_file = std::fs::read_dir(&doc_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .ok_or_else(|| anyhow::anyhow!("No .json file found in {}", doc_dir.display()))?;

    Ok(json_file)
}

// ── JSON parsing ──────────────────────────────────────────────────────────────

/// Information extracted from rustdoc for a single item.
#[derive(Debug)]
struct ResolvedInfo {
    /// For functions: the resolved return type as a string.
    /// For types: comma-separated list of directly-implemented traits.
    resolved_type: String,
}

/// Parse the rustdoc JSON and return a map of `module_path → ResolvedInfo`.
///
/// We use `serde_json::Value` rather than the `rustdoc-types` crate so we
/// are not locked to a specific format version — the nightly JSON format
/// evolves with each Rust release.
///
/// Key format: last two path segments joined by `::` (e.g. `"MyStruct::new"`
/// for a method, `"my_fn"` for a top-level function, `"MyStruct"` for a type).
fn parse_rustdoc_json(path: &Path) -> Result<HashMap<String, ResolvedInfo>> {
    let text = std::fs::read_to_string(path)?;
    let root: serde_json::Value = serde_json::from_str(&text)?;

    let index = root
        .get("index")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow::anyhow!("rustdoc JSON missing `index` object"))?;
    let paths = root.get("paths").and_then(|v| v.as_object());

    let mut map: HashMap<String, ResolvedInfo> = HashMap::new();

    for (id, item) in index {
        let kind = item.get("kind").and_then(|v| v.as_str()).unwrap_or("");

        match kind {
            "function" | "method" => {
                if let Some(info) = extract_function_info(item) {
                    let key = path_key(id, item, paths);
                    map.insert(key, info);
                }
            }
            "struct" | "enum" | "union" => {
                if let Some(info) = extract_type_impl_info(id, index) {
                    let key = path_key(id, item, paths);
                    map.insert(key, info);
                }
            }
            _ => {}
        }
    }

    Ok(map)
}

/// Build a short lookup key for a rustdoc item.
/// Strategy: use the `paths` table if available (gives full crate-relative
/// path), otherwise fall back to the item `name`.
fn path_key(
    id: &str,
    item: &serde_json::Value,
    paths: Option<&serde_json::Map<String, serde_json::Value>>,
) -> String {
    if let Some(paths) = paths {
        if let Some(path_entry) = paths.get(id) {
            if let Some(arr) = path_entry.get("path").and_then(|v| v.as_array()) {
                let segs: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                // Use last two segments for matching against our FQNs
                // (crate name stripped, last path components kept).
                let start = if segs.len() > 2 { segs.len() - 2 } else { 0 };
                return segs[start..].join("::");
            }
        }
    }
    item.get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Extract the return type of a function item as a human-readable string.
fn extract_function_info(item: &serde_json::Value) -> Option<ResolvedInfo> {
    let output = item
        .pointer("/inner/sig/output")
        .or_else(|| item.pointer("/inner/decl/output"))?;
    let type_str = type_to_string(output);
    if type_str.is_empty() || type_str == "()" {
        return None;
    }
    Some(ResolvedInfo {
        resolved_type: type_str,
    })
}

/// Find all trait impls for a type and return them as a comma-separated string.
fn extract_type_impl_info(
    type_id: &str,
    index: &serde_json::Map<String, serde_json::Value>,
) -> Option<ResolvedInfo> {
    let mut traits: Vec<String> = Vec::new();

    for item in index.values() {
        let kind = item.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        if kind != "impl" {
            continue;
        }
        // The impl's `for` field should match our type id.
        let for_id = item
            .pointer("/inner/for/resolved_path/id")
            .and_then(|v| v.as_str());
        if for_id != Some(type_id) {
            continue;
        }
        // Extract the trait name if present (None = inherent impl).
        if let Some(trait_name) = item
            .pointer("/inner/trait/resolved_path/name")
            .and_then(|v| v.as_str())
        {
            // Strip common crate prefixes to keep it short.
            let short = trait_name.rsplit("::").next().unwrap_or(trait_name);
            traits.push(short.to_string());
        }
    }

    if traits.is_empty() {
        return None;
    }
    traits.sort();
    traits.dedup();
    Some(ResolvedInfo {
        resolved_type: traits.join(", "),
    })
}

/// Convert a rustdoc `Type` JSON value to a human-readable string.
fn type_to_string(v: &serde_json::Value) -> String {
    if let Some(name) = v
        .get("resolved_path")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
    {
        // Include generic args if present
        if let Some(args) = v.pointer("/resolved_path/args/angle_bracketed/args") {
            if let Some(arr) = args.as_array() {
                if !arr.is_empty() {
                    let inner: Vec<String> = arr
                        .iter()
                        .filter_map(|a| a.get("type").map(type_to_string))
                        .collect();
                    if !inner.is_empty() {
                        return format!("{}<{}>", name, inner.join(", "));
                    }
                }
            }
        }
        return name.to_string();
    }
    if let Some(inner) = v.get("borrowed_ref") {
        let lt = inner.get("lifetime").and_then(|v| v.as_str()).unwrap_or("");
        let mutable = inner
            .get("is_mutable")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let inner_type = inner.get("type").map(type_to_string).unwrap_or_default();
        let prefix = if mutable { "&mut " } else { "&" };
        let lt_part = if lt.is_empty() {
            String::new()
        } else {
            format!("{} ", lt)
        };
        return format!("{}{}{}", prefix, lt_part, inner_type);
    }
    if let Some(inner) = v.get("tuple") {
        if let Some(arr) = inner.as_array() {
            if arr.is_empty() {
                return "()".to_string();
            }
            let parts: Vec<String> = arr.iter().map(type_to_string).collect();
            return format!("({})", parts.join(", "));
        }
    }
    if let Some(inner) = v.get("slice") {
        return format!("[{}]", type_to_string(inner));
    }
    if v.get("primitive").and_then(|v| v.as_str()).is_some() {
        return v["primitive"].as_str().unwrap_or("").to_string();
    }
    String::new()
}

// ── Merge pass ────────────────────────────────────────────────────────────────

/// Merge rustdoc resolved-type information into `symbols`.
///
/// Matching strategy: for each symbol, try to find a rustdoc entry whose key
/// is a suffix of the symbol's `fqn` (with `::` boundary alignment).
/// This handles file-path-prefixed FQNs like `src/engine.rs::CoreEngine::new`.
///
/// Returns the number of symbols that were enriched.
fn merge_resolved_types(symbols: &mut [Symbol], map: &HashMap<String, ResolvedInfo>) -> usize {
    let mut count = 0;
    for sym in symbols.iter_mut() {
        // Only enrich Rust symbols.
        if sym.language != crate::language::Language::Rust {
            continue;
        }
        if let Some(info) = find_match(&sym.fqn, &sym.name, map) {
            sym.resolved_type = Some(info.resolved_type.clone());
            if sym.source.is_none() {
                sym.source = Some("rustdoc".to_string());
            }
            count += 1;
        }
    }
    count
}

/// Find a rustdoc `ResolvedInfo` entry that matches the given FQN.
fn find_match<'a>(
    fqn: &str,
    name: &str,
    map: &'a HashMap<String, ResolvedInfo>,
) -> Option<&'a ResolvedInfo> {
    // 1. Direct name match (most common for top-level items)
    if let Some(v) = map.get(name) {
        return Some(v);
    }

    // 2. Try each key: check if fqn ends with "::<key>" (suffix match)
    for (key, info) in map {
        if fqn_ends_with(fqn, key) {
            return Some(info);
        }
    }
    None
}

/// Returns true if `fqn` ends with `suffix` at a `::` boundary.
/// e.g. fqn = "src/engine.rs::CoreEngine::new", suffix = "CoreEngine::new" → true
fn fqn_ends_with(fqn: &str, suffix: &str) -> bool {
    if fqn == suffix {
        return true;
    }
    if let Some(pos) = fqn.rfind(suffix) {
        let before = &fqn[..pos];
        return before.ends_with("::");
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_to_string_primitive() {
        let v = serde_json::json!({"primitive": "bool"});
        assert_eq!(type_to_string(&v), "bool");
    }

    #[test]
    fn type_to_string_resolved_path() {
        let v = serde_json::json!({
            "resolved_path": { "name": "String", "id": "0:0", "args": null }
        });
        assert_eq!(type_to_string(&v), "String");
    }

    #[test]
    fn type_to_string_tuple_unit() {
        let v = serde_json::json!({"tuple": []});
        assert_eq!(type_to_string(&v), "()");
    }

    #[test]
    fn type_to_string_option() {
        let v = serde_json::json!({
            "resolved_path": {
                "name": "Option",
                "id": "0:0",
                "args": {
                    "angle_bracketed": {
                        "args": [{"type": {"primitive": "u32"}}],
                        "bindings": []
                    }
                }
            }
        });
        assert_eq!(type_to_string(&v), "Option<u32>");
    }

    #[test]
    fn fqn_ends_with_suffix() {
        assert!(fqn_ends_with(
            "src/engine.rs::CoreEngine::new",
            "CoreEngine::new"
        ));
        assert!(fqn_ends_with("src/lib.rs::my_fn", "my_fn"));
        assert!(!fqn_ends_with("src/lib.rs::my_fn", "other_fn"));
        // Must respect :: boundary — partial name match should fail
        assert!(!fqn_ends_with("src/lib.rs::my_fn", "fn"));
    }

    #[test]
    fn nightly_gate_skips_without_cargo_toml() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("index.db");
        let db = crate::db::Database::open(&db_path).expect("db");
        // No Cargo.toml → must return 0 without panicking
        let count = run_rustdoc_enrichment(dir.path(), &mut [], &db);
        assert_eq!(count, 0);
    }

    #[test]
    fn rustdoc_incremental_cache_hit() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("index.db");
        let db = crate::db::Database::open(&db_path).expect("db");

        // Gate 1: Cargo.toml present.
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname=\"x\"\n").unwrap();

        // Prime the DB with the current Cargo.lock hash.
        std::fs::write(
            dir.path().join("Cargo.lock"),
            b"# This file is automatically @generated",
        )
        .unwrap();
        let lock_hash = cargo_lock_hash(dir.path());
        db.set_macro_expand_hash("__rustdoc__", &lock_hash).unwrap();

        // Gate 3 fires: must return 0 without running cargo doc.
        let count = run_rustdoc_enrichment(dir.path(), &mut [], &db);
        assert_eq!(count, 0, "cache hit must short-circuit to 0");
    }

    fn make_rust_sym(name: &str) -> crate::symbol::Symbol {
        use crate::language::Language;
        use crate::symbol::{Symbol, SymbolKind};
        let mut sym = Symbol::new(
            "src/lib.rs",
            name,
            SymbolKind::Function,
            1,
            5,
            format!("fn {}() -> String", name),
            None,
            format!("fn {}() -> String {{ String::new() }}", name),
            Language::Rust,
        );
        sym.fqn = format!("src/lib.rs::{}", name);
        sym
    }

    #[test]
    fn merge_enriches_rust_symbol_by_name() {
        let mut map = HashMap::new();
        map.insert(
            "my_fn".to_string(),
            ResolvedInfo {
                resolved_type: "String".to_string(),
            },
        );
        let mut sym = make_rust_sym("my_fn");
        let count = merge_resolved_types(std::slice::from_mut(&mut sym), &map);
        assert_eq!(count, 1);
        assert_eq!(sym.resolved_type.as_deref(), Some("String"));
        assert_eq!(sym.source.as_deref(), Some("rustdoc"));
    }

    #[test]
    fn merge_enriches_rust_symbol_by_fqn_suffix() {
        let mut map = HashMap::new();
        map.insert(
            "CoreEngine::new".to_string(),
            ResolvedInfo {
                resolved_type: "Self".to_string(),
            },
        );
        let mut sym = make_rust_sym("new");
        sym.fqn = "src/engine.rs::CoreEngine::new".to_string();
        let count = merge_resolved_types(std::slice::from_mut(&mut sym), &map);
        assert_eq!(count, 1);
        assert_eq!(sym.resolved_type.as_deref(), Some("Self"));
    }

    #[test]
    fn merge_skips_non_rust_symbol() {
        use crate::language::Language;
        use crate::symbol::{Symbol, SymbolKind};
        let mut map = HashMap::new();
        map.insert(
            "compute".to_string(),
            ResolvedInfo {
                resolved_type: "String".to_string(),
            },
        );
        let mut sym = Symbol::new(
            "src/app.py",
            "compute",
            SymbolKind::Function,
            1,
            3,
            "def compute():".to_string(),
            None,
            "def compute(): pass".to_string(),
            Language::Python,
        );
        let count = merge_resolved_types(std::slice::from_mut(&mut sym), &map);
        assert_eq!(count, 0, "Python symbols must not be enriched by rustdoc");
        assert!(sym.resolved_type.is_none());
    }

    #[test]
    fn merge_preserves_existing_source() {
        let mut map = HashMap::new();
        map.insert(
            "my_fn".to_string(),
            ResolvedInfo {
                resolved_type: "u32".to_string(),
            },
        );
        let mut sym = make_rust_sym("my_fn");
        sym.source = Some("proc-macro".to_string());

        let count = merge_resolved_types(std::slice::from_mut(&mut sym), &map);
        assert_eq!(count, 1);
        assert_eq!(
            sym.source.as_deref(),
            Some("proc-macro"),
            "pre-existing source must not be overwritten"
        );
    }

    #[test]
    fn merge_empty_map_returns_zero() {
        let map: HashMap<String, ResolvedInfo> = HashMap::new();
        let mut sym = make_rust_sym("my_fn");
        let count = merge_resolved_types(std::slice::from_mut(&mut sym), &map);
        assert_eq!(count, 0);
        assert!(sym.resolved_type.is_none());
    }

    #[test]
    fn parse_rustdoc_json_malformed_returns_empty() {
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("crate.json");
        let mut f = std::fs::File::create(&json_path).unwrap();
        f.write_all(b"not valid json at all").unwrap();
        let result = parse_rustdoc_json(&json_path);
        // Must not panic; an Err is acceptable, empty map is also fine.
        match result {
            Ok(map) => assert!(map.is_empty(), "malformed JSON should yield empty map"),
            Err(_) => {} // parse error is acceptable
        }
    }

    #[test]
    fn parse_rustdoc_json_missing_index_returns_err() {
        use std::io::Write as _;
        let dir = tempfile::tempdir().unwrap();
        let json_path = dir.path().join("crate.json");
        let mut f = std::fs::File::create(&json_path).unwrap();
        f.write_all(br#"{"version": 1}"#).unwrap(); // valid JSON but no "index" key
        let result = parse_rustdoc_json(&json_path);
        assert!(result.is_err(), "missing `index` key must return an error");
    }

    #[test]
    fn fqn_ends_with_partial_name_no_match() {
        // "fn" is a suffix of "my_fn" but not at a :: boundary — must not match.
        assert!(!fqn_ends_with("src/lib.rs::my_fn", "fn"));
        // Empty suffix edge case.
        assert!(fqn_ends_with("a::b", "a::b"));
    }
}
