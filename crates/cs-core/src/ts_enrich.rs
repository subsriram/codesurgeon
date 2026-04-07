//! TypeScript/JavaScript compiler shim enrichment pass.
//!
//! At index time, invokes a small Node.js shim that uses the workspace's
//! existing `typescript` dev dependency to run `ts.createProgram()` +
//! `TypeChecker` and annotate symbols with resolved types.
//!
//! Enabled by `[indexing] ts_types = true` in `.codesurgeon/config.toml`.
//! Skipped gracefully when:
//! - `node` is not available on PATH
//! - no `tsconfig.json` found in workspace root
//! - `typescript` package not found in node_modules or globally
//!
//! Incremental: re-run is gated on `tsconfig.json` content hash. When the
//! tsconfig hasn't changed the pass is skipped and existing `resolved_type`
//! values are preserved in the DB.

use crate::db::Database;
use crate::language::Language;
use crate::symbol::Symbol;
use crate::watcher::hash_content;
use anyhow::Result;
use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

/// The Node.js enricher shim, embedded at compile time from `assets/ts-enricher.js`.
const TS_ENRICHER_JS: &str = include_str!("../assets/ts-enricher.js");

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the TypeScript compiler enrichment pass.
///
/// Mutates symbols in `all_symbols` in-place by setting `resolved_type` and
/// `source = "ts-compiler"` on matched entries.  Returns the number of symbols
/// enriched so the caller can log/stat.
///
/// Silently returns 0 when:
/// - `node` is not available
/// - `tsconfig.json` is absent from `workspace_root`
/// - `tsconfig.json` hash is unchanged since last run (incremental skip)
/// - the Node.js subprocess fails (logged at WARN)
pub fn run_ts_enrichment(
    workspace_root: &Path,
    all_symbols: &mut [Symbol],
    db: &Database,
) -> usize {
    // Gate 1: workspace must have a tsconfig.json
    let tsconfig_path = workspace_root.join("tsconfig.json");
    if !tsconfig_path.exists() {
        return 0;
    }

    // Gate 2: node must be available
    if !node_available() {
        tracing::info!("node not found on PATH — skipping TypeScript enrichment");
        return 0;
    }

    // Gate 3: incremental — skip if tsconfig.json hash is unchanged
    let tsconfig_hash = file_hash(&tsconfig_path);
    match db.get_macro_expand_hash("__ts_enrich__") {
        Ok(Some(cached)) if cached == tsconfig_hash => {
            tracing::debug!("ts-enrich cache hit (tsconfig.json unchanged)");
            return 0;
        }
        _ => {}
    }

    // Write the shim to a temp file
    let shim_path = match write_shim() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("ts-enrich: failed to write shim: {}", e);
            return 0;
        }
    };

    // Run the shim
    let stdout_text = match run_shim(&shim_path, workspace_root) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("ts-enrich: shim execution failed: {}", e);
            return 0;
        }
    };

    // Parse NDJSON output
    let resolved_map = parse_ndjson_output(&stdout_text);
    tracing::debug!(
        "ts-enrich: shim emitted {} type annotations",
        resolved_map.len()
    );

    // Merge into symbols
    let count = merge_resolved_types(all_symbols, &resolved_map);

    // Update cache regardless of whether any symbols were enriched, so we
    // don't re-spawn the compiler on every index when a project has no TS.
    if let Err(e) = db.set_macro_expand_hash("__ts_enrich__", &tsconfig_hash) {
        tracing::warn!("ts-enrich: cache write failed: {}", e);
    }

    count
}

// ── Detection helpers ─────────────────────────────────────────────────────────

static NODE_AVAILABLE: OnceLock<bool> = OnceLock::new();

fn node_available() -> bool {
    *NODE_AVAILABLE.get_or_init(|| {
        Command::new("node")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

fn file_hash(path: &Path) -> String {
    std::fs::read(path)
        .map(|b| hash_content(&b))
        .unwrap_or_else(|_| "no_file".to_string())
}

// ── Shim execution ────────────────────────────────────────────────────────────

/// Write the embedded JS shim to a well-known temp path and return it.
/// Overwrites on every call so the shim is always up-to-date after a
/// codesurgeon upgrade without requiring a cache flush.
fn write_shim() -> Result<std::path::PathBuf> {
    let tmp = std::env::temp_dir().join("codesurgeon-ts-enricher.js");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(TS_ENRICHER_JS.as_bytes())?;
    Ok(tmp)
}

/// Run `node <shim> <workspace_root>` and return stdout.
/// Stderr is forwarded to tracing at DEBUG level.
fn run_shim(shim_path: &Path, workspace_root: &Path) -> Result<String> {
    let output = Command::new("node")
        .arg(shim_path)
        .arg(workspace_root)
        .current_dir(workspace_root)
        .output()?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.trim().is_empty() {
        tracing::debug!("ts-enrich shim stderr: {}", stderr.trim());
    }

    if !output.status.success() {
        anyhow::bail!("ts-enricher.js exited {}: {}", output.status, stderr.trim());
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

// ── NDJSON parsing ────────────────────────────────────────────────────────────

/// Resolved type information for a single symbol emitted by the shim.
#[derive(Debug)]
struct TsResolvedInfo {
    resolved_type: String,
}

/// Parse NDJSON output from the shim into a map of `fqn → TsResolvedInfo`.
fn parse_ndjson_output(output: &str) -> HashMap<String, TsResolvedInfo> {
    let mut map = HashMap::new();
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(val) = serde_json::from_str::<serde_json::Value>(line) else {
            tracing::debug!("ts-enrich: skipping malformed NDJSON line: {}", line);
            continue;
        };
        let Some(fqn) = val.get("fqn").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(resolved_type) = val.get("resolved_type").and_then(|v| v.as_str()) else {
            continue;
        };
        if resolved_type.is_empty() {
            continue;
        }
        map.insert(
            fqn.to_string(),
            TsResolvedInfo {
                resolved_type: resolved_type.to_string(),
            },
        );
    }
    map
}

// ── Merge pass ────────────────────────────────────────────────────────────────

/// Merge shim output into `symbols`, returning the count enriched.
fn merge_resolved_types(symbols: &mut [Symbol], map: &HashMap<String, TsResolvedInfo>) -> usize {
    let mut count = 0;
    for sym in symbols.iter_mut() {
        if !is_ts_js(&sym.language) {
            continue;
        }
        if let Some(info) = find_match(&sym.fqn, &sym.name, map) {
            sym.resolved_type = Some(info.resolved_type.clone());
            if sym.source.is_none() {
                sym.source = Some("ts-compiler".to_string());
            }
            count += 1;
        }
    }
    count
}

fn is_ts_js(lang: &Language) -> bool {
    matches!(
        lang,
        Language::TypeScript | Language::Tsx | Language::JavaScript | Language::Jsx
    )
}

fn find_match<'a>(
    fqn: &str,
    name: &str,
    map: &'a HashMap<String, TsResolvedInfo>,
) -> Option<&'a TsResolvedInfo> {
    // 1. Exact FQN match — the shim outputs full relative-path FQNs, so this
    //    is the common case when both sides agree on the file path.
    if let Some(v) = map.get(fqn) {
        return Some(v);
    }

    // 2. Suffix match — handles path normalisation differences.
    //    e.g. shim emits `src/foo.ts::Cls::method`, symbol FQN is
    //    `./src/foo.ts::Cls::method` or similar.
    for (key, info) in map {
        if fqn_ends_with(fqn, key) {
            return Some(info);
        }
    }

    // 3. Name-only fallback for simple top-level functions.
    if let Some(v) = map.get(name) {
        return Some(v);
    }

    None
}

/// Returns true if `fqn` ends with `suffix` at a `::` segment boundary.
///
/// e.g. `"src/foo.ts::Cls::method"` ends with `"Cls::method"` → true
///      `"src/foo.ts::method"` ends with `"ethod"` → false (no boundary)
fn fqn_ends_with(fqn: &str, suffix: &str) -> bool {
    if fqn == suffix {
        return true;
    }
    if let Some(rest) = fqn.strip_suffix(suffix) {
        return rest.ends_with("::");
    }
    false
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::Language;
    use crate::symbol::{Symbol, SymbolKind};

    fn make_sym(fqn: &str, name: &str, lang: Language) -> Symbol {
        Symbol::new(
            fqn.split("::").next().unwrap_or("file.ts"),
            name,
            SymbolKind::Function,
            1,
            10,
            format!("function {}() {{}}", name),
            None,
            String::new(),
            lang,
        )
    }

    #[test]
    fn parse_ndjson_basic() {
        let input = concat!(
            r#"{"fqn":"src/foo.ts::MyClass::myMethod","resolved_type":"Promise<string>","line":10}"#,
            "\n",
            r#"{"fqn":"src/bar.ts::greet","resolved_type":"string","line":3}"#,
            "\n",
        );
        let map = parse_ndjson_output(input);
        assert_eq!(map.len(), 2);
        assert_eq!(
            map["src/foo.ts::MyClass::myMethod"].resolved_type,
            "Promise<string>"
        );
        assert_eq!(map["src/bar.ts::greet"].resolved_type, "string");
    }

    #[test]
    fn parse_ndjson_skips_empty_type() {
        let input = concat!(
            r#"{"fqn":"src/foo.ts::fn1","resolved_type":"","line":1}"#,
            "\n",
            r#"{"fqn":"src/foo.ts::fn2","resolved_type":"number","line":2}"#,
            "\n",
        );
        let map = parse_ndjson_output(input);
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("src/foo.ts::fn2"));
    }

    #[test]
    fn parse_ndjson_skips_malformed() {
        let input = "not json\n{\"fqn\":\"a::b\",\"resolved_type\":\"T\"}\n";
        let map = parse_ndjson_output(input);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn fqn_ends_with_positive() {
        assert!(fqn_ends_with(
            "src/foo.ts::MyClass::method",
            "MyClass::method"
        ));
        assert!(fqn_ends_with("src/bar.ts::greet", "greet"));
        assert!(fqn_ends_with("a::b::c", "b::c"));
    }

    #[test]
    fn fqn_ends_with_negative() {
        // Must respect :: boundary — a partial name suffix must not match.
        assert!(!fqn_ends_with("src/bar.ts::greet", "reet"));
        assert!(!fqn_ends_with("src/a.ts::fn1", "fn2"));
        // Ensure we don't match mid-segment.
        assert!(!fqn_ends_with("src/foo.ts::method", "ethod"));
    }

    #[test]
    fn fqn_ends_with_repeated_segment_name() {
        // The segment name "foo" appears in both file path and as the symbol name.
        // strip_suffix must match at the END, not the first occurrence found by find().
        assert!(fqn_ends_with("src/foo.ts::foo::foo", "foo"));
        assert!(fqn_ends_with("src/foo.ts::foo::foo", "foo::foo"));
        // But a prefix of the repeated name must not match.
        assert!(!fqn_ends_with("src/foo.ts::foo::foo", "oo"));
    }

    #[test]
    fn gate_skips_without_tsconfig() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("index.db");
        let db = crate::db::Database::open(&db_path).expect("db open");
        // No tsconfig.json → must return 0 without panicking.
        let count = run_ts_enrichment(dir.path(), &mut [], &db);
        assert_eq!(count, 0);
    }

    #[test]
    fn ts_incremental_cache_hit() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("index.db");
        let db = crate::db::Database::open(&db_path).expect("db open");

        // Gate 1: tsconfig.json must exist for gate 1 to pass.
        let tsconfig = dir.path().join("tsconfig.json");
        std::fs::write(&tsconfig, r#"{"compilerOptions":{}}"#).unwrap();

        // Prime the DB with the hash of tsconfig.json, exactly as run_ts_enrichment does.
        let hash = file_hash(&tsconfig);
        db.set_macro_expand_hash("__ts_enrich__", &hash).unwrap();

        // Gate 3 fires: must return 0 without invoking node, regardless of node availability.
        let mut syms = vec![make_sym("src/a.ts::fn1", "fn1", Language::TypeScript)];
        let count = run_ts_enrichment(dir.path(), &mut syms, &db);
        assert_eq!(count, 0, "cache hit must short-circuit to 0");
        assert!(
            syms[0].resolved_type.is_none(),
            "symbol must not be mutated on cache hit"
        );
    }

    #[test]
    fn merge_preserves_existing_source() {
        // When a symbol already has source set, the merge pass must not overwrite it.
        let mut map = HashMap::new();
        map.insert(
            "src/foo.ts::fn1".to_string(),
            TsResolvedInfo {
                resolved_type: "string".to_string(),
            },
        );
        let mut sym = make_sym("src/foo.ts::fn1", "fn1", Language::TypeScript);
        sym.fqn = "src/foo.ts::fn1".to_string();
        sym.source = Some("lsp".to_string());

        let count = merge_resolved_types(std::slice::from_mut(&mut sym), &map);
        assert_eq!(count, 1, "symbol must still be counted as enriched");
        assert_eq!(
            sym.source.as_deref(),
            Some("lsp"),
            "pre-existing source must not be overwritten"
        );
        assert_eq!(sym.resolved_type.as_deref(), Some("string"));
    }

    #[test]
    fn merge_empty_map_returns_zero() {
        let map: HashMap<String, TsResolvedInfo> = HashMap::new();
        let mut sym = make_sym("src/foo.ts::fn1", "fn1", Language::TypeScript);
        sym.fqn = "src/foo.ts::fn1".to_string();
        let count = merge_resolved_types(std::slice::from_mut(&mut sym), &map);
        assert_eq!(count, 0);
        assert!(sym.resolved_type.is_none());
    }

    #[test]
    fn merge_name_only_fallback() {
        // Map key is the bare symbol name; FQN is path-qualified.
        // find_match resolves this via suffix matching (step 2).
        let mut map = HashMap::new();
        map.insert(
            "greet".to_string(),
            TsResolvedInfo {
                resolved_type: "void".to_string(),
            },
        );
        let mut sym = make_sym("src/utils.ts::greet", "greet", Language::TypeScript);
        sym.fqn = "src/utils.ts::greet".to_string();
        let count = merge_resolved_types(std::slice::from_mut(&mut sym), &map);
        assert_eq!(count, 1, "name-only fallback must enrich the symbol");
        assert_eq!(sym.resolved_type.as_deref(), Some("void"));
    }

    #[test]
    fn merge_only_ts_js_symbols() {
        let mut map = HashMap::new();
        map.insert(
            "src/foo.ts::fn1".to_string(),
            TsResolvedInfo {
                resolved_type: "string".to_string(),
            },
        );

        let mut syms = vec![
            make_sym("src/foo.rs::fn1", "fn1", Language::Rust),
            make_sym("src/foo.ts::fn1", "fn1", Language::TypeScript),
        ];
        // Fix up FQNs to match the map keys (Symbol::new builds its own FQN).
        syms[1].fqn = "src/foo.ts::fn1".to_string();

        let count = merge_resolved_types(&mut syms, &map);
        assert_eq!(count, 1);
        assert!(
            syms[0].resolved_type.is_none(),
            "Rust symbol must not be enriched"
        );
        assert_eq!(syms[1].resolved_type.as_deref(), Some("string"));
        assert_eq!(syms[1].source.as_deref(), Some("ts-compiler"));
    }

    #[test]
    fn merge_jsx_tsx_enriched() {
        let mut map = HashMap::new();
        map.insert(
            "src/Comp.tsx::render".to_string(),
            TsResolvedInfo {
                resolved_type: "JSX.Element".to_string(),
            },
        );
        let mut syms = vec![make_sym("src/Comp.tsx::render", "render", Language::Tsx)];
        syms[0].fqn = "src/Comp.tsx::render".to_string();

        let count = merge_resolved_types(&mut syms, &map);
        assert_eq!(count, 1);
        assert_eq!(syms[0].resolved_type.as_deref(), Some("JSX.Element"));
    }
}
