//! cargo-expand macro enrichment pass.
//!
//! After the base tree-sitter index is built, this pass identifies Rust source
//! files that contain proc-macro or derive invocations, runs `cargo expand
//! <module>` on them, and adds the generated symbols to the index.
//!
//! Enabled by `[indexing] rust_expand_macros = true` in `.codesurgeon/config.toml`.
//! Skipped gracefully if `cargo-expand` is not installed.
//! Gated on `Cargo.toml` presence in the workspace root.

use crate::db::Database;
use crate::indexer::parse_rust_source;
use crate::symbol::Symbol;
use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the macro enrichment pass over the Rust files represented by
/// `file_data`.  Each tuple is `(rel_path, file_hash, symbols)`.
///
/// Returns additional `Symbol`s with `source = Some("macro_expanded")` to be
/// appended to the main index.  Symbols already present in the base pass (by
/// FQN) are excluded to avoid duplicates.
///
/// Silently returns an empty `Vec` when:
/// - `Cargo.toml` is absent from `workspace_root`
/// - `cargo-expand` is not installed
/// - the expand step fails for a given file (logged at WARN)
pub fn run_macro_enrichment(
    workspace_root: &Path,
    file_data: &[(String, String, Vec<Symbol>)],
    db: &Database,
) -> Vec<Symbol> {
    // Gate 1: workspace must have a Cargo.toml
    if !workspace_root.join("Cargo.toml").exists() {
        return vec![];
    }

    // Gate 2: cargo-expand must be installed
    if !cargo_expand_available() {
        tracing::info!(
            "cargo-expand not found — skipping macro enrichment. \
             Install with: cargo install cargo-expand"
        );
        return vec![];
    }

    // Collect FQNs already in the base index so we can exclude duplicates.
    let base_fqns: HashSet<String> = file_data
        .iter()
        .flat_map(|(_, _, syms)| syms.iter().map(|s| s.fqn.clone()))
        .collect();

    let mut result: Vec<Symbol> = Vec::new();

    for (rel_path, file_hash, base_symbols) in file_data {
        // Only process Rust source files.
        if !rel_path.ends_with(".rs") {
            continue;
        }
        // Only process files that have proc-macro / derive invocations.
        // We check the body of all symbols from this file; if any contain
        // attribute syntax that cargo-expand would expand, proceed.
        if !base_symbols
            .iter()
            .any(|s| body_has_macro_invocation(&s.body))
        {
            continue;
        }

        // Incremental: skip if the file hasn't changed since last expansion.
        match db.get_macro_expand_hash(rel_path) {
            Ok(Some(cached)) if cached == *file_hash => {
                tracing::debug!("macro-expand cache hit: {}", rel_path);
                continue;
            }
            _ => {}
        }

        match expand_and_extract(workspace_root, rel_path, &base_fqns) {
            Ok(mut syms) => {
                tracing::info!(
                    "macro-expand: {} new symbol(s) from {}",
                    syms.len(),
                    rel_path
                );
                // Update cache whether or not we got new symbols.
                if let Err(e) = db.set_macro_expand_hash(rel_path, file_hash) {
                    tracing::warn!("macro-expand cache write failed for {}: {}", rel_path, e);
                }
                result.append(&mut syms);
            }
            Err(e) => {
                tracing::warn!("macro-expand failed for {}: {}", rel_path, e);
            }
        }
    }

    result
}

// ── Internal helpers ──────────────────────────────────────────────────────────

static CARGO_EXPAND_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Returns true if `cargo expand` can be invoked on this machine.
fn cargo_expand_available() -> bool {
    *CARGO_EXPAND_AVAILABLE.get_or_init(|| {
        Command::new("cargo")
            .args(["expand", "--version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Returns true if the symbol body contains a proc-macro or derive attribute
/// that `cargo-expand` would expand.
fn body_has_macro_invocation(body: &str) -> bool {
    body.contains("#[derive(")
        || body.contains("#[proc_macro")
        || body.contains("#[async_trait")
        || body.contains("#[tokio::main")
        || body.contains("#[actix_web::")
        || body.contains("#[rocket::")
        || body.contains("#[error(")   // thiserror
        || body.contains("thiserror")
}

/// Derive the `cargo expand` module argument from a relative file path.
///
/// Examples:
/// - `src/lib.rs`      → no module arg (expands root)
/// - `src/main.rs`     → no module arg (expands root)
/// - `src/foo.rs`      → `["foo"]`
/// - `src/foo/mod.rs`  → `["foo"]`
/// - `src/foo/bar.rs`  → `["foo::bar"]`
fn file_path_to_expand_args(rel_path: &str) -> Vec<String> {
    // Normalise to forward slashes on all platforms.
    let rel = rel_path.replace('\\', "/");

    // Strip `src/` prefix; if the file is not under src/, skip expansion.
    let Some(after_src) = rel.strip_prefix("src/") else {
        return vec![];
    };

    if after_src == "lib.rs" || after_src == "main.rs" {
        return vec![];
    }

    // e.g. "foo/bar.rs" → "foo::bar"
    //      "foo/mod.rs" → "foo"
    let without_ext = after_src.trim_end_matches(".rs");
    let module_path = without_ext
        .replace("/mod", "") // strip trailing /mod
        .replace('/', "::"); // remaining slashes → ::

    if module_path.is_empty() {
        vec![]
    } else {
        vec![module_path]
    }
}

/// Run `cargo expand [<module>]` in `workspace_root` and parse the output.
/// Returns only symbols whose FQN is not already in `base_fqns`, marked with
/// `source = Some("macro_expanded")`.
fn expand_and_extract(
    workspace_root: &Path,
    rel_path: &str,
    base_fqns: &HashSet<String>,
) -> Result<Vec<Symbol>> {
    let module_args = file_path_to_expand_args(rel_path);

    // If we couldn't derive a module path, skip.
    // We still attempt lib/main (empty args) for root files.
    let mut cmd = Command::new("cargo");
    cmd.arg("expand")
        .args(&module_args)
        // Suppress colour codes — we're parsing the output.
        .env("NO_COLOR", "1")
        .env("CARGO_TERM_COLOR", "never")
        .current_dir(workspace_root);

    let output = cmd.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("cargo expand exited {}: {}", output.status, stderr.trim());
    }

    let expanded = String::from_utf8_lossy(&output.stdout);
    let mut symbols = parse_rust_source(rel_path, &expanded)?;

    // Mark all as macro_expanded and filter to those not in the base pass.
    for sym in &mut symbols {
        sym.source = Some("macro_expanded".to_string());
    }
    symbols.retain(|s| !base_fqns.contains(&s.fqn));

    Ok(symbols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_path_to_expand_args_lib() {
        assert_eq!(file_path_to_expand_args("src/lib.rs"), Vec::<String>::new());
    }

    #[test]
    fn test_file_path_to_expand_args_main() {
        assert_eq!(
            file_path_to_expand_args("src/main.rs"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn test_file_path_to_expand_args_module() {
        assert_eq!(
            file_path_to_expand_args("src/engine.rs"),
            vec!["engine".to_string()]
        );
    }

    #[test]
    fn test_file_path_to_expand_args_nested() {
        assert_eq!(
            file_path_to_expand_args("src/foo/bar.rs"),
            vec!["foo::bar".to_string()]
        );
    }

    #[test]
    fn test_file_path_to_expand_args_mod_rs() {
        assert_eq!(
            file_path_to_expand_args("src/foo/mod.rs"),
            vec!["foo".to_string()]
        );
    }

    #[test]
    fn test_file_path_outside_src() {
        assert_eq!(
            file_path_to_expand_args("tests/integration.rs"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn test_body_has_macro_invocation() {
        assert!(body_has_macro_invocation("#[derive(Debug, Serialize)]"));
        assert!(body_has_macro_invocation("#[error(\"not found\")]"));
        assert!(!body_has_macro_invocation("fn plain_function() {}"));
    }
}
