//! Token savings micro-benchmark (issue #27, PLAN.md §B2).
//!
//! Runs a fixed set of 20 representative queries against a tempdir copy of
//! the `cs-core` source tree and records the token count of each
//! `run_pipeline` capsule. Writes `target/token_savings.json` and prints a
//! markdown table to stdout. Advisory — no CI gate.
//!
//! Paired with `cs-benchmark/scripts/bench_summary.py` (in a separate repo
//! at `~/projects/cs-benchmark/`), which diffs against the committed
//! `benches/token_baseline.json` there and renders the Δ column for PR
//! comments. There's also a Python parallel — `cs-benchmark/scripts/token_savings.py`
//! — that drives the released binary against an arbitrary workspace.
//!
//! Usage:
//!     cargo run --release --example token_savings -p cs-core

use std::fs;
use std::path::{Path, PathBuf};

use cs_core::capsule::estimate_tokens;
use cs_core::{CoreEngine, EngineConfig};

/// 20 queries chosen to exercise the major code paths in `cs-core`. The
/// expected pivot files in the comments are a sanity check — if future
/// ranking changes stop hitting these files, update the expectations.
const QUERIES: &[&str] = &[
    "fix the retry logic",           // engine.rs + capsule.rs
    "add a new language parser",     // parser_*.rs + symbol.rs
    "token budget assembly",         // capsule.rs
    "how does BM25 search work",     // search.rs
    "session memory observations",   // memory.rs
    "embedding cache refresh",       // engine.rs
    "tree-sitter rust parsing",      // parser_rust.rs
    "sqlite schema migration",       // db.rs
    "graph centrality calculation",  // graph.rs
    "rerank search results",         // engine.rs / search.rs
    "generate module documentation", // engine.rs (generate_module_docs)
    "impact graph blast radius",     // engine.rs
    "skeleton file API surface",     // skeletonizer.rs
    "workspace incremental index",   // engine.rs
    "observation staleness score",   // memory.rs
    "stub file indexing",            // engine.rs
    "swift enrichment xcode",        // engine.rs (swift_enrichment_hint)
    "diff capsule for PRs",          // engine.rs (get_diff_capsule)
    "search logic flow path",        // engine.rs / graph.rs find_path
    "intent detection routing",      // search.rs SearchIntent
];

fn corpus_source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn copy_dir(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create_dir_all");
    for entry in fs::read_dir(src).expect("read_dir") {
        let entry = entry.expect("dir entry");
        let ty = entry.file_type().expect("file_type");
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir(&entry.path(), &to);
        } else if ty.is_file() {
            fs::copy(entry.path(), to).expect("copy");
        }
    }
}

/// Sum token estimates across every .rs file in the corpus — the "send the
/// whole workspace" baseline the savings percentage is computed against.
fn workspace_tokens(root: &Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "rs") {
                if let Ok(text) = fs::read_to_string(&p) {
                    total += estimate_tokens(&text) as u64;
                }
            }
        }
    }
    total
}

fn main() {
    // Build an isolated workspace copy so the example never touches the real
    // `.codesurgeon` state of whichever repo invokes it.
    let tmp = tempfile::Builder::new()
        .prefix("cs-token-savings-")
        .tempdir()
        .expect("tempdir");
    copy_dir(&corpus_source(), &tmp.path().join("src"));

    let mut cfg = EngineConfig::new(tmp.path()).without_embedder();
    // A/B knob: set CS_BENCH_CENTRALITY_K=15.0 to pin the smoothing constant
    // and reproduce pre-#82 behaviour for comparison runs. Unset → corpus
    // median (issue #82 default).
    if let Ok(s) = std::env::var("CS_BENCH_CENTRALITY_K") {
        if let Ok(k) = s.parse::<f32>() {
            cfg.centrality_k_override = Some(k);
        }
    }
    let engine = CoreEngine::new(cfg).expect("engine");
    engine.index_workspace().expect("index");

    let ws_tokens = workspace_tokens(tmp.path());

    // Per-query measurements. `run_pipeline` returns the formatted capsule
    // string; `estimate_tokens` gives a char/4 token count that matches what
    // downstream agents consume.
    let mut results: Vec<(String, u64)> = Vec::with_capacity(QUERIES.len());
    for q in QUERIES {
        let capsule = engine
            .run_pipeline(q, None, None, None)
            .expect("run_pipeline");
        let tokens = estimate_tokens(&capsule) as u64;
        results.push((q.to_string(), tokens));
    }

    let total_capsule: u64 = results.iter().map(|(_, t)| t).sum();
    let avg_capsule = total_capsule as f64 / results.len() as f64;
    let savings_pct = if ws_tokens > 0 {
        (1.0 - avg_capsule / ws_tokens as f64) * 100.0
    } else {
        0.0
    };

    // Write machine-readable output for the summary script.
    let out_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target/token_savings.json");
    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let per_query_json: Vec<String> = results
        .iter()
        .map(|(q, t)| format!("    {:?}: {}", q, t))
        .collect();
    let json = format!(
        "{{\n  \"workspace_tokens\": {},\n  \"query_count\": {},\n  \"avg_capsule_tokens\": {:.1},\n  \"savings_pct\": {:.2},\n  \"per_query\": {{\n{}\n  }}\n}}\n",
        ws_tokens,
        results.len(),
        avg_capsule,
        savings_pct,
        per_query_json.join(",\n"),
    );
    fs::write(&out_path, &json).expect("write token_savings.json");

    // Markdown to stdout for PR comments / local inspection.
    println!("### Token savings");
    println!();
    println!("Corpus: cs-core source tree ({ws_tokens} workspace tokens)");
    println!();
    println!("| Query | Capsule tokens |");
    println!("|---|---:|");
    for (q, t) in &results {
        println!("| {q} | {t} |");
    }
    println!();
    println!("**Average capsule:** {avg_capsule:.0} tokens");
    println!("**Workspace savings:** {savings_pct:.1}%");
    println!();
    println!("_Written: {}_", out_path.display());

    drop(tmp);
}
