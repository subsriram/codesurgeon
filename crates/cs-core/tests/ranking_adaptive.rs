//! v1.5 adaptive pivot-count tests.
//!
//! When anchors resolve cleanly the capsule shrinks to 3 pivots; on partial
//! resolution it caps at 5; when no anchor fires (or fires but all stats are
//! zero) it uses the default 8.
//!
//! Each test seeds a pool of ≥ 8 retrievable symbols so the observed pivot
//! count is driven by the adaptive cap, not by candidate-pool starvation.

use cs_core::{CoreEngine, EngineConfig};
use tempfile::TempDir;

fn engine(dir: &TempDir) -> CoreEngine {
    let config = EngineConfig::new(dir.path()).without_embedder();
    CoreEngine::new(config).expect("engine init failed")
}

/// Seed N non-test helper files whose symbol names *and* docstrings contain
/// tokens that overlap with the test queries so BM25 retrieval surfaces them.
/// They must NOT be exact-name matches for the target anchor.
fn seed_helpers(dir: &TempDir, n: usize) {
    for i in 0..n {
        let name = format!("helper_{}.py", i);
        let body = format!(
            "def parse_latex_helper_{i}():\n    \"\"\"parse_latex_xyz helper variant {i}\"\"\"\n    return {i}\n",
            i = i
        );
        std::fs::write(dir.path().join(&name), body).unwrap();
    }
}

fn pivot_count(capsule: &str) -> usize {
    capsule.matches("#### `").count()
}

/// Clean anchor: 3 exact-name hits across 3 distinct non-test files + enough
/// background candidates that, without the adaptive cap, 5–8 pivots would be
/// selected. Expectation: capsule is capped at 3 pivots.
#[test]
fn clean_anchor_shrinks_to_three_pivots() {
    let dir = tempfile::tempdir().unwrap();
    seed_helpers(&dir, 6);
    for i in 0..3 {
        let name = format!("mod_{}.py", i);
        std::fs::write(
            dir.path().join(&name),
            "def parse_latex_xyz():\n    \"\"\"parse_latex_xyz core variant\"\"\"\n    return 1\n",
        )
        .unwrap();
    }

    let e = engine(&dir);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline("refactor parse_latex_xyz helper", Some(16000), None, None)
        .expect("run_pipeline");
    let n = pivot_count(&out);
    assert_eq!(n, 3, "expected 3 pivots (clean cap), got {}\n{}", n, out);
}

/// Medium anchor: 1 exact-name hit (distinct_source_files = 1 → not clean) +
/// enough BM25-matchable helpers that the default 8 would otherwise apply.
/// Expectation: capsule is capped at 5 pivots.
#[test]
fn medium_anchor_caps_at_five_pivots() {
    let dir = tempfile::tempdir().unwrap();
    seed_helpers(&dir, 8);
    std::fs::write(
        dir.path().join("mod_unique.py"),
        "def parse_latex_xyz():\n    \"\"\"parse_latex_xyz sole variant\"\"\"\n    return 1\n",
    )
    .unwrap();

    let e = engine(&dir);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline("fix parse_latex_xyz", Some(16000), None, None)
        .expect("run_pipeline");
    let n = pivot_count(&out);
    assert_eq!(n, 5, "expected 5 pivots (medium cap), got {}\n{}", n, out);
}

/// No anchor: query contains prose that's not extractable as identifiers,
/// matching lots of BM25 body content. Effective pivots = default 8.
#[test]
fn no_anchor_uses_default_eight_pivots() {
    let dir = tempfile::tempdir().unwrap();
    // Seed 10 non-anchor-matching helpers whose bodies will BM25-match the
    // query tokens. The query itself uses identifier-shaped words that do NOT
    // exist as symbols in the index, and for which the bm25-name fallback
    // also returns zero hits — so anchor stats are all zero and the default
    // (8) bucket applies.
    for i in 0..10 {
        let name = format!("module_{}.py", i);
        let body = format!(
            "def module_function_{i}():\n    \"\"\"refactoring analysis module pipeline variant {i}\"\"\"\n    return {i}\n",
            i = i
        );
        std::fs::write(dir.path().join(&name), body).unwrap();
    }

    let e = engine(&dir);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline(
            "refactoring analysis pipeline module",
            Some(32000),
            None,
            None,
        )
        .expect("run_pipeline");
    let n = pivot_count(&out);
    assert!(
        n > 5,
        "expected >5 pivots (default 8 bucket), got {}\n{}",
        n,
        out
    );
    assert!(n <= 8, "default bucket is 8, got {}\n{}", n, out);
}

/// Edge case: 3 exact-name hits all in test files → distinct_source_files = 0
/// → does not qualify for the clean-bucket 3-cap, falls back to medium (5).
#[test]
fn anchor_hits_only_in_tests_falls_back_to_medium() {
    let dir = tempfile::tempdir().unwrap();
    seed_helpers(&dir, 8);
    let test_dir = dir.path().join("tests");
    std::fs::create_dir_all(&test_dir).unwrap();
    for i in 0..3 {
        let name = format!("test_mod_{}.py", i);
        std::fs::write(
            test_dir.join(&name),
            "def parse_latex_xyz():\n    \"\"\"parse_latex_xyz test variant\"\"\"\n    return 1\n",
        )
        .unwrap();
    }

    let e = engine(&dir);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline("fix parse_latex_xyz", Some(16000), None, None)
        .expect("run_pipeline");
    let n = pivot_count(&out);
    assert_eq!(
        n, 5,
        "3 test-only exact hits should land in medium bucket (5), got {}\n{}",
        n, out
    );
}
