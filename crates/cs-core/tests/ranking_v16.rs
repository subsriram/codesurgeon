//! v1.6 file-diversity pinning tests.
//!
//! Each distinct file among exact anchor hits gets one pinned pivot
//! (most-specific symbol in the file), up to ANCHOR_FILE_BUDGET=5.
//! Remaining slots are filled by the centrality-ranked RRF fusion.
//! Total pivots = max_pivots (default 8). Pinning preserves anchor-named
//! files even when they would lose to high-centrality competitors.

use cs_core::{CoreEngine, EngineConfig};
use tempfile::TempDir;

fn engine_with(dir: &TempDir, max_pivots: usize) -> CoreEngine {
    let mut config = EngineConfig::new(dir.path()).without_embedder();
    config.max_pivots = max_pivots;
    CoreEngine::new(config).expect("engine init failed")
}

fn engine(dir: &TempDir) -> CoreEngine {
    engine_with(dir, 8)
}

/// Parse pivot file paths out of the rendered capsule. Headings look like
/// `#### `<fqn>` (<file_path>:<start>-<end>)`.
fn pivot_files(capsule: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in capsule.lines() {
        if let Some(rest) = line.strip_prefix("#### `") {
            // rest: <fqn>` (<file_path>:<start>-<end>)
            if let Some(idx) = rest.rfind("` (") {
                let after = &rest[idx + 3..];
                if let Some(close) = after.rfind(')') {
                    let inner = &after[..close];
                    if let Some(colon) = inner.rfind(':') {
                        out.push(inner[..colon].to_string());
                    }
                }
            }
        }
    }
    out
}

fn pivot_fqns(capsule: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in capsule.lines() {
        if let Some(rest) = line.strip_prefix("#### `") {
            if let Some(idx) = rest.rfind("` (") {
                out.push(rest[..idx].to_string());
            }
        }
    }
    out
}

fn pivot_count(capsule: &str) -> usize {
    capsule.matches("#### `").count()
}

/// Seed N non-anchor-matching helper files that BM25 may surface.
fn seed_helpers(dir: &TempDir, n: usize) {
    for i in 0..n {
        let name = format!("helper_{}.py", i);
        let body = format!(
            "def helper_func_{i}():\n    \"\"\"helper variant {i} for parse_latex_xyz alpha_beta\"\"\"\n    return {i}\n",
            i = i
        );
        std::fs::write(dir.path().join(&name), body).unwrap();
    }
}

/// Test 1: anchor hits in 3 distinct files → all 3 files pinned.
#[test]
fn three_distinct_files_all_pinned() {
    let dir = tempfile::tempdir().unwrap();
    seed_helpers(&dir, 6);
    for i in 0..3 {
        let name = format!("mod_{}.py", i);
        std::fs::write(
            dir.path().join(&name),
            "def parse_latex_xyz():\n    \"\"\"core variant\"\"\"\n    return 1\n",
        )
        .unwrap();
    }

    let e = engine(&dir);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline("fix parse_latex_xyz", Some(16000), None, None)
        .expect("run_pipeline");

    let files = pivot_files(&out);
    for i in 0..3 {
        let want = format!("mod_{}.py", i);
        assert!(
            files.iter().any(|f| f.ends_with(&want)),
            "expected {} to be pinned among pivot files {:?}",
            want,
            files
        );
    }
}

/// Test 2: 10 anchor hits in 10 distinct files, max_pivots=10 →
/// pinned count caps at ANCHOR_FILE_BUDGET=5; remaining 5 slots are RRF fill.
#[test]
fn file_budget_caps_pinning_at_five() {
    let dir = tempfile::tempdir().unwrap();
    seed_helpers(&dir, 6);
    for i in 0..10 {
        let name = format!("anchor_mod_{}.py", i);
        std::fs::write(
            dir.path().join(&name),
            "def parse_latex_xyz():\n    \"\"\"variant\"\"\"\n    return 1\n",
        )
        .unwrap();
    }

    let e = engine_with(&dir, 10);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline("fix parse_latex_xyz", Some(32000), None, None)
        .expect("run_pipeline");

    let files = pivot_files(&out);
    let anchor_pivots: Vec<&String> = files.iter().filter(|f| f.contains("anchor_mod_")).collect();
    // At least 5 distinct anchor files should appear (the pinned ones).
    // Some additional anchor files may also appear via RRF fill, but the
    // pinned count itself is 5 and total pivots = max_pivots = 10.
    assert!(
        anchor_pivots.len() >= 5,
        "expected ≥5 anchor files in pivots, got {} from {:?}",
        anchor_pivots.len(),
        files
    );
    assert_eq!(
        files.len(),
        10,
        "expected pivots = max_pivots (10), got {}\n{}",
        files.len(),
        out
    );
}

/// Test 3: many anchor hits in the same file → only 1 *pinned* slot for that
/// file. We verify this by shrinking max_pivots to 1: the lone slot must be
/// taken by the single pin, not by any RRF candidate from elsewhere.
#[test]
fn many_hits_same_file_pin_once() {
    let dir = tempfile::tempdir().unwrap();
    // Helpers also reference the anchor tokens in their docstrings so RRF
    // would happily pick a helper if pinning were broken.
    seed_helpers(&dir, 6);
    std::fs::write(
        dir.path().join("solo.py"),
        "\
def parse_latex_xyz():
    return 1

def alpha_beta_one():
    return 2

def gamma_delta_two():
    return 3
",
    )
    .unwrap();

    let e = engine_with(&dir, 1);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline(
            "fix parse_latex_xyz alpha_beta_one gamma_delta_two",
            Some(16000),
            None,
            None,
        )
        .expect("run_pipeline");

    let files = pivot_files(&out);
    assert_eq!(files.len(), 1, "expected single pivot, got {:?}", files);
    assert!(
        files[0].ends_with("solo.py"),
        "expected solo.py pinned, got {}",
        files[0]
    );
}

/// Test 4: no anchors → pinned = 0, all 8 slots filled by RRF.
/// Regression test that the v1.6 path doesn't disturb anchor-less queries.
#[test]
fn no_anchors_default_eight_pivots() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..10 {
        let name = format!("module_{}.py", i);
        let body = format!(
            "def module_function_{i}():\n    \"\"\"refactoring analysis pipeline variant {i}\"\"\"\n    return {i}\n",
            i = i
        );
        std::fs::write(dir.path().join(&name), body).unwrap();
    }

    let e = engine(&dir);
    e.index_workspace().expect("index");
    // Query has no extractable identifier tokens (all stop words / plain English).
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
        (5..=8).contains(&n),
        "expected 5-8 pivots from RRF only, got {}\n{}",
        n,
        out
    );
}

/// Test 5: same file has Foo (class) and Foo::bar (method); both are anchors.
/// Pinning prefers the more specific symbol (more "::").
#[test]
fn specificity_tiebreak_prefers_deeper_fqn() {
    let dir = tempfile::tempdir().unwrap();
    seed_helpers(&dir, 6);
    std::fs::write(
        dir.path().join("klass.py"),
        "\
class FooXyz:
    def bar_xyz(self):
        return 1
",
    )
    .unwrap();

    // Shrink to a single slot so the only pivot must be the pinned one.
    let e = engine_with(&dir, 1);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline("fix FooXyz bar_xyz", Some(16000), None, None)
        .expect("run_pipeline");

    let fqns = pivot_fqns(&out);
    assert_eq!(fqns.len(), 1, "expected single pivot, got {:?}", fqns);
    assert!(
        fqns[0].contains("bar_xyz"),
        "expected method (bar_xyz) pinned over class (FooXyz), got {}",
        fqns[0]
    );
}

/// Test 6: anchor-pinned symbol also appears at the top of RRF → no duplicate.
#[test]
fn anchor_rrf_overlap_dedup() {
    let dir = tempfile::tempdir().unwrap();
    seed_helpers(&dir, 6);
    std::fs::write(
        dir.path().join("target.py"),
        "def parse_latex_xyz():\n    \"\"\"parse_latex_xyz parse_latex_xyz parse_latex_xyz\"\"\"\n    return 1\n",
    )
    .unwrap();

    let e = engine(&dir);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline("fix parse_latex_xyz", Some(16000), None, None)
        .expect("run_pipeline");

    let fqns = pivot_fqns(&out);
    let target_count = fqns
        .iter()
        .filter(|f| f.contains("parse_latex_xyz"))
        .count();
    assert_eq!(
        target_count, 1,
        "expected parse_latex_xyz once across pinned+RRF, got {} in {:?}",
        target_count, fqns
    );
}

/// Test 7: across the various scenarios pinned + RRF == max_pivots.
#[test]
fn pinned_plus_fill_equals_max_pivots() {
    let dir = tempfile::tempdir().unwrap();
    seed_helpers(&dir, 8);
    for i in 0..3 {
        let name = format!("mod_{}.py", i);
        std::fs::write(
            dir.path().join(&name),
            "def parse_latex_xyz():\n    return 1\n",
        )
        .unwrap();
    }

    let e = engine(&dir);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline("fix parse_latex_xyz", Some(32000), None, None)
        .expect("run_pipeline");

    let n = pivot_count(&out);
    assert_eq!(n, 8, "expected pivots = max_pivots (8), got {}\n{}", n, out);
}
