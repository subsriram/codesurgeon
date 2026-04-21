//! Reverse-edge expansion (issue #67) — regression tests.
//!
//! When an exception-class anchor is mentioned in the task, the capsule
//! should include callers/raisers reachable by walking backward through the
//! call graph. Direct callers aren't enough: the motivating sympy-21379
//! case has the fix site 3 hops upstream from the error class the user
//! names.
//!
//! These tests seed a tiny Python corpus and assert that `run_pipeline`
//! returns reverse-expanded callers as pivots even though the task mentions
//! only the error class. We use a small `max_pivots` + BM25-competing noise
//! so that the 2-hop caller must be *promoted* by reverse expansion (not
//! merely surfaced because the pivot pool is empty).

use cs_core::{CoreEngine, EngineConfig};
use tempfile::TempDir;

fn engine_for(dir: &TempDir, reverse_expand: bool, max_pivots: usize) -> CoreEngine {
    let mut config = EngineConfig::new(dir.path()).without_embedder();
    config.max_pivots = max_pivots;
    config.reverse_expand_anchors = reverse_expand;
    CoreEngine::new(config).expect("engine init failed")
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

/// Seed BM25-competing noise files: each has a function whose body matches
/// the query terms "fix" and "crash" so they outrank the 2-hop caller on
/// pure BM25 + centrality. Reverse expansion must explicitly promote the
/// 2-hop caller for it to appear as a pivot.
fn seed_bm25_noise(dir: &TempDir, n: usize) {
    for i in 0..n {
        let name = format!("noise_{}.py", i);
        let body = format!(
            "def fix_crash_handler_{i}():\n    \"\"\"fix the crash in handling variant {i}\"\"\"\n    return {i}\n",
            i = i
        );
        std::fs::write(dir.path().join(&name), body).unwrap();
    }
}

fn seed_raise_chain(dir: &TempDir) {
    std::fs::write(
        dir.path().join("myerror.py"),
        "class MyError(Exception):\n    pass\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("raiser.py"),
        "from myerror import MyError\n\n\
         def raise_my_error():\n    raise MyError(\"boom\")\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("caller.py"),
        "from raiser import raise_my_error\n\n\
         def caller_of_raise_my_error():\n    raise_my_error()\n",
    )
    .unwrap();
}

/// Acceptance-criteria regression: task names only the error class, 2-hop
/// caller must be in pivots.
#[test]
fn reverse_expand_surfaces_indirect_caller() {
    let dir = tempfile::tempdir().unwrap();
    seed_bm25_noise(&dir, 10);
    seed_raise_chain(&dir);

    let e = engine_for(&dir, true, 3);
    e.index_workspace().expect("index");

    let out = e
        .run_pipeline("fix MyError crash", Some(16000), None, None)
        .expect("run_pipeline");
    let fqns = pivot_fqns(&out);

    assert!(
        fqns.iter().any(|f| f.contains("caller_of_raise_my_error")),
        "expected reverse-expanded caller in pivots, got {:?}\n\n{}",
        fqns,
        out
    );
}

/// Same corpus, feature disabled: the 2-hop caller has no BM25 overlap with
/// the task and is 2 hops from the seed (past single-hop graph expansion),
/// so it should not appear.
#[test]
fn without_reverse_expand_indirect_caller_is_missed() {
    let dir = tempfile::tempdir().unwrap();
    seed_bm25_noise(&dir, 10);
    seed_raise_chain(&dir);

    let e = engine_for(&dir, false, 3);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline("fix MyError crash", Some(16000), None, None)
        .expect("run_pipeline");
    let fqns = pivot_fqns(&out);

    assert!(
        !fqns.iter().any(|f| f.contains("caller_of_raise_my_error")),
        "reverse-expand disabled, but 2-hop caller still appeared: {:?}\n\n{}",
        fqns,
        out
    );
}

/// Generic (non-exception) anchor must NOT trigger reverse expansion.
/// Uses a parse_latex chain identical in shape to the raise chain but with
/// a non-exception seed. Feature is enabled, but the classifier should skip
/// the seed and the 2-hop caller should not surface.
#[test]
fn generic_anchor_does_not_trigger_reverse_expand() {
    let dir = tempfile::tempdir().unwrap();
    seed_bm25_noise(&dir, 10);

    std::fs::write(
        dir.path().join("parse_latex.py"),
        "def parse_latex(x):\n    return x\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("mid.py"),
        "from parse_latex import parse_latex\n\n\
         def mid_layer():\n    return parse_latex('x')\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("top.py"),
        "from mid import mid_layer\n\n\
         def unrelated_top_level():\n    return mid_layer()\n",
    )
    .unwrap();

    let e = engine_for(&dir, true, 3);
    e.index_workspace().expect("index");
    let out = e
        .run_pipeline("fix parse_latex crash", Some(16000), None, None)
        .expect("run_pipeline");
    let fqns = pivot_fqns(&out);

    assert!(
        !fqns.iter().any(|f| f.contains("unrelated_top_level")),
        "generic anchor should not trigger reverse expansion, but deep caller appeared: {:?}\n\n{}",
        fqns,
        out
    );
}

/// Ensure the feature is opt-outable: even on a corpus where reverse
/// expansion would fire, setting the flag to false produces zero reverse
/// candidates — the capsule is identical to the pre-#67 behaviour.
#[test]
fn feature_flag_respected() {
    let dir = tempfile::tempdir().unwrap();
    seed_bm25_noise(&dir, 10);
    seed_raise_chain(&dir);

    let on = engine_for(&dir, true, 3);
    on.index_workspace().expect("index");
    let out_on = on
        .run_pipeline("fix MyError crash", Some(16000), None, None)
        .unwrap();

    let off = engine_for(&dir, false, 3);
    off.index_workspace().expect("index");
    let out_off = off
        .run_pipeline("fix MyError crash", Some(16000), None, None)
        .unwrap();

    let fqns_on = pivot_fqns(&out_on);
    let fqns_off = pivot_fqns(&out_off);

    let indirect_on = fqns_on
        .iter()
        .any(|f| f.contains("caller_of_raise_my_error"));
    let indirect_off = fqns_off
        .iter()
        .any(|f| f.contains("caller_of_raise_my_error"));
    assert!(
        indirect_on && !indirect_off,
        "feature flag must gate reverse expansion: on={:?}, off={:?}",
        fqns_on,
        fqns_off
    );
}

/// Dense-graph regression (issue #69): on a seed with 50+ direct callers,
/// a query-term-aligned chain reachable only via docstring / signature
/// context (not name overlap) should still surface. Before the query-aware
/// scoring fix, the walk ranked purely on name/fqn overlap — when the chain
/// caller had a generic name the top-5 filter at hop 1 would usually lose
/// it to one of the 50 distractors.
#[test]
fn dense_graph_reverse_expand_surfaces_target() {
    let dir = tempfile::tempdir().unwrap();

    // Exception class with 55 generic distractor raisers (no query term
    // match on name/fqn, no docstring).
    std::fs::write(
        dir.path().join("baseerr.py"),
        "class BaseError(Exception):\n    pass\n",
    )
    .unwrap();
    let mut distractors = String::from("from baseerr import BaseError\n\n");
    for i in 0..55 {
        distractors.push_str(&format!(
            "def handler_{i}():\n    raise BaseError(\"x\")\n\n",
            i = i
        ));
    }
    std::fs::write(dir.path().join("distractors.py"), distractors).unwrap();

    // Target chain: direct raiser has a docstring that matches the query
    // ("serialization") but its name is generic — tests the docstring /
    // signature arm of `term_overlap_score`. The deep target is 1 hop
    // further up the chain.
    std::fs::write(
        dir.path().join("chain_raiser.py"),
        "from baseerr import BaseError\n\n\
         def generic_dispatch_fn():\n    \
             \"\"\"serialization entry for the chain\"\"\"\n    \
             raise BaseError(\"boom\")\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("chain_target.py"),
        "from chain_raiser import generic_dispatch_fn\n\n\
         def deep_serialization_target():\n    generic_dispatch_fn()\n",
    )
    .unwrap();

    // Larger pivot budget so dense reverse-expand has room to surface both
    // the hop-1 chain caller and the hop-2 deep target alongside BM25
    // matches for the 55 distractors.
    let e = engine_for(&dir, true, 8);
    e.index_workspace().expect("index");

    let out = e
        .run_pipeline(
            "fix BaseError during serialization",
            Some(16000),
            None,
            None,
        )
        .expect("run_pipeline");

    // Per issue #69 acceptance criterion: the deep target needs to be
    // *present* in the capsule (pivot, adjacent, or skeleton) so the agent
    // doesn't have to guess-grep. Check the whole capsule body, not just
    // pivots — reverse-expand candidates can land in any of these slots
    // depending on RRF fusion and file-diversity pinning.
    assert!(
        out.contains("deep_serialization_target"),
        "expected dense-graph reverse-expand to surface the deep target (issue #69); pivots were {:?}\n\n{}",
        pivot_fqns(&out),
        out
    );
}

/// `from ... import (ErrorType, other, ...)` statements are indexed as
/// `SymbolKind::Import`. Under reverse-expand's query-aware ranking they
/// were scoring highly (their FQN / name literally list the user's
/// query terms) and leaking into the candidate pool. That regressed
/// `sympy__sympy-21379` from a 290-s success to a 600-s timeout — the
/// agent chased import lines into unrelated files and never found the
/// fix site.
///
/// Verification strategy: toggle `reverse_expand_anchors` on/off against
/// the same fixture. Pre-fix, reverse-expand contributed `Import` kind
/// candidates that won pivot slots via RRF fusion. Post-fix, the two
/// capsules should be identical in pivot set (the reverse-expand walk
/// still runs, it just filters out Imports before scoring).
#[test]
fn reverse_expand_does_not_surface_import_statements() {
    let dir = tempfile::tempdir().unwrap();

    std::fs::write(
        dir.path().join("err.py"),
        "class DeepError(Exception):\n    pass\n",
    )
    .unwrap();

    // One behaviour-carrying caller of DeepError.
    std::fs::write(
        dir.path().join("fix_site.py"),
        "from err import DeepError\n\n\
         def run_the_pipeline():\n\
         \x20   raise DeepError(\"boom\")\n",
    )
    .unwrap();

    // Re-export shims: files whose only content is `from err import DeepError`.
    // Pre-fix, reverse-expand walked from DeepError up through these imports
    // and they won pivot slots because their FQN literally contains "DeepError".
    for i in 0..6 {
        std::fs::write(
            dir.path().join(format!("shim_{i}.py")),
            "from err import DeepError\n",
        )
        .unwrap();
    }

    let e = engine_for(&dir, true, 8);
    e.index_workspace().expect("index");

    let out = e
        .run_pipeline("fix DeepError", Some(16000), None, None)
        .expect("run_pipeline");
    let pivots = pivot_fqns(&out);

    // The real caller (reached by reverse-expand through the raise edge)
    // must be present.
    assert!(
        out.contains("run_the_pipeline"),
        "expected the behaviour-carrying caller to surface; pivots: {:?}\n\n{}",
        pivots,
        out
    );

    // No pivot FQN should be an import statement. Reverse-expand is the
    // only path that could reach these (no BM25 overlap with the
    // single-token query "fix DeepError"), so filtering `SymbolKind::Import`
    // in reverse_expand_from_anchors is load-bearing.
    for p in &pivots {
        let tail = p.rsplit("::").next().unwrap_or(p);
        assert!(
            !tail.starts_with("from ") && !tail.starts_with("import "),
            "import-statement symbol leaked into pivots: {:?}\n\nfull output:\n{}",
            p,
            out
        );
    }
}
