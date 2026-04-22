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

/// `from ... import (ErrorType, other, ...)` statements are indexed as
/// `SymbolKind::Import`. Under reverse-expand's candidate scoring they
/// score highly (their FQN / name literally list the query terms) and
/// can leak into the pivot pool. The motivating regression was seen on
/// `sympy__sympy-21379` where bare import lines from `sympy/__init__.py`
/// won pivot slots and sent the agent into unrelated files until it
/// timed out. Filter drops `SymbolKind::Import` from reverse-expand
/// candidates and from pivot eligibility in general.
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

/// Trivial exception-class stubs (`class FooError(Base): pass`) rank highly
/// on BM25 when the task names the exception, but their bodies are a single
/// declaration line and carry no behaviour. Before this filter, on the
/// sympy-21379 corpus `PolynomialError` (a 1-line `pass` stub) took pivot
/// slot #7 and `Mod.eval` — the actual fix site, reachable via
/// reverse-expand through the raise chain — lost the slot.
///
/// With the filter, the stub is dropped from pivot eligibility and the
/// behaviour-carrying caller takes the slot instead. The stub is still
/// available as a reverse-expand seed (we want the walk to run) — it just
/// can't *occupy* a pivot slot on its own.
#[test]
fn trivial_exception_stub_is_excluded_from_pivots() {
    let dir = tempfile::tempdir().unwrap();

    // 1-line exception stub — matches the `PolynomialError(BasePolynomialError): pass` shape.
    std::fs::write(
        dir.path().join("errors.py"),
        "class BaseAppError(Exception):\n    pass\n\n\
         class AppError(BaseAppError):\n    pass\n",
    )
    .unwrap();

    // Behaviour-carrying caller that raises the stub exception.
    std::fs::write(
        dir.path().join("fix_site.py"),
        "from errors import AppError\n\n\
         def run_the_pipeline():\n\
         \x20   raise AppError(\"boom\")\n",
    )
    .unwrap();

    let e = engine_for(&dir, true, 8);
    e.index_workspace().expect("index");

    let out = e
        .run_pipeline("fix AppError crash", Some(16000), None, None)
        .expect("run_pipeline");
    let pivots = pivot_fqns(&out);

    // The stub itself must NOT occupy a pivot slot: its body is `pass`, no
    // behaviour to show.
    assert!(
        !pivots.iter().any(|p| p.ends_with("::AppError")),
        "trivial exception stub AppError should not occupy a pivot slot; pivots: {:?}\n\n{}",
        pivots,
        out
    );

    // The behaviour-carrying caller (reached via reverse-expand) must be
    // present — the point is that its slot is no longer contested by the
    // stub.
    assert!(
        pivots.iter().any(|p| p.contains("run_the_pipeline")),
        "expected the raiser of AppError to surface as a pivot; pivots: {:?}\n\n{}",
        pivots,
        out
    );
}

/// Exception classes with real methods (`__init__`, `__str__`, custom
/// formatting, validation, etc.) carry behaviour worth showing as pivots.
/// The filter must NOT drop them — only trivial `pass` stubs.
#[test]
fn non_trivial_exception_classes_remain_eligible_as_pivots() {
    let dir = tempfile::tempdir().unwrap();

    // Exception with real methods — > 3 non-blank body lines.
    std::fs::write(
        dir.path().join("errors.py"),
        "class RichError(Exception):\n\
         \x20   \"\"\"An error with real machinery.\"\"\"\n\
         \x20   def __init__(self, code, msg):\n\
         \x20       super().__init__(msg)\n\
         \x20       self.code = code\n\
         \x20   def __str__(self):\n\
         \x20       return f\"[{self.code}] {self.args[0]}\"\n",
    )
    .unwrap();

    let e = engine_for(&dir, true, 8);
    e.index_workspace().expect("index");

    let out = e
        .run_pipeline("fix RichError crash", Some(16000), None, None)
        .expect("run_pipeline");
    let pivots = pivot_fqns(&out);

    // The rich exception class should surface as a pivot — its __init__ and
    // __str__ carry information the agent can use.
    assert!(
        pivots.iter().any(|p| p.ends_with("::RichError")),
        "non-trivial exception class with real methods should occupy a pivot slot; pivots: {:?}\n\n{}",
        pivots,
        out
    );
}
