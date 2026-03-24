use cs_core::watcher::ChangeKind;
use cs_core::{CoreEngine, EngineConfig};
use std::sync::Arc;
use tempfile::TempDir;

fn test_engine(dir: &TempDir) -> CoreEngine {
    let config = EngineConfig::new(dir.path()).without_embedder();
    CoreEngine::new(config).expect("engine init failed")
}

fn indexed_engine_with_two_langs(dir: &TempDir) -> CoreEngine {
    std::fs::write(dir.path().join("lib.rs"), "pub fn rust_fn() {}\n").unwrap();
    std::fs::write(dir.path().join("script.py"), "def py_fn(): pass\n").unwrap();
    let engine = test_engine(dir);
    engine.index_workspace().expect("index failed");
    engine
}

/// A corrupt SQLite file must cause `CoreEngine::new` to return `Err`,
/// not panic or silently succeed with a broken state.
#[test]
fn corrupt_sqlite_returns_err_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let db_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&db_dir).unwrap();
    std::fs::write(db_dir.join("index.db"), b"not a sqlite database\xff\xfe").unwrap();

    let config = EngineConfig::new(dir.path()).without_embedder();
    let result = CoreEngine::new(config);
    assert!(result.is_err(), "expected Err for corrupt db, got Ok");
}

/// Parallel calls to `run_pipeline` must not deadlock or panic.
/// This guards against lock-ordering bugs between graph/search/db.
#[test]
fn parallel_queries_do_not_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Arc::new(test_engine(&dir));

    let handles: Vec<_> = (0..8)
        .map(|i| {
            let e = Arc::clone(&engine);
            std::thread::spawn(move || {
                let query = format!("query number {}", i);
                let _ = e.run_pipeline(&query, Some(500), None, None);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }
}

/// Concurrent `reindex_file` calls for the same file must not corrupt the
/// index or deadlock. Each call should complete without panicking.
#[test]
fn concurrent_reindex_same_file() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("lib.rs");
    std::fs::write(&file_path, "pub fn foo() {}\npub fn bar() {}\n").unwrap();

    let engine = Arc::new(test_engine(&dir));

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let e = Arc::clone(&engine);
            let p = file_path.clone();
            std::thread::spawn(move || {
                e.reindex_file(&p, ChangeKind::Modified)
                    .expect("reindex_file failed");
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    let _ = engine.run_pipeline("foo", Some(500), None, None);
}

/// `run_pipeline` with `language="rust"` must not return Python symbols.
#[test]
fn run_pipeline_language_filter_excludes_other_langs() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);
    let out = engine
        .run_pipeline("fn", Some(4000), Some("rust"), None)
        .expect("run_pipeline failed");
    assert!(!out.contains("script.py"), "Python file should be filtered out: {}", out);
}

/// `run_pipeline` with `file_hint` must restrict results to matching file paths.
#[test]
fn run_pipeline_file_hint_restricts_to_matching_file() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);
    let out = engine
        .run_pipeline("fn", Some(4000), None, Some("script.py"))
        .expect("run_pipeline failed");
    assert!(!out.contains("lib.rs"), "Rust file should be filtered out: {}", out);
}

/// `get_context_capsule` with `max_results=1` must return at most one pivot.
#[test]
fn get_context_capsule_max_results_caps_pivots() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);
    let out = engine
        .get_context_capsule("fn", Some(4000), Some(1), None)
        .expect("get_context_capsule failed");
    let pivot_count = out.matches("#### `").count();
    assert!(pivot_count <= 1, "expected ≤1 pivot, got {}: {}", pivot_count, out);
}

/// `get_context_capsule` with `min_score` above any real score yields no pivots.
#[test]
fn get_context_capsule_min_score_filters_all_below_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);
    let out = engine
        .get_context_capsule("fn", Some(4000), None, Some(f32::MAX))
        .expect("get_context_capsule failed");
    assert!(!out.contains("#### `"), "expected no pivots with max min_score: {}", out);
}

/// `get_impact_graph` with `include_tests=false` must exclude test files.
#[test]
fn get_impact_graph_exclude_tests_filters_test_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        "pub fn target() {}\npub fn caller() { target(); }\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("lib_test.rs"),
        "fn test_target() { super::target(); }\n",
    )
    .unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    let with_tests = engine.get_impact_graph("lib.rs::target", None, true).unwrap();
    let without_tests = engine.get_impact_graph("lib.rs::target", None, false).unwrap();
    for dep in &without_tests.direct_dependents {
        assert!(
            !dep.file_path.contains("_test"),
            "test file leaked into production-only impact: {}",
            dep.file_path
        );
    }
    assert!(without_tests.total_affected <= with_tests.total_affected);
}

/// `get_impact_graph` with `max_depth=1` must not exceed depth-1 transitive results.
#[test]
fn get_impact_graph_max_depth_limits_traversal() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        "pub fn base() {}\npub fn mid() { base(); }\npub fn top() { mid(); }\n",
    )
    .unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    let shallow = engine.get_impact_graph("lib.rs::base", Some(1), true).unwrap();
    let deep = engine.get_impact_graph("lib.rs::base", Some(5), true).unwrap();
    assert!(
        shallow.total_affected <= deep.total_affected,
        "shallow traversal should find ≤ symbols than deep"
    );
}

/// `get_skeleton` with `max_depth=1` must return only top-level symbols.
#[test]
fn get_skeleton_max_depth_filters_nested_symbols() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        "pub struct Foo {}\nimpl Foo { pub fn method(&self) {} }\npub fn top_fn() {}\n",
    )
    .unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    let shallow = engine.get_skeleton("lib.rs", Some(1)).unwrap();
    let full = engine.get_skeleton("lib.rs", None).unwrap();
    assert!(
        shallow.symbols.len() <= full.symbols.len(),
        "max_depth=1 should have fewer symbols than unrestricted"
    );
    for sym in &shallow.symbols {
        let after_file = sym.fqn.splitn(2, "::").nth(1).unwrap_or("");
        assert!(
            !after_file.contains("::"),
            "depth-1 symbol has nested FQN: {}",
            sym.fqn
        );
    }
}

/// Queries issued while indexing is flagged as in-progress must succeed
/// (possibly with partial results) and must not panic.
#[test]
fn query_during_indexing_does_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.py"), "def alpha(): pass\n").unwrap();
    std::fs::write(dir.path().join("b.py"), "def beta(): pass\n").unwrap();

    let engine = Arc::new(test_engine(&dir));

    let e_idx = Arc::clone(&engine);
    let indexer = std::thread::spawn(move || {
        e_idx.index_workspace().expect("index_workspace failed");
    });

    let query_handles: Vec<_> = (0..4)
        .map(|_| {
            let e = Arc::clone(&engine);
            std::thread::spawn(move || {
                let _ = e.run_pipeline("alpha", Some(500), None, None);
            })
        })
        .collect();

    indexer.join().expect("indexer thread panicked");
    for h in query_handles {
        h.join().expect("query thread panicked");
    }
}
