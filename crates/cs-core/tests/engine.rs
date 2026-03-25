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

/// `run_pipeline` must write an `auto` observation when pivots are found.
#[test]
fn run_pipeline_writes_auto_observation() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);
    engine.run_pipeline("rust fn", Some(4000), None, None).expect("run_pipeline failed");

    let observations = engine.get_session_context().expect("get_session_context failed");
    let auto_obs: Vec<_> = observations
        .iter()
        .filter(|o| o.kind.as_str() == "auto")
        .collect();
    assert!(!auto_obs.is_empty(), "expected at least one auto observation after run_pipeline");
    assert!(
        auto_obs[0].content.starts_with("Agent queried:"),
        "unexpected auto observation format: {}",
        auto_obs[0].content
    );
}

/// `get_context_capsule` must write an `auto` observation when pivots are found.
#[test]
fn get_context_capsule_writes_auto_observation() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);
    engine
        .get_context_capsule("py fn", Some(4000), None, None)
        .expect("get_context_capsule failed");

    let observations = engine.get_session_context().expect("get_session_context failed");
    let auto_obs: Vec<_> = observations
        .iter()
        .filter(|o| o.kind.as_str() == "auto")
        .collect();
    assert!(!auto_obs.is_empty(), "expected at least one auto observation after get_context_capsule");
}

/// Calling `run_pipeline` twice with the same task must deduplicate — only one auto observation.
#[test]
fn run_pipeline_deduplicates_auto_observations() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);
    engine.run_pipeline("rust fn", Some(4000), None, None).expect("first call failed");
    engine.run_pipeline("rust fn", Some(4000), None, None).expect("second call failed");

    let observations = engine.get_session_context().expect("get_session_context failed");
    let auto_count = observations
        .iter()
        .filter(|o| o.kind.as_str() == "auto" && o.content.contains("rust fn"))
        .count();
    assert_eq!(auto_count, 1, "duplicate auto observation should have been suppressed");
}

/// `run_pipeline` with no matching symbols must not write an auto observation.
#[test]
fn run_pipeline_no_pivots_skips_auto_observation() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(&dir); // empty index — no symbols
    engine
        .run_pipeline("xyzzy no match", Some(4000), None, None)
        .expect("run_pipeline failed");

    let observations = engine.get_session_context().expect("get_session_context failed");
    let auto_obs: Vec<_> = observations
        .iter()
        .filter(|o| o.kind.as_str() == "auto")
        .collect();
    assert!(
        auto_obs.is_empty(),
        "expected no auto observation when there are no pivots, got: {:?}",
        auto_obs.iter().map(|o| &o.content).collect::<Vec<_>>()
    );
}

/// Queries issued while indexing is flagged as in-progress must succeed
/// Stub files under `node_modules/@types/` must be indexed with `is_stub=true`
/// and must never appear as pivots in `run_pipeline` results.
#[test]
fn stub_files_indexed_but_not_returned_as_pivots() {
    let dir = tempfile::tempdir().unwrap();

    // Project source file
    std::fs::write(dir.path().join("app.ts"), "export function greet(name: string): string { return `Hello ${name}`; }\n").unwrap();

    // Simulate a .d.ts stub under node_modules/@types/
    let types_dir = dir.path().join("node_modules").join("@types").join("node");
    std::fs::create_dir_all(&types_dir).unwrap();
    std::fs::write(
        types_dir.join("index.d.ts"),
        "declare function require(id: string): any;\ndeclare var process: NodeJS.Process;\n",
    ).unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index_workspace failed");

    let stats = engine.index_stats().expect("index_stats failed");
    assert!(stats.stub_symbol_count > 0, "expected stub symbols to be indexed");

    // run_pipeline on a stub-only query — stubs must not appear as pivots
    let out = engine.run_pipeline("require", Some(4000), None, None).expect("run_pipeline failed");
    // Pivots are formatted as `#### \`...\``; stub symbols must not be pivots
    assert!(
        !out.contains("#### `node_modules/"),
        "stub symbol appeared as pivot: {}",
        out
    );
}

/// Regular project source files must NOT be marked as stubs.
#[test]
fn project_files_are_not_stubs() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "pub fn real_fn() {}\n").unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index_workspace failed");

    let stats = engine.index_stats().expect("index_stats failed");
    assert_eq!(stats.stub_symbol_count, 0, "project symbols should not be stubs");
    assert!(stats.symbol_count > 0, "project symbols should be indexed");
}

/// Python `.pyi` stubs under a virtual-env `site-packages/` directory must be indexed
/// and marked as stubs.
#[test]
fn python_pyi_stubs_indexed() {
    let dir = tempfile::tempdir().unwrap();

    // Simulate .venv/lib/python3.x/site-packages/requests.pyi
    let site_packages = dir
        .path()
        .join(".venv")
        .join("lib")
        .join("python3.11")
        .join("site-packages");
    std::fs::create_dir_all(&site_packages).unwrap();
    std::fs::write(
        site_packages.join("requests.pyi"),
        "def get(url: str, **kwargs) -> Response: ...\nclass Response:\n    status_code: int\n",
    )
    .unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index_workspace failed");

    let stats = engine.index_stats().expect("index_stats failed");
    assert!(stats.stub_symbol_count > 0, "Python pyi stubs should be indexed");
}

/// Swift `.swiftinterface` stubs under `.build/` must be indexed and marked as stubs.
#[test]
fn swift_swiftinterface_stubs_indexed() {
    let dir = tempfile::tempdir().unwrap();

    // Simulate .build/release/Modules/Foundation.swiftinterface
    let modules_dir = dir
        .path()
        .join(".build")
        .join("release")
        .join("Modules");
    std::fs::create_dir_all(&modules_dir).unwrap();
    std::fs::write(
        modules_dir.join("MyLib.swiftinterface"),
        "public func doSomething() -> Void\npublic class MyClass {}\n",
    )
    .unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index_workspace failed");

    let stats = engine.index_stats().expect("index_stats failed");
    assert!(stats.stub_symbol_count > 0, "Swift swiftinterface stubs should be indexed");
}

/// When `index_stubs = false`, stub directories are skipped entirely.
#[test]
fn index_stubs_false_skips_stub_files() {
    let dir = tempfile::tempdir().unwrap();

    let types_dir = dir.path().join("node_modules").join("@types").join("node");
    std::fs::create_dir_all(&types_dir).unwrap();
    std::fs::write(
        types_dir.join("index.d.ts"),
        "declare function require(id: string): any;\n",
    )
    .unwrap();

    let config = cs_core::EngineConfig {
        index_stubs: false,
        ..cs_core::EngineConfig::new(dir.path()).without_embedder()
    };
    let engine = cs_core::CoreEngine::new(config).expect("engine init failed");
    engine.index_workspace().expect("index_workspace failed");

    let stats = engine.index_stats().expect("index_stats failed");
    assert_eq!(stats.stub_symbol_count, 0, "stubs should not be indexed when disabled");
}

/// Sensitive files (.env, *.key, files named *secret*, etc.) must not be indexed.
#[test]
fn sensitive_files_are_not_indexed() {
    let dir = tempfile::tempdir().unwrap();

    // Normal file — should be indexed
    std::fs::write(dir.path().join("app.py"), "def hello(): pass\n").unwrap();

    // Sensitive files — must be excluded
    std::fs::write(dir.path().join(".env"), "DB_PASSWORD=hunter2\n").unwrap();
    std::fs::write(dir.path().join(".env.local"), "SECRET=abc\n").unwrap();
    std::fs::write(dir.path().join("my_secret.py"), "API_KEY = 'abc'\n").unwrap();
    std::fs::write(dir.path().join("credentials.json"), "{\"key\":\"val\"}\n").unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    let out = engine
        .run_pipeline("hello", Some(4000), None, None)
        .expect("run_pipeline failed");

    assert!(!out.contains(".env"), "env file should be excluded: {}", out);
    assert!(!out.contains("my_secret"), "secret file should be excluded: {}", out);
    assert!(!out.contains("credentials"), "credentials file should be excluded: {}", out);
}

/// Files containing known API key patterns in the first 4 KB must not be indexed.
#[test]
fn file_with_api_key_content_is_not_indexed() {
    let dir = tempfile::tempdir().unwrap();

    // Normal file
    std::fs::write(dir.path().join("lib.py"), "def safe(): pass\n").unwrap();
    // File with an AWS key literal embedded
    std::fs::write(
        dir.path().join("config.py"),
        "AWS_KEY = \"AKIAIOSFODNN7EXAMPLE123\"\n",
    )
    .unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    let out = engine
        .run_pipeline("AWS_KEY safe", Some(4000), None, None)
        .expect("run_pipeline failed");

    assert!(!out.contains("config.py"), "file with embedded API key should be excluded: {}", out);
}

/// Files listed in `.codesurgeonignore` must not appear in the index.
#[test]
fn codesurgeonignore_excludes_files() {
    let dir = tempfile::tempdir().unwrap();

    std::fs::write(dir.path().join("app.py"), "def keep(): pass\n").unwrap();
    std::fs::write(dir.path().join("generated.py"), "def generated(): pass\n").unwrap();
    std::fs::write(dir.path().join(".codesurgeonignore"), "generated.py\n").unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    let out = engine
        .run_pipeline("generated keep", Some(4000), None, None)
        .expect("run_pipeline failed");

    assert!(!out.contains("generated.py"), "ignored file should be excluded: {}", out);
}

/// Glob patterns in `.codesurgeonignore` must work (e.g. `fixtures/`).
#[test]
fn codesurgeonignore_glob_pattern_excludes_directory() {
    let dir = tempfile::tempdir().unwrap();

    std::fs::write(dir.path().join("app.py"), "def main(): pass\n").unwrap();

    let fixtures = dir.path().join("fixtures");
    std::fs::create_dir_all(&fixtures).unwrap();
    std::fs::write(fixtures.join("data.py"), "def fixture_fn(): pass\n").unwrap();

    std::fs::write(dir.path().join(".codesurgeonignore"), "fixtures/\n").unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    let out = engine
        .run_pipeline("fixture_fn", Some(4000), None, None)
        .expect("run_pipeline failed");

    assert!(!out.contains("fixtures/"), "fixtures dir should be excluded: {}", out);
}

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
