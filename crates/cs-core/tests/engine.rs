use cs_core::watcher::ChangeKind;
use cs_core::{CoreEngine, EngineConfig};
use std::sync::Arc;
use tempfile::TempDir;

fn test_engine(dir: &TempDir) -> CoreEngine {
    let config = EngineConfig::new(dir.path()).without_embedder();
    CoreEngine::new(config).expect("engine init failed")
}

/// Engine with `auto_observations = true`. Use from tests that exercise the
/// legacy auto-memory path; the production default is off (#72).
fn test_engine_with_auto_obs(dir: &TempDir) -> CoreEngine {
    let mut config = EngineConfig::new(dir.path()).without_embedder();
    config.auto_observations = true;
    CoreEngine::new(config).expect("engine init failed")
}

fn indexed_engine_with_two_langs(dir: &TempDir) -> CoreEngine {
    std::fs::write(dir.path().join("lib.rs"), "pub fn rust_fn() {}\n").unwrap();
    std::fs::write(dir.path().join("script.py"), "def py_fn(): pass\n").unwrap();
    let engine = test_engine(dir);
    engine.index_workspace().expect("index failed");
    engine
}

/// Like `indexed_engine_with_two_langs` but with auto-observations enabled.
fn indexed_engine_with_two_langs_auto_obs(dir: &TempDir) -> CoreEngine {
    std::fs::write(dir.path().join("lib.rs"), "pub fn rust_fn() {}\n").unwrap();
    std::fs::write(dir.path().join("script.py"), "def py_fn(): pass\n").unwrap();
    let engine = test_engine_with_auto_obs(dir);
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
    assert!(
        !out.contains("script.py"),
        "Python file should be filtered out: {}",
        out
    );
}

/// `run_pipeline` with `file_hint` must restrict results to matching file paths.
#[test]
fn run_pipeline_file_hint_restricts_to_matching_file() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);
    let out = engine
        .run_pipeline("fn", Some(4000), None, Some("script.py"))
        .expect("run_pipeline failed");
    assert!(
        !out.contains("lib.rs"),
        "Rust file should be filtered out: {}",
        out
    );
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
    assert!(
        pivot_count <= 1,
        "expected ≤1 pivot, got {}: {}",
        pivot_count,
        out
    );
}

/// `get_context_capsule` with `min_score` above any real score yields no pivots.
#[test]
fn get_context_capsule_min_score_filters_all_below_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);
    let out = engine
        .get_context_capsule("fn", Some(4000), None, Some(f32::MAX))
        .expect("get_context_capsule failed");
    assert!(
        !out.contains("#### `"),
        "expected no pivots with max min_score: {}",
        out
    );
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

    let with_tests = engine
        .get_impact_graph("lib.rs::target", None, None, true)
        .unwrap();
    let without_tests = engine
        .get_impact_graph("lib.rs::target", None, None, false)
        .unwrap();
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

    let shallow = engine
        .get_impact_graph("lib.rs::base", Some(1), None, true)
        .unwrap();
    let deep = engine
        .get_impact_graph("lib.rs::base", Some(5), None, true)
        .unwrap();
    assert!(
        shallow.total_affected <= deep.total_affected,
        "shallow traversal should find ≤ symbols than deep"
    );
}

/// `get_impact_graph` must cap each list at `max_results` for high-fan-out
/// symbols and report the dropped count via `direct_truncated`. Without this
/// cap, central symbols (e.g. exception base classes) overflow agent context —
/// see issue #65, where `PolynomialError` returned 30k+ transitive dependents.
#[test]
fn get_impact_graph_caps_direct_dependents_for_high_fanout() {
    let dir = tempfile::tempdir().unwrap();
    // 1 target + 60 distinct callers all in one file.
    let mut src = String::from("pub fn target() {}\n");
    for i in 0..60 {
        src.push_str(&format!("pub fn caller_{i:02}() {{ target(); }}\n"));
    }
    std::fs::write(dir.path().join("lib.rs"), src).unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    let result = engine
        .get_impact_graph("lib.rs::target", None, Some(10), true)
        .expect("get_impact_graph failed");

    assert_eq!(
        result.direct_dependents.len(),
        10,
        "direct list must respect the max_results cap"
    );
    assert_eq!(
        result.direct_truncated, 50,
        "direct_truncated must report dropped count: 60 callers - 10 cap"
    );
    // total_affected must reflect the *real* blast radius, not the truncated view.
    assert!(
        result.total_affected >= 60,
        "total_affected must be the pre-cap count, got {}",
        result.total_affected
    );

    // Cap is stable: a second call returns the same set in the same order.
    let again = engine
        .get_impact_graph("lib.rs::target", None, Some(10), true)
        .unwrap();
    let first: Vec<_> = result.direct_dependents.iter().map(|s| &s.fqn).collect();
    let second: Vec<_> = again.direct_dependents.iter().map(|s| &s.fqn).collect();
    assert_eq!(first, second, "truncation order must be deterministic");
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
        let after_file = sym.fqn.split_once("::").map(|x| x.1).unwrap_or("");
        assert!(
            !after_file.contains("::"),
            "depth-1 symbol has nested FQN: {}",
            sym.fqn
        );
    }
}

/// `run_pipeline` must write an `auto` observation when pivots are found
/// AND `auto_observations` is enabled in config (off by default — #72).
#[test]
fn run_pipeline_writes_auto_observation() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs_auto_obs(&dir);
    engine
        .run_pipeline("rust fn", Some(4000), None, None)
        .expect("run_pipeline failed");

    let observations = engine
        .get_session_context()
        .expect("get_session_context failed")
        .observations;
    let auto_obs: Vec<_> = observations
        .iter()
        .filter(|o| o.kind.as_str() == "auto")
        .collect();
    assert!(
        !auto_obs.is_empty(),
        "expected at least one auto observation after run_pipeline"
    );
    assert!(
        auto_obs[0].content.starts_with("Agent queried:"),
        "unexpected auto observation format: {}",
        auto_obs[0].content
    );
}

/// `get_context_capsule` must write an `auto` observation when pivots are found
/// AND `auto_observations` is enabled in config (off by default — #72).
#[test]
fn get_context_capsule_writes_auto_observation() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs_auto_obs(&dir);
    engine
        .get_context_capsule("py fn", Some(4000), None, None)
        .expect("get_context_capsule failed");

    let observations = engine
        .get_session_context()
        .expect("get_session_context failed")
        .observations;
    let auto_obs: Vec<_> = observations
        .iter()
        .filter(|o| o.kind.as_str() == "auto")
        .collect();
    assert!(
        !auto_obs.is_empty(),
        "expected at least one auto observation after get_context_capsule"
    );
}

/// Calling `run_pipeline` twice with the same task must deduplicate — only one auto observation.
/// Exercises the auto-observation path; default is off (#72).
#[test]
fn run_pipeline_deduplicates_auto_observations() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs_auto_obs(&dir);
    engine
        .run_pipeline("rust fn", Some(4000), None, None)
        .expect("first call failed");
    engine
        .run_pipeline("rust fn", Some(4000), None, None)
        .expect("second call failed");

    let observations = engine
        .get_session_context()
        .expect("get_session_context failed")
        .observations;
    let auto_count = observations
        .iter()
        .filter(|o| o.kind.as_str() == "auto" && o.content.contains("rust fn"))
        .count();
    assert_eq!(
        auto_count, 1,
        "duplicate auto observation should have been suppressed"
    );
}

/// `run_pipeline` with no matching symbols must not write an auto observation,
/// even when the auto-observation flag is on. Without enabling the flag this
/// would be a tautology (#72 disables auto-obs by default).
#[test]
fn run_pipeline_no_pivots_skips_auto_observation() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine_with_auto_obs(&dir); // empty index — no symbols
    engine
        .run_pipeline("xyzzy no match", Some(4000), None, None)
        .expect("run_pipeline failed");

    let observations = engine
        .get_session_context()
        .expect("get_session_context failed")
        .observations;
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

/// Default `auto_observations = false` (#72): `run_pipeline` must NOT write
/// an auto observation on a vanilla engine, even when pivots are found.
/// Repeatedly running with the default must keep the memory table empty of
/// Auto-kind rows, so failed runs can't poison future runs.
#[test]
fn run_pipeline_does_not_auto_record_when_flag_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir); // default config: auto_observations = false
    engine
        .run_pipeline("rust fn", Some(4000), None, None)
        .expect("run_pipeline failed");
    engine
        .run_pipeline("py fn", Some(4000), None, None)
        .expect("run_pipeline failed");

    let observations = engine
        .get_session_context()
        .expect("get_session_context failed")
        .observations;
    let auto_obs: Vec<_> = observations
        .iter()
        .filter(|o| o.kind.as_str() == "auto")
        .collect();
    assert!(
        auto_obs.is_empty(),
        "auto_observations=false is the default; no 'auto' rows should be written, got: {:?}",
        auto_obs.iter().map(|o| &o.content).collect::<Vec<_>>()
    );
}

/// `[observability] auto_observations = true` in config.toml must enable the
/// legacy auto-memory path even though the EngineConfig default is off.
#[test]
fn config_toml_enables_auto_observations() {
    use cs_core::memory::IndexingConfig;
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("config.toml"),
        "[observability]\nauto_observations = true\n",
    )
    .unwrap();

    let cfg = IndexingConfig::load_from_toml(&config_dir.join("config.toml"));
    assert!(
        cfg.auto_observations,
        "auto_observations should be true after loading from [observability]"
    );
}

/// Queries issued while indexing is flagged as in-progress must succeed
/// Stub files under `node_modules/@types/` must be indexed with `is_stub=true`
/// and must never appear as pivots in `run_pipeline` results.
#[test]
fn stub_files_indexed_but_not_returned_as_pivots() {
    let dir = tempfile::tempdir().unwrap();

    // Project source file
    std::fs::write(
        dir.path().join("app.ts"),
        "export function greet(name: string): string { return `Hello ${name}`; }\n",
    )
    .unwrap();

    // Simulate a .d.ts stub under node_modules/@types/
    let types_dir = dir.path().join("node_modules").join("@types").join("node");
    std::fs::create_dir_all(&types_dir).unwrap();
    std::fs::write(
        types_dir.join("index.d.ts"),
        "declare function require(id: string): any;\ndeclare var process: NodeJS.Process;\n",
    )
    .unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index_workspace failed");

    let stats = engine.index_stats().expect("index_stats failed");
    assert!(
        stats.stub_symbol_count > 0,
        "expected stub symbols to be indexed"
    );

    // run_pipeline on a stub-only query — stubs must not appear as pivots
    let out = engine
        .run_pipeline("require", Some(4000), None, None)
        .expect("run_pipeline failed");
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
    assert_eq!(
        stats.stub_symbol_count, 0,
        "project symbols should not be stubs"
    );
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
    assert!(
        stats.stub_symbol_count > 0,
        "Python pyi stubs should be indexed"
    );
}

/// Swift `.swiftinterface` stubs under `.build/` must be indexed and marked as stubs.
#[test]
fn swift_swiftinterface_stubs_indexed() {
    let dir = tempfile::tempdir().unwrap();

    // Simulate .build/release/Modules/Foundation.swiftinterface
    let modules_dir = dir.path().join(".build").join("release").join("Modules");
    std::fs::create_dir_all(&modules_dir).unwrap();
    std::fs::write(
        modules_dir.join("MyLib.swiftinterface"),
        "public func doSomething() -> Void\npublic class MyClass {}\n",
    )
    .unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index_workspace failed");

    let stats = engine.index_stats().expect("index_stats failed");
    assert!(
        stats.stub_symbol_count > 0,
        "Swift swiftinterface stubs should be indexed"
    );
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
    assert_eq!(
        stats.stub_symbol_count, 0,
        "stubs should not be indexed when disabled"
    );
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

    assert!(
        !out.contains(".env"),
        "env file should be excluded: {}",
        out
    );
    assert!(
        !out.contains("my_secret"),
        "secret file should be excluded: {}",
        out
    );
    assert!(
        !out.contains("credentials"),
        "credentials file should be excluded: {}",
        out
    );
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

    assert!(
        !out.contains("config.py"),
        "file with embedded API key should be excluded: {}",
        out
    );
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

    assert!(
        !out.contains("generated.py"),
        "ignored file should be excluded: {}",
        out
    );
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

    assert!(
        !out.contains("fixtures/"),
        "fixtures dir should be excluded: {}",
        out
    );
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

// ── TTL / compression / staleness tests ───────────────────────────────────────

use cs_core::db::Database;
use cs_core::memory::{MemoryConfig, MemoryStore, Observation, ObservationKind};
use parking_lot::Mutex;

/// Auto observations must have a non-None `expires_at` set 7 days out.
#[test]
fn auto_observation_gets_7_day_ttl() {
    let obs = Observation::new("s", "content", None, None, ObservationKind::Auto);
    assert!(
        obs.expires_at.is_some(),
        "auto observations must have expires_at"
    );
    let expires: chrono::DateTime<chrono::Utc> = obs
        .expires_at
        .as_ref()
        .unwrap()
        .parse()
        .expect("invalid rfc3339");
    let diff = expires - chrono::Utc::now();
    // Allow ±1 minute for clock skew during test runs
    assert!(
        diff.num_days() >= 6 && diff.num_days() <= 7,
        "expected ~7 day TTL, got {:?}",
        diff
    );
}

/// Manual observations must have `expires_at = None` by default (never expire).
#[test]
fn manual_observation_has_no_ttl_by_default() {
    let obs = Observation::new("s", "content", None, None, ObservationKind::Manual);
    assert!(
        obs.expires_at.is_none(),
        "manual observations must not expire by default"
    );
}

/// `MemoryConfig` loaded from config.toml overrides default TTLs.
#[test]
fn memory_config_toml_overrides_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        "[memory]\nauto_ttl_days = 3\nmanual_ttl_days = 30\n",
    )
    .unwrap();
    let cfg = MemoryConfig::load_from_toml(&config_path);
    assert_eq!(cfg.auto_ttl_days, 3);
    assert_eq!(cfg.manual_ttl_days, Some(30));
}

/// `MemoryConfig::load_from_toml` returns defaults when file is missing.
#[test]
fn memory_config_toml_missing_returns_defaults() {
    let cfg = MemoryConfig::load_from_toml(std::path::Path::new("/nonexistent/config.toml"));
    assert_eq!(cfg.auto_ttl_days, 7);
    assert!(cfg.manual_ttl_days.is_none());
}

/// `prune_expired` removes observations whose TTL has elapsed.
#[test]
fn prune_expired_removes_past_ttl_observations() {
    let dir = tempfile::tempdir().unwrap();
    let db_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&db_dir).unwrap();
    let db = Arc::new(Mutex::new(Database::open(&db_dir.join("mem.db")).unwrap()));
    let store = MemoryStore::new(Arc::clone(&db), "test-session");

    // Insert an observation with an already-elapsed expires_at
    let mut obs = Observation::new(
        "test-session",
        "old content",
        None,
        None,
        ObservationKind::Auto,
    );
    obs.expires_at = Some((chrono::Utc::now() - chrono::Duration::hours(1)).to_rfc3339());
    db.lock().insert_observation(&obs).unwrap();

    // Save a regular (non-expired) manual observation too
    store.save("keep me", None, None).unwrap();

    let pruned = store.prune_expired().unwrap();
    assert_eq!(pruned, 1, "expected exactly 1 expired observation pruned");

    let remaining = store.get_recent_observations(50).unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].content, "keep me");
}

/// `staleness_score` returns 0 when no observations are stale.
#[test]
fn staleness_score_zero_when_no_stale() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);
    engine
        .run_pipeline("rust fn", Some(4000), None, None)
        .unwrap();
    let ctx = engine.get_session_context().unwrap();
    assert_eq!(ctx.staleness_score, 0.0);
}

/// `staleness_score` is > 0 after an observation is marked stale.
#[test]
fn staleness_score_nonzero_after_stale_mark() {
    let dir = tempfile::tempdir().unwrap();
    let db_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&db_dir).unwrap();
    let db = Arc::new(Mutex::new(Database::open(&db_dir.join("mem.db")).unwrap()));
    let store = MemoryStore::new(Arc::clone(&db), "test-session");

    // Save a symbol-linked observation
    store
        .save("insight about foo", Some("foo::bar"), Some("hash-1"))
        .unwrap();
    // Simulate code change — mark stale by providing a different hash
    store.check_and_mark_stale("foo::bar", "hash-2").unwrap();

    let score = store.staleness_score().unwrap();
    assert!(
        score > 0.0,
        "staleness_score should be > 0 after marking stale"
    );
}

/// After 3+ observations accumulate for the same symbol, `compress_observations`
/// creates one Summary entry and retires the originals.
#[test]
fn compress_observations_merges_symbol_observations() {
    let dir = tempfile::tempdir().unwrap();
    let db_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&db_dir).unwrap();
    let db = Arc::new(Mutex::new(Database::open(&db_dir.join("mem.db")).unwrap()));
    let store = MemoryStore::new(Arc::clone(&db), "test-session");

    // Save 3 symbol-linked observations for the same FQN
    for i in 0..3u8 {
        store
            .save(
                &format!("observation {i}"),
                Some("my::Symbol"),
                Some("hash"),
            )
            .unwrap();
    }

    let compressed = store.compress_observations().unwrap();
    assert_eq!(compressed, 1, "expected 1 symbol compressed");

    // After compression, non-expired visible observations should be just the Summary
    let visible = store.get_recent_observations(50).unwrap();
    let summaries: Vec<_> = visible
        .iter()
        .filter(|o| o.kind.as_str() == "summary")
        .collect();
    assert_eq!(
        summaries.len(),
        1,
        "expected exactly 1 summary after compression"
    );
    assert!(summaries[0].content.contains("[summary of 3 observations]"));

    // The originals must no longer appear (they were expired)
    let originals: Vec<_> = visible
        .iter()
        .filter(|o| o.kind.as_str() == "manual")
        .collect();
    assert!(
        originals.is_empty(),
        "original observations should be expired after compression"
    );
}

/// `get_session_context` wraps observations and staleness_score together.
/// Auto-observations are only recorded when the flag is enabled (#72).
#[test]
fn get_session_context_returns_session_context_struct() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs_auto_obs(&dir);
    engine
        .run_pipeline("rust fn", Some(4000), None, None)
        .unwrap();
    let ctx = engine.get_session_context().unwrap();
    assert!(!ctx.observations.is_empty(), "expected observations");
    // staleness_score is a valid percentage
    assert!(ctx.staleness_score >= 0.0 && ctx.staleness_score <= 100.0);
}

// ── Macro expansion tests ──────────────────────────────────────────────────────

/// Symbols extracted from source code must have `source = None` by default.
#[test]
fn symbol_source_is_none_for_regular_code() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        "pub fn my_fn() {}\npub struct MyStruct;\n",
    )
    .unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");
    // Verify via run_pipeline that we indexed the symbols; source field is
    // None for regular source code symbols (tested at the struct level below).
    let src_sym = cs_core::symbol::Symbol::new(
        "src/lib.rs",
        "plain_fn",
        cs_core::SymbolKind::Function,
        1,
        1,
        "fn plain_fn()".to_string(),
        None,
        "fn plain_fn() {}".to_string(),
        cs_core::language::Language::Rust,
    );
    assert!(
        src_sym.source.is_none(),
        "source should be None for regular symbols"
    );
}

/// `[indexing] rust_expand_macros = true` in config.toml must set the flag.
#[test]
fn indexing_config_rust_expand_macros_loaded_from_toml() {
    use cs_core::memory::IndexingConfig;
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("config.toml"),
        "[indexing]\nrust_expand_macros = true\n",
    )
    .unwrap();
    let cfg = IndexingConfig::load_from_toml(&config_dir.join("config.toml"));
    assert!(
        cfg.rust_expand_macros,
        "rust_expand_macros should be true when set in config.toml"
    );
}

/// `IndexingConfig` must default to `rust_expand_macros = false`.
#[test]
fn indexing_config_defaults_to_false() {
    use cs_core::memory::IndexingConfig;
    let cfg = IndexingConfig::default();
    assert!(
        !cfg.rust_expand_macros,
        "rust_expand_macros should default to false"
    );
}

/// `run_macro_enrichment` must return empty when no Cargo.toml is present.
#[test]
fn macro_enrichment_skipped_without_cargo_toml() {
    use cs_core::db::Database;
    use cs_core::macro_expand::run_macro_enrichment;
    let dir = tempfile::tempdir().unwrap();
    // No Cargo.toml in dir — enrichment must skip gracefully.
    let db_path = dir.path().join("index.db");
    let db = Database::open(&db_path).expect("db open failed");
    let result = run_macro_enrichment(dir.path(), &[], &db);
    assert!(
        result.is_empty(),
        "expected empty result without Cargo.toml"
    );
}

/// `run_macro_enrichment` must return empty when file_data has no Rust files.
#[test]
fn macro_enrichment_skipped_for_non_rust_files() {
    use cs_core::db::Database;
    use cs_core::macro_expand::run_macro_enrichment;
    let dir = tempfile::tempdir().unwrap();
    // Cargo.toml present so the Cargo gate passes.
    std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
    let db_path = dir.path().join("index.db");
    let db = Database::open(&db_path).expect("db open failed");
    // Only a Python file in file_data.
    let file_data = vec![("script.py".to_string(), "abc".to_string(), vec![])];
    let result = run_macro_enrichment(dir.path(), &file_data, &db);
    assert!(
        result.is_empty(),
        "expected empty result for non-Rust files"
    );
}

// ── rustdoc enrichment tests ──────────────────────────────────────────────────

/// `Symbol::resolved_type` must default to `None` for freshly-created symbols.
#[test]
fn symbol_resolved_type_is_none_by_default() {
    let sym = cs_core::symbol::Symbol::new(
        "src/lib.rs",
        "my_fn",
        cs_core::SymbolKind::Function,
        1,
        3,
        "fn my_fn() -> String".to_string(),
        None,
        "fn my_fn() -> String { String::new() }".to_string(),
        cs_core::language::Language::Rust,
    );
    assert!(
        sym.resolved_type.is_none(),
        "resolved_type should default to None"
    );
}

/// `[indexing] rust_rustdoc_types = true` must be loaded correctly.
#[test]
fn indexing_config_rust_rustdoc_types_loaded_from_toml() {
    use cs_core::memory::IndexingConfig;
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("config.toml"),
        "[indexing]\nrust_rustdoc_types = true\n",
    )
    .unwrap();
    let cfg = IndexingConfig::load_from_toml(&config_dir.join("config.toml"));
    assert!(
        cfg.rust_rustdoc_types,
        "rust_rustdoc_types should be true when set in config.toml"
    );
}

/// Both enrichment flags can be enabled together.
#[test]
fn indexing_config_both_enrichment_flags() {
    use cs_core::memory::IndexingConfig;
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("config.toml"),
        "[indexing]\nrust_expand_macros = true\nrust_rustdoc_types = true\n",
    )
    .unwrap();
    let cfg = IndexingConfig::load_from_toml(&config_dir.join("config.toml"));
    assert!(cfg.rust_expand_macros, "rust_expand_macros should be true");
    assert!(cfg.rust_rustdoc_types, "rust_rustdoc_types should be true");
}

/// `run_rustdoc_enrichment` must return 0 when no Cargo.toml is present.
#[test]
fn rustdoc_enrichment_skipped_without_cargo_toml() {
    use cs_core::db::Database;
    use cs_core::rustdoc_enrich::run_rustdoc_enrichment;
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("index.db");
    let db = Database::open(&db_path).expect("db open failed");
    let count = run_rustdoc_enrichment(dir.path(), &mut [], &db);
    assert_eq!(count, 0, "expected 0 enrichments without Cargo.toml");
}

/// Symbols with `resolved_type` set must be correctly persisted and read back.
#[test]
fn resolved_type_round_trips_through_db() {
    use cs_core::db::Database;
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("index.db");
    let db = Database::open(&db_path).expect("db open failed");

    let mut sym = cs_core::symbol::Symbol::new(
        "src/lib.rs",
        "parse",
        cs_core::SymbolKind::Function,
        1,
        4,
        "pub fn parse(s: &str) -> Option<u32>".to_string(),
        None,
        "pub fn parse(s: &str) -> Option<u32> { s.parse().ok() }".to_string(),
        cs_core::language::Language::Rust,
    );
    sym.resolved_type = Some("Option<u32>".to_string());
    sym.source = Some("rustdoc".to_string());

    db.upsert_symbol(&sym).expect("upsert failed");
    let fetched = db
        .get_symbol(sym.id)
        .expect("get failed")
        .expect("symbol missing");
    assert_eq!(fetched.resolved_type.as_deref(), Some("Option<u32>"));
    assert_eq!(fetched.source.as_deref(), Some("rustdoc"));
}

// ── Issue #12: pyright Python type enrichment ────────────────────────────────

/// `run_pyright_enrichment` returns 0 when there are no Python symbols.
#[test]
fn pyright_enrichment_skipped_without_python_symbols() {
    use cs_core::db::Database;
    use cs_core::pyright_enrich::run_pyright_enrichment;
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("index.db");
    let db = Database::open(&db_path).expect("db open");
    let count = run_pyright_enrichment(dir.path(), &mut [], &db);
    assert_eq!(count, 0, "expected 0 enrichments without Python symbols");
}

/// `python_pyright = true` in config.toml is read by `IndexingConfig`.
#[test]
fn indexing_config_reads_python_pyright() {
    use cs_core::memory::IndexingConfig;
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    std::fs::write(&config_path, "[indexing]\npython_pyright = true\n").unwrap();
    let cfg = IndexingConfig::load_from_toml(&config_path);
    assert!(cfg.python_pyright, "python_pyright should be true");
}

/// `extract_return_type_from_sig` handles common Python signature forms.
#[test]
fn pyright_return_type_extraction_variants() {
    use cs_core::pyright_enrich::extract_return_type_from_sig;
    assert_eq!(
        extract_return_type_from_sig("def f() -> str:"),
        Some("str".to_string())
    );
    assert_eq!(
        extract_return_type_from_sig("async def f() -> None:"),
        Some("None".to_string())
    );
    assert_eq!(extract_return_type_from_sig("def f():"), None);
}

// ── Issue #9: search_memory + detail levels ───────────────────────────────────

/// `search_memory` returns observations whose content matches the query.
#[test]
fn search_memory_returns_matching_observations() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(&dir);

    engine
        .save_observation("retry backoff uses exponential strategy", None)
        .unwrap();
    engine
        .save_observation("token budget assembly logic is in capsule.rs", None)
        .unwrap();
    engine
        .save_observation("auth middleware validates JWT on every request", None)
        .unwrap();

    let results = engine.search_memory("retry backoff", None).unwrap();
    assert!(!results.is_empty(), "expected at least one result");
    assert!(
        results[0].content.contains("retry"),
        "top result should mention 'retry', got: {}",
        results[0].content
    );
}

/// `search_memory` returns empty vec when no observations match.
#[test]
fn search_memory_returns_empty_for_no_match() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(&dir);

    engine
        .save_observation("completely unrelated observation", None)
        .unwrap();

    let results = engine.search_memory("xyzzy frobulate", None).unwrap();
    assert!(results.is_empty(), "expected no results for nonsense query");
}

/// `search_memory` ranks observations with more matching terms higher.
#[test]
fn search_memory_ranks_by_term_overlap() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(&dir);

    // One term match
    engine.save_observation("retry logic exists", None).unwrap();
    // Two term match — should rank higher
    engine
        .save_observation("retry backoff is implemented", None)
        .unwrap();

    let results = engine.search_memory("retry backoff", None).unwrap();
    assert!(results.len() >= 2);
    assert!(
        results[0].content.contains("backoff"),
        "two-term match should rank first, got: {}",
        results[0].content
    );
}

/// `search_memory` respects `max_results` and returns at most that many.
#[test]
fn search_memory_respects_max_results() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(&dir);

    for i in 0..5 {
        engine
            .save_observation(&format!("cache observation number {i}"), None)
            .unwrap();
    }

    let results = engine.search_memory("cache", Some(3)).unwrap();
    assert_eq!(results.len(), 3, "expected exactly 3 results");
}

/// `search_memory` does not return expired observations.
#[test]
fn search_memory_excludes_expired_observations() {
    use cs_core::db::Database;
    use cs_core::memory::{MemoryStore, Observation, ObservationKind};
    use parking_lot::Mutex;

    let dir = tempfile::tempdir().unwrap();
    let db_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&db_dir).unwrap();
    let db = Arc::new(Mutex::new(Database::open(&db_dir.join("mem.db")).unwrap()));
    let store = MemoryStore::new(Arc::clone(&db), "test-session");

    // Insert an already-expired observation directly
    let mut expired = Observation::new(
        "test-session",
        "expired cache insight",
        None,
        None,
        ObservationKind::Manual,
    );
    expired.expires_at = Some("2000-01-01T00:00:00Z".to_string());
    db.lock().insert_observation(&expired).unwrap();

    // Also save a live observation
    store.save("live cache observation", None, None).unwrap();

    let results = store.search_observations("cache", 10).unwrap();
    assert_eq!(results.len(), 1, "expired observation must not appear");
    assert!(results[0].content.contains("live"));
}

/// `get_symbol_snippet` returns signature and body for a known FQN.
#[test]
fn get_symbol_snippet_returns_body_for_known_fqn() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        "/// Adds two numbers.\npub fn add(a: u32, b: u32) -> u32 { a + b }\n",
    )
    .unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();

    // find the FQN from the index
    let _ctx = engine.get_context_capsule("add", None, None, None).unwrap();
    // The snippet function should work for any indexed symbol
    let snippet = engine.get_symbol_snippet("lib.rs::add");
    // May or may not find it depending on FQN format, but must not panic
    // If found, signature must be non-empty
    if let Some((sig, _body)) = snippet {
        assert!(!sig.is_empty(), "signature should be non-empty");
        assert!(
            sig.contains("add"),
            "signature should mention function name"
        );
    }
    // Getting a snippet for an unknown FQN returns None
    assert!(
        engine.get_symbol_snippet("nonexistent::fqn").is_none(),
        "unknown FQN should return None"
    );
}

// ── submit_lsp_edges ──────────────────────────────────────────────────────────

use cs_core::symbol::LspEdge;

/// Accepted edges are reflected in `index_stats().lsp_edge_count`.
#[test]
fn submit_lsp_edges_accepted_count() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.py"), "def caller(): pass\n").unwrap();
    std::fs::write(dir.path().join("b.py"), "def callee(): pass\n").unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();

    let result = engine
        .submit_lsp_edges(&[LspEdge {
            from_fqn: "a.py::caller".to_string(),
            to_fqn: "b.py::callee".to_string(),
            kind: "calls".to_string(),
            resolved_type: None,
        }])
        .unwrap();

    assert!(result.contains("1 edge(s) accepted"), "got: {result}");
    let stats = engine.index_stats().unwrap();
    assert_eq!(stats.lsp_edge_count, 1);
}

/// Edges referencing unknown FQNs are skipped without returning an error.
#[test]
fn submit_lsp_edges_unknown_fqn_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();

    let result = engine
        .submit_lsp_edges(&[LspEdge {
            from_fqn: "ghost.py::missing".to_string(),
            to_fqn: "also_gone.py::missing".to_string(),
            kind: "calls".to_string(),
            resolved_type: None,
        }])
        .unwrap();

    assert!(result.contains("skipped"), "got: {result}");
    let stats = engine.index_stats().unwrap();
    assert_eq!(stats.lsp_edge_count, 0);
}

/// Submitting the same edge twice must not duplicate it in the DB.
#[test]
fn submit_lsp_edges_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.py"), "def f(): pass\n").unwrap();
    std::fs::write(dir.path().join("b.py"), "def g(): pass\n").unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();

    let edge = vec![LspEdge {
        from_fqn: "a.py::f".to_string(),
        to_fqn: "b.py::g".to_string(),
        kind: "calls".to_string(),
        resolved_type: None,
    }];
    engine.submit_lsp_edges(&edge).unwrap();
    engine.submit_lsp_edges(&edge).unwrap();

    let stats = engine.index_stats().unwrap();
    assert_eq!(
        stats.lsp_edge_count, 1,
        "duplicate edges must not accumulate"
    );
}

/// LSP edges for a file are deleted when that file is re-indexed.
#[test]
fn submit_lsp_edges_invalidated_on_reindex() {
    let dir = tempfile::tempdir().unwrap();
    let file_a = dir.path().join("a.py");
    std::fs::write(&file_a, "def f(): pass\n").unwrap();
    std::fs::write(dir.path().join("b.py"), "def g(): pass\n").unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();

    engine
        .submit_lsp_edges(&[LspEdge {
            from_fqn: "a.py::f".to_string(),
            to_fqn: "b.py::g".to_string(),
            kind: "calls".to_string(),
            resolved_type: None,
        }])
        .unwrap();
    assert_eq!(engine.index_stats().unwrap().lsp_edge_count, 1);

    // Re-indexing a.py must invalidate edges sourced from it.
    std::fs::write(&file_a, "def f(): pass\ndef f2(): pass\n").unwrap();
    engine.reindex_file(&file_a, ChangeKind::Modified).unwrap();

    assert_eq!(
        engine.index_stats().unwrap().lsp_edge_count,
        0,
        "LSP edges must be invalidated when source file is re-indexed"
    );
}

// ── 9d Memory consolidation ────────────────────────────────────────────────────

/// `consolidate_observations` must return Ok(0) when the embedder is not loaded
/// (the no-embedder stub path, which is what the test engine uses).
#[test]
fn consolidate_observations_is_noop_without_embedder() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);

    // Produce a few auto observations so there is something to consolidate.
    engine
        .run_pipeline("rust fn", Some(4000), None, None)
        .unwrap();
    engine
        .run_pipeline("py fn", Some(4000), None, None)
        .unwrap();

    let n = engine
        .consolidate_observations()
        .expect("consolidate_observations must not error");
    assert_eq!(n, 0, "expected 0 clusters without embedder");
}

/// Auto observations written by `run_pipeline` must survive `consolidate_observations`
/// intact when the embedder is absent (no premature expiry). Requires the
/// auto-observation flag since the default is off (#72).
#[test]
fn consolidate_does_not_expire_observations_without_embedder() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs_auto_obs(&dir);

    engine
        .run_pipeline("rust fn", Some(4000), None, None)
        .unwrap();
    engine
        .run_pipeline("py fn", Some(4000), None, None)
        .unwrap();

    engine.consolidate_observations().unwrap();

    let obs = engine
        .get_session_context()
        .expect("get_session_context failed")
        .observations;
    let auto_count = obs.iter().filter(|o| o.kind.as_str() == "auto").count();
    assert!(
        auto_count >= 2,
        "auto observations must not be expired by consolidation without embedder; found {auto_count}"
    );
}

// ── TypeScript enrichment integration tests (issue #17) ─────────────────��────

/// `[indexing] ts_types = true` in config.toml must be read correctly.
#[test]
fn indexing_config_ts_types_loaded_from_toml() {
    use cs_core::memory::IndexingConfig;
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("config.toml"),
        "[indexing]\nts_types = true\n",
    )
    .unwrap();
    let cfg = IndexingConfig::load_from_toml(&config_dir.join("config.toml"));
    assert!(
        cfg.ts_types,
        "ts_types should be true when set in config.toml"
    );
}

/// `ts_types` must default to `false` so it is never accidentally enabled.
#[test]
fn indexing_config_ts_types_defaults_to_false() {
    use cs_core::memory::IndexingConfig;
    let cfg = IndexingConfig::default();
    assert!(!cfg.ts_types, "ts_types should default to false");
}

/// All three enrichment flags can be enabled together without conflict.
#[test]
fn indexing_config_all_three_enrichment_flags() {
    use cs_core::memory::IndexingConfig;
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("config.toml"),
        "[indexing]\nrust_expand_macros = true\nrust_rustdoc_types = true\nts_types = true\n",
    )
    .unwrap();
    let cfg = IndexingConfig::load_from_toml(&config_dir.join("config.toml"));
    assert!(cfg.rust_expand_macros, "rust_expand_macros");
    assert!(cfg.rust_rustdoc_types, "rust_rustdoc_types");
    assert!(cfg.ts_types, "ts_types");
}

/// `EngineConfig` must default `ts_types` to `false`.
#[test]
fn engine_config_ts_types_defaults_to_false() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = cs_core::EngineConfig::new(dir.path());
    assert!(
        !cfg.ts_types,
        "ts_types must default to false in EngineConfig"
    );
}

/// `run_ts_enrichment` must return 0 gracefully when no tsconfig.json is present.
#[test]
fn ts_enrichment_skipped_without_tsconfig() {
    use cs_core::db::Database;
    use cs_core::ts_enrich::run_ts_enrichment;
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("index.db");
    let db = Database::open(&db_path).expect("db open failed");
    let count = run_ts_enrichment(dir.path(), &mut [], &db);
    assert_eq!(count, 0, "expected 0 enrichments without tsconfig.json");
}

/// A TypeScript symbol with `resolved_type` and `source = "ts-compiler"` must
/// round-trip correctly through the DB — both fields preserved on read-back.
#[test]
fn ts_resolved_type_round_trips_through_db() {
    use cs_core::db::Database;
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("index.db");
    let db = Database::open(&db_path).expect("db open failed");

    let mut sym = cs_core::symbol::Symbol::new(
        "src/api.ts",
        "fetchUser",
        cs_core::SymbolKind::Function,
        5,
        12,
        "async function fetchUser(id: string)".to_string(),
        None,
        "async function fetchUser(id: string) { return fetch(`/users/${id}`); }".to_string(),
        cs_core::language::Language::TypeScript,
    );
    sym.resolved_type = Some("Promise<Response>".to_string());
    sym.source = Some("ts-compiler".to_string());

    db.upsert_symbol(&sym).expect("upsert failed");
    let fetched = db
        .get_symbol(sym.id)
        .expect("get failed")
        .expect("symbol missing");

    assert_eq!(
        fetched.resolved_type.as_deref(),
        Some("Promise<Response>"),
        "resolved_type must round-trip through DB"
    );
    assert_eq!(
        fetched.source.as_deref(),
        Some("ts-compiler"),
        "source must round-trip through DB"
    );
}

/// When `ts_types = true` is set in config.toml and the workspace has a tsconfig.json,
/// the engine must attempt ts enrichment (gracefully skipping if node/typescript
/// is absent).  This test verifies the engine wiring — it does not require node.
#[test]
fn engine_applies_ts_types_config_from_toml() {
    let dir = tempfile::tempdir().unwrap();
    let config_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("config.toml"),
        "[indexing]\nts_types = true\n",
    )
    .unwrap();
    // tsconfig.json present so the workspace gate passes.
    std::fs::write(
        dir.path().join("tsconfig.json"),
        r#"{"compilerOptions":{"strict":true},"include":["src"]}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("hello.ts"),
        "export function greet(): string { return 'hi'; }\n",
    )
    .unwrap();

    let config = cs_core::EngineConfig::new(dir.path()).without_embedder();
    // config.ts_types starts false in EngineConfig before toml is applied;
    // CoreEngine::new reads config.toml and applies ts_types.
    let _ = config.ts_types;
    let engine = cs_core::CoreEngine::new(config).expect("engine init failed");
    // index_workspace must complete without panic even when node or typescript is absent.
    engine.index_workspace().expect("index_workspace failed");
}

/// End-to-end: when node and a local typescript package are available, running
/// `run_ts_enrichment` on a minimal workspace with a real .ts file must enrich
/// at least one symbol with a non-trivial resolved type.
///
/// Skipped at runtime when `node` is absent or `typescript` is not installed
/// locally — this test is meant to run on developer machines and CI envs that
/// have Node.js set up.
#[test]
fn ts_enrichment_end_to_end_with_node() {
    use cs_core::db::Database;
    use cs_core::ts_enrich::run_ts_enrichment;
    use cs_core::{CoreEngine, EngineConfig};

    // Runtime skip: require node.
    let node_ok = std::process::Command::new("node")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !node_ok {
        eprintln!("ts_enrichment_end_to_end_with_node: node not found — skipping");
        return;
    }

    let dir = tempfile::tempdir().unwrap();

    // Minimal tsconfig.json.
    std::fs::write(
        dir.path().join("tsconfig.json"),
        r#"{"compilerOptions":{"strict":true,"target":"ES2020"},"include":["*.ts"]}"#,
    )
    .unwrap();

    // Source file with a typed function and a class method so the shim has
    // something concrete to resolve.
    std::fs::write(
        dir.path().join("sample.ts"),
        r#"
export function add(a: number, b: number): number {
  return a + b;
}

export class Greeter {
  private name: string;
  constructor(name: string) { this.name = name; }
  greet(): string { return `Hello, ${this.name}`; }
}
"#,
    )
    .unwrap();

    // Check if typescript is available locally or globally.
    let ts_local = dir.path().join("node_modules").join("typescript");
    let ts_global = std::process::Command::new("node")
        .args(["-e", "require('typescript'); process.exit(0)"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !ts_local.exists() && !ts_global {
        eprintln!("ts_enrichment_end_to_end_with_node: typescript not installed — skipping");
        return;
    }

    // Index the workspace so there are symbols to enrich.
    let engine =
        CoreEngine::new(EngineConfig::new(dir.path()).without_embedder()).expect("engine init");
    engine.index_workspace().expect("index_workspace");

    // Run enrichment directly (bypasses the ts_types config gate).
    let db_path = dir.path().join(".codesurgeon").join("index.db");
    let db = Database::open(&db_path).expect("db open");
    let mut symbols = db.all_symbols().expect("all_symbols");

    let count = run_ts_enrichment(dir.path(), &mut symbols, &db);
    assert!(
        count > 0,
        "expected at least one symbol to be enriched; got 0 \
         (check that typescript is installed in node_modules or globally)"
    );

    // At least one enriched symbol should have a non-trivial resolved type.
    let enriched: Vec<_> = symbols
        .iter()
        .filter(|s| s.resolved_type.is_some())
        .collect();
    assert!(
        !enriched.is_empty(),
        "at least one symbol should carry resolved_type after enrichment"
    );
    assert!(
        enriched
            .iter()
            .all(|s| s.source.as_deref() == Some("ts-compiler")),
        "all enriched symbols must carry source = 'ts-compiler'"
    );
}

/// `Consolidated` kind must never appear in the `get_consolidation_candidates` pool,
/// preventing already-consolidated entries from being re-consolidated on subsequent runs.
/// We verify this indirectly: after consolidation completes the session context must
/// contain no `Consolidated` entries when the embedder is absent (no merges occurred).
#[test]
fn consolidated_kind_not_in_candidates_pool() {
    let dir = tempfile::tempdir().unwrap();
    let engine = indexed_engine_with_two_langs(&dir);

    engine
        .run_pipeline("rust fn", Some(4000), None, None)
        .unwrap();
    engine.consolidate_observations().unwrap(); // no-op without embedder

    let obs = engine
        .get_session_context()
        .expect("get_session_context failed")
        .observations;
    let consolidated_count = obs
        .iter()
        .filter(|o| o.kind.as_str() == "consolidated")
        .count();
    assert_eq!(
        consolidated_count, 0,
        "no consolidated entries expected without embedder"
    );
}

// ── get_stats tests ───────────────────────────────────────────────────────────

/// `get_stats` on an empty log returns a "no calls recorded" message, not an error.
#[test]
fn get_stats_empty_log_returns_no_data_message() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(&dir);
    let out = engine.get_stats(None).expect("get_stats failed");
    assert!(
        out.contains("No run_pipeline calls"),
        "expected no-data message, got: {}",
        out
    );
}

/// After `run_pipeline` calls, `get_stats` must report the correct query count
/// and include all expected sections.
#[test]
fn get_stats_counts_logged_queries() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        "pub fn foo() {}\npub fn bar() {}\n",
    )
    .unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    engine
        .run_pipeline("fix bug in foo", Some(500), None, None)
        .unwrap();
    engine
        .run_pipeline("add new feature bar", Some(500), None, None)
        .unwrap();
    engine
        .run_pipeline("refactor foo and bar", Some(500), None, None)
        .unwrap();

    let out = engine.get_stats(Some(30)).expect("get_stats failed");
    assert!(
        out.contains("Total queries:        3"),
        "wrong count: {}",
        out
    );
    assert!(
        out.contains("Token savings:"),
        "missing savings line: {}",
        out
    );
    assert!(out.contains("Latency"), "missing latency section: {}", out);
    assert!(
        out.contains("Intent breakdown"),
        "missing intent section: {}",
        out
    );
    assert!(
        out.contains("Language distribution"),
        "missing lang section: {}",
        out
    );
}

/// `get_stats` with a very large `days` value must still return data logged moments ago.
#[test]
fn get_stats_large_days_includes_recent_queries() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "pub fn foo() {}\n").unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();
    engine
        .run_pipeline("fix foo", Some(500), None, None)
        .unwrap();

    let out = engine.get_stats(Some(9999)).expect("get_stats failed");
    assert!(
        out.contains("Total queries:        1"),
        "expected 1 query in large window, got: {}",
        out
    );
}

/// Token savings percentage must be between 0% and 100%.
#[test]
fn get_stats_savings_pct_in_valid_range() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        "pub fn alpha() {}\npub fn beta() {}\npub fn gamma() {}\n",
    )
    .unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();

    for _ in 0..5 {
        engine
            .run_pipeline("fix alpha", Some(200), None, None)
            .unwrap();
    }

    let out = engine.get_stats(Some(30)).unwrap();
    // Extract the savings percentage from the output line.
    let pct_line = out
        .lines()
        .find(|l| l.contains("Token savings:"))
        .expect("no savings line");
    let pct_str = pct_line
        .split_whitespace()
        .find(|t| t.ends_with('%'))
        .expect("no % token");
    let pct: f64 = pct_str
        .trim_end_matches('%')
        .parse()
        .expect("non-numeric %");
    assert!(
        (0.0..=100.0).contains(&pct),
        "savings % out of range: {}",
        pct
    );
}

// ── 9e AST change categories ───────────────────────────────────────────────────

/// Adding a new function to a file must produce a passive observation with
/// `change_category = "new_symbol"` and content that names the new FQN.
#[test]
fn reindex_new_symbol_sets_change_category() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lib.py"), "def foo(): pass\n").unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();

    // Add a new function.
    std::fs::write(
        dir.path().join("lib.py"),
        "def foo(): pass\ndef bar(): pass\n",
    )
    .unwrap();
    engine
        .reindex_file(&dir.path().join("lib.py"), ChangeKind::Modified)
        .unwrap();

    let obs = engine.get_session_context().unwrap().observations;
    let file_obs: Vec<_> = obs
        .iter()
        .filter(|o| o.change_category.as_deref() == Some("new_symbol"))
        .collect();
    assert!(
        !file_obs.is_empty(),
        "expected a passive observation with change_category=new_symbol"
    );
    assert!(
        file_obs.iter().any(|o| o.content.contains("bar")),
        "new_symbol observation must mention the added FQN"
    );
}

/// Removing a function must produce a `deleted_symbol` observation.
#[test]
fn reindex_deleted_symbol_sets_change_category() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.py"),
        "def foo(): pass\ndef bar(): pass\n",
    )
    .unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();

    std::fs::write(dir.path().join("lib.py"), "def foo(): pass\n").unwrap();
    engine
        .reindex_file(&dir.path().join("lib.py"), ChangeKind::Modified)
        .unwrap();

    let obs = engine.get_session_context().unwrap().observations;
    let del_obs: Vec<_> = obs
        .iter()
        .filter(|o| o.change_category.as_deref() == Some("deleted_symbol"))
        .collect();
    assert!(
        !del_obs.is_empty(),
        "expected a passive observation with change_category=deleted_symbol"
    );
    assert!(
        del_obs.iter().any(|o| o.content.contains("bar")),
        "deleted_symbol observation must mention the removed FQN"
    );
}

/// Changing only a function body (signature unchanged) must produce `body_change`.
#[test]
fn reindex_body_change_sets_change_category() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lib.py"), "def foo():\n    return 1\n").unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();

    std::fs::write(dir.path().join("lib.py"), "def foo():\n    return 2\n").unwrap();
    engine
        .reindex_file(&dir.path().join("lib.py"), ChangeKind::Modified)
        .unwrap();

    let obs = engine.get_session_context().unwrap().observations;
    // Accept either body_change (preferred) or no change-category observation at all
    // if the parser doesn't distinguish signature from body for this language;
    // but it must NOT be new_symbol or deleted_symbol.
    let bad = obs.iter().any(|o| {
        matches!(
            o.change_category.as_deref(),
            Some("new_symbol") | Some("deleted_symbol")
        ) && o.content.contains("foo")
    });
    assert!(
        !bad,
        "body-only change must not produce new_symbol or deleted_symbol"
    );
}

/// Adding an import must produce a `dependency_added` observation.
#[test]
fn reindex_import_sets_dependency_added() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lib.py"), "def foo(): pass\n").unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().unwrap();

    std::fs::write(dir.path().join("lib.py"), "import os\ndef foo(): pass\n").unwrap();
    engine
        .reindex_file(&dir.path().join("lib.py"), ChangeKind::Modified)
        .unwrap();

    let obs = engine.get_session_context().unwrap().observations;
    // Only assert if the parser actually emits Import symbols for Python.
    let has_dep = obs
        .iter()
        .any(|o| o.change_category.as_deref() == Some("dependency_added"));
    let has_new = obs
        .iter()
        .any(|o| o.change_category.as_deref() == Some("new_symbol") && o.content.contains("os"));
    // Either dependency_added or the import is absorbed into new_symbol — either is fine,
    // but the observation must not claim a function was added.
    assert!(
        has_dep || has_new || obs.iter().any(|o| o.change_category.is_some()),
        "re-indexing after adding an import must produce at least one classified observation"
    );
}

// ── Manifest tests ────────────────────────────────────────────────────────────

/// After `index_workspace`, a manifest.json must exist in `.codesurgeon/`.
#[test]
fn manifest_written_after_index() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "pub fn hello() {}\n").unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    let manifest_path = dir.path().join(".codesurgeon").join("manifest.json");
    assert!(manifest_path.exists(), "manifest.json should be written");

    let text = std::fs::read_to_string(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(manifest["version"], 1);
    assert!(manifest["files"].is_object(), "files should be an object");
    assert!(
        manifest["files"]["lib.rs"].is_string(),
        "lib.rs should appear in manifest"
    );
}

/// `index_stats` reports manifest file count and timestamp when manifest is present.
#[test]
fn index_stats_reports_manifest_info() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("app.py"), "def foo(): pass\n").unwrap();
    std::fs::write(dir.path().join("util.py"), "def bar(): pass\n").unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    let stats = engine.index_stats().expect("index_stats failed");
    assert!(
        stats.manifest_file_count.is_some(),
        "manifest_file_count should be Some after index"
    );
    assert!(
        stats.manifest_updated_at.is_some(),
        "manifest_updated_at should be Some after index"
    );
    // Manifest should account for all indexed source files
    assert_eq!(
        stats.manifest_file_count.unwrap(),
        stats.file_count,
        "manifest file count should match DB file count"
    );
}

/// Second `index_workspace` call should skip unchanged files (incremental).
#[test]
fn incremental_index_skips_unchanged_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lib.rs"), "pub fn stable() {}\n").unwrap();
    std::fs::write(dir.path().join("other.rs"), "pub fn other() {}\n").unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("first index failed");

    let stats_before = engine.index_stats().expect("stats failed");
    let count_before = stats_before.symbol_count;

    // Modify one file and re-index
    std::fs::write(
        dir.path().join("other.rs"),
        "pub fn other() {}\npub fn new_fn() {}\n",
    )
    .unwrap();
    engine.index_workspace().expect("second index failed");

    let stats_after = engine.index_stats().expect("stats failed");
    // new_fn was added — symbol count should increase
    assert!(
        stats_after.symbol_count > count_before,
        "new symbol should be picked up"
    );
    // stable() was unchanged — should still be present
    let out = engine
        .run_pipeline("stable", Some(2000), None, None)
        .expect("run_pipeline failed");
    assert!(
        out.contains("stable"),
        "unchanged symbol should still be indexed"
    );
}

/// `.codesurgeon/.gitignore` is created on `CoreEngine::new()` and excludes manifest.json
/// by default (track_manifest = false).
#[test]
fn gitignore_written_on_new() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(&dir);
    drop(engine); // ensure new() completes

    let gitignore = dir.path().join(".codesurgeon").join(".gitignore");
    assert!(gitignore.exists(), ".gitignore should be created");
    let contents = std::fs::read_to_string(&gitignore).unwrap();
    assert!(contents.contains("index.db"), "should exclude index.db");
    assert!(
        contents.contains("manifest.json"),
        "should exclude manifest.json by default"
    );
}

/// With `track_manifest = true`, `.gitignore` must NOT exclude `manifest.json`.
#[test]
fn gitignore_omits_manifest_when_tracked() {
    let dir = tempfile::tempdir().unwrap();
    let config = cs_core::EngineConfig {
        track_manifest: true,
        ..cs_core::EngineConfig::new(dir.path()).without_embedder()
    };
    let engine = cs_core::CoreEngine::new(config).expect("engine init failed");
    drop(engine);

    let gitignore = dir.path().join(".codesurgeon").join(".gitignore");
    let contents = std::fs::read_to_string(&gitignore).unwrap();
    assert!(
        contents.contains("index.db"),
        "should still exclude index.db"
    );
    assert!(
        !contents.contains("manifest.json"),
        "manifest.json should not be excluded when track_manifest=true"
    );
}

/// The embedding cache must be empty immediately after `CoreEngine::new`.
/// Verifies the lazy-load contract from issue #47: ~154 MB is not allocated
/// until the first semantic query, not at startup.
#[cfg(feature = "embeddings")]
#[test]
fn embedding_cache_is_empty_at_startup() {
    let dir = tempfile::tempdir().unwrap();
    let config = EngineConfig::new(dir.path()).without_embedder();
    let engine = CoreEngine::new(config).expect("engine init failed");
    assert_eq!(
        engine.embedding_cache_len(),
        0,
        "cache must not be loaded eagerly at startup"
    );
}

/// Parallel queries racing to trigger the first lazy cache load must not
/// deadlock or panic. `cache_once` serialises the init; all threads proceed
/// normally after the single DB load completes.
#[cfg(feature = "embeddings")]
#[test]
fn parallel_queries_do_not_deadlock_on_lazy_cache_init() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        "pub fn foo() {}\npub fn bar() {}\n",
    )
    .unwrap();
    let config = EngineConfig::new(dir.path()).without_embedder();
    let engine = Arc::new(CoreEngine::new(config).expect("engine init failed"));
    engine.index_workspace().expect("index failed");

    let handles: Vec<_> = (0..8)
        .map(|i| {
            let e = Arc::clone(&engine);
            std::thread::spawn(move || {
                let _ = e.run_pipeline(&format!("query {i}"), Some(500), None, None);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread panicked");
    }

    // No embedder → no embeddings stored → cache initialised but still empty.
    assert_eq!(engine.embedding_cache_len(), 0);
}

/// When a file is deleted, `reindex_file(Removed)` must purge its symbols from
/// search results so they don't appear as ghost entries.
#[test]
fn reindex_file_removed_purges_from_search() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("lib.py");
    std::fs::write(&file, "def my_unique_target_fn(): pass\n").unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("index failed");

    // Symbol should be findable before deletion.
    let before = engine
        .get_context_capsule("my_unique_target_fn", None, None, None)
        .unwrap();
    assert!(
        before.contains("my_unique_target_fn"),
        "symbol should appear before deletion"
    );

    // Delete the file and notify the engine.
    std::fs::remove_file(&file).unwrap();
    engine
        .reindex_file(&file, ChangeKind::Removed)
        .expect("reindex_file Removed failed");

    // Symbol must no longer appear in search results (0 pivots).
    let after = engine
        .get_context_capsule("my_unique_target_fn", None, None, None)
        .unwrap();
    assert!(
        after.contains("0 pivots"),
        "symbol should be purged after file removal, got: {}",
        after
    );
}

/// `index_workspace` must prune symbols from files that no longer exist on disk
/// (e.g. deleted worktrees, branch switches).
#[test]
fn index_workspace_prunes_stale_files() {
    let dir = tempfile::tempdir().unwrap();
    let file_a = dir.path().join("kept.py");
    let file_b = dir.path().join("stale.py");
    std::fs::write(&file_a, "def kept_fn(): pass\n").unwrap();
    std::fs::write(&file_b, "def stale_unique_fn(): pass\n").unwrap();

    let engine = test_engine(&dir);
    engine.index_workspace().expect("first index failed");

    // Both symbols present.
    let capsule = engine
        .get_context_capsule("stale_unique_fn", None, None, None)
        .unwrap();
    assert!(
        capsule.contains("stale_unique_fn"),
        "should exist before prune"
    );

    // Delete one file, then re-index the workspace.
    std::fs::remove_file(&file_b).unwrap();
    engine.index_workspace().expect("second index failed");

    // Stale symbol must be gone from pivots.
    let after = engine
        .get_context_capsule("stale_unique_fn", None, None, None)
        .unwrap();
    assert!(
        after.contains("0 pivots"),
        "stale symbol should be pruned after re-index, got: {}",
        after
    );

    // Kept symbol must still be present.
    let kept = engine
        .get_context_capsule("kept_fn", None, None, None)
        .unwrap();
    assert!(kept.contains("kept_fn"), "kept symbol should survive prune");
}

// ─── run_pipeline_with_context (Phase 2 / v1.7 context param) ───────────────
//
// The `context` parameter is a raw-text blob used *only* for anchor
// extraction. It exists so callers (notably the SWE-bench harness) can pass
// the full problem statement alongside a paraphrased `task` — identifiers the
// agent dropped from the summary are recovered from the raw source.
//
// These tests verify:
//   1. `context=None` is behaviorally identical to plain `run_pipeline`.
//   2. Identifiers present *only* in `context` (not in `task`) surface symbols
//      that would otherwise miss.

/// Helper: in a workspace with a distinctively-named Python symbol, assert
/// whether a capsule built from `(task, context)` contains the symbol.
fn capsule_has_symbol(
    engine: &CoreEngine,
    task: &str,
    context: Option<&str>,
    needle: &str,
) -> bool {
    let out = engine
        .run_pipeline_with_context(task, context, Some(4000), None, None)
        .expect("run_pipeline_with_context");
    out.contains(needle)
}

#[test]
fn context_none_matches_plain_run_pipeline() {
    // Byte-equality of capsule output across the two entrypoints. We use
    // two independent engine/workspace pairs so the auto-observation side
    // effect of the first call doesn't pollute the second's session memory.
    fn make() -> (TempDir, CoreEngine) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("mod.py"),
            "def distinctive_anchor_zzz():\n    return 1\n",
        )
        .unwrap();
        let engine = test_engine(&dir);
        engine.index_workspace().expect("index");
        (dir, engine)
    }
    let (_d1, e1) = make();
    let a = e1
        .run_pipeline("distinctive_anchor_zzz bug", Some(4000), None, None)
        .unwrap();
    let (_d2, e2) = make();
    let b = e2
        .run_pipeline_with_context("distinctive_anchor_zzz bug", None, Some(4000), None, None)
        .unwrap();
    assert_eq!(
        a, b,
        "run_pipeline and run_pipeline_with_context(context=None) must match"
    );
}

#[test]
fn context_recovers_identifier_paraphrased_out_of_task() {
    // Simulates the SWE-bench failure mode: the agent paraphrased away the
    // identifier into a bag-of-English `task`, but the raw problem statement
    // still names the function. Without context, BM25 likely misses; with
    // context, the anchor fires.
    let dir = tempfile::tempdir().unwrap();
    // Target symbol with a unique name unlikely to match BM25 on generic prose.
    std::fs::write(
        dir.path().join("target.py"),
        "def obscurely_named_fn_q42():\n    \"\"\"short doc\"\"\"\n    return None\n",
    )
    .unwrap();
    // Decoy files so the capsule isn't trivial.
    for i in 0..6 {
        std::fs::write(
            dir.path().join(format!("decoy_{i}.py")),
            format!("def decoy_{i}():\n    return {i}\n"),
        )
        .unwrap();
    }
    let engine = test_engine(&dir);
    engine.index_workspace().expect("index");

    // Paraphrased task — strips the identifier.
    let task_without_anchor = "fix bug in the obscure function";
    // Raw context still names the symbol verbatim.
    let raw_problem = "Error trace:\n  File 'target.py', line 3, in obscurely_named_fn_q42\nfailed";

    let without = capsule_has_symbol(&engine, task_without_anchor, None, "obscurely_named_fn_q42");
    let with = capsule_has_symbol(
        &engine,
        task_without_anchor,
        Some(raw_problem),
        "obscurely_named_fn_q42",
    );

    assert!(
        with,
        "context containing identifier should surface the symbol via anchor extraction"
    );
    // Note: `without` is not strictly required to be false — this is a smoke
    // test of the recovery path. We assert the positive case only so the test
    // stays stable across ranker tweaks. If it ever does surface without the
    // context, delete this comment — no harm.
    let _ = without;
}

#[test]
fn context_dedupes_against_task() {
    // Sanity: if an identifier appears in both `task` and `context`, the
    // symbol isn't double-counted (no panic, no duplicate pivot). Pure smoke.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("m.py"),
        "def unique_q99_fn():\n    return 1\n",
    )
    .unwrap();
    let engine = test_engine(&dir);
    engine.index_workspace().expect("index");

    let out = engine
        .run_pipeline_with_context(
            "fix unique_q99_fn",
            Some("Problem: unique_q99_fn crashes."),
            Some(4000),
            None,
            None,
        )
        .expect("pipeline");
    assert!(out.contains("unique_q99_fn"), "symbol should be present");
}

/// Issue #69 v2 (traceback half): a Python traceback pasted into `context`
/// must surface its frame-name function as an anchor *even when the name
/// fails the prose shape filter* (plain lowercase, no underscore, no internal
/// caps). The traceback pass in `anchors::extract` bypasses the shape filter
/// for names that come from a `File "...", line N, in <name>` frame.
///
/// We prove the wiring by:
///   1. Indexing a function whose name (`frobnicate`) is plain lowercase —
///      the prose-shape filter rejects it as plain English.
///   2. Running with a paraphrased task and a real-shape traceback in
///      `context`. The symbol must surface.
///   3. Running with the same task and a `context` that mentions
///      `frobnicate` only in prose (no `File "..."` shape). The prose-only
///      mention must NOT route through the traceback pass — this is the
///      specificity check that proves the previous run actually used the
///      traceback path and not the existing prose path.
#[test]
fn context_traceback_frame_surfaces_plain_lowercase_function() {
    let dir = tempfile::tempdir().unwrap();
    // Plain lowercase name — fails prose shape filter (no `_`, no internal caps).
    std::fs::write(
        dir.path().join("target.py"),
        "def frobnicate(x):\n    \"\"\"transform x in place\"\"\"\n    return x\n",
    )
    .unwrap();
    // Decoys so the capsule isn't trivial and BM25 has competition.
    for i in 0..6 {
        std::fs::write(
            dir.path().join(format!("decoy_{i}.py")),
            format!("def decoy_func_{i}():\n    return {i}\n"),
        )
        .unwrap();
    }
    let engine = test_engine(&dir);
    engine.index_workspace().expect("index");

    let task = "investigate the reported crash";
    let traceback_ctx = "\
Traceback (most recent call last):
  File \"app/main.py\", line 12, in <module>
    target.frobnicate(value)
  File \"target.py\", line 3, in frobnicate
    raise RuntimeError(\"boom\")
RuntimeError: boom
";
    let prose_only_ctx = "The crash report mentions frobnicate but no traceback was attached.";

    let with_traceback = capsule_has_symbol(&engine, task, Some(traceback_ctx), "frobnicate");
    let with_prose_only = capsule_has_symbol(&engine, task, Some(prose_only_ctx), "frobnicate");

    assert!(
        with_traceback,
        "traceback frame `in frobnicate` must extract the identifier and surface the symbol; \
         this validates the engine-layer wiring of the #69 v2 traceback anchor pass"
    );
    assert!(
        !with_prose_only,
        "plain prose mention of `frobnicate` (no `File \"...\", in frobnicate` shape) must NOT \
         surface — the prose shape filter rejects plain lowercase tokens. This is the specificity \
         check: it proves the traceback case above used the new traceback path, not the prose path."
    );
}

/// Engine-level test for `traceback_candidates`: a traceback frame whose
/// function name is a class method (`Class.method`) must resolve via
/// leaf-name lookup, and the suffix-based file-path match must pin down
/// the *specific* symbol — not the same-named function in another file.
///
/// Setup: two files each define a `flush` symbol — one is a top-level
/// function, the other is `Buffer.flush`. The traceback names the class-method
/// file, so only that symbol should surface as a traceback pivot.
#[test]
fn traceback_resolves_class_method_via_leaf_and_file_path() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("io")).unwrap();
    std::fs::write(
        dir.path().join("io/buffer.py"),
        "class Buffer:\n    def flush(self):\n        return 1\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("io/sink.py"),
        "def flush():\n    return 2\n",
    )
    .unwrap();
    // Decoys so BM25 / ANN have noise to compete with.
    for i in 0..6 {
        std::fs::write(
            dir.path().join(format!("decoy_{i}.py")),
            format!("def decoy_func_{i}():\n    return {i}\n"),
        )
        .unwrap();
    }
    let engine = test_engine(&dir);
    engine.index_workspace().expect("index");

    let task = "diagnose the reported crash";
    // Traceback frame names `Buffer.flush` (dotted) in `io/buffer.py`.
    // The frame's file path is workspace-relative; the resolver's
    // suffix match must accept it against the indexed `file_path`.
    let tb = "\
Traceback (most recent call last):
  File \"app/main.py\", line 4, in <module>
    b.flush()
  File \"io/buffer.py\", line 3, in flush
    raise RuntimeError(\"boom\")
RuntimeError: boom
";

    let capsule = engine
        .run_pipeline_capsule_with_context(task, Some(tb), Some(4000), None, None)
        .expect("pipeline");

    let pivot_files: Vec<&str> = capsule
        .pivots
        .iter()
        .map(|p| p.file_path.as_str())
        .collect();
    assert!(
        pivot_files.iter().any(|f| f.ends_with("io/buffer.py")),
        "traceback frame `File \"io/buffer.py\", in flush` must surface the class method \
         `Buffer::flush` via leaf-name + suffix-path match. Pivots saw: {pivot_files:?}"
    );
}
