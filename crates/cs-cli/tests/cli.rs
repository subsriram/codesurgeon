//! Integration tests for the `codesurgeon` CLI binary.
//!
//! Each test spawns the real binary against a throwaway tempdir workspace.
//! Tests verify exit codes, stdout content, and error handling — not the
//! internal engine state.
//!
//! Run:  cargo test -p cs-cli --test cli

use std::process::{Command, Output};
use tempfile::TempDir;

const BIN: &str = env!("CARGO_BIN_EXE_codesurgeon");

// ── Helpers ───────────────────────────────────────────────────────────────────

fn run(dir: &TempDir, args: &[&str]) -> Output {
    Command::new(BIN)
        .env("CS_WORKSPACE", dir.path())
        .env("CS_LOG", "error")
        .args(args)
        .output()
        .expect("failed to spawn codesurgeon")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

// ── Argument validation ────────────────────────────────────────────────────────

/// Running with no subcommand must exit non-zero and print usage.
#[test]
fn no_subcommand_exits_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &[]);
    assert!(
        !out.status.success(),
        "expected non-zero exit with no subcommand"
    );
}

/// An unknown subcommand must exit non-zero.
#[test]
fn unknown_subcommand_exits_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &["definitely-not-a-command"]);
    assert!(
        !out.status.success(),
        "expected non-zero exit for unknown subcommand"
    );
}

/// `search` requires a query argument; omitting it must exit non-zero.
#[test]
fn search_without_query_exits_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &["search"]);
    assert!(
        !out.status.success(),
        "expected non-zero exit when query is missing"
    );
}

/// `--budget` must be a positive integer; a negative string should be rejected
/// by clap before the engine ever runs.
#[test]
fn search_with_negative_budget_exits_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    // clap parses budget as u32, so "-1" fails argument parsing.
    let out = run(&dir, &["search", "foo", "--budget", "-1"]);
    assert!(
        !out.status.success(),
        "expected non-zero exit for negative budget"
    );
}

// ── status ────────────────────────────────────────────────────────────────────

/// `status` on a fresh empty workspace must succeed and show 0 symbols.
#[test]
fn status_on_empty_workspace_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &["status"]);
    assert!(out.status.success(), "status failed: {}", stderr(&out));
    let text = stdout(&out);
    assert!(
        text.contains("Symbols"),
        "expected 'Symbols' in status output: {text}"
    );
}

// ── index ─────────────────────────────────────────────────────────────────────

/// `index` on a workspace with one Python file must succeed and report > 0 symbols.
#[test]
fn index_workspace_with_python_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("main.py"),
        "def hello():\n    return 'world'\n",
    )
    .unwrap();

    let out = run(&dir, &["index"]);
    assert!(out.status.success(), "index failed: {}", stderr(&out));

    let text = stdout(&out);
    assert!(
        text.contains("Done"),
        "expected 'Done' in index output: {text}"
    );
    // Should report at least 1 symbol.
    assert!(
        text.contains("symbols"),
        "expected symbol count in output: {text}"
    );
}

/// Indexing an empty workspace must succeed (0 symbols is valid).
#[test]
fn index_empty_workspace_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &["index"]);
    assert!(out.status.success(), "index failed: {}", stderr(&out));
}

/// `index --force` must succeed and re-parse all files.
#[test]
fn index_force_reparses_all_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lib.py"), "def hello():\n    return 1\n").unwrap();

    // First index.
    let idx1 = run(&dir, &["index"]);
    assert!(
        idx1.status.success(),
        "first index failed: {}",
        stderr(&idx1)
    );

    // Second index without --force should skip unchanged files.
    let idx2 = run(&dir, &["index"]);
    assert!(idx2.status.success());
    let err2 = stderr(&idx2);
    assert!(
        err2.contains("skipped") || err2.contains("unchanged"),
        "expected incremental skip message: {err2}"
    );

    // Third index with --force should re-parse everything.
    let idx3 = run(&dir, &["index", "--force"]);
    assert!(
        idx3.status.success(),
        "force index failed: {}",
        stderr(&idx3)
    );
    let out3 = stdout(&idx3);
    assert!(
        out3.contains("(force)"),
        "expected '(force)' in output: {out3}"
    );
}

// ── search ────────────────────────────────────────────────────────────────────

/// `search` on an empty workspace must exit zero (no results is not an error).
#[test]
fn search_on_empty_workspace_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &["search", "anything"]);
    assert!(
        out.status.success(),
        "search failed on empty workspace: {}",
        stderr(&out)
    );
}

/// After indexing, `search` for a known symbol name must produce output that
/// includes the symbol.
#[test]
fn search_finds_indexed_symbol() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        "pub fn compute_result(x: i32) -> i32 { x * 2 }\n",
    )
    .unwrap();

    // Index first.
    let idx = run(&dir, &["index"]);
    assert!(idx.status.success(), "index failed: {}", stderr(&idx));

    // Then search.
    let out = run(&dir, &["search", "compute_result"]);
    assert!(out.status.success(), "search failed: {}", stderr(&out));
    let text = stdout(&out);
    assert!(
        text.contains("compute_result"),
        "expected 'compute_result' in search output: {text}"
    );
}

// ── observe / memory ──────────────────────────────────────────────────────────

/// `observe` must save and `memory` must return the observation.
#[test]
fn observe_then_memory_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let sentinel = "test-observation-sentinel-xyz";

    let obs = run(&dir, &["observe", sentinel]);
    assert!(obs.status.success(), "observe failed: {}", stderr(&obs));

    let mem = run(&dir, &["memory"]);
    assert!(mem.status.success(), "memory failed: {}", stderr(&mem));
    assert!(
        stdout(&mem).contains(sentinel),
        "observation not found in memory output: {}",
        stdout(&mem)
    );
}

/// `memory` output must include a UUID id field for each observation.
#[test]
fn memory_output_includes_id() {
    let dir = tempfile::tempdir().unwrap();

    let obs = run(&dir, &["observe", "check-id-present"]);
    assert!(obs.status.success(), "observe failed: {}", stderr(&obs));

    let mem = run(&dir, &["memory"]);
    assert!(mem.status.success(), "memory failed: {}", stderr(&mem));
    let out = stdout(&mem);
    // Output format: [...] (id: <uuid>): <content>
    assert!(
        out.contains("(id: "),
        "memory output missing id field: {out}"
    );
}

/// `memory --delete <id>` must remove the observation so it no longer appears.
#[test]
fn memory_delete_removes_observation() {
    let dir = tempfile::tempdir().unwrap();
    let sentinel = "delete-me-sentinel";

    let obs = run(&dir, &["observe", sentinel]);
    assert!(obs.status.success(), "observe failed: {}", stderr(&obs));

    let mem = run(&dir, &["memory"]);
    let out = stdout(&mem);
    assert!(
        out.contains(sentinel),
        "observation not found before delete"
    );

    // Extract the id from the output line containing the sentinel
    let id = out
        .lines()
        .find(|l| l.contains(sentinel))
        .and_then(|l| l.split("(id: ").nth(1))
        .and_then(|s| s.split(')').next())
        .expect("could not parse id from memory output")
        .to_string();

    let del = run(&dir, &["memory", "--delete", &id]);
    assert!(del.status.success(), "delete failed: {}", stderr(&del));
    assert!(
        stdout(&del).contains("Deleted"),
        "unexpected delete output: {}",
        stdout(&del)
    );

    let mem2 = run(&dir, &["memory"]);
    assert!(
        !stdout(&mem2).contains(sentinel),
        "observation still present after delete: {}",
        stdout(&mem2)
    );
}

/// `memory --delete` with an unknown id must exit non-zero.
#[test]
fn memory_delete_nonexistent_exits_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let del = run(
        &dir,
        &["memory", "--delete", "00000000-0000-0000-0000-000000000000"],
    );
    assert!(
        !del.status.success(),
        "expected non-zero exit for unknown id"
    );
}

// ── submit-lsp-edges ──────────────────────────────────────────────────────────

/// Valid JSON LSP edges piped via stdin must exit zero and report accepted/skipped counts.
#[test]
fn submit_lsp_edges_stdin_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let edges = r#"[{"from_fqn":"src/lib.rs::foo","to_fqn":"src/lib.rs::bar","kind":"calls","resolved_type":null}]"#;

    let out = Command::new(BIN)
        .env("CS_WORKSPACE", dir.path())
        .env("CS_LOG", "error")
        .args(["submit-lsp-edges"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(edges.as_bytes())
                .unwrap();
            child.wait_with_output()
        })
        .expect("failed to spawn codesurgeon");

    assert!(
        out.status.success(),
        "submit-lsp-edges failed: {}",
        stderr(&out)
    );
    let text = stdout(&out);
    // Unknown symbols are skipped, but the command itself must succeed.
    assert!(
        text.contains("accepted") && text.contains("skipped"),
        "unexpected output: {text}"
    );
}

/// `submit-lsp-edges` with a valid JSON file must exit zero.
#[test]
fn submit_lsp_edges_file_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let edges_path = dir.path().join("edges.json");
    std::fs::write(
        &edges_path,
        r#"[{"from_fqn":"a::b","to_fqn":"c::d","kind":"imports","resolved_type":null}]"#,
    )
    .unwrap();

    let out = run(&dir, &["submit-lsp-edges", edges_path.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "submit-lsp-edges failed: {}",
        stderr(&out)
    );
    let text = stdout(&out);
    assert!(
        text.contains("accepted") && text.contains("skipped"),
        "unexpected output: {text}"
    );
}

/// `submit-lsp-edges` with malformed JSON must exit non-zero.
#[test]
fn submit_lsp_edges_bad_json_exits_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let edges_path = dir.path().join("bad.json");
    std::fs::write(&edges_path, "this is not json").unwrap();

    let out = run(&dir, &["submit-lsp-edges", edges_path.to_str().unwrap()]);
    assert!(
        !out.status.success(),
        "expected non-zero exit for malformed JSON"
    );
}

/// `submit-lsp-edges` with a nonexistent file must exit non-zero.
#[test]
fn submit_lsp_edges_missing_file_exits_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &["submit-lsp-edges", "/tmp/does-not-exist-xyz.json"]);
    assert!(
        !out.status.success(),
        "expected non-zero exit for missing file"
    );
}

// ── context ──────────────────────────────────────────────────────────────────

/// `context` on an empty workspace must exit zero (no results is not an error).
#[test]
fn context_on_empty_workspace_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &["context", "find authentication logic"]);
    assert!(
        out.status.success(),
        "context failed on empty workspace: {}",
        stderr(&out)
    );
}

/// `context` with --budget flag must be accepted by clap.
#[test]
fn context_accepts_budget_flag() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &["context", "test query", "--budget", "2000"]);
    assert!(
        out.status.success(),
        "context with --budget failed: {}",
        stderr(&out)
    );
}

/// `context` with --language and --file-hint flags must be accepted.
#[test]
fn context_accepts_language_and_file_hint_flags() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(
        &dir,
        &[
            "context",
            "test query",
            "--language",
            "rust",
            "--file-hint",
            "src/lib",
        ],
    );
    assert!(
        out.status.success(),
        "context with flags failed: {}",
        stderr(&out)
    );
}

/// `context` without a task argument must exit non-zero.
#[test]
fn context_without_task_exits_nonzero() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &["context"]);
    assert!(
        !out.status.success(),
        "expected non-zero exit when task is missing"
    );
}

// ── config ───────────────────────────────────────────────────────────────────

/// `config` on workspace without config file must exit zero and show defaults message.
#[test]
fn config_shows_defaults_when_missing() {
    let dir = tempfile::tempdir().unwrap();
    let out = run(&dir, &["config"]);
    assert!(out.status.success(), "config failed: {}", stderr(&out));
    let text = stdout(&out);
    assert!(
        text.contains("defaults"),
        "expected 'defaults' message in config output: {text}"
    );
}

/// `config` with a config file present must display its contents.
#[test]
fn config_displays_config_contents() {
    let dir = tempfile::tempdir().unwrap();
    let cs_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&cs_dir).unwrap();
    std::fs::write(cs_dir.join("config.toml"), "[indexing]\nts_types = true\n").unwrap();

    let out = run(&dir, &["config"]);
    assert!(out.status.success(), "config failed: {}", stderr(&out));
    let text = stdout(&out);
    assert!(
        text.contains("ts_types = true"),
        "expected config file contents in output: {text}"
    );
}

/// `config` shows effective skeleton_detail and token_budget from [context] section.
#[test]
fn config_shows_context_settings() {
    let dir = tempfile::tempdir().unwrap();
    let cs_dir = dir.path().join(".codesurgeon");
    std::fs::create_dir_all(&cs_dir).unwrap();
    std::fs::write(
        cs_dir.join("config.toml"),
        "[context]\nmax_tokens = 8000\nskeleton_detail = \"detailed\"\n",
    )
    .unwrap();

    let out = run(&dir, &["config"]);
    assert!(out.status.success(), "config failed: {}", stderr(&out));
    let text = stdout(&out);
    assert!(
        text.contains("Detailed"),
        "expected 'Detailed' skeleton_detail: {text}"
    );
    assert!(text.contains("8000"), "expected token_budget 8000: {text}");
}

// ── indexing progress ────────────────────────────────────────────────────────

/// `index` must produce progress output on stderr with [codesurgeon] prefix.
#[test]
fn index_progress_output_to_stderr() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("app.py"), "def greet():\n    return 'hi'\n").unwrap();

    let out = run(&dir, &["index"]);
    assert!(out.status.success(), "index failed: {}", stderr(&out));
    let err = stderr(&out);
    assert!(
        err.contains("[codesurgeon]"),
        "expected [codesurgeon] progress prefix in stderr: {err}"
    );
    assert!(
        err.contains("done"),
        "expected 'done' progress line in stderr: {err}"
    );
}
