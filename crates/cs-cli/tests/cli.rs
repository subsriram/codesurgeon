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
    assert!(
        out.status.success(),
        "status failed: {}",
        stderr(&out)
    );
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
    assert!(out.contains(sentinel), "observation not found before delete");

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
    let del = run(&dir, &["memory", "--delete", "00000000-0000-0000-0000-000000000000"]);
    assert!(
        !del.status.success(),
        "expected non-zero exit for unknown id"
    );
}
