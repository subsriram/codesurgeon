//! Capture git SHA + build timestamp into compile-time env vars so both
//! `codesurgeon` and `codesurgeon-mcp` can report which build is running.
//!
//! Exposed as `cs_core::GIT_SHA` / `cs_core::BUILD_TIME` / `cs_core::VERSION`
//! (see `src/lib.rs`). The version string is what `--version` prints.
//!
//! Falls back to `unknown` for the SHA when not in a git checkout (e.g.
//! crate published to crates.io). Marks `+dirty` when the working tree
//! has uncommitted changes — important for benchmark reproducibility,
//! since a `+dirty` build doesn't correspond to any merged commit.

use std::process::Command;

fn main() {
    // Re-run when HEAD moves (commit, branch switch). The .git path is
    // resolved relative to the workspace root; cs-core lives at
    // `crates/cs-core/` so we go up two levels.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");

    let sha = git_short_sha().unwrap_or_else(|| "unknown".to_string());
    let dirty = git_is_dirty().unwrap_or(false);
    let sha_full = if dirty { format!("{}+dirty", sha) } else { sha };

    let build_time = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

    println!("cargo:rustc-env=CS_GIT_SHA={}", sha_full);
    println!("cargo:rustc-env=CS_BUILD_TIME={}", build_time);
}

fn git_short_sha() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn git_is_dirty() -> Option<bool> {
    let out = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(!out.stdout.is_empty())
}
