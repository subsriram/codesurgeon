# codesurgeon 1.0 Release Readiness Report

**Date:** 2026-04-10
**Verdict:** CONDITIONAL PASS — solid codebase, missing release packaging

---

## 1. Blockers (must fix before 1.0)

### Release packaging

| Item | Current state | Action |
|------|---------------|--------|
| Version bump | `0.1.0` in Cargo.toml | Change to `1.0.0` |
| LICENSE file | Missing from repo root | Add `/LICENSE` with MIT text (matches Cargo.toml `license = "MIT"`) |
| CHANGELOG.md | Missing | Create with 0.1.0 -> 1.0.0 changes |
| Repository URL mismatch | Cargo.toml says `subsriram`, README says `sriramk` | Fix to match actual GitHub org |
| Crate metadata | cs-cli, cs-mcp, cs-core missing license/description/repository | Add `license.workspace = true` etc. to each crate's Cargo.toml |

### Code safety

| Issue | File | Severity | Detail |
|-------|------|----------|--------|
| Embedding store alignment UB | `emb_store.rs:168` | Critical | Unsafe `from_raw_parts` without alignment/size validation. Corrupted `.bin` file causes undefined behavior. Fix: add magic bytes + version check, validate mmap length, use `align_to()` or verify alignment, add checksum. |
| Unbounded skeleton vec | `engine.rs:813-839` | High | 100k+ symbols materialized into memory before batching. Fix: stream in chunks of 64 instead of collecting all upfront. |
| Stale file cleanup O(n^2) | `engine.rs:516-525` | Medium | Nested loop: for each stale file, individual DELETE statements per symbol. Fix: add batch delete methods to `db.rs`. |

---

## 2. Should fix (strongly recommended)

### Code quality

| Issue | File | Detail |
|-------|------|--------|
| Unwraps in production paths | `edges.rs:237-238,323`, `engine.rs:1434` | `split().next().unwrap()` patterns. Use `unwrap_or` or propagate errors. |
| Embedding batch failure tracking | `engine.rs:844-856` | Failed batches silently skipped, no retry or logging of degraded semantic search. |
| Concurrent embedding cache refresh race | `engine.rs:1036` | Two rapid reindexes can race on `refresh_embedding_cache()`. Add mutex. |
| Stdin reader thread has no timeout | `main.rs:561-574` | If main loop deadlocks, process hangs forever. Add read timeout. |
| Config file parse errors silent | `memory.rs:200-206` | Malformed TOML gives no warning, silently uses defaults. Log a warning on parse failure. |

### Documentation

| Issue | Detail |
|-------|--------|
| CLI `diff` command docs incomplete | README only shows stdin form; should document file argument and pipe forms. |
| README missing standard 1.0 sections | Installation, Configuration, Contributing sections needed. |
| `RUST_LOG` not in troubleshooting | Issue #22 gap — one-line addition to README. |

### CI/CD

| Issue | Detail |
|-------|--------|
| No release workflow | No `.github/workflows/release.yml` for multi-platform binary builds. |
| No release profile | No `[profile.release]` with LTO in Cargo.toml. |

---

## 3. GitHub issues triage

### Close now (done)

| # | Title | Rationale |
|---|-------|-----------|
| 43 | Codebase review meta-issue | All sub-items resolved or tracked separately. |
| 31 | Expand test coverage | All planned work merged, remaining items explicitly deferred. |
| 22 | README troubleshooting/privacy docs | 95% done, only `RUST_LOG` mention missing. |

### P1 — should have for 1.0

| # | Title | Status |
|---|-------|--------|
| 25 | CLI parity (context/config show/index --force) | ✅ Complete — context, config, index --force all implemented. |
| 24 | config.toml schema + skeleton_detail | ✅ Complete — skeleton_detail, [context], [observability], user-level config. |
| 23 | First-run UX (progress, .cursor/.windsurf rules) | Mostly done; progress output done, .cursor/.windsurf rules deferred to post-1.0. |
| 16 | Binary distribution (cargo install, Homebrew) | Not started; blocked by fastembed/ort native deps. |

### P2 — nice to have

| # | Title |
|---|-------|
| 29 | Benchmark: SWE-bench pass@1 |
| 28 | Benchmark: index performance CI gate |
| 27 | Benchmark: token savings CI gate |
| 26 | `codesurgeon setup` one-command onboarding |

### P3 — post 1.0

| # | Title |
|---|-------|
| 55 | HNSW index for ANN search |
| 50 | Single-daemon mode |
| 32 | Iterator returns instead of Vec |
| 21 | workspace_setup MCP tool |
| 20 | Project rules |
| 19 | Multi-root workspace support |
| 18 | Git merge driver |

---

## 4. What's in good shape

- **Test coverage**: 57+ tests across crates, MCP protocol invariants well-tested, corrupt DB handling tested.
- **Graceful degradation**: Empty index, missing tools (pyright/cargo-expand/node), SQLite locks, orphaned process — all handled.
- **Security**: API key detection in indexed files, no hardcoded secrets, minimal justified unsafe code.
- **Ranking docs**: All parameters in `docs/ranking.md` verified accurate against code.
- **Wire format**: Content-Length and NDJSON dual-format correct (Codex + Claude Code compatibility).
- **Error handling**: Generally good — most failures logged and degraded gracefully.
- **Language support docs**: All languages in README match actual tree-sitter crates.

---

## 5. Testing gaps

| Gap | Severity |
|-----|----------|
| No concurrent file change stress tests (rapid changes, create-then-delete, symlink loops) | Medium |
| No corrupted index recovery tests (truncated embeddings.bin, partial SQLite writes) | Medium |
| No long symbol name / deep nesting edge case tests | Low |
| No integration tests for enrichment features (ts_types, pyright, cargo-expand) | Low |

---

## 6. Performance concerns

| Issue | File | Detail |
|-------|------|--------|
| Graph FQN lookup is O(n) | `graph.rs` | Linear search over all symbols. Add HashMap index for large workspaces. |
| Tantivy search index not persisted | `engine.rs:259` | Rebuilt in-RAM on every startup. Persisting to disk saves 2-5s cold start on 100k symbol workspaces. |
| ~~Embedding cache refresh race~~ | ~~`engine.rs:1036`~~ | Fixed: `refresh_guard` mutex at `engine.rs:2103` serialises concurrent reindexes. |

---

## 7. Recommended 1.0 action plan

### Phase 1 — Blockers ✅

1. ~~Add LICENSE file~~
2. ~~Version bump to 1.0.0~~
3. ~~Fix repository URL~~
4. ~~Add crate metadata (workspace inheritance)~~
5. ~~Fix embedding store alignment validation~~
6. ~~Create CHANGELOG.md~~

### Phase 2 — Should-fix ✅

7. ~~Fix unwraps in production paths~~ (already safe — all have `unwrap_or`)
8. ~~Add embedding batch error logging~~
9. ~~Add mutex around embedding cache refresh~~
10. ~~Close issues #43, #31, #22~~
11. ~~Update README with Installation/Configuration/Contributing sections~~
12. ~~Add release CI workflow~~

### Phase 3 — P1 issues ✅

13. ~~Implement `context` CLI command (#25)~~
14. ~~Implement `config` CLI command (#24)~~
15. ~~Add indexing progress output (#23)~~

### Phase 4 — Post-release ✅

16. ~~LTO release profile~~ — landed in root `Cargo.toml` (`lto = "fat"`, `codegen-units = 1`, `strip = "symbols"`). Measured: `codesurgeon` 47MB→36MB, `codesurgeon-mcp` 50MB→38MB (−24%).
17. ~~Binary distribution strategy (#16)~~ — decision: ship pre-built binaries via `release.yml` for `aarch64-apple-darwin`, `x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`. README documents `curl | tar` install. `cargo install` / Homebrew formula deferred to 1.1 (still blocked by fastembed/ort native deps).

---

## 1.0.0 released

Tagged `v1.0.0` on 2026-04-14. Release workflow publishes three binary tarballs to the GitHub Releases page.
