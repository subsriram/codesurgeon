# codesurgeon enrichment passes

> **Keep this doc up to date whenever an enrichment pass is added, removed, or its
> gates / incremental strategy change.**
> Enrichment passes live in `crates/cs-core/src/{macro_expand,rustdoc_enrich,pyright_enrich}.rs`
> and are orchestrated in `crates/cs-core/src/engine.rs::index_workspace_inner`.

---

## Overview

After the base tree-sitter index is built, optional enrichment passes merge
additional type information into existing symbols.  Each pass:

- is **opt-in** via `[indexing]` in `.codesurgeon/config.toml`
- **skips gracefully** when its external tool is absent
- is **incremental** — gated on a content/stat hash so re-runs are cheap

```
index_workspace_inner
  ├── tree-sitter parse (all languages)          base symbol index
  ├── rust_expand_macros   cargo-expand pass      Rust proc-macro symbols
  ├── rust_rustdoc_types   rustdoc JSON pass       Rust resolved types + traits
  ├── python_pyright       pyright pass            Python return types + bases
  └── stub file indexing   .d.ts / .pyi / .swiftinterface
```

All enrichment happens **before** stub indexing and embedding so that enriched
`resolved_type` values are present in the DB before any downstream read.

---

## Pass 1 — Rust macro expansion (`rust_expand_macros`)

**Config:** `[indexing] rust_expand_macros = true`
**Tool:** `cargo-expand` (`cargo install cargo-expand`)
**Module:** `crates/cs-core/src/macro_expand.rs`

### What it does

Identifies Rust source files that contain proc-macro or derive invocations
(`#[derive(`, `#[proc_macro`, etc.) and runs `cargo expand <module>` on each.
The expanded source is re-parsed with tree-sitter and the resulting symbols are
added to the index with `source = "macro_expanded"`.

Symbols already present in the base pass (matched by FQN) are excluded to
avoid duplicates.

### Gates

| # | Condition | Behaviour |
|---|-----------|-----------|
| 1 | `Cargo.toml` absent | return empty vec |
| 2 | `cargo-expand` not installed | return empty vec (logged at INFO) |

### Incremental strategy

No per-file incremental skip — `cargo expand` is run for every file that
contains a macro invocation on each full re-index.  The cost is bounded by
the number of macro-heavy files in the project.

### Symbol fields set

| Field | Value |
|-------|-------|
| `source` | `"macro_expanded"` |
| `resolved_type` | not set |

---

## Pass 2 — Rust rustdoc resolved types (`rust_rustdoc_types`)

**Config:** `[indexing] rust_rustdoc_types = true`
**Tool:** nightly Rust (`rustup toolchain install nightly`)
**Module:** `crates/cs-core/src/rustdoc_enrich.rs`

### What it does

Runs `cargo +nightly doc --output-format json --no-deps` and parses the
produced JSON file (one per crate under `target/doc/<crate>.json`).

For each Rust symbol in the index it tries to find a matching rustdoc entry
(by FQN suffix) and merges:

- **Functions/methods** — resolved return type (e.g. `Option<String>`, `Vec<u8>`)
- **Structs/enums/unions** — comma-separated directly-implemented trait names
  (e.g. `"Debug, Serialize, Clone"`)

### Gates

| # | Condition | Behaviour |
|---|-----------|-----------|
| 1 | `Cargo.toml` absent | return 0 |
| 2 | nightly Rust not installed | return 0 (logged at INFO) |
| 3 | `Cargo.lock` hash unchanged | return 0 (incremental cache hit, logged at DEBUG) |
| 4 | `cargo doc` fails | return 0 (logged at WARN) |

### Incremental strategy

A blake3 hash of `Cargo.lock` is stored in the DB under the key `__rustdoc__`.
The pass is skipped when the stored hash matches the current one.  This means
the pass re-runs whenever a dependency is added, removed, or updated — exactly
when resolved types might change.

### Symbol fields set

| Field | Value |
|-------|-------|
| `source` | `"rustdoc"` (only if `source` was previously `None`) |
| `resolved_type` | resolved return type or trait list |

---

## Pass 3 — Python pyright type enrichment (`python_pyright`)

**Config:** `[indexing] python_pyright = true`
**Tool:** pyright (`npm install -g pyright`)
**Module:** `crates/cs-core/src/pyright_enrich.rs`

### What it does

Merges Python type annotations into existing Python symbols using two sources:

1. **Explicit annotations** already captured in the `signature` field by tree-sitter
   (e.g. `def f(x: int) -> str:` → `resolved_type = "str"`).  These are extracted
   without invoking pyright at all.

2. **Inferred types** from `information`-severity pyright diagnostics matching the
   pattern `Type of "name" is "T"`.  Covers cases where the return type is inferred
   rather than explicitly annotated.

For class symbols, base-class names from the signature are stored as `resolved_type`
(e.g. `class MyView(APIView, LogMixin):` → `resolved_type = "APIView, LogMixin"`).

### Gates

| # | Condition | Behaviour |
|---|-----------|-----------|
| 1 | No Python symbols in index | return 0 immediately |
| 2 | `pyright` not on PATH | return 0 (logged at INFO with install hint) |
| 3 | Python file stats hash unchanged | return 0 (incremental cache hit, logged at DEBUG) |
| 4 | `pyright` exits with code ≥ 2 | return 0 (logged at WARN); exit codes 0 and 1 are both treated as success |

### Incremental strategy

A hash is computed over all `.py` file stats (path + size + mtime) in the
workspace — a cheap `stat`-only scan, no file reads.  The hash is stored in
the DB under the key `__pyright__`.  The pass is skipped when the stored hash
matches the current one.

Directories skipped during stat collection: `.`, `node_modules`, `__pycache__`,
`.venv`, `venv`, `site-packages`.

### Symbol fields set

| Field | Value |
|-------|-------|
| `source` | `"pyright"` |
| `resolved_type` | return type string (functions/methods) or base-class names (classes) |

### Symbols enriched

| `SymbolKind` | Source of `resolved_type` |
|---|---|
| `Function`, `AsyncFunction` | `-> T` in signature, then pyright diagnostic |
| `Method`, `AsyncMethod` | `-> T` in signature, then pyright diagnostic |
| `Class` | base-class names from signature |
| all others | not enriched |

---

## Adding a new enrichment pass

1. Create `crates/cs-core/src/<name>_enrich.rs` with a `run_<name>_enrichment(workspace_root, all_symbols, db)` entry point.
2. Add the config flag to `IndexingConfig` in `memory.rs` and wire it into `EngineConfig` in `engine.rs`.
3. Call the pass in `index_workspace_inner` after the existing enrichment block, before stub indexing.
4. Flush enriched symbols back to SQLite (see the rustdoc or pyright pass for the pattern).
5. Update this document.
