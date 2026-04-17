# Design: Explicit Symbol-Name Anchors in the Ranking Pipeline

> **Status**: design / proposal
> **Target**: `crates/cs-core/src/engine.rs::build_context_capsule`
> **Related**: `docs/ranking.md`, SWE-bench benchmark report `benches/swebench/report_29c_interim.md`
> **Motivation**: SWE-bench #29c revealed that capsule ranking misses the target
> file in 5 out of 6 regression tasks even with semantic (embedding) retrieval
> enabled. The failure mode is always the same: the task explicitly names the
> target symbol, but the ranker treats it as bag-of-words and surfaces
> tangentially-related files instead.

---

## The problem, with evidence

### Case study 1 — `sphinx-doc/sphinx-9711` (prose-mentioned name)

Problem statement (first line):
> "`needs_extensions` checks versions using strings"

The task literally names the function `needs_extensions` in the title.
Running `run_pipeline` with this task:

| Ranking signal | Top pivot |
|---|---|
| BM25 only | `sphinx/domains/cfamily.py::cfamily` (matched "extensions" as English plural noun → C++ file-extension parser) |
| BM25 + embeddings | `utils/bump_version.py::bump_version` (matched "version comparison using strings") |
| **Ground truth** | **`sphinx/extension.py::needs_extensions`** |

Both rankings miss. The target function has a ~10-line body and sparse docstring — semantically under-resourced compared to `bump_version.py` which is a full release script dedicated to version manipulation. BM25 tokenises `needs_extensions` into `needs` + `extensions` and scores each independently, neither hitting the target file strongly.

Result: with-arm walltime 41.6s (embeddings on) vs 30.1s (embeddings off) vs ~16s baseline without codesurgeon. The capsule is net-negative.

### Case study 2 — `pydata/xarray-7229` (code-snippet API call)

Problem statement includes a reproducing example:
```python
import xarray as xr
ds = xr.tutorial.load_dataset("air_temperature")
xr.where(True, ds.air, ds.air, keep_attrs=True).time.attrs
```

The task calls `xr.where(...)`. The fix is in `xarray/core/computation.py` where `where()` is defined.

| Ranking signal | Top pivots |
|---|---|
| BM25 + embeddings | `pydap_.py::_fix_attributes`, `conventions.py::_update_bounds_attributes` (both matched "attributes" heavily) |
| **Ground truth** | **`xarray/core/computation.py::where`** |

The ranker latched onto the noun "attributes" (appearing 8+ times) and ignored that `where` is the specific function the user called. `computation.py` wasn't even in the top 5.

Result: with-arm walltime 528s vs without-arm 194s — **+172% walltime** regression.

### Case study 3 — `sympy/sympy-21612` (path-segment semantics)

Problem statement:
> "Latex parsing of fractions yields wrong expression"
>
> ```python
> from sympy.parsing.latex import parse_latex
> parse_latex("\\frac{\\frac{a^3+b}{c}}{\\frac{1}{c^2}}")
> ```

The task calls `parse_latex(...)` from `sympy.parsing.latex`. The fix is in `sympy/parsing/latex/_parse_latex_antlr.py`.

| Ranking signal | Top pivot |
|---|---|
| BM25 + embeddings | `sympy/printing/latex.py::latex` (LaTeX **output** printer) |
| **Ground truth** | **`sympy/parsing/latex/_parse_latex_antlr.py`** (LaTeX **input** parser) |

`parsing/` vs `printing/` are opposite-direction path segments sharing the same domain word. Pure BM25 + embeddings can't disambiguate. But the task literally imports from `sympy.parsing.latex`, and calls `parse_latex` — those are ground-truth anchors.

Result: 1133s vs 336s — **+237% walltime** regression.

---

## Proposed solution

Add a new retrieval source, **Explicit Anchors**, that extracts symbol names and module paths from the problem statement and boosts any indexed symbol that matches. This source runs in parallel to BM25, semantic, and graph retrieval, and feeds into the same RRF fusion.

### Two extraction modes (both fire on every query)

#### (a) Prose-mentioned symbol names

Tokenize the problem statement and cross-reference every identifier-shaped token against the symbol-name FTS index. Matches should be exact on the `name` field (not `fqn`, not `signature`).

```rust
// Identifier pattern: snake_case or camelCase, min 4 chars to avoid noise
// Avoid matching stop words like "with", "this", "that", "when", "where"
let re = Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b").unwrap();
let candidates: HashSet<String> = re.find_iter(query)
    .map(|m| m.as_str().to_string())
    .filter(|tok| !STOP_WORDS.contains(tok.as_str()))
    .collect();

// For each candidate, check if an indexed symbol has that exact name.
// Use a direct name index (separate from the full-text Tantivy index) to avoid
// BM25 scoring — we want exact match or nothing.
let mut anchors: Vec<(u64, f32)> = vec![];
for tok in &candidates {
    for symbol_id in db.symbols_by_exact_name(tok)? {
        anchors.push((symbol_id, 1.0));  // flat score — position in RRF is what matters
    }
}
```

For `sphinx-9711`, this extracts `needs_extensions` and directly looks up the symbol by name. A single hit → `sphinx/extension.py::needs_extensions` gets injected at rank 1 in the anchor list. RRF merge promotes it into the capsule top-3.

#### (b) Code-snippet API calls

Parse fenced code blocks and extract function/method calls using a light tokenizer — no full Python parser needed, just regex for `identifier.identifier(` and `ClassName(`.

```rust
fn extract_api_calls(query: &str) -> Vec<String> {
    // Find fenced code blocks (```lang ... ``` or indented 4 spaces)
    let code_blocks = extract_code_blocks(query);
    let mut calls = vec![];
    // Match things like `xr.where(`, `np.array(`, `MyClass(`,
    // also handle multi-level: `a.b.c(`
    let call_re = Regex::new(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(").unwrap();
    for block in code_blocks {
        for cap in call_re.captures_iter(&block) {
            let full = cap.get(1).unwrap().as_str();
            // Split dotted path and add each segment as an anchor candidate.
            // xr.where → ["xr.where", "where"]
            // urllib.request.urlopen → ["urllib.request.urlopen", "request.urlopen", "urlopen"]
            calls.push(full.to_string());
            if let Some(last) = full.rsplit('.').next() {
                if last != full {
                    calls.push(last.to_string());
                }
            }
        }
    }
    calls
}
```

For `xarray-7229`, this extracts `xr.where` → `["xr.where", "where"]`. Looking up `where` by exact name finds `xarray/core/computation.py::where` (among others). Rank 1 anchor match.

For `sympy-21612`, this extracts `parse_latex` → single anchor → `sympy/parsing/latex/_parse_latex_antlr.py::parse_latex` ranks 1. Path-segment disambiguation is a free side effect.

#### (c) Bonus — import statements

Also cheap to extract:
```python
from sympy.parsing.latex import parse_latex
import xarray as xr
```

The `from X.Y import Z` statement is extremely informative — it directly names both a module path AND a symbol. Penalise files whose path doesn't share any segment with the imported module path, and boost those that do.

```rust
// Extract "from a.b.c import foo, bar" and "import a.b.c as x"
// Match against file paths: prefer files under a/b/c/ or whose basename is c.py
```

---

## Integration point

The minimal change is one new candidate source added to the RRF merge in `build_context_capsule` (around `engine.rs:2167`):

```rust
// New source: explicit anchors from the query (exact symbol-name matches
// from prose and from code-snippet API calls).
let anchor_results = self.anchor_candidates(query, ANCHOR_CANDIDATES);

#[cfg(feature = "embeddings")]
let mut search_results = {
    let ann_results = self.ann_candidates(query, ANN_CANDIDATES);
    rrf_merge(&[
        &bm25_results,
        &graph_results,
        &ann_results,
        &anchor_results,  // ← new
    ], RRF_K)
};
```

### Key design decisions

1. **Exact match only, no fuzzy.** We already have BM25 for fuzzy. Anchors are meant to be unambiguous ground truth; if a token doesn't map to an exact symbol name, drop it.

2. **Flat scoring.** All anchor hits get score 1.0. RRF handles rank-based fusion — anchor hits ending up at positions 1..N in the anchor list is what matters, not their raw scores.

3. **Boost anchor contribution in RRF.** Optionally, give the anchor list a stronger k constant (say `k=30` vs the usual `k=60`), so anchor rank 1 contributes `1/31 = 0.032` to the fused score vs `1/61 = 0.016` for BM25 rank 1. This is a tuning knob; start without it and add if the benchmark demands.

4. **Stop words matter.** Without a stop list (`with`, `where`, `when`, `this`, `that`, `size`, `type`, `name`, `list`, `dict`, `len`, `str`, `int`, ...), every English sentence will produce dozens of false-positive matches. Curate a small list of English common words that are also common programming identifiers.

5. **Never let anchors dominate.** If a task contains no extractable identifiers (pure prose bug report), anchors return empty and the pipeline degrades to current behaviour. If anchors return 50 matches, cap the list at `ANCHOR_CANDIDATES = 20` before RRF merge so it doesn't drown out BM25.

6. **Respect `file_hint` even more strongly with anchors.** If the user already narrowed by file, intersect anchor matches with that file's symbols before RRF.

---

## Implementation sketch

### 1. New module: `crates/cs-core/src/anchors.rs`

```rust
//! Explicit symbol-name anchor extraction for ranking.
//!
//! Extracts identifiers from the task query that match exact symbol names in
//! the index. Three sources:
//!   1. Prose tokens (top-level words that look like identifiers)
//!   2. Function/method calls in fenced code blocks (`foo.bar(...)`)
//!   3. `from X.Y import Z` / `import X.Y as Z` statements
//!
//! All matches are flat-scored — ranking within the anchor list is
//! "extraction order" which roughly correlates with position in the query.
//! RRF fusion handles blending with BM25/ANN/graph.

use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// English stop words that are also common programming identifiers.
/// Used to filter prose tokens; code-snippet extraction ignores this list.
const STOP_WORDS: &[&str] = &[
    "with", "when", "where", "this", "that", "from", "into", "have", "been",
    "just", "like", "make", "many", "more", "most", "must", "only", "over",
    "such", "than", "then", "they", "were", "will", "into", "upon",
    // common type/collection names we don't want to anchor on
    "none", "true", "false", "int", "str", "dict", "list", "set", "tuple",
    "bool", "float", "bytes", "type", "kind", "name", "value", "values",
    "size", "length", "index", "data", "item", "items", "path", "file",
    "files", "line", "lines", "test", "tests", "error", "errors", "cause",
    "fail", "pass", "call", "calls", "version", "using", "should", "result",
];

fn identifier_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b").unwrap())
}

fn call_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Matches `a.b.c(`, `Foo(`, `self.bar(`
    RE.get_or_init(|| {
        Regex::new(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(").unwrap()
    })
}

fn import_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?m)^\s*(?:from\s+([\w.]+)\s+import\s+([\w, ]+)|import\s+([\w.]+)(?:\s+as\s+(\w+))?)").unwrap()
    })
}

fn code_block_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"```[\w]*\n([\s\S]*?)```").unwrap())
}

/// Extracted anchors, in order of discovery.
#[derive(Debug, Default)]
pub struct Anchors {
    /// Symbol names to try looking up exactly.
    pub symbol_names: Vec<String>,
    /// Module paths (from import statements).
    pub module_paths: Vec<String>,
}

pub fn extract(query: &str) -> Anchors {
    let mut out = Anchors::default();
    let mut seen: HashSet<String> = HashSet::new();

    // 1. Code-block API calls — highest priority
    for block_cap in code_block_re().captures_iter(query) {
        let block = &block_cap[1];

        // Import statements inside code
        for imp in import_re().captures_iter(block) {
            if let Some(m) = imp.get(1) { out.module_paths.push(m.as_str().to_string()); }
            if let Some(m) = imp.get(3) { out.module_paths.push(m.as_str().to_string()); }
            // Imported symbol names
            if let Some(names) = imp.get(2) {
                for n in names.as_str().split(',') {
                    let n = n.trim();
                    if !n.is_empty() && seen.insert(n.to_string()) {
                        out.symbol_names.push(n.to_string());
                    }
                }
            }
        }

        // Function/method calls
        for cap in call_re().captures_iter(block) {
            let full = cap[1].to_string();
            if seen.insert(full.clone()) {
                out.symbol_names.push(full.clone());
            }
            // Also add the last segment: `xr.where` → `where`
            if let Some(last) = full.rsplit('.').next() {
                if last.len() > 3 && seen.insert(last.to_string()) {
                    out.symbol_names.push(last.to_string());
                }
            }
        }
    }

    // 2. Prose identifiers — lower priority, filtered by stop words
    for m in identifier_re().find_iter(query) {
        let tok = m.as_str();
        let lower = tok.to_lowercase();
        if STOP_WORDS.contains(&lower.as_str()) { continue; }
        // Require either underscore or camelCase — filters out English words
        let has_snake = tok.contains('_');
        let has_camel = tok.chars().any(|c| c.is_uppercase()) &&
                        tok.chars().any(|c| c.is_lowercase());
        if !has_snake && !has_camel { continue; }
        if seen.insert(tok.to_string()) {
            out.symbol_names.push(tok.to_string());
        }
    }

    out
}
```

### 2. New DB method in `crates/cs-core/src/db.rs`

```rust
impl Db {
    /// Look up symbol IDs by exact name (not FQN). Used for anchor retrieval.
    /// Returns at most `limit` matches per name.
    pub fn symbols_by_exact_name(&self, name: &str, limit: usize) -> Result<Vec<u64>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT id FROM symbols WHERE name = ?1 LIMIT ?2"
        )?;
        let rows = stmt.query_map((name, limit as i64), |r| r.get::<_, i64>(0).map(|v| v as u64))?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(Into::into)
    }
}
```

Note: `symbols.name` is already indexed in SQLite (see `crates/cs-core/src/db.rs` schema).

### 3. New engine method in `crates/cs-core/src/engine.rs`

```rust
const ANCHOR_CANDIDATES: usize = 20;

impl CoreEngine {
    /// Returns anchor candidates — exact symbol-name hits from explicit
    /// identifiers in the query. Score is flat 1.0 per hit; order reflects
    /// extraction priority (code-snippet calls first, then prose tokens).
    fn anchor_candidates(&self, query: &str, limit: usize) -> Vec<(u64, f32)> {
        let anchors = crate::anchors::extract(query);
        let mut out: Vec<(u64, f32)> = Vec::with_capacity(limit);
        let db = self.db.lock();
        let mut seen: HashSet<u64> = HashSet::new();
        for name in &anchors.symbol_names {
            // For "xr.where" try "where" (last segment); for "needs_extensions" try as-is
            let lookup = name.rsplit('.').next().unwrap_or(name);
            if let Ok(ids) = db.symbols_by_exact_name(lookup, 5) {
                for id in ids {
                    if seen.insert(id) {
                        out.push((id, 1.0));
                        if out.len() >= limit { return out; }
                    }
                }
            }
        }
        out
    }
}
```

### 4. Integration in `build_context_capsule`

```rust
// In engine.rs:2160 (inside build_context_capsule, before the RRF merge):
let anchor_results = self.anchor_candidates(query, ANCHOR_CANDIDATES);

#[cfg(feature = "embeddings")]
let mut search_results = {
    let ann_results = self.ann_candidates(query, ANN_CANDIDATES);
    rrf_merge(
        &[&bm25_results, &graph_results, &ann_results, &anchor_results],
        RRF_K,
    )
};
#[cfg(not(feature = "embeddings"))]
let mut search_results = rrf_merge(
    &[&bm25_results, &graph_results, &anchor_results],
    RRF_K,
);
```

### 5. Add ranking constant in `crates/cs-core/src/ranking.rs`

```rust
/// Explicit anchor candidate pool size. Anchors are exact symbol-name matches
/// extracted from the query (either prose identifiers or code-snippet API calls).
/// Small because we want high precision, not recall.
pub(crate) const ANCHOR_CANDIDATES: usize = 20;
```

---

## Tests

### Unit tests in `crates/cs-core/src/anchors.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_prose_snake_case() {
        let a = extract("The `needs_extensions` check is handy for verifying versions");
        assert!(a.symbol_names.contains(&"needs_extensions".to_string()));
    }

    #[test]
    fn extract_code_block_api_call() {
        let q = "```python\nxr.where(True, ds.air, ds.air, keep_attrs=True)\n```";
        let a = extract(q);
        assert!(a.symbol_names.contains(&"xr.where".to_string()));
        assert!(a.symbol_names.contains(&"where".to_string()));
    }

    #[test]
    fn extract_import_statement() {
        let q = "```python\nfrom sympy.parsing.latex import parse_latex\nparse_latex('x')\n```";
        let a = extract(q);
        assert!(a.symbol_names.contains(&"parse_latex".to_string()));
        assert!(a.module_paths.contains(&"sympy.parsing.latex".to_string()));
    }

    #[test]
    fn stop_words_filtered() {
        let a = extract("This is a simple test with some common words");
        assert!(a.symbol_names.is_empty()); // nothing looks like a symbol
    }

    #[test]
    fn camelcase_accepted() {
        let a = extract("The BuildEnvironment class handles the case");
        assert!(a.symbol_names.contains(&"BuildEnvironment".to_string()));
    }

    #[test]
    fn plain_english_rejected() {
        let a = extract("The function should return an empty dict for fields");
        // "function", "should", "return", "empty", "fields" — all plain English
        // None should survive: "function" is in stop list (or rejected as no _ or camelCase)
        // "BuildEnvironment" would pass. "fields" is plain lowercase, gets rejected.
        assert!(!a.symbol_names.iter().any(|s| s == "function"));
        assert!(!a.symbol_names.iter().any(|s| s == "fields"));
    }
}
```

### Integration test: verify the three regression tasks now hit

Add a test that seeds a tiny in-memory corpus with symbols named `needs_extensions`, `where`, `parse_latex`, and the three regression-task queries, and asserts `anchor_candidates` surfaces the right symbol for each.

### Benchmark validation

After implementing:

```bash
# Just re-run the 3 regression case studies; walltime should drop back to <60s each
python3 benches/swebench/run.py \
  --instance-ids sphinx-doc__sphinx-9711,pydata__xarray-7229,sympy__sympy-21612 \
  --arms with --max-budget-usd 3.00 --timeout 300 --clean
```

Success criteria:
- `sphinx-9711`: capsule contains `sphinx/extension.py::needs_extensions` in top-5 pivots
- `xarray-7229`: capsule contains `xarray/core/computation.py::where` in top-5 pivots
- `sympy-21612`: capsule contains `sympy/parsing/latex/_parse_latex_antlr.py::parse_latex` in top-5 pivots
- Walltime for each: ≤ without-arm baseline (16s, 194s, 336s respectively)

---

## Rollout

1. **Feature-flag at engine level**: gate behind `EngineConfig::anchor_retrieval_enabled` (default `true`). Lets us disable for debugging or A/B testing.
2. **Log anchor hits at `debug` level**: `tracing::debug!("anchors: {} extracted, {} resolved", extracted, resolved);`. Helps diagnose when anchors fire vs not.
3. **Track in stats**: add `anchor_hits` column to `query_log` so we can measure how often anchors contributed after the fact.
4. **Update `docs/ranking.md`** with the new Stage 1 source and the parameter `ANCHOR_CANDIDATES`.

---

## Out of scope (file follow-ups separately)

- **Reverse-edge expansion from error types** (issue: walk callers/raisers of symbols in the current capsule). Addresses sphinx-9711's `VersionRequirementError → needs_extensions` case *even without anchors*. Complementary, not a substitute.
- **Path-segment scoring for antonym segments** (`parsing/` vs `printing/`). Anchors solve sympy-21612 directly, but path-segment scoring would catch the generalized case where no API call is quoted.
- **Short-body function floor** — a symbol whose body is < N tokens should get a bonus based on exact name match to a query token. Overlaps partly with anchors but useful when the user describes the bug without naming the function.

---

## File tree of changes

```
crates/cs-core/src/
├── anchors.rs          # NEW ~150 lines, pure function + tests
├── engine.rs           # ~20 lines added to build_context_capsule + anchor_candidates method
├── db.rs               # ~10 lines for symbols_by_exact_name
├── ranking.rs          # 3 lines — new constant
└── lib.rs              # pub mod anchors;

docs/
└── ranking.md          # update Stage 1 diagram + parameter table

crates/cs-core/tests/
└── ranking_anchors.rs  # NEW integration test seeded with the 3 regression cases
```

Total net change: ~250 lines of Rust + tests.

---

## Open questions for the implementing agent

1. Should we re-tokenize on the go's side or reuse Tantivy's tokenizer? Probably reuse for consistency with BM25 term matching.
2. Is the stop-word list language-aware? Current list is English-only. Revisit if we support non-English problem statements.
3. Should `ANCHOR_CANDIDATES` be per-intent (structural intent might not benefit)? Start uniform, tune after benchmark.
4. How to avoid double-counting when an anchor hit is *also* in BM25? RRF handles this correctly — agreement between sources amplifies the candidate, which is what we want.
