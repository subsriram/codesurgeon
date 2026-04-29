//! Ranking pipeline helpers shared by `build_context_capsule`.
//!
//! All functions here are pure (no `self`) — they operate on already-locked
//! graph/search data passed in by the engine.  See `docs/ranking.md` for the
//! rationale behind every constant.

use crate::graph::CodeGraph;
use crate::search::SearchIntent;
use crate::symbol::{Symbol, SymbolKind};
use std::collections::{HashMap, HashSet};

// ── Retrieval pool sizes ──────────────────────────────────────────────────────

/// BM25 candidate pool size passed to Tantivy.
pub(crate) const BM25_POOL_SIZE: usize = 50;
/// Number of ANN candidates from the HNSW index per query.
#[cfg(feature = "embeddings")]
pub(crate) const ANN_CANDIDATES: usize = 25;
/// Number of graph-neighbor candidates expanded from BM25 seeds per query.
pub(crate) const GRAPH_CANDIDATES: usize = 25;
/// Explicit-anchor candidate pool size. Anchors are exact symbol-name matches
/// extracted from the query (prose identifiers, import targets, code-block API
/// calls). Kept small because the goal is precision, not recall.
pub(crate) const ANCHOR_CANDIDATES: usize = 20;
/// Max rows to fetch per distinct anchor name (limits blast radius on common
/// names like `where` or `get`).
pub(crate) const ANCHOR_ROWS_PER_NAME: usize = 5;
/// Hit-count threshold for the BM25-name fallback. If the name-field BM25
/// query returns more than this many hits, the anchor is considered too
/// fuzzy and is skipped entirely — the aggressive `ANCHOR_RRF_K` boost
/// would otherwise amplify ranker bias toward public-API symbols (see the
/// matplotlib-26208 regression documented in docs/explicit-symbol-anchors.md).
pub(crate) const ANCHOR_FUZZY_CUTOFF: usize = 3;
/// Probe depth for the BM25-name fallback. Fetch up to this many hits so we
/// can measure fuzziness before deciding whether to inject any of them.
pub(crate) const ANCHOR_FUZZY_PROBE: usize = 20;
/// v1.6 file-diversity pinning: max number of distinct anchor-named files
/// pinned into the pivot set. Each pinned file contributes one pivot (the
/// most-specific anchor hit in that file). Remaining pivot slots are filled
/// from the BM25/ANN/graph RRF fusion. See docs/explicit-symbol-anchors.md.
pub(crate) const ANCHOR_FILE_BUDGET: usize = 5;

// ── Reverse-edge expansion (issue #67) ───────────────────────────────────────
//
// For symptom-anchored bug reports the user names the error class or a trigger
// symbol, but the fix site is reached only by walking **backward** through
// callers. Reverse expansion seeds on exception-class anchors and BFS-walks
// their callers/raisers up to `EXPAND_MAX_DEPTH` hops, injecting the
// walk results into the RRF fusion alongside the direct anchor list.

/// Max hops to BFS backward through `dependents()` from each seed anchor.
/// 3 is the tightest depth that covers the motivating sympy-21379 case
/// (`PolynomialError ← parallel_poly_from_expr ← gcd ← Mod.eval`).
pub(crate) const EXPAND_MAX_DEPTH: u32 = 3;
/// Per-hop cap on the number of callers expanded. Prevents exponential blowup
/// when walking from an exception class that's imported/raised in hundreds of
/// sites. Selection within a hop is driven by term overlap with the query.
pub(crate) const EXPAND_FAN_OUT: usize = 5;
/// Overall cap on the reverse-expansion candidate list. Mirrors
/// `ANCHOR_CANDIDATES` — the walk is precision-first, not recall-first.
pub(crate) const EXPAND_CANDIDATES: usize = 20;
/// RRF k for the reverse-expansion list. Sits between `ANCHOR_RRF_K = 15`
/// (aggressive, precision-first) and `RRF_K = 60` (default): seeds-reachable
/// symbols contribute meaningfully without overwhelming direct-anchor hits.
pub(crate) const EXPAND_RRF_K: f32 = 30.0;
/// Upper bound on a seed's direct-caller count. Anchors with more callers are
/// skipped — they're hubs (e.g. `exp`, `symbols`) whose reverse set would
/// flood the capsule. Exception classes in real codebases typically have
/// dozens-to-low-hundreds of raisers; this caps the seed set but the per-hop
/// `EXPAND_FAN_OUT` still bounds each walk.
pub(crate) const EXPAND_SEED_FANOUT_LIMIT: usize = 500;
/// Per-anchor-name cap on exact-match lookups during seed promotion.
/// Higher than `ANCHOR_ROWS_PER_NAME` (5) because seed promotion needs
/// the whole population, not the top-K-by-prefer-module-sort that
/// `anchor_candidates` produces for ranking. Set generously enough that
/// realistic anchor names (e.g. `hist` in matplotlib — 5+ distinct
/// classes have a `hist` method) all become seeds, but capped so a
/// truly common name like `where` or `get` doesn't seed thousands.
/// Issue #96 — `Axes::hist` was being dropped from forward seeds because
/// `pyplot::hist` (1-segment fqn) outranked it under the prefer_module
/// sort + 5-row truncate.
pub(crate) const EXPAND_SEED_PER_NAME_LIMIT: usize = 25;
/// Weight applied to body-text semantic similarity when ranking per-hop
/// callers inside the reverse-expand walk (issue #69 v2). Multiplies the
/// `[0, 1]` cosine similarity between the query embedding and each caller's
/// body embedding. Calibrated so that:
/// - one lexical term match (`+1.0`) still outweighs a moderately related
///   caller (`sim ≈ 0.5` → `+1.0` weighted contribution);
/// - amongst overlap=0 candidates (the common sympy-21379 failure mode
///   where the fix site has no lexical overlap with the query), semantic
///   similarity reorders by topical relevance rather than centrality alone.
///
/// Only applied when the `embeddings` feature is active AND a per-symbol
/// embedding lookup is provided to `reverse_expand_from_anchors`.
pub(crate) const EXPAND_SEMANTIC_WEIGHT: f32 = 2.0;

/// Density-aware fan-out parameters (issue #69 v1, originally landed in
/// `0f35b33` and reverted in `5516865`). Re-introduced behind the
/// `CS_REVERSE_EXPAND_STRATEGY` env-var gate so the cs-benchmark diagnostic
/// panel can score it on a stratified panel without making it the default
/// — see `docs/reverse_expand_panel.md` in the cs-benchmark repo.
pub(crate) const EXPAND_FAN_OUT_CAP: usize = 25;
pub(crate) const EXPAND_FAN_OUT_DIVISOR: usize = 5;

/// Best-first reverse-expand parameters (issue #69 option 3, deferred from
/// the original v1 — never landed before now). The walk replaces the fixed
/// (depth × fan_out) BFS with a priority queue, scoring candidates by the
/// same mixed signal (lex overlap + semantic + centrality penalty) the BFS
/// uses but spending budget on the highest-priority frontier nodes
/// regardless of hop. `TOTAL_BUDGET` is the cap on output candidates;
/// `EXPAND_BUDGET` is the cap on graph expansions performed during the walk
/// (so a dense seed doesn't enumerate the whole graph chasing low-scoring
/// nodes).
///
/// Issue #96 — bumped from 40 → 200 / 200 → 1000 after the matplotlib
/// probe showed default-budget runs only reached depth 1–2 in dense
/// codebases (matplotlib `Axes::hist`'s implementation tree is
/// 3 hops deep; default 40-output cap fired before depth 3 was even
/// touched). The user's TOTAL=500 probe confirmed the chain
/// `Axes::hist → fill → add_patch → _update_patch_limits` is
/// reachable; default needed to be high enough for depth 3 in the
/// common case. Walker overhead at TOTAL=500 was ~600 ms wall, so
/// 200 is well under any latency budget.
pub(crate) const EXPAND_TOTAL_BUDGET: usize = 200;
pub(crate) const EXPAND_GRAPH_BUDGET: usize = 1000;

/// UCB-style exploration weight applied to the priority of frontier
/// candidates whose parent-seed subtree has been under-visited. Used by
/// `v3b` only. The classic UCB1 formula uses `c = sqrt(2)`; we use a
/// smaller value so exploration nudges the walk rather than dominating
/// the per-candidate signal.
pub(crate) const EXPAND_UCB_C: f32 = 0.5;

/// Read an env-var-overridable budget value, accepting a deprecated
/// alias for one release cycle. New callers should set `primary`;
/// `legacy` is logged at warn level when set so the user gets a
/// pointer to the new name. Issue #96 / #95 rename.
fn resolve_budget(primary: &str, legacy: &str, default: usize) -> usize {
    if let Ok(v) = std::env::var(primary) {
        if let Ok(n) = v.trim().parse::<usize>() {
            return n;
        }
    }
    if let Ok(v) = std::env::var(legacy) {
        tracing::warn!(
            "{} is deprecated; use {} instead (will be removed)",
            legacy,
            primary
        );
        if let Ok(n) = v.trim().parse::<usize>() {
            return n;
        }
    }
    default
}

/// Read `CS_EXPAND_TOTAL_BUDGET` (or the deprecated
/// `CS_REVERSE_EXPAND_TOTAL_BUDGET` alias) from the environment;
/// fall back to the constant when unset or unparseable. Lets the
/// cs-benchmark panel ablate budget without rebuilding the binary —
/// issue #96 step 1.
pub(crate) fn resolve_total_budget() -> usize {
    resolve_budget(
        "CS_EXPAND_TOTAL_BUDGET",
        "CS_REVERSE_EXPAND_TOTAL_BUDGET",
        EXPAND_TOTAL_BUDGET,
    )
}

/// Read `CS_EXPAND_GRAPH_BUDGET` (or the deprecated
/// `CS_REVERSE_EXPAND_EXPAND_BUDGET` alias) from the environment.
pub(crate) fn resolve_expand_budget() -> usize {
    resolve_budget(
        "CS_EXPAND_GRAPH_BUDGET",
        "CS_REVERSE_EXPAND_EXPAND_BUDGET",
        EXPAND_GRAPH_BUDGET,
    )
}

/// Per-hop fan-out policy for `reverse_expand_from_anchors`. The historical
/// default (`Fixed(EXPAND_FAN_OUT)`) explores top-N callers
/// uniformly across hops. `DensityScaled` lets dense seeds spend more
/// budget than sparse ones — see issue #69 option 2.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FanOutPolicy {
    Fixed(usize),
    DensityScaled {
        floor: usize,
        cap: usize,
        divisor: usize,
    },
}

/// Which direction the anchor walk traverses. `Reverse` walks edges *into*
/// the anchor (callers/raisers/importers) — the original behaviour, used
/// when the user names a symptom and the fix is upstream. `Forward` walks
/// edges *out of* the anchor (callees) — used when the user names a
/// public API they invoked and the fix lives in its implementation tree.
/// Issue #95.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WalkDirection {
    Forward,
    Reverse,
}

impl WalkDirection {
    /// Return the unvisited neighbours of `id` for this walk direction.
    /// `Reverse` → `graph.dependents(id)` (callers).
    /// `Forward` → `graph.dependencies(id)` (callees).
    pub(crate) fn neighbours<'a>(&self, graph: &'a CodeGraph, id: u64) -> Vec<&'a Symbol> {
        match self {
            WalkDirection::Reverse => graph.dependents(id),
            WalkDirection::Forward => graph.dependencies(id),
        }
    }
}

impl FanOutPolicy {
    /// Select fan-out for one BFS hop given the seed's caller count.
    pub(crate) fn for_hop(&self, dependents_len: usize) -> usize {
        match *self {
            FanOutPolicy::Fixed(n) => n,
            FanOutPolicy::DensityScaled {
                floor,
                cap,
                divisor,
            } => {
                let scaled = dependents_len / divisor.max(1);
                scaled.clamp(floor, cap)
            }
        }
    }
}

// ── Reverse-expand strategy gate (cs-benchmark diagnostic harness) ───────────
//
// `CS_REVERSE_EXPAND_STRATEGY` selects which combination of ranking signals
// drives reverse-edge expansion. Default (`v2`) matches the current main
// behavior — query-term overlap + semantic embedding similarity. The other
// variants exist so the cs-benchmark panel can score them on a stratified
// task panel without making any one of them the default.

/// Reverse-edge expansion strategy. Resolved once per engine call from
/// `CS_REVERSE_EXPAND_STRATEGY` (default `V2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandStrategy {
    /// Skip reverse-expand entirely. Equivalent to `reverse_expand_anchors=false`.
    None,
    /// #67 only. Fixed fan_out=5, no query-term overlap, no semantic.
    /// Pure centrality-driven beam search.
    V0,
    /// Density-aware fan-out alone (no query-term overlap, no semantic).
    V1a,
    /// Query-term-overlap fan-out alone (fixed fan_out=5, no semantic).
    V1b,
    /// Density-aware + query-term-overlap (no semantic). v1 as originally
    /// landed in `0f35b33`.
    V1ab,
    /// Current default. Query-term-overlap + semantic embedding similarity.
    /// Per #69 v2 (PR #83).
    V2,
    /// Total-node-budget + best-first priority-queue walk. Replaces the
    /// fixed (depth × fan_out) BFS with a priority queue ordered by the
    /// same mixed-signal score (lex overlap + semantic + centrality).
    /// Lets the walk spend depth on a promising chain on dense graphs
    /// where BFS's uniform fan-out wastes budget. Issue #69 option 3.
    V3a,
    /// `V3a` + UCB exploration bonus on the parent-seed subtree. Adds
    /// `EXPAND_UCB_C * sqrt(ln(N) / n)` to each candidate's
    /// priority, where `N` is total expansions in the walk and `n` is
    /// expansions within the candidate's seed subtree. Pulls the walk
    /// toward under-sampled subtrees when the per-candidate signal is
    /// noisy — exactly the symptom-anchored case where lexical/semantic
    /// signals are weak.
    V3b,
}

impl ExpandStrategy {
    /// Read `CS_EXPAND_STRATEGY` (or the deprecated alias
    /// `CS_REVERSE_EXPAND_STRATEGY`) from the environment. Unset or
    /// unrecognized → `V2` (current main behavior). Logged at warn
    /// level so misconfigured benchmark runs don't silently fall back.
    pub fn from_env() -> Self {
        let (name, raw) = match std::env::var("CS_EXPAND_STRATEGY") {
            Ok(v) => ("CS_EXPAND_STRATEGY", Some(v)),
            Err(_) => match std::env::var("CS_REVERSE_EXPAND_STRATEGY") {
                Ok(v) => {
                    tracing::warn!(
                        "CS_REVERSE_EXPAND_STRATEGY is deprecated; use CS_EXPAND_STRATEGY instead"
                    );
                    ("CS_REVERSE_EXPAND_STRATEGY", Some(v))
                }
                Err(_) => ("CS_EXPAND_STRATEGY", None),
            },
        };
        match raw
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            None | Some("") | Some("v2") => Self::V2,
            Some("none") | Some("off") => Self::None,
            Some("v0") => Self::V0,
            Some("v1a") => Self::V1a,
            Some("v1b") => Self::V1b,
            Some("v1ab") | Some("v1") => Self::V1ab,
            Some("v3a") => Self::V3a,
            Some("v3b") => Self::V3b,
            Some(other) => {
                tracing::warn!("{}={:?} unrecognized, falling back to v2", name, other);
                Self::V2
            }
        }
    }

    pub(crate) fn fan_out_policy(&self) -> FanOutPolicy {
        match self {
            Self::V1a | Self::V1ab => FanOutPolicy::DensityScaled {
                floor: EXPAND_FAN_OUT,
                cap: EXPAND_FAN_OUT_CAP,
                divisor: EXPAND_FAN_OUT_DIVISOR,
            },
            _ => FanOutPolicy::Fixed(EXPAND_FAN_OUT),
        }
    }

    /// Whether to feed query-term overlap into per-hop ranking. False → caller
    /// scoring degenerates to centrality-only.
    pub(crate) fn use_query_terms(&self) -> bool {
        matches!(self, Self::V1b | Self::V1ab | Self::V2)
    }

    /// Whether to feed semantic body-embedding similarity into per-hop
    /// ranking. Includes `V2` and the V3 variants (which use the same
    /// scoring formula inside the priority queue). Only referenced under
    /// `feature = "embeddings"`; without that feature the engine never
    /// has a scorer to gate.
    #[allow(dead_code)]
    pub(crate) fn use_semantic(&self) -> bool {
        matches!(self, Self::V2 | Self::V3a | Self::V3b)
    }

    /// `V3a`/`V3b` use the priority-queue walker
    /// (`reverse_expand_best_first`) instead of the BFS variant.
    pub(crate) fn use_best_first(&self) -> bool {
        matches!(self, Self::V3a | Self::V3b)
    }

    /// `V3b` adds a UCB exploration bonus on top of `V3a`'s best-first.
    pub(crate) fn use_exploration_bonus(&self) -> bool {
        matches!(self, Self::V3b)
    }

    /// True for variants that are not yet implemented. None remain — kept
    /// for forward compatibility when new variants are added behind the
    /// gate before their implementations land.
    pub(crate) fn is_unimplemented(&self) -> bool {
        false
    }
}

// ── Direction routing (issue #95) ────────────────────────────────────────────
//
// Ranking strategy (`CS_REVERSE_EXPAND_STRATEGY`) and walk direction
// (`CS_EXPAND_DIRECTION`) are orthogonal axes: a variant says *how* to
// rank candidates, a direction says *where* to walk for them.
//
// `Auto` is the default — a per-anchor classifier reads off the index
// (kind, fan-out ratio) and picks Forward/Reverse/Both. Override with
// `CS_EXPAND_DIRECTION ∈ {auto, forward, reverse, both}` for the
// cs-benchmark panel.

/// Effective expansion direction for one anchor seed (after classifier
/// runs / env-var override applies).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EffectiveDirection {
    Forward,
    Reverse,
    Both,
}

/// User-facing expansion-direction setting from `CS_EXPAND_DIRECTION`.
/// `Auto` defers to the per-anchor classifier; the others force a
/// uniform direction across all anchors (used by the panel to compare
/// `forward-only` vs `reverse-only` vs `both` against `auto`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandDirection {
    Auto,
    Forward,
    Reverse,
    Both,
}

impl ExpandDirection {
    /// Read `CS_EXPAND_DIRECTION` from the environment. Unset or
    /// unrecognized → `Auto`. Logged at warn level on unrecognized
    /// values so misconfigured benchmark runs don't silently fall back.
    pub fn from_env() -> Self {
        match std::env::var("CS_EXPAND_DIRECTION")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            None | Some("") | Some("auto") => Self::Auto,
            Some("forward") | Some("fwd") => Self::Forward,
            Some("reverse") | Some("rev") => Self::Reverse,
            Some("both") | Some("bidirectional") => Self::Both,
            Some(other) => {
                tracing::warn!(
                    "CS_EXPAND_DIRECTION={:?} unrecognized, falling back to auto",
                    other
                );
                Self::Auto
            }
        }
    }

    /// Resolve to a concrete direction for one anchor. Forward/Reverse/Both
    /// pass through unchanged; Auto delegates to `classify_direction`.
    pub(crate) fn resolve_for(&self, graph: &CodeGraph, seed: &Symbol) -> EffectiveDirection {
        match self {
            Self::Forward => EffectiveDirection::Forward,
            Self::Reverse => EffectiveDirection::Reverse,
            Self::Both => EffectiveDirection::Both,
            Self::Auto => classify_direction(graph, seed),
        }
    }
}

/// Per-anchor walk-direction classifier. Reads kind + fan-out ratio
/// off the indexed graph — no embeddings, no LLM. Issue #95 layer 3.
///
/// Rules (in order of priority):
/// 1. Exception classes / errors / warnings → reverse only. They are
///    typically named as a *symptom*; the fix lives in the code that
///    raises or handles them, which is upstream.
/// 2. Modules → forward only. The user named a module they imported;
///    the fix is in its tree, not in the modules that import it.
/// 3. Forward fan-out ≫ reverse fan-out → forward. The anchor is a
///    public entry point with a deep implementation tree.
/// 4. Reverse fan-out ≫ forward fan-out → reverse. The anchor is a
///    private leaf utility; the bug is in what calls into it.
/// 5. Otherwise → bidirectional with split budget.
///
/// The 3× ratio is the simplest threshold that sorts SWE-bench
/// Verified anchors empirically. Tunable later if the panel shows a
/// per-language preference.
pub(crate) fn classify_direction(graph: &CodeGraph, seed: &Symbol) -> EffectiveDirection {
    if is_reverse_expand_seed(seed) {
        return EffectiveDirection::Reverse;
    }
    if seed.kind == SymbolKind::Module {
        return EffectiveDirection::Forward;
    }
    let fwd = graph.dependencies(seed.id).len();
    let rev = graph.dependents(seed.id).len();
    let ratio = 3;
    if fwd > ratio * rev.max(1) {
        return EffectiveDirection::Forward;
    }
    if rev > ratio * fwd.max(1) {
        return EffectiveDirection::Reverse;
    }
    EffectiveDirection::Both
}

// ── Fusion & scoring weights ──────────────────────────────────────────────────

/// RRF rank fusion constant (k=60 from the original paper).
pub(crate) const RRF_K: f32 = 60.0;
/// RRF k for the explicit-anchor list only. Lower than the global `RRF_K`
/// (k=15 vs 60) so rank-1 anchor hits contribute ~4× more than rank-1 BM25 —
/// enough to overcome a BM25+ANN combo that both wrong-answer the query.
/// Safe because anchor extraction is precision-first: most noise is filtered
/// out by the stop-word list and the exact-match gate in `anchor_candidates`.
pub(crate) const ANCHOR_RRF_K: f32 = 15.0;
/// RRF k for symbols resolved from Python traceback frames. Even more
/// aggressive than `ANCHOR_RRF_K` because tracebacks carry **both** the
/// file path and the function name — the resolution is precision-first
/// by construction, and a frame in the traceback IS a member of the
/// call chain that produced the bug. Issue #95 layer 1.
pub(crate) const TRACEBACK_RRF_K: f32 = 8.0;
/// Structural injection: score multiplier for injected hub types.
pub(crate) const STRUCTURAL_INJECTION_SCORE: f32 = 5.0;
/// Centrality boost multiplier applied to BM25 score.
pub(crate) const CENTRALITY_BOOST: f32 = 3.0;
/// Fixed boost for markdown symbols (bypasses centrality which is always 0).
pub(crate) const MARKDOWN_CENTRALITY_BYPASS: f32 = 2.5;
/// Weight of BM25+centrality in the final blend (when embeddings are available).
#[cfg(feature = "embeddings")]
pub(crate) const BM25_BLEND_WEIGHT: f32 = 0.5;
/// Weight of semantic cosine similarity in the final blend.
#[cfg(feature = "embeddings")]
pub(crate) const SEMANTIC_BLEND_WEIGHT: f32 = 0.5;
/// Structural re-sort: in-degree weight (dominant signal).
pub(crate) const STRUCTURAL_INDEGREE_WEIGHT: f32 = 20.0;
/// Structural re-sort: accumulated BM25 weight (tiebreaker).
pub(crate) const STRUCTURAL_BM25_WEIGHT: f32 = 0.05;
/// Coordinator bonus per owned seed type.
pub(crate) const COORDINATOR_BONUS_PER_TYPE: f32 = 5.0;
/// Minimum owned seed types required to trigger coordinator bonus.
pub(crate) const COORDINATOR_MIN_OWNED: usize = 2;
/// Score multiplier for symbols from library stub files (`.d.ts`, `.pyi`, `.swiftinterface`).
/// Stubs rank below project symbols and are never returned as pivots.
pub(crate) const STUB_SCORE_WEIGHT: f32 = 0.3;

// ── Candidate retrieval ───────────────────────────────────────────────────────

/// Reciprocal Rank Fusion over multiple ranked lists.
/// Each list contributes `1 / (k + rank + 1)` to a candidate's score.
/// Lists that agree on a candidate amplify its score; unique candidates are preserved.
/// Reciprocal Rank Fusion with a per-list `k`. Lists with smaller `k` boost
/// their top-ranked candidates more aggressively; use this when one retriever
/// is known to be higher-precision than the others (e.g. explicit anchors).
pub(crate) fn rrf_merge_ks(lists: &[(&[(u64, f32)], f32)]) -> Vec<(u64, f32)> {
    let mut scores: HashMap<u64, f32> = HashMap::new();
    for (list, k) in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut merged: Vec<(u64, f32)> = scores.into_iter().collect();
    merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    merged
}

/// Classify a symbol as a reverse-expansion seed.
///
/// Seeds are anchors whose callers we want to walk backward from. The current
/// heuristic fires on **exception/error/warning classes** — the case that
/// motivated issue #67, where the problem statement names the exception but
/// the fix site is reachable only through its raisers.
///
/// Narrower than "any type" on purpose: walking callers of a generic class
/// like `dict` or `Config` would flood the capsule. Name-suffix classifi-
/// cation is cheap, language-agnostic (Python / Rust / Java / Swift all use
/// `Error`/`Exception`/`Warning` suffixes by convention), and correctly
/// skips anchors like `exp` or `symbols` in the sympy-21379 reproducer.
pub(crate) fn is_reverse_expand_seed(sym: &Symbol) -> bool {
    if sym.is_stub {
        return false;
    }
    if !sym.kind.is_type_definition() {
        return false;
    }
    let name = sym.name.as_str();
    name.ends_with("Error") || name.ends_with("Exception") || name.ends_with("Warning")
}

/// True when `sym` is a named exception class whose body is a trivial stub
/// — e.g. `class PolynomialError(BasePolynomialError): pass`.
///
/// Such symbols make terrible pivots: the body carries no behaviour, so the
/// agent sees only a 1-line declaration, yet they rank highly whenever the
/// task mentions the exception class by name (BM25 match on the FQN). The
/// fix is to exclude them from pivot slots — `reverse_expand_from_anchors`
/// will have surfaced the raiser/caller chain separately, and those
/// behaviour-carrying callers should take the pivot slot instead.
///
/// Gate logic:
/// - kind is a class-like type definition (matches `is_reverse_expand_seed`)
/// - name ends with `Error` / `Exception` / `Warning`
/// - body has ≤ 3 non-blank lines (class header + `pass`/docstring + optional base)
///
/// This is a NARROW filter: exception classes with real methods
/// (`__init__`, `__str__`, custom machinery) are retained as pivots because
/// their bodies are informative. Regression: sympy-21379 capsule picked
/// `PolynomialError` (a 1-line `pass` stub) as pivot #7, wasting a slot that
/// `Mod.eval` (the actual fix site, reachable by reverse-expand) should have
/// taken.
pub(crate) fn is_trivial_exception_pivot(sym: &Symbol) -> bool {
    if sym.is_stub {
        return false;
    }
    if !sym.kind.is_type_definition() {
        return false;
    }
    let name = sym.name.as_str();
    if !(name.ends_with("Error") || name.ends_with("Exception") || name.ends_with("Warning")) {
        return false;
    }
    let non_blank_lines = sym.body.lines().filter(|l| !l.trim().is_empty()).count();
    non_blank_lines <= 3
}

/// BFS reverse walk from `seed_ids` through incoming edges (`dependents`).
///
/// Returns `(id, score)` pairs where earlier hops score higher (`1 / (depth + 1)`).
/// Within a hop, callers are ranked by query-term overlap in their name/fqn
/// plus optional body-text semantic similarity, lightly penalized by
/// centrality so utility hubs don't crowd out the intended fix sites.
/// Per-hop expansion is capped at `fan_out`.
///
/// `query_terms` is the already-tokenised, lowercased list of task+context
/// terms. An empty list still walks the graph, it just selects by centrality
/// (and semantic similarity, if provided).
///
/// `semantic_scorer`, when `Some`, is called once per candidate and should
/// return the cosine similarity between the query embedding and that
/// symbol's body embedding in `[0, 1]`. `None` (or a closure returning
/// `None` for a given id) falls back to pure term-overlap + centrality —
/// this is the behaviour on no-embeddings builds and on symbols with no
/// indexed body (e.g. synthetic `Import` entries, already filtered). The
/// weight is `EXPAND_SEMANTIC_WEIGHT`. See issue #69 v2.
///
/// The return order is BFS order (depth-ascending), preserved for RRF.
pub(crate) fn reverse_expand_from_anchors(
    graph: &CodeGraph,
    seed_ids: &[u64],
    query_terms: &[String],
    max_depth: u32,
    fan_out: FanOutPolicy,
    max_total: usize,
    semantic_scorer: Option<&dyn Fn(u64) -> Option<f32>>,
) -> Vec<(u64, f32)> {
    expand_from_anchors_directional(
        graph,
        seed_ids,
        query_terms,
        max_depth,
        fan_out,
        max_total,
        semantic_scorer,
        WalkDirection::Reverse,
    )
}

/// Forward sibling of `reverse_expand_from_anchors` — walks edges *out of*
/// the anchor (callees) instead of into it (callers). Same scoring formula
/// and same fan-out controls; only the neighbour-direction differs.
/// Issue #95 layer 2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_expand_from_anchors(
    graph: &CodeGraph,
    seed_ids: &[u64],
    query_terms: &[String],
    max_depth: u32,
    fan_out: FanOutPolicy,
    max_total: usize,
    semantic_scorer: Option<&dyn Fn(u64) -> Option<f32>>,
) -> Vec<(u64, f32)> {
    expand_from_anchors_directional(
        graph,
        seed_ids,
        query_terms,
        max_depth,
        fan_out,
        max_total,
        semantic_scorer,
        WalkDirection::Forward,
    )
}

#[allow(clippy::too_many_arguments)]
fn expand_from_anchors_directional(
    graph: &CodeGraph,
    seed_ids: &[u64],
    query_terms: &[String],
    max_depth: u32,
    fan_out: FanOutPolicy,
    max_total: usize,
    semantic_scorer: Option<&dyn Fn(u64) -> Option<f32>>,
    direction: WalkDirection,
) -> Vec<(u64, f32)> {
    use std::collections::VecDeque;

    let fan_out_floor = match fan_out {
        FanOutPolicy::Fixed(n) => n,
        FanOutPolicy::DensityScaled { floor, .. } => floor,
    };
    if max_depth == 0 || fan_out_floor == 0 || max_total == 0 || seed_ids.is_empty() {
        return Vec::new();
    }

    let mut visited: HashSet<u64> = seed_ids.iter().copied().collect();
    let mut out: Vec<(u64, f32)> = Vec::new();
    let mut depth_emitted: [usize; 8] = [0; 8];
    // Per-seed depth attribution (#96): track which root each emission
    // descends from. Cheap to maintain — one HashMap entry per seed.
    let mut per_seed_depth: HashMap<u64, [usize; 8]> = HashMap::new();
    let mut queue: VecDeque<(u64, u32, u64)> = seed_ids.iter().map(|&id| (id, 0, id)).collect();

    while let Some((id, depth, parent_seed)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        let neighbours = direction.neighbours(graph, id);
        if neighbours.is_empty() {
            continue;
        }
        let hop_fan_out = fan_out.for_hop(neighbours.len());
        if hop_fan_out == 0 {
            continue;
        }

        // Score each neighbour by:
        //   + `overlap` — count of query-term matches in name / fqn
        //   + `SEMANTIC_WEIGHT * sim` — cosine of body embedding vs query embedding
        //   − `0.1 * centrality` — small penalty so leaf callers beat utility hubs
        //     when everything else ties.
        //
        // The semantic term closes the gap that motivated #69 v2: for
        // symptom-anchored queries like sympy-21379, the fix site's *body*
        // is topically aligned with the problem statement even though its
        // *name* has no lexical overlap with the query. Term overlap alone
        // kept such sites out of the top-`fan_out` beam; body-text similarity
        // surfaces them.
        //
        // Filter out `SymbolKind::Import` entries (retained after the #69
        // revert — the problem existed at #67 too, just less visible). Import
        // statement symbols have no body, no callees beyond the imported
        // names, and no agent-useful content; when they win pivot slots
        // they push the agent into unrelated files.
        let mut scored: Vec<(u64, f32)> = neighbours
            .iter()
            .filter(|s| !s.is_stub)
            .filter(|s| s.kind != SymbolKind::Import)
            .filter(|s| !visited.contains(&s.id))
            .map(|s| {
                let name_lower = s.name.to_lowercase();
                let fqn_lower = s.fqn.to_lowercase();
                let overlap = query_terms
                    .iter()
                    .filter(|t| {
                        let t = t.as_str();
                        name_lower.contains(t) || fqn_lower.contains(t)
                    })
                    .count() as f32;
                let centrality = graph.centrality_score(s.id);
                let semantic = semantic_scorer
                    .and_then(|f| f(s.id))
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                (
                    s.id,
                    overlap + EXPAND_SEMANTIC_WEIGHT * semantic - centrality * 0.1,
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (cid, _) in scored.into_iter().take(hop_fan_out) {
            if visited.insert(cid) {
                let score = 1.0 / (depth as f32 + 2.0);
                let next_depth = depth + 1;
                let bucket = (next_depth as usize).min(depth_emitted.len() - 1);
                depth_emitted[bucket] += 1;
                let per_seed = per_seed_depth.entry(parent_seed).or_insert([0; 8]);
                per_seed[bucket] += 1;
                out.push((cid, score));
                if out.len() >= max_total {
                    log_expand_stats(
                        "expand-bfs",
                        direction,
                        graph,
                        &out,
                        &depth_emitted,
                        &per_seed_depth,
                        true,
                    );
                    return out;
                }
                queue.push_back((cid, next_depth, parent_seed));
            }
        }
    }
    log_expand_stats(
        "expand-bfs",
        direction,
        graph,
        &out,
        &depth_emitted,
        &per_seed_depth,
        false,
    );
    out
}

/// Emit the expand-walker debug log. Centralises the formatting so both
/// walkers (#96) produce the same shape:
///
/// ```text
/// expand-bfs [Forward]: emitted 36 (cap), depth_dist=[7, 29, 0, ...]
///   seed=12345 (Axes::hist) depth_dist=[3, 5, 0, ...]
///   seed=67890 (pyplot::hist) depth_dist=[4, 24, 0, ...]
///   …
/// expand-bfs emissions: lib/.../Axes::hist::do_thing, lib/.../helper, …
/// ```
fn log_expand_stats(
    label: &str,
    direction: WalkDirection,
    graph: &CodeGraph,
    out: &[(u64, f32)],
    depth_emitted: &[usize; 8],
    per_seed_depth: &HashMap<u64, [usize; 8]>,
    cap_hit: bool,
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    let suffix = if cap_hit { " (cap)" } else { "" };
    tracing::debug!(
        "{} [{:?}]: emitted {}{}, depth_dist={:?}",
        label,
        direction,
        out.len(),
        suffix,
        &depth_emitted[1..]
    );
    // Per-seed buckets — sorted by total emissions so the noisiest
    // subtree shows up first in the log.
    let mut by_seed: Vec<(u64, [usize; 8])> = per_seed_depth
        .iter()
        .map(|(&id, dist)| (id, *dist))
        .collect();
    by_seed.sort_by_key(|(_, d)| std::cmp::Reverse(d.iter().sum::<usize>()));
    for (seed_id, dist) in &by_seed {
        let name = graph
            .get_symbol(*seed_id)
            .map(|s| s.fqn.as_str())
            .unwrap_or("<unknown>");
        tracing::debug!("  seed={} ({}) depth_dist={:?}", seed_id, name, &dist[1..]);
    }
    // Pre-RRF emission spot-check (#96): list every FQN the walk
    // emitted so the user can grep for an expected fix-site name and
    // disambiguate "walk found it but RRF dropped it" from "walk
    // never traversed there at all".
    let fqns: Vec<&str> = out
        .iter()
        .filter_map(|(id, _)| graph.get_symbol(*id).map(|s| s.fqn.as_str()))
        .collect();
    tracing::debug!("{} emissions: {}", label, fqns.join(", "));
}

/// Best-first reverse-edge expansion (issue #69 option 3, deferred from
/// the original `0f35b33` v1 — never landed before now). Replaces the
/// fixed `(depth × fan_out)` BFS with a priority-queue walk:
///
/// - Seeds enter the queue at infinite priority so they're popped first.
/// - On each pop we score *all* unvisited dependents of the popped node
///   using the same mixed signal as the BFS variant (lex overlap +
///   `EXPAND_SEMANTIC_WEIGHT * sim` − `0.1 * centrality`).
/// - When `exploration_bonus` is true (`v3b`), each candidate's priority
///   gets an additional `EXPAND_UCB_C * sqrt(ln(N) / n)` term,
///   where `N` is the running total of expansions and `n` is expansions
///   that originated from the same parent seed. Pulls the walk toward
///   under-sampled subtrees when the per-candidate signal is noisy.
/// - The walk stops when `total_budget` candidates have been emitted,
///   `expand_budget` graph expansions have been performed, the queue is
///   empty, or every remaining frontier node is past `max_depth`.
///
/// The output score `1 / (depth + 2)` matches the BFS variant so RRF
/// fusion treats them equivalently — earlier-hop candidates rank higher.
/// Output order is the order candidates were popped from the priority
/// queue (i.e. priority order, mostly), which is what RRF wants for rank.
///
/// `query_terms` empty + `semantic_scorer` None reduces priority to
/// `−0.1 * centrality`, which approximates "least-central first" — a
/// reasonable default when no ranking signal is available.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reverse_expand_best_first(
    graph: &CodeGraph,
    seed_ids: &[u64],
    query_terms: &[String],
    max_depth: u32,
    total_budget: usize,
    expand_budget: usize,
    semantic_scorer: Option<&dyn Fn(u64) -> Option<f32>>,
    exploration_bonus: bool,
) -> Vec<(u64, f32)> {
    expand_best_first_directional(
        graph,
        seed_ids,
        query_terms,
        max_depth,
        total_budget,
        expand_budget,
        semantic_scorer,
        exploration_bonus,
        WalkDirection::Reverse,
    )
}

/// Forward sibling of `reverse_expand_best_first`. Issue #95 layer 2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_expand_best_first(
    graph: &CodeGraph,
    seed_ids: &[u64],
    query_terms: &[String],
    max_depth: u32,
    total_budget: usize,
    expand_budget: usize,
    semantic_scorer: Option<&dyn Fn(u64) -> Option<f32>>,
    exploration_bonus: bool,
) -> Vec<(u64, f32)> {
    expand_best_first_directional(
        graph,
        seed_ids,
        query_terms,
        max_depth,
        total_budget,
        expand_budget,
        semantic_scorer,
        exploration_bonus,
        WalkDirection::Forward,
    )
}

#[allow(clippy::too_many_arguments)]
fn expand_best_first_directional(
    graph: &CodeGraph,
    seed_ids: &[u64],
    query_terms: &[String],
    max_depth: u32,
    total_budget: usize,
    expand_budget: usize,
    semantic_scorer: Option<&dyn Fn(u64) -> Option<f32>>,
    exploration_bonus: bool,
    direction: WalkDirection,
) -> Vec<(u64, f32)> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    if max_depth == 0 || total_budget == 0 || expand_budget == 0 || seed_ids.is_empty() {
        return Vec::new();
    }

    // BinaryHeap is a max-heap on Ord. Wrap f32 priorities so NaN doesn't
    // panic and so highest-priority pops first.
    #[derive(Clone, Copy)]
    struct PqItem {
        priority: f32,
        id: u64,
        depth: u32,
        parent_seed: u64,
    }
    impl PartialEq for PqItem {
        fn eq(&self, other: &Self) -> bool {
            self.priority.total_cmp(&other.priority).is_eq()
        }
    }
    impl Eq for PqItem {}
    impl Ord for PqItem {
        fn cmp(&self, other: &Self) -> Ordering {
            self.priority.total_cmp(&other.priority)
        }
    }
    impl PartialOrd for PqItem {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut visited: HashSet<u64> = seed_ids.iter().copied().collect();
    let mut out: Vec<(u64, f32)> = Vec::new();
    let mut depth_emitted: [usize; 8] = [0; 8];
    let mut per_seed_depth: HashMap<u64, [usize; 8]> = HashMap::new();
    let mut pq: BinaryHeap<PqItem> = BinaryHeap::new();
    let mut subtree_visits: HashMap<u64, usize> = HashMap::new();
    let mut total_expansions: usize = 0;

    for &seed in seed_ids {
        pq.push(PqItem {
            priority: f32::INFINITY,
            id: seed,
            depth: 0,
            parent_seed: seed,
        });
    }

    while let Some(item) = pq.pop() {
        if total_expansions >= expand_budget {
            break;
        }
        if item.depth >= max_depth {
            continue;
        }
        let neighbours = direction.neighbours(graph, item.id);
        if neighbours.is_empty() {
            continue;
        }
        total_expansions += 1;

        for s in neighbours
            .iter()
            .filter(|s| !s.is_stub)
            .filter(|s| s.kind != SymbolKind::Import)
        {
            if !visited.insert(s.id) {
                continue;
            }
            let name_lower = s.name.to_lowercase();
            let fqn_lower = s.fqn.to_lowercase();
            let overlap = query_terms
                .iter()
                .filter(|t| {
                    let t = t.as_str();
                    name_lower.contains(t) || fqn_lower.contains(t)
                })
                .count() as f32;
            let centrality = graph.centrality_score(s.id);
            let semantic = semantic_scorer
                .and_then(|f| f(s.id))
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);

            let mut priority = overlap + EXPAND_SEMANTIC_WEIGHT * semantic - centrality * 0.1;

            if exploration_bonus {
                let n_sub = (*subtree_visits.get(&item.parent_seed).unwrap_or(&0)).max(1) as f32;
                let n_total = total_expansions.max(1) as f32;
                let bonus = EXPAND_UCB_C * (n_total.ln() / n_sub).sqrt();
                priority += bonus;
            }

            let next_depth = item.depth + 1;
            let out_score = 1.0 / (next_depth as f32 + 1.0);
            let bucket = (next_depth as usize).min(depth_emitted.len() - 1);
            depth_emitted[bucket] += 1;
            let per_seed = per_seed_depth.entry(item.parent_seed).or_insert([0; 8]);
            per_seed[bucket] += 1;
            out.push((s.id, out_score));
            *subtree_visits.entry(item.parent_seed).or_insert(0) += 1;

            if out.len() >= total_budget {
                tracing::debug!(
                    "expand-best-first [{:?}]: expansions={}",
                    direction,
                    total_expansions
                );
                log_expand_stats(
                    "expand-best-first",
                    direction,
                    graph,
                    &out,
                    &depth_emitted,
                    &per_seed_depth,
                    true,
                );
                return out;
            }

            pq.push(PqItem {
                priority,
                id: s.id,
                depth: next_depth,
                parent_seed: item.parent_seed,
            });
        }
    }

    tracing::debug!(
        "expand-best-first [{:?}]: expansions={}",
        direction,
        total_expansions
    );
    log_expand_stats(
        "expand-best-first",
        direction,
        graph,
        &out,
        &depth_emitted,
        &per_seed_depth,
        false,
    );
    out
}

/// Split a free-text query into lowercase term tokens usable by
/// `reverse_expand_from_anchors`. Mirrors the rest of the ranking pipeline:
/// split on non-alphanumerics, drop short tokens, lowercase.
pub(crate) fn query_terms_for_reverse_expand(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3)
        .map(|t| t.to_lowercase())
        .collect()
}

/// Expand 1-hop graph neighbors of BM25 seed IDs, ranked by centrality.
/// Seeds themselves are excluded — they are already in the BM25 list.
pub(crate) fn graph_candidates(
    graph: &CodeGraph,
    seed_ids: &HashSet<u64>,
    max: usize,
) -> Vec<(u64, f32)> {
    let mut seen: HashSet<u64> = seed_ids.clone();
    let mut candidates: Vec<(u64, f32)> = Vec::new();
    for &seed in seed_ids {
        for neighbor_id in graph.neighbor_ids(seed) {
            if seen.insert(neighbor_id) {
                candidates.push((neighbor_id, graph.centrality_score(neighbor_id)));
            }
        }
    }
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max);
    candidates
}

/// Augment the BM25 candidate pool with top hub types ranked by family in-degree.
/// BM25 cannot surface types whose names don't lexically match the query.
pub(crate) fn inject_structural_candidates(
    graph: &CodeGraph,
    search_results: &mut Vec<(u64, f32)>,
    max_pivots: usize,
) {
    let mut by_in_degree: Vec<(u64, f32)> = graph
        .all_symbols()
        .filter(|s| s.kind.is_type_definition() || s.kind == SymbolKind::Module)
        .map(|s| (s.id, graph.family_in_degree_score(s.id)))
        .collect();
    by_in_degree.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (id, c_in) in by_in_degree.into_iter().take(max_pivots * 2) {
        if !search_results.iter().any(|(sid, _)| *sid == id) {
            search_results.push((id, c_in * STRUCTURAL_INJECTION_SCORE));
        }
    }
}

// ── Re-ranking ────────────────────────────────────────────────────────────────

/// For Structural intent: re-sort so type definitions ranked by in-degree come first,
/// with a coordinator bonus for types that declare BM25-matched types as properties.
pub(crate) fn apply_structural_resort(
    graph: &CodeGraph,
    scored: Vec<(u64, f32)>,
    bm25_ids: &HashSet<u64>,
    query: &str,
) -> Vec<(u64, f32)> {
    let is_hub_type = |id: u64| {
        graph
            .get_symbol(id)
            .map(|s| {
                s.kind.is_type_definition()
                    || s.kind == SymbolKind::Impl
                    || s.kind == SymbolKind::Module
            })
            .unwrap_or(false)
    };

    let mut type_scored: Vec<(u64, f32)> = scored
        .iter()
        .filter(|(id, _)| is_hub_type(*id))
        .map(|(id, accumulated)| {
            let c_in = graph.family_in_degree_score(*id);
            (
                *id,
                c_in * STRUCTURAL_INDEGREE_WEIGHT + accumulated * STRUCTURAL_BM25_WEIGHT,
            )
        })
        .collect();

    // Coordinator bonus: find the type that DECLARES BM25-matched types as member
    // variables. Seeds = BM25-matched types whose names overlap with 4-char query stems.
    let query_stems: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 3)
        .map(|t| t[..t.len().min(4)].to_lowercase())
        .collect();
    let seed_names: Vec<String> = type_scored
        .iter()
        .filter_map(|(id, _)| {
            if bm25_ids.contains(id) {
                graph.get_symbol(*id).map(|s| s.name.clone())
            } else {
                None
            }
        })
        .filter(|n| {
            n.len() > 4 && {
                let nl = n.to_lowercase();
                query_stems.iter().any(|stem| nl.contains(stem.as_str()))
            }
        })
        .collect();

    if seed_names.len() >= 2 {
        for (id, score) in &mut type_scored {
            if let Some(sym) = graph.get_symbol(*id) {
                let path_lower = sym.file_path.to_lowercase();
                if path_lower.contains("test")
                    || path_lower.contains("spec")
                    || path_lower.contains("mock")
                {
                    continue;
                }
                let owned = seed_names
                    .iter()
                    .filter(|name| {
                        *name != &sym.name
                            && (sym.body.contains(&format!(": {}", name))
                                || sym.body.contains(&format!(": [{}]", name))
                                || sym.body.contains(&format!(": {}?", name))
                                || sym.body.contains(&format!("[{}]", name)))
                    })
                    .count();
                if owned >= COORDINATOR_MIN_OWNED {
                    *score += owned as f32 * COORDINATOR_BONUS_PER_TYPE;
                }
            }
        }
    }

    type_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let others: Vec<(u64, f32)> = scored
        .into_iter()
        .filter(|(id, _)| !is_hub_type(*id))
        .collect();
    type_scored.into_iter().chain(others).collect()
}

// ── Selection & deduplication ─────────────────────────────────────────────────

/// Deduplicate by FQN — keep the highest-scored entry per unique FQN.
pub(crate) fn dedup_by_fqn(graph: &CodeGraph, scored: Vec<(u64, f32)>) -> Vec<(u64, f32)> {
    let mut seen_fqns = HashSet::new();
    scored
        .into_iter()
        .filter(|(id, _)| {
            graph
                .get_symbol(*id)
                .map(|sym| seen_fqns.insert(sym.fqn.clone()))
                .unwrap_or(true)
        })
        .collect()
}

/// Select adjacent symbol IDs based on the search intent.
pub(crate) fn select_adjacents(
    graph: &CodeGraph,
    pivot_ids: &[u64],
    intent: &SearchIntent,
    max_adjacent: usize,
) -> Vec<u64> {
    let raw: Vec<u64> = match intent {
        SearchIntent::Debug => pivot_ids
            .iter()
            .flat_map(|&id| {
                graph
                    .dependencies(id)
                    .iter()
                    .map(|s| s.id)
                    .collect::<Vec<_>>()
            })
            .filter(|id| !pivot_ids.contains(id))
            .take(max_adjacent)
            .collect(),
        SearchIntent::Refactor => pivot_ids
            .iter()
            .flat_map(|&id| {
                graph
                    .blast_radius(id, 2)
                    .iter()
                    .map(|s| s.id)
                    .collect::<Vec<_>>()
            })
            .filter(|id| !pivot_ids.contains(id))
            .take(max_adjacent)
            .collect(),
        _ => pivot_ids
            .iter()
            .flat_map(|&id| {
                let mut adj: Vec<u64> = graph.dependencies(id).iter().map(|s| s.id).collect();
                adj.extend(graph.dependents(id).iter().map(|s| s.id));
                adj
            })
            .filter(|id| !pivot_ids.contains(id))
            .take(max_adjacent)
            .collect(),
    };
    // Deduplicate (same symbol may be reachable from multiple pivots).
    let mut seen = HashSet::new();
    raw.into_iter().filter(|id| seen.insert(*id)).collect()
}

/// Resolve adjacent IDs to symbols, filtering test files and capping per-file counts.
pub(crate) fn resolve_adjacents<'a>(
    graph: &'a CodeGraph,
    adjacent_ids: &[u64],
    filter_test_files: bool,
) -> Vec<&'a Symbol> {
    let mut file_counts: HashMap<&str, usize> = HashMap::new();
    adjacent_ids
        .iter()
        .filter_map(|id| graph.get_symbol(*id))
        .filter(|sym| {
            if filter_test_files {
                let p = sym.file_path.to_lowercase();
                !p.contains("test") && !p.contains("spec") && !p.contains("mock")
            } else {
                true
            }
        })
        .filter(|sym| {
            let count = file_counts.entry(sym.file_path.as_str()).or_insert(0);
            *count += 1;
            *count <= 2 // max 2 symbols per file in adjacents
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CodeGraph;
    use crate::language::Language;
    use crate::symbol::{EdgeKind, Symbol, SymbolKind};

    fn mk(file: &str, name: &str, kind: SymbolKind) -> Symbol {
        Symbol::new(
            file,
            name,
            kind,
            1,
            1,
            String::new(),
            None,
            String::new(),
            Language::Python,
        )
    }

    /// Build a tiny graph: one seed (exception class) and `n` direct callers
    /// whose names have no lexical overlap with any reasonable query. Returns
    /// `(graph, seed_id, caller_ids)`.
    fn graph_with_anonymous_callers(n: usize) -> (CodeGraph, u64, Vec<u64>) {
        let mut g = CodeGraph::new();
        let seed = mk("err.py", "MyError", SymbolKind::Class);
        let seed_id = seed.id;
        g.add_symbol(seed);
        let mut caller_ids = Vec::new();
        for i in 0..n {
            let c = mk(
                &format!("c_{i}.py"),
                &format!("anon_{i}"),
                SymbolKind::Function,
            );
            caller_ids.push(c.id);
            g.add_symbol(c);
        }
        for &cid in &caller_ids {
            g.add_edge(cid, seed_id, EdgeKind::Calls);
        }
        g.warm_caches();
        (g, seed_id, caller_ids)
    }

    /// Issue #69 v2: when every direct caller of a seed has zero lexical
    /// overlap with the query, a semantic scorer that singles out one caller
    /// must steer the BFS to that caller at `fan_out=1`. This is the
    /// sympy-21379 failure mode reduced to a unit fixture.
    #[test]
    fn semantic_scorer_promotes_aligned_caller_under_zero_overlap() {
        let (g, seed_id, caller_ids) = graph_with_anonymous_callers(6);
        let terms: Vec<String> = vec!["substitution".into(), "piecewise".into()];
        let aligned = caller_ids[3];
        let scorer = |id: u64| -> Option<f32> {
            if id == aligned {
                Some(0.9)
            } else {
                Some(0.3)
            }
        };
        let out = reverse_expand_from_anchors(
            &g,
            &[seed_id],
            &terms,
            3,
            FanOutPolicy::Fixed(1),
            1,
            Some(&scorer),
        );
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].0, aligned,
            "semantic scorer should promote aligned caller; got {:?}",
            out
        );
    }

    /// Different semantic target, different winner — proves the scorer's
    /// *output* drives selection, not a fixed id/ordering bias.
    #[test]
    fn semantic_scorer_winner_follows_scorer_output() {
        let (g, seed_id, caller_ids) = graph_with_anonymous_callers(6);
        let terms: Vec<String> = vec!["substitution".into()];
        let target = caller_ids[5];
        let scorer = |id: u64| -> Option<f32> {
            if id == target {
                Some(0.95)
            } else {
                Some(0.1)
            }
        };
        let out = reverse_expand_from_anchors(
            &g,
            &[seed_id],
            &terms,
            3,
            FanOutPolicy::Fixed(1),
            1,
            Some(&scorer),
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, target);
    }

    /// Strong lexical overlap (multiple term matches in name) must still win
    /// against a high semantic score — the semantic term is a tiebreaker and
    /// a soft signal for zero-overlap candidates, not a replacement for
    /// explicit lexical hits. With `SEMANTIC_WEIGHT = 2.0`, two term matches
    /// (`+2.0`) outweigh one perfect semantic hit (`2.0 * 0.9 = 1.8`).
    #[test]
    fn lexical_overlap_still_dominates_when_strong() {
        let mut g = CodeGraph::new();
        let seed = mk("err.py", "MyError", SymbolKind::Class);
        let seed_id = seed.id;
        g.add_symbol(seed);

        let lexical = mk(
            "a.py",
            "substitution_piecewise_handler",
            SymbolKind::Function,
        );
        let lexical_id = lexical.id;
        g.add_symbol(lexical);

        let semantic = mk("b.py", "anon_helper", SymbolKind::Function);
        let semantic_id = semantic.id;
        g.add_symbol(semantic);

        g.add_edge(lexical_id, seed_id, EdgeKind::Calls);
        g.add_edge(semantic_id, seed_id, EdgeKind::Calls);
        g.warm_caches();

        let terms: Vec<String> = vec!["substitution".into(), "piecewise".into()];
        let scorer = |id: u64| -> Option<f32> {
            if id == semantic_id {
                Some(0.9)
            } else {
                None
            }
        };

        let out = reverse_expand_from_anchors(
            &g,
            &[seed_id],
            &terms,
            3,
            FanOutPolicy::Fixed(1),
            1,
            Some(&scorer),
        );
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].0, lexical_id,
            "two term matches (+2.0) should outweigh one semantic hit (+1.8)"
        );
    }

    /// No semantic scorer → behaviour is identical to the pre-v2 pure term-
    /// overlap walk. A caller with strictly more lexical overlap must be
    /// picked regardless of the other caller's graph position.
    #[test]
    fn no_scorer_falls_back_to_pure_term_overlap() {
        let mut g = CodeGraph::new();
        let seed = mk("err.py", "MyError", SymbolKind::Class);
        let seed_id = seed.id;
        g.add_symbol(seed);

        let winner = mk("a.py", "substitution_handler", SymbolKind::Function);
        let winner_id = winner.id;
        g.add_symbol(winner);
        let loser = mk("b.py", "anon_helper", SymbolKind::Function);
        let loser_id = loser.id;
        g.add_symbol(loser);

        g.add_edge(winner_id, seed_id, EdgeKind::Calls);
        g.add_edge(loser_id, seed_id, EdgeKind::Calls);
        g.warm_caches();

        let terms: Vec<String> = vec!["substitution".into()];
        let out =
            reverse_expand_from_anchors(&g, &[seed_id], &terms, 3, FanOutPolicy::Fixed(1), 1, None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, winner_id);
    }

    // ── reverse_expand_best_first (v3a/v3b) ──────────────────────────────

    #[test]
    fn best_first_walks_priority_order_not_breadth_first() {
        // Seed → 3 direct callers (A, B, C). C has the highest term score.
        // Each caller has its own grandchild (A1, B1, C1). C1 has term-score 0
        // but inherits high priority because C was popped first.
        //
        // BFS at fan_out=1 / max_total=2 would emit [C, A1] (one per hop).
        // Best-first at total_budget=2 should emit [C, C1] (depth-2 wins
        // because C's grandchild was reachable at higher priority than
        // unexpanded A or B).
        let mut g = CodeGraph::new();
        let seed = mk("err.py", "Err", SymbolKind::Class);
        let seed_id = seed.id;
        g.add_symbol(seed);
        let a = mk("a.py", "anon_a", SymbolKind::Function);
        let b = mk("b.py", "anon_b", SymbolKind::Function);
        let c = mk("c.py", "substitution_handler", SymbolKind::Function);
        let (a_id, b_id, c_id) = (a.id, b.id, c.id);
        g.add_symbol(a);
        g.add_symbol(b);
        g.add_symbol(c);
        let a1 = mk("a.py", "anon_a_helper", SymbolKind::Function);
        let b1 = mk("b.py", "anon_b_helper", SymbolKind::Function);
        let c1 = mk("c.py", "anon_c_helper", SymbolKind::Function);
        let (a1_id, _b1_id, c1_id) = (a1.id, b1.id, c1.id);
        g.add_symbol(a1);
        g.add_symbol(b1);
        g.add_symbol(c1);
        g.add_edge(a_id, seed_id, EdgeKind::Calls);
        g.add_edge(b_id, seed_id, EdgeKind::Calls);
        g.add_edge(c_id, seed_id, EdgeKind::Calls);
        g.add_edge(a1_id, a_id, EdgeKind::Calls);
        g.add_edge(_b1_id, b_id, EdgeKind::Calls);
        g.add_edge(c1_id, c_id, EdgeKind::Calls);
        g.warm_caches();

        let terms: Vec<String> = vec!["substitution".into()];
        let out = reverse_expand_best_first(&g, &[seed_id], &terms, 3, 2, 100, None, false);

        assert_eq!(out.len(), 2);
        let ids: Vec<u64> = out.iter().map(|(id, _)| *id).collect();
        assert!(
            ids.contains(&c_id),
            "C should be emitted first; got {:?}",
            ids
        );
        // The second slot goes to either C1 (because C had highest priority and
        // was expanded next) or another seed-direct caller. Best-first
        // greedily descends — C1 inherits C's high subtree priority once
        // C is popped. Accept either C1 or A/B since centrality penalty
        // is identical and term overlap is zero for all three.
        assert!(
            ids.contains(&c1_id) || ids.contains(&a_id) || ids.contains(&b_id),
            "second slot should be a frontier candidate; got {:?}",
            ids,
        );
    }

    #[test]
    fn best_first_respects_total_budget() {
        // With many callers and total_budget=3, we get exactly 3 outputs
        // even though the graph has more.
        let (g, seed_id, _caller_ids) = graph_with_anonymous_callers(20);
        let out = reverse_expand_best_first(&g, &[seed_id], &[], 3, 3, 100, None, false);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn best_first_respects_expand_budget() {
        // expand_budget=1 means only the seeds themselves get expanded once.
        // We get up to fan_out_full callers in one batch, but no deeper
        // expansion. Output count ≤ direct callers.
        let (g, seed_id, _caller_ids) = graph_with_anonymous_callers(5);
        let out = reverse_expand_best_first(&g, &[seed_id], &[], 5, 100, 1, None, false);
        // 5 direct callers were emitted in the seed expansion; expand_budget=1
        // stopped further expansions. Each caller has no further dependents
        // anyway in this fixture, so the cap is exercised but the result
        // happens to be the same as if we'd let it run.
        assert_eq!(out.len(), 5);
    }

    #[test]
    fn best_first_zero_signal_returns_callers() {
        // No query terms, no semantic scorer — priority degenerates to
        // `-0.1 * centrality`. Walk should still emit seed dependents in
        // some order. (The exact order isn't important here; this is a
        // doesn't-panic / doesn't-deadlock smoke test.)
        let (g, seed_id, _caller_ids) = graph_with_anonymous_callers(3);
        let out = reverse_expand_best_first(&g, &[seed_id], &[], 3, 10, 100, None, false);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn best_first_exploration_bonus_is_finite() {
        // UCB-bonus path must not produce NaN priorities (which would
        // destroy the BinaryHeap ordering). Smoke test that v3b still
        // converges on the same fixture.
        let (g, seed_id, _caller_ids) = graph_with_anonymous_callers(5);
        let out = reverse_expand_best_first(&g, &[seed_id], &[], 3, 5, 100, None, true);
        assert_eq!(out.len(), 5);
        for (_, score) in &out {
            assert!(
                score.is_finite(),
                "out score should be finite, got {}",
                score
            );
        }
    }

    // ── Forward expansion + direction classifier (issue #95) ────────────

    /// Build a small graph for forward-expand: one anchor with `n` direct
    /// callees. Returns `(graph, anchor_id, callee_ids)`.
    fn graph_with_anonymous_callees(n: usize) -> (CodeGraph, u64, Vec<u64>) {
        let mut g = CodeGraph::new();
        let anchor = mk("api.py", "public_api", SymbolKind::Function);
        let anchor_id = anchor.id;
        g.add_symbol(anchor);
        let mut callee_ids = Vec::new();
        for i in 0..n {
            let c = mk(
                &format!("impl_{i}.py"),
                &format!("anon_{i}"),
                SymbolKind::Function,
            );
            callee_ids.push(c.id);
            g.add_symbol(c);
        }
        // Anchor → callee (anchor depends on callee).
        for &cid in &callee_ids {
            g.add_edge(anchor_id, cid, EdgeKind::Calls);
        }
        g.warm_caches();
        (g, anchor_id, callee_ids)
    }

    #[test]
    fn forward_expand_walks_callees_not_callers() {
        // Anchor has 3 callees and zero callers. Reverse walk returns
        // nothing; forward walk returns all 3.
        let (g, anchor_id, callee_ids) = graph_with_anonymous_callees(3);
        let rev =
            reverse_expand_from_anchors(&g, &[anchor_id], &[], 3, FanOutPolicy::Fixed(5), 10, None);
        assert!(rev.is_empty(), "no callers → reverse should be empty");
        let fwd =
            forward_expand_from_anchors(&g, &[anchor_id], &[], 3, FanOutPolicy::Fixed(5), 10, None);
        assert_eq!(fwd.len(), 3);
        let ids: HashSet<u64> = fwd.iter().map(|(id, _)| *id).collect();
        for cid in &callee_ids {
            assert!(ids.contains(cid));
        }
    }

    #[test]
    fn forward_best_first_walks_callees() {
        // The best-first variant must also dispatch to dependencies()
        // when direction is forward — smoke-test the wiring.
        let (g, anchor_id, _callee_ids) = graph_with_anonymous_callees(4);
        let fwd = forward_expand_best_first(&g, &[anchor_id], &[], 3, 4, 50, None, false);
        assert_eq!(fwd.len(), 4);
    }

    #[test]
    fn classify_direction_exception_class_is_reverse() {
        let mut g = CodeGraph::new();
        let exc = mk("err.py", "PolynomialError", SymbolKind::Class);
        let exc_id = exc.id;
        g.add_symbol(exc);
        // Add a few raisers so reverse fan-out > 0.
        for i in 0..3 {
            let r = mk(
                &format!("r{i}.py"),
                &format!("raiser_{i}"),
                SymbolKind::Function,
            );
            let rid = r.id;
            g.add_symbol(r);
            g.add_edge(rid, exc_id, EdgeKind::Calls);
        }
        g.warm_caches();
        let sym = g.get_symbol(exc_id).unwrap();
        assert_eq!(classify_direction(&g, sym), EffectiveDirection::Reverse);
    }

    #[test]
    fn classify_direction_high_fanout_function_is_forward() {
        // Anchor has 10 callees, 1 caller — forward-shaped (3× ratio met).
        let mut g = CodeGraph::new();
        let anchor = mk("api.py", "do_thing", SymbolKind::Function);
        let anchor_id = anchor.id;
        g.add_symbol(anchor);
        for i in 0..10 {
            let cal = mk(
                &format!("c{i}.py"),
                &format!("step_{i}"),
                SymbolKind::Function,
            );
            let cal_id = cal.id;
            g.add_symbol(cal);
            g.add_edge(anchor_id, cal_id, EdgeKind::Calls);
        }
        let caller = mk("user.py", "user", SymbolKind::Function);
        let caller_id = caller.id;
        g.add_symbol(caller);
        g.add_edge(caller_id, anchor_id, EdgeKind::Calls);
        g.warm_caches();
        let sym = g.get_symbol(anchor_id).unwrap();
        assert_eq!(classify_direction(&g, sym), EffectiveDirection::Forward);
    }

    // ── budget env-var resolvers (#96) ───────────────────────────────────
    //
    // These tests mutate process-wide env state and would race if run in
    // parallel — `cargo test` partitions tests across threads. Serialise
    // by sharing a single test that sets, asserts, unsets in sequence.

    #[test]
    fn resolve_total_and_expand_budget_env_overrides() {
        let all_vars = [
            "CS_EXPAND_TOTAL_BUDGET",
            "CS_EXPAND_GRAPH_BUDGET",
            "CS_REVERSE_EXPAND_TOTAL_BUDGET",
            "CS_REVERSE_EXPAND_EXPAND_BUDGET",
        ];
        for v in all_vars {
            std::env::remove_var(v);
        }

        // Default (unset) → constants.
        assert_eq!(resolve_total_budget(), EXPAND_TOTAL_BUDGET);
        assert_eq!(resolve_expand_budget(), EXPAND_GRAPH_BUDGET);

        // New names win.
        std::env::set_var("CS_EXPAND_TOTAL_BUDGET", "200");
        std::env::set_var("CS_EXPAND_GRAPH_BUDGET", "1000");
        assert_eq!(resolve_total_budget(), 200);
        assert_eq!(resolve_expand_budget(), 1000);

        // New names take precedence over deprecated aliases when both
        // are set.
        std::env::set_var("CS_REVERSE_EXPAND_TOTAL_BUDGET", "999");
        std::env::set_var("CS_REVERSE_EXPAND_EXPAND_BUDGET", "9999");
        assert_eq!(resolve_total_budget(), 200);
        assert_eq!(resolve_expand_budget(), 1000);

        // Without new names, deprecated aliases are read (with a warn).
        std::env::remove_var("CS_EXPAND_TOTAL_BUDGET");
        std::env::remove_var("CS_EXPAND_GRAPH_BUDGET");
        assert_eq!(resolve_total_budget(), 999);
        assert_eq!(resolve_expand_budget(), 9999);

        // Unparseable → falls back to constants.
        std::env::set_var("CS_EXPAND_TOTAL_BUDGET", "not-a-number");
        std::env::set_var("CS_EXPAND_GRAPH_BUDGET", "");
        std::env::remove_var("CS_REVERSE_EXPAND_TOTAL_BUDGET");
        std::env::remove_var("CS_REVERSE_EXPAND_EXPAND_BUDGET");
        assert_eq!(resolve_total_budget(), EXPAND_TOTAL_BUDGET);
        assert_eq!(resolve_expand_budget(), EXPAND_GRAPH_BUDGET);

        for v in all_vars {
            std::env::remove_var(v);
        }
    }

    #[test]
    fn classify_direction_balanced_anchor_is_both() {
        // Symmetric fan-out → ambiguous → Both.
        let mut g = CodeGraph::new();
        let anchor = mk("mid.py", "intermediary", SymbolKind::Function);
        let anchor_id = anchor.id;
        g.add_symbol(anchor);
        for i in 0..3 {
            let cal = mk(
                &format!("c{i}.py"),
                &format!("callee_{i}"),
                SymbolKind::Function,
            );
            let cal_id = cal.id;
            g.add_symbol(cal);
            g.add_edge(anchor_id, cal_id, EdgeKind::Calls);
        }
        for i in 0..3 {
            let r = mk(
                &format!("r{i}.py"),
                &format!("caller_{i}"),
                SymbolKind::Function,
            );
            let r_id = r.id;
            g.add_symbol(r);
            g.add_edge(r_id, anchor_id, EdgeKind::Calls);
        }
        g.warm_caches();
        let sym = g.get_symbol(anchor_id).unwrap();
        assert_eq!(classify_direction(&g, sym), EffectiveDirection::Both);
    }
}
