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
// their callers/raisers up to `REVERSE_EXPAND_MAX_DEPTH` hops, injecting the
// walk results into the RRF fusion alongside the direct anchor list.

/// Max hops to BFS backward through `dependents()` from each seed anchor.
/// 3 is the tightest depth that covers the motivating sympy-21379 case
/// (`PolynomialError ← parallel_poly_from_expr ← gcd ← Mod.eval`).
pub(crate) const REVERSE_EXPAND_MAX_DEPTH: u32 = 3;
/// Per-hop cap on the number of callers expanded. Prevents exponential blowup
/// when walking from an exception class that's imported/raised in hundreds of
/// sites. Selection within a hop is driven by term overlap with the query.
pub(crate) const REVERSE_EXPAND_FAN_OUT: usize = 5;
/// Overall cap on the reverse-expansion candidate list. Mirrors
/// `ANCHOR_CANDIDATES` — the walk is precision-first, not recall-first.
pub(crate) const REVERSE_EXPAND_CANDIDATES: usize = 20;
/// RRF k for the reverse-expansion list. Sits between `ANCHOR_RRF_K = 15`
/// (aggressive, precision-first) and `RRF_K = 60` (default): seeds-reachable
/// symbols contribute meaningfully without overwhelming direct-anchor hits.
pub(crate) const REVERSE_EXPAND_RRF_K: f32 = 30.0;
/// Upper bound on a seed's direct-caller count. Anchors with more callers are
/// skipped — they're hubs (e.g. `exp`, `symbols`) whose reverse set would
/// flood the capsule. Exception classes in real codebases typically have
/// dozens-to-low-hundreds of raisers; this caps the seed set but the per-hop
/// `REVERSE_EXPAND_FAN_OUT` still bounds each walk.
pub(crate) const REVERSE_EXPAND_SEED_MAX_CALLERS: usize = 500;
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
pub(crate) const REVERSE_EXPAND_SEMANTIC_WEIGHT: f32 = 2.0;

// ── Fusion & scoring weights ──────────────────────────────────────────────────

/// RRF rank fusion constant (k=60 from the original paper).
pub(crate) const RRF_K: f32 = 60.0;
/// RRF k for the explicit-anchor list only. Lower than the global `RRF_K`
/// (k=15 vs 60) so rank-1 anchor hits contribute ~4× more than rank-1 BM25 —
/// enough to overcome a BM25+ANN combo that both wrong-answer the query.
/// Safe because anchor extraction is precision-first: most noise is filtered
/// out by the stop-word list and the exact-match gate in `anchor_candidates`.
pub(crate) const ANCHOR_RRF_K: f32 = 15.0;
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
/// weight is `REVERSE_EXPAND_SEMANTIC_WEIGHT`. See issue #69 v2.
///
/// The return order is BFS order (depth-ascending), preserved for RRF.
pub(crate) fn reverse_expand_from_anchors(
    graph: &CodeGraph,
    seed_ids: &[u64],
    query_terms: &[String],
    max_depth: u32,
    fan_out: usize,
    max_total: usize,
    semantic_scorer: Option<&dyn Fn(u64) -> Option<f32>>,
) -> Vec<(u64, f32)> {
    use std::collections::VecDeque;

    if max_depth == 0 || fan_out == 0 || max_total == 0 || seed_ids.is_empty() {
        return Vec::new();
    }

    let mut visited: HashSet<u64> = seed_ids.iter().copied().collect();
    let mut out: Vec<(u64, f32)> = Vec::new();
    let mut queue: VecDeque<(u64, u32)> = seed_ids.iter().map(|&id| (id, 0)).collect();

    while let Some((id, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        let dependents = graph.dependents(id);
        if dependents.is_empty() {
            continue;
        }

        // Score each caller by:
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
        let mut scored: Vec<(u64, f32)> = dependents
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
                    overlap + REVERSE_EXPAND_SEMANTIC_WEIGHT * semantic - centrality * 0.1,
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (cid, _) in scored.into_iter().take(fan_out) {
            if visited.insert(cid) {
                let score = 1.0 / (depth as f32 + 2.0);
                out.push((cid, score));
                if out.len() >= max_total {
                    return out;
                }
                queue.push_back((cid, depth + 1));
            }
        }
    }
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
        let out = reverse_expand_from_anchors(&g, &[seed_id], &terms, 3, 1, 1, Some(&scorer));
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
        let out = reverse_expand_from_anchors(&g, &[seed_id], &terms, 3, 1, 1, Some(&scorer));
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

        let out = reverse_expand_from_anchors(&g, &[seed_id], &terms, 3, 1, 1, Some(&scorer));
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
        let out = reverse_expand_from_anchors(&g, &[seed_id], &terms, 3, 1, 1, None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, winner_id);
    }
}
