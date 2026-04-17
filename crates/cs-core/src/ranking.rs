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
