use crate::skeletonizer::skeletonize;
use crate::symbol::Symbol;
use serde::{Deserialize, Serialize};

/// Default token budget if not specified by the caller.
pub const DEFAULT_TOKEN_BUDGET: u32 = 4_000;

/// The assembled context capsule returned to the agent.
#[derive(Debug, Serialize, Deserialize)]
pub struct Capsule {
    /// Pivot symbols — returned with their full source body.
    pub pivots: Vec<PivotEntry>,

    /// Adjacent symbols — returned as skeletons only.
    pub skeletons: Vec<SkeletonEntry>,

    /// Observations from previous sessions (if any).
    pub session_memories: Vec<MemoryEntry>,

    pub stats: CapsuleStats,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PivotEntry {
    pub fqn: String,
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub kind: String,
    pub body: String,
    pub token_estimate: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SkeletonEntry {
    pub fqn: String,
    pub file_path: String,
    pub start_line: u32,
    pub kind: String,
    pub skeleton: String,
    pub token_estimate: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub content: String,
    pub symbol_fqn: Option<String>,
    pub is_stale: bool,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CapsuleStats {
    pub pivot_count: usize,
    pub skeleton_count: usize,
    pub memory_count: usize,
    pub total_tokens: u32,
    pub budget_tokens: u32,
    pub budget_used_pct: f32,
}

/// Assemble a token-budgeted capsule from ranked results.
///
/// `pivots` are the most relevant symbols (returned with full body).
/// `adjacent` are nearby symbols (returned as skeletons).
/// `memories` are past session observations to surface.
/// `budget` is the max token count for the entire capsule.
/// `query` is used for semantic chunking of large pivot bodies.
pub fn build_capsule(
    pivots: Vec<&Symbol>,
    adjacent: Vec<&Symbol>,
    memories: Vec<MemoryEntry>,
    budget: u32,
    query: Option<&str>,
) -> Capsule {
    let mut remaining = budget;
    let mut pivot_entries = Vec::new();
    let mut skeleton_entries = Vec::new();

    // 1. Reserve ~15% of budget for memories
    let memory_budget = (budget as f32 * 0.15) as u32;
    remaining = remaining.saturating_sub(memory_budget);

    // 2. Fit pivots (full body) — most important, spend budget here first
    // Large bodies are semantically chunked to the query-relevant portion.
    const CHUNK_THRESHOLD_TOKENS: u32 = 300;
    for sym in &pivots {
        let raw_tokens = sym.token_estimate();
        // Chunk if body is large and we have a query to guide selection
        let (body, tokens) = if raw_tokens > CHUNK_THRESHOLD_TOKENS {
            if let Some(q) = query {
                let chunked =
                    chunk_for_query(&sym.body, q, remaining.min(CHUNK_THRESHOLD_TOKENS * 2));
                let t = estimate_tokens(&chunked);
                (chunked, t)
            } else {
                (sym.body.clone(), raw_tokens)
            }
        } else {
            (sym.body.clone(), raw_tokens)
        };

        if tokens > remaining {
            // Try to fit as a skeleton instead
            let skel_tokens = sym.skeleton_token_estimate();
            if skel_tokens <= remaining {
                skeleton_entries.push(sym_to_skeleton(sym));
                remaining = remaining.saturating_sub(skel_tokens);
            }
            continue;
        }
        pivot_entries.push(sym_to_pivot_with_body(sym, body, tokens));
        remaining = remaining.saturating_sub(tokens);
    }

    // 3. Fit adjacent as skeletons
    for sym in &adjacent {
        let skel_tokens = sym.skeleton_token_estimate();
        if skel_tokens > remaining {
            break;
        }
        skeleton_entries.push(sym_to_skeleton(sym));
        remaining = remaining.saturating_sub(skel_tokens);
    }

    // 4. Fit memories within their budget
    let mut memory_entries = Vec::new();
    let mut mem_remaining = memory_budget;
    for mem in memories {
        let tokens = estimate_tokens(&mem.content);
        if tokens > mem_remaining {
            break;
        }
        mem_remaining = mem_remaining.saturating_sub(tokens);
        memory_entries.push(mem);
    }

    let total_tokens = budget - remaining;
    let stats = CapsuleStats {
        pivot_count: pivot_entries.len(),
        skeleton_count: skeleton_entries.len(),
        memory_count: memory_entries.len(),
        total_tokens,
        budget_tokens: budget,
        budget_used_pct: total_tokens as f32 / budget as f32 * 100.0,
    };

    Capsule {
        pivots: pivot_entries,
        skeletons: skeleton_entries,
        session_memories: memory_entries,
        stats,
    }
}

/// Format the capsule as a markdown string suitable for injecting into a prompt.
pub fn format_capsule(capsule: &Capsule) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "## codesurgeon context capsule\n\
         > {} pivots · {} skeletons · {} memories · ~{} tokens ({:.0}% of budget)\n\n",
        capsule.stats.pivot_count,
        capsule.stats.skeleton_count,
        capsule.stats.memory_count,
        capsule.stats.total_tokens,
        capsule.stats.budget_used_pct,
    ));

    if !capsule.pivots.is_empty() {
        out.push_str("### Pivot files (full source)\n\n");
        for p in &capsule.pivots {
            out.push_str(&format!(
                "#### `{}` ({}:{}-{})\n```\n{}\n```\n\n",
                p.fqn, p.file_path, p.start_line, p.end_line, p.body
            ));
        }
    }

    if !capsule.skeletons.is_empty() {
        out.push_str("### Related symbols (skeletons)\n\n");
        for s in &capsule.skeletons {
            out.push_str(&format!(
                "- `{}` @ `{}:{}`\n  ```\n  {}\n  ```\n",
                s.fqn,
                s.file_path,
                s.start_line,
                s.skeleton.replace('\n', "\n  ")
            ));
        }
        out.push('\n');
    }

    if !capsule.session_memories.is_empty() {
        out.push_str("### Session memory\n\n");
        for m in &capsule.session_memories {
            let stale_tag = if m.is_stale { " ⚠️ stale" } else { "" };
            let sym_tag = m
                .symbol_fqn
                .as_deref()
                .map(|f| format!(" (re: `{}`)", f))
                .unwrap_or_default();
            out.push_str(&format!(
                "- {}{}{}: {}\n",
                m.created_at, sym_tag, stale_tag, m.content
            ));
        }
        out.push('\n');
    }

    out
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sym_to_pivot_with_body(sym: &Symbol, body: String, token_estimate: u32) -> PivotEntry {
    PivotEntry {
        fqn: sym.fqn.clone(),
        file_path: sym.file_path.clone(),
        start_line: sym.start_line,
        end_line: sym.end_line,
        kind: sym.kind.to_string(),
        body,
        token_estimate,
    }
}

/// Semantic chunking: given a large function body and a query,
/// return the most query-relevant contiguous window of lines.
///
/// Splits the body into overlapping windows of ~`max_tokens` tokens,
/// scores each window against query terms, and returns the best window.
pub fn chunk_for_query(body: &str, query: &str, max_tokens: u32) -> String {
    let lines: Vec<&str> = body.lines().collect();
    if lines.is_empty() {
        return body.to_string();
    }

    // Tokenise the query into lowercase terms
    let query_terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(|t| t.to_lowercase())
        .collect();
    if query_terms.is_empty() {
        return body.to_string();
    }

    let max_chars = (max_tokens * 4) as usize;

    // Build windows: greedily fill each window up to max_chars, advancing by half
    let mut windows: Vec<(usize, usize)> = Vec::new(); // (start_line, end_line) exclusive
    let mut start = 0;
    while start < lines.len() {
        let mut end = start;
        let mut chars = 0;
        while end < lines.len() && chars + lines[end].len() < max_chars {
            chars += lines[end].len() + 1;
            end += 1;
        }
        if end == start {
            end = (start + 1).min(lines.len());
        }
        windows.push((start, end));
        // advance by half the window so windows overlap
        let step = ((end - start) / 2).max(1);
        start += step;
        if end == lines.len() {
            break;
        }
    }

    if windows.len() <= 1 {
        return body.to_string();
    }

    // Score each window: count query term occurrences (case-insensitive)
    let best_start = windows
        .iter()
        .max_by_key(|(s, e)| {
            let window_text = lines[*s..*e].join("\n").to_lowercase();
            query_terms
                .iter()
                .map(|t| window_text.matches(t.as_str()).count())
                .sum::<usize>()
        })
        .map(|(s, _)| *s)
        .unwrap_or(0);

    // Return the best window, always including the first line (signature)
    let mut result_lines: Vec<&str> = Vec::new();
    if best_start > 0 {
        result_lines.push(lines[0]); // always include signature
        result_lines.push("  // ... (lines omitted) ...");
    }
    let (_, best_end) = windows
        .iter()
        .find(|(s, _)| *s == best_start)
        .copied()
        .unwrap();
    result_lines.extend_from_slice(&lines[best_start..best_end]);
    if best_end < lines.len() {
        result_lines.push("  // ... (lines omitted) ...");
    }
    result_lines.join("\n")
}

fn sym_to_skeleton(sym: &Symbol) -> SkeletonEntry {
    let skeleton = skeletonize(sym);
    let token_estimate = (skeleton.len() / 4) as u32;
    SkeletonEntry {
        fqn: sym.fqn.clone(),
        file_path: sym.file_path.clone(),
        start_line: sym.start_line,
        kind: sym.kind.to_string(),
        skeleton,
        token_estimate,
    }
}

/// Rough token estimate: characters / 4.
pub fn estimate_tokens(text: &str) -> u32 {
    (text.len() / 4) as u32
}
