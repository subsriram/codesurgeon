//! Edge extraction: builds the dependency graph edges from indexed symbols.
//!
//! Four edge types are extracted in a single pass over all symbols:
//! - `Imports`    — import/use statements referencing other symbols by name
//! - `Implements` — `impl Trait for Type` relationships
//! - `Calls`      — function bodies that call other named functions
//! - `References` — callables that mention a type in their signature

use crate::language::Language;
use crate::symbol::{Edge, EdgeKind, Symbol, SymbolKind};
use std::collections::{HashMap, HashSet};

// ── Public API ────────────────────────────────────────────────────────────────

/// Build import edges between already-indexed symbols.
/// Parses import statement text to extract actual imported names.
pub fn extract_import_edges(symbols: &[Symbol]) -> Vec<Edge> {
    let mut name_to_ids: HashMap<&str, Vec<u64>> = HashMap::new();
    for sym in symbols {
        name_to_ids.entry(sym.name.as_str()).or_default().push(sym.id);
    }

    let mut edges = Vec::new();
    for sym in symbols {
        if sym.kind != SymbolKind::Import {
            continue;
        }
        for name in extract_imported_names(sym) {
            if let Some(targets) = name_to_ids.get(name.as_str()) {
                for &target_id in targets {
                    if target_id != sym.id {
                        edges.push(
                            Edge::new(sym.id, target_id, EdgeKind::Imports)
                                .with_label(name.clone()),
                        );
                    }
                }
            }
        }
    }
    edges
}

/// Build Implements edges from `impl Trait for Type` symbols.
pub fn extract_impl_edges(symbols: &[Symbol]) -> Vec<Edge> {
    let mut name_to_ids: HashMap<&str, Vec<u64>> = HashMap::new();
    for sym in symbols {
        name_to_ids.entry(sym.name.as_str()).or_default().push(sym.id);
    }

    let mut edges = Vec::new();
    for sym in symbols {
        if sym.kind != SymbolKind::Impl {
            continue;
        }
        let label = sym.name.trim_start_matches("impl::");
        if let Some((trait_part, type_part)) = label.split_once(" for ") {
            let trait_name = trait_part.trim().split('<').next().unwrap_or("").trim();
            let type_name = type_part.trim().split('<').next().unwrap_or("").trim();
            if let (Some(type_ids), Some(trait_ids)) =
                (name_to_ids.get(type_name), name_to_ids.get(trait_name))
            {
                for &type_id in type_ids {
                    for &trait_id in trait_ids {
                        if type_id != trait_id {
                            edges.push(
                                Edge::new(type_id, trait_id, EdgeKind::Implements)
                                    .with_label(label.to_string()),
                            );
                        }
                    }
                }
            }
        }
    }
    edges
}

/// Build Calls edges by scanning function bodies for `identifier(args)` patterns.
/// Edge labels include a short args snippet for call-site annotation.
pub fn extract_call_edges(symbols: &[Symbol]) -> Vec<Edge> {
    let mut name_to_ids: HashMap<&str, Vec<u64>> = HashMap::new();
    for sym in symbols {
        name_to_ids.entry(sym.name.as_str()).or_default().push(sym.id);
        if let Some(simple) = sym.name.rsplit("::").next() {
            if simple != sym.name {
                name_to_ids.entry(simple).or_default().push(sym.id);
            }
        }
    }

    let mut edges = Vec::new();
    for sym in symbols {
        if !sym.kind.is_callable() || sym.body.len() < 20 {
            continue;
        }
        let mut seen: HashMap<u64, String> = HashMap::new();
        for (callee_name, args_snippet) in calls_in_body(&sym.body) {
            if let Some(targets) = name_to_ids.get(callee_name.as_str()) {
                for &target_id in targets {
                    if target_id != sym.id {
                        seen.entry(target_id).or_insert_with(|| {
                            if args_snippet.is_empty() {
                                callee_name.clone()
                            } else {
                                format!("{}({})", callee_name, args_snippet)
                            }
                        });
                    }
                }
            }
        }
        for (target_id, label) in seen {
            edges.push(Edge::new(sym.id, target_id, EdgeKind::Calls).with_label(label));
        }
    }
    edges
}

/// Build type-flow edges: functions that mention a type in their signature depend on that type.
/// Creates `References` edges from callables to the type symbols they reference.
pub fn extract_type_flow_edges(symbols: &[Symbol]) -> Vec<Edge> {
    let mut type_name_to_ids: HashMap<&str, Vec<u64>> = HashMap::new();
    for sym in symbols {
        if sym.kind.is_type_definition() {
            type_name_to_ids.entry(sym.name.as_str()).or_default().push(sym.id);
        }
    }
    if type_name_to_ids.is_empty() {
        return vec![];
    }

    let mut edges = Vec::new();
    for sym in symbols {
        if !sym.kind.is_callable() {
            continue;
        }
        let sig = sym.body.lines().next().unwrap_or("").trim();
        for type_name in pascal_case_identifiers(sig) {
            if let Some(type_ids) = type_name_to_ids.get(type_name.as_str()) {
                for &type_id in type_ids {
                    if type_id != sym.id {
                        edges.push(
                            Edge::new(sym.id, type_id, EdgeKind::References)
                                .with_label(type_name.clone()),
                        );
                    }
                }
            }
        }
    }
    edges
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Parse an import symbol's body text to get the actual imported names.
fn extract_imported_names(sym: &Symbol) -> Vec<String> {
    let text = sym.body.trim();
    match sym.language {
        Language::Python => {
            if let Some(rest) = text.strip_prefix("from ") {
                if let Some(import_part) = rest.split(" import ").nth(1) {
                    return import_part
                        .split(',')
                        .map(|s| {
                            s.trim()
                                .trim_matches(|c| c == '(' || c == ')' || c == '\\')
                                .split_whitespace()
                                .next()
                                .unwrap_or("")
                                .to_string()
                        })
                        .filter(|s| !s.is_empty() && s != "*")
                        .collect();
                }
            }
            if let Some(rest) = text.strip_prefix("import ") {
                return rest
                    .split(',')
                    .map(|s| s.split_whitespace().next().unwrap_or("").to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            vec![]
        }
        Language::TypeScript | Language::Tsx | Language::JavaScript | Language::Jsx => {
            if let (Some(start), Some(end)) = (text.find('{'), text.find('}')) {
                return text[start + 1..end]
                    .split(',')
                    .map(|s| s.split_whitespace().next().unwrap_or("").to_string())
                    .filter(|s| !s.is_empty() && s != "type")
                    .collect();
            }
            if let Some(rest) = text.strip_prefix("import ") {
                let first = rest.split_whitespace().next().unwrap_or("");
                if !first.is_empty() && first != "{" && first != "*" && first != "type" {
                    return vec![first.to_string()];
                }
            }
            vec![]
        }
        Language::Rust => {
            let path = text.trim_start_matches("use ").trim_end_matches(';').trim();
            if let (Some(start), Some(end)) = (path.find('{'), path.rfind('}')) {
                return path[start + 1..end]
                    .split(',')
                    .map(|s| s.split_whitespace().next().unwrap_or("").to_string())
                    .filter(|s| !s.is_empty() && s != "*" && s != "self")
                    .collect();
            }
            if let Some(last) = path.split("::").last() {
                let name = last.trim().to_string();
                if !name.is_empty() && name != "*" && name != "self" {
                    return vec![name];
                }
            }
            vec![]
        }
        _ => vec![],
    }
}

/// Extract call sites from source text.
/// Returns `(callee_name, args_snippet)` pairs.
fn calls_in_body(body: &str) -> Vec<(String, String)> {
    const SKIP: &[&str] = &[
        "if", "for", "while", "match", "fn", "let", "mut", "pub", "use", "mod", "struct",
        "enum", "impl", "trait", "type", "async", "await", "return", "where", "loop",
        "continue", "break", "Some", "None", "Ok", "Err", "Box", "Vec", "HashMap", "HashSet",
        "BTreeMap", "String", "Option", "Result", "Arc", "Mutex", "RwLock", "format",
        "println", "eprintln", "print", "eprint", "vec", "assert", "panic", "todo",
        "unimplemented",
    ];

    let mut calls = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();
    let bytes = body.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i] == b'(' && i > 0 {
            let mut j = i.saturating_sub(1);
            while j > 0 && bytes[j] == b' ' {
                j -= 1;
            }
            if bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_' {
                let end = j + 1;
                while j > 0 && (bytes[j - 1].is_ascii_alphanumeric() || bytes[j - 1] == b'_') {
                    j -= 1;
                }
                if let Ok(name) = std::str::from_utf8(&bytes[j..end]) {
                    if name.len() > 2 && !SKIP.contains(&name) && !seen_names.contains(name) {
                        seen_names.insert(name.to_string());
                        let args_snippet = extract_args_snippet(bytes, i + 1, 60);
                        calls.push((name.to_string(), args_snippet));
                    }
                }
            }
        }
        i += 1;
    }
    calls
}

/// Capture up to `max_chars` of argument text starting at `start` (just after the opening `(`).
/// Stops at the matching `)` or at `max_chars`.
pub(crate) fn extract_args_snippet(bytes: &[u8], start: usize, max_chars: usize) -> String {
    let len = bytes.len();
    let mut depth = 1i32;
    let mut end = start;
    while end < len && depth > 0 {
        match bytes[end] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {}
        }
        if depth > 0 {
            end += 1;
        }
    }
    let raw = std::str::from_utf8(&bytes[start..end.min(len)])
        .unwrap_or("")
        .trim();
    if raw.len() <= max_chars {
        raw.to_string()
    } else {
        let mut boundary = max_chars;
        while boundary > 0 && !raw.is_char_boundary(boundary) {
            boundary -= 1;
        }
        format!("{}…", &raw[..boundary])
    }
}

/// Extract PascalCase identifiers from a string (type names in signatures).
fn pascal_case_identifiers(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();
    while i < len {
        if bytes[i].is_ascii_uppercase() {
            let start = i;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            if i - start >= 2 {
                if let Ok(name) = std::str::from_utf8(&bytes[start..i]) {
                    result.push(name.to_string());
                }
            }
        } else {
            i += 1;
        }
    }
    result
}
