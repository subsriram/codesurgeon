use crate::language::{detect_language, Language};
use crate::symbol::{Edge, EdgeKind, Symbol, SymbolKind};
use anyhow::Result;
use std::path::Path;
use tree_sitter::{Node, Parser};

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Parse a single file and return all symbols found in it.
pub fn index_file(workspace_root: &Path, abs_path: &Path, content: &str) -> Result<Vec<Symbol>> {
    let lang = match detect_language(abs_path) {
        Some(l) => l,
        None => return Ok(vec![]),
    };

    let rel_path = abs_path
        .strip_prefix(workspace_root)
        .unwrap_or(abs_path)
        .to_string_lossy()
        .to_string();

    match lang {
        Language::Python => extract_python(&rel_path, content),
        Language::TypeScript | Language::Tsx => extract_ts_js(&rel_path, content, lang),
        Language::JavaScript | Language::Jsx => extract_ts_js(&rel_path, content, lang),
        Language::Shell => extract_shell(&rel_path, content),
        Language::Html => extract_html(&rel_path, content),
        Language::Rust => extract_rust(&rel_path, content),
        Language::Swift => extract_swift(&rel_path, content),
        Language::Sql => extract_sql(&rel_path, content),
        Language::Markdown => extract_markdown(&rel_path, content),
    }
}

/// Build import edges between already-indexed symbols.
/// Parses import statement text to extract actual imported names.
pub fn extract_import_edges(symbols: &[Symbol]) -> Vec<Edge> {
    let mut name_to_ids: std::collections::HashMap<&str, Vec<u64>> =
        std::collections::HashMap::new();
    for sym in symbols {
        name_to_ids
            .entry(sym.name.as_str())
            .or_default()
            .push(sym.id);
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

/// Parse an import symbol's body text to get the actual imported names.
fn extract_imported_names(sym: &Symbol) -> Vec<String> {
    let text = sym.body.trim();
    match sym.language {
        Language::Python => {
            // "from foo import Bar, Baz" → ["Bar", "Baz"]
            // "import os, sys" → ["os", "sys"]
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
                    .map(|s| s.trim().split_whitespace().next().unwrap_or("").to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            vec![]
        }
        Language::TypeScript | Language::Tsx | Language::JavaScript | Language::Jsx => {
            // "import { Foo, Bar } from './module'" → ["Foo", "Bar"]
            // "import DefaultExport from './x'" → ["DefaultExport"]
            if let (Some(start), Some(end)) = (text.find('{'), text.find('}')) {
                return text[start + 1..end]
                    .split(',')
                    .map(|s| {
                        // "foo as bar" → "foo"
                        s.trim().split_whitespace().next().unwrap_or("").to_string()
                    })
                    .filter(|s| !s.is_empty() && s != "type")
                    .collect();
            }
            // "import Foo from '...'" or "import * as Foo from '...'"
            if let Some(rest) = text.strip_prefix("import ") {
                let first = rest.trim().split_whitespace().next().unwrap_or("");
                if !first.is_empty() && first != "{" && first != "*" && first != "type" {
                    return vec![first.to_string()];
                }
            }
            vec![]
        }
        Language::Rust => {
            // "use foo::bar::{Bar, Baz}" → ["Bar", "Baz"]
            // "use foo::bar::Baz" → ["Baz"]
            let path = text.trim_start_matches("use ").trim_end_matches(';').trim();
            if let (Some(start), Some(end)) = (path.find('{'), path.rfind('}')) {
                return path[start + 1..end]
                    .split(',')
                    .map(|s| {
                        // "Foo as F" → "Foo"
                        s.trim().split_whitespace().next().unwrap_or("").to_string()
                    })
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

/// Build Implements edges from `impl Trait for Type` symbols.
pub fn extract_impl_edges(symbols: &[Symbol]) -> Vec<Edge> {
    let mut name_to_ids: std::collections::HashMap<&str, Vec<u64>> =
        std::collections::HashMap::new();
    for sym in symbols {
        name_to_ids
            .entry(sym.name.as_str())
            .or_default()
            .push(sym.id);
    }

    let mut edges = Vec::new();

    for sym in symbols {
        if sym.kind != SymbolKind::Impl {
            continue;
        }
        // Name is "impl::TraitName for TypeName" or "impl::TypeName"
        let label = sym.name.trim_start_matches("impl::");
        if let Some((trait_part, type_part)) = label.split_once(" for ") {
            // Strip generic parameters: "Display" not "Display<T>"
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
    let mut name_to_ids: std::collections::HashMap<&str, Vec<u64>> =
        std::collections::HashMap::new();
    for sym in symbols {
        name_to_ids
            .entry(sym.name.as_str())
            .or_default()
            .push(sym.id);
        // Also index by simple name (after last "::") so that method call sites
        // like `foo()` resolve to `Type::foo` entries in the map.
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
        // Deduplicate: one edge per (caller, callee) pair with the first args snippet seen
        let mut seen: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
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

/// Extract call sites from source text.
/// Returns `(callee_name, args_snippet)` pairs.
/// `args_snippet` is a truncated view of the argument text (≤60 chars).
fn calls_in_body(body: &str) -> Vec<(String, String)> {
    const SKIP: &[&str] = &[
        "if",
        "for",
        "while",
        "match",
        "fn",
        "let",
        "mut",
        "pub",
        "use",
        "mod",
        "struct",
        "enum",
        "impl",
        "trait",
        "type",
        "async",
        "await",
        "return",
        "where",
        "loop",
        "continue",
        "break",
        "Some",
        "None",
        "Ok",
        "Err",
        "Box",
        "Vec",
        "HashMap",
        "HashSet",
        "BTreeMap",
        "String",
        "Option",
        "Result",
        "Arc",
        "Mutex",
        "RwLock",
        "format",
        "println",
        "eprintln",
        "print",
        "eprint",
        "vec",
        "assert",
        "panic",
        "todo",
        "unimplemented",
    ];

    let mut calls = Vec::new();
    let mut seen_names = std::collections::HashSet::new();
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
                        // Capture args snippet: scan forward, balance parens, take first 60 chars
                        let args_start = i + 1;
                        let args_snippet = extract_args_snippet(bytes, args_start, 60);
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
fn extract_args_snippet(bytes: &[u8], start: usize, max_chars: usize) -> String {
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
        format!("{}…", &raw[..max_chars])
    }
}

/// Build type-flow edges: functions that mention a type in their signature depend on that type.
/// Creates `References` edges from callables to the type symbols they reference.
pub fn extract_type_flow_edges(symbols: &[Symbol]) -> Vec<Edge> {
    // Build a map from type name → symbol IDs for all type-defining symbols
    let mut type_name_to_ids: std::collections::HashMap<&str, Vec<u64>> =
        std::collections::HashMap::new();
    for sym in symbols {
        if sym.kind.is_type_definition() {
            type_name_to_ids
                .entry(sym.name.as_str())
                .or_default()
                .push(sym.id);
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
        // Extract the signature line (everything up to the first `{` or end of first line)
        let sig = sym.body.lines().next().unwrap_or("").trim();
        // Find PascalCase identifiers in the signature that match known types
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

// ──────────────────────────────────────────────────────────────────────────────
// Helper: parse with tree-sitter and walk nodes
// ──────────────────────────────────────────────────────────────────────────────

fn make_parser(lang: &Language) -> Option<Parser> {
    let ts_lang = lang.tree_sitter_language()?;
    let mut parser = Parser::new();
    parser.set_language(&ts_lang).ok()?;
    Some(parser)
}

fn node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

fn node_lines(node: &Node) -> (u32, u32) {
    (
        node.start_position().row as u32 + 1,
        node.end_position().row as u32 + 1,
    )
}

/// Extract the first string literal child (docstring) from a block node.
fn extract_docstring(node: &Node, source: &str) -> Option<String> {
    // Only inspect the very first child statement
    let mut cursor = node.walk();
    let first = node.children(&mut cursor).next()?;
    if first.kind() == "expression_statement" {
        if let Some(inner) = first.child(0) {
            if inner.kind() == "string" || inner.kind() == "string_literal" {
                let raw = node_text(&inner, source);
                let trimmed = raw.trim_matches('"').trim_matches('\'').trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    None
}

// ──────────────────────────────────────────────────────────────────────────────
// Python
// ──────────────────────────────────────────────────────────────────────────────

fn extract_python(file_path: &str, source: &str) -> Result<Vec<Symbol>> {
    let mut parser = match make_parser(&Language::Python) {
        Some(p) => p,
        None => return Ok(vec![]),
    };

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("parse failed"))?;
    let root = tree.root_node();
    let mut symbols = Vec::new();

    walk_python_node(&root, source, file_path, None, &mut symbols);

    Ok(symbols)
}

fn walk_python_node(
    node: &Node,
    source: &str,
    file_path: &str,
    parent_class: Option<&str>,
    symbols: &mut Vec<Symbol>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_definition" | "decorated_definition" => {
                let func_node = if child.kind() == "decorated_definition" {
                    child.child_by_field_name("definition").unwrap_or(child)
                } else {
                    child
                };

                if func_node.kind() == "function_definition" {
                    let name_node = func_node.child_by_field_name("name");
                    let name = name_node
                        .map(|n| node_text(&n, source))
                        .unwrap_or("_unknown");

                    // Check if async
                    let is_async = func_node
                        .child(0)
                        .map(|c| node_text(&c, source) == "async")
                        .unwrap_or(false);

                    let kind = if parent_class.is_some() {
                        if is_async {
                            SymbolKind::AsyncMethod
                        } else {
                            SymbolKind::Method
                        }
                    } else {
                        if is_async {
                            SymbolKind::AsyncFunction
                        } else {
                            SymbolKind::Function
                        }
                    };

                    let (start, end) = node_lines(&child);
                    let body = node_text(&child, source).to_string();

                    // Signature = everything up to the colon on the def line
                    let params = func_node
                        .child_by_field_name("parameters")
                        .map(|p| node_text(&p, source))
                        .unwrap_or("()");
                    let return_type = func_node
                        .child_by_field_name("return_type")
                        .map(|r| format!(" -> {}", node_text(&r, source)))
                        .unwrap_or_default();
                    let sig = format!("def {}{}{}:", name, params, return_type);

                    // Docstring from body block
                    let docstring = func_node
                        .child_by_field_name("body")
                        .and_then(|b| extract_docstring(&b, source));

                    // Qualify name if inside a class
                    let qualified = match parent_class {
                        Some(cls) => format!("{}::{}", cls, name),
                        None => name.to_string(),
                    };

                    symbols.push(Symbol::new(
                        file_path,
                        &qualified,
                        kind,
                        start,
                        end,
                        sig,
                        docstring,
                        body,
                        Language::Python,
                    ));

                    // Recurse into function body for nested functions
                    if let Some(body_node) = func_node.child_by_field_name("body") {
                        walk_python_node(&body_node, source, file_path, parent_class, symbols);
                    }
                }
            }

            "class_definition" => {
                let name_node = child.child_by_field_name("name");
                let name = name_node
                    .map(|n| node_text(&n, source))
                    .unwrap_or("_unknown");
                let (start, end) = node_lines(&child);
                let body = node_text(&child, source).to_string();

                // Superclasses
                let superclasses = child
                    .child_by_field_name("superclasses")
                    .map(|s| format!("({})", node_text(&s, source)))
                    .unwrap_or_default();
                let sig = format!("class {}{}:", name, superclasses);

                let docstring = child
                    .child_by_field_name("body")
                    .and_then(|b| extract_docstring(&b, source));

                symbols.push(Symbol::new(
                    file_path,
                    name,
                    SymbolKind::Class,
                    start,
                    end,
                    sig,
                    docstring,
                    body,
                    Language::Python,
                ));

                // Recurse for methods
                if let Some(body_node) = child.child_by_field_name("body") {
                    walk_python_node(&body_node, source, file_path, Some(name), symbols);
                }
            }

            "import_statement" | "import_from_statement" => {
                let (start, end) = node_lines(&child);
                let text = node_text(&child, source).to_string();
                // Use the whole import line as name for now
                symbols.push(Symbol::new(
                    file_path,
                    &text.lines().next().unwrap_or("import").trim().to_string(),
                    SymbolKind::Import,
                    start,
                    end,
                    text.clone(),
                    None,
                    text,
                    Language::Python,
                ));
            }

            _ => {
                // Recurse into top-level blocks
                walk_python_node(&child, source, file_path, parent_class, symbols);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TypeScript / JavaScript (shared grammar)
// ──────────────────────────────────────────────────────────────────────────────

fn extract_ts_js(file_path: &str, source: &str, lang: Language) -> Result<Vec<Symbol>> {
    let mut parser = match make_parser(&lang) {
        Some(p) => p,
        None => return Ok(vec![]),
    };

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("parse failed"))?;
    let root = tree.root_node();
    let mut symbols = Vec::new();

    walk_ts_node(&root, source, file_path, lang, None, &mut symbols);

    Ok(symbols)
}

fn walk_ts_node(
    node: &Node,
    source: &str,
    file_path: &str,
    lang: Language,
    parent_class: Option<&str>,
    symbols: &mut Vec<Symbol>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_declaration" | "function" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<anonymous>".to_string());

                let is_async = node_text(&child, source).trim_start().starts_with("async");
                let kind = if is_async {
                    SymbolKind::AsyncFunction
                } else {
                    SymbolKind::Function
                };

                push_ts_symbol(
                    &child,
                    source,
                    file_path,
                    lang,
                    &name,
                    kind,
                    parent_class,
                    symbols,
                );
            }

            "method_definition" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<method>".to_string());

                let is_async = child
                    .child_by_field_name("value")
                    .map(|v| node_text(&v, source).trim_start().starts_with("async"))
                    .unwrap_or(false);

                let kind = if is_async {
                    SymbolKind::AsyncMethod
                } else {
                    SymbolKind::Method
                };
                let qualified = match parent_class {
                    Some(cls) => format!("{}::{}", cls, name),
                    None => name.clone(),
                };

                push_ts_symbol(
                    &child,
                    source,
                    file_path,
                    lang,
                    &qualified,
                    kind,
                    parent_class,
                    symbols,
                );
            }

            "class_declaration" | "class" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<class>".to_string());

                push_ts_symbol(
                    &child,
                    source,
                    file_path,
                    lang,
                    &name,
                    SymbolKind::Class,
                    None,
                    symbols,
                );

                // Recurse for methods
                if let Some(body) = child.child_by_field_name("body") {
                    walk_ts_node(&body, source, file_path, lang, Some(&name), symbols);
                }
            }

            // TypeScript-specific
            "interface_declaration" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<interface>".to_string());
                push_ts_symbol(
                    &child,
                    source,
                    file_path,
                    lang,
                    &name,
                    SymbolKind::Interface,
                    None,
                    symbols,
                );
            }

            "type_alias_declaration" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<type>".to_string());
                push_ts_symbol(
                    &child,
                    source,
                    file_path,
                    lang,
                    &name,
                    SymbolKind::TypeAlias,
                    None,
                    symbols,
                );
            }

            "enum_declaration" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<enum>".to_string());
                push_ts_symbol(
                    &child,
                    source,
                    file_path,
                    lang,
                    &name,
                    SymbolKind::Enum,
                    None,
                    symbols,
                );
            }

            "import_statement" | "import_declaration" => {
                let (start, end) = node_lines(&child);
                let text = node_text(&child, source).to_string();
                let name = text.lines().next().unwrap_or("import").trim().to_string();
                symbols.push(Symbol::new(
                    file_path,
                    &name,
                    SymbolKind::Import,
                    start,
                    end,
                    text.clone(),
                    None,
                    text,
                    lang,
                ));
            }

            "lexical_declaration" | "variable_declaration" => {
                // Capture top-level const/let arrow functions as functions
                // e.g. const foo = async (x: T) => { ... }
                if let Some(decl) = child.child_by_field_name("declarator").or_else(|| {
                    let mut c = child.walk();
                    let found = child
                        .children(&mut c)
                        .find(|n| n.kind() == "variable_declarator");
                    found
                }) {
                    if let Some(value) = decl.child_by_field_name("value") {
                        if matches!(value.kind(), "arrow_function" | "function") {
                            if let Some(name_node) = decl.child_by_field_name("name") {
                                let name = node_text(&name_node, source).to_string();
                                let is_async =
                                    node_text(&value, source).trim_start().starts_with("async");
                                let kind = if is_async {
                                    SymbolKind::AsyncFunction
                                } else {
                                    SymbolKind::Function
                                };
                                push_ts_symbol(
                                    &child, source, file_path, lang, &name, kind, None, symbols,
                                );
                            }
                        }
                    }
                }
            }

            _ => {
                walk_ts_node(&child, source, file_path, lang, parent_class, symbols);
            }
        }
    }
}

fn push_ts_symbol(
    node: &Node,
    source: &str,
    file_path: &str,
    lang: Language,
    name: &str,
    kind: SymbolKind,
    _parent_class: Option<&str>,
    symbols: &mut Vec<Symbol>,
) {
    let (start, end) = node_lines(node);
    let body = node_text(node, source).to_string();
    // Signature = first line
    let sig = body.lines().next().unwrap_or("").trim().to_string();

    // Look for JSDoc comment immediately before this node
    let docstring = extract_jsdoc(node, source);

    symbols.push(Symbol::new(
        file_path, name, kind, start, end, sig, docstring, body, lang,
    ));
}

fn extract_jsdoc(node: &Node, source: &str) -> Option<String> {
    // A JSDoc comment immediately precedes the node.
    // tree-sitter stores comments as siblings; we check the prev_named_sibling.
    let prev = node.prev_named_sibling()?;
    if prev.kind() == "comment" {
        let text = node_text(&prev, source);
        if text.starts_with("/**") {
            let inner = text
                .trim_start_matches("/**")
                .trim_end_matches("*/")
                .lines()
                .map(|l| l.trim_start_matches('*').trim())
                .collect::<Vec<_>>()
                .join(" ");
            return Some(inner.trim().to_string());
        }
    }
    None
}

// ──────────────────────────────────────────────────────────────────────────────
// Shell
// ──────────────────────────────────────────────────────────────────────────────

fn extract_shell(file_path: &str, source: &str) -> Result<Vec<Symbol>> {
    let mut parser = match make_parser(&Language::Shell) {
        Some(p) => p,
        None => return Ok(vec![]),
    };

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("parse failed"))?;
    let root = tree.root_node();
    let mut symbols = Vec::new();

    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() == "function_definition" {
            let name = child
                .child_by_field_name("name")
                .map(|n| node_text(&n, source).to_string())
                .unwrap_or_else(|| "<func>".to_string());

            let (start, end) = node_lines(&child);
            let body = node_text(&child, source).to_string();
            let sig = format!("{}()", name);

            symbols.push(Symbol::new(
                file_path,
                &name,
                SymbolKind::Function,
                start,
                end,
                sig,
                None,
                body,
                Language::Shell,
            ));
        }
    }

    Ok(symbols)
}

// ──────────────────────────────────────────────────────────────────────────────
// HTML
// ──────────────────────────────────────────────────────────────────────────────

fn extract_html(file_path: &str, source: &str) -> Result<Vec<Symbol>> {
    let mut parser = match make_parser(&Language::Html) {
        Some(p) => p,
        None => return Ok(vec![]),
    };

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("parse failed"))?;
    let root = tree.root_node();
    let mut symbols = Vec::new();

    // Walk and find <script> and <style> elements
    walk_html(&root, source, file_path, &mut symbols);

    Ok(symbols)
}

fn walk_html(node: &Node, source: &str, file_path: &str, symbols: &mut Vec<Symbol>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "script_element" {
            let (start, end) = node_lines(&child);
            let body = node_text(&child, source).to_string();
            // Count script blocks by line number
            let name = format!("script@{}", start);
            symbols.push(Symbol::new(
                file_path,
                &name,
                SymbolKind::ScriptBlock,
                start,
                end,
                format!("<script> block at line {}", start),
                None,
                body,
                Language::Html,
            ));
        } else if child.kind() == "style_element" {
            let (start, end) = node_lines(&child);
            let body = node_text(&child, source).to_string();
            let name = format!("style@{}", start);
            symbols.push(Symbol::new(
                file_path,
                &name,
                SymbolKind::StyleBlock,
                start,
                end,
                format!("<style> block at line {}", start),
                None,
                body,
                Language::Html,
            ));
        } else {
            walk_html(&child, source, file_path, symbols);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Rust
// ──────────────────────────────────────────────────────────────────────────────

fn extract_rust(file_path: &str, source: &str) -> Result<Vec<Symbol>> {
    let mut parser = match make_parser(&Language::Rust) {
        Some(p) => p,
        None => return Ok(vec![]),
    };

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("parse failed"))?;
    let root = tree.root_node();
    let mut symbols = Vec::new();

    walk_rust(&root, source, file_path, None, &mut symbols);

    Ok(symbols)
}

fn walk_rust(
    node: &Node,
    source: &str,
    file_path: &str,
    impl_context: Option<&str>,
    symbols: &mut Vec<Symbol>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_item" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<fn>".to_string());

                let qualified = match impl_context {
                    Some(ctx) => format!("{}::{}", ctx, name),
                    None => name.clone(),
                };

                let (start, end) = node_lines(&child);
                let body = node_text(&child, source).to_string();
                // Signature = everything up to the opening brace
                let sig = body
                    .find('{')
                    .map(|i| body[..i].trim().to_string())
                    .unwrap_or_else(|| body.lines().next().unwrap_or("").to_string());

                let docstring = extract_rust_doc(&child, source);

                symbols.push(Symbol::new(
                    file_path,
                    &qualified,
                    SymbolKind::Function,
                    start,
                    end,
                    sig,
                    docstring,
                    body,
                    Language::Rust,
                ));

                // Recurse for nested fns / closures (rare but valid)
                if let Some(body_node) = child.child_by_field_name("body") {
                    walk_rust(&body_node, source, file_path, impl_context, symbols);
                }
            }

            "struct_item" => {
                push_rust_item(&child, source, file_path, SymbolKind::Struct, symbols);
            }

            "enum_item" => {
                push_rust_item(&child, source, file_path, SymbolKind::Enum, symbols);
            }

            "trait_item" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<trait>".to_string());
                push_rust_item(&child, source, file_path, SymbolKind::Trait, symbols);
                // Recurse for default method impls
                if let Some(body) = child.child_by_field_name("body") {
                    walk_rust(&body, source, file_path, Some(&name), symbols);
                }
            }

            "impl_item" => {
                // impl Foo or impl Trait for Foo
                let type_name = child
                    .child_by_field_name("type")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<impl>".to_string());

                let trait_name = child
                    .child_by_field_name("trait")
                    .map(|n| format!("{} for ", node_text(&n, source)));

                let impl_label = match trait_name {
                    Some(t) => format!("{}{}", t, type_name),
                    None => type_name.clone(),
                };

                let (start, end) = node_lines(&child);
                let body = node_text(&child, source).to_string();
                let sig = body.lines().next().unwrap_or("").trim().to_string();

                symbols.push(Symbol::new(
                    file_path,
                    &format!("impl::{}", impl_label),
                    SymbolKind::Impl,
                    start,
                    end,
                    sig,
                    None,
                    body,
                    Language::Rust,
                ));

                // Recurse with impl type as context
                if let Some(body_node) = child.child_by_field_name("body") {
                    walk_rust(&body_node, source, file_path, Some(&type_name), symbols);
                }
            }

            "mod_item" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<mod>".to_string());

                let (start, end) = node_lines(&child);
                let body_text = node_text(&child, source).to_string();
                let sig = format!("mod {}", name);

                symbols.push(Symbol::new(
                    file_path,
                    &format!("mod::{}", name),
                    SymbolKind::Module,
                    start,
                    end,
                    sig,
                    None,
                    body_text,
                    Language::Rust,
                ));

                // Recurse into inline modules
                if let Some(body_node) = child.child_by_field_name("body") {
                    walk_rust(&body_node, source, file_path, None, symbols);
                }
            }

            "macro_definition" => {
                push_rust_item(&child, source, file_path, SymbolKind::Macro, symbols);
            }

            "use_declaration" => {
                let (start, end) = node_lines(&child);
                let text = node_text(&child, source).to_string();
                let name = text.lines().next().unwrap_or("use").trim().to_string();
                symbols.push(Symbol::new(
                    file_path,
                    &name,
                    SymbolKind::Import,
                    start,
                    end,
                    text.clone(),
                    None,
                    text,
                    Language::Rust,
                ));
            }

            _ => {
                walk_rust(&child, source, file_path, impl_context, symbols);
            }
        }
    }
}

fn push_rust_item(
    node: &Node,
    source: &str,
    file_path: &str,
    kind: SymbolKind,
    symbols: &mut Vec<Symbol>,
) {
    let name = node
        .child_by_field_name("name")
        .map(|n| node_text(&n, source).to_string())
        .unwrap_or_else(|| "<item>".to_string());

    let (start, end) = node_lines(node);
    let body = node_text(node, source).to_string();
    let sig = body
        .find('{')
        .or_else(|| body.find(';'))
        .map(|i| body[..i].trim().to_string())
        .unwrap_or_else(|| body.lines().next().unwrap_or("").to_string());

    let docstring = extract_rust_doc(node, source);

    symbols.push(Symbol::new(
        file_path,
        &name,
        kind,
        start,
        end,
        sig,
        docstring,
        body,
        Language::Rust,
    ));
}

fn extract_rust_doc(node: &Node, source: &str) -> Option<String> {
    let prev = node.prev_named_sibling()?;
    if prev.kind() == "line_comment" || prev.kind() == "block_comment" {
        let text = node_text(&prev, source);
        // Only /// doc comments
        if text.starts_with("///") || text.starts_with("/**") {
            let cleaned: String = text
                .lines()
                .map(|l| l.trim_start_matches('/').trim_start_matches('*').trim())
                .collect::<Vec<_>>()
                .join(" ");
            return Some(cleaned.trim().to_string());
        }
    }
    None
}

// ──────────────────────────────────────────────────────────────────────────────
// Swift (tree-sitter)
// ──────────────────────────────────────────────────────────────────────────────

fn extract_swift(file_path: &str, source: &str) -> Result<Vec<Symbol>> {
    let mut parser = match make_parser(&Language::Swift) {
        Some(p) => p,
        None => return Ok(vec![]),
    };

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("swift parse failed"))?;
    let root = tree.root_node();
    let mut symbols = Vec::new();

    walk_swift(&root, source, file_path, None, &mut symbols);

    Ok(symbols)
}

fn walk_swift(
    node: &Node,
    source: &str,
    file_path: &str,
    parent_class: Option<&str>,
    symbols: &mut Vec<Symbol>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_declaration" | "init_declaration" | "deinit_declaration" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| match child.kind() {
                        "init_declaration" => "init".to_string(),
                        "deinit_declaration" => "deinit".to_string(),
                        _ => "<func>".to_string(),
                    });

                let qualified = match parent_class {
                    Some(cls) => format!("{}::{}", cls, name),
                    None => name.clone(),
                };

                let (start, end) = node_lines(&child);
                let body = node_text(&child, source).to_string();
                let sig = body
                    .find('{')
                    .map(|i| body[..i].trim().to_string())
                    .unwrap_or_else(|| body.lines().next().unwrap_or("").to_string());

                let kind = if parent_class.is_some() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };

                symbols.push(Symbol::new(
                    file_path,
                    &qualified,
                    kind,
                    start,
                    end,
                    sig,
                    None,
                    body,
                    Language::Swift,
                ));

                if let Some(body_node) = child.child_by_field_name("body") {
                    walk_swift(&body_node, source, file_path, parent_class, symbols);
                }
            }

            // class_declaration covers class, struct, enum, extension, actor via declaration_kind
            "class_declaration" => {
                let decl_kind = child
                    .child_by_field_name("declaration_kind")
                    .map(|n| node_text(&n, source))
                    .unwrap_or("class");

                let sym_kind = match decl_kind {
                    "struct" => SymbolKind::Struct,
                    "enum" => SymbolKind::Enum,
                    _ => SymbolKind::Class, // class, actor
                };

                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<type>".to_string());

                let (start, end) = node_lines(&child);
                let body = node_text(&child, source).to_string();
                let sig = body.lines().next().unwrap_or("").trim().to_string();

                // For extensions, don't create a new symbol — just recurse with type context
                if decl_kind == "extension" {
                    if let Some(body_node) = child.child_by_field_name("body") {
                        walk_swift(&body_node, source, file_path, Some(name.as_str()), symbols);
                    }
                    continue;
                }

                let name_copy = name.clone();
                symbols.push(Symbol::new(
                    file_path,
                    &name,
                    sym_kind,
                    start,
                    end,
                    sig,
                    None,
                    body,
                    Language::Swift,
                ));

                if let Some(body_node) = child.child_by_field_name("body") {
                    walk_swift(&body_node, source, file_path, Some(&name_copy), symbols);
                }
            }

            "protocol_declaration" => {
                let name = child
                    .child_by_field_name("name")
                    .map(|n| node_text(&n, source).to_string())
                    .unwrap_or_else(|| "<protocol>".to_string());

                let (start, end) = node_lines(&child);
                let body = node_text(&child, source).to_string();
                let sig = body.lines().next().unwrap_or("").trim().to_string();
                let name_copy = name.clone();

                symbols.push(Symbol::new(
                    file_path,
                    &name,
                    SymbolKind::Interface,
                    start,
                    end,
                    sig,
                    None,
                    body,
                    Language::Swift,
                ));

                if let Some(body_node) = child.child_by_field_name("body") {
                    walk_swift(&body_node, source, file_path, Some(&name_copy), symbols);
                }
            }

            "import_declaration" => {
                let (start, end) = node_lines(&child);
                let text = node_text(&child, source).to_string();
                symbols.push(Symbol::new(
                    file_path,
                    &text.trim().to_string(),
                    SymbolKind::Import,
                    start,
                    end,
                    text.clone(),
                    None,
                    text,
                    Language::Swift,
                ));
            }

            _ => {
                walk_swift(&child, source, file_path, parent_class, symbols);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SQL (tree-sitter-sequel)
// ──────────────────────────────────────────────────────────────────────────────

fn extract_sql(file_path: &str, source: &str) -> Result<Vec<Symbol>> {
    let mut parser = match make_parser(&Language::Sql) {
        Some(p) => p,
        None => return Ok(vec![]),
    };

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("sql parse failed"))?;
    let root = tree.root_node();
    let mut symbols = Vec::new();

    walk_sql(&root, source, file_path, &mut symbols);

    Ok(symbols)
}

fn walk_sql(node: &Node, source: &str, file_path: &str, symbols: &mut Vec<Symbol>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        let sym_kind = match kind {
            "create_table" => Some(SymbolKind::Struct),
            "create_view" | "create_materialized_view" => Some(SymbolKind::TypeAlias),
            "create_function" => Some(SymbolKind::Function),
            "create_index" => Some(SymbolKind::Constant),
            "create_type" => Some(SymbolKind::Enum),
            _ => None,
        };

        if let Some(sym_kind) = sym_kind {
            // Name lives in the first `object_reference` child's `name` field
            let name = child
                .named_children(&mut child.walk())
                .find(|n| n.kind() == "object_reference")
                .and_then(|obj_ref| obj_ref.child_by_field_name("name"))
                .map(|n| node_text(&n, source).to_string())
                .unwrap_or_default();

            if !name.is_empty() {
                let (start, end) = node_lines(&child);
                let body = node_text(&child, source).to_string();
                let sig = body.lines().next().unwrap_or("").trim().to_string();
                symbols.push(Symbol::new(
                    file_path,
                    &name,
                    sym_kind,
                    start,
                    end,
                    sig,
                    None,
                    body,
                    Language::Sql,
                ));
            }
        } else {
            walk_sql(&child, source, file_path, symbols);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Markdown
// ──────────────────────────────────────────────────────────────────────────────

fn extract_markdown(file_path: &str, source: &str) -> Result<Vec<Symbol>> {
    let mut parser = match make_parser(&Language::Markdown) {
        Some(p) => p,
        None => return Ok(vec![]),
    };

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("markdown parse failed"))?;

    let mut symbols = Vec::new();
    walk_markdown(tree.root_node(), source, file_path, &mut symbols);
    Ok(symbols)
}

fn heading_level(node: &Node, source: &str) -> Option<u8> {
    // ATX headings: look for atx_h1_marker .. atx_h6_marker child
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let level = match child.kind() {
            "atx_h1_marker" => 1,
            "atx_h2_marker" => 2,
            "atx_h3_marker" => 3,
            "atx_h4_marker" => 4,
            "atx_h5_marker" => 5,
            "atx_h6_marker" => 6,
            // Setext headings use underline nodes
            "setext_h1_underline" => 1,
            "setext_h2_underline" => 2,
            _ => continue,
        };
        let _ = source; // suppress unused warning
        return Some(level);
    }
    None
}

fn heading_text<'a>(node: &Node, source: &'a str) -> &'a str {
    // The inline child holds the heading text
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "inline" {
            return node_text(&child, source);
        }
    }
    // Fallback: strip the marker prefix from the whole node text
    let raw = node_text(node, source);
    raw.trim_start_matches('#').trim()
}

fn walk_markdown(node: Node, source: &str, file_path: &str, symbols: &mut Vec<Symbol>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "section" => {
                // A section wraps a heading + its content. Walk children to find the heading.
                let mut inner = child.walk();
                for inner_child in child.children(&mut inner) {
                    if inner_child.kind() == "atx_heading" || inner_child.kind() == "setext_heading" {
                        let level = heading_level(&inner_child, source).unwrap_or(1);
                        let name = heading_text(&inner_child, source).trim().to_string();
                        let (start, _) = node_lines(&inner_child);
                        let (_, end) = node_lines(&child);
                        let sig = format!("{} {}", "#".repeat(level as usize), name);
                        let body = node_text(&child, source).to_string();
                        symbols.push(Symbol::new(
                            file_path,
                            &name,
                            SymbolKind::Module,
                            start,
                            end,
                            sig,
                            None,
                            body,
                            Language::Markdown,
                        ));
                        break;
                    }
                }
                // Recurse into section for nested sections
                walk_markdown(child, source, file_path, symbols);
            }
            "atx_heading" | "setext_heading" => {
                // Top-level heading not wrapped in a section
                let level = heading_level(&child, source).unwrap_or(1);
                let name = heading_text(&child, source).trim().to_string();
                let (start, end) = node_lines(&child);
                let sig = format!("{} {}", "#".repeat(level as usize), name);
                let body = node_text(&child, source).to_string();
                symbols.push(Symbol::new(
                    file_path,
                    &name,
                    SymbolKind::Module,
                    start,
                    end,
                    sig,
                    None,
                    body,
                    Language::Markdown,
                ));
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_python_function() {
        let src = r#"
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
"#;
        let symbols = extract_python("test.py", src).expect("python parse");
        let func = symbols
            .iter()
            .find(|s| s.name == "greet")
            .expect("greet not found");
        assert_eq!(func.kind, SymbolKind::Function);
        assert!(func.docstring.is_some());
    }

    #[test]
    fn indexes_rust_struct() {
        let src = r#"
/// A simple point.
pub struct Point {
    pub x: f64,
    pub y: f64,
}
"#;
        let symbols = extract_rust("lib.rs", src).expect("rust parse");
        let s = symbols
            .iter()
            .find(|s| s.name == "Point")
            .expect("Point not found");
        assert_eq!(s.kind, SymbolKind::Struct);
    }

    #[test]
    fn call_edges_include_args_snippet() {
        let src = r#"
fn caller() {
    let x = compute(1, 2, "hello");
    other_fn();
}
fn compute(a: i32, b: i32, c: &str) -> i32 { a + b }
fn other_fn() {}
"#;
        let symbols = extract_rust("lib.rs", src).expect("rust parse");
        let edges = extract_call_edges(&symbols);
        // Should have a Calls edge from caller → compute
        let compute_sym = symbols.iter().find(|s| s.name == "compute").unwrap();
        let caller_sym = symbols.iter().find(|s| s.name == "caller").unwrap();
        let edge = edges
            .iter()
            .find(|e| e.from_id == caller_sym.id && e.to_id == compute_sym.id);
        assert!(edge.is_some(), "expected Calls edge from caller to compute");
        // Label should include args snippet
        let label = edge.unwrap().label.as_deref().unwrap_or("");
        assert!(
            label.contains("compute"),
            "edge label should contain callee name"
        );
        assert!(label.contains('('), "edge label should contain args");
    }

    #[test]
    fn import_edges_resolve_python_names() {
        let src = r#"
from utils import validate, parse_request

def validate(x):
    pass

def parse_request(req):
    pass
"#;
        let symbols = extract_python("app.py", src).expect("python parse");
        let edges = extract_import_edges(&symbols);
        assert!(
            !edges.is_empty(),
            "expected import edges for validate/parse_request"
        );
        assert!(
            edges.iter().any(|e| e.label.as_deref() == Some("validate")),
            "expected edge labelled 'validate'"
        );
    }

    #[test]
    fn type_flow_edges_from_signatures() {
        let src = r#"
pub struct UserToken {
    pub value: String,
}
pub fn validate_token(token: UserToken) -> bool {
    true
}
"#;
        let symbols = extract_rust("lib.rs", src).expect("rust parse");
        let edges = extract_type_flow_edges(&symbols);
        let token_sym = symbols.iter().find(|s| s.name == "UserToken").unwrap();
        let fn_sym = symbols.iter().find(|s| s.name == "validate_token").unwrap();
        let edge = edges
            .iter()
            .find(|e| e.from_id == fn_sym.id && e.to_id == token_sym.id);
        assert!(
            edge.is_some(),
            "expected References edge from validate_token to UserToken"
        );
    }

    #[test]
    fn indexes_markdown_headings() {
        let src = r#"# Introduction

Some intro text.

## Installation

Install with cargo.

### Advanced

Details here.

## Usage

Basic usage.
"#;
        let symbols = extract_markdown("README.md", src).expect("md parse");
        assert!(
            symbols.iter().any(|s| s.name == "Introduction" && s.kind == SymbolKind::Module),
            "expected h1"
        );
        assert!(
            symbols.iter().any(|s| s.name == "Installation"),
            "expected h2"
        );
        assert!(
            symbols.iter().any(|s| s.name == "Advanced"),
            "expected h3"
        );
        assert!(
            symbols.iter().any(|s| s.name == "Usage"),
            "expected second h2"
        );
    }

    #[test]
    fn indexes_sql_create_table() {
        let src = r#"
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE VIEW active_users AS SELECT * FROM users WHERE active = true;

CREATE FUNCTION get_user(p_id INT) RETURNS users AS $$
BEGIN
  RETURN QUERY SELECT * FROM users WHERE id = p_id;
END;
$$ LANGUAGE plpgsql;
"#;
        let symbols = extract_sql("schema.sql", src).expect("sql parse");
        assert!(
            symbols.iter().any(|s| s.name == "users" && s.kind == SymbolKind::Struct),
            "expected users table"
        );
        assert!(
            symbols.iter().any(|s| s.name == "active_users" && s.kind == SymbolKind::TypeAlias),
            "expected active_users view"
        );
        assert!(
            symbols.iter().any(|s| s.name == "get_user" && s.kind == SymbolKind::Function),
            "expected get_user function"
        );
    }
}
