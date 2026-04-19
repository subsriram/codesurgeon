use crate::language::{detect_language, Language};
use crate::symbol::{Symbol, SymbolKind};
use anyhow::Result;
use std::path::Path;
use tree_sitter::{Node, Parser};

// Re-export edge extraction so callers can use `crate::indexer::extract_*`
// without knowing about the edges module (source-compatible with old import paths).
pub use crate::edges::{
    extract_call_edges, extract_impl_edges, extract_import_edges, extract_shell_call_edges,
    extract_sql_ref_edges, extract_type_flow_edges,
};

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Parse raw Rust source (e.g. `cargo-expand` output) against a given
/// logical `file_path` and return all symbols. Used by the macro enrichment pass.
pub fn parse_rust_source(file_path: &str, source: &str) -> Result<Vec<Symbol>> {
    extract_rust(file_path, source)
}

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

// ──────────────────────────────────────────────────────────────────────────────
// Helper: parse with tree-sitter and walk nodes
// ──────────────────────────────────────────────────────────────────────────────

fn make_parser(lang: &Language) -> Option<Parser> {
    let ts_lang = match lang.tree_sitter_language() {
        Some(l) => l,
        None => {
            tracing::warn!("No tree-sitter grammar available for {:?}", lang);
            return None;
        }
    };
    let mut parser = Parser::new();
    if let Err(e) = parser.set_language(&ts_lang) {
        tracing::warn!("Failed to set tree-sitter language for {:?}: {}", lang, e);
        return None;
    }
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
        // For decorated_definition, the outer node spans the decorators + def
        // (used for line range and body text so decorators are captured), while
        // the inner definition node is what we introspect for name/params/etc.
        let (outer, inner) = if child.kind() == "decorated_definition" {
            let inner = child.child_by_field_name("definition").unwrap_or(child);
            (child, inner)
        } else {
            (child, child)
        };

        match inner.kind() {
            "function_definition" => {
                let name_node = inner.child_by_field_name("name");
                let name = name_node
                    .map(|n| node_text(&n, source))
                    .unwrap_or("<anonymous>");

                // Check if async
                let is_async = inner
                    .child(0)
                    .map(|c| node_text(&c, source) == "async")
                    .unwrap_or(false);

                let kind = if parent_class.is_some() {
                    if is_async {
                        SymbolKind::AsyncMethod
                    } else {
                        SymbolKind::Method
                    }
                } else if is_async {
                    SymbolKind::AsyncFunction
                } else {
                    SymbolKind::Function
                };

                let (start, end) = node_lines(&outer);
                let body = node_text(&outer, source).to_string();

                // Signature = everything up to the colon on the def line
                let params = inner
                    .child_by_field_name("parameters")
                    .map(|p| node_text(&p, source))
                    .unwrap_or("()");
                let return_type = inner
                    .child_by_field_name("return_type")
                    .map(|r| format!(" -> {}", node_text(&r, source)))
                    .unwrap_or_default();
                let sig = format!("def {}{}{}:", name, params, return_type);

                // Docstring from body block
                let docstring = inner
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
                if let Some(body_node) = inner.child_by_field_name("body") {
                    walk_python_node(&body_node, source, file_path, parent_class, symbols);
                }
            }

            "class_definition" => {
                let name_node = inner.child_by_field_name("name");
                let name = name_node
                    .map(|n| node_text(&n, source))
                    .unwrap_or("<anonymous>");
                let (start, end) = node_lines(&outer);
                let body = node_text(&outer, source).to_string();

                // Superclasses
                let superclasses = inner
                    .child_by_field_name("superclasses")
                    .map(|s| format!("({})", node_text(&s, source)))
                    .unwrap_or_default();
                let sig = format!("class {}{}:", name, superclasses);

                let docstring = inner
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
                if let Some(body_node) = inner.child_by_field_name("body") {
                    walk_python_node(&body_node, source, file_path, Some(name), symbols);
                }
            }

            "import_statement" | "import_from_statement" => {
                let (start, end) = node_lines(&child);
                let text = node_text(&child, source).to_string();
                // Use the whole import line as name for now
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

#[allow(clippy::too_many_arguments)]
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
                .unwrap_or_else(|| "<anonymous>".to_string());

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
                    .unwrap_or_else(|| "<anonymous>".to_string());

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
                let name = text.trim().to_string();
                symbols.push(Symbol::new(
                    file_path,
                    &name,
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
    // The inline child holds the heading text (ATX headings).
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "inline" {
            return node_text(&child, source);
        }
    }
    // Setext headings: the text is in the paragraph_content child that precedes
    // the underline marker. Look for any child that is NOT the underline.
    let mut cursor2 = node.walk();
    for child in node.children(&mut cursor2) {
        match child.kind() {
            "setext_h1_underline"
            | "setext_h2_underline"
            | "atx_h1_marker"
            | "atx_h2_marker"
            | "atx_h3_marker"
            | "atx_h4_marker"
            | "atx_h5_marker"
            | "atx_h6_marker" => continue,
            _ => {
                let text = node_text(&child, source).trim();
                if !text.is_empty() {
                    // Safety: the returned lifetime is tied to `source`, and
                    // `node_text` returns a slice of `source`. We trim a `&str`
                    // that borrows `source`, so the cast is valid.
                    return unsafe {
                        std::str::from_utf8_unchecked(
                            &source.as_bytes()[child.start_byte()..child.end_byte()],
                        )
                        .trim()
                    };
                }
            }
        }
    }
    // Last-resort fallback: strip ATX marker prefix and any trailing underline line.
    let raw = node_text(node, source).trim_start_matches('#').trim();
    // For setext: the underline is the last line; drop it.
    if let Some(last_newline) = raw.rfind('\n') {
        let last_line = raw[last_newline + 1..].trim();
        if last_line.chars().all(|c| c == '=' || c == '-') && !last_line.is_empty() {
            return raw[..last_newline].trim();
        }
    }
    raw
}

fn walk_markdown(node: Node, source: &str, file_path: &str, symbols: &mut Vec<Symbol>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "section" {
            // A section wraps a heading + its content.
            //
            // ATX headings: each sub-level gets its own nested `section` node, so
            // only the primary heading of this section appears as a direct child.
            //
            // Setext headings: tree-sitter does NOT create nested sections — all
            // headings in the document appear as siblings inside one flat section.
            // We therefore process every heading child we encounter (no break).
            let mut inner = child.walk();
            for inner_child in child.children(&mut inner) {
                if inner_child.kind() == "atx_heading" || inner_child.kind() == "setext_heading" {
                    let level = heading_level(&inner_child, source).unwrap_or(1);
                    let name = heading_text(&inner_child, source).trim().to_string();
                    if name.is_empty() {
                        continue;
                    }
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
                    // For ATX headings, sub-levels live in nested sections that
                    // walk_markdown will visit via the recursive call below.
                    // For setext headings, siblings are peers in this section and
                    // we must NOT break — continue the inner loop.
                    if inner_child.kind() == "atx_heading" {
                        break;
                    }
                }
            }
            // Recurse into section for nested ATX sections.
            walk_markdown(child, source, file_path, symbols);
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

    /// Regression: a decorated class with a dotted-path superclass used to be
    /// silently dropped, along with all of its methods. See matplotlib's
    /// `_AxesBase` in `lib/matplotlib/axes/_base.py` for the real-world case.
    #[test]
    fn indexes_decorated_python_class_and_methods() {
        let src = r#"
@some_decorator({"key": ["val"]})
class Foo(pkg.BaseArtist):
    """Decorated class."""

    name = "foo"

    def __init__(self):
        self.x = 1

    @property
    def bar(self):
        return self.x

    def baz(self, n):
        return n + 1
"#;
        let symbols = extract_python("decor.py", src).expect("python parse");
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"Foo"),
            "decorated class must be indexed; got: {:?}",
            names
        );
        let foo = symbols.iter().find(|s| s.name == "Foo").unwrap();
        assert_eq!(foo.kind, SymbolKind::Class);
        assert!(
            foo.signature.contains("pkg.BaseArtist"),
            "class signature should include dotted superclass; got: {}",
            foo.signature
        );
        for method in ["__init__", "bar", "baz"] {
            let fqn = format!("Foo::{}", method);
            assert!(
                symbols.iter().any(|s| s.fqn.ends_with(&fqn)),
                "method {} must be indexed under Foo; got fqns: {:?}",
                method,
                symbols.iter().map(|s| &s.fqn).collect::<Vec<_>>()
            );
        }
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

    /// Regression for issue #62: identifiers inside a Python docstring must
    /// not turn into `calls` edges. Before the fix, `evalf`, `dict`, and
    /// `sqrt` in the `>>> ...` doctest block would fan out to every symbol
    /// with those names elsewhere in the workspace.
    #[test]
    fn call_edges_skip_python_docstring_doctest() {
        let src = r#"
class SomeType:
    def evalf(self, n):
        pass

class Something:
    def dict(self):
        pass

def parse_latex(text):
    r"""Parse a LaTeX string into a SymPy expression.

    Examples
    ========

    >>> expr = parse_latex(r"\frac{1 + \sqrt{a}}{b}")
    >>> expr.evalf(4, subs=dict(a=5, b=2))
    1.618
    """
    result = _parse_it(text)
    return result

def _parse_it(text):
    return text
"#;
        let symbols = extract_python("parser.py", src).expect("python parse");
        let edges = extract_call_edges(&symbols);
        let parse_latex = symbols
            .iter()
            .find(|s| s.name == "parse_latex")
            .expect("parse_latex missing");

        let outgoing: Vec<&crate::symbol::Edge> = edges
            .iter()
            .filter(|e| e.from_id == parse_latex.id)
            .collect();

        // The only real call is `_parse_it(text)`.
        let labels: Vec<&str> = outgoing.iter().filter_map(|e| e.label.as_deref()).collect();
        assert!(
            labels.iter().any(|l| l.starts_with("_parse_it")),
            "expected call edge to _parse_it, got labels: {:?}",
            labels
        );
        // None of the docstring-mentioned identifiers should produce edges.
        for name in ["evalf", "dict"] {
            assert!(
                !labels.iter().any(|l| l.starts_with(name)),
                "docstring identifier '{}' should not produce a call edge (labels: {:?})",
                name,
                labels
            );
        }
    }

    /// Triple-quoted strings used as embedded SQL / HTML / templates also
    /// contain identifier-shaped tokens that are not real calls.
    #[test]
    fn call_edges_skip_python_triple_quoted_non_docstring() {
        let src = r#"
def run_query(conn):
    sql = """
        SELECT * FROM users WHERE status = dict(active=True)
    """
    return conn.execute(sql)

def dict(x):
    return x

def execute(q):
    return q
"#;
        let symbols = extract_python("q.py", src).expect("python parse");
        let edges = extract_call_edges(&symbols);
        let run_query = symbols.iter().find(|s| s.name == "run_query").unwrap();
        let dict_sym = symbols.iter().find(|s| s.name == "dict").unwrap();
        assert!(
            !edges
                .iter()
                .any(|e| e.from_id == run_query.id && e.to_id == dict_sym.id),
            "`dict(...)` inside a triple-quoted string should not create a call edge"
        );
    }

    /// Control: a real call outside any string literal still produces an edge
    /// after docstring stripping.
    #[test]
    fn call_edges_python_real_call_still_resolved() {
        let src = r#"
def helper(x):
    return x + 1

def caller():
    """Do some work.

    >>> helper(1)
    2
    """
    return helper(42)
"#;
        let symbols = extract_python("m.py", src).expect("python parse");
        let edges = extract_call_edges(&symbols);
        let caller = symbols.iter().find(|s| s.name == "caller").unwrap();
        let helper = symbols.iter().find(|s| s.name == "helper").unwrap();
        // Exactly one edge caller→helper (from the real `return helper(42)`,
        // not the doctest `>>> helper(1)` — but either way, dedup'd to one).
        let matching: Vec<_> = edges
            .iter()
            .filter(|e| e.from_id == caller.id && e.to_id == helper.id)
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "expected exactly one caller→helper edge, got {}",
            matching.len()
        );
        // And the label's args snippet should be from the real call, not the doctest.
        let label = matching[0].label.as_deref().unwrap_or("");
        assert!(
            label.contains("42"),
            "expected real-call args in label, got {:?}",
            label
        );
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
            symbols
                .iter()
                .any(|s| s.name == "Introduction" && s.kind == SymbolKind::Module),
            "expected h1"
        );
        assert!(
            symbols.iter().any(|s| s.name == "Installation"),
            "expected h2"
        );
        assert!(symbols.iter().any(|s| s.name == "Advanced"), "expected h3");
        assert!(
            symbols.iter().any(|s| s.name == "Usage"),
            "expected second h2"
        );
        // Bodies must contain section content, not just the heading line
        let install = symbols.iter().find(|s| s.name == "Installation").unwrap();
        assert!(
            install.body.contains("Install with cargo"),
            "Installation body should include section content"
        );
        // No duplicate symbols — each heading should appear exactly once
        let intro_count = symbols.iter().filter(|s| s.name == "Introduction").count();
        assert_eq!(intro_count, 1, "Introduction should appear exactly once");
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
            symbols
                .iter()
                .any(|s| s.name == "users" && s.kind == SymbolKind::Struct),
            "expected users table"
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.name == "active_users" && s.kind == SymbolKind::TypeAlias),
            "expected active_users view"
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.name == "get_user" && s.kind == SymbolKind::Function),
            "expected get_user function"
        );
    }

    // ── extract_args_snippet edge cases ──────────────────────────────────────

    /// The function was previously panicking when a multi-byte UTF-8 character
    /// straddled the truncation boundary. Verify it now truncates cleanly.
    #[test]
    fn args_snippet_truncates_at_utf8_boundary() {
        // Each CJK character is 3 bytes; with max_chars=5 the boundary falls
        // in the middle of the second character. The function must not panic
        // and must produce a valid UTF-8 string shorter than the input.
        let src = "fn f(你好世界abc) {}";
        let bytes = src.as_bytes();
        // Find the '(' position and step past it to get args_start.
        let args_start = src.find('(').unwrap() + 1;
        let result = crate::edges::extract_args_snippet(bytes, args_start, 5);
        // Must not panic and must be valid UTF-8.
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
        // Must be no longer than max_chars + the ellipsis.
        assert!(
            result.chars().count() <= 6,
            "snippet too long: {:?}",
            result
        );
    }

    #[test]
    fn args_snippet_empty_parens() {
        let src = b"foo()";
        // args_start points at the char after '('
        let result = crate::edges::extract_args_snippet(src, 4, 60);
        assert!(
            result.is_empty(),
            "expected empty snippet, got {:?}",
            result
        );
    }

    #[test]
    fn args_snippet_unclosed_paren_does_not_panic() {
        // No closing ')' — depth never reaches 0.
        let src = b"foo(bar, baz";
        let result = crate::edges::extract_args_snippet(src, 4, 60);
        // Should return whatever it found without panicking.
        assert!(!result.is_empty());
    }

    // ── Empty / whitespace-only files ────────────────────────────────────────

    #[test]
    fn empty_file_python() {
        let symbols = extract_python("empty.py", "").expect("python parse");
        assert!(symbols.is_empty());
    }

    #[test]
    fn whitespace_only_python() {
        let symbols = extract_python("blank.py", "   \n\t\n   ").expect("python parse");
        assert!(symbols.is_empty());
    }

    #[test]
    fn comments_only_python() {
        let src = "# This is a comment\n# Another comment\n";
        let symbols = extract_python("comments.py", src).expect("python parse");
        assert!(symbols.is_empty());
    }

    #[test]
    fn empty_file_rust() {
        let symbols = extract_rust("empty.rs", "").expect("rust parse");
        assert!(symbols.is_empty());
    }

    #[test]
    fn empty_file_shell() {
        let symbols = extract_shell("empty.sh", "").expect("shell parse");
        assert!(symbols.is_empty());
    }

    #[test]
    fn empty_file_sql() {
        let symbols = extract_sql("empty.sql", "").expect("sql parse");
        assert!(symbols.is_empty());
    }

    #[test]
    fn empty_file_markdown() {
        let symbols = extract_markdown("empty.md", "").expect("md parse");
        assert!(symbols.is_empty());
    }

    // ── Unicode / multi-byte identifiers ─────────────────────────────────────

    /// Python allows Unicode identifiers; ensure the indexer doesn't panic or
    /// silently drop the symbol.
    #[test]
    fn python_unicode_function_name() {
        let src = "def 処理する(x):\n    return x\n";
        let symbols = extract_python("unicode.py", src).expect("python parse");
        assert!(
            symbols.iter().any(|s| s.name == "処理する"),
            "expected Unicode function name to be indexed; got: {:?}",
            symbols.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }

    /// Rust identifiers can include Unicode letters; ensure no panic.
    #[test]
    fn rust_unicode_struct_name() {
        // Rust supports non-ASCII identifiers (RFC 2457).
        let src = "pub struct Häuser { pub count: u32 }\n";
        let symbols = extract_rust("unicode.rs", src).expect("rust parse");
        assert!(
            symbols.iter().any(|s| s.name == "Häuser"),
            "expected Unicode struct name; got: {:?}",
            symbols.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }

    /// Call edges where the args string contains multi-byte UTF-8 must not panic.
    #[test]
    fn call_edges_with_multibyte_utf8_args() {
        let src = r#"
fn caller() {
    process("你好", 42);
}
fn process(s: &str, n: i32) {}
"#;
        let symbols = extract_rust("lib.rs", src).expect("rust parse");
        // extract_call_edges must not panic on multibyte args.
        let edges = extract_call_edges(&symbols);
        let process_sym = symbols.iter().find(|s| s.name == "process").unwrap();
        let caller_sym = symbols.iter().find(|s| s.name == "caller").unwrap();
        assert!(
            edges
                .iter()
                .any(|e| e.from_id == caller_sym.id && e.to_id == process_sym.id),
            "expected Calls edge from caller to process"
        );
    }

    // ── Shell call-graph edges ────────────────────────────────────────────────

    #[test]
    fn shell_call_edges_basic() {
        let src = r#"
bar() {
  echo "bar"
}

baz() {
  echo "baz"
}

foo() {
  bar
  baz
}
"#;
        let symbols = extract_shell("script.sh", src).expect("shell parse");
        let edges = extract_shell_call_edges(&symbols);
        let foo_sym = symbols.iter().find(|s| s.name == "foo").expect("foo");
        let bar_sym = symbols.iter().find(|s| s.name == "bar").expect("bar");
        let baz_sym = symbols.iter().find(|s| s.name == "baz").expect("baz");
        assert!(
            edges
                .iter()
                .any(|e| e.from_id == foo_sym.id && e.to_id == bar_sym.id),
            "expected foo → bar edge"
        );
        assert!(
            edges
                .iter()
                .any(|e| e.from_id == foo_sym.id && e.to_id == baz_sym.id),
            "expected foo → baz edge"
        );
    }

    #[test]
    fn shell_call_edges_no_self_loop() {
        let src = "foo() {\n  foo\n}\n";
        let symbols = extract_shell("script.sh", src).expect("shell parse");
        let edges = extract_shell_call_edges(&symbols);
        assert!(
            edges.iter().all(|e| e.from_id != e.to_id),
            "self-loop edges must not be produced"
        );
    }

    // ── SQL reference edges ───────────────────────────────────────────────────

    #[test]
    fn sql_ref_edges_view_to_table() {
        let src = r#"
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE VIEW active_users AS SELECT * FROM users WHERE active = true;
"#;
        let symbols = extract_sql("schema.sql", src).expect("sql parse");
        let edges = extract_sql_ref_edges(&symbols);
        let users_sym = symbols
            .iter()
            .find(|s| s.name == "users")
            .expect("users table");
        let view_sym = symbols
            .iter()
            .find(|s| s.name == "active_users")
            .expect("active_users view");
        assert!(
            edges
                .iter()
                .any(|e| e.from_id == view_sym.id && e.to_id == users_sym.id),
            "expected active_users → users edge; edges={:?}",
            edges
        );
    }

    #[test]
    fn sql_ref_edges_function_to_table() {
        let src = r#"
CREATE TABLE users (
    id SERIAL PRIMARY KEY
);

CREATE FUNCTION get_user(p_id INT) RETURNS users AS $$
BEGIN
  RETURN QUERY SELECT * FROM users WHERE id = p_id;
END;
$$ LANGUAGE plpgsql;
"#;
        let symbols = extract_sql("schema.sql", src).expect("sql parse");
        let edges = extract_sql_ref_edges(&symbols);
        let users_sym = symbols
            .iter()
            .find(|s| s.name == "users")
            .expect("users table");
        let fn_sym = symbols
            .iter()
            .find(|s| s.name == "get_user")
            .expect("get_user");
        assert!(
            edges
                .iter()
                .any(|e| e.from_id == fn_sym.id && e.to_id == users_sym.id),
            "expected get_user → users edge; edges={:?}",
            edges
        );
    }

    // ── Markdown setext headings ──────────────────────────────────────────────

    /// Setext-style headings (underlined with `===` or `---`) must be indexed.
    #[test]
    fn markdown_setext_headings() {
        let src = "Top Level\n=========\n\nSome text.\n\nSubsection\n-----------\n\nMore text.\n";
        let symbols = extract_markdown("setext.md", src).expect("md parse");
        assert!(
            symbols.iter().any(|s| s.name == "Top Level"),
            "expected setext h1 'Top Level'; got: {:?}",
            symbols.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
        assert!(
            symbols.iter().any(|s| s.name == "Subsection"),
            "expected setext h2 'Subsection'; got: {:?}",
            symbols.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }
}
