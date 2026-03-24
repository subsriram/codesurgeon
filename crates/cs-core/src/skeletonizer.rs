use crate::symbol::{Symbol, SymbolKind};

/// Returns a skeleton representation of a symbol:
/// signature + optional docstring, with the implementation body stripped.
/// This gives the agent the API surface without the token cost of the body.
pub fn skeletonize(sym: &Symbol) -> String {
    match sym.kind {
        // For imports and small items, return as-is
        SymbolKind::Import | SymbolKind::Constant | SymbolKind::Variable => sym.signature.clone(),

        // For type definitions, include the full signature but truncate long bodies
        SymbolKind::Enum | SymbolKind::Struct | SymbolKind::TypeAlias => skeleton_type(sym),

        // For callables, signature + docstring only
        SymbolKind::Function
        | SymbolKind::AsyncFunction
        | SymbolKind::Method
        | SymbolKind::AsyncMethod => skeleton_callable(sym),

        // For containers, show the header + member signatures
        SymbolKind::Class
        | SymbolKind::Interface
        | SymbolKind::Trait
        | SymbolKind::Impl
        | SymbolKind::Module => skeleton_container(sym),

        // For macro/script blocks, just the first line
        SymbolKind::Macro | SymbolKind::ScriptBlock | SymbolKind::StyleBlock => {
            sym.signature.clone()
        }
    }
}

fn skeleton_callable(sym: &Symbol) -> String {
    let mut out = String::new();

    if let Some(doc) = &sym.docstring {
        // Include docstring / JSDoc comment
        out.push_str(doc);
        out.push('\n');
    }

    // Include full signature line(s)
    out.push_str(&sym.signature);

    // Add a "..." body placeholder so the agent knows there IS a body
    match sym.language {
        crate::language::Language::Python => {
            out.push_str("\n    ...");
        }
        crate::language::Language::Rust => {
            out.push_str(" { ... }");
        }
        _ => {
            out.push_str(" { ... }");
        }
    }

    out
}

fn skeleton_type(sym: &Symbol) -> String {
    let lines: Vec<&str> = sym.body.lines().collect();

    // For short types (≤ 20 lines), include the full body
    if lines.len() <= 20 {
        return sym.body.clone();
    }

    // For long types, include signature + first few fields + "..."
    let mut out = sym.signature.clone();
    out.push_str(" {\n");

    // Show up to 8 member lines
    let member_lines: Vec<&str> = lines
        .iter()
        .skip(1) // skip opening line
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && t != "{" && t != "}"
        })
        .take(8)
        .cloned()
        .collect();

    for line in &member_lines {
        out.push_str(line);
        out.push('\n');
    }

    if lines.len() > 10 {
        out.push_str("    // ... more fields\n");
    }
    out.push('}');
    out
}

fn skeleton_container(sym: &Symbol) -> String {
    let lines: Vec<&str> = sym.body.lines().collect();

    // For very small containers, return as-is
    if lines.len() <= 5 {
        return sym.body.clone();
    }

    let mut out = String::new();

    if let Some(doc) = &sym.docstring {
        out.push_str(doc);
        out.push('\n');
    }

    // Opening line (class Foo extends Bar {)
    out.push_str(&sym.signature);
    out.push_str(" {\n");

    // Extract method/field signature lines without their bodies
    // We look for lines that look like function/method declarations
    let method_sigs: Vec<String> = extract_member_signatures(&lines, &sym.language);

    for sig in &method_sigs {
        out.push_str("  ");
        out.push_str(sig);
        out.push('\n');
    }

    if method_sigs.is_empty() {
        out.push_str("  // ...\n");
    }

    out.push('}');
    out
}

fn extract_member_signatures(lines: &[&str], lang: &crate::language::Language) -> Vec<String> {
    use crate::language::Language;

    let mut sigs = Vec::new();
    let mut depth = 0i32;

    for line in lines {
        let trimmed = line.trim();

        // Track brace depth
        for c in trimmed.chars() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }

        // Only pick up top-level members (depth == 1 after opening brace)
        if depth != 1 {
            continue;
        }

        let is_member = match lang {
            Language::Rust => {
                trimmed.starts_with("pub fn")
                    || trimmed.starts_with("fn ")
                    || trimmed.starts_with("pub async fn")
                    || trimmed.starts_with("async fn")
                    || trimmed.starts_with("pub ")
            }
            Language::Python => trimmed.starts_with("def ") || trimmed.starts_with("async def "),
            Language::TypeScript | Language::Tsx | Language::JavaScript | Language::Jsx => {
                !trimmed.starts_with("//")
                    && !trimmed.starts_with("*")
                    && !trimmed.is_empty()
                    && !trimmed.starts_with("}")
            }
            Language::Swift => {
                trimmed.starts_with("func ")
                    || trimmed.starts_with("var ")
                    || trimmed.starts_with("let ")
            }
            _ => false,
        };

        if is_member {
            // Trim to just the signature (up to opening brace or colon)
            let sig = trimmed
                .find('{')
                .or_else(|| trimmed.find(':'))
                .map(|i| trimmed[..i].trim().to_string())
                .unwrap_or_else(|| trimmed.to_string());

            if !sig.is_empty() {
                sigs.push(format!("{}  // ...", sig));
            }
        }
    }

    sigs
}

/// Estimate how many tokens a skeleton will consume.
pub fn skeleton_token_estimate(sym: &Symbol) -> u32 {
    let skeleton = skeletonize(sym);
    (skeleton.len() / 4) as u32
}

