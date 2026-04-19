//! Explicit symbol-name anchor extraction for ranking.
//!
//! Extracts identifiers from the task query that look like exact symbol names.
//! Three sources:
//!   1. Function/method calls in fenced code blocks (`foo.bar(...)`)
//!   2. `from X.Y import Z` / `import X.Y as Z` statements
//!   3. Prose tokens that look like identifiers (snake_case or CamelCase)
//!
//! All matches are flat-scored — ranking within the anchor list is
//! "extraction order" (code snippets first, then prose). RRF fusion handles
//! blending with BM25/ANN/graph.

use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// English stop words that are also common programming identifiers.
/// Used to filter prose tokens; code-snippet extraction ignores this list.
const STOP_WORDS: &[&str] = &[
    // English filler
    "with",
    "when",
    "where",
    "this",
    "that",
    "from",
    "into",
    "have",
    "been",
    "just",
    "like",
    "make",
    "many",
    "more",
    "most",
    "must",
    "only",
    "over",
    "such",
    "than",
    "then",
    "they",
    "were",
    "will",
    "upon",
    "what",
    "about",
    "your",
    "there",
    "their",
    "some",
    "them",
    "these",
    "those",
    "which",
    "would",
    "could",
    "should",
    "after",
    "before",
    "while",
    "also",
    "does",
    "other",
    "each",
    "same",
    "here",
    "because",
    "both",
    // common type/collection names we don't want to anchor on
    "none",
    "true",
    "false",
    "null",
    "int",
    "str",
    "dict",
    "list",
    "set",
    "tuple",
    "bool",
    "float",
    "bytes",
    "type",
    "kind",
    "name",
    "value",
    "values",
    "size",
    "length",
    "index",
    "data",
    "item",
    "items",
    "path",
    "file",
    "files",
    "line",
    "lines",
    "test",
    "tests",
    "error",
    "errors",
    "cause",
    "fail",
    "pass",
    "call",
    "calls",
    "version",
    "using",
    "result",
    "return",
    "returns",
    "function",
    "method",
    "class",
    "object",
    "module",
    "package",
    "import",
    "input",
    "output",
    "field",
    "fields",
    "attribute",
    "attributes",
    "example",
    "empty",
    "true",
    "false",
    "print",
    "self",
];

fn identifier_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // At least 4 characters to avoid matching 2-letter stop words and noise.
    RE.get_or_init(|| Regex::new(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b").unwrap())
}

fn call_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Matches `a.b.c(`, `Foo(`, `self.bar(` — captures the full dotted path.
    RE.get_or_init(|| {
        Regex::new(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\(").unwrap()
    })
}

fn import_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `from X.Y import A, B` or `import X.Y as Z` / `import X.Y`.
    RE.get_or_init(|| {
        // Match either at the start of a line (canonical Python code) or after
        // a sentence boundary / whitespace (inline mentions in prose bug reports).
        Regex::new(
            r"(?m)(?:^|[\s:;])\s*(?:from\s+([\w.]+)\s+import\s+([\w, ]+)|import\s+([\w.]+)(?:\s+as\s+(\w+))?)",
        )
        .unwrap()
    })
}

fn code_block_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"```[\w]*\n([\s\S]*?)```").unwrap())
}

fn dotted_prose_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Identifier-dotted-identifier chain of at least 2 segments. Each segment
    // must start with a letter or underscore. Captures the full chain (e.g.
    // `xr.where`, `urllib.request.urlopen`).
    RE.get_or_init(|| {
        Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b").unwrap()
    })
}

/// Extracted anchors, in order of discovery.
#[derive(Debug, Default, Clone)]
pub struct Anchors {
    /// Symbol names to try looking up exactly (deduplicated, in order).
    pub symbol_names: Vec<String>,
    /// Module paths (from import statements).
    pub module_paths: Vec<String>,
    /// Names that came from a dotted call (e.g. `xr.where`, whether from a
    /// code block or inline prose). When multiple exact matches exist for such
    /// a name, the anchor resolver prefers module-level fqns (1 `::`) over
    /// class methods (2+ `::`), since a dotted call is almost always a
    /// module-level function, not a method.
    pub from_dotted_call: HashSet<String>,
}

/// Extract anchor candidates from a free-form query.
///
/// Order matters — code-block calls and imports come first (highest priority
/// for RRF), then prose tokens.
pub fn extract(query: &str) -> Anchors {
    let mut out = Anchors::default();
    let mut seen: HashSet<String> = HashSet::new();
    let push = |out: &mut Anchors, seen: &mut HashSet<String>, name: &str| {
        if name.len() >= 3 && seen.insert(name.to_string()) {
            out.symbol_names.push(name.to_string());
        }
    };

    // 1. Code-block imports + API calls — highest priority.
    for block_cap in code_block_re().captures_iter(query) {
        let block = &block_cap[1];

        // Imports inside the code block.
        for imp in import_re().captures_iter(block) {
            if let Some(m) = imp.get(1) {
                out.module_paths.push(m.as_str().to_string());
            }
            if let Some(m) = imp.get(3) {
                out.module_paths.push(m.as_str().to_string());
            }
            if let Some(names) = imp.get(2) {
                for n in names.as_str().split(',') {
                    let n = n.trim();
                    if !n.is_empty() {
                        push(&mut out, &mut seen, n);
                    }
                }
            }
        }

        // Function/method calls inside the code block.
        for cap in call_re().captures_iter(block) {
            let full = &cap[1];
            let is_dotted = full.contains('.');
            push(&mut out, &mut seen, full);
            if is_dotted {
                out.from_dotted_call.insert(full.to_string());
            }
            if let Some(last) = full.rsplit('.').next() {
                if last != full {
                    push(&mut out, &mut seen, last);
                    if is_dotted && last.len() > 2 {
                        out.from_dotted_call.insert(last.to_string());
                    }
                }
            }
        }
    }

    // 2. Top-level import statements outside code fences (still machine-precise).
    for imp in import_re().captures_iter(query) {
        if let Some(m) = imp.get(1) {
            out.module_paths.push(m.as_str().to_string());
        }
        if let Some(m) = imp.get(3) {
            out.module_paths.push(m.as_str().to_string());
        }
        if let Some(names) = imp.get(2) {
            for n in names.as_str().split(',') {
                let n = n.trim();
                if !n.is_empty() {
                    push(&mut out, &mut seen, n);
                }
            }
        }
    }

    // 3. Prose identifiers — lower priority, filtered by stop words + shape.
    for m in identifier_re().find_iter(query) {
        let tok = m.as_str();
        let lower = tok.to_lowercase();
        if STOP_WORDS.contains(&lower.as_str()) {
            continue;
        }
        // Require either underscore or camelCase — filters out plain English words.
        let has_snake = tok.contains('_');
        let has_camel =
            tok.chars().any(|c| c.is_uppercase()) && tok.chars().any(|c| c.is_lowercase());
        if !has_snake && !has_camel {
            continue;
        }
        push(&mut out, &mut seen, tok);
    }

    // 4. Dotted-identifier chains anywhere in the query (inline prose API
    //    calls like `xr.where`). The prose regex above stops at `.` so it
    //    would miss these; this pass catches them and marks them as
    //    originating from a dotted call so the resolver prefers module-level
    //    symbols over class methods.
    for m in dotted_prose_re().find_iter(query) {
        let full = m.as_str();
        push(&mut out, &mut seen, full);
        out.from_dotted_call.insert(full.to_string());
        if let Some(last) = full.rsplit('.').next() {
            if last != full && last.len() > 2 {
                push(&mut out, &mut seen, last);
                out.from_dotted_call.insert(last.to_string());
            }
        }
    }

    // Deduplicate module_paths while preserving order.
    let mut seen_paths: HashSet<String> = HashSet::new();
    out.module_paths.retain(|p| seen_paths.insert(p.clone()));

    out
}

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
        assert!(a.symbol_names.is_empty());
    }

    #[test]
    fn camelcase_accepted() {
        let a = extract("The BuildEnvironment class handles the case");
        assert!(a.symbol_names.contains(&"BuildEnvironment".to_string()));
    }

    #[test]
    fn plain_english_rejected() {
        let a = extract("The function should return an empty dict for fields");
        assert!(!a.symbol_names.iter().any(|s| s == "function"));
        assert!(!a.symbol_names.iter().any(|s| s == "fields"));
        assert!(!a.symbol_names.iter().any(|s| s == "return"));
    }

    #[test]
    fn code_block_before_prose() {
        // Call in a code block should be extracted before prose identifiers.
        let q = "Some description mentioning MyClass, and:\n\
                 ```python\nmy_func(1)\n```";
        let a = extract(q);
        let pos_func = a.symbol_names.iter().position(|s| s == "my_func");
        let pos_class = a.symbol_names.iter().position(|s| s == "MyClass");
        assert!(pos_func.is_some());
        assert!(pos_class.is_some());
        assert!(pos_func.unwrap() < pos_class.unwrap());
    }

    #[test]
    fn dedup() {
        let q = "```python\nfoo_bar()\nfoo_bar()\n```\nAnd foo_bar is mentioned again.";
        let a = extract(q);
        let count = a.symbol_names.iter().filter(|s| *s == "foo_bar").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn import_outside_code_fence() {
        let q = "This reproduces with: from sympy.parsing.latex import parse_latex";
        let a = extract(q);
        assert!(a.module_paths.contains(&"sympy.parsing.latex".to_string()));
        assert!(a.symbol_names.contains(&"parse_latex".to_string()));
    }

    #[test]
    fn empty_query() {
        let a = extract("");
        assert!(a.symbol_names.is_empty());
        assert!(a.module_paths.is_empty());
    }

    #[test]
    fn dotted_prose_call_extracted() {
        // v1.2.b: "fix xr.where keep_attrs overwriting coordinate attributes"
        // should extract xr.where (dotted) and where (last segment), both
        // marked as from_dotted_call — and also keep_attrs as a regular prose
        // identifier.
        let a = extract("fix xr.where keep_attrs overwriting coordinate attributes");
        assert!(a.symbol_names.contains(&"xr.where".to_string()));
        assert!(a.symbol_names.contains(&"where".to_string()));
        assert!(a.symbol_names.contains(&"keep_attrs".to_string()));
        assert!(a.from_dotted_call.contains("xr.where"));
        assert!(a.from_dotted_call.contains("where"));
        // keep_attrs is a plain snake_case identifier, not from a dotted call.
        assert!(!a.from_dotted_call.contains("keep_attrs"));
    }

    #[test]
    fn dotted_prose_three_segments() {
        // Multi-level dotted chains: only the full chain and the last segment
        // are pushed; intermediate segments are not (keeps precision high).
        let a = extract("calls urllib.request.urlopen internally");
        assert!(a
            .symbol_names
            .contains(&"urllib.request.urlopen".to_string()));
        assert!(a.symbol_names.contains(&"urlopen".to_string()));
        assert!(a.from_dotted_call.contains("urllib.request.urlopen"));
        assert!(a.from_dotted_call.contains("urlopen"));
    }

    #[test]
    fn code_block_dotted_marked_as_from_dotted_call() {
        // v1.2.c: code-block dotted calls should also populate from_dotted_call.
        let q = "```python\nxr.where(cond, a, b)\n```";
        let a = extract(q);
        assert!(a.from_dotted_call.contains("xr.where"));
        assert!(a.from_dotted_call.contains("where"));
    }
}
