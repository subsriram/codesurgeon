use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Language {
    Python,
    TypeScript,
    Tsx,
    JavaScript,
    Jsx,
    Shell,
    Html,
    Rust,
    Swift,
    Sql,
    Markdown,
}

impl Language {
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Python => "python",
            Language::TypeScript => "typescript",
            Language::Tsx => "tsx",
            Language::JavaScript => "javascript",
            Language::Jsx => "jsx",
            Language::Shell => "shell",
            Language::Html => "html",
            Language::Rust => "rust",
            Language::Swift => "swift",
            Language::Sql => "sql",
            Language::Markdown => "markdown",
        }
    }

    /// Returns the tree-sitter Language for grammars we have bindings for.
    pub fn tree_sitter_language(&self) -> Option<tree_sitter::Language> {
        match self {
            Language::Python => Some(tree_sitter_python::LANGUAGE.into()),
            Language::TypeScript => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            Language::Tsx => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
            Language::JavaScript | Language::Jsx => Some(tree_sitter_javascript::LANGUAGE.into()),
            Language::Shell => Some(tree_sitter_bash::LANGUAGE.into()),
            Language::Html => Some(tree_sitter_html::LANGUAGE.into()),
            Language::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
            Language::Swift => Some(tree_sitter_swift::LANGUAGE.into()),
            Language::Sql => Some(tree_sitter_sequel::LANGUAGE.into()),
            Language::Markdown => Some(tree_sitter_md_025::LANGUAGE.into()),
        }
    }

    pub fn uses_tree_sitter(&self) -> bool {
        true
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Detect language from file extension.
pub fn detect_language(path: &Path) -> Option<Language> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    match ext.as_str() {
        "py" | "pyw" | "pyi" => Some(Language::Python),
        "ts" => Some(Language::TypeScript),
        "tsx" => Some(Language::Tsx),
        "js" | "mjs" | "cjs" => Some(Language::JavaScript),
        "jsx" => Some(Language::Jsx),
        "sh" | "bash" | "zsh" | "fish" => Some(Language::Shell),
        "html" | "htm" => Some(Language::Html),
        "rs" => Some(Language::Rust),
        "swift" | "swiftinterface" => Some(Language::Swift),
        "sql" => Some(Language::Sql),
        "md" | "mdx" => Some(Language::Markdown),
        _ => None,
    }
}

