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
        }
    }

    /// Returns the tree-sitter Language for grammars we have bindings for.
    /// SQL uses a regex fallback (no grammar crate).
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
            // SQL: no stable grammar crate — use regex fallback
            Language::Sql => None,
        }
    }

    pub fn uses_tree_sitter(&self) -> bool {
        !matches!(self, Language::Sql)
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
        "py" | "pyw" => Some(Language::Python),
        "ts" => Some(Language::TypeScript),
        "tsx" => Some(Language::Tsx),
        "js" | "mjs" | "cjs" => Some(Language::JavaScript),
        "jsx" => Some(Language::Jsx),
        "sh" | "bash" | "zsh" | "fish" => Some(Language::Shell),
        "html" | "htm" => Some(Language::Html),
        "rs" => Some(Language::Rust),
        "swift" => Some(Language::Swift),
        "sql" => Some(Language::Sql),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn detects_python() {
        assert_eq!(
            detect_language(&PathBuf::from("foo.py")),
            Some(Language::Python)
        );
    }

    #[test]
    fn detects_tsx() {
        assert_eq!(
            detect_language(&PathBuf::from("App.tsx")),
            Some(Language::Tsx)
        );
    }

    #[test]
    fn detects_rust() {
        assert_eq!(
            detect_language(&PathBuf::from("main.rs")),
            Some(Language::Rust)
        );
    }
}
