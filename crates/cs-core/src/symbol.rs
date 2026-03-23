use crate::language::Language;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SymbolKind {
    Function,
    AsyncFunction,
    Class,
    Method,
    AsyncMethod,
    Interface,
    TypeAlias,
    Enum,
    Struct,
    Trait,
    Impl,
    Constant,
    Variable,
    Import,
    Module,
    Macro,
    // For HTML script/style blocks
    ScriptBlock,
    StyleBlock,
}

impl SymbolKind {
    pub fn is_callable(&self) -> bool {
        matches!(
            self,
            SymbolKind::Function
                | SymbolKind::AsyncFunction
                | SymbolKind::Method
                | SymbolKind::AsyncMethod
        )
    }

    pub fn is_type_definition(&self) -> bool {
        matches!(
            self,
            SymbolKind::Class
                | SymbolKind::Interface
                | SymbolKind::TypeAlias
                | SymbolKind::Enum
                | SymbolKind::Struct
                | SymbolKind::Trait
        )
    }
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            SymbolKind::Function => "fn",
            SymbolKind::AsyncFunction => "async fn",
            SymbolKind::Class => "class",
            SymbolKind::Method => "method",
            SymbolKind::AsyncMethod => "async method",
            SymbolKind::Interface => "interface",
            SymbolKind::TypeAlias => "type",
            SymbolKind::Enum => "enum",
            SymbolKind::Struct => "struct",
            SymbolKind::Trait => "trait",
            SymbolKind::Impl => "impl",
            SymbolKind::Constant => "const",
            SymbolKind::Variable => "var",
            SymbolKind::Import => "import",
            SymbolKind::Module => "mod",
            SymbolKind::Macro => "macro",
            SymbolKind::ScriptBlock => "script",
            SymbolKind::StyleBlock => "style",
        };
        write!(f, "{s}")
    }
}

/// A single named entity in the codebase.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Symbol {
    /// Stable ID: hash of (file_path + "::" + name + "::" + start_line).
    pub id: u64,

    /// Fully-qualified name, e.g. "src/auth/service.py::AuthService::validate_token"
    pub fqn: String,

    /// Simple name, e.g. "validate_token"
    pub name: String,

    pub kind: SymbolKind,

    /// Path relative to workspace root
    pub file_path: String,

    pub start_line: u32,
    pub end_line: u32,

    /// First line(s) only: function signature, class header, etc.
    pub signature: String,

    /// Docstring / JSDoc / rustdoc, if present
    pub docstring: Option<String>,

    /// Full source text of the symbol (including signature + body)
    pub body: String,

    pub language: Language,

    /// blake3 hash of `body` — used for stale observation detection
    pub content_hash: String,
}

impl Symbol {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        file_path: &str,
        name: &str,
        kind: SymbolKind,
        start_line: u32,
        end_line: u32,
        signature: String,
        docstring: Option<String>,
        body: String,
        language: Language,
    ) -> Self {
        let id = compute_id(file_path, name, start_line);
        let fqn = format!("{}::{}", file_path, name);
        let content_hash = blake3::hash(body.as_bytes()).to_hex().to_string();

        Symbol {
            id,
            fqn,
            name: name.to_string(),
            kind,
            file_path: file_path.to_string(),
            start_line,
            end_line,
            signature,
            docstring,
            body,
            language,
            content_hash,
        }
    }

    /// Returns the skeleton: signature + optional docstring, no body.
    pub fn skeleton(&self) -> String {
        match &self.docstring {
            Some(doc) => format!("{}\n{}", doc, self.signature),
            None => self.signature.clone(),
        }
    }

    /// Rough token estimate (chars / 4).
    pub fn token_estimate(&self) -> u32 {
        (self.body.len() / 4) as u32
    }

    pub fn skeleton_token_estimate(&self) -> u32 {
        (self.skeleton().len() / 4) as u32
    }
}

fn compute_id(file_path: &str, name: &str, start_line: u32) -> u64 {
    let mut h = DefaultHasher::new();
    file_path.hash(&mut h);
    "::".hash(&mut h);
    name.hash(&mut h);
    "::".hash(&mut h);
    start_line.hash(&mut h);
    h.finish()
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EdgeKind {
    /// A imports B
    Imports,
    /// A calls B
    Calls,
    /// A implements interface/trait B
    Implements,
    /// A extends/inherits B
    Inherits,
    /// A references type B
    References,
    /// A is defined inside B (e.g. method inside class)
    DefinedIn,
}

impl std::fmt::Display for EdgeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            EdgeKind::Imports => "imports",
            EdgeKind::Calls => "calls",
            EdgeKind::Implements => "implements",
            EdgeKind::Inherits => "inherits",
            EdgeKind::References => "references",
            EdgeKind::DefinedIn => "defined_in",
        };
        write!(f, "{s}")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Edge {
    pub from_id: u64,
    pub to_id: u64,
    pub kind: EdgeKind,
    /// The source text that caused this edge (e.g. the import path)
    pub label: Option<String>,
}

impl Edge {
    pub fn new(from_id: u64, to_id: u64, kind: EdgeKind) -> Self {
        Edge {
            from_id,
            to_id,
            kind,
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}
