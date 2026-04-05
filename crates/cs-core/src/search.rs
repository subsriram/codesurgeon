use crate::language::Language;
use crate::symbol::{Symbol, SymbolKind};
use anyhow::Result;
use tantivy::{
    collector::TopDocs,
    doc,
    query::QueryParser,
    schema::{Schema, SchemaBuilder, Value, FAST, STORED, STRING, TEXT},
    Index, IndexWriter, ReloadPolicy,
};

// ── Rerank constants ─────────────────────────────────────────────────────────
// See docs/ranking.md for rationale. Update both when tuning.

/// Boost per query term found in symbol name.
const NAME_TERM_BOOST: f32 = 2.0;
/// Boost per query term found in symbol signature.
const SIG_TERM_BOOST: f32 = 1.0;
/// Boost per query term found in symbol body.
const BODY_TERM_BOOST: f32 = 0.3;
/// Score multiplier for test/spec/mock files.
const TEST_FILE_PENALTY: f32 = 0.25;
/// Score multiplier for utility scripts (check-*, run-*, etc.).
const UTILITY_FILE_PENALTY: f32 = 0.3;
/// Type definition boost for Structural/Explore intent.
const TYPE_DEF_BOOST: f32 = 2.5;
/// Impl block boost for Structural/Explore intent.
const IMPL_BOOST: f32 = 1.5;
/// Callable penalty for Structural/Explore intent.
const CALLABLE_PENALTY: f32 = 0.6;
/// Markdown language boost.
const MARKDOWN_BOOST: f32 = 1.5;

/// Wraps a Tantivy in-RAM index for BM25 full-text search over symbols.
/// Complements the SQLite FTS5 search with richer scoring.
pub struct SearchIndex {
    index: Index,
    writer: IndexWriter,
    schema: SearchSchema,
}

struct SearchSchema {
    _schema: Schema,
    f_id: tantivy::schema::Field,
    f_name: tantivy::schema::Field,
    f_fqn: tantivy::schema::Field,
    f_signature: tantivy::schema::Field,
    f_docstring: tantivy::schema::Field,
    f_body: tantivy::schema::Field,
    f_file: tantivy::schema::Field,
}

impl SearchIndex {
    pub fn new() -> Result<Self> {
        let mut b: SchemaBuilder = Schema::builder();

        let f_id = b.add_u64_field("id", STORED | FAST);
        let f_name = b.add_text_field("name", TEXT | STORED);
        let f_fqn = b.add_text_field("fqn", STRING | STORED);
        let f_signature = b.add_text_field("signature", TEXT);
        let f_docstring = b.add_text_field("docstring", TEXT);
        let f_body = b.add_text_field("body", TEXT);
        let f_file = b.add_text_field("file", STRING | STORED);

        let schema = b.build();
        let index = Index::create_in_ram(schema.clone());
        let writer = index.writer(50_000_000)?; // 50 MB heap

        Ok(SearchIndex {
            index,
            writer,
            schema: SearchSchema {
                _schema: schema,
                f_id,
                f_name,
                f_fqn,
                f_signature,
                f_docstring,
                f_body,
                f_file,
            },
        })
    }

    /// Add or update a symbol in the search index.
    pub fn index_symbol(&mut self, sym: &Symbol) -> Result<()> {
        // Delete existing entry for this symbol id to avoid duplicates
        let id_term = tantivy::Term::from_field_u64(self.schema.f_id, sym.id);
        self.writer.delete_term(id_term);

        self.writer.add_document(doc!(
            self.schema.f_id        => sym.id,
            self.schema.f_name      => sym.name.as_str(),
            self.schema.f_fqn       => sym.fqn.as_str(),
            self.schema.f_signature => sym.signature.as_str(),
            self.schema.f_docstring => sym.docstring.as_deref().unwrap_or(""),
            self.schema.f_body      => sym.body.as_str(),
            self.schema.f_file      => sym.file_path.as_str(),
        ))?;
        Ok(())
    }

    /// Commit all pending writes.
    pub fn commit(&mut self) -> Result<()> {
        self.writer.commit()?;
        Ok(())
    }

    /// Search across name, signature, docstring, and body.
    /// Returns symbol IDs ranked by BM25 score.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<(u64, f32)>> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let searcher = reader.searcher();

        let query_parser = QueryParser::for_index(
            &self.index,
            vec![
                self.schema.f_name,
                self.schema.f_signature,
                self.schema.f_docstring,
                self.schema.f_body,
            ],
        );

        // Gracefully handle malformed queries
        let parsed = match query_parser.parse_query(query) {
            Ok(q) => q,
            Err(_) => {
                // Fall back to a fuzzy name search
                query_parser
                    .parse_query(&escape_for_tantivy(query))
                    .unwrap_or_else(|_| query_parser.parse_query("*").unwrap())
            }
        };

        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(limit))?;

        let mut results = Vec::new();
        for (score, doc_addr) in top_docs {
            let retrieved: tantivy::TantivyDocument = searcher.doc(doc_addr)?;
            if let Some(id_val) = retrieved.get_first(self.schema.f_id) {
                if let Some(id) = id_val.as_u64() {
                    results.push((id, score));
                }
            }
        }
        Ok(results)
    }

    /// TF-IDF re-ranking on top of BM25 results.
    /// Applies name/signature term boosts, file path penalties (test/utility files),
    /// and symbol-kind boosts for structural/explore queries.
    pub fn rerank_by_query_proximity(
        results: Vec<(u64, f32)>,
        symbols: &[&Symbol],
        query: &str,
        intent: &SearchIntent,
    ) -> Vec<(u64, f32)> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut rescored: Vec<(u64, f32)> = results
            .into_iter()
            .map(|(id, score)| {
                let boost = symbols
                    .iter()
                    .find(|s| s.id == id)
                    .map(|sym| {
                        let name_lower = sym.name.to_lowercase();
                        let sig_lower = sym.signature.to_lowercase();
                        let path_lower = sym.file_path.to_lowercase();
                        let filename = path_lower.rsplit('/').next().unwrap_or(&path_lower);
                        let mut b = 1.0f32;

                        // Name/signature/body term matching
                        let body_lower = sym.body.to_lowercase();
                        for term in &query_terms {
                            if name_lower.contains(term) {
                                b += NAME_TERM_BOOST;
                            }
                            if sig_lower.contains(term) {
                                b += SIG_TERM_BOOST;
                            }
                            if body_lower.contains(term) {
                                b += BODY_TERM_BOOST;
                            }
                        }

                        // Test/spec file penalty — test setup/mocks are rarely the architectural answer
                        let is_test = path_lower.contains("test")
                            || path_lower.contains("spec")
                            || path_lower.contains("mock")
                            || path_lower.contains("uitest");
                        if is_test {
                            b *= TEST_FILE_PENALTY;
                        }

                        // Utility script penalty — check-*, run-*, setup*, generate*, etc.
                        let is_utility = filename.starts_with("check-")
                            || filename.starts_with("run-")
                            || filename.starts_with("setup")
                            || filename.starts_with("generate")
                            || filename.starts_with("gen-")
                            || filename.starts_with("build-")
                            || filename.starts_with("deploy-");
                        if is_utility {
                            b *= UTILITY_FILE_PENALTY;
                        }

                        // For Structural/Explore intents: boost type definitions, reduce raw callables
                        match intent {
                            SearchIntent::Structural | SearchIntent::Explore => {
                                if sym.kind.is_type_definition() {
                                    b *= TYPE_DEF_BOOST;
                                } else if sym.kind == SymbolKind::Impl {
                                    b *= IMPL_BOOST;
                                } else if sym.kind.is_callable() && !is_test {
                                    b *= CALLABLE_PENALTY;
                                }
                            }
                            _ => {}
                        }

                        // Markdown docs boost — documentation sections are preferred for
                        // conceptual queries where terms appear in prose, not code names.
                        if sym.language == Language::Markdown {
                            b *= MARKDOWN_BOOST;
                        }

                        b
                    })
                    .unwrap_or(1.0);
                (id, score * boost)
            })
            .collect();

        rescored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored
    }
}

fn escape_for_tantivy(query: &str) -> String {
    // Escape special tantivy query chars
    query
        .chars()
        .map(|c| {
            if "+-&|!(){}[]^\"~*?:\\".contains(c) {
                format!("\\{}", c)
            } else {
                c.to_string()
            }
        })
        .collect()
}

/// Intent detection: map a task description to a search strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchIntent {
    /// "fix", "bug", "error", "exception" — follow error paths
    Debug,
    /// "refactor", "rename", "move", "restructure" — blast-radius mode
    Refactor,
    /// "add", "implement", "create", "feature" — exploration mode
    Add,
    /// "understand", "explain", "how does", "what is" — exploration
    Explore,
    /// "coordinator", "central", "manager", "architecture" — boost type definitions
    Structural,
    /// Default
    General,
}

impl SearchIntent {
    pub fn as_str(&self) -> &'static str {
        match self {
            SearchIntent::Debug => "debug",
            SearchIntent::Refactor => "refactor",
            SearchIntent::Add => "add",
            SearchIntent::Explore => "explore",
            SearchIntent::Structural => "structural",
            SearchIntent::General => "general",
        }
    }

    pub fn detect(task: &str) -> Self {
        let lower = task.to_lowercase();
        if contains_any(
            &lower,
            &[
                "fix",
                "bug",
                "error",
                "exception",
                "crash",
                "fail",
                "broken",
            ],
        ) {
            SearchIntent::Debug
        } else if contains_any(
            &lower,
            &[
                "refactor",
                "rename",
                "move",
                "restructure",
                "migrate",
                "replace",
            ],
        ) {
            SearchIntent::Refactor
        } else if contains_any(
            &lower,
            &[
                "coordinator",
                "central",
                "manager",
                "architecture",
                "orchestrat",
                "hub",
                "entry point",
                "main class",
                "core class",
                "state machine",
                "controller",
            ],
        ) {
            SearchIntent::Structural
        } else if contains_any(
            &lower,
            &["understand", "explain", "how does", "what is", "what does"],
        ) {
            SearchIntent::Explore
        } else if contains_any(
            &lower,
            &[
                "add",
                "implement",
                "create",
                "build",
                "new feature",
                "support",
            ],
        ) {
            SearchIntent::Add
        } else {
            SearchIntent::General
        }
    }
}

fn contains_any(s: &str, terms: &[&str]) -> bool {
    terms.iter().any(|t| s.contains(t))
}
