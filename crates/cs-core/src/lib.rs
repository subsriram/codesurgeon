pub mod capsule;
pub mod db;
pub mod diff;
pub mod edges;
#[cfg(feature = "embeddings")]
pub mod emb_store;
pub mod embedder;
pub mod engine;
pub mod graph;
pub mod indexer;
pub mod language;
pub mod macro_expand;
pub mod memory;
pub mod module_docs;
pub mod pyright_enrich;
pub mod ranking;
pub mod rustdoc_enrich;
pub mod search;
pub mod skeletonizer;
pub mod symbol;
pub mod ts_enrich;
pub mod watcher;

// Re-export the main entry point
pub use capsule::Capsule;
pub use engine::{CoreEngine, EngineConfig, IndexStats, SessionContext};
pub use memory::Observation;
pub use symbol::{Edge, EdgeKind, Symbol, SymbolKind};
