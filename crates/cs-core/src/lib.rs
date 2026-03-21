pub mod capsule;
pub mod db;
pub mod embedder;
pub mod engine;
pub mod graph;
pub mod indexer;
pub mod language;
pub mod memory;
pub mod search;
pub mod skeletonizer;
pub mod symbol;
pub mod watcher;

// Re-export the main entry point
pub use capsule::Capsule;
pub use engine::{CoreEngine, EngineConfig, IndexStats};
pub use memory::Observation;
pub use symbol::{Edge, EdgeKind, Symbol, SymbolKind};
