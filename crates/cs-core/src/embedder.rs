//! Optional local embeddings via fastembed.
//!
//! Compiled only when the `embeddings` or `metal` feature is enabled:
//!
//! ```bash
//! cargo build --features embeddings          # CPU (any platform)
//! cargo build --features metal               # CPU + Apple Accelerate BLAS (Apple Silicon)
//! ```
//!
//! Uses `NomicEmbedTextV15Q` (nomic-ai/nomic-embed-text-v1.5, quantized) — 768-dim, 8192 context.
//! The quantized variant uses int8 weights (~130 MB vs ~520 MB for the full model) with
//! negligible quality loss for code search, and loads ~4x less RAM at runtime.
//! Embeddings are 768-dimensional, L2-normalised, so dot-product == cosine similarity.

#[cfg(feature = "embeddings")]
pub use inner::{cosine_similarity, Embedder};

#[cfg(feature = "embeddings")]
mod inner {
    use anyhow::Result;
    use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
    use parking_lot::Mutex;

    /// Wraps fastembed's `TextEmbedding` behind a `Mutex` because its `embed` method
    /// requires `&mut self` while `CoreEngine` methods take `&self`.
    pub struct Embedder {
        model: Mutex<TextEmbedding>,
    }

    impl Embedder {
        /// Load the model, downloading it on first use.
        ///
        /// The model is cached in `~/.cache/fastembed` (shared across all workspaces)
        /// so it is only downloaded once and never duplicated per-project.
        pub fn new() -> Result<Self> {
            // Use a single shared cache dir so every workspace finds the model
            // in the same place and we avoid per-workspace 49 MB copies.
            let cache_dir = std::env::var("HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join(".cache")
                .join("fastembed");

            let model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::NomicEmbedTextV15Q)
                    .with_cache_dir(cache_dir)
                    .with_show_download_progress(false),
            )
            .map_err(|e| anyhow::anyhow!("fastembed init failed: {e}"))?;
            Ok(Self {
                model: Mutex::new(model),
            })
        }

        /// Embed a batch of texts. Returns one 384-dim vector per input, L2-normalised.
        pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            self.model
                .lock()
                .embed(texts.to_vec(), None)
                .map_err(|e| anyhow::anyhow!("embed failed: {e}"))
        }

        /// Embed a single string.
        pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
            let mut out = self.embed_batch(&[text])?;
            out.pop()
                .ok_or_else(|| anyhow::anyhow!("empty embedding result"))
        }
    }

    /// Cosine similarity between two L2-normalised vectors (dot product).
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}
