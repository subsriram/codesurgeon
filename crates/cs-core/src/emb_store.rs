//! Flat-binary embedding store that can be memory-mapped.
//!
//! File layout (all little-endian):
//!   [magic: u8×8][count: u64][dim: u32][entries: (symbol_id: u64, f32×dim) × count]
//!
//! Using memmap2 keeps the data in OS-managed virtual memory so pages that
//! are not accessed in a given session are never brought into RSS.

#[cfg(feature = "embeddings")]
use anyhow::{bail, Result};
#[cfg(feature = "embeddings")]
use memmap2::Mmap;
#[cfg(feature = "embeddings")]
use std::fs::{File, OpenOptions};
#[cfg(feature = "embeddings")]
use std::path::Path;

#[cfg(feature = "embeddings")]
const MAGIC: &[u8; 8] = b"CSEMB001";
/// magic(8) + count(8) + dim(4)
#[cfg(feature = "embeddings")]
const HEADER_SIZE: usize = 20;

/// A flat embedding store backed by either a memory-mapped file or a heap Vec.
///
/// The mmap variant lets the OS page-out embeddings that are not accessed,
/// while the heap variant is used during the initial index build before the
/// file has been written.
#[cfg(feature = "embeddings")]
pub struct EmbeddingStore {
    inner: StoreInner,
    pub count: usize,
    pub dim: usize,
}

#[cfg(feature = "embeddings")]
enum StoreInner {
    Mmap(Mmap),
    Heap(Vec<(u64, Vec<f32>)>),
}

#[cfg(feature = "embeddings")]
impl EmbeddingStore {
    /// Open an existing `embeddings.bin` via mmap.
    /// Returns `None` if the file is absent, has a wrong magic, or its size is inconsistent.
    pub fn open(path: &Path) -> Option<Self> {
        let file = File::open(path).ok()?;
        let mmap = unsafe { Mmap::map(&file) }.ok()?;

        if mmap.len() < HEADER_SIZE {
            return None;
        }
        if &mmap[0..8] != MAGIC {
            return None;
        }

        let count = u64::from_le_bytes(mmap[8..16].try_into().ok()?) as usize;
        let dim = u32::from_le_bytes(mmap[16..20].try_into().ok()?) as usize;

        // Guard against corrupt dim/count that would overflow the size arithmetic.
        if dim == 0 || dim > 16384 || count > usize::MAX / (8 + dim * 4) {
            return None;
        }

        let expected = HEADER_SIZE + count * (8 + dim * 4);
        if mmap.len() != expected {
            return None;
        }

        Some(EmbeddingStore {
            inner: StoreInner::Mmap(mmap),
            count,
            dim,
        })
    }

    /// Write `entries` to `path` as a flat binary file, then mmap the result.
    pub fn write_and_open(path: &Path, entries: &[(u64, Vec<f32>)]) -> Result<Self> {
        if entries.is_empty() {
            bail!("cannot write empty embedding store");
        }
        let dim = entries[0].1.len();
        let count = entries.len();
        let total = HEADER_SIZE + count * (8 + dim * 4);

        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            file.set_len(total as u64)?;

            let mut m = unsafe { memmap2::MmapMut::map_mut(&file)? };
            m[0..8].copy_from_slice(MAGIC);
            m[8..16].copy_from_slice(&(count as u64).to_le_bytes());
            m[16..20].copy_from_slice(&(dim as u32).to_le_bytes());

            let mut off = HEADER_SIZE;
            for (id, vec) in entries {
                m[off..off + 8].copy_from_slice(&id.to_le_bytes());
                off += 8;
                for &f in vec {
                    m[off..off + 4].copy_from_slice(&f.to_le_bytes());
                    off += 4;
                }
            }
            m.flush()?;
        }

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(EmbeddingStore {
            inner: StoreInner::Mmap(mmap),
            count,
            dim,
        })
    }

    /// In-memory store built from a heap Vec (used before the file is written).
    pub fn from_heap(entries: Vec<(u64, Vec<f32>)>) -> Self {
        let (count, dim) = if entries.is_empty() {
            (0, 0)
        } else {
            (entries.len(), entries[0].1.len())
        };
        EmbeddingStore {
            inner: StoreInner::Heap(entries),
            count,
            dim,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Iterate over `(symbol_id, embedding_slice)` pairs.
    ///
    /// For the mmap backend the slices point directly into the mapped file — no
    /// heap allocation.  For the heap backend they borrow from the Vec.
    pub fn iter(&self) -> EmbeddingIter<'_> {
        EmbeddingIter {
            store: self,
            pos: 0,
        }
    }
}

#[cfg(feature = "embeddings")]
pub struct EmbeddingIter<'a> {
    store: &'a EmbeddingStore,
    pos: usize,
}

#[cfg(feature = "embeddings")]
impl<'a> Iterator for EmbeddingIter<'a> {
    type Item = (u64, &'a [f32]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.store.count {
            return None;
        }
        let item = match &self.store.inner {
            StoreInner::Mmap(mmap) => {
                let entry_size = 8 + self.store.dim * 4;
                let off = HEADER_SIZE + self.pos * entry_size;
                // Safety: open() validated mmap length == HEADER_SIZE + count * entry_size,
                // and pos < count is checked above, so this slice is always 8 bytes.
                let id = u64::from_le_bytes(match mmap[off..off + 8].try_into() {
                    Ok(b) => b,
                    Err(_) => return None,
                });
                let floats_bytes = &mmap[off + 8..off + entry_size];
                // Safety: the file layout guarantees 4-byte alignment for the
                // float region (HEADER_SIZE=20 + entry_size multiples of 4) and
                // the mmap starts at a page-aligned address, so all f32 slices
                // within the mapped region are properly aligned.
                let floats: &[f32] = unsafe {
                    std::slice::from_raw_parts(floats_bytes.as_ptr() as *const f32, self.store.dim)
                };
                (id, floats)
            }
            StoreInner::Heap(entries) => {
                let (id, vec) = &entries[self.pos];
                (*id, vec.as_slice())
            }
        };
        self.pos += 1;
        Some(item)
    }
}

#[cfg(all(test, feature = "embeddings"))]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_entries(n: usize, dim: usize) -> Vec<(u64, Vec<f32>)> {
        (0..n)
            .map(|i| {
                let id = (i + 1) as u64 * 100;
                let vec: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 / 1000.0).collect();
                (id, vec)
            })
            .collect()
    }

    #[test]
    fn heap_store_iter_matches_input() {
        let entries = make_entries(5, 4);
        let store = EmbeddingStore::from_heap(entries.clone());
        assert_eq!(store.len(), 5);
        assert!(!store.is_empty());
        let collected: Vec<(u64, Vec<f32>)> =
            store.iter().map(|(id, v)| (id, v.to_vec())).collect();
        assert_eq!(collected, entries);
    }

    #[test]
    fn empty_heap_store() {
        let store = EmbeddingStore::from_heap(vec![]);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.iter().count(), 0);
    }

    #[test]
    fn write_and_open_round_trips_data() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("embeddings.bin");
        let entries = make_entries(10, 8);

        let store = EmbeddingStore::write_and_open(&path, &entries).unwrap();
        assert_eq!(store.len(), 10);
        assert_eq!(store.dim, 8);
        let collected: Vec<(u64, Vec<f32>)> =
            store.iter().map(|(id, v)| (id, v.to_vec())).collect();
        assert_eq!(collected, entries);
    }

    #[test]
    fn open_mmap_after_write_round_trips_data() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("embeddings.bin");
        let entries = make_entries(6, 16);
        EmbeddingStore::write_and_open(&path, &entries).unwrap();

        // Reopen simulating a new process startup.
        let store = EmbeddingStore::open(&path).expect("open should succeed");
        assert_eq!(store.len(), 6);
        assert_eq!(store.dim, 16);
        let collected: Vec<(u64, Vec<f32>)> =
            store.iter().map(|(id, v)| (id, v.to_vec())).collect();
        assert_eq!(collected, entries);
    }

    #[test]
    fn open_missing_file_returns_none() {
        let dir = TempDir::new().unwrap();
        assert!(EmbeddingStore::open(&dir.path().join("nonexistent.bin")).is_none());
    }

    #[test]
    fn open_wrong_magic_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.bin");
        std::fs::write(&path, b"BADMAGIC00000000000000000000").unwrap();
        assert!(EmbeddingStore::open(&path).is_none());
    }

    #[test]
    fn open_truncated_file_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("truncated.bin");
        // Valid header claiming count=1 dim=4, but no entry bytes.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"CSEMB001");
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        std::fs::write(&path, &buf).unwrap();
        assert!(EmbeddingStore::open(&path).is_none());
    }

    #[test]
    fn write_errors_on_empty_input() {
        let dir = TempDir::new().unwrap();
        assert!(EmbeddingStore::write_and_open(&dir.path().join("empty.bin"), &[]).is_err());
    }

    #[test]
    fn open_dim_zero_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("dim0.bin");
        let mut buf = Vec::new();
        buf.extend_from_slice(b"CSEMB001");
        buf.extend_from_slice(&1u64.to_le_bytes()); // count = 1
        buf.extend_from_slice(&0u32.to_le_bytes()); // dim = 0 (invalid)
        std::fs::write(&path, &buf).unwrap();
        assert!(EmbeddingStore::open(&path).is_none());
    }

    #[test]
    fn open_dim_overflow_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("dim_huge.bin");
        let mut buf = Vec::new();
        buf.extend_from_slice(b"CSEMB001");
        buf.extend_from_slice(&1u64.to_le_bytes()); // count = 1
        buf.extend_from_slice(&20000u32.to_le_bytes()); // dim > 16384 limit
        std::fs::write(&path, &buf).unwrap();
        assert!(EmbeddingStore::open(&path).is_none());
    }

    #[test]
    fn open_count_overflow_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("count_huge.bin");
        let mut buf = Vec::new();
        buf.extend_from_slice(b"CSEMB001");
        buf.extend_from_slice(&u64::MAX.to_le_bytes()); // count = MAX (overflow)
        buf.extend_from_slice(&4u32.to_le_bytes()); // dim = 4
        std::fs::write(&path, &buf).unwrap();
        assert!(EmbeddingStore::open(&path).is_none());
    }
}
