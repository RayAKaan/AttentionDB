use crate::hnsw_index::HNSWIndex;
use crate::persistence::error::PersistenceError;
use crate::persistence::index_persistence;
use std::path::Path;

/// Trait for HNSW index persistence strategies.
///
/// The current `VectorPersistence` implementation saves vectors + metadata
/// and rebuilds the HNSW graph on load. In the future, a `GraphPersistence`
/// implementation could serialize the actual graph structure for faster loads.
pub trait IndexPersistence {
    /// Save the index to disk.
    fn save(&self, index: &HNSWIndex, dir: &Path) -> Result<(), PersistenceError>;

    /// Load the index from disk.
    fn load(&self, dir: &Path) -> Result<HNSWIndex, PersistenceError>;

    /// Load the index with progress reporting.
    fn load_with_progress<F>(&self, dir: &Path, callback: F) -> Result<HNSWIndex, PersistenceError>
    where
        F: FnMut(index_persistence::LoadProgress);

    /// Append vectors to an existing persisted index.
    fn append(&self, dir: &Path, vectors: &[(u64, Vec<f32>)]) -> Result<usize, PersistenceError>;
}

/// Current vector-only persistence strategy (rebuilds graph on load).
pub struct VectorPersistence;

impl IndexPersistence for VectorPersistence {
    fn save(&self, index: &HNSWIndex, dir: &Path) -> Result<(), PersistenceError> {
        index_persistence::save_index(index, dir)
    }

    fn load(&self, dir: &Path) -> Result<HNSWIndex, PersistenceError> {
        index_persistence::load_index(dir, None::<fn(index_persistence::LoadProgress)>)
    }

    fn load_with_progress<F>(&self, dir: &Path, callback: F) -> Result<HNSWIndex, PersistenceError>
    where
        F: FnMut(index_persistence::LoadProgress),
    {
        index_persistence::load_index(dir, Some(callback))
    }

    fn append(&self, dir: &Path, vectors: &[(u64, Vec<f32>)]) -> Result<usize, PersistenceError> {
        index_persistence::append_vectors(dir, vectors)
    }
}
