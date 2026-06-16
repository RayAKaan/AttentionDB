use crate::hnsw_index::HNSWIndex;
use crate::persistence::error::PersistenceError;
use std::path::Path;

/// Strategy interface for persisting and loading HNSW indexes.
pub trait PersistenceStrategy {
    /// Save the index to the given directory.
    fn save(&self, index: &HNSWIndex, dir: &Path) -> Result<(), PersistenceError>;

    /// Load an index from the given directory.
    fn load(&self, dir: &Path) -> Result<HNSWIndex, PersistenceError>;

    /// Append new vectors to an existing persisted index.
    fn append(&self, dir: &Path, vectors: &[(u64, Vec<f32>)]) -> Result<usize, PersistenceError>;
}

/// Current default implementation: Save vectors + metadata, rebuild graph on load.
pub struct VectorRebuildPersistence;

impl PersistenceStrategy for VectorRebuildPersistence {
    fn save(&self, index: &HNSWIndex, dir: &Path) -> Result<(), PersistenceError> {
        crate::persistence::index_persistence::save_index(index, dir)
    }

    fn load(&self, dir: &Path) -> Result<HNSWIndex, PersistenceError> {
        crate::persistence::index_persistence::load_index(dir, None::<fn(crate::persistence::LoadProgress)>)
    }

    fn append(&self, dir: &Path, vectors: &[(u64, Vec<f32>)]) -> Result<usize, PersistenceError> {
        crate::persistence::index_persistence::append_vectors(dir, vectors)
    }
}
