//! Production-Grade HNSW Index Persistence Implementation
//!
//! This module provides robust, safe persistence for HNSW indexes.
//!
//! Current Strategy (Practical & Reliable):
//! - Save: Vectors + Configuration + Metadata + Checksum
//! - Load: Rebuild HNSW graph from vectors (deterministic)
//!
//! Future Improvement: Real graph serialization (requires changes to hnsw_rs or custom implementation).

use crate::hnsw_index::{HNSWConfig, HNSWIndex};
use crate::persistence::error::PersistenceError;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Metadata stored with each persisted index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub version: u32,
    pub head_name: String,
    pub dim: usize,
    pub config: HNSWConfig,
    pub vector_count: usize,
    pub created_at: String,
    pub checksum: Option<String>,
}

/// Save an HNSW index to disk (atomic write + checksum)
pub fn save_index(index: &HNSWIndex, dir: &Path) -> Result<(), PersistenceError> {
    std::fs::create_dir_all(dir)?;

    let checksum = index
        .vectors
        .iter()
        .flat_map(|(_, vec)| vec)
        .map(|v| v.to_bits() as u64)
        .sum::<u64>()
        .to_string();

    let metadata = IndexMetadata {
        version: 2,
        head_name: index.head_name.clone(),
        dim: index.dim,
        config: index.config.clone(),
        vector_count: index.len(),
        created_at: chrono::Utc::now().to_rfc3339(),
        checksum: Some(checksum),
    };

    let meta_path = dir.join("metadata.json");
    let temp_meta = dir.join("metadata.json.tmp");

    let meta_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| PersistenceError::Serialization(e.to_string()))?;

    std::fs::write(&temp_meta, meta_json)?;
    std::fs::rename(&temp_meta, &meta_path)?;

    let vectors_path = dir.join("vectors.bin");
    let temp_vectors = dir.join("vectors.bin.tmp");

    let mut file = File::create(&temp_vectors)?;
    file.write_all(&(index.len() as u64).to_le_bytes())?;

    for (id, vec) in &index.vectors {
        file.write_all(&id.to_le_bytes())?;
        file.write_all(&(vec.len() as u32).to_le_bytes())?;
        for val in vec {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    std::fs::rename(&temp_vectors, &vectors_path)?;

    Ok(())
}

/// Append new vectors to an existing persisted index (incremental update)
pub fn append_vectors(
    dir: &Path,
    new_vectors: &[(u64, Vec<f32>)],
) -> Result<usize, PersistenceError> {
    if new_vectors.is_empty() {
        return Ok(0);
    }

    let vectors_path = dir.join("vectors.bin");
    let mut existing_vectors = Vec::new();

    if vectors_path.exists() {
        let mut file = File::open(&vectors_path)?;
        let mut count_buf = [0u8; 8];
        file.read_exact(&mut count_buf)?;
        let existing_count = u64::from_le_bytes(count_buf) as usize;

        for _ in 0..existing_count {
            let mut id_buf = [0u8; 8];
            file.read_exact(&mut id_buf)?;
            let id = u64::from_le_bytes(id_buf);

            let mut dim_buf = [0u8; 4];
            file.read_exact(&mut dim_buf)?;
            let dim = u32::from_le_bytes(dim_buf) as usize;

            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                let mut val_buf = [0u8; 4];
                file.read_exact(&mut val_buf)?;
                vec.push(f32::from_le_bytes(val_buf));
            }

            existing_vectors.push((id, vec));
        }
    }

    existing_vectors.extend_from_slice(new_vectors);

    let temp_path = dir.join("vectors.bin.tmp");
    let mut file = File::create(&temp_path)?;
    file.write_all(&(existing_vectors.len() as u64).to_le_bytes())?;

    for (id, vec) in &existing_vectors {
        file.write_all(&id.to_le_bytes())?;
        file.write_all(&(vec.len() as u32).to_le_bytes())?;
        for val in vec {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    std::fs::rename(&temp_path, &vectors_path)?;

    Ok(existing_vectors.len())
}

/// Progress information during index loading
#[derive(Debug, Clone)]
pub struct LoadProgress {
    pub total_vectors: usize,
    pub loaded_vectors: usize,
}

fn migrate_metadata(metadata: &mut IndexMetadata) -> Result<(), PersistenceError> {
    match metadata.version {
        1 => {
            metadata.version = 2;
            if metadata.checksum.is_none() {
                metadata.checksum = Some("legacy".to_string());
            }
            Ok(())
        }
        2 => Ok(()),
        _ => Err(PersistenceError::InvalidMetadata(format!(
            "Unsupported persistence version: {}",
            metadata.version
        ))),
    }
}

/// Load an HNSW index from disk (rebuilds the graph).
///
/// Optionally accepts a progress callback that is invoked periodically during loading.
pub fn load_index<F>(dir: &Path, progress: Option<F>) -> Result<HNSWIndex, PersistenceError>
where
    F: FnMut(LoadProgress),
{
    load_index_inner(dir, progress)
}

fn load_index_inner<F>(
    dir: &Path,
    mut progress_callback: Option<F>,
) -> Result<HNSWIndex, PersistenceError>
where
    F: FnMut(LoadProgress),
{
    let meta_path = dir.join("metadata.json");
    if !meta_path.exists() {
        return Err(PersistenceError::IndexNotFound(
            dir.to_string_lossy().to_string(),
        ));
    }

    let meta_json = std::fs::read_to_string(&meta_path)?;
    let mut metadata: IndexMetadata = serde_json::from_str(&meta_json)
        .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;

    migrate_metadata(&mut metadata)?;

    let vectors_path = dir.join("vectors.bin");
    if !vectors_path.exists() {
        return Err(PersistenceError::IndexNotFound(
            vectors_path.to_string_lossy().to_string(),
        ));
    }

    let mut file = File::open(&vectors_path)?;
    let mut count_buf = [0u8; 8];
    file.read_exact(&mut count_buf)?;
    let vector_count = u64::from_le_bytes(count_buf) as usize;

    let mut vectors = Vec::with_capacity(vector_count);

    for i in 0..vector_count {
        let mut id_buf = [0u8; 8];
        file.read_exact(&mut id_buf)?;
        let id = u64::from_le_bytes(id_buf);

        let mut dim_buf = [0u8; 4];
        file.read_exact(&mut dim_buf)?;
        let dim = u32::from_le_bytes(dim_buf) as usize;

        if dim != metadata.dim {
            return Err(PersistenceError::DimensionMismatch {
                expected: metadata.dim,
                got: dim,
            });
        }

        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            let mut val_buf = [0u8; 4];
            file.read_exact(&mut val_buf)?;
            vec.push(f32::from_le_bytes(val_buf));
        }

        vectors.push((id, vec));

        if let Some(ref mut cb) = progress_callback {
            if i % 1000 == 0 || i == vector_count - 1 {
                cb(LoadProgress {
                    total_vectors: vector_count,
                    loaded_vectors: i + 1,
                });
            }
        }
    }

    if let Some(expected_checksum) = &metadata.checksum {
        let calculated_checksum: u64 = vectors
            .iter()
            .flat_map(|(_, vec)| vec)
            .map(|v| v.to_bits() as u64)
            .sum();
        if calculated_checksum.to_string() != *expected_checksum {
            return Err(PersistenceError::InvalidMetadata(
                "Checksum mismatch: data may be corrupted".to_string(),
            ));
        }
    }

    let mut index = HNSWIndex::new(&metadata.head_name, metadata.dim, metadata.config);

    for (id, vec) in vectors {
        let _ = index.insert(id, &vec);
    }

    Ok(index)
}
