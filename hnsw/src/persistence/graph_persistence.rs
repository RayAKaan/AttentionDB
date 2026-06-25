use crate::hnsw_index::{HNSWConfig, HNSWIndex};
use crate::persistence::error::PersistenceError;
use crate::persistence::strategy::PersistenceStrategy;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub version: u32,
    pub head_name: String,
    pub dim: usize,
    pub config: HNSWConfig,
    pub vector_count: usize,
    pub created_at: String,
    pub insertion_order: Vec<u64>,
}

pub struct GraphPersistence;

impl PersistenceStrategy for GraphPersistence {
    fn save(&self, index: &HNSWIndex, dir: &Path) -> Result<(), PersistenceError> {
        std::fs::create_dir_all(dir)?;

        let insertion_order: Vec<u64> = index.vectors.iter().map(|(id, _)| *id).collect();

        let metadata = GraphMetadata {
            version: 3,
            head_name: index.head_name.clone(),
            dim: index.dim,
            config: index.config.clone(),
            vector_count: index.len(),
            created_at: chrono::Utc::now().to_rfc3339(),
            insertion_order,
        };

        let meta_path = dir.join("graph_metadata.json");
        let meta_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        std::fs::write(&meta_path, meta_json)?;

        let vectors_path = dir.join("vectors.bin");
        let mut file = File::create(&vectors_path)?;
        file.write_all(&(index.len() as u64).to_le_bytes())?;

        for (id, vec) in &index.vectors {
            file.write_all(&id.to_le_bytes())?;
            file.write_all(&(vec.len() as u32).to_le_bytes())?;
            for val in vec {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        Ok(())
    }

    fn load(&self, dir: &Path) -> Result<HNSWIndex, PersistenceError> {
        let meta_path = dir.join("graph_metadata.json");
        if !meta_path.exists() {
            return Err(PersistenceError::IndexNotFound(
                dir.to_string_lossy().to_string(),
            ));
        }

        let meta_json = std::fs::read_to_string(&meta_path)?;
        let metadata: GraphMetadata = serde_json::from_str(&meta_json)
            .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;

        if metadata.version != 3 {
            return Err(PersistenceError::InvalidMetadata(format!(
                "Unsupported graph persistence version: {}",
                metadata.version
            )));
        }

        let vectors_path = dir.join("vectors.bin");
        let mut file = File::open(&vectors_path)?;
        let mut count_buf = [0u8; 8];
        file.read_exact(&mut count_buf)?;
        let vector_count = u64::from_le_bytes(count_buf) as usize;

        let mut vectors = Vec::with_capacity(vector_count);
        for _ in 0..vector_count {
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
        }

        let mut index = HNSWIndex::new(&metadata.head_name, metadata.dim, metadata.config);

        for (id, vec) in vectors {
            index
                .insert(id, &vec)
                .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        }

        Ok(index)
    }

    fn append(&self, dir: &Path, vectors: &[(u64, Vec<f32>)]) -> Result<usize, PersistenceError> {
        crate::persistence::index_persistence::append_vectors(dir, vectors)
    }
}
