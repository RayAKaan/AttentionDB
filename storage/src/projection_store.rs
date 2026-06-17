use std::collections::HashMap;
use std::fs::File;
use std::io::{Write, Read, Seek, SeekFrom};
use std::path::Path;
use crate::error::StorageError;

pub struct ProjectionStore {
    file: File,
    head_dim: usize,
    offsets: HashMap<String, u64>,
}

impl ProjectionStore {
    pub fn new(path: &Path, head_dim: usize) -> Result<Self, StorageError> {
        let file = File::create(path)?;
        Ok(Self { file, head_dim, offsets: HashMap::new() })
    }

    pub fn open(path: &Path, head_dim: usize) -> Result<Self, StorageError> {
        let file = File::open(path)?;
        Ok(Self { file, head_dim, offsets: HashMap::new() })
    }

    pub fn append_k_vector(&mut self, head: &str, vec: &[f32]) -> Result<u64, StorageError> {
        let offset = self.file.seek(SeekFrom::End(0))
            .map_err(|e| StorageError::Projection(e.to_string()))?;
        let bytes: Vec<u8> = vec.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        self.file.write_all(&bytes)
            .map_err(|e| StorageError::Projection(e.to_string()))?;
        self.offsets.entry(head.to_string()).or_insert(offset);
        Ok(offset)
    }

    pub fn read_vector(&mut self, offset: u64) -> Result<Vec<f32>, StorageError> {
        self.file.seek(SeekFrom::Start(offset))
            .map_err(|e| StorageError::Projection(e.to_string()))?;
        let mut buf = vec![0u8; self.head_dim * 4];
        self.file.read_exact(&mut buf)
            .map_err(|e| StorageError::Projection(e.to_string()))?;
        let mut vec = Vec::with_capacity(self.head_dim);
        for chunk in buf.chunks_exact(4) {
            if let Ok(bytes) = chunk.try_into() {
                vec.push(f32::from_le_bytes(bytes));
            } else {
                return Err(StorageError::Projection("Malformed float vector bytes".to_string()));
            }
        }
        Ok(vec)
    }

    pub fn read_head_vectors(&mut self, head: &str) -> Result<Vec<Vec<f32>>, StorageError> {
        let start_offset = self.offsets.get(head).copied()
            .ok_or_else(|| StorageError::NotFound(format!("Head {} not found", head)))?;
        let file_len = self.file.seek(SeekFrom::End(0))
            .map_err(|e| StorageError::Projection(e.to_string()))?;
        let vec_size = (self.head_dim * 4) as u64;

        let mut result = Vec::new();
        let mut offset = start_offset;
        while offset < file_len {
            let vec = self.read_vector(offset)?;
            result.push(vec);
            offset += vec_size;
        }
        Ok(result)
    }

    pub fn head_count(&self) -> usize {
        self.offsets.len()
    }
}
