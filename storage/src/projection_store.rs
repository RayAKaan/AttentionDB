//! Columnar Projection Store (.kv)
//!
//! Stores K and V vectors contiguously per head for fast SIMD dot-product access.

use std::fs::File;
use std::io::{Write, Read, Seek, SeekFrom};
use std::path::Path;
use crate::error::StorageError;

pub struct ProjectionStore {
    file: File,
    head_dim: usize,
}

impl ProjectionStore {
    pub fn new(path: &Path, head_dim: usize) -> Result<Self, StorageError> {
        let file = File::create(path)?;
        Ok(Self { file, head_dim })
    }

    pub fn append_vectors(&mut self, k_vecs: &[Vec<f32>], v_vecs: &[Vec<f32>]) -> Result<u64, StorageError> {
        let offset = self.file.seek(SeekFrom::End(0))?;

        for vec in k_vecs.iter().chain(v_vecs.iter()) {
            let bytes: Vec<u8> = vec.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            self.file.write_all(&bytes)?;
        }

        Ok(offset)
    }

    pub fn read_vector(&mut self, offset: u64, head_idx: usize) -> Result<Vec<f32>, StorageError> {
        self.file.seek(SeekFrom::Start(offset + (head_idx * self.head_dim * 4) as u64))?;
        let mut buf = vec![0u8; self.head_dim * 4];
        self.file.read_exact(&mut buf)?;

        let vec: Vec<f32> = buf.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok(vec)
    }
}
