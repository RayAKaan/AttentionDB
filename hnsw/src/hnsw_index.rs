use hnsw_rs::hnsw::{Hnsw, Neighbour};
use hnsw_rs::prelude::*;
use std::path::Path;
use std::fs::File;
use std::io::{Read, Write};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::error::HNSWError;
use crate::gpu::{GpuBackend, CpuBackend};

#[cfg(feature = "cuda")]
use crate::gpu::CudaBackend;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    pub max_nb_connection: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub store_vectors: bool,
    pub max_elements: usize,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            max_nb_connection: 16,
            ef_construction: 400,
            ef_search: 64,
            store_vectors: true,
            max_elements: 100_000,
        }
    }
}

impl HNSWConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    pub fn with_max_connections(mut self, max_conn: usize) -> Self {
        self.max_nb_connection = max_conn;
        self
    }

    pub fn with_ef_construction(mut self, ef_constr: usize) -> Self {
        self.ef_construction = ef_constr;
        self
    }

    pub fn with_vector_storage(mut self, store: bool) -> Self {
        self.store_vectors = store;
        self
    }

    pub fn with_max_elements(mut self, max_elements: usize) -> Self {
        self.max_elements = max_elements;
        self
    }
}

pub struct HNSWIndex {
    pub head_name: String,
    pub dim: usize,
    inner: Hnsw<'static, f32, DistCosine>,
    pub config: HNSWConfig,
    pub is_built: bool,
    vectors: Vec<(u64, Vec<f32>)>,
    id_to_idx: HashMap<u64, usize>,
    _pins: Vec<Box<[f32]>>,
    gpu_backend: Box<dyn GpuBackend>,
}

impl HNSWIndex {
    pub fn new(head_name: &str, dim: usize, config: HNSWConfig) -> Self {
        let inner = Hnsw::<f32, DistCosine>::new(
            config.max_nb_connection,
            config.max_elements,
            16,
            config.ef_construction,
            DistCosine {},
        );

        Self {
            head_name: head_name.to_string(),
            dim,
            inner,
            config,
            is_built: false,
            vectors: Vec::new(),
            id_to_idx: HashMap::new(),
            _pins: Vec::new(),
            gpu_backend: Box::new(CpuBackend),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn enable_cuda(&mut self) -> Result<(), HNSWError> {
        self.gpu_backend = Box::new(CudaBackend::new()?);
        Ok(())
    }

    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), HNSWError> {
        if vector.len() != self.dim {
            return Err(HNSWError::DimensionMismatch { expected: self.dim, got: vector.len() });
        }

        let boxed: Box<[f32]> = vector.to_vec().into_boxed_slice();
        let reference: &'static [f32] = unsafe { &*(&*boxed as *const [f32]) };
        self._pins.push(boxed);
        self.inner.insert((reference, id as usize));
        self.is_built = true;

        if self.config.store_vectors {
            let idx = self.vectors.len();
            self.vectors.push((id, vector.to_vec()));
            self.id_to_idx.insert(id, idx);
        }

        Ok(())
    }

    pub fn insert_batch(&mut self, items: &[(u64, Vec<f32>)]) -> Result<(), HNSWError> {
        for (id, vec) in items {
            self.insert(*id, vec)?;
        }
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize, ef: Option<usize>) -> Result<Vec<(u64, f32)>, HNSWError> {
        if query.len() != self.dim {
            return Err(HNSWError::DimensionMismatch { expected: self.dim, got: query.len() });
        }
        if !self.is_built {
            return Err(HNSWError::IndexNotBuilt);
        }

        let ef = ef.unwrap_or(self.config.ef_search);
        let neighbors: Vec<Neighbour> = self.inner.search(query, k, ef);
        let results: Vec<(u64, f32)> = neighbors.into_iter()
            .map(|n| (n.d_id as u64, n.distance))
            .collect();

        Ok(results)
    }

    pub fn rerank_exact(&self, query: &[f32], candidates: &[u64], k: usize) -> Result<Vec<(u64, f32)>, HNSWError> {
        if query.len() != self.dim {
            return Err(HNSWError::DimensionMismatch { expected: self.dim, got: query.len() });
        }

        let candidate_vectors: Vec<(u64, Vec<f32>)> = candidates
            .iter()
            .filter_map(|id| {
                self.id_to_idx.get(id).and_then(|&idx| {
                    self.vectors.get(idx).map(|(_, vec)| (*id, vec.clone()))
                })
            })
            .collect();

        let results = self.gpu_backend.rerank_exact(query, &candidate_vectors, k)
            .unwrap_or_else(|_| {
                let mut scored: Vec<(u64, f32)> = candidate_vectors.iter()
                    .map(|(id, vec)| {
                        let score: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                        (*id, score)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(k);
                scored
            });

        Ok(results)
    }

    pub fn search_with_rerank(&self, query: &[f32], k: usize, ef: Option<usize>) -> Result<Vec<(u64, f32)>, HNSWError> {
        let candidates = self.search(query, k * 3, ef)?;
        let candidate_ids: Vec<u64> = candidates.into_iter().map(|(id, _)| id).collect();
        self.rerank_exact(query, &candidate_ids, k)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn get_vector(&self, id: u64) -> Option<&[f32]> {
        self.id_to_idx.get(&id)
            .and_then(|&idx| self.vectors.get(idx))
            .map(|(_, v)| v.as_slice())
    }

    pub fn set_ef_search(&mut self, ef: usize) {
        self.config.ef_search = ef;
    }

    pub fn save(&self, path: &Path) -> Result<(), HNSWError> {
        let mut file = File::create(path)?;
        let config_json = serde_json::to_vec(&self.config)
            .map_err(|e| HNSWError::Serialization(e.to_string()))?;
        file.write_all(&(config_json.len() as u32).to_le_bytes())?;
        file.write_all(&config_json)?;

        file.write_all(&(self.vectors.len() as u32).to_le_bytes())?;
        for (id, vec) in &self.vectors {
            file.write_all(&id.to_le_bytes())?;
            file.write_all(&(vec.len() as u32).to_le_bytes())?;
            for val in vec {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        Ok(())
    }

    pub fn load(path: &Path, head_name: &str, dim: usize) -> Result<Self, HNSWError> {
        let mut file = File::open(path)?;

        let config_len = {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            u32::from_le_bytes(buf) as usize
        };

        let mut config_buf = vec![0u8; config_len];
        file.read_exact(&mut config_buf)?;

        let config: HNSWConfig = serde_json::from_slice(&config_buf)
            .map_err(|e| HNSWError::Serialization(e.to_string()))?;

        let inner = Hnsw::<f32, DistCosine>::new(
            config.max_nb_connection,
            config.max_elements,
            16,
            config.ef_construction,
            DistCosine {},
        );

        let vec_count = {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            u32::from_le_bytes(buf) as usize
        };

        let mut vectors = Vec::with_capacity(vec_count);
        let mut id_to_idx = HashMap::new();
        let mut _pins = Vec::with_capacity(vec_count);

        for i in 0..vec_count {
            let mut id_buf = [0u8; 8];
            file.read_exact(&mut id_buf)?;
            let id = u64::from_le_bytes(id_buf);

            let mut len_buf = [0u8; 4];
            file.read_exact(&mut len_buf)?;
            let vlen = u32::from_le_bytes(len_buf) as usize;

            let mut vec = vec![0.0; vlen];
            for v in &mut vec {
                let mut val_buf = [0u8; 4];
                file.read_exact(&mut val_buf)?;
                *v = f32::from_le_bytes(val_buf);
            }

            let boxed: Box<[f32]> = vec.clone().into_boxed_slice();
            let reference: &'static [f32] = unsafe { &*(&*boxed as *const [f32]) };
            _pins.push(boxed);
            inner.insert((reference, id as usize));
            vectors.push((id, vec));
            id_to_idx.insert(id, i);
        }

        Ok(Self {
            head_name: head_name.to_string(),
            dim,
            inner,
            config,
            is_built: !vectors.is_empty(),
            vectors,
            id_to_idx,
            _pins,
            gpu_backend: Box::new(CpuBackend),
        })
    }
}
