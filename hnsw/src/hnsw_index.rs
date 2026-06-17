use hnsw_rs::hnsw::{Hnsw, Neighbour};
use hnsw_rs::prelude::*;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use crate::error::HNSWError;
use crate::gpu::{GpuBackend, CpuBackend};
use crate::persistence::strategy::PersistenceStrategy;
use crate::settings::CollectionSettings;

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

impl From<CollectionSettings> for HNSWConfig {
    fn from(settings: CollectionSettings) -> Self {
        HNSWConfig {
            max_nb_connection: settings.max_nb_connection,
            ef_construction: settings.ef_construction,
            ef_search: settings.ef_search,
            store_vectors: true,
            max_elements: 100_000,
        }
    }
}

pub struct HNSWIndex {
    pub head_name: String,
    pub dim: usize,
    inner: Hnsw<'static, f32, DistCosine>,
    pub config: HNSWConfig,
    pub settings: CollectionSettings,
    pub is_built: bool,
    pub(crate) vectors: Vec<(u64, Vec<f32>)>,
    id_to_idx: HashMap<u64, usize>,
    /// Arc references keep vector data alive for the lifetime of the index.
    /// When `insert` is called, the vector is cloned into a `Vec<f32>` that
    /// is stored both in `vectors` and in `_arc_refs`. The `Arc` ensures the
    /// backing memory is not freed while `hnsw_rs` holds a raw reference.
    _arc_refs: Vec<Arc<Vec<f32>>>,
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
            settings: CollectionSettings::default(),
            is_built: false,
            vectors: Vec::new(),
            id_to_idx: HashMap::new(),
            _arc_refs: Vec::new(),
            gpu_backend: Box::new(CpuBackend),
        }
    }

    pub fn with_settings(
        head_name: &str,
        dim: usize,
        config: HNSWConfig,
        settings: CollectionSettings,
    ) -> Result<Self, HNSWError> {
        settings.validate().map_err(|e| HNSWError::InvalidConfig(e))?;
        let mut index = Self::new(head_name, dim, config);
        index.settings = settings;
        Ok(index)
    }

    pub fn update_settings(&mut self, settings: CollectionSettings) -> Result<(), HNSWError> {
        settings.validate().map_err(|e| HNSWError::InvalidConfig(e))?;
        self.settings = settings;
        Ok(())
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

        let arc_vec = Arc::new(vector.to_vec());
        let reference: &[f32] = &arc_vec;
        // SAFETY: hnsw_rs stores a 'static reference, but we keep the Arc alive
        // in _arc_refs for the entire lifetime of the index, so the reference
        // remains valid.
        let static_ref: &'static [f32] = unsafe { &*(reference as *const [f32]) };
        self.inner.insert((static_ref, id as usize));
        self._arc_refs.push(arc_vec);
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

        let ef = ef.unwrap_or(self.settings.ef_search);
        let neighbors: Vec<Neighbour> = self.inner.search(query, k, ef);
        let results: Vec<(u64, f32)> = neighbors.into_iter()
            .map(|n| (n.d_id as u64, 1.0 - n.distance))
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

    // ==================== GPU PROJECTIONS ====================

    /// Enable GPU acceleration for projection operations (requires `cuda` feature)
    #[cfg(feature = "cuda")]
    pub fn enable_gpu_projections(&mut self) -> Result<(), HNSWError> {
        if !self.gpu_backend.is_available() {
            self.gpu_backend = Box::new(CudaBackend::new()?);
        }
        Ok(())
    }

    /// Perform batched projection using GPU if available, otherwise CPU
    pub fn project_batch(&self, matrix: &[f32], vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if self.settings.enable_gpu_fusion && self.gpu_backend.is_available() {
            match self.gpu_backend.project_batch(matrix, vectors) {
                Ok(result) => return result,
                Err(_) => {}
            }
        }
        let dim = if vectors.is_empty() { return vec![]; } else { vectors[0].len() };
        let mut results = Vec::with_capacity(vectors.len());
        for vec in vectors {
            let mut output = vec![0.0; dim];
            for i in 0..dim {
                for j in 0..dim {
                    output[i] += matrix[i * dim + j] * vec[j];
                }
            }
            results.push(output);
        }
        results
    }

    /// Project a single vector (convenience wrapper)
    pub fn project_vector(&self, matrix: &[f32], vector: &[f32]) -> Vec<f32> {
        self.project_batch(matrix, &[vector.to_vec()])
            .into_iter()
            .next()
            .unwrap_or_default()
    }

    pub fn save(&self, dir: &Path) -> Result<(), HNSWError> {
        crate::persistence::save_index(self, dir)
            .map_err(|e| HNSWError::Persistence(e.to_string()))?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self, HNSWError> {
        let index = crate::persistence::load_index(dir, None::<fn(crate::persistence::LoadProgress)>)
            .map_err(|e| HNSWError::Persistence(e.to_string()))?;
        Ok(index)
    }

    pub fn load_with_progress<F>(
        dir: &Path,
        progress_callback: F,
    ) -> Result<Self, HNSWError>
    where
        F: FnMut(crate::persistence::LoadProgress),
    {
        let index = crate::persistence::load_index(dir, Some(progress_callback))
            .map_err(|e| HNSWError::Persistence(e.to_string()))?;
        Ok(index)
    }

    pub fn append_vectors(
        &mut self,
        dir: &Path,
        new_vectors: &[(u64, Vec<f32>)],
    ) -> Result<usize, HNSWError> {
        for (id, vec) in new_vectors {
            let _ = self.insert(*id, vec);
        }
        crate::persistence::append_vectors(dir, new_vectors)
            .map_err(|e| HNSWError::Persistence(e.to_string()))
    }

    /// Save using graph-aware persistence (preserves insertion order)
    pub fn save_graph(&self, dir: &Path) -> Result<(), HNSWError> {
        let strategy = crate::persistence::GraphPersistence;
        strategy.save(self, dir)
            .map_err(|e| HNSWError::Persistence(e.to_string()))
    }

    /// Load using graph-aware persistence
    pub fn load_graph(dir: &Path) -> Result<Self, HNSWError> {
        let strategy = crate::persistence::GraphPersistence;
        strategy.load(dir)
            .map_err(|e| HNSWError::Persistence(e.to_string()))
    }
}
