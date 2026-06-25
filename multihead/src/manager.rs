use crate::head::HeadConfig;
use crate::gating::GatingNetwork;
use crate::fusion::{fuse_scores, weighted_fuse};
use crate::error::MultiHeadError;
use std::collections::HashMap;
#[cfg(feature = "gpu")]
use tracing::{info, warn};

#[cfg(feature = "gpu")]
use attentiondb_hnsw::gpu::{GpuBackend, CpuBackend};
#[cfg(feature = "gpu")]
use attentiondb_hnsw::gpu::CudaBackend;

pub struct MultiHeadManager {
    pub heads: HashMap<String, HeadConfig>,
    pub gating: GatingNetwork,
    pub dim: usize,
    #[cfg(feature = "gpu")]
    gpu_backend: Box<dyn GpuBackend>,
    #[cfg(feature = "gpu")]
    gpu_available: bool,
}

impl MultiHeadManager {
    pub fn new(dim: usize, num_heads: usize) -> Self {
        #[cfg(feature = "gpu")]
        let (gpu_backend, gpu_available) = match CudaBackend::new() {
            Ok(cuda) => {
                info!("[MultiHeadManager] CUDA backend initialized successfully");
                (Box::new(cuda) as Box<dyn GpuBackend>, true)
            }
            Err(e) => {
                warn!("[MultiHeadManager] CUDA init failed (falling back to CPU): {:?}", e);
                (Box::new(CpuBackend) as Box<dyn GpuBackend>, false)
            }
        };

        #[cfg(not(feature = "gpu"))]
        let _gpu_backend: () = ();

        Self {
            heads: HashMap::new(),
            gating: GatingNetwork::new(dim, num_heads),
            dim,
            #[cfg(feature = "gpu")]
            gpu_backend,
            #[cfg(feature = "gpu")]
            gpu_available,
        }
    }

    pub fn add_head(&mut self, config: HeadConfig) {
        self.heads.insert(config.name.clone(), config);
    }

    pub fn get_head(&self, name: &str) -> Result<&HeadConfig, MultiHeadError> {
        self.heads.get(name).ok_or_else(|| MultiHeadError::HeadNotFound(name.to_string()))
    }

    pub fn get_head_mut(&mut self, name: &str) -> Result<&mut HeadConfig, MultiHeadError> {
        self.heads.get_mut(name).ok_or_else(|| MultiHeadError::HeadNotFound(name.to_string()))
    }

    #[cfg(feature = "gpu")]
    pub fn enable_gpu_fusion(&mut self) -> Result<(), MultiHeadError> {
        if self.gpu_available {
            return Ok(());
        }
        match CudaBackend::new() {
            Ok(cuda) => {
                info!("[MultiHeadManager] GPU fusion enabled with CUDA");
                self.gpu_backend = Box::new(cuda);
                self.gpu_available = true;
                Ok(())
            }
            Err(e) => {
                warn!("[MultiHeadManager] GPU fusion requested but CUDA unavailable: {:?}", e);
                self.gpu_backend = Box::new(CpuBackend);
                self.gpu_available = false;
                Ok(())
            }
        }
    }

    pub fn fuse(
        &self,
        query_embedding: &[f32],
        head_results: &[(String, Vec<(u64, f32)>)],
    ) -> Result<Vec<(u64, f32)>, MultiHeadError> {
        if query_embedding.len() != self.dim {
            return Err(MultiHeadError::InvalidConfig(
                format!("Query embedding dimension mismatch: expected {}, got {}", self.dim, query_embedding.len())
            ));
        }

        #[cfg(feature = "gpu")]
        {
            if self.gpu_available {
                let gate_weights = self.gating.forward(query_embedding);
                match self.gpu_backend.fuse_scores(head_results, &gate_weights) {
                    Ok(fused) => return Ok(fused),
                    Err(e) => {
                        warn!("[MultiHeadManager] GPU fusion failed (falling back to CPU): {:?}", e);
                    }
                }
            }
        }

        let gate_weights = self.gating.forward(query_embedding);
        let fused = fuse_scores(head_results, &gate_weights);
        Ok(fused)
    }

    pub fn fuse_weighted(
        &self,
        head_results: &[(String, Vec<(u64, f32)>)],
        explicit_weights: &[(String, f32)],
    ) -> Vec<(u64, f32)> {
        let weight_refs: Vec<(&str, f32)> = explicit_weights.iter()
            .map(|(n, w)| (n.as_str(), *w))
            .collect();
        weighted_fuse(head_results, &weight_refs)
    }

    /// Fuse with gating weights directly (bypasses per-head weight map).
    pub fn fuse_weighted_with_gating(
        &self,
        head_results: &[(String, Vec<(u64, f32)>)],
        gate_weights: &[f32],
    ) -> Vec<(u64, f32)> {
        fuse_scores(head_results, gate_weights)
    }

    pub fn list_heads(&self) -> Vec<String> {
        self.heads.keys().cloned().collect()
    }

    pub fn head_count(&self) -> usize {
        self.heads.len()
    }

    pub fn get_head_weights(&self) -> HashMap<String, f32> {
        self.heads.iter().map(|(k, v)| (k.clone(), v.weight)).collect()
    }

    #[cfg(feature = "gpu")]
    pub fn is_gpu_fusion_enabled(&self) -> bool {
        self.gpu_available
    }

    #[cfg(not(feature = "gpu"))]
    pub fn is_gpu_fusion_enabled(&self) -> bool {
        false
    }

    pub fn create_hnsw_index_for_head(
        &self,
        head_name: &str,
        dim: usize,
        base_config: attentiondb_hnsw::HNSWConfig,
    ) -> Result<attentiondb_hnsw::HNSWIndex, MultiHeadError> {
        let head = self.get_head(head_name)?;
        let settings = head.settings.clone().unwrap_or_default();
        let mut final_config = base_config;
        final_config.ef_search = settings.ef_search;
        final_config.ef_construction = settings.ef_construction;
        final_config.max_nb_connection = settings.max_nb_connection;
        let index = attentiondb_hnsw::HNSWIndex::with_settings(
            head_name,
            dim,
            final_config,
            settings,
        ).map_err(|e| MultiHeadError::InvalidConfig(e.to_string()))?;
        Ok(index)
    }
}