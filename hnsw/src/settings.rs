//! Tunable Retrieval Parameters (Collection-level Settings)
//!
//! These parameters allow users to tune the recall vs speed trade-off
//! per collection, without recompiling.

use serde::{Deserialize, Serialize};

/// Collection-level settings for HNSW retrieval behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSettings {
    /// ef_search: candidates explored during search. Higher = better recall, slower.
    pub ef_search: usize,

    /// ef_construction: candidates considered during index building.
    /// Higher = better graph quality, slower indexing.
    pub ef_construction: usize,

    /// max_nb_connection: maximum connections per node in the HNSW graph.
    pub max_nb_connection: usize,

    /// similarity_metric: "cosine", "dot_product", or "l2"
    pub similarity_metric: String,

    /// enable_exact_reranking: whether to rerank exact after HNSW search
    pub enable_exact_reranking: bool,

    /// enable_gpu_fusion: whether to use GPU for multi-head score fusion
    pub enable_gpu_fusion: bool,

    /// enable_gpu_projections: whether to use GPU for W_Q / W_K / W_V projections
    pub enable_gpu_projections: bool,
}

impl Default for CollectionSettings {
    fn default() -> Self {
        Self {
            ef_search: 64,
            ef_construction: 400,
            max_nb_connection: 16,
            similarity_metric: "cosine".to_string(),
            enable_exact_reranking: true,
            enable_gpu_fusion: false,
            enable_gpu_projections: false,
        }
    }
}

impl CollectionSettings {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate all settings, returning an error string if any are invalid
    pub fn validate(&self) -> Result<(), String> {
        if self.ef_search < 1 {
            return Err("ef_search must be at least 1".to_string());
        }
        if self.ef_construction < 10 {
            return Err("ef_construction must be at least 10".to_string());
        }
        if self.max_nb_connection < 4 {
            return Err("max_nb_connection must be at least 4".to_string());
        }
        if !["cosine", "dot_product", "l2"].contains(&self.similarity_metric.as_str()) {
            return Err("similarity_metric must be one of: cosine, dot_product, l2".to_string());
        }
        Ok(())
    }

    pub fn high_recall() -> Self {
        Self {
            ef_search: 256,
            ef_construction: 800,
            max_nb_connection: 48,
            similarity_metric: "cosine".to_string(),
            enable_exact_reranking: true,
            enable_gpu_fusion: true,
            enable_gpu_projections: true,
        }
    }

    pub fn low_latency() -> Self {
        Self {
            ef_search: 32,
            ef_construction: 200,
            max_nb_connection: 12,
            similarity_metric: "cosine".to_string(),
            enable_exact_reranking: false,
            enable_gpu_fusion: false,
            enable_gpu_projections: false,
        }
    }
}
