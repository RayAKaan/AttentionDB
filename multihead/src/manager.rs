use crate::head::HeadConfig;
use crate::gating::GatingNetwork;
use crate::fusion::{fuse_scores, weighted_fuse};
use crate::error::MultiHeadError;
use std::collections::HashMap;

pub struct MultiHeadManager {
    pub heads: HashMap<String, HeadConfig>,
    pub gating: GatingNetwork,
    pub dim: usize,
}

impl MultiHeadManager {
    pub fn new(dim: usize, num_heads: usize) -> Self {
        Self {
            heads: HashMap::new(),
            gating: GatingNetwork::new(dim, num_heads),
            dim,
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

    /// Fuse results from multiple heads using learned gating
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

        let gate_weights = self.gating.forward(query_embedding);
        Ok(fuse_scores(head_results, &gate_weights))
    }

    /// Fuse using explicit head weights (bypass gating)
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

    pub fn list_heads(&self) -> Vec<String> {
        self.heads.keys().cloned().collect()
    }

    pub fn head_count(&self) -> usize {
        self.heads.len()
    }

    pub fn get_head_weights(&self) -> HashMap<String, f32> {
        self.heads.iter().map(|(k, v)| (k.clone(), v.weight)).collect()
    }
}
