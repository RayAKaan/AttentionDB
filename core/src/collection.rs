use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, debug};
use attentiondb_hnsw::{HeadIndexManager, HNSWConfig};
use attentiondb_multihead::{MultiHeadManager, HeadConfig, HeadType, GatingNetwork};
use crate::bm25::Bm25Index;
use crate::error::CoreError;

/// Overfetch multiplier: query more candidates per head for better gating.
const OVERFETCH_FACTOR: usize = 3;

/// Whether to normalize scores within each head before gating fusion.
const SCORE_NORMALIZATION: bool = true;

pub struct Collection {
    pub name: String,
    pub dim: usize,
    pub head_manager: Arc<RwLock<HeadIndexManager>>,
    pub multihead_manager: Arc<RwLock<MultiHeadManager>>,
    pub settings: RwLock<attentiondb_hnsw::CollectionSettings>,
    pub bm25: Bm25Index,
    pub gating_network: RwLock<Option<GatingNetwork>>,
}

impl Collection {
    pub fn new(name: &str, dim: usize) -> Self {
        let head_manager = Arc::new(RwLock::new(HeadIndexManager::new(dim)));
        let multihead_manager = Arc::new(RwLock::new(MultiHeadManager::new(dim, 1)));
        Self {
            name: name.to_string(),
            dim,
            head_manager,
            multihead_manager,
            settings: RwLock::new(attentiondb_hnsw::CollectionSettings::default()),
            bm25: Bm25Index::default(),
            gating_network: RwLock::new(None),
        }
    }

    pub fn add_default_head(&self, name: &str) -> Result<(), CoreError> {
        let config = HNSWConfig::default();
        {
            let heads = self.head_manager.read();
            heads.add_head_with_config(name, config);
        }
        let head_config = HeadConfig::new(name, HeadType::Semantic, self.dim);
        {
            let mut mh = self.multihead_manager.write();
            mh.add_head(head_config);
        }
        Ok(())
    }

    pub fn add_head_with_settings(
        &self,
        name: &str,
        config: HNSWConfig,
        head_type: HeadType,
    ) -> Result<(), CoreError> {
        {
            let heads = self.head_manager.read();
            heads.add_head_with_config(name, config);
        }
        let head_config = HeadConfig::new(name, head_type, self.dim);
        {
            let mut mh = self.multihead_manager.write();
            mh.add_head(head_config);
        }
        Ok(())
    }

    pub fn insert_vector(
        &self,
        head: &str,
        id: u64,
        vector: &[f32],
    ) -> Result<(), CoreError> {
        if self.head_manager.read().get_head(head).is_err() {
            let config = HNSWConfig::default();
            self.head_manager.read().add_head_with_config(head, config);
        }
        {
            let mut mh = self.multihead_manager.write();
            if mh.get_head(head).is_err() {
                let head_config = HeadConfig::new(head, HeadType::Semantic, self.dim);
                mh.add_head(head_config);
            }
        }
        let heads = self.head_manager.read();
        heads.insert(head, id, vector)?;
        Ok(())
    }

    /// Get learned gating weights from the gating network, or fallback to uniform.
    fn get_gated_weights(&self, query: &[f32], head_names: &[String]) -> Vec<f32> {
        let gating = self.gating_network.read();
        if let Some(ref net) = *gating {
            let weights = net.forward(query);
            if weights.len() == head_names.len() {
                return weights;
            }
            debug!(
                "[Collection] Gating network returned {} weights for {} heads; using uniform fallback",
                weights.len(),
                head_names.len()
            );
        }
        let w = 1.0 / head_names.len().max(1) as f32;
        vec![w; head_names.len()]
    }

    /// Load a gating network for learned weight assignment.
    pub fn load_gating_network(&self, gating: GatingNetwork) {
        let mut net = self.gating_network.write();
        *net = Some(gating);
        info!("[Collection] Gating network loaded for '{}'", self.name);
    }

    pub fn attend(
        &self,
        heads: &[String],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let heads_read = self.head_manager.read();
        let num_heads = heads.len().max(1);
        let per_head_k = top_k * OVERFETCH_FACTOR * num_heads;

        let head_results: Vec<(String, Vec<(u64, f32)>)> = heads
            .iter()
            .filter_map(|h| {
                let idx = heads_read.get_head(h).ok()?;
                let mut results = idx.read().search(query, per_head_k, None).ok()?;
                if SCORE_NORMALIZATION {
                    if let Some(max_score) = results.iter().map(|(_, s)| *s).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) {
                        if max_score > 0.0 {
                            for (_, s) in results.iter_mut() {
                                *s /= max_score;
                            }
                        }
                    }
                }
                Some((h.clone(), results))
            })
            .collect();

        let gate_weights = self.get_gated_weights(query, heads);
        let mh = self.multihead_manager.read();
        let mut fused = mh.fuse_weighted_with_gating(&head_results, &gate_weights);
        if fused.len() > top_k {
            fused.truncate(top_k);
        }
        Ok(fused)
    }

    pub fn attend_weighted(
        &self,
        heads: &[(String, f32)],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let heads_read = self.head_manager.read();
        let num_heads = heads.len().max(1);
        let per_head_k = top_k * OVERFETCH_FACTOR * num_heads;
        let head_results: Vec<(String, Vec<(u64, f32)>)> = heads
            .iter()
            .filter_map(|(h, _)| {
                let idx = heads_read.get_head(h).ok()?;
                let results = idx.read().search(query, per_head_k, None).ok()?;
                Some((h.clone(), results))
            })
            .collect();

        let mh = self.multihead_manager.read();
        let mut fused = mh.fuse_weighted(&head_results, heads);
        if fused.len() > top_k {
            fused.truncate(top_k);
        }
        Ok(fused)
    }

    pub fn attend_hybrid(
        &self,
        heads: &[String],
        query_vector: &[f32],
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let num_heads = heads.len().max(1);
        let per_head_k = top_k * OVERFETCH_FACTOR * num_heads;
        let dense_results = self.attend(heads, query_vector, per_head_k)?;
        let sparse_results = self.bm25.search(query_text, per_head_k);
        let hybrid_fused = crate::bm25::reciprocal_rank_fusion(&dense_results, &sparse_results, top_k);
        Ok(hybrid_fused)
    }

    pub fn list_heads(&self) -> Vec<String> {
        let heads = self.head_manager.read();
        heads.list_heads()
    }

    pub fn total_vectors(&self) -> usize {
        let heads = self.head_manager.read();
        heads.total_vectors()
    }

    pub fn head_count(&self) -> usize {
        let heads = self.head_manager.read();
        heads.head_count()
    }
}