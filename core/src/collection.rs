use std::sync::Arc;
use parking_lot::RwLock;
use attentiondb_hnsw::{HeadIndexManager, HNSWConfig};
use attentiondb_multihead::{MultiHeadManager, HeadConfig, HeadType, GatingNetwork};
use crate::bm25::Bm25Index;
use crate::error::CoreError;

pub const OVERFETCH_MULTIPLIER: usize = 5;
pub const MIN_CANDIDATES_PER_HEAD: usize = 20;

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
        Self {
            name: name.to_string(),
            dim,
            head_manager: Arc::new(RwLock::new(HeadIndexManager::new(dim))),
            multihead_manager: Arc::new(RwLock::new(MultiHeadManager::new(dim, 1))),
            settings: RwLock::new(attentiondb_hnsw::CollectionSettings::default()),
            bm25: Bm25Index::default(),
            gating_network: RwLock::new(None),
        }
    }

    pub fn add_default_head(&self, name: &str) -> Result<(), CoreError> {
        let config = HNSWConfig::default();
        self.head_manager.read().add_head_with_config(name, config);
        let head_config = HeadConfig::new(name, HeadType::Semantic, self.dim);
        self.multihead_manager.write().add_head(head_config);
        Ok(())
    }

    pub fn add_head_with_settings(
        &self,
        name: &str,
        config: HNSWConfig,
        head_type: HeadType,
    ) -> Result<(), CoreError> {
        self.head_manager.read().add_head_with_config(name, config);
        self.multihead_manager.write().add_head(HeadConfig::new(name, head_type, self.dim));
        Ok(())
    }

    pub fn insert_vector(&self, head: &str, id: u64, vector: &[f32]) -> Result<(), CoreError> {
        if self.head_manager.read().get_head(head).is_err() {
            self.head_manager.read().add_head_with_config(head, HNSWConfig::default());
        }
        if self.multihead_manager.read().get_head(head).is_err() {
            self.multihead_manager.write().add_head(HeadConfig::new(head, HeadType::Semantic, self.dim));
        }
        self.head_manager.read().insert(head, id, vector)?;
        Ok(())
    }

    fn get_gated_weights(&self, query: &[f32], head_names: &[String]) -> Vec<f32> {
        let gating = self.gating_network.read();
        if let Some(ref net) = *gating {
            let w = net.forward(query);
            if w.len() == head_names.len() {
                return w;
            }
        }
        let w = 1.0 / head_names.len().max(1) as f32;
        vec![w; head_names.len()]
    }

    fn normalize_head_scores(results: &mut Vec<(u64, f32)>) {
        let max_score = results.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        if max_score > 0.0 {
            for (_, s) in results.iter_mut() {
                *s /= max_score;
            }
        }
    }

    pub fn load_gating_network_from(&self, net: GatingNetwork) {
        *self.gating_network.write() = Some(net);
    }

    pub fn attend(
        &self,
        heads: &[String],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let heads_read = self.head_manager.read();
        let num_heads = heads.len().max(1);
        let effective_top_k = (top_k * OVERFETCH_MULTIPLIER).max(MIN_CANDIDATES_PER_HEAD);

        let mut head_results: Vec<(String, Vec<(u64, f32)>)> = heads
            .iter()
            .filter_map(|h| {
                let idx = heads_read.get_head(h).ok()?;
                let mut results = idx.read().search(query, effective_top_k, None).ok()?;
                Self::normalize_head_scores(&mut results);
                Some((h.clone(), results))
            })
            .collect();

        if head_results.is_empty() {
            return Ok(vec![]);
        }

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
        let effective_top_k = (top_k * OVERFETCH_MULTIPLIER).max(MIN_CANDIDATES_PER_HEAD);
        let head_results: Vec<(String, Vec<(u64, f32)>)> = heads
            .iter()
            .filter_map(|(h, _)| {
                let idx = heads_read.get_head(h).ok()?;
                let mut results = idx.read().search(query, effective_top_k, None).ok()?;
                Self::normalize_head_scores(&mut results);
                Some((h.clone(), results))
            })
            .collect();
        let mh = self.multihead_manager.read();
        let mut fused = mh.fuse_weighted(&head_results, heads);
        if fused.len() > top_k { fused.truncate(top_k); }
        Ok(fused)
    }

    pub fn attend_hybrid(
        &self,
        heads: &[String],
        query_vector: &[f32],
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let effective_top_k = (top_k * OVERFETCH_MULTIPLIER).max(MIN_CANDIDATES_PER_HEAD);
        let dense = self.attend(heads, query_vector, effective_top_k)?;
        let sparse = self.bm25.search(query_text, effective_top_k);
        Ok(crate::bm25::reciprocal_rank_fusion(&dense, &sparse, top_k))
    }

    pub fn list_heads(&self) -> Vec<String> {
        self.head_manager.read().list_heads()
    }

    pub fn total_vectors(&self) -> usize {
        self.head_manager.read().total_vectors()
    }

    pub fn head_count(&self) -> usize {
        self.head_manager.read().head_count()
    }
}