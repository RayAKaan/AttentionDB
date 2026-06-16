use std::sync::Arc;
use parking_lot::RwLock;
use attentiondb_hnsw::{HeadIndexManager, HNSWConfig};
use attentiondb_multihead::{MultiHeadManager, HeadConfig, HeadType};
use crate::error::CoreError;

pub struct Collection {
    pub name: String,
    pub dim: usize,
    pub head_manager: Arc<RwLock<HeadIndexManager>>,
    pub multihead_manager: Arc<RwLock<MultiHeadManager>>,
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
        let heads = self.head_manager.read();
        heads.insert(head, id, vector)?;
        Ok(())
    }

    pub fn attend(
        &self,
        heads: &[String],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let heads_read = self.head_manager.read();
        let head_results: Vec<(String, Vec<(u64, f32)>)> = heads
            .iter()
            .filter_map(|h| {
                let idx = heads_read.get_head(h).ok()?;
                let results = idx.read().search(query, top_k, None).ok()?;
                Some((h.clone(), results))
            })
            .collect();

        let weights: Vec<(String, f32)> = heads.iter().map(|h| (h.clone(), 1.0)).collect();
        let mh = self.multihead_manager.read();
        let fused = mh.fuse_weighted(&head_results, &weights);
        Ok(fused)
    }

    pub fn attend_weighted(
        &self,
        heads: &[(String, f32)],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let heads_read = self.head_manager.read();
        let head_results: Vec<(String, Vec<(u64, f32)>)> = heads
            .iter()
            .filter_map(|(h, _)| {
                let idx = heads_read.get_head(h).ok()?;
                let results = idx.read().search(query, top_k, None).ok()?;
                Some((h.clone(), results))
            })
            .collect();

        let mh = self.multihead_manager.read();
        let fused = mh.fuse_weighted(&head_results, heads);
        Ok(fused)
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
