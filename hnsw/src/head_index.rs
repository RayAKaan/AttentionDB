use crate::error::HNSWError;
use crate::hnsw_index::{HNSWConfig, HNSWIndex};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub struct HeadIndexManager {
    heads: RwLock<HashMap<String, Arc<RwLock<HNSWIndex>>>>,
    pub dim: usize,
}

impl HeadIndexManager {
    pub fn new(dim: usize) -> Self {
        Self {
            heads: RwLock::new(HashMap::new()),
            dim,
        }
    }

    pub fn add_head(&self, name: &str) {
        let config = HNSWConfig::default();
        let index = Arc::new(RwLock::new(HNSWIndex::new(name, self.dim, config)));
        self.heads.write().insert(name.to_string(), index);
    }

    pub fn add_head_with_config(&self, name: &str, config: HNSWConfig) {
        let index = Arc::new(RwLock::new(HNSWIndex::new(name, self.dim, config)));
        self.heads.write().insert(name.to_string(), index);
    }

    pub fn get_head(&self, name: &str) -> Result<Arc<RwLock<HNSWIndex>>, HNSWError> {
        self.heads
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| HNSWError::HeadNotFound(name.to_string()))
    }

    pub fn insert(&self, head: &str, id: u64, vector: &[f32]) -> Result<(), HNSWError> {
        let idx = self.get_head(head)?;
        let result = idx.write().insert(id, vector);
        result
    }

    pub fn search_multi(
        &self,
        heads: &[&str],
        query: &[f32],
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(u64, f32)>, HNSWError> {
        let mut all_results: Vec<(u64, f32)> = Vec::new();
        for head_name in heads {
            if let Ok(idx) = self.get_head(head_name) {
                let results = idx.read().search(query, k * 2, ef)?;
                all_results.extend(results);
            }
        }
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);
        Ok(all_results)
    }

    pub fn search_multi_weighted(
        &self,
        heads: &[(&str, f32)],
        query: &[f32],
        k: usize,
        ef: Option<usize>,
    ) -> Result<Vec<(u64, f32)>, HNSWError> {
        let mut all_results: Vec<(u64, f32)> = Vec::new();
        for (head_name, weight) in heads {
            if let Ok(idx) = self.get_head(head_name) {
                let results = idx.read().search(query, k * 2, ef)?;
                for (id, dist) in results {
                    all_results.push((id, dist * weight));
                }
            }
        }
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);
        Ok(all_results)
    }

    pub fn remove_head(&self, name: &str) -> Result<(), HNSWError> {
        self.heads
            .write()
            .remove(name)
            .ok_or_else(|| HNSWError::HeadNotFound(name.to_string()))?;
        Ok(())
    }

    pub fn save_all(&self, dir: &Path) -> Result<(), HNSWError> {
        std::fs::create_dir_all(dir)?;
        let heads = self.heads.read();
        for (name, index) in heads.iter() {
            let idx = index.read();
            idx.save(&dir.join(name))?;
        }
        Ok(())
    }

    pub fn list_heads(&self) -> Vec<String> {
        self.heads.read().keys().cloned().collect()
    }

    pub fn total_vectors(&self) -> usize {
        self.heads.read().values().map(|idx| idx.read().len()).sum()
    }

    pub fn head_count(&self) -> usize {
        self.heads.read().len()
    }
}
