use std::collections::HashMap;
use std::path::Path;
use crate::hnsw_index::{HNSWIndex, HNSWConfig};
use crate::error::HNSWError;

pub struct HeadIndexManager {
    pub heads: HashMap<String, HNSWIndex>,
    pub dim: usize,
}

impl HeadIndexManager {
    pub fn new(dim: usize) -> Self {
        Self { heads: HashMap::new(), dim }
    }

    pub fn add_head(&mut self, name: &str) {
        let config = HNSWConfig::default();
        let index = HNSWIndex::new(name, self.dim, config);
        self.heads.insert(name.to_string(), index);
    }

    pub fn add_head_with_config(&mut self, name: &str, config: HNSWConfig) {
        let index = HNSWIndex::new(name, self.dim, config);
        self.heads.insert(name.to_string(), index);
    }

    pub fn get_head(&self, name: &str) -> Result<&HNSWIndex, HNSWError> {
        self.heads.get(name).ok_or_else(|| HNSWError::HeadNotFound(name.to_string()))
    }

    pub fn get_head_mut(&mut self, name: &str) -> Result<&mut HNSWIndex, HNSWError> {
        self.heads.get_mut(name).ok_or_else(|| HNSWError::HeadNotFound(name.to_string()))
    }

    pub fn insert(&mut self, head: &str, id: u64, vector: &[f32]) -> Result<(), HNSWError> {
        self.get_head_mut(head)?.insert(id, vector)
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
            if let Ok(index) = self.get_head(head_name) {
                let results = index.search(query, k * 2, ef)?;
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
            if let Ok(index) = self.get_head(head_name) {
                let results = index.search(query, k * 2, ef)?;
                for (id, dist) in results {
                    all_results.push((id, dist * weight));
                }
            }
        }
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);
        Ok(all_results)
    }

    pub fn remove_head(&mut self, name: &str) -> Result<(), HNSWError> {
        self.heads.remove(name).ok_or_else(|| HNSWError::HeadNotFound(name.to_string()))?;
        Ok(())
    }

    pub fn save_all(&self, dir: &Path) -> Result<(), HNSWError> {
        std::fs::create_dir_all(dir)?;
        for (name, index) in &self.heads {
            index.save(&dir.join(name))?;
        }
        Ok(())
    }

    pub fn list_heads(&self) -> Vec<String> {
        self.heads.keys().cloned().collect()
    }

    pub fn total_vectors(&self) -> usize {
        self.heads.values().map(|h| h.len()).sum()
    }

    pub fn head_count(&self) -> usize {
        self.heads.len()
    }
}
