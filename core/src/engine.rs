use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use crate::collection::Collection;
use crate::error::CoreError;

pub struct AttentionEngine {
    collections: Arc<RwLock<HashMap<String, Arc<Collection>>>>,
}

pub struct EngineStats {
    pub collection_count: usize,
    pub total_heads: usize,
    pub total_vectors: usize,
}

impl AttentionEngine {
    pub fn new() -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn create_collection(
        &self,
        name: &str,
        dim: usize,
        heads: &[&str],
    ) -> Result<(), CoreError> {
        let mut collections = self.collections.write();
        if collections.contains_key(name) {
            return Err(CoreError::CollectionAlreadyExists(name.to_string()));
        }

        let collection = Arc::new(Collection::new(name, dim));
        for head_name in heads {
            collection.add_default_head(head_name)?;
        }

        collections.insert(name.to_string(), collection);
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Result<Arc<Collection>, CoreError> {
        let collections = self.collections.read();
        collections
            .get(name)
            .cloned()
            .ok_or_else(|| CoreError::CollectionNotFound(name.to_string()))
    }

    pub fn attend(
        &self,
        collection_name: &str,
        heads: &[String],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let collection = self.get_collection(collection_name)?;
        collection.attend(heads, query, top_k)
    }

    pub fn attend_weighted(
        &self,
        collection_name: &str,
        heads: &[(String, f32)],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let collection = self.get_collection(collection_name)?;
        collection.attend_weighted(heads, query, top_k)
    }

    pub fn insert_vector(
        &self,
        collection_name: &str,
        head: &str,
        id: u64,
        vector: &[f32],
    ) -> Result<(), CoreError> {
        let collection = self.get_collection(collection_name)?;
        collection.insert_vector(head, id, vector)
    }

    pub fn delete_collection(&self, name: &str) -> Result<(), CoreError> {
        let mut collections = self.collections.write();
        collections.remove(name);
        Ok(())
    }

    pub fn list_collections(&self) -> Vec<String> {
        let collections = self.collections.read();
        collections.keys().cloned().collect()
    }

    pub fn stats(&self) -> EngineStats {
        let collections = self.collections.read();
        let collection_count = collections.len();
        let total_heads = collections
            .values()
            .map(|c| c.head_manager.read().head_count())
            .sum();
        let total_vectors = collections
            .values()
            .map(|c| c.head_manager.read().total_vectors())
            .sum();
        EngineStats {
            collection_count,
            total_heads,
            total_vectors,
        }
    }
}

impl Default for AttentionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_collection() {
        let engine = AttentionEngine::new();
        engine
            .create_collection("test", 128, &["semantic", "temporal"])
            .unwrap();
        let stats = engine.stats();
        assert_eq!(stats.collection_count, 1);
        assert_eq!(stats.total_heads, 2);
    }

    #[test]
    fn test_duplicate_collection_fails() {
        let engine = AttentionEngine::new();
        engine.create_collection("dup", 64, &["default"]).unwrap();
        let result = engine.create_collection("dup", 64, &["default"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_collection() {
        let engine = AttentionEngine::new();
        engine.create_collection("docs", 64, &["default"]).unwrap();
        assert_eq!(engine.stats().collection_count, 1);
        engine.delete_collection("docs").unwrap();
        assert_eq!(engine.stats().collection_count, 0);
    }

    #[test]
    fn test_list_collections() {
        let engine = AttentionEngine::new();
        engine.create_collection("a", 64, &["h1"]).unwrap();
        engine.create_collection("b", 64, &["h1"]).unwrap();
        let mut list = engine.list_collections();
        list.sort();
        assert_eq!(list, vec!["a", "b"]);
    }

    #[test]
    fn test_get_nonexistent_collection() {
        let engine = AttentionEngine::new();
        let result = engine.get_collection("nope");
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_and_attend() {
        let engine = AttentionEngine::new();
        engine
            .create_collection("docs", 4, &["semantic"])
            .unwrap();

        engine
            .insert_vector("docs", "semantic", 1, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        engine
            .insert_vector("docs", "semantic", 2, &[0.0, 1.0, 0.0, 0.0])
            .unwrap();

        let results = engine
            .attend("docs", &["semantic".to_string()], &[1.0, 0.0, 0.0, 0.0], 2)
            .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_engine_stats() {
        let engine = AttentionEngine::new();
        engine
            .create_collection("c1", 64, &["a", "b"])
            .unwrap();
        engine.create_collection("c2", 64, &["a"]).unwrap();
        let stats = engine.stats();
        assert_eq!(stats.collection_count, 2);
        assert_eq!(stats.total_heads, 3);
    }
}
