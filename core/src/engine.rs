use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::path::Path;
use parking_lot::RwLock;
use attentiondb_query::parse_aql;
use attentiondb_storage::{Wal, OpType, Record};
use crate::collection::Collection;
use crate::error::CoreError;
use crate::transaction::TransactionManager;

/// Deterministically convert a UUID to a stable u64 hash.
///
/// This avoids silently truncating the UUID's high 64 bits and retains a
/// low probability of collision using a deterministic hash over the full 128 bits.
fn uuid_to_u64(id: &uuid::Uuid) -> u64 {
    let mut hasher = DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
}

pub struct AttentionEngine {
    collections: Arc<RwLock<HashMap<String, Arc<Collection>>>>,
    wal: Arc<parking_lot::Mutex<Option<Wal>>>,
    pub document_store: Arc<RwLock<attentiondb_storage::DocumentStore>>,
    id_map: Arc<RwLock<HashMap<u64, uuid::Uuid>>>,
    pub txn_manager: TransactionManager,
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
            wal: Arc::new(parking_lot::Mutex::new(None)),
            document_store: Arc::new(RwLock::new(attentiondb_storage::DocumentStore::new())),
            id_map: Arc::new(RwLock::new(HashMap::new())),
            txn_manager: TransactionManager::new(),
        }
    }

    pub fn open(wal_path: &str, durability: attentiondb_storage::Durability) -> Result<Self, CoreError> {
        let wal = Wal::new(Path::new(wal_path))?.with_durability(durability);
        let dir = Path::new(wal_path).parent().unwrap_or(Path::new(".")).to_path_buf();
        let document_store = attentiondb_storage::DocumentStore::open(dir)?;

        Ok(Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            wal: Arc::new(parking_lot::Mutex::new(Some(wal))),
            document_store: Arc::new(RwLock::new(document_store)),
            id_map: Arc::new(RwLock::new(HashMap::new())),
            txn_manager: TransactionManager::new(),
        })
    }

    pub fn create_collection(
        &self,
        name: &str,
        dim: usize,
        heads: &[&str],
    ) -> Result<(), CoreError> {
        self.create_collection_with_settings(name, dim, heads, attentiondb_hnsw::CollectionSettings::default())
    }

    pub fn create_collection_with_settings(
        &self,
        name: &str,
        dim: usize,
        heads: &[&str],
        settings: attentiondb_hnsw::CollectionSettings,
    ) -> Result<(), CoreError> {
        let mut collections = self.collections.write();
        if collections.contains_key(name) {
            return Err(CoreError::CollectionAlreadyExists(name.to_string()));
        }

        let collection = Arc::new(Collection::new(name, dim));
        *collection.settings.write() = settings;
        for head_name in heads {
            collection.add_default_head(head_name)?;
        }

        collections.insert(name.to_string(), collection);
        Ok(())
    }

    pub fn alter_collection_settings(
        &self,
        name: &str,
        settings: attentiondb_hnsw::CollectionSettings,
    ) -> Result<(), CoreError> {
        let collection = self.get_collection(name)?;
        *collection.settings.write() = settings;
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

    pub fn insert_document(&self, collection_name: &str, record: Record) -> Result<String, CoreError> {
        let collection = self.get_collection(collection_name)?;

        let id = record.id.to_string();
        let numeric_id = uuid_to_u64(&record.id);

        {
            let mut id_map = self.id_map.write();
            id_map.insert(numeric_id, record.id);
        }

        {
            let mut store = self.document_store.write();
            store.insert(record.clone())?;
        }

        {
            let mut wal_guard = self.wal.lock();
            if let Some(wal) = wal_guard.as_mut() {
                let record_bytes = record.to_msgpack()?;
                wal.append(OpType::Insert, collection_name, record.id, record_bytes)?;
            }
        }

        for (head_name, vec_data) in &record.k_vecs {
            collection.insert_vector(head_name, numeric_id, vec_data)?;
        }

        let mut full_text = String::new();
        for value in record.fields.values() {
            if let serde_json::Value::String(s) = value {
                full_text.push_str(s);
                full_text.push(' ');
            }
        }
        collection.bm25.insert(numeric_id, &full_text);

        Ok(id)
    }

    pub fn attend_hybrid(
        &self,
        collection_name: &str,
        heads: &[String],
        query_vector: &[f32],
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let collection = self.get_collection(collection_name)?;
        collection.attend_hybrid(heads, query_vector, query_text, top_k)
    }

    pub fn begin_transaction(&self, collection_name: &str) -> u64 {
        self.txn_manager.begin_transaction(collection_name)
    }

    pub fn record_transaction_operation(&self, txn_id: u64, op: crate::transaction::TxnOp) -> Result<(), CoreError> {
        self.txn_manager.record_operation(txn_id, op)
    }

    pub fn rollback_transaction(&self, txn_id: u64) -> Result<bool, CoreError> {
        self.txn_manager.rollback_transaction(txn_id)
    }

    pub fn commit_transaction(&self, txn_id: u64) -> Result<bool, CoreError> {
        if let Some(txn) = self.txn_manager.get_staged_transaction(txn_id) {
            let collection_name = &txn.collection_name;
            for op in txn.operations {
                match op {
                    crate::transaction::TxnOp::Insert(rec) => {
                        self.insert_document(collection_name, rec)?;
                    }
                    crate::transaction::TxnOp::Delete(uuid) => {
                        self.delete_document(collection_name, &uuid.to_string())?;
                    }
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn get_document_fields(&self, numeric_id: u64) -> HashMap<String, String> {
        let id_map = self.id_map.read();
        if let Some(uuid) = id_map.get(&numeric_id) {
            let store = self.document_store.read();
            if let Some(record) = store.get(uuid) {
                return record.fields.iter().map(|(k, v)| {
                    let s = match v {
                        serde_json::Value::String(str) => str.clone(),
                        other => other.to_string(),
                    };
                    (k.clone(), s)
                }).collect();
            }
        }
        HashMap::new()
    }

    pub fn delete_document(&self, collection_name: &str, id_str: &str) -> Result<bool, CoreError> {
        let _collection = self.get_collection(collection_name)?;
        if let Ok(uuid) = uuid::Uuid::parse_str(id_str) {
            let mut store = self.document_store.write();
            store.delete(&uuid)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn execute_aql(&self, aql: &str) -> Result<String, CoreError> {
        self.execute_aql_with_vector(aql, None)
    }

    pub fn execute_aql_with_vector(&self, aql: &str, query_vector: Option<&[f32]>) -> Result<String, CoreError> {
        let statement = parse_aql(aql)?;

        match statement {
            attentiondb_query::AQLStatement::Query(query) => {
                let collection = self.get_collection(&query.collection)?;
                let heads: Vec<String> = if query.heads.is_empty() {
                    collection.list_heads()
                } else {
                    query.heads.clone()
                };

                let vec = query_vector.ok_or_else(|| CoreError::InvalidOperation(
                    format!(
                        "ATTEND query requires a pre-computed query vector. query_text='{}' is a semantic label, not a vector. Call execute_aql_with_vector() with the embedded vector.",
                        query.query_text
                    )
                ))?;

                let results = collection.attend(&heads, vec, query.top_k)?;
                Ok(format!("[{} results] for '{}' on {}", results.len(), query.query_text, query.collection))
            }
            attentiondb_query::AQLStatement::CreateCollection(coll) => {
                let heads: Vec<&str> = if coll.head_settings.is_empty() {
                    vec!["default"]
                } else {
                    coll.head_settings.keys().map(|s| s.as_str()).collect()
                };
                self.create_collection(&coll.collection, 64, &heads)?;
                Ok(format!("Created collection '{}'", coll.collection))
            }
            attentiondb_query::AQLStatement::AlterCollection(alter) => {
                let _collection = self.get_collection(&alter.collection)?;
                Ok(format!("Altered collection '{}'", alter.collection))
            }
        }
    }

    pub fn execute_reprojection_job(&self, job: &attentiondb_learned::ReprojectionJob) -> Result<(), CoreError> {
        let target_collection = &job.collection;
        let collection = self.get_collection(target_collection)?;

        let store = self.document_store.read();
        let records = store.list_all_records();
        drop(store);

        let mut updated_records = Vec::new();

        for mut record in records {
            if record.tags.contains(&format!("collection:{}", target_collection)) {
                let numeric_id = uuid_to_u64(&record.id);

                let mut new_k_vecs = HashMap::new();
                for (head_name, vec) in &record.k_vecs {
                    let reprojected = job.new_projection.project_key(vec);
                    collection.insert_vector(head_name, numeric_id, &reprojected)?;
                    new_k_vecs.insert(head_name.clone(), reprojected);
                }

                record.k_vecs = new_k_vecs;
                updated_records.push(record);
            }
        }

        let mut store_write = self.document_store.write();
        for record in updated_records {
            store_write.update_record(record)?;
        }

        Ok(())
    }

    pub fn is_persistent(&self) -> bool {
        self.wal.lock().is_some()
    }

    pub fn flush_wal(&self) -> Result<(), CoreError> {
        let mut wal_guard = self.wal.lock();
        if let Some(wal) = wal_guard.as_mut() {
            wal.fsync().map_err(|e| CoreError::InvalidOperation(e.to_string()))
        } else {
            Ok(())
        }
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
    fn test_execute_reprojection_job() {
        let engine = AttentionEngine::new();
        engine.create_collection("papers", 4, &["semantic"]).unwrap();

        let mut fields = HashMap::new();
        fields.insert("title".to_string(), serde_json::json!("Test"));
        let mut rec = Record::new(fields);
        rec.k_vecs.insert("semantic".to_string(), vec![1.0, 0.0, 0.0, 0.0]);

        engine.insert_document("papers", rec).unwrap();

        let config = attentiondb_learned::ProjectionConfig { dim: 4, num_heads: 1, head_dim: 4 };
        let old = attentiondb_learned::ProjectionMatrix::new(config.clone());
        let new = attentiondb_learned::ProjectionMatrix::new(config);
        let job = attentiondb_learned::ReprojectionJob::new("papers", old, new);

        assert!(engine.execute_reprojection_job(&job).is_ok());
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
