use crate::collection::Collection;
use crate::error::CoreError;
use crate::transaction::TransactionManager;
use attentiondb_query::parse_aql;
use attentiondb_storage::{OpType, Record, Wal};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

/// Bidirectional IdMapper — no hashing, no collisions.
pub struct IdMapper {
    u64_to_uuid: HashMap<u64, Uuid>,
    uuid_to_u64: HashMap<Uuid, u64>,
    next_id: u64,
}

impl IdMapper {
    pub fn new() -> Self {
        Self {
            u64_to_uuid: HashMap::new(),
            uuid_to_u64: HashMap::new(),
            next_id: 1,
        }
    }

    /// Idempotent: returns existing numeric ID if UUID already mapped.
    pub fn register(&mut self, uuid: Uuid) -> u64 {
        if let Some(&id) = self.uuid_to_u64.get(&uuid) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.uuid_to_u64.insert(uuid, id);
        self.u64_to_uuid.insert(id, uuid);
        id
    }

    /// O(1) lookup — no hash collision possible.
    pub fn uuid_to_id(&self, uuid: &Uuid) -> Option<u64> {
        self.uuid_to_u64.get(uuid).copied()
    }

    /// O(1) reverse lookup.
    pub fn id_to_uuid(&self, id: u64) -> Option<&Uuid> {
        self.u64_to_uuid.get(&id)
    }

    pub fn len(&self) -> usize {
        self.uuid_to_u64.len()
    }

    pub fn is_empty(&self) -> bool {
        self.uuid_to_u64.is_empty()
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mappings: Vec<serde_json::Value> = self
            .uuid_to_u64
            .iter()
            .map(|(uuid, id)| {
                serde_json::json!({
                    "uuid": uuid.to_string(),
                    "numeric_id": id
                })
            })
            .collect();
        serde_json::json!({
            "mappings": mappings,
            "next_id": self.next_id
        })
    }

    pub fn from_json(&mut self, value: &serde_json::Value) {
        self.u64_to_uuid.clear();
        self.uuid_to_u64.clear();
        if let Some(mappings) = value.get("mappings").and_then(|m| m.as_array()) {
            for entry in mappings {
                if let (Some(uuid_str), Some(id)) = (
                    entry.get("uuid").and_then(|u| u.as_str()),
                    entry.get("numeric_id").and_then(|n| n.as_u64()),
                ) {
                    if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                        self.uuid_to_u64.insert(uuid, id);
                        self.u64_to_uuid.insert(id, uuid);
                    }
                }
            }
        }
        if let Some(next) = value.get("next_id").and_then(|n| n.as_u64()) {
            self.next_id = next.max(self.uuid_to_u64.len() as u64 + 1);
        }
    }

    pub fn persist_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json();
        let content = serde_json::to_string_pretty(&json)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    pub fn load_from_file(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&content)?;
        self.from_json(&value);
        Ok(())
    }
}

impl Default for IdMapper {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AttentionEngine {
    collections: Arc<RwLock<HashMap<String, Arc<Collection>>>>,
    wal: Arc<parking_lot::Mutex<Option<Wal>>>,
    pub document_store: Arc<RwLock<attentiondb_storage::DocumentStore>>,
    pub id_mapper: Arc<RwLock<IdMapper>>,
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
            id_mapper: Arc::new(RwLock::new(IdMapper::new())),
            txn_manager: TransactionManager::new(),
        }
    }

    pub fn open(
        wal_path: &str,
        durability: attentiondb_storage::Durability,
    ) -> Result<Self, CoreError> {
        let wal = Wal::new(Path::new(wal_path))?.with_durability(durability);
        let dir = Path::new(wal_path)
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();
        let document_store = attentiondb_storage::DocumentStore::open(dir)?;
        Ok(Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            wal: Arc::new(parking_lot::Mutex::new(Some(wal))),
            document_store: Arc::new(RwLock::new(document_store)),
            id_mapper: Arc::new(RwLock::new(IdMapper::new())),
            txn_manager: TransactionManager::new(),
        })
    }

    pub fn create_collection(
        &self,
        name: &str,
        dim: usize,
        heads: &[&str],
    ) -> Result<(), CoreError> {
        self.create_collection_with_settings(
            name,
            dim,
            heads,
            attentiondb_hnsw::CollectionSettings::default(),
        )
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
        for &h in heads {
            collection.add_default_head(h)?;
        }
        collections.insert(name.to_string(), collection);
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Result<Arc<Collection>, CoreError> {
        self.collections
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| CoreError::CollectionNotFound(name.to_string()))
    }

    pub fn attend(
        &self,
        collection: &str,
        heads: &[String],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        self.get_collection(collection)?.attend(heads, query, top_k)
    }

    pub fn attend_weighted(
        &self,
        collection: &str,
        heads: &[(String, f32)],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        self.get_collection(collection)?
            .attend_weighted(heads, query, top_k)
    }

    pub fn insert_vector(
        &self,
        collection: &str,
        head: &str,
        id: u64,
        vector: &[f32],
    ) -> Result<(), CoreError> {
        self.get_collection(collection)?
            .insert_vector(head, id, vector)
    }

    pub fn delete_collection(&self, name: &str) -> Result<(), CoreError> {
        self.collections.write().remove(name);
        Ok(())
    }

    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    pub fn insert_document(
        &self,
        collection_name: &str,
        record: Record,
    ) -> Result<String, CoreError> {
        let collection = self.get_collection(collection_name)?;
        let uuid = record.id;
        let numeric_id = self.id_mapper.write().register(uuid);
        self.document_store.write().insert(record.clone())?;
        if let Some(ref mut wal) = *self.wal.lock() {
            wal.append(OpType::Insert, collection_name, uuid, record.to_msgpack()?)?;
        }
        for (head, vec) in &record.k_vecs {
            collection.insert_vector(head, numeric_id, vec)?;
        }
        let full_text: String = record
            .fields
            .values()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        collection.bm25.insert(numeric_id, &full_text);
        Ok(uuid.to_string())
    }

    pub fn attend_hybrid(
        &self,
        collection: &str,
        heads: &[String],
        query: &[f32],
        text: &str,
        top_k: usize,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        self.get_collection(collection)?
            .attend_hybrid(heads, query, text, top_k)
    }

    pub fn begin_transaction(&self, collection: &str) -> u64 {
        self.txn_manager.begin_transaction(collection)
    }

    pub fn record_transaction_operation(
        &self,
        id: u64,
        op: crate::transaction::TxnOp,
    ) -> Result<(), CoreError> {
        self.txn_manager.record_operation(id, op)
    }

    pub fn rollback_transaction(&self, id: u64) -> Result<bool, CoreError> {
        self.txn_manager.rollback_transaction(id)
    }

    pub fn commit_transaction(&self, txn_id: u64) -> Result<bool, CoreError> {
        if let Some(txn) = self.txn_manager.get_staged_transaction(txn_id) {
            let cn = &txn.collection_name;
            for op in txn.operations {
                match op {
                    crate::transaction::TxnOp::Insert(r) => {
                        self.insert_document(cn, r)?;
                    }
                    crate::transaction::TxnOp::Delete(u) => {
                        self.delete_document(cn, &u.to_string())?;
                    }
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn get_document_fields(&self, numeric_id: u64) -> HashMap<String, String> {
        let mapper = self.id_mapper.read();
        if let Some(uuid) = mapper.id_to_uuid(numeric_id) {
            let store = self.document_store.read();
            if let Some(rec) = store.get(uuid) {
                return rec
                    .fields
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            match v {
                                serde_json::Value::String(s) => s.clone(),
                                o => o.to_string(),
                            },
                        )
                    })
                    .collect();
            }
        }
        HashMap::new()
    }

    pub fn delete_document(&self, collection: &str, id_str: &str) -> Result<bool, CoreError> {
        self.get_collection(collection)?;
        if let Ok(uuid) = Uuid::parse_str(id_str) {
            self.document_store.write().delete(&uuid)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn execute_aql(&self, aql: &str) -> Result<String, CoreError> {
        self.execute_aql_with_vector(aql, None)
    }

    pub fn execute_aql_with_vector(
        &self,
        aql: &str,
        query_vector: Option<&[f32]>,
    ) -> Result<String, CoreError> {
        let stmt = parse_aql(aql)?;
        match stmt {
            attentiondb_query::AQLStatement::Query(q) => {
                let c = self.get_collection(&q.collection)?;
                let heads = if q.heads.is_empty() {
                    c.list_heads()
                } else {
                    q.heads
                };
                let vec = query_vector.ok_or_else(|| {
                    CoreError::InvalidOperation("ATTEND requires a query vector".into())
                })?;
                let r = c.attend(&heads, vec, q.top_k)?;
                Ok(format!("[{} results]", r.len()))
            }
            attentiondb_query::AQLStatement::CreateCollection(coll) => {
                let heads: Vec<&str> = if coll.head_settings.is_empty() {
                    vec!["default"]
                } else {
                    coll.head_settings.keys().map(|s| s.as_str()).collect()
                };
                self.create_collection(&coll.collection, 64, &heads)?;
                Ok(format!("Created '{}'", coll.collection))
            }
            attentiondb_query::AQLStatement::AlterCollection(a) => {
                self.get_collection(&a.collection)?;
                Ok(format!("Altered '{}'", a.collection))
            }
        }
    }

    pub fn execute_reprojection_job(
        &self,
        job: &attentiondb_learned::ReprojectionJob,
    ) -> Result<(), CoreError> {
        let c = self.get_collection(&job.collection)?;
        let records = self.document_store.read().list_all_records();
        let mut updated = Vec::new();
        for mut rec in records {
            if rec.tags.contains(&format!("collection:{}", job.collection)) {
                let nid = self.id_mapper.write().register(rec.id);
                let mut new_kv = HashMap::new();
                for (h, v) in &rec.k_vecs {
                    let rp = job.new_projection.project_key(v);
                    c.insert_vector(h, nid, &rp)?;
                    new_kv.insert(h.clone(), rp);
                }
                rec.k_vecs = new_kv;
                updated.push(rec);
            }
        }
        for rec in updated {
            self.document_store.write().update_record(rec)?;
        }
        Ok(())
    }

    pub fn is_persistent(&self) -> bool {
        self.wal.lock().is_some()
    }

    pub fn flush_wal(&self) -> Result<(), CoreError> {
        if let Some(ref mut wal) = *self.wal.lock() {
            wal.fsync()
                .map_err(|e| CoreError::InvalidOperation(e.to_string()))
        } else {
            Ok(())
        }
    }

    pub fn stats(&self) -> EngineStats {
        let cols = self.collections.read();
        EngineStats {
            collection_count: cols.len(),
            total_heads: cols
                .values()
                .map(|c| c.head_manager.read().head_count())
                .sum(),
            total_vectors: cols
                .values()
                .map(|c| c.head_manager.read().total_vectors())
                .sum(),
        }
    }

    pub fn persist_id_mapper(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.id_mapper.read().persist_to_file(path)
    }

    pub fn load_id_mapper(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.id_mapper.write().load_from_file(path)
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
    use std::collections::HashMap;

    #[test]
    fn test_id_mapper_no_duplicates() {
        let mut m = IdMapper::new();
        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        assert_ne!(m.register(u1), m.register(u2));
        assert_eq!(m.register(u1), m.register(u1));
    }

    #[test]
    fn test_id_mapper_roundtrip() {
        let mut m = IdMapper::new();
        let u = Uuid::new_v4();
        let id = m.register(u);
        assert_eq!(m.id_to_uuid(id), Some(&u));
        assert_eq!(m.uuid_to_id(&u), Some(id));
    }

    #[test]
    fn test_id_mapper_persist_json() {
        let mut m = IdMapper::new();
        let u = Uuid::new_v4();
        m.register(u);
        let json = m.to_json();
        let mut m2 = IdMapper::new();
        m2.from_json(&json);
        assert_eq!(m.len(), m2.len());
        assert_eq!(m2.uuid_to_id(&u), Some(1));
    }

    #[test]
    fn test_create_collection() {
        let e = AttentionEngine::new();
        e.create_collection("t", 128, &["a", "b"]).unwrap();
        assert_eq!(e.stats().collection_count, 1);
    }

    #[test]
    fn test_duplicate_collection_fails() {
        let e = AttentionEngine::new();
        e.create_collection("x", 64, &["h"]).unwrap();
        assert!(e.create_collection("x", 64, &["h"]).is_err());
    }

    #[test]
    fn test_insert_and_attend() {
        let e = AttentionEngine::new();
        e.create_collection("d", 4, &["s"]).unwrap();
        e.insert_vector("d", "s", 1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let r = e
            .attend("d", &["s".into()], &[1.0, 0.0, 0.0, 0.0], 5)
            .unwrap();
        assert!(!r.is_empty());
    }

    #[test]
    fn test_insert_document_with_id_mapper() {
        let e = AttentionEngine::new();
        e.create_collection("c", 4, &["s"]).unwrap();
        let mut rec = Record::new(HashMap::new());
        rec.k_vecs.insert("s".into(), vec![0.1; 4]);
        let id_str = e.insert_document("c", rec).unwrap();
        let uuid = Uuid::parse_str(&id_str).unwrap();
        assert_eq!(e.id_mapper.read().uuid_to_id(&uuid), Some(1));
    }

    #[test]
    fn test_engine_stats() {
        let e = AttentionEngine::new();
        e.create_collection("c1", 64, &["a", "b"]).unwrap();
        e.create_collection("c2", 64, &["a"]).unwrap();
        assert_eq!(e.stats().collection_count, 2);
        assert_eq!(e.stats().total_heads, 3);
    }
}
