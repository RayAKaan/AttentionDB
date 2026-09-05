//! AttentionEngine — the single-node durable database engine (Phase 1).
//!
//! Authoritative state = catalog (manifest) + SSTables + WAL segments + idmap
//! snapshot. Every logical mutation is written to the authoritative WAL before
//! it is applied to subsystems; recovery = load catalog → load SSTables →
//! load idmap → replay WAL → rebuild indexes deterministically → validate → READY.
//!
//! See docs/consistency-model.md for the invariants enforced here and
//! docs/recovery.md for the startup state machine.

use crate::collection::Collection;
use crate::error::CoreError;
use crate::transaction::{Transaction, TransactionManager, TxnOp};
use attentiondb_query::parse_aql;
use attentiondb_storage::{
    Catalog, CollectionMeta, Durability, HeadMeta, IdMapSnapshot, Record, RecordKind, Wal,
    WalRecord, DATABASE_FORMAT_VERSION, WAL_DIR_NAME,
};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

/// Maximum operations buffered for a single transaction during replay
/// (resource limit — a transaction larger than this is a corruption error).
const MAX_TXN_OPS: usize = 100_000;

/// Database lifecycle states (docs/recovery.md). Traffic is only accepted in `Ready`.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineState {
    Closed,
    Opening,
    LoadingCatalog,
    LoadingStorage,
    LoadingIndexes,
    ReplayingWal,
    Validating,
    Ready,
    RecoveryFailed(String),
}

/// Bidirectional IdMapper — no hashing, no collisions. Phase 1 adds:
/// exact persistence ([`IdMapper::snapshot`] / [`IdMapper::restore_snapshot`]),
/// retired ids (delete/update never reuse a numeric id — INV-6), and forced-id
/// restore used by WAL replay.
pub struct IdMapper {
    u64_to_uuid: HashMap<u64, Uuid>,
    uuid_to_u64: HashMap<Uuid, u64>,
    retired: std::collections::HashSet<u64>,
    next_id: u64,
}

impl IdMapper {
    pub fn new() -> Self {
        Self {
            u64_to_uuid: HashMap::new(),
            uuid_to_u64: HashMap::new(),
            retired: std::collections::HashSet::new(),
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

    /// Force-restore an exact mapping (WAL replay / snapshot load). Never mints.
    pub fn restore(&mut self, uuid: Uuid, id: u64) {
        self.u64_to_uuid.insert(id, uuid);
        self.uuid_to_u64.insert(uuid, id);
        self.retired.remove(&id);
        if id >= self.next_id {
            self.next_id = id + 1;
        }
    }

    /// Mint a fresh numeric id for a uuid whose old id is being retired
    /// (update path). The old id can never be reused (INV-6).
    pub fn remap(&mut self, uuid: Uuid) -> (u64, Option<u64>) {
        let old = self.uuid_to_u64.remove(&uuid);
        if let Some(o) = old {
            self.u64_to_uuid.remove(&o);
            self.retired.insert(o);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.uuid_to_u64.insert(uuid, id);
        self.u64_to_uuid.insert(id, uuid);
        (id, old)
    }

    /// Retire a document: mappings removed, numeric id permanently retired.
    pub fn retire(&mut self, uuid: &Uuid) -> Option<u64> {
        let id = self.uuid_to_u64.remove(uuid)?;
        self.u64_to_uuid.remove(&id);
        self.retired.insert(id);
        Some(id)
    }

    /// Retire a raw numeric id without a uuid (update/delete replay paths).
    pub fn retire_mark(&mut self, id: u64) {
        self.retired.insert(id);
    }

    pub fn is_retired(&self, id: u64) -> bool {
        self.retired.contains(&id)
    }

    pub fn retired_ids(&self) -> Vec<u64> {
        self.retired.iter().copied().collect()
    }

    /// O(1) lookup — no hash collision possible.
    pub fn uuid_to_id(&self, uuid: &Uuid) -> Option<u64> {
        self.uuid_to_u64.get(uuid).copied()
    }

    /// O(1) reverse lookup.
    pub fn id_to_uuid(&self, id: u64) -> Option<&Uuid> {
        self.u64_to_uuid.get(&id)
    }

    pub fn contains_uuid(&self, uuid: &Uuid) -> bool {
        self.uuid_to_u64.contains_key(uuid)
    }

    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    pub fn len(&self) -> usize {
        self.uuid_to_u64.len()
    }

    pub fn is_empty(&self) -> bool {
        self.uuid_to_u64.is_empty()
    }

    pub fn snapshot(&self) -> IdMapSnapshot {
        IdMapSnapshot {
            format_version: attentiondb_storage::IDMAP_FORMAT_VERSION,
            next_id: self.next_id,
            mappings: self
                .uuid_to_u64
                .iter()
                .map(|(u, i)| (u.to_string(), *i))
                .collect(),
            retired: self.retired.iter().copied().collect(),
        }
    }

    pub fn restore_snapshot(&mut self, snap: &IdMapSnapshot) {
        self.u64_to_uuid.clear();
        self.uuid_to_u64.clear();
        self.retired.clear();
        self.next_id = snap.next_id;
        for (uuid_str, id) in &snap.mappings {
            if let Ok(uuid) = Uuid::parse_str(uuid_str) {
                self.uuid_to_u64.insert(uuid, *id);
                self.u64_to_uuid.insert(*id, uuid);
            }
        }
        for id in &snap.retired {
            self.retired.insert(*id);
        }
        // Safety: next_id must exceed every known id (INV-6).
        let max_known = self
            .u64_to_uuid
            .keys()
            .copied()
            .chain(self.retired.iter().copied())
            .max()
            .unwrap_or(0);
        if self.next_id <= max_known {
            self.next_id = max_known + 1;
        }
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

#[derive(Debug, Clone)]
pub struct EngineStats {
    pub collection_count: usize,
    pub total_heads: usize,
    pub total_vectors: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CheckpointInfo {
    pub checkpoint_seq: u64,
    pub manifest_generation: u64,
    pub documents: usize,
    pub collections: usize,
    pub duration_ms: f64,
}

/// Buffered transaction operations during replay (applied on COMMIT only).
#[derive(Default)]
struct TxnBuffer {
    ops: Vec<ReplayOp>,
}

/// A decoded, applicable operation extracted from a WAL record or a txn buffer.
/// Some payload fields (ids, schema) are carried for the checker/future tooling.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ReplayOp {
    CreateCollection {
        id: u32,
        name: String,
        dim: usize,
        heads: Vec<HeadMeta>,
        settings_json: String,
        schema_fields: Vec<(String, String)>,
    },
    DropCollection {
        name: String,
    },
    AlterCollection {
        name: String,
        settings_json: String,
    },
    InsertDocument {
        collection: String,
        uuid: Uuid,
        numeric_id: u64,
        record: Record,
    },
    UpdateDocument {
        collection: String,
        uuid: Uuid,
        old_numeric_id: u64,
        new_numeric_id: u64,
        record: Record,
    },
    DeleteDocument {
        collection: String,
        uuid: Uuid,
        numeric_id: u64,
    },
}

pub struct AttentionEngine {
    pub collections: Arc<RwLock<HashMap<String, Arc<Collection>>>>,
    pub document_store: Arc<RwLock<attentiondb_storage::DocumentStore>>,
    pub id_mapper: Arc<RwLock<IdMapper>>,
    pub txn_manager: TransactionManager,
    catalog: Arc<Mutex<Catalog>>,
    data_dir: Option<PathBuf>,
    wal: Arc<Mutex<Option<Wal>>>,
    durability: Durability,
    state: Arc<RwLock<EngineState>>,
    manifest_generation: AtomicU64,
    /// Serializes mutations against checkpoints/backups (correctness > concurrency).
    mutation_gate: Arc<Mutex<()>>,
    /// Hard query limits (§18) enforced at every query entry point.
    pub query_limits: crate::planner::QueryLimits,
    /// Set during shutdown: rejects new writes (graceful shutdown step 1).
    accepting_writes: Arc<AtomicBool>,
}

impl AttentionEngine {
    // ================================================================
    // Construction
    // ================================================================

    /// In-memory engine (tests / ephemeral use). Nothing is persisted.
    pub fn new() -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            document_store: Arc::new(RwLock::new(attentiondb_storage::DocumentStore::new())),
            id_mapper: Arc::new(RwLock::new(IdMapper::new())),
            txn_manager: TransactionManager::new(),
            catalog: Arc::new(Mutex::new(Catalog::fresh())),
            data_dir: None,
            wal: Arc::new(Mutex::new(None)),
            durability: Durability::GroupCommit,
            state: Arc::new(RwLock::new(EngineState::Ready)),
            manifest_generation: AtomicU64::new(0),
            mutation_gate: Arc::new(Mutex::new(())),
            query_limits: crate::planner::QueryLimits::default(),
            accepting_writes: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Legacy entry point kept for compatibility: opens the database directory
    /// that contains `wal_path`. New code should call [`AttentionEngine::open_dir`].
    pub fn open(wal_path: &str, durability: Durability) -> Result<Self, CoreError> {
        let db_dir = Path::new(wal_path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        Self::open_dir(&db_dir, durability)
    }

    /// Durable open with full recovery. Runs the Phase 1 recovery pipeline:
    ///
    /// ```text
    /// OPENING → LOADING_CATALOG → LOADING_STORAGE → LOADING_INDEXES
    ///         → REPLAYING_WAL → VALIDATING → READY
    /// ```
    ///
    /// Any failure returns `Err(RecoveryFailed | Corruption | …)` — the caller
    /// MUST NOT serve traffic (INV-11). There is no in-memory fallback.
    pub fn open_dir(db_dir: &Path, durability: Durability) -> Result<Self, CoreError> {
        let started = std::time::Instant::now();
        let set_state = |s: EngineState| {
            tracing::info!(state = ?s, "recovery: stage");
        };

        set_state(EngineState::Opening);
        Catalog::ensure_dirs(db_dir)?;

        // ---- 1. Catalog -----------------------------------------------------
        set_state(EngineState::LoadingCatalog);
        let (catalog, catalog_fallback) = Catalog::load(db_dir)?;
        if catalog.database_format_version > DATABASE_FORMAT_VERSION {
            return Err(CoreError::RecoveryFailed(format!(
                "database format v{} is newer than supported v{} — upgrade required",
                catalog.database_format_version, DATABASE_FORMAT_VERSION
            )));
        }
        if catalog_fallback {
            tracing::warn!(
                "catalog recovered from a previous manifest generation (CURRENT was unusable)"
            );
        }

        // ---- 2. Storage (SSTables, timestamp-merged, strict) -----------------
        set_state(EngineState::LoadingStorage);
        let document_store =
            attentiondb_storage::DocumentStore::open_without_wal(Catalog::sst_dir(db_dir))?;

        // ---- 3. Id map snapshot ---------------------------------------------
        let mut id_mapper = IdMapper::new();
        if let Some(snap) = IdMapSnapshot::load(db_dir)? {
            id_mapper.restore_snapshot(&snap);
        }

        // ---- 4. WAL replay ---------------------------------------------------
        set_state(EngineState::ReplayingWal);
        let wal_dir = db_dir.join(WAL_DIR_NAME);
        let mut wal = Wal::open(&wal_dir, durability, default_max_segment_bytes())?;
        let replay_started = std::time::Instant::now();
        let outcome = wal
            .replay(catalog.checkpoint_seq)
            .map_err(|e| CoreError::RecoveryFailed(format!("WAL replay failed: {e}")))?;
        let replayed = outcome.records.len();
        if outcome.torn_tail {
            tracing::warn!("WAL had a torn tail (crash during append); tail was truncated — data after the last valid record was not acknowledged");
        }

        let engine = Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            document_store: Arc::new(RwLock::new(document_store)),
            id_mapper: Arc::new(RwLock::new(id_mapper)),
            txn_manager: TransactionManager::new(),
            catalog: Arc::new(Mutex::new(catalog.clone())),
            data_dir: Some(db_dir.to_path_buf()),
            wal: Arc::new(Mutex::new(Some(wal))),
            durability,
            state: Arc::new(RwLock::new(EngineState::Opening)),
            manifest_generation: AtomicU64::new(0),
            mutation_gate: Arc::new(Mutex::new(())),
            query_limits: crate::planner::QueryLimits::default(),
            accepting_writes: Arc::new(AtomicBool::new(true)),
        };

        // Materialize collections from the catalog first (INV-7/8), then let the
        // WAL replay converge anything that happened after the last manifest save.
        for meta in &catalog.collections {
            if !meta.dropped {
                let coll = engine.materialize_collection(meta)?;
                engine.collections.write().insert(meta.name.clone(), coll);
            }
        }

        let mut txn_buf: HashMap<u64, TxnBuffer> = HashMap::new();
        for record in &outcome.records {
            engine.apply_replay_record(record, &mut txn_buf, replayed)?;
        }
        // Uncommitted transactions at the end of the log are discarded (documented:
        // commit marker = atomicity boundary). Aborts are implicit.
        if !txn_buf.is_empty() {
            tracing::warn!(
                count = txn_buf.len(),
                "discarded uncommitted transaction(s) found at end of WAL (never committed)"
            );
        }
        tracing::info!(
            wal_replay_start = outcome.first_seq,
            wal_replay_end = outcome.last_seq,
            records_replayed = replayed,
            replay_ms = replay_started.elapsed().as_millis() as u64,
            "WAL replay complete"
        );

        // ---- 5. Index rebuild (deterministic) --------------------------------
        set_state(EngineState::LoadingIndexes);
        let rebuild_started = std::time::Instant::now();
        engine.rebuild_all_indexes()?;
        tracing::info!(
            rebuild_ms = rebuild_started.elapsed().as_millis() as u64,
            "indexes rebuilt deterministically from authoritative records"
        );

        // ---- 6. Validate ------------------------------------------------------
        set_state(EngineState::Validating);
        for coll in engine.collections.read().values() {
            coll.purge_retired_from_vector_store();
        }
        let issues = crate::checker::check_engine(&engine);
        let errors: Vec<_> = issues.iter().filter(|i| i.severity.is_error()).collect();
        if !errors.is_empty() {
            for i in &errors {
                tracing::error!(code = %i.code, collection = %i.collection, detail = %i.detail, "consistency check failed");
            }
            return Err(CoreError::RecoveryFailed(format!(
                "{} consistency error(s) detected after recovery — refusing to become READY",
                errors.len()
            )));
        }

        // ---- 7. Ready ----------------------------------------------------------
        *engine.state.write() = EngineState::Ready;
        engine
            .manifest_generation
            .store(catalog.checkpoint_seq, Ordering::SeqCst);
        tracing::info!(
            database = %db_dir.display(),
            format_version = DATABASE_FORMAT_VERSION,
            checkpoint_seq = catalog.checkpoint_seq,
            collections = engine.collections.read().len(),
            documents = engine.document_store.read().len(),
            recovery_ms = started.elapsed().as_millis() as u64,
            status = "READY",
            "AttentionDB recovery complete"
        );
        Ok(engine)
    }

    // ================================================================
    // State / introspection
    // ================================================================

    pub fn state(&self) -> EngineState {
        self.state.read().clone()
    }

    pub fn is_ready(&self) -> bool {
        matches!(*self.state.read(), EngineState::Ready)
    }

    pub fn data_dir(&self) -> Option<&Path> {
        self.data_dir.as_deref()
    }

    pub fn durability(&self) -> Durability {
        self.durability
    }

    pub fn is_persistent(&self) -> bool {
        self.data_dir.is_some()
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

    pub fn catalog_snapshot(&self) -> Catalog {
        self.catalog.lock().clone()
    }

    // ================================================================
    // Collections (durable)
    // ================================================================

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
        let _gate = self.mutation_gate.lock();
        self.check_writable()?;
        {
            let cols = self.collections.read();
            if cols.contains_key(name) {
                return Err(CoreError::CollectionAlreadyExists(name.to_string()));
            }
        }
        settings.validate().map_err(CoreError::InvalidConfig)?;
        let settings_json =
            serde_json::to_string(&settings).map_err(|e| CoreError::Internal(e.to_string()))?;
        let head_metas: Vec<HeadMeta> = heads
            .iter()
            .map(|h| HeadMeta {
                name: h.to_string(),
                settings_json: String::new(),
            })
            .collect();

        let (coll_id, seq) = {
            let mut cat = self.catalog.lock();
            if cat.collections.iter().any(|c| c.name == name && !c.dropped) {
                return Err(CoreError::CollectionAlreadyExists(name.to_string()));
            }
            let id = cat.next_collection_id;
            cat.next_collection_id += 1;
            let seq = self.next_seq_hint(&cat);
            cat.collections.push(CollectionMeta {
                id,
                name: name.to_string(),
                dim,
                heads: head_metas.clone(),
                settings_json: settings_json.clone(),
                schema_fields: Vec::new(),
                created_seq: seq,
                dropped: false,
            });
            (id, seq)
        };

        // WAL first (durability), then manifest, then RAM (INV-7 + fail-stop).
        let mut rec = WalRecord::new(0, RecordKind::CreateCollection);
        rec.collection = name.to_string();
        rec.numeric_id = coll_id as u64;
        rec.payload = serde_json::to_vec(&serde_json::json!({
            "dim": dim,
            "heads": head_metas,
            "settings": &settings_json,
            "seq": seq,
        }))
        .map_err(|e| CoreError::Internal(e.to_string()))?;
        self.wal_append(rec)?;

        self.persist_catalog()?;

        let coll = {
            let cat = self.catalog.lock();
            let meta = cat
                .live_collection(name)
                .ok_or_else(|| CoreError::Internal("catalog entry vanished".into()))?
                .clone();
            self.materialize_collection(&meta)?
        };
        self.collections.write().insert(name.to_string(), coll);
        Ok(())
    }

    /// Drop a collection: tombstoned in catalog, recorded in WAL. Physical data
    /// is reclaimed at the next checkpoint's full compaction.
    pub fn drop_collection(&self, name: &str) -> Result<(), CoreError> {
        let _gate = self.mutation_gate.lock();
        self.check_writable()?;
        {
            let cols = self.collections.read();
            if !cols.contains_key(name) {
                return Err(CoreError::CollectionNotFound(name.to_string()));
            }
        }
        let mut rec = WalRecord::new(0, RecordKind::DropCollection);
        rec.collection = name.to_string();
        self.wal_append(rec)?;

        {
            let mut cat = self.catalog.lock();
            if let Some(meta) = cat.collections.iter_mut().find(|c| c.name == name) {
                meta.dropped = true;
            }
        }
        self.persist_catalog()?;
        self.collections.write().remove(name);
        Ok(())
    }

    pub fn alter_collection_settings(
        &self,
        name: &str,
        settings: attentiondb_hnsw::CollectionSettings,
    ) -> Result<(), CoreError> {
        let _gate = self.mutation_gate.lock();
        self.check_writable()?;
        settings.validate().map_err(CoreError::InvalidConfig)?;
        let collection = self.get_collection(name)?;
        let settings_json =
            serde_json::to_string(&settings).map_err(|e| CoreError::Internal(e.to_string()))?;

        let mut rec = WalRecord::new(0, RecordKind::AlterCollection);
        rec.collection = name.to_string();
        rec.payload = settings_json.clone().into_bytes();
        self.wal_append(rec)?;

        {
            let mut cat = self.catalog.lock();
            let meta = cat
                .collections
                .iter_mut()
                .find(|c| c.name == name && !c.dropped)
                .ok_or_else(|| CoreError::CollectionNotFound(name.to_string()))?;
            meta.settings_json = settings_json;
        }
        self.persist_catalog()?;
        *collection.settings.write() = settings;
        Ok(())
    }

    pub fn get_collection(&self, name: &str) -> Result<Arc<Collection>, CoreError> {
        self.collections
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| CoreError::CollectionNotFound(name.to_string()))
    }

    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    // ================================================================
    // Documents (durable insert / update / upsert / delete)
    // ================================================================

    /// Durable insert: WAL record (with pre-assigned numeric id) → apply to
    /// DocumentStore → heads → BM25. The collection membership tag is stored in
    /// the record itself so recovery can rebuild per-collection membership.
    pub fn insert_document(
        &self,
        collection_name: &str,
        mut record: Record,
    ) -> Result<String, CoreError> {
        let _gate = self.mutation_gate.lock();
        self.check_writable()?;
        let collection = self.get_collection(collection_name)?;
        self.validate_record_for_collection(&collection, &record)?;
        let uuid = record.id;
        let numeric_id = self.id_mapper.write().register(uuid);
        let membership_tag = format!("collection:{collection_name}");
        if !record.tags.contains(&membership_tag) {
            record.tags.push(membership_tag);
        }

        let mut rec = WalRecord::new(0, RecordKind::InsertDocument);
        rec.collection = collection_name.to_string();
        rec.doc_id = *uuid.as_bytes();
        rec.numeric_id = numeric_id;
        rec.payload = record.to_msgpack()?;
        self.wal_append(rec)?;

        self.apply_insert(&collection, uuid, numeric_id, record)?;
        Ok(uuid.to_string())
    }

    /// Durable update: retire the old vector id, mint a new one, replace fields
    /// and vectors. Old representation can never be returned (INV-4/5).
    pub fn update_document(
        &self,
        collection_name: &str,
        id_str: &str,
        fields: HashMap<String, serde_json::Value>,
        k_vecs: HashMap<String, Vec<f32>>,
    ) -> Result<String, CoreError> {
        let _gate = self.mutation_gate.lock();
        self.check_writable()?;
        let collection = self.get_collection(collection_name)?;
        let uuid = Uuid::parse_str(id_str)
            .map_err(|_| CoreError::InvalidArgument(format!("invalid document id '{id_str}'")))?;

        let old_numeric = self
            .id_mapper
            .read()
            .uuid_to_id(&uuid)
            .ok_or_else(|| CoreError::NotFound(format!("document {id_str} not found")))?;
        if self.document_store.read().get(&uuid).is_none() {
            return Err(CoreError::NotFound(format!("document {id_str} not found")));
        }

        // Build the new record, preserving identity + membership tags.
        let old_record = self
            .document_store
            .read()
            .get(&uuid)
            .cloned()
            .ok_or_else(|| CoreError::NotFound(format!("document {id_str} not found")))?;
        let mut new_record = old_record;
        new_record.fields = fields;
        new_record.k_vecs = k_vecs;
        new_record.version += 1;
        new_record.timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        let tag = format!("collection:{collection_name}");
        if !new_record.tags.contains(&tag) {
            new_record.tags.push(tag);
        }

        self.validate_record_for_collection(&collection, &new_record)?;

        let (new_numeric, old_retired) = self.id_mapper.write().remap(uuid);
        let _ = old_retired;

        let mut rec = WalRecord::new(0, RecordKind::UpdateDocument);
        rec.collection = collection_name.to_string();
        rec.doc_id = *uuid.as_bytes();
        rec.numeric_id = new_numeric;
        let mut payload = Vec::with_capacity(8 + 64);
        payload.extend_from_slice(&old_numeric.to_be_bytes());
        payload.extend_from_slice(&new_record.to_msgpack()?);
        rec.payload = payload;
        self.wal_append(rec)?;

        // Apply: retire old id everywhere, then insert new representation.
        collection.retire_id(old_numeric);
        collection.bm25.remove(old_numeric);
        self.apply_insert(&collection, uuid, new_numeric, new_record)?;
        Ok(uuid.to_string())
    }

    /// Durable upsert: update when the uuid exists, insert otherwise.
    pub fn upsert_document(
        &self,
        collection_name: &str,
        uuid: Uuid,
        fields: HashMap<String, serde_json::Value>,
        k_vecs: HashMap<String, Vec<f32>>,
    ) -> Result<String, CoreError> {
        let exists = self
            .id_mapper
            .read()
            .uuid_to_id(&uuid)
            .map(|nid| !self.id_mapper.read().is_retired(nid))
            .unwrap_or(false)
            && self.document_store.read().get(&uuid).is_some();
        if exists {
            self.update_document(collection_name, &uuid.to_string(), fields, k_vecs)
        } else {
            let mut record = Record::new(fields);
            record.id = uuid;
            record.k_vecs = k_vecs;
            self.insert_document(collection_name, record)
        }
    }

    /// Durable delete: WAL → tombstone in DocumentStore → retire id → BM25
    /// removal. The retired id can never be returned by any head (INV-3).
    pub fn delete_document(&self, collection: &str, id_str: &str) -> Result<bool, CoreError> {
        let _gate = self.mutation_gate.lock();
        self.check_writable()?;
        let coll = self.get_collection(collection)?;
        let uuid = Uuid::parse_str(id_str)
            .map_err(|_| CoreError::InvalidArgument(format!("invalid document id '{id_str}'")))?;
        let numeric = match self.id_mapper.read().uuid_to_id(&uuid) {
            Some(n) if !self.id_mapper.read().is_retired(n) => n,
            _ => return Ok(false),
        };
        if self.document_store.read().get(&uuid).is_none() {
            return Ok(false);
        }

        let mut rec = WalRecord::new(0, RecordKind::DeleteDocument);
        rec.collection = collection.to_string();
        rec.doc_id = *uuid.as_bytes();
        rec.numeric_id = numeric;
        self.wal_append(rec)?;

        coll.retire_id(numeric);
        coll.bm25.remove(numeric);
        self.id_mapper.write().retire(&uuid);
        self.document_store.write().delete_nolog(&uuid)?;
        Ok(true)
    }

    // ================================================================
    // Transactions (commit-marker atomicity — docs/transactions.md)
    // ================================================================

    pub fn begin_transaction(&self, collection: &str) -> u64 {
        self.txn_manager.begin_transaction(collection)
    }

    pub fn record_transaction_operation(&self, id: u64, op: TxnOp) -> Result<(), CoreError> {
        self.txn_manager.record_operation(id, op)
    }

    pub fn rollback_transaction(&self, id: u64) -> Result<bool, CoreError> {
        self.check_writable()?;
        self.txn_manager.rollback_transaction(id)
    }

    /// Commit: BEGIN + ops + COMMIT are appended to the WAL (atomicity boundary
    /// = the COMMIT record), then applied. Crash before COMMIT → nothing applied.
    /// Crash during apply → replay re-applies the whole transaction idempotently.
    pub fn commit_transaction(&self, txn_id: u64) -> Result<bool, CoreError> {
        let _gate = self.mutation_gate.lock();
        self.check_writable()?;
        let txn: Transaction = match self.txn_manager.get_staged_transaction(txn_id) {
            Some(t) => t,
            None => return Ok(false),
        };
        let collection = txn.collection_name.clone();
        let coll = self.get_collection(&collection)?;

        // Pre-validate all ops so apply failures are ~impossible after commit.
        for op in &txn.operations {
            match op {
                TxnOp::Insert(r) => {
                    self.validate_record_for_collection(&coll, r)?;
                }
                TxnOp::Delete(_) => {}
            }
        }

        // 1) BEGIN
        let mut begin = WalRecord::new(0, RecordKind::BeginTxn);
        begin.txn_id = txn_id;
        begin.collection = collection.clone();
        self.wal_append(begin)?;

        // 2) Ops (ids are assigned now, deterministically, in order)
        for op in &txn.operations {
            match op {
                TxnOp::Insert(record) => {
                    let numeric = self.id_mapper.write().register(record.id);
                    // Membership tag is baked into the logged payload so replay
                    // rebuild classifies the record into this collection.
                    let mut tagged = record.clone();
                    let tag = format!("collection:{collection}");
                    if !tagged.tags.contains(&tag) {
                        tagged.tags.push(tag);
                    }
                    let mut rec = WalRecord::new(0, RecordKind::TxnOp);
                    rec.txn_id = txn_id;
                    rec.collection = collection.clone();
                    rec.doc_id = *record.id.as_bytes();
                    rec.numeric_id = numeric;
                    rec.payload = tagged.to_msgpack()?;
                    self.wal_append(rec)?;
                }
                TxnOp::Delete(uuid) => {
                    let numeric = self.id_mapper.read().uuid_to_id(uuid).unwrap_or(0);
                    let mut rec = WalRecord::new(0, RecordKind::TxnOp);
                    rec.txn_id = txn_id;
                    rec.collection = collection.clone();
                    rec.doc_id = *uuid.as_bytes();
                    rec.numeric_id = numeric; // 0 = "delete if present"
                    rec.payload = Vec::new();
                    self.wal_append(rec)?;
                }
            }
        }

        // 3) COMMIT — durability boundary
        let mut commit = WalRecord::new(0, RecordKind::CommitTxn);
        commit.txn_id = txn_id;
        self.wal_append(commit)?;

        // 4) Apply (idempotent primitives). Iterate ALL ops — never zip with the
        // id list (deletes have no staged id; zipping silently drops them).
        for op in txn.operations.iter() {
            match op {
                TxnOp::Insert(record) => {
                    let numeric = self.id_mapper.read().uuid_to_id(&record.id).unwrap_or(0);
                    let mut record = record.clone();
                    let tag = format!("collection:{collection}");
                    if !record.tags.contains(&tag) {
                        record.tags.push(tag);
                    }
                    self.apply_insert(&coll, record.id, numeric, record)?;
                }
                TxnOp::Delete(uuid) => {
                    let numeric_now = self.id_mapper.read().uuid_to_id(uuid);
                    if let Some(n) = numeric_now {
                        coll.retire_id(n);
                        coll.bm25.remove(n);
                        self.id_mapper.write().retire(uuid);
                        self.document_store.write().delete_nolog(uuid)?;
                    }
                }
            }
        }
        Ok(true)
    }

    // ================================================================
    // Search (unchanged semantics; retired ids filtered in Collection)
    // ================================================================

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

    /// Filtered retrieval (Phase 2 §11–13). Post-filters the staged pipeline's
    /// candidates and expands the candidate pool (bounded rounds) when the
    /// filter is selective. HARD GUARANTEE: every returned document satisfies
    /// the filter — post-filtering can never leak a non-matching document;
    /// when expansion is exhausted the result is simply smaller than top_k.
    pub fn attend_filtered(
        &self,
        collection_name: &str,
        heads: &[String],
        query: &[f32],
        top_k: usize,
        filter: Option<&attentiondb_query::filter::FilterExpr>,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        self.attend_filtered_with_deadline(collection_name, heads, query, top_k, filter, None)
    }

    /// Filtered retrieval with a query deadline (§19): the deadline is checked
    /// at every pipeline stage boundary; a miss surfaces as
    /// `CoreError::Timeout` — never a partial unfiltered result.
    pub fn attend_filtered_with_deadline(
        &self,
        collection_name: &str,
        heads: &[String],
        query: &[f32],
        top_k: usize,
        filter: Option<&attentiondb_query::filter::FilterExpr>,
        deadline: Option<std::time::Instant>,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        let coll = self.get_collection(collection_name)?;
        let Some(filter) = filter else {
            return coll.attend(heads, query, top_k);
        };
        if top_k == 0 {
            return Ok(vec![]);
        }
        let considered = std::sync::atomic::AtomicU64::new(0);
        let matched_ct = std::sync::atomic::AtomicU64::new(0);
        let base = coll.retrieval_config.read().clone();
        let mut multiplier = base.candidate_multiplier.max(1);
        let mut budget = base.candidate_budget.max(top_k);
        let mut survivors: Vec<crate::retrieval::RankedCandidate> = Vec::new();
        for round in 0..3 {
            let cfg = crate::collection::RetrievalConfig {
                candidate_multiplier: multiplier,
                candidate_budget: budget,
                ..base.clone()
            };
            // Fetch the FULL candidate pool (budget-sized): attend_detailed's
            // final top-K truncation must happen AFTER filtering, never before,
            // or expansion could never recover selective-filter results.
            let ranked =
                coll.attend_detailed(heads, query, budget, None, None, None, Some(&cfg), deadline)?;
            survivors = ranked
                .into_iter()
                .filter(|rc| {
                    considered.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let m = self.candidate_matches(collection_name, rc.id, filter);
                    if m {
                        matched_ct.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    m
                })
                .collect();
            if survivors.len() >= top_k || round == 2 {
                break;
            }
            multiplier = multiplier.saturating_mul(8).min(100);
            budget = budget.saturating_mul(8).min(100_000);
        }
        metrics::histogram!("attentiondb_filter_selectivity").record(
            matched_ct.load(std::sync::atomic::Ordering::Relaxed) as f64
                / considered.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64,
        );
        let pairs: Vec<(u64, f32)> = survivors
            .into_iter()
            .map(|r| (r.id, r.final_score))
            .collect();
        Ok(crate::retrieval::deterministic_top_k(pairs, top_k))
    }

    /// Hybrid retrieval under a metadata filter (§13): BOTH channels are
    /// filtered before fusion, so the §13 hard guarantee holds — a
    /// non-matching document can never leak through either channel.
    /// Rrf: filter each channel's budget-sized candidate list, then RRF-fuse.
    /// Fusion: the (filtered) sparse channel feeds the staged pipeline and the
    /// expansion loop guarantees enough filtered survivors when possible.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_hybrid_filtered_with_deadline(
        &self,
        collection_name: &str,
        heads: &[String],
        query: &[f32],
        text: &str,
        top_k: usize,
        filter: &attentiondb_query::filter::FilterExpr,
        deadline: Option<std::time::Instant>,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        if top_k == 0 {
            return Ok(vec![]);
        }
        let coll = self.get_collection(collection_name)?;
        let cfg = coll.retrieval_config.read().clone();
        let considered = std::sync::atomic::AtomicU64::new(0);
        let matched_ct = std::sync::atomic::AtomicU64::new(0);
        let matches = |id: u64| {
            considered.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let m = self.candidate_matches(collection_name, id, filter);
            if m {
                matched_ct.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            m
        };
        let budget = {
            let base = (cfg.candidate_budget.max(top_k)).saturating_mul(8);
            base.min(100_000)
        };
        // Sparse channel: fetch budget-sized, filter, keep order.
        let sparse_f: Vec<(u64, f32)> = coll
            .bm25_channel(text, budget)
            .into_iter()
            .filter(|(id, _)| matches(*id))
            .collect();
        // Dense channel: expansion rounds over the staged pipeline.
        let mut dense_f: Vec<(u64, f32)> = Vec::new();
        for _round in 0..3 {
            let pool =
                coll.attend_detailed(heads, query, budget, None, None, None, None, deadline)?;
            let pool_len = pool.len();
            dense_f = pool
                .into_iter()
                .map(|r| (r.id, r.final_score))
                .filter(|(id, _)| matches(*id))
                .collect();
            if dense_f.len() >= top_k || pool_len < budget {
                break;
            }
        }
        metrics::histogram!("attentiondb_filter_selectivity").record(
            matched_ct.load(std::sync::atomic::Ordering::Relaxed) as f64
                / considered.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64,
        );
        match cfg.hybrid_strategy {
            crate::collection::HybridStrategy::Rrf => Ok(crate::retrieval::rrf_fuse(
                &[&dense_f, &sparse_f],
                cfg.rrf_k,
                top_k,
            )),
            crate::collection::HybridStrategy::Fusion => {
                // Sparse channel participates INSIDE the staged pipeline; rerun
                // with expansion until enough filtered survivors.
                let mut fused: Vec<(u64, f32)> = Vec::new();
                for _round in 0..3 {
                    let pool = coll.attend_detailed(
                        heads,
                        query,
                        budget,
                        Some(&sparse_f),
                        None,
                        None,
                        None,
                        deadline,
                    )?;
                    let pool_len = pool.len();
                    fused = pool
                        .into_iter()
                        .map(|r| (r.id, r.final_score))
                        .filter(|(id, _)| matches(*id))
                        .collect();
                    if fused.len() >= top_k || pool_len < budget {
                        break;
                    }
                }
                Ok(crate::retrieval::deterministic_top_k(fused, top_k))
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 2B — trained gating model lifecycle (§14, §17, §18)
    // ------------------------------------------------------------------

    /// Activate the registry's active gating model for a collection
    /// (§17/§18). Validates compatibility (§15): head count must match the
    /// collection, card must pass §16 validation, and the card's head names
    /// (when present) must cover the collection's heads. On success the new
    /// weights take effect for subsequent queries WITHOUT restart; a query
    /// in flight sees old-or-new, never partial (Arc swap).
    pub fn activate_gating_model(
        &self,
        collection: &str,
        registry_root: &std::path::Path,
    ) -> Result<String, CoreError> {
        use attentiondb_learned::registry::{ModelRegistry, RegistryError};
        let reg = ModelRegistry::new(registry_root);
        let card = reg
            .load_active()
            .map_err(|e| match e {
                RegistryError::NoActiveModel => {
                    CoreError::InvalidArgument("no active gating model in registry".into())
                }
                other => CoreError::InvalidArgument(format!("registry error: {other}")),
            })?
            .ok_or_else(|| {
                CoreError::InvalidArgument("no active gating model in registry".into())
            })?;
        self.install_gating_card(collection, card)
    }

    /// Validate + install a card directly (used by tests and the bench).
    pub fn install_gating_card(
        &self,
        collection: &str,
        card: attentiondb_learned::gating_v2::ModelCard,
    ) -> Result<String, CoreError> {
        card.validate()
            .map_err(|e| CoreError::InvalidArgument(format!("model invalid: {e}")))?;
        let coll = self.get_collection(collection)?;
        let head_names = coll.list_heads();
        if card.num_heads != head_names.len() {
            return Err(CoreError::InvalidArgument(format!(
                "model trained for {} heads, collection has {} — refusing (§15)",
                card.num_heads,
                head_names.len()
            )));
        }
        if let Some(names) = &card.head_names {
            for h in &head_names {
                if !names.contains(h) {
                    return Err(CoreError::InvalidArgument(format!(
                        "model does not cover collection head '{h}' — refusing (§15)"
                    )));
                }
            }
        }
        let model_id = card.model_id.clone();
        *coll.gating_model.write() = Some(Arc::new(card));
        Ok(model_id)
    }

    /// Deactivate: queries fall back to the deterministic configured fusion
    /// (§14). No error when nothing is active.
    pub fn deactivate_gating_model(&self, collection: &str) -> Result<(), CoreError> {
        let coll = self.get_collection(collection)?;
        *coll.gating_model.write() = None;
        Ok(())
    }

    /// Inspect the currently installed model's metadata (§17 inspect).
    pub fn inspect_gating_model(&self, collection: &str) -> Result<Option<String>, CoreError> {
        let coll = self.get_collection(collection)?;
        let summary = coll.gating_model.read().as_ref().map(|c| {
            format!(
                "model_id={} arch={} input_dim={} hidden={} num_heads={} loss={} trained(seed={}, dataset_hash={:#x}, commit={})",
                c.model_id,
                c.arch,
                c.input_dim,
                c.hidden,
                c.num_heads,
                c.loss,
                c.training.seed,
                c.training.dataset_hash,
                c.training.code_commit
            )
        });
        Ok(summary)
    }

    /// Plan a retrieval without executing it; returns EXPLAIN text (§17).
    pub fn explain(
        &self,
        collection_name: &str,
        heads: &[String],
        top_k: usize,
        filter: Option<&attentiondb_query::filter::FilterExpr>,
        has_text: bool,
    ) -> Result<String, CoreError> {
        let coll = self.get_collection(collection_name)?;
        let cfg = coll.retrieval_config.read().clone();
        let facts = crate::planner::PlanFacts {
            existing_heads: coll.list_heads(),
            ef_search: coll.settings.read().ef_search,
        };
        let plan = crate::planner::plan_retrieval(
            collection_name,
            heads,
            top_k,
            filter,
            has_text,
            &facts,
            &cfg,
            &self.query_limits,
        )?;
        Ok(plan.explain_text())
    }

    /// Deadline-bounded unfiltered retrieval (§19).
    pub fn attend_with_deadline_ms(
        &self,
        collection_name: &str,
        heads: &[String],
        query: &[f32],
        top_k: usize,
        timeout_ms: u64,
    ) -> Result<Vec<(u64, f32)>, CoreError> {
        self.query_limits.validate_top_k(top_k)?;
        self.query_limits.validate_deadline_ms(timeout_ms)?;
        self.query_limits.validate_heads(heads)?;
        self.query_limits.validate_vector_dim(query.len())?;
        let coll = self.get_collection(collection_name)?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        let ranked =
            coll.attend_detailed(heads, query, top_k, None, None, None, None, Some(deadline))?;
        Ok(ranked.into_iter().map(|r| (r.id, r.final_score)).collect())
    }

    fn candidate_matches(
        &self,
        collection_name: &str,
        numeric: u64,
        filter: &attentiondb_query::filter::FilterExpr,
    ) -> bool {
        let _ = collection_name;
        let Some(uuid) = self.id_mapper.read().id_to_uuid(numeric).copied() else {
            return false;
        };
        match self.document_store.read().get(&uuid) {
            Some(rec) => filter.eval(&rec.fields),
            None => false,
        }
    }

    /// Filter-only query: deterministic scan of the collection's documents
    /// (numeric id ASC), no vector ranking. O(N) over the collection's records;
    /// acceptable for selective operational queries (cost documented — index-
    /// accelerated filters are future work, see docs/retrieval/filtering.md).
    pub fn scan_filtered(
        &self,
        collection_name: &str,
        filter: Option<&attentiondb_query::filter::FilterExpr>,
        limit: usize,
    ) -> Result<Vec<(String, u64)>, CoreError> {
        if limit == 0 {
            return Ok(vec![]);
        }
        self.get_collection(collection_name)?;
        let tag = format!("collection:{collection_name}");
        let mapper = self.id_mapper.read();
        let store = self.document_store.read();
        // Collect ALL matches, sort by numeric id ASC, THEN truncate: limiting
        // before sorting would return an arbitrary (HashMap-order) subset.
        let mut out: Vec<(String, u64)> = Vec::new();
        for rec in store.list_all_records() {
            if !rec.tags.contains(&tag) {
                continue;
            }
            if let Some(f) = filter {
                if !f.eval(&rec.fields) {
                    continue;
                }
            }
            if let Some(n) = mapper.uuid_to_id(&rec.id) {
                out.push((rec.id.to_string(), n));
            }
        }
        out.sort_by_key(|item| item.1);
        out.truncate(limit);
        Ok(out)
    }

    pub fn insert_vector(
        &self,
        collection: &str,
        head: &str,
        id: u64,
        vector: &[f32],
    ) -> Result<(), CoreError> {
        // Raw vector path (AQL/REPL tooling). Ephemeral-engine only in Phase 1:
        // durable writes must go through insert_document so WAL stays complete.
        if self.is_persistent() {
            return Err(CoreError::InvalidOperation(
                "insert_vector is not durable; use insert_document (with k_vecs) on a persistent engine".into(),
            ));
        }
        self.get_collection(collection)?
            .insert_vector(head, id, vector)
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

    // ================================================================
    // AQL (unchanged surface, durable engine underneath)
    // ================================================================

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
                // AQL ALTER without concrete settings payload is a no-op validation;
                // REST/gRPC ALTER carry full settings and persist them.
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

    // ================================================================
    // Durability operations
    // ================================================================

    /// Flush WAL buffers (GroupCommit semantics already flush; Async does not).
    pub fn flush_wal(&self) -> Result<(), CoreError> {
        if let Some(ref mut wal) = *self.wal.lock() {
            wal.flush().map_err(CoreError::Storage)?;
        }
        Ok(())
    }

    /// Checkpoint: everything up to the current WAL sequence becomes represented
    /// in durable SSTables + idmap snapshot + manifest; older WAL segments are
    /// then trimmed. Serialized against mutations via the mutation gate.
    pub fn checkpoint(&self) -> Result<CheckpointInfo, CoreError> {
        let _gate = self.mutation_gate.lock();
        self.checkpoint_locked()
    }

    /// Checkpoint body; caller must hold `mutation_gate`.
    fn checkpoint_locked(&self) -> Result<CheckpointInfo, CoreError> {
        let started = std::time::Instant::now();
        let db_dir = self
            .data_dir
            .clone()
            .ok_or_else(|| CoreError::InvalidOperation("engine is not persistent".into()))?;

        // 1) Make the WAL durable up to `seq`.
        let seq = {
            let mut wal_guard = self.wal.lock();
            let wal = wal_guard
                .as_mut()
                .ok_or_else(|| CoreError::InvalidOperation("engine is not persistent".into()))?;
            let seq = wal.next_seq() - 1; // last appended seq
            wal.fsync()?;
            seq
        };

        // 2) Flush memtable → atomic SSTables.
        self.document_store.write().flush()?;

        // 3) Idmap snapshot (exact allocator + retired ids — INV-6).
        let snap = self.id_mapper.read().snapshot();
        snap.save(&db_dir)?;

        // 4) Manifest: point at the new state (crash-safe install).
        let info = {
            let mut cat = self.catalog.lock();
            cat.checkpoint_seq = seq;
            cat.next_doc_numeric_id = self.id_mapper.read().next_id();
            let gen = cat.save(&db_dir)?;
            CheckpointInfo {
                checkpoint_seq: seq,
                manifest_generation: gen,
                documents: self.document_store.read().len(),
                collections: self.collections.read().len(),
                duration_ms: started.elapsed().as_secs_f64() * 1000.0,
            }
        };

        // 5) Rotate + trim segments fully covered by the checkpoint.
        {
            let mut wal_guard = self.wal.lock();
            let wal = wal_guard.as_mut().expect("persistent engine");
            wal.rotate()?;
            let removed = wal.retain_from(seq)?;
            if removed > 0 {
                tracing::info!(
                    segments_removed = removed,
                    through_seq = seq,
                    "WAL segments trimmed after checkpoint"
                );
            }
        }

        // 6) Bound manifest generations (keep the newest two for fallback).
        cleanup_old_manifests(&db_dir)?;

        tracing::info!(
            checkpoint_seq = info.checkpoint_seq,
            generation = info.manifest_generation,
            duration_ms = info.duration_ms,
            "checkpoint complete"
        );
        Ok(info)
    }

    /// Graceful shutdown: stop accepting writes → flush → checkpoint → CLOSED.
    /// Returns an error if durability steps failed (never reports clean shutdown
    /// falsely — §22 of the Phase 1 spec).
    pub fn close(&self) -> Result<(), CoreError> {
        self.accepting_writes.store(false, Ordering::SeqCst);
        if !self.is_persistent() {
            *self.state.write() = EngineState::Closed;
            return Ok(());
        }
        self.checkpoint()?;
        {
            let mut wal_guard = self.wal.lock();
            if let Some(wal) = wal_guard.as_mut() {
                wal.fsync()?;
            }
        }
        *self.state.write() = EngineState::Closed;
        tracing::info!("engine closed cleanly (checkpointed)");
        Ok(())
    }

    /// Backup: checkpoint, then copy the authoritative state set into `dest`.
    /// See core/src/backup.rs — this method only performs the quiesce + checkpoint
    /// and delegates the copy.
    pub fn backup_to(&self, dest: &Path) -> Result<CheckpointInfo, CoreError> {
        // Hold the mutation gate across checkpoint + copy so the snapshot is
        // internally consistent (never a torn copy of a live directory).
        let _gate = self.mutation_gate.lock();
        let info = self.checkpoint_locked()?;
        crate::backup::copy_database_dir(
            self.data_dir
                .as_deref()
                .ok_or_else(|| CoreError::InvalidOperation("engine is not persistent".into()))?,
            dest,
        )?;
        Ok(info)
    }

    // ================================================================
    // Internal helpers
    // ================================================================

    fn check_writable(&self) -> Result<(), CoreError> {
        if !self.accepting_writes.load(Ordering::SeqCst) {
            return Err(CoreError::Unavailable(
                "engine is shutting down: writes are rejected".into(),
            ));
        }
        if !self.is_ready() {
            return Err(CoreError::Unavailable(format!(
                "engine not ready (state {:?})",
                *self.state.read()
            )));
        }
        Ok(())
    }

    fn next_seq_hint(&self, _cat: &Catalog) -> u64 {
        self.wal.lock().as_ref().map(|w| w.next_seq()).unwrap_or(0)
    }

    fn wal_append(&self, record: WalRecord) -> Result<u64, CoreError> {
        let mut wal_guard = self.wal.lock();
        let wal = match wal_guard.as_mut() {
            Some(w) => w,
            // Ephemeral engine: mutations apply to RAM only (documented mode).
            None => return Ok(0),
        };
        let seq = wal.append(record).map_err(CoreError::Storage)?;
        Ok(seq)
    }

    fn persist_catalog(&self) -> Result<(), CoreError> {
        let db_dir = match &self.data_dir {
            Some(d) => d.clone(),
            // Ephemeral engine: catalog lives in RAM only.
            None => return Ok(()),
        };
        let cat = self.catalog.lock().clone();
        let gen = cat.save(&db_dir)?;
        self.manifest_generation.store(gen, Ordering::SeqCst);
        Ok(())
    }

    fn validate_record_for_collection(
        &self,
        collection: &Arc<Collection>,
        record: &Record,
    ) -> Result<(), CoreError> {
        if record.k_vecs.len() > crate::constants::MAX_HEADS_PER_DOCUMENT {
            return Err(CoreError::ResourceExhausted(format!(
                "document has {} heads; max {}",
                record.k_vecs.len(),
                crate::constants::MAX_HEADS_PER_DOCUMENT
            )));
        }
        for (head, vec) in &record.k_vecs {
            if vec.len() != collection.dim {
                return Err(CoreError::InvalidArgument(format!(
                    "vector for head '{head}' has dimension {}, collection dimension is {}",
                    vec.len(),
                    collection.dim
                )));
            }
            if vec.iter().any(|f| !f.is_finite()) {
                return Err(CoreError::InvalidArgument(format!(
                    "vector for head '{head}' contains non-finite values"
                )));
            }
        }
        Ok(())
    }

    /// Apply an insert to all subsystems (idempotent — safe to re-apply in replay).
    fn apply_insert(
        &self,
        collection: &Arc<Collection>,
        uuid: Uuid,
        numeric_id: u64,
        record: Record,
    ) -> Result<(), CoreError> {
        self.document_store.write().insert_nolog(record.clone())?;
        for (head, vec) in &record.k_vecs {
            collection.insert_vector(head, numeric_id, vec)?;
        }
        let text: String = record
            .fields
            .values()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        collection.bm25.insert(numeric_id, &text);
        let _ = uuid;
        Ok(())
    }

    fn materialize_collection(&self, meta: &CollectionMeta) -> Result<Arc<Collection>, CoreError> {
        let coll = Arc::new(Collection::new(&meta.name, meta.dim));
        let settings: attentiondb_hnsw::CollectionSettings = if meta.settings_json.is_empty() {
            Default::default()
        } else {
            serde_json::from_str(&meta.settings_json).map_err(|e| {
                CoreError::Corruption(format!("bad settings for {}: {e}", meta.name))
            })?
        };
        *coll.settings.write() = settings;
        for h in &meta.heads {
            coll.add_default_head(&h.name)?;
        }
        Ok(coll)
    }

    /// Apply one replayed WAL record. Indexes are NOT touched here — they are
    /// rebuilt once, deterministically, from final record state after replay
    /// (idempotent by construction).
    fn apply_replay_record(
        &self,
        record: &WalRecord,
        txn_buf: &mut HashMap<u64, TxnBuffer>,
        _total: usize,
    ) -> Result<(), CoreError> {
        let kind = record.kind().ok_or_else(|| {
            CoreError::Corruption(format!(
                "unknown WAL record kind {} at seq {}",
                record.kind, record.seq
            ))
        })?;
        match kind {
            RecordKind::BeginTxn => {
                txn_buf.insert(record.txn_id, TxnBuffer::default());
            }
            RecordKind::TxnOp => {
                let buf = txn_buf.get_mut(&record.txn_id).ok_or_else(|| {
                    CoreError::Corruption(format!(
                        "TxnOp for unknown/unstarted txn {} at seq {}",
                        record.txn_id, record.seq
                    ))
                })?;
                if buf.ops.len() >= MAX_TXN_OPS {
                    return Err(CoreError::Corruption(format!(
                        "transaction {} exceeds MAX_TXN_OPS ({})",
                        record.txn_id, MAX_TXN_OPS
                    )));
                }
                // TxnOp payload is an insert (record msgpack) or a delete (empty).
                let uuid = Uuid::from_bytes(record.doc_id);
                if record.payload.is_empty() {
                    buf.ops.push(ReplayOp::DeleteDocument {
                        collection: record.collection.clone(),
                        uuid,
                        numeric_id: record.numeric_id,
                    });
                } else {
                    let rec = Record::from_msgpack(&record.payload)
                        .map_err(|e| CoreError::Corruption(format!("txn payload: {e}")))?;
                    buf.ops.push(ReplayOp::InsertDocument {
                        collection: record.collection.clone(),
                        uuid,
                        numeric_id: record.numeric_id,
                        record: rec,
                    });
                }
            }
            RecordKind::CommitTxn => {
                let buf = txn_buf.remove(&record.txn_id).ok_or_else(|| {
                    CoreError::Corruption(format!(
                        "COMMIT for unknown txn {} at seq {}",
                        record.txn_id, record.seq
                    ))
                })?;
                for op in buf.ops {
                    self.apply_replay_op(op)?;
                }
            }
            RecordKind::AbortTxn => {
                txn_buf.remove(&record.txn_id);
            }
            RecordKind::CreateCollection
            | RecordKind::DropCollection
            | RecordKind::AlterCollection
            | RecordKind::InsertDocument
            | RecordKind::UpsertDocument
            | RecordKind::UpdateDocument
            | RecordKind::DeleteDocument
            | RecordKind::CheckpointMarker => {
                let op = self.decode_op(record)?;
                self.apply_replay_op(op)?;
            }
        }
        Ok(())
    }

    fn decode_op(&self, record: &WalRecord) -> Result<ReplayOp, CoreError> {
        let kind = record.kind().expect("checked earlier");
        let uuid = Uuid::from_bytes(record.doc_id);
        Ok(match kind {
            RecordKind::CreateCollection => {
                let v: serde_json::Value = serde_json::from_slice(&record.payload)
                    .map_err(|e| CoreError::Corruption(format!("create payload: {e}")))?;
                let heads: Vec<HeadMeta> = serde_json::from_value(
                    v.get("heads").cloned().unwrap_or(serde_json::json!([])),
                )
                .map_err(|e| CoreError::Corruption(format!("create heads: {e}")))?;
                ReplayOp::CreateCollection {
                    id: record.numeric_id as u32,
                    name: record.collection.clone(),
                    dim: v.get("dim").and_then(|d| d.as_u64()).unwrap_or(64) as usize,
                    heads,
                    settings_json: v
                        .get("settings")
                        .and_then(|s| s.as_str())
                        .unwrap_or("")
                        .to_string(),
                    schema_fields: Vec::new(),
                }
            }
            RecordKind::DropCollection => ReplayOp::DropCollection {
                name: record.collection.clone(),
            },
            RecordKind::AlterCollection => ReplayOp::AlterCollection {
                name: record.collection.clone(),
                settings_json: String::from_utf8_lossy(&record.payload).to_string(),
            },
            RecordKind::InsertDocument | RecordKind::UpsertDocument => {
                let rec = Record::from_msgpack(&record.payload)
                    .map_err(|e| CoreError::Corruption(format!("insert payload: {e}")))?;
                ReplayOp::InsertDocument {
                    collection: record.collection.clone(),
                    uuid,
                    numeric_id: record.numeric_id,
                    record: rec,
                }
            }
            RecordKind::UpdateDocument => {
                if record.payload.len() < 8 {
                    return Err(CoreError::Corruption(format!(
                        "update record at seq {} has truncated payload",
                        record.seq
                    )));
                }
                let mut old_bytes = [0u8; 8];
                old_bytes.copy_from_slice(&record.payload[..8]);
                let rec = Record::from_msgpack(&record.payload[8..])
                    .map_err(|e| CoreError::Corruption(format!("update record: {e}")))?;
                ReplayOp::UpdateDocument {
                    collection: record.collection.clone(),
                    uuid,
                    old_numeric_id: u64::from_be_bytes(old_bytes),
                    new_numeric_id: record.numeric_id,
                    record: rec,
                }
            }
            RecordKind::DeleteDocument => ReplayOp::DeleteDocument {
                collection: record.collection.clone(),
                uuid,
                numeric_id: record.numeric_id,
            },
            RecordKind::CheckpointMarker => ReplayOp::AlterCollection {
                name: String::new(),
                settings_json: String::new(),
            },
            RecordKind::BeginTxn
            | RecordKind::TxnOp
            | RecordKind::CommitTxn
            | RecordKind::AbortTxn => {
                unreachable!("handled in apply_replay_record")
            }
        })
    }

    fn apply_replay_op(&self, op: ReplayOp) -> Result<(), CoreError> {
        match op {
            ReplayOp::CreateCollection {
                id: _,
                name,
                dim,
                heads,
                settings_json,
                schema_fields: _,
            } => {
                let mut cols = self.collections.write();
                if !cols.contains_key(&name) {
                    let meta = CollectionMeta {
                        id: 0,
                        name: name.clone(),
                        dim,
                        heads,
                        settings_json,
                        schema_fields: Vec::new(),
                        created_seq: 0,
                        dropped: false,
                    };
                    let coll = self.materialize_collection(&meta)?;
                    cols.insert(name.clone(), coll);
                }
                // Also converge catalog (WAL may be ahead of last manifest save).
                let mut cat = self.catalog.lock();
                if cat.live_collection(&name).is_none() {
                    if let Some(existing) = cat.collections.iter_mut().find(|c| c.name == name) {
                        existing.dropped = false;
                    } else {
                        let id = cat.next_collection_id;
                        cat.next_collection_id += 1;
                        cat.collections.push(CollectionMeta {
                            id,
                            name,
                            dim,
                            heads: Vec::new(),
                            settings_json: String::new(),
                            schema_fields: Vec::new(),
                            created_seq: 0,
                            dropped: false,
                        });
                    }
                }
            }
            ReplayOp::DropCollection { name } => {
                self.collections.write().remove(&name);
                let mut cat = self.catalog.lock();
                if let Some(meta) = cat.collections.iter_mut().find(|c| c.name == name) {
                    meta.dropped = true;
                }
            }
            ReplayOp::AlterCollection {
                name,
                settings_json,
            } => {
                if let Some(coll) = self.collections.read().get(&name).cloned() {
                    if let Ok(s) =
                        serde_json::from_str::<attentiondb_hnsw::CollectionSettings>(&settings_json)
                    {
                        *coll.settings.write() = s;
                    }
                }
                let mut cat = self.catalog.lock();
                if let Some(meta) = cat.collections.iter_mut().find(|c| c.name == name) {
                    meta.settings_json = settings_json;
                }
            }
            ReplayOp::InsertDocument {
                collection: _,
                uuid,
                numeric_id,
                record,
            } => {
                // Only mapper + storage: indexes are rebuilt once after replay
                // from final record state (idempotent by construction).
                self.id_mapper.write().restore(uuid, numeric_id);
                self.document_store.write().insert_nolog(record)?;
            }
            ReplayOp::UpdateDocument {
                collection: _,
                uuid,
                old_numeric_id,
                new_numeric_id,
                record,
            } => {
                self.id_mapper.write().restore(uuid, new_numeric_id);
                self.id_mapper.write().retire_mark(old_numeric_id);
                self.document_store.write().insert_nolog(record.clone())?;
                // vectors: rebuilt post-replay; retired id filtered/purged there
            }
            ReplayOp::DeleteDocument {
                collection: _,
                uuid,
                numeric_id,
            } => {
                self.id_mapper.write().retire(&uuid);
                if numeric_id > 0 {
                    self.id_mapper.write().retire_mark(numeric_id);
                }
                self.document_store.write().delete_nolog(&uuid)?;
            }
        }
        Ok(())
    }

    /// Rebuild all collection indexes (HNSW + BM25) deterministically from the
    /// authoritative records in DocumentStore. Phase 1 index strategy
    /// (docs/indexes.md): deterministic rebuild instead of graph persistence,
    /// because hnsw_rs cannot delete and its load path is unvalidated.
    fn rebuild_all_indexes(&self) -> Result<(), CoreError> {
        let records = self.document_store.read().list_all_records();
        // Group records by collection membership tag (tags are part of the
        // durable record, so membership survives restart — INV-7/8).
        let mut by_collection: HashMap<String, Vec<(u64, Record)>> = HashMap::new();
        {
            let mapper = self.id_mapper.read();
            for rec in &records {
                let numeric = match mapper.uuid_to_id(&rec.id) {
                    Some(n) => n,
                    None => continue, // orphan record; reported by the checker
                };
                for tag in &rec.tags {
                    if let Some(coll_name) = tag.strip_prefix("collection:") {
                        by_collection
                            .entry(coll_name.to_string())
                            .or_default()
                            .push((numeric, rec.clone()));
                    }
                }
            }
        }

        let cols = self.collections.read();
        for (name, recs) in by_collection {
            if let Some(coll) = cols.get(&name) {
                coll.rebuild_from_records(&recs)?;
            }
            // Records tagged for a dropped/unknown collection are ignored (the
            // catalog is authoritative); they remain in storage.
        }
        Ok(())
    }
}

impl Default for AttentionEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn default_max_segment_bytes() -> u64 {
    std::env::var("ATTENTIONDB_WAL_SEGMENT_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64 * 1024 * 1024)
}

fn cleanup_old_manifests(db_dir: &Path) -> Result<(), CoreError> {
    let dir = Catalog::manifest_dir(db_dir);
    let mut gens: Vec<u64> = Vec::new();
    for e in std::fs::read_dir(&dir)? {
        let p = e?.path();
        if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
            if let Some(num) = stem.strip_prefix("manifest-") {
                if let Ok(g) = num.parse::<u64>() {
                    gens.push(g);
                }
            }
        }
    }
    if gens.len() <= 2 {
        return Ok(());
    }
    gens.sort_unstable();
    let keep_from = gens.len() - 2;
    for g in &gens[..keep_from] {
        let _ = std::fs::remove_file(dir.join(format!("manifest-{g:09}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn rec(i: usize, dim: usize) -> Record {
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(i));
        fields.insert("body".to_string(), serde_json::json!(format!("body {i}")));
        let mut r = Record::new(fields);
        r.k_vecs.insert("default".to_string(), vec![i as f32; dim]);
        r
    }

    /// One-hot vector: unique direction per index → deterministic top-1 ranking.
    fn one_hot(i: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[i % dim] = 1.0;
        v
    }

    fn one_hot_rec(i: usize, dim: usize) -> Record {
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(i));
        fields.insert("body".to_string(), serde_json::json!(format!("body {i}")));
        let mut r = Record::new(fields);
        r.k_vecs.insert("default".to_string(), one_hot(i, dim));
        r
    }

    // ---------- IdMapper ----------

    #[test]
    fn test_id_mapper_no_duplicates() {
        let mut m = IdMapper::new();
        let u1 = Uuid::new_v4();
        let u2 = Uuid::new_v4();
        assert_ne!(m.register(u1), m.register(u2));
        assert_eq!(m.register(u1), m.register(u1));
    }

    #[test]
    fn test_id_mapper_remap_retires_old_id() {
        let mut m = IdMapper::new();
        let u = Uuid::new_v4();
        let id1 = m.register(u);
        let (id2, old) = m.remap(u);
        assert_eq!(old, Some(id1));
        assert_ne!(id1, id2);
        assert!(m.is_retired(id1));
        assert_eq!(m.uuid_to_id(&u), Some(id2));
        // Retired ids are never reused.
        let u3 = Uuid::new_v4();
        let id3 = m.register(u3);
        assert!(id3 > id2 && id3 > id1);
    }

    #[test]
    fn test_id_mapper_snapshot_roundtrip() {
        let mut m = IdMapper::new();
        let u1 = Uuid::new_v4();
        m.register(u1);
        let u2 = Uuid::new_v4();
        let id2 = m.register(u2);
        m.retire(&u2);
        let snap = m.snapshot();
        let mut m2 = IdMapper::new();
        m2.restore_snapshot(&snap);
        assert_eq!(m2.uuid_to_id(&u1), m.uuid_to_id(&u1));
        assert!(m2.is_retired(id2));
        // new registrations continue after the max id (INV-6)
        let fresh = m2.register(Uuid::new_v4());
        assert!(fresh > id2);
    }

    // ---------- Ephemeral engine (back-compat) ----------

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
        let id_str = e.insert_document("c", rec(1, 4)).unwrap();
        let uuid = Uuid::parse_str(&id_str).unwrap();
        assert_eq!(e.id_mapper.read().uuid_to_id(&uuid), Some(1));
    }

    // ---------- Durable engine: create → insert → restart → search ----------

    #[test]
    fn test_durable_basic_lifecycle() {
        let dir = tmp();
        let dim = 16;
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            e.create_collection("papers", dim, &["default"]).unwrap();
            for i in 0..10 {
                e.insert_document("papers", one_hot_rec(i, dim)).unwrap();
            }
            // Deterministic ranking: one-hot query hits exactly its document.
            let r = e
                .attend("papers", &["default".into()], &one_hot(9, dim), 3)
                .unwrap();
            let all = e.document_store.read().list_all_records();
            let target_uuid = all
                .iter()
                .find(|rec| rec.fields.get("idx").and_then(|v| v.as_i64()) == Some(9))
                .unwrap()
                .id;
            let expected_id = e.id_mapper.read().uuid_to_id(&target_uuid).unwrap();
            assert_eq!(r[0].0, expected_id);
            e.close().unwrap();
        }
        // Restart: collections + documents + search must survive (INV-7).
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            assert_eq!(e.list_collections(), vec!["papers".to_string()]);
            assert_eq!(e.stats().collection_count, 1);
            let coll = e.get_collection("papers").unwrap();
            assert_eq!(coll.head_count(), 1);
            assert_eq!(coll.total_vectors(), 10);
            let r = e
                .attend("papers", &["default".into()], &one_hot(9, dim), 3)
                .unwrap();
            assert!(!r.is_empty());
            // The same query returns the same document after restart (T2).
            let all = e.document_store.read().list_all_records();
            let target_uuid = all
                .iter()
                .find(|rec| rec.fields.get("idx").and_then(|v| v.as_i64()) == Some(9))
                .unwrap()
                .id;
            let expected_id = e.id_mapper.read().uuid_to_id(&target_uuid).unwrap();
            assert_eq!(r[0].0, expected_id);
            assert_eq!(e.document_store.read().len(), 10);
            e.close().unwrap();
        }
    }

    #[test]
    fn test_durable_delete_and_update_survive_restart() {
        let dir = tmp();
        let mut uuids = Vec::new();
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            e.create_collection("c", 4, &["default"]).unwrap();
            for i in 0..6 {
                uuids.push(e.insert_document("c", rec(i, 4)).unwrap());
            }
            // delete doc 0
            assert!(e.delete_document("c", &uuids[0]).unwrap());
            // update doc 1: vector becomes [-1,-1,-1,-1]
            let mut fields = HashMap::new();
            fields.insert("idx".to_string(), serde_json::json!(1));
            fields.insert("body".to_string(), serde_json::json!("UPDATED"));
            let mut kv = HashMap::new();
            kv.insert("default".to_string(), vec![-1.0; 4]);
            e.update_document("c", &uuids[1], fields, kv).unwrap();
            e.close().unwrap();
        }
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            // INV-3: the deleted document is gone from every subsystem —
            // its uuid no longer maps to a live id and it is not searchable.
            let deleted_uuid = Uuid::parse_str(&uuids[0]).unwrap();
            assert!(e.id_mapper.read().uuid_to_id(&deleted_uuid).is_none());
            assert!(e.document_store.read().get(&deleted_uuid).is_none());
            let r = e.attend("c", &["default".into()], &[-1.0; 4], 2).unwrap();
            assert!(!r.is_empty());
            // updated doc resolves to UPDATED body (INV-4)
            let top = r[0].0;
            let fields = e.get_document_fields(top);
            assert_eq!(fields.get("body").map(String::as_str), Some("UPDATED"));
            e.close().unwrap();
        }
    }

    #[test]
    fn test_upsert_idempotent_by_uuid() {
        let dir = tmp();
        let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
        e.create_collection("c", 4, &["default"]).unwrap();
        let uuid = Uuid::new_v4();
        let mut fields = HashMap::new();
        fields.insert("v".to_string(), serde_json::json!("a"));
        let mut kv = HashMap::new();
        kv.insert("default".to_string(), vec![1.0; 4]);
        e.upsert_document("c", uuid, fields.clone(), kv.clone())
            .unwrap();
        e.upsert_document("c", uuid, fields, kv).unwrap();
        // exactly one logical document (INV-5): one live record, one live id.
        assert_eq!(e.document_store.read().len(), 1);
        let mapper = e.id_mapper.read();
        assert_eq!(mapper.snapshot().mappings.len(), 1);
        // search returns exactly one result (old representation filtered)
        let r = e.attend("c", &["default".into()], &[1.0; 4], 10).unwrap();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn test_wal_replay_after_unclean_stop() {
        let dir = tmp();
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            e.create_collection("c", 4, &["default"]).unwrap();
            for i in 0..5 {
                e.insert_document("c", rec(i, 4)).unwrap();
            }
            // NO close(): simulate crash (engine state on disk = WAL only)
            std::mem::forget(e);
        }
        let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
        assert_eq!(e.list_collections(), vec!["c".to_string()]);
        assert_eq!(e.document_store.read().len(), 5);
    }

    #[test]
    fn test_checkpoint_trims_wal_and_state_survives() {
        let dir = tmp();
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            e.create_collection("c", 4, &["default"]).unwrap();
            for i in 0..20 {
                e.insert_document("c", rec(i, 4)).unwrap();
            }
            let info = e.checkpoint().unwrap();
            assert!(info.checkpoint_seq >= 21);
            for i in 20..30 {
                e.insert_document("c", rec(i, 4)).unwrap();
            }
            e.close().unwrap();
        }
        let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
        assert_eq!(e.document_store.read().len(), 30);
        let coll = e.get_collection("c").unwrap();
        assert_eq!(coll.total_vectors(), 30);
    }

    #[test]
    fn test_corrupt_wal_middle_is_fatal() {
        let dir = tmp();
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            e.create_collection("c", 4, &["default"]).unwrap();
            for i in 0..5 {
                e.insert_document("c", rec(i, 4)).unwrap();
            }
            // Crash WITHOUT close/checkpoint: all records live only in the WAL.
            std::mem::forget(e);
        }
        // Corrupt a byte in the middle of the (non-empty) WAL segment.
        let wal_dir = dir.path().join("WAL");
        let seg = std::fs::read_dir(&wal_dir)
            .unwrap()
            .filter_map(|x| x.ok())
            .map(|x| x.path())
            .find(|p| std::fs::metadata(p).map(|m| m.len() > 32).unwrap_or(false))
            .expect("non-empty WAL segment");
        let mut bytes = std::fs::read(&seg).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&seg, &bytes).unwrap();
        let result = AttentionEngine::open_dir(dir.path(), Durability::Sync);
        assert!(result.is_err(), "mid-log corruption must be fatal");
    }

    #[test]
    fn test_uncommitted_txn_never_applies() {
        let dir = tmp();
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            e.create_collection("c", 4, &["default"]).unwrap();
            let t = e.begin_transaction("c");
            for i in 0..4 {
                e.record_transaction_operation(t, TxnOp::Insert(rec(i, 4)))
                    .unwrap();
            }
            // No commit — crash (forget) drops the staging area; nothing applied.
            std::mem::forget(e);
        }
        let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
        assert_eq!(
            e.document_store.read().len(),
            0,
            "uncommitted txn must not apply"
        );
    }

    #[test]
    fn test_committed_txn_applies_atomically() {
        let dir = tmp();
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            e.create_collection("c", 4, &["default"]).unwrap();
            // pre-existing doc we will delete inside the txn
            let victim = e.insert_document("c", rec(100, 4)).unwrap();

            let t = e.begin_transaction("c");
            for i in 0..3 {
                e.record_transaction_operation(t, TxnOp::Insert(rec(i, 4)))
                    .unwrap();
            }
            e.record_transaction_operation(t, TxnOp::Delete(Uuid::parse_str(&victim).unwrap()))
                .unwrap();
            assert!(e.commit_transaction(t).unwrap());
            assert_eq!(e.document_store.read().len(), 3);
            std::mem::forget(e); // crash right after commit+apply
        }
        // Replay must converge to the same state (INV-9/10).
        let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
        assert_eq!(e.document_store.read().len(), 3);
    }

    #[test]
    fn test_backup_restore_roundtrip() {
        let dir = tmp();
        let backup = tmp();
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            e.create_collection("c", 4, &["default"]).unwrap();
            for i in 0..8 {
                e.insert_document("c", rec(i, 4)).unwrap();
            }
            let victim = e.insert_document("c", rec(99, 4)).unwrap();
            e.delete_document("c", &victim).unwrap();
            e.backup_to(backup.path()).unwrap();
            e.close().unwrap();
        }
        let restored = tmp();
        crate::backup::restore_backup(backup.path(), restored.path()).unwrap();
        let e = AttentionEngine::open_dir(restored.path(), Durability::Sync).unwrap();
        assert_eq!(e.list_collections(), vec!["c".to_string()]);
        assert_eq!(e.document_store.read().len(), 8);
        let r = e.attend("c", &["default".into()], &[7.0; 4], 1).unwrap();
        assert!(!r.is_empty());
    }

    #[test]
    fn test_open_refuses_newer_format_version() {
        let dir = tmp();
        {
            let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
            e.create_collection("c", 4, &["default"]).unwrap();
            e.close().unwrap();
        }
        // Tamper: claim a future database format version in the manifest.
        let cat = attentiondb_storage::Catalog::load(dir.path()).unwrap().0;
        let mut tampered = cat.clone();
        tampered.database_format_version = 999;
        tampered.save(dir.path()).unwrap();
        let err = match AttentionEngine::open_dir(dir.path(), Durability::Sync) {
            Err(e) => e,
            Ok(_) => panic!("expected RecoveryFailed for newer format version"),
        };
        assert!(matches!(err, CoreError::RecoveryFailed(_)));
    }

    #[test]
    fn test_ephemeral_engine_rejects_after_close() {
        let dir = tmp();
        let e = AttentionEngine::open_dir(dir.path(), Durability::Sync).unwrap();
        e.create_collection("c", 4, &["default"]).unwrap();
        e.close().unwrap();
        assert!(matches!(
            e.insert_document("c", rec(1, 4)),
            Err(CoreError::Unavailable(_))
        ));
    }
}
