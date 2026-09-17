use parking_lot::RwLock;
use std::collections::{BinaryHeap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use uuid::Uuid;

use crate::error::StorageError;
use crate::record::Record;
use crate::sstable::{SSTableReader, SSTableWriter};

/// Heap entry ordered by access_time (oldest first for LRU eviction).
#[derive(Debug, Clone)]
pub struct LruHeapEntry {
    pub access_time: u64,
    pub id: Uuid,
}

impl PartialEq for LruHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.access_time == other.access_time
    }
}
impl Eq for LruHeapEntry {}
impl PartialOrd for LruHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for LruHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.access_time.cmp(&self.access_time)
    }
}

#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: AtomicUsize,
    pub misses: AtomicUsize,
    pub evictions: AtomicUsize,
    pub size_bytes: AtomicUsize,
}

impl Clone for CacheStats {
    fn clone(&self) -> Self {
        Self {
            hits: AtomicUsize::new(self.hits.load(Ordering::Relaxed)),
            misses: AtomicUsize::new(self.misses.load(Ordering::Relaxed)),
            evictions: AtomicUsize::new(self.evictions.load(Ordering::Relaxed)),
            size_bytes: AtomicUsize::new(self.size_bytes.load(Ordering::Relaxed)),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheStatsSnapshot {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub size_bytes: usize,
    pub entries: usize,
}

/// LRU block cache using BinaryHeap<LruHeapEntry>.
pub struct BlockCache {
    pub cache: HashMap<Uuid, Record>,
    pub lru: BinaryHeap<LruHeapEntry>,
    pub capacity: usize,
    pub memory_budget: usize,
    pub memory_used: usize,
    pub stats: CacheStats,
    access_counter: u64,
}

impl BlockCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            lru: BinaryHeap::with_capacity(capacity),
            capacity,
            memory_budget: 50 * 1024 * 1024,
            memory_used: 0,
            stats: CacheStats::default(),
            access_counter: 0,
        }
    }

    pub fn get(&mut self, id: &Uuid) -> Option<Record> {
        if let Some(record) = self.cache.get(id) {
            self.access_counter += 1;
            self.lru.push(LruHeapEntry {
                access_time: self.access_counter,
                id: *id,
            });
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Some(record.clone());
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub fn insert(&mut self, id: Uuid, record: Record) {
        self.access_counter += 1;
        let size = estimate_record_size(&record);
        // Lazy deletion: pop stale entries
        while let Some(top) = self.lru.peek() {
            if self.cache.contains_key(&top.id) {
                break;
            }
            self.lru.pop();
        }
        // Evict until under capacity and budget
        while self.cache.len() >= self.capacity || (self.memory_used + size > self.memory_budget) {
            if let Some(entry) = self.lru.pop() {
                if self.cache.remove(&entry.id).is_some() {
                    self.memory_used = self.memory_used.saturating_sub(256);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }
        self.memory_used += size;
        self.cache.insert(id, record);
        self.lru.push(LruHeapEntry {
            access_time: self.access_counter,
            id,
        });
    }

    pub fn remove(&mut self, id: &Uuid) {
        if let Some(record) = self.cache.remove(id) {
            let size = estimate_record_size(&record);
            self.memory_used = self.memory_used.saturating_sub(size);
        }
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru.clear();
        self.memory_used = 0;
        self.stats.size_bytes.store(0, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            size_bytes: self.memory_used,
            entries: self.cache.len(),
        }
    }
}

fn estimate_record_size(r: &Record) -> usize {
    std::mem::size_of::<Record>() + r.fields.len() * 64 + r.k_vecs.len() * 128 + r.tags.len() * 32
}

fn tombstone_record(id: Uuid) -> Record {
    let mut t = Record::new(HashMap::new());
    t.id = id;
    t.tags.push("__TOMBSTONE__".into());
    t
}

pub struct DocumentStore {
    memtable: HashMap<Uuid, Record>,
    flushed_records: HashMap<Uuid, Record>,
    storage_dir: Option<PathBuf>,
    memtable_threshold: usize,
    sstables: Vec<SSTableReader>,
    pub block_cache: RwLock<BlockCache>,
}

impl Default for DocumentStore {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            memtable: HashMap::new(),
            flushed_records: HashMap::new(),
            storage_dir: None,
            memtable_threshold: 1000,
            sstables: Vec::new(),
            block_cache: RwLock::new(BlockCache::new(50_000)),
        }
    }

    /// Compatibility shim: the per-store WAL was removed in Phase 1 — the engine's
    /// authoritative WAL is the only mutation log (docs/wal.md). Durability for
    /// standalone use is provided by `flush_memtable`.
    pub fn with_wal(self, _wal: crate::wal::Wal) -> Self {
        self
    }
    pub fn with_storage_dir(mut self, dir: PathBuf) -> Result<Self, StorageError> {
        std::fs::create_dir_all(&dir)?;
        self.storage_dir = Some(dir);
        Ok(self)
    }
    pub fn with_memtable_threshold(mut self, t: usize) -> Self {
        self.memtable_threshold = t;
        self
    }

    pub fn open(dir: PathBuf) -> Result<Self, StorageError> {
        Self::open_inner(dir)
    }

    /// Alias of `open`. Since Phase 1 there is no per-store WAL: the engine's
    /// authoritative WAL is the only mutation log (docs/wal.md).
    pub fn open_without_wal(dir: PathBuf) -> Result<Self, StorageError> {
        Self::open_inner(dir)
    }

    fn open_inner(dir: PathBuf) -> Result<Self, StorageError> {
        std::fs::create_dir_all(&dir)?;
        let mut sstables = Vec::new();
        // Timestamp-aware merge: for each key, the entry with the highest
        // (timestamp, file order) wins. This is generation-safe — unlike filename
        // ordering, `compacted_*` vs `sstable_*` naming cannot resurrect stale
        // versions or undo deletes (INV-12).
        // Value: (record, timestamp, file_index); tombstones keep their record.
        let mut merged: HashMap<uuid::Uuid, (Record, i64, u64)> = HashMap::new();
        let mut paths: Vec<std::path::PathBuf> = Vec::new();
        let mut skipped_tmp = 0usize;
        let entries = std::fs::read_dir(&dir)
            .map_err(|e| StorageError::RecoveryFailed(format!("read {}: {e}", dir.display())))?;
        for e in entries {
            let p = e?.path();
            let ext = p.extension().and_then(|s| s.to_str());
            match ext {
                Some("sst") => paths.push(p),
                Some("tmp") => {
                    // Incomplete SSTable from an interrupted flush: not valid state.
                    // Safe to remove (the original is either intact or never existed).
                    let _ = std::fs::remove_file(&p);
                    skipped_tmp += 1;
                }
                _ => {}
            }
        }
        if skipped_tmp > 0 {
            tracing::info!(removed = skipped_tmp, "discarded incomplete .tmp SSTables");
        }
        paths.sort();
        for (file_idx, p) in paths.into_iter().enumerate() {
            let file_name = p.display().to_string();
            // Corruption is never silently skipped: recovery must fail loudly.
            let reader = SSTableReader::open(&p).map_err(|e| {
                StorageError::RecoveryFailed(format!("SSTable {file_name} unusable: {e}"))
            })?;
            for e in reader.iter() {
                if let Ok(rec) = Record::from_msgpack(&e.value) {
                    let entry = merged.entry(rec.id);
                    match entry {
                        std::collections::hash_map::Entry::Vacant(v) => {
                            v.insert((rec, e.timestamp, file_idx as u64));
                        }
                        std::collections::hash_map::Entry::Occupied(mut occ) => {
                            let (_, cur_ts, cur_ord) = occ.get();
                            let newer = e.timestamp > *cur_ts
                                || (e.timestamp == *cur_ts && (file_idx as u64) > *cur_ord);
                            if newer {
                                occ.insert((rec, e.timestamp, file_idx as u64));
                            }
                        }
                    }
                } else {
                    return Err(StorageError::RecoveryFailed(format!(
                        "SSTable {file_name} contains an undecodable record (key {:02x?})",
                        e.key.iter().take(8).collect::<Vec<_>>()
                    )));
                }
            }
            sstables.push(reader);
        }
        let flushed: HashMap<uuid::Uuid, Record> = merged
            .into_iter()
            .map(|(id, (rec, _, _))| (id, rec))
            .collect();

        Ok(Self {
            memtable: HashMap::new(),
            flushed_records: flushed,
            storage_dir: Some(dir),
            memtable_threshold: 1000,
            sstables,
            block_cache: RwLock::new(BlockCache::new(50_000)),
        })
    }

    /// Insert (materialize) a record. NOT logged: since Phase 1 the engine's
    /// authoritative WAL is the only mutation log; this method only applies state.
    pub fn insert(&mut self, record: Record) -> Result<Uuid, StorageError> {
        let id = record.id;
        self.apply_insert(record);
        Ok(id)
    }

    /// Explicit "apply without logging" form used by engine replay paths.
    pub fn insert_nolog(&mut self, record: Record) -> Result<(), StorageError> {
        self.apply_insert(record);
        Ok(())
    }

    fn apply_insert(&mut self, record: Record) {
        let id = record.id;
        self.memtable.insert(id, record.clone());
        self.block_cache.write().insert(id, record);
        if self.memtable.len() >= self.memtable_threshold && self.storage_dir.is_some() {
            if let Err(e) = self.flush_memtable() {
                tracing::error!(error = %e, "memtable flush failed during insert");
            }
        }
    }

    pub fn flush_memtable(&mut self) -> Result<(), StorageError> {
        if self.memtable.is_empty() {
            return Ok(());
        }
        if let Some(ref dir) = self.storage_dir {
            std::fs::create_dir_all(dir)?;
            let ts = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
            let p = dir.join(format!("sstable_{}.sst", ts));
            let mut w = SSTableWriter::new(&p)?;
            for (id, rec) in &self.memtable {
                w.append(id.as_bytes().to_vec(), rec.to_msgpack()?)?;
            }
            w.flush()?;
            // E3 window: SST fully written at its FINAL name (no tmp->rename
            // in this path — audit finding F-A), not yet installed.
            crate::crashgate::GATE_SST_AFTER_WRITE.hit();
            self.sstables.push(SSTableReader::open(&p)?);
            let config = crate::compaction::CompactionConfig::default();
            match crate::compaction::compact(dir, &config) {
                Ok(Some(result)) => {
                    if let Err(e) = crate::compaction::cleanup_merged_files(&result) {
                        tracing::warn!(error = %e, "compaction cleanup failed (harmless: extra files remain)");
                    }
                    self.sstables.clear();
                    self.sstables = Self::load_sstables(dir)?;
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(error = %e, "post-flush compaction failed (data remains safe in SSTables)")
                }
            }
        }
        for (id, rec) in self.memtable.drain() {
            if rec.tags.contains(&"__TOMBSTONE__".into()) {
                self.flushed_records.remove(&id);
            } else {
                self.flushed_records.insert(id, rec);
            }
        }
        Ok(())
    }

    /// Reload the SSTable reader list from disk (after compaction/restore).
    fn load_sstables(dir: &PathBuf) -> Result<Vec<SSTableReader>, StorageError> {
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sst"))
            .collect();
        paths.sort();
        let mut out = Vec::with_capacity(paths.len());
        for p in paths {
            out.push(SSTableReader::open(&p)?);
        }
        Ok(out)
    }

    /// Public flush: make all memtable state durable in SSTables (checkpoint path).
    pub fn flush(&mut self) -> Result<(), StorageError> {
        self.flush_memtable()
    }

    /// Number of records waiting in the memtable (checkpoint/observability).
    pub fn memtable_len(&self) -> usize {
        self.memtable.len()
    }

    pub fn get(&self, id: &Uuid) -> Option<&Record> {
        if let Some(r) = self.memtable.get(id) {
            if r.tags.contains(&"__TOMBSTONE__".into()) {
                return None;
            }
            return Some(r);
        }
        if let Some(r) = self.flushed_records.get(id) {
            if r.tags.contains(&"__TOMBSTONE__".into()) {
                return None;
            }
            return Some(r);
        }
        None
    }

    pub fn get_record(&self, id: &Uuid) -> Option<Record> {
        if let Some(r) = self.memtable.get(id) {
            if r.tags.contains(&"__TOMBSTONE__".into()) {
                return None;
            }
            return Some(r.clone());
        }
        if let Some(r) = self.flushed_records.get(id) {
            if r.tags.contains(&"__TOMBSTONE__".into()) {
                return None;
            }
            return Some(r.clone());
        }
        if let Some(cached) = self.block_cache.write().get(id) {
            if cached.tags.contains(&"__TOMBSTONE__".into()) {
                return None;
            }
            return Some(cached);
        }
        for sst in self.sstables.iter().rev() {
            if let Some(entry) = sst.get(id.as_bytes()) {
                if let Ok(rec) = Record::from_msgpack(&entry.value) {
                    self.block_cache.write().insert(*id, rec.clone());
                    if rec.tags.contains(&"__TOMBSTONE__".into()) {
                        return None;
                    }
                    return Some(rec);
                }
            }
        }
        None
    }

    /// Delete (tombstone) a record. NOT logged: the engine's WAL is authoritative.
    pub fn delete(&mut self, id: &Uuid) -> Result<(), StorageError> {
        self.apply_delete(id);
        Ok(())
    }

    /// Apply a delete without logging (engine WAL is authoritative).
    pub fn delete_nolog(&mut self, id: &Uuid) -> Result<(), StorageError> {
        self.apply_delete(id);
        Ok(())
    }

    fn apply_delete(&mut self, id: &Uuid) {
        if self.storage_dir.is_some() {
            let tombstone = tombstone_record(*id);
            self.memtable.insert(*id, tombstone.clone());
            self.block_cache.write().insert(*id, tombstone);
            if self.memtable.len() >= self.memtable_threshold {
                if let Err(e) = self.flush_memtable() {
                    tracing::error!(error = %e, "memtable flush failed during delete");
                }
            }
        } else {
            self.memtable.remove(id);
            self.flushed_records.remove(id);
            self.block_cache.write().remove(id);
        }
    }

    pub fn len(&self) -> usize {
        let mc = self
            .memtable
            .values()
            .filter(|r| !r.tags.contains(&"__TOMBSTONE__".into()))
            .count();
        let fc = self
            .flushed_records
            .values()
            .filter(|r| {
                !r.tags.contains(&"__TOMBSTONE__".into()) && !self.memtable.contains_key(&r.id)
            })
            .count();
        mc + fc
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn list_all_records(&self) -> Vec<Record> {
        let mut r: HashMap<Uuid, Record> = self
            .flushed_records
            .iter()
            .filter(|(_, rec)| !rec.tags.contains(&"__TOMBSTONE__".into()))
            .map(|(id, rec)| (*id, rec.clone()))
            .collect();
        for (id, rec) in &self.memtable {
            if rec.tags.contains(&"__TOMBSTONE__".into()) {
                r.remove(id);
            } else {
                r.insert(*id, rec.clone());
            }
        }
        r.into_values().collect()
    }

    pub fn update_record(&mut self, record: Record) -> Result<(), StorageError> {
        self.insert(record)?;
        Ok(())
    }

    pub fn cache_stats(&self) -> CacheStatsSnapshot {
        self.block_cache.read().snapshot()
    }
}
