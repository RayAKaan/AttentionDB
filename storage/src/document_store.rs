//! Real LSM-style Document Store (.adb)
//!
//! Implements memtable buffering, WAL logging, SSTable flushing,
//! highly concurrent LRU disk frame block caching, and disk recovery.

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use parking_lot::RwLock;
use uuid::Uuid;
use tracing::info;
use crate::record::Record;
use crate::error::StorageError;
use crate::sstable::{SSTableWriter, SSTableReader};

/// A cache entry ordered by reverse access time for true LRU eviction.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub record: Record,
    /// Monotonically increasing access counter (higher = more recent).
    pub access_order: u64,
}

impl PartialEq for CacheEntry {
    fn eq(&self, other: &Self) -> bool {
        self.access_order == other.access_order
    }
}

impl Eq for CacheEntry {}

impl PartialOrd for CacheEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CacheEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.access_order.cmp(&other.access_order)
    }
}

/// Statistics for the block cache.
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: AtomicUsize,
    pub misses: AtomicUsize,
    pub evictions: AtomicUsize,
    pub current_entries: AtomicUsize,
    pub memory_usage_bytes: AtomicU64,
}

impl Clone for CacheStats {
    fn clone(&self) -> Self {
        Self {
            hits: AtomicUsize::new(self.hits.load(Ordering::Relaxed)),
            misses: AtomicUsize::new(self.misses.load(Ordering::Relaxed)),
            evictions: AtomicUsize::new(self.evictions.load(Ordering::Relaxed)),
            current_entries: AtomicUsize::new(self.current_entries.load(Ordering::Relaxed)),
            memory_usage_bytes: AtomicU64::new(self.memory_usage_bytes.load(Ordering::Relaxed)),
        }
    }
}

impl CacheStats {
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            current_entries: self.current_entries.load(Ordering::Relaxed),
            memory_usage_bytes: self.memory_usage_bytes.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheStatsSnapshot {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub current_entries: usize,
    pub memory_usage_bytes: u64,
}

/// Block cache using a BinaryHeap for true O(log n) LRU eviction.
pub struct BlockCache {
    pub cache: HashMap<Uuid, CacheEntry>,
    pub lru_heap: BinaryHeap<Reverse<(u64, Uuid)>>,
    pub capacity: usize,
    pub stats: CacheStats,
    /// Approximate memory budget in bytes (0 = unlimited).
    pub memory_budget: u64,
    access_counter: AtomicU64,
}

impl BlockCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            lru_heap: BinaryHeap::with_capacity(capacity),
            capacity,
            stats: CacheStats::default(),
            memory_budget: 0,
            access_counter: AtomicU64::new(1),
        }
    }

    pub fn with_memory_budget(mut self, bytes: u64) -> Self {
        self.memory_budget = bytes;
        self
    }

    fn next_access(&self) -> u64 {
        self.access_counter.fetch_add(1, Ordering::Relaxed)
    }

    pub fn get(&mut self, id: &Uuid) -> Option<Record> {
        if self.cache.contains_key(id) {
            let new_order = self.next_access();
            if let Some(entry) = self.cache.get_mut(id) {
                entry.access_order = new_order;
                self.lru_heap.push(Reverse((new_order, *id)));
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Some(entry.record.clone());
            }
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub fn insert(&mut self, id: Uuid, record: Record) {
        let access_order = self.next_access();
        let size_estimate = estimate_record_size(&record);

        // Evict until capacity or memory budget is satisfied
        while self.cache.len() >= self.capacity || (self.memory_budget > 0 && self.estimate_memory_usage() + size_estimate as u64 > self.memory_budget) {
            if let Some(Reverse((_, evict_id))) = self.lru_heap.pop() {
                if self.cache.remove(&evict_id).is_some() {
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                    self.stats.current_entries.fetch_sub(1, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }

        // Clean stale heap entries lazily
        while let Some(Reverse((order, id))) = self.lru_heap.peek() {
            if let Some(entry) = self.cache.get(id) {
                if entry.access_order != *order {
                    self.lru_heap.pop();
                } else {
                    break;
                }
            } else {
                self.lru_heap.pop();
            }
        }

        self.cache.insert(id, CacheEntry { record, access_order });
        self.lru_heap.push(Reverse((access_order, id)));
        self.stats.current_entries.fetch_add(1, Ordering::Relaxed);
        self.stats.memory_usage_bytes.fetch_add(size_estimate as u64, Ordering::Relaxed);
    }

    pub fn remove(&mut self, id: &Uuid) {
        if self.cache.remove(id).is_some() {
            self.stats.current_entries.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_heap.clear();
        self.stats.current_entries.store(0, Ordering::Relaxed);
        self.stats.memory_usage_bytes.store(0, Ordering::Relaxed);
    }

    fn estimate_memory_usage(&self) -> u64 {
        self.stats.memory_usage_bytes.load(Ordering::Relaxed)
    }

    /// Resize the cache capacity (triggers eviction if shrinking).
    pub fn resize(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
        while self.cache.len() > self.capacity {
            if let Some(Reverse((_, evict_id))) = self.lru_heap.pop() {
                if self.cache.remove(&evict_id).is_some() {
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                    self.stats.current_entries.fetch_sub(1, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }
        info!("[BlockCache] Resized to capacity={}", new_capacity);
    }
}

fn estimate_record_size(record: &Record) -> usize {
    let mut size = std::mem::size_of::<Record>();
    size += record.fields.len() * 64;
    size += record.k_vecs.len() * 128;
    size += record.tags.len() * 32;
    size
}

pub struct DocumentStore {
    memtable: HashMap<Uuid, Record>,
    flushed_records: HashMap<Uuid, Record>,
    wal: Option<crate::wal::Wal>,
    storage_dir: Option<PathBuf>,
    memtable_threshold: usize,
    sstables: Vec<SSTableReader>,
    pub block_cache: RwLock<BlockCache>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            memtable: HashMap::new(),
            flushed_records: HashMap::new(),
            wal: None,
            storage_dir: None,
            memtable_threshold: 1000,
            sstables: Vec::new(),
            block_cache: RwLock::new(BlockCache::new(50_000)),
        }
    }

    pub fn with_wal(mut self, wal: crate::wal::Wal) -> Self {
        self.wal = Some(wal);
        self
    }

    pub fn with_storage_dir(mut self, dir: PathBuf) -> Result<Self, StorageError> {
        std::fs::create_dir_all(&dir)?;
        self.storage_dir = Some(dir);
        Ok(self)
    }

    pub fn with_memtable_threshold(mut self, threshold: usize) -> Self {
        self.memtable_threshold = threshold;
        self
    }

    pub fn open(dir: PathBuf) -> Result<Self, StorageError> {
        std::fs::create_dir_all(&dir)?;
        let mut sstables = Vec::new();
        let mut flushed_records = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut paths: Vec<_> = entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sst"))
                .collect();
            paths.sort();
            for path in paths {
                if let Ok(reader) = SSTableReader::open(&path) {
                    for entry in reader.iter() {
                        if let Ok(record) = Record::from_msgpack(&entry.value) {
                            if record.tags.contains(&"__TOMBSTONE__".to_string()) {
                                flushed_records.remove(&record.id);
                            } else {
                                flushed_records.insert(record.id, record);
                            }
                        }
                    }
                    sstables.push(reader);
                }
            }
        }
        let wal_path = dir.join("store.wal");
        let mut wal = crate::wal::Wal::new(&wal_path)?;
        let wal_entries = wal.replay()?;
        let mut memtable = HashMap::new();
        for entry in wal_entries {
            match entry.op {
                crate::wal::OpType::Insert => {
                    if let Ok(record) = Record::from_msgpack(&entry.data) {
                        memtable.insert(record.id, record);
                    }
                }
                crate::wal::OpType::Delete => {
                    memtable.remove(&entry.record_id);
                    flushed_records.remove(&entry.record_id);
                }
                _ => {}
            }
        }
        Ok(Self {
            memtable,
            flushed_records,
            wal: Some(wal),
            storage_dir: Some(dir),
            memtable_threshold: 1000,
            sstables,
            block_cache: RwLock::new(BlockCache::new(50_000)),
        })
    }

    pub fn insert(&mut self, record: Record) -> Result<Uuid, StorageError> {
        let id = record.id;
        if let Some(ref mut wal) = self.wal {
            let data = record.to_msgpack()?;
            wal.append(crate::wal::OpType::Insert, "default", id, data)?;
        }
        self.memtable.insert(id, record.clone());
        self.block_cache.write().insert(id, record);
        if self.memtable.len() >= self.memtable_threshold {
            if self.storage_dir.is_some() {
                self.flush_memtable()?;
            }
        }
        Ok(id)
    }

    pub fn flush_memtable(&mut self) -> Result<(), StorageError> {
        if self.memtable.is_empty() {
            return Ok(());
        }
        if let Some(ref dir) = self.storage_dir {
            std::fs::create_dir_all(dir)?;
            let now = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
            let sst_path = dir.join(format!("sstable_{}.sst", now));
            let mut writer = SSTableWriter::new(&sst_path)?;
            for (id, record) in &self.memtable {
                let key = id.as_bytes().to_vec();
                let value = record.to_msgpack()?;
                writer.append(key, value)?;
            }
            writer.flush()?;
            let reader = SSTableReader::open(&sst_path)?;
            self.sstables.push(reader);
            // Trigger compaction
            let config = crate::compaction::CompactionConfig::default();
            if let Ok(Some(result)) = crate::compaction::compact(dir, &config) {
                let _ = crate::compaction::cleanup_merged_files(&result);
                self.sstables.clear();
                if let Ok(entries) = std::fs::read_dir(dir) {
                    let mut paths: Vec<_> = entries
                        .filter_map(Result::ok)
                        .map(|e| e.path())
                        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sst"))
                        .collect();
                    paths.sort();
                    for path in paths {
                        if let Ok(reader) = SSTableReader::open(&path) {
                            self.sstables.push(reader);
                        }
                    }
                }
            }
        }
        for (id, record) in self.memtable.drain() {
            if record.tags.contains(&"__TOMBSTONE__".to_string()) {
                self.flushed_records.remove(&id);
            } else {
                self.flushed_records.insert(id, record);
            }
        }
        Ok(())
    }

    pub fn get(&self, id: &Uuid) -> Option<&Record> {
        if let Some(record) = self.memtable.get(id) {
            if record.tags.contains(&"__TOMBSTONE__".to_string()) {
                return None;
            }
            return Some(record);
        }
        if let Some(record) = self.flushed_records.get(id) {
            if record.tags.contains(&"__TOMBSTONE__".to_string()) {
                return None;
            }
            return Some(record);
        }
        None
    }

    pub fn get_record(&self, id: &Uuid) -> Option<Record> {
        if let Some(record) = self.memtable.get(id) {
            if record.tags.contains(&"__TOMBSTONE__".to_string()) {
                return None;
            }
            return Some(record.clone());
        }
        if let Some(record) = self.flushed_records.get(id) {
            if record.tags.contains(&"__TOMBSTONE__".to_string()) {
                return None;
            }
            return Some(record.clone());
        }
        if let Some(cached) = self.block_cache.write().get(id) {
            if cached.tags.contains(&"__TOMBSTONE__".to_string()) {
                return None;
            }
            return Some(cached);
        }
        for sst in self.sstables.iter().rev() {
            if let Some(entry) = sst.get(id.as_bytes()) {
                if let Ok(record) = Record::from_msgpack(&entry.value) {
                    self.block_cache.write().insert(*id, record.clone());
                    if record.tags.contains(&"__TOMBSTONE__".to_string()) {
                        return None;
                    }
                    return Some(record);
                }
            }
        }
        None
    }

    pub fn delete(&mut self, id: &Uuid) -> Result<(), StorageError> {
        if let Some(ref mut wal) = self.wal {
            wal.append(crate::wal::OpType::Delete, "default", *id, vec![])?;
        }
        if self.storage_dir.is_some() {
            let mut tombstone = Record::new(HashMap::new());
            tombstone.id = *id;
            tombstone.tags.push("__TOMBSTONE__".to_string());
            self.memtable.insert(*id, tombstone.clone());
            self.block_cache.write().insert(*id, tombstone);
            if self.memtable.len() >= self.memtable_threshold {
                self.flush_memtable()?;
            }
        } else {
            self.memtable.remove(id);
            self.flushed_records.remove(id);
            self.block_cache.write().remove(id);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        let mem_count = self.memtable.values().filter(|r| !r.tags.contains(&"__TOMBSTONE__".to_string())).count();
        let flushed_count = self.flushed_records.values().filter(|r| !r.tags.contains(&"__TOMBSTONE__".to_string()) && !self.memtable.contains_key(&r.id)).count();
        mem_count + flushed_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn list_all_records(&self) -> Vec<Record> {
        let mut results: HashMap<Uuid, Record> = HashMap::new();
        for (id, rec) in &self.flushed_records {
            if !rec.tags.contains(&"__TOMBSTONE__".to_string()) {
                results.insert(*id, rec.clone());
            }
        }
        for (id, rec) in &self.memtable {
            if rec.tags.contains(&"__TOMBSTONE__".to_string()) {
                results.remove(id);
            } else {
                results.insert(*id, rec.clone());
            }
        }
        results.into_values().collect()
    }

    pub fn update_record(&mut self, record: Record) -> Result<(), StorageError> {
        self.insert(record)?;
        Ok(())
    }

    /// Return cache statistics.
    pub fn cache_stats(&self) -> CacheStatsSnapshot {
        self.block_cache.read().stats.snapshot()
    }
}