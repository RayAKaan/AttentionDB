use std::collections::{HashMap, BinaryHeap};
use std::cmp::Reverse;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use parking_lot::RwLock;
use uuid::Uuid;

use crate::record::Record;
use crate::error::StorageError;
use crate::sstable::{SSTableWriter, SSTableReader};

/// Heap entry ordered by access_time (oldest first for LRU eviction).
#[derive(Debug, Clone)]
pub struct LruHeapEntry {
    pub access_time: u64,
    pub id: Uuid,
}

impl PartialEq for LruHeapEntry {
    fn eq(&self, other: &Self) -> bool { self.access_time == other.access_time }
}
impl Eq for LruHeapEntry {}
impl PartialOrd for LruHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
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
    pub hits: usize, pub misses: usize, pub evictions: usize,
    pub size_bytes: usize, pub entries: usize,
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
            self.lru.push(LruHeapEntry { access_time: self.access_counter, id: *id });
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
            if self.cache.contains_key(&top.id) { break; }
            self.lru.pop();
        }
        // Evict until under capacity and budget
        while self.cache.len() >= self.capacity || (self.memory_used + size > self.memory_budget) {
            if let Some(entry) = self.lru.pop() {
                if self.cache.remove(&entry.id).is_some() {
                    self.memory_used = self.memory_used.saturating_sub(256);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            } else { break; }
        }
        self.memory_used += size;
        self.cache.insert(id, record);
        self.lru.push(LruHeapEntry { access_time: self.access_counter, id });
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

    pub fn with_wal(mut self, wal: crate::wal::Wal) -> Self { self.wal = Some(wal); self }
    pub fn with_storage_dir(mut self, dir: PathBuf) -> Result<Self, StorageError> {
        std::fs::create_dir_all(&dir)?; self.storage_dir = Some(dir); Ok(self)
    }
    pub fn with_memtable_threshold(mut self, t: usize) -> Self { self.memtable_threshold = t; self }

    pub fn open(dir: PathBuf) -> Result<Self, StorageError> {
        std::fs::create_dir_all(&dir)?;
        let mut sstables = Vec::new();
        let mut flushed = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut paths: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path())
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sst")).collect();
            paths.sort();
            for p in paths {
                if let Ok(r) = SSTableReader::open(&p) {
                    for e in r.iter() {
                        if let Ok(rec) = Record::from_msgpack(&e.value) {
                            if rec.tags.contains(&"__TOMBSTONE__".into()) { flushed.remove(&rec.id); }
                            else { flushed.insert(rec.id, rec); }
                        }
                    }
                    sstables.push(r);
                }
            }
        }
        let wal_path = dir.join("store.wal");
        let mut wal = crate::wal::Wal::new(&wal_path)?;
        let entries = wal.replay()?;
        let mut memtable = HashMap::new();
        for e in entries {
            match e.op {
                crate::wal::OpType::Insert => {
                    if let Ok(rec) = Record::from_msgpack(&e.data) { memtable.insert(rec.id, rec); }
                }
                crate::wal::OpType::Delete => { memtable.remove(&e.record_id); flushed.remove(&e.record_id); }
                _ => {}
            }
        }
        Ok(Self {
            memtable, flushed_records: flushed, wal: Some(wal), storage_dir: Some(dir),
            memtable_threshold: 1000, sstables, block_cache: RwLock::new(BlockCache::new(50_000)),
        })
    }

    pub fn insert(&mut self, record: Record) -> Result<Uuid, StorageError> {
        let id = record.id;
        if let Some(ref mut w) = self.wal {
            w.append(crate::wal::OpType::Insert, "default", id, record.to_msgpack()?)?;
        }
        self.memtable.insert(id, record.clone());
        self.block_cache.write().insert(id, record);
        if self.memtable.len() >= self.memtable_threshold && self.storage_dir.is_some() {
            self.flush_memtable()?;
        }
        Ok(id)
    }

    pub fn flush_memtable(&mut self) -> Result<(), StorageError> {
        if self.memtable.is_empty() { return Ok(()); }
        if let Some(ref dir) = self.storage_dir {
            std::fs::create_dir_all(dir)?;
            let ts = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
            let p = dir.join(format!("sstable_{}.sst", ts));
            let mut w = SSTableWriter::new(&p)?;
            for (id, rec) in &self.memtable { w.append(id.as_bytes().to_vec(), rec.to_msgpack()?)?; }
            w.flush()?;
            self.sstables.push(SSTableReader::open(&p)?);
            let config = crate::compaction::CompactionConfig::default();
            if let Ok(Some(result)) = crate::compaction::compact(dir, &config) {
                let _ = crate::compaction::cleanup_merged_files(&result);
                self.sstables.clear();
                if let Ok(entries) = std::fs::read_dir(dir) {
                    let mut paths: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path())
                        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sst")).collect();
                    paths.sort();
                    for p in paths { if let Ok(r) = SSTableReader::open(&p) { self.sstables.push(r); } }
                }
            }
        }
        for (id, rec) in self.memtable.drain() {
            if rec.tags.contains(&"__TOMBSTONE__".into()) { self.flushed_records.remove(&id); }
            else { self.flushed_records.insert(id, rec); }
        }
        Ok(())
    }

    pub fn get(&self, id: &Uuid) -> Option<&Record> {
        if let Some(r) = self.memtable.get(id) {
            if r.tags.contains(&"__TOMBSTONE__".into()) { return None; }
            return Some(r);
        }
        if let Some(r) = self.flushed_records.get(id) {
            if r.tags.contains(&"__TOMBSTONE__".into()) { return None; }
            return Some(r);
        }
        None
    }

    pub fn get_record(&self, id: &Uuid) -> Option<Record> {
        if let Some(r) = self.memtable.get(id) {
            if r.tags.contains(&"__TOMBSTONE__".into()) { return None; }
            return Some(r.clone());
        }
        if let Some(r) = self.flushed_records.get(id) {
            if r.tags.contains(&"__TOMBSTONE__".into()) { return None; }
            return Some(r.clone());
        }
        if let Some(cached) = self.block_cache.write().get(id) {
            if cached.tags.contains(&"__TOMBSTONE__".into()) { return None; }
            return Some(cached);
        }
        for sst in self.sstables.iter().rev() {
            if let Some(entry) = sst.get(id.as_bytes()) {
                if let Ok(rec) = Record::from_msgpack(&entry.value) {
                    self.block_cache.write().insert(*id, rec.clone());
                    if rec.tags.contains(&"__TOMBSTONE__".into()) { return None; }
                    return Some(rec);
                }
            }
        }
        None
    }

    pub fn delete(&mut self, id: &Uuid) -> Result<(), StorageError> {
        if let Some(ref mut w) = self.wal {
            w.append(crate::wal::OpType::Delete, "default", *id, vec![])?;
        }
        if self.storage_dir.is_some() {
            let mut tombstone = Record::new(HashMap::new());
            tombstone.id = *id;
            tombstone.tags.push("__TOMBSTONE__".into());
            self.memtable.insert(*id, tombstone.clone());
            self.block_cache.write().insert(*id, tombstone);
            if self.memtable.len() >= self.memtable_threshold { self.flush_memtable()?; }
        } else {
            self.memtable.remove(id);
            self.flushed_records.remove(id);
            self.block_cache.write().remove(id);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        let mc = self.memtable.values().filter(|r| !r.tags.contains(&"__TOMBSTONE__".into())).count();
        let fc = self.flushed_records.values().filter(|r| !r.tags.contains(&"__TOMBSTONE__".into()) && !self.memtable.contains_key(&r.id)).count();
        mc + fc
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn list_all_records(&self) -> Vec<Record> {
        let mut r: HashMap<Uuid, Record> = self.flushed_records.iter()
            .filter(|(_, rec)| !rec.tags.contains(&"__TOMBSTONE__".into()))
            .map(|(id, rec)| (*id, rec.clone())).collect();
        for (id, rec) in &self.memtable {
            if rec.tags.contains(&"__TOMBSTONE__".into()) { r.remove(id); }
            else { r.insert(*id, rec.clone()); }
        }
        r.into_values().collect()
    }

    pub fn update_record(&mut self, record: Record) -> Result<(), StorageError> {
        self.insert(record)?; Ok(())
    }

    pub fn cache_stats(&self) -> CacheStatsSnapshot {
        self.block_cache.read().snapshot()
    }
}