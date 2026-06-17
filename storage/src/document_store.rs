use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;
use crate::record::Record;
use crate::error::StorageError;
use crate::sstable::{SSTableWriter, SSTableReader};

pub struct DocumentStore {
    memtable: HashMap<Uuid, Record>,
    flushed_records: HashMap<Uuid, Record>,
    wal: Option<crate::wal::Wal>,
    storage_dir: Option<PathBuf>,
    memtable_threshold: usize,
    sstables: Vec<SSTableReader>,
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
        })
    }

    pub fn insert(&mut self, record: Record) -> Result<Uuid, StorageError> {
        let id = record.id;

        if let Some(ref mut wal) = self.wal {
            let data = record.to_msgpack()?;
            wal.append(crate::wal::OpType::Insert, "default", id, data)?;
        }

        self.memtable.insert(id, record);

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

            // Trigger compaction if enough SST files have accumulated.
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

    pub fn delete(&mut self, id: &Uuid) -> Result<(), StorageError> {
        if let Some(ref mut wal) = self.wal {
            wal.append(crate::wal::OpType::Delete, "default", *id, vec![])?;
        }
        if self.storage_dir.is_some() {
            let mut tombstone = Record::new(HashMap::new());
            tombstone.id = *id;
            tombstone.tags.push("__TOMBSTONE__".to_string());
            self.memtable.insert(*id, tombstone);
            if self.memtable.len() >= self.memtable_threshold {
                self.flush_memtable()?;
            }
        } else {
            self.memtable.remove(id);
            self.flushed_records.remove(id);
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
}
