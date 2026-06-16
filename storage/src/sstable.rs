use std::fs::File;
use std::io::{Write, Read};
use std::path::Path;
use serde::{Serialize, Deserialize};
use crate::error::StorageError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSTableEntry {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub timestamp: i64,
}

pub struct SSTableWriter {
    file: File,
    entries: Vec<SSTableEntry>,
    path: String,
}

impl SSTableWriter {
    pub fn new(path: &Path) -> Result<Self, StorageError> {
        let file = File::create(path)?;
        Ok(Self {
            file,
            entries: Vec::new(),
            path: path.to_string_lossy().to_string(),
        })
    }

    pub fn append(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<(), StorageError> {
        let entry = SSTableEntry {
            key,
            value,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        self.entries.push(entry);
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), StorageError> {
        self.entries.sort_by(|a, b| a.key.cmp(&b.key));

        let serialized = bincode::serialize(&self.entries)
            .map_err(|e| StorageError::Sstable(e.to_string()))?;

        self.file.write_all(&serialized)?;
        self.file.sync_all()?;
        Ok(())
    }

    pub fn path(&self) -> &str {
        &self.path
    }
}

pub struct SSTableReader {
    entries: Vec<SSTableEntry>,
}

impl SSTableReader {
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let entries: Vec<SSTableEntry> = bincode::deserialize(&buf)
            .map_err(|e| StorageError::Sstable(e.to_string()))?;

        Ok(Self { entries })
    }

    pub fn get(&self, key: &[u8]) -> Option<&SSTableEntry> {
        self.entries
            .binary_search_by(|e| e.key.as_slice().cmp(key))
            .ok()
            .map(|idx| &self.entries[idx])
    }

    pub fn range(&self, start: &[u8], end: &[u8]) -> Vec<&SSTableEntry> {
        let start_idx = self.entries
            .binary_search_by(|e| e.key.as_slice().cmp(start))
            .unwrap_or_else(|i| i);
        let end_idx = self.entries
            .binary_search_by(|e| e.key.as_slice().cmp(end))
            .unwrap_or_else(|i| i);
        self.entries[start_idx..end_idx].iter().collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = &SSTableEntry> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

pub fn sstable_entries_from_map(map: std::collections::BTreeMap<Vec<u8>, Vec<u8>>) -> Vec<SSTableEntry> {
    let now = chrono::Utc::now().timestamp_millis();
    map.into_iter()
        .map(|(key, value)| SSTableEntry { key, value, timestamp: now })
        .collect()
}
