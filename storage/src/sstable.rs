use crate::error::StorageError;
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

const SSTABLE_MAGIC: u32 = 0xA54D_4244; // 'A' 'M' 'B' 'D'
/// SSTable format version. v2 = [magic][fmt_ver u32][crc32][bincode payload];
/// v1 (legacy, no version field) = [magic][crc32][payload] and remains readable.
pub const SSTABLE_FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSTableEntry {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
    pub timestamp: i64,
}

pub struct SSTableWriter {
    file: Option<File>,
    entries: Vec<SSTableEntry>,
    path: String,
    /// Temporary path used for crash-safe installation (write tmp → fsync → rename).
    tmp_path: String,
    finished: bool,
}

impl SSTableWriter {
    pub fn new(path: &Path) -> Result<Self, StorageError> {
        // Write to a `.tmp` sibling so a crash mid-flush can never leave a partial
        // file at the final path that a later open() might mistake for valid state.
        let path_str = path.to_string_lossy().to_string();
        let tmp = format!("{path_str}.tmp");
        let file = File::create(&tmp)?;
        Ok(Self {
            file: Some(file),
            entries: Vec::new(),
            path: path_str,
            tmp_path: tmp,
            finished: false,
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

    /// Append with an EXPLICIT logical timestamp. Compaction MUST use this:
    /// re-writing an old record with a fresh wall-clock timestamp would make it
    /// beat newer tombstones/updates living in files outside the merge set,
    /// resurrecting deleted data (INV-12 timestamp-merge correctness).
    pub fn append_with_timestamp(
        &mut self,
        key: Vec<u8>,
        value: Vec<u8>,
        timestamp: i64,
    ) -> Result<(), StorageError> {
        let entry = SSTableEntry {
            key,
            value,
            timestamp,
        };
        self.entries.push(entry);
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), StorageError> {
        if self.finished {
            return Err(StorageError::Sstable(format!(
                "SSTableWriter for {} already finished",
                self.path
            )));
        }
        self.entries.sort_by(|a, b| a.key.cmp(&b.key));

        let payload =
            bincode::serialize(&self.entries).map_err(|e| StorageError::Sstable(e.to_string()))?;
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let crc32 = hasher.finalize();

        let mut file = self
            .file
            .take()
            .ok_or_else(|| StorageError::Sstable("writer already closed".into()))?;
        file.write_all(&SSTABLE_MAGIC.to_be_bytes())?;
        file.write_all(&SSTABLE_FORMAT_VERSION.to_be_bytes())?;
        file.write_all(&crc32.to_be_bytes())?;
        file.write_all(&payload)?;
        file.sync_all()?;
        drop(file);

        // Atomically install: readers only ever see the complete file.
        let final_path = Path::new(&self.path);
        std::fs::rename(&self.tmp_path, final_path)?;
        if let Some(parent) = final_path.parent() {
            crate::catalog::fsync_dir(parent)?;
        }
        self.finished = true;
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
        let file_name = path.display().to_string();
        let entries = if buf.len() >= 12 {
            let magic = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
            if magic != SSTABLE_MAGIC {
                return Err(StorageError::corruption(&file_name, "bad SSTable magic"));
            }
            // Try v2 first: [magic][fmt_ver u32][crc u32][payload]. The version
            // field must equal SSTABLE_FORMAT_VERSION *and* the checksum+payload
            // must validate. Otherwise fall back to legacy v1
            // [magic][crc u32][payload] for backward compatibility. Anything that
            // validates as neither is corruption — never silently reinterpreted.
            let ver = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
            let mut parsed: Option<Vec<SSTableEntry>> = None;
            if ver == SSTABLE_FORMAT_VERSION {
                parsed = Self::parse_with_layout(&buf, 12, 8, &file_name).ok();
            }
            if parsed.is_none() {
                // legacy v1
                parsed = Some(Self::parse_with_layout(&buf, 8, 4, &file_name)?);
            }
            parsed.unwrap()
        } else if buf.is_empty() {
            return Err(StorageError::corruption(&file_name, "empty SSTable file"));
        } else {
            return Err(StorageError::corruption(&file_name, "SSTable too short"));
        };

        Ok(Self { entries })
    }

    /// Parse entries with a given (payload_offset, crc_offset) layout.
    fn parse_with_layout(
        buf: &[u8],
        payload_offset: usize,
        crc_offset: usize,
        file_name: &str,
    ) -> Result<Vec<SSTableEntry>, StorageError> {
        let payload = &buf[payload_offset..];
        let expected_crc = u32::from_be_bytes([
            buf[crc_offset],
            buf[crc_offset + 1],
            buf[crc_offset + 2],
            buf[crc_offset + 3],
        ]);
        let mut hasher = Hasher::new();
        hasher.update(payload);
        if hasher.finalize() != expected_crc {
            return Err(StorageError::ChecksumMismatch {
                file: file_name.to_string(),
            });
        }
        bincode::deserialize(payload)
            .map_err(|e| StorageError::corruption(file_name, e.to_string()))
    }

    pub fn get(&self, key: &[u8]) -> Option<&SSTableEntry> {
        self.entries
            .binary_search_by(|e| e.key.as_slice().cmp(key))
            .ok()
            .map(|idx| &self.entries[idx])
    }

    pub fn range(&self, start: &[u8], end: &[u8]) -> Vec<&SSTableEntry> {
        let start_idx = self
            .entries
            .binary_search_by(|e| e.key.as_slice().cmp(start))
            .unwrap_or_else(|i| i);
        let end_idx = self
            .entries
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

pub fn sstable_entries_from_map(
    map: std::collections::BTreeMap<Vec<u8>, Vec<u8>>,
) -> Vec<SSTableEntry> {
    let now = chrono::Utc::now().timestamp_millis();
    map.into_iter()
        .map(|(key, value)| SSTableEntry {
            key,
            value,
            timestamp: now,
        })
        .collect()
}
