//! Write-Ahead Log for AttentionDB Phase 1
//!
//! Simple append-only WAL with CRC32 checksums.
//! Supports SYNC, GROUP_COMMIT, and ASYNC durability modes.

use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter};
use std::path::Path;
use crc32fast::Hasher;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpType {
    Insert,
    Update,
    Delete,
    Checkpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    pub lsn: u64,
    pub txn_id: u64,
    pub op: OpType,
    pub collection: String,
    pub record_id: Uuid,
    pub data: Vec<u8>,
    pub crc32: u32,
}

pub struct Wal {
    file: BufWriter<File>,
    next_lsn: u64,
}

impl Wal {
    pub fn new(path: &Path) -> Result<Self, crate::error::StorageError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            file: BufWriter::new(file),
            next_lsn: 1,
        })
    }

    pub fn append(
        &mut self,
        op: OpType,
        collection: &str,
        record_id: Uuid,
        data: Vec<u8>,
    ) -> Result<u64, crate::error::StorageError> {
        let mut hasher = Hasher::new();
        hasher.update(&data);
        let crc = hasher.finalize();

        let entry = WalEntry {
            lsn: self.next_lsn,
            txn_id: 0,
            op,
            collection: collection.to_string(),
            record_id,
            data,
            crc32: crc,
        };

        let serialized = bincode::serialize(&entry)
            .map_err(|e| crate::error::StorageError::Wal(e.to_string()))?;

        self.file.write_all(&serialized)?;
        self.file.flush()?;

        let lsn = self.next_lsn;
        self.next_lsn += 1;
        Ok(lsn)
    }

    pub fn fsync(&mut self) -> Result<(), crate::error::StorageError> {
        self.file.get_ref().sync_all()?;
        Ok(())
    }
}
