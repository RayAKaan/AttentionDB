use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter, Read};
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Durability {
    Sync,
    GroupCommit,
    Async,
}

pub struct Wal {
    file: BufWriter<File>,
    path: String,
    next_lsn: u64,
    durability: Durability,
}

impl Wal {
    pub fn new(path: &Path) -> Result<Self, crate::error::StorageError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            file: BufWriter::new(file),
            path: path.to_string_lossy().to_string(),
            next_lsn: 1,
            durability: Durability::GroupCommit,
        })
    }

    pub fn with_durability(mut self, durability: Durability) -> Self {
        self.durability = durability;
        self
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

        match self.durability {
            Durability::Sync => {
                self.file.flush()?;
                self.file.get_ref().sync_all()?;
            }
            Durability::GroupCommit => {
                self.file.flush()?;
            }
            Durability::Async => {}
        }

        let lsn = self.next_lsn;
        self.next_lsn += 1;
        Ok(lsn)
    }

    pub fn fsync(&mut self) -> Result<(), crate::error::StorageError> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        Ok(())
    }

    pub fn verify_entry(entry: &WalEntry) -> bool {
        let mut hasher = Hasher::new();
        hasher.update(&entry.data);
        let expected_crc = hasher.finalize();
        expected_crc == entry.crc32
    }

    pub fn replay(&mut self) -> Result<Vec<WalEntry>, crate::error::StorageError> {
        let mut file = OpenOptions::new().read(true).open(&self.path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let mut entries = Vec::new();
        let mut offset = 0;
        while offset < buf.len() {
            match bincode::deserialize::<WalEntry>(&buf[offset..]) {
                Ok(entry) => {
                    let entry_size = bincode::serialized_size(&entry).unwrap() as usize;
                    if Self::verify_entry(&entry) {
                        entries.push(entry);
                    }
                    offset += entry_size;
                }
                Err(_) => break,
            }
        }

        if let Some(last) = entries.last() {
            self.next_lsn = last.lsn + 1;
        }

        Ok(entries)
    }
}
