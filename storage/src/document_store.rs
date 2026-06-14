//! Simple LSM-style Document Store (.adb)
//!
//! Phase 1: In-memory + append-only file (real LSM in later phases)

use std::collections::HashMap;
use uuid::Uuid;
use crate::record::Record;
use crate::error::StorageError;

pub struct DocumentStore {
    memtable: HashMap<Uuid, Record>,
    wal: Option<crate::wal::Wal>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self {
            memtable: HashMap::new(),
            wal: None,
        }
    }

    pub fn with_wal(mut self, wal: crate::wal::Wal) -> Self {
        self.wal = Some(wal);
        self
    }

    pub fn insert(&mut self, record: Record) -> Result<Uuid, StorageError> {
        let id = record.id;

        if let Some(ref mut wal) = self.wal {
            let data = record.to_msgpack()?;
            wal.append(crate::wal::OpType::Insert, "default", id, data)?;
        }

        self.memtable.insert(id, record);
        Ok(id)
    }

    pub fn get(&self, id: &Uuid) -> Option<&Record> {
        self.memtable.get(id)
    }

    pub fn delete(&mut self, id: &Uuid) -> Result<(), StorageError> {
        if let Some(ref mut wal) = self.wal {
            wal.append(crate::wal::OpType::Delete, "default", *id, vec![])?;
        }
        self.memtable.remove(id);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.memtable.len()
    }
}
