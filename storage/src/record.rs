//! Record definition and serialization for AttentionDB Phase 1

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: Uuid,
    pub version: u64,
    pub timestamp: i64,
    pub fields: HashMap<String, serde_json::Value>,
    pub k_vecs: Vec<Vec<f32>>,
    pub v_vecs: Vec<Vec<f32>>,
    pub t_embed: Vec<f32>,
    pub schema_id: Option<u32>,
    pub tags: Vec<String>,
}

impl Record {
    pub fn new(fields: HashMap<String, serde_json::Value>) -> Self {
        Self {
            id: Uuid::new_v4(),
            version: 1,
            timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            fields,
            k_vecs: vec![],
            v_vecs: vec![],
            t_embed: vec![0.0; 64],
            schema_id: None,
            tags: vec![],
        }
    }

    pub fn to_msgpack(&self) -> Result<Vec<u8>, crate::error::StorageError> {
        rmp_serde::to_vec(self)
            .map_err(|e| crate::error::StorageError::Serialization(e.to_string()))
    }

    pub fn from_msgpack(data: &[u8]) -> Result<Self, crate::error::StorageError> {
        rmp_serde::from_slice(data)
            .map_err(|e| crate::error::StorageError::Serialization(e.to_string()))
    }
}
