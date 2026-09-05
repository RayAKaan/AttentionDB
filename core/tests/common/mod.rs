//! Shared helpers for the Phase 1 integration test suite.

#![allow(dead_code)]

use attentiondb_core::AttentionEngine;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;

pub fn temp_db() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

pub fn open_sync(dir: &std::path::Path) -> AttentionEngine {
    AttentionEngine::open_dir(dir, Durability::Sync).unwrap()
}

pub fn try_open_sync(
    dir: &std::path::Path,
) -> Result<AttentionEngine, attentiondb_core::CoreError> {
    AttentionEngine::open_dir(dir, Durability::Sync)
}

pub fn doc(idx: i64, vector: &[f32], body: &str) -> Record {
    let mut fields = HashMap::new();
    fields.insert("idx".to_string(), serde_json::json!(idx));
    fields.insert("body".to_string(), serde_json::json!(body));
    let mut r = Record::new(fields);
    r.k_vecs.insert("default".to_string(), vector.to_vec());
    r
}

pub fn one_hot(i: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0; dim];
    v[i % dim] = 1.0;
    v
}

/// Find the numeric id of the document whose `idx` field equals `idx`.
pub fn id_of_idx(engine: &AttentionEngine, collection: &str, idx: i64) -> Option<u64> {
    let _ = collection;
    let all = engine.document_store.read().list_all_records();
    let target = all
        .into_iter()
        .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(idx))?;
    engine.id_mapper.read().uuid_to_id(&target.id)
}

/// Simple deterministic PRNG (xorshift64*) for property tests.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}
