//! REGRESSION (E8 soak find — ENGINE DEFECT #2): same-millisecond flushes let
//! a deleted record's own flush shadow its tombstone (lexical file-order
//! inversion), and full-compaction tombstone GC then leaked the dead record;
//! recovery rightly refused (MISSING_MAPPING orphan).
//!
//! Fix under test: flush entry timestamps are LOGICALLY MONOTONIC
//! (max(now_ms, last+1)), so a tombstone flushed after its record always
//! resolves newer regardless of file names/order.
//!
//! Preserved failing state: research/phase3/raw/runs/PH3E-SOAK-004/
//! bug-preserved-e8d-refusal/ (18 leaked records, deterministic refusal).
use attentiondb_core::engine::AttentionEngine;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;

const DIM: usize = 8;
const HEAD: &str = "head";

fn mk(idx: u32, ver: u64) -> Record {
    let mut fields = HashMap::new();
    fields.insert("idx".to_string(), serde_json::json!(idx));
    fields.insert("version".to_string(), serde_json::json!(ver));
    let mut r = Record::new(fields);
    r.id = uuid::Uuid::from_u128(((idx as u128) << 64) | (ver as u128));
    let mut v = vec![0.0f32; DIM];
    v[(idx % 7) as usize] = 1.0;
    r.k_vecs.insert(HEAD.to_string(), v);
    r
}

fn uuid_of(idx: u32, ver: u64) -> uuid::Uuid {
    uuid::Uuid::from_u128(((idx as u128) << 64) | (ver as u128))
}

/// Minimal repro of the E8d leak: insert→flush then delete→flush back-to-back
/// (same-millisecond flush pairs), a partial compaction (threshold flush does
/// this), then a full compaction (tombstone GC), close, reopen. Pre-fix the
/// dead record leaked into the compacted SST and recovery refused.
#[test]
fn regression_no_tombstone_shadow_leak() {
    let dir = std::env::temp_dir().join("e8reg-tombstone-shadow");
    let _ = std::fs::remove_dir_all(&dir);
    let e = AttentionEngine::open_dir(&dir, Durability::Sync).expect("open");
    e.create_collection("bench", DIM, &[HEAD]).unwrap();

    // 64 keys: insert v1, flush; delete v1, flush — back-to-back so the two
    // flushes land in the same wall-clock millisecond (the pre-fix trigger).
    for k in 0..64u32 {
        e.insert_document("bench", mk(k, 1)).unwrap();
        e.checkpoint().unwrap(); // flush #1 (record)
        let t = e.begin_transaction("bench");
        e.record_transaction_operation(
            t,
            attentiondb_core::transaction::TxnOp::Delete(uuid_of(k, 1)),
        )
        .unwrap();
        assert!(e.commit_transaction(t).unwrap());
        e.checkpoint().unwrap(); // flush #2 (tombstone) — back-to-back with #1
    }
    // partial compaction (puts tombstones into compacted_* names), then
    // another churn round so later full compaction merges everything (GC).
    e.compact_storage().unwrap(); // full merge + tombstone GC
    for k in 0..64u32 {
        e.insert_document("bench", mk(k, 2)).unwrap();
    }
    e.checkpoint().unwrap();
    e.compact_storage().unwrap(); // final full merge
    e.close().unwrap();

    // Judge: fresh open must succeed and show exactly the v2 docs.
    let e2 = AttentionEngine::open_dir(&dir, Durability::Sync)
        .expect("REGRESSION: recovery refused — tombstone shadow leak re-appeared");
    for k in 0..64u32 {
        assert!(
            e2.document_store.read().get(&uuid_of(k, 2)).is_some(),
            "live doc ({k},2) missing after recovery"
        );
        assert!(
            e2.document_store.read().get(&uuid_of(k, 1)).is_none(),
            "deleted doc ({k},1) RESURRECTED after recovery"
        );
    }
    e2.close().unwrap();
}
