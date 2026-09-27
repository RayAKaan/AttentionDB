//! E7 investigation: two txns each [Delete U, Insert U same uuid] — live
//! apply succeeds; does fresh-process replay agree?

use attentiondb_core::engine::AttentionEngine;
use attentiondb_core::transaction::TxnOp;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;

fn vec_for(idx: u32) -> Vec<f32> {
    let mut v = vec![0.0f32; 32];
    v[(idx as usize) % 32] = 1.0;
    v
}
fn rec(idx: u32, v: u64, num: i64) -> Record {
    let mut f = HashMap::new();
    f.insert("idx".to_string(), serde_json::json!(idx));
    f.insert("num".to_string(), serde_json::json!(num));
    let mut r = Record::new(f);
    r.id = uuid::Uuid::from_u128(((idx as u128) << 64) | (v as u128));
    r.k_vecs.insert("h".to_string(), vec_for(idx));
    r
}

#[test]
fn replay_same_uuid_delete_reinsert_two_txns() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing_subscriber::filter::LevelFilter::ERROR)
        .with_test_writer()
        .try_init();
    let dir = std::env::temp_dir().join("e7-replay-probe");
    let _ = std::fs::remove_dir_all(&dir);
    {
        let e = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
        e.create_collection("bench", 32, &["h"]).unwrap();
        e.insert_document("bench", rec(6100, 0, 0)).unwrap();
        e.checkpoint().unwrap();
        for num in [100i64, 200] {
            let t = e.begin_transaction("bench");
            e.record_transaction_operation(t, TxnOp::Delete(uuid::Uuid::from_u128(6100u128 << 64)))
                .unwrap();
            e.record_transaction_operation(t, TxnOp::Insert(rec(6100, 0, num)))
                .unwrap();
            let ok = e.commit_transaction(t).unwrap();
            println!("commit num={num} -> {ok}");
        }
        let alpha = e.get_collection("bench").unwrap();
        println!("live total_vectors = {}", alpha.total_vectors());
        e.close().unwrap();
    }
    match AttentionEngine::open_dir(&dir, Durability::Sync) {
        Ok(e) => {
            println!("replay OPENED ok");
            e.close().unwrap();
        }
        Err(err) => {
            println!("replay REFUSED: {err}");
            // surface the underlying issues: replay manually via open_dir is
            // all-or-nothing, so re-run with the internal checker exposed
            // through a raw replay: use the debug hook if available.
        }
    }
}

/// Variant 2: NO clean close — drop the engine so the WAL retains the raw
/// TxnOp records; fresh open_dir must replay them (decode_op carries the
/// WAL numeric ids) and pass the post-recovery consistency checker.
/// This is the crash-recovery path E7y exercises.
#[test]
fn replay_same_uuid_delete_reinsert_wal_no_close() {
    let dir = std::env::temp_dir().join("e7-replay-probe-wal");
    let _ = std::fs::remove_dir_all(&dir);
    {
        let e = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
        e.create_collection("bench", 32, &["h"]).unwrap();
        e.insert_document("bench", rec(6100, 0, 0)).unwrap();
        e.checkpoint().unwrap();
        for num in [100i64, 200] {
            let t = e.begin_transaction("bench");
            e.record_transaction_operation(t, TxnOp::Delete(uuid::Uuid::from_u128(6100u128 << 64)))
                .unwrap();
            e.record_transaction_operation(t, TxnOp::Insert(rec(6100, 0, num)))
                .unwrap();
            let ok = e.commit_transaction(t).unwrap();
            assert!(ok, "commit {num} must succeed");
        }
        // Intentionally NO close(): simulate crash — WAL keeps TxnOps.
        drop(e);
    }
    match AttentionEngine::open_dir(&dir, Durability::Sync) {
        Ok(e) => {
            let alpha = e.get_collection("bench").unwrap();
            let total = alpha.total_vectors();
            e.close().unwrap();
            assert_eq!(total, 1, "exactly the reinserted doc must survive");
        }
        Err(err) => panic!("WAL replay refused after committed same-uuid txns: {err}"),
    }
}
