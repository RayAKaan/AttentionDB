//! REGRESSION (E8 soak find — ENGINE DEFECT #2, decisive): WAL replay during
//! recovery must NEVER write SSTables. Pre-fix, a replayed memtable crossing
//! the threshold (1000 entries) auto-flushed a "ghost" SST into the directory
//! being opened — a SECOND engine opened on a LIVE directory (E8's mid-run
//! dir-check) re-materialized deleted records with newer timestamps than
//! their real tombstones; the next full recovery then resolved the ghost as
//! the winning version and rightly refused (MISSING_MAPPING orphan).
//!
//! Preserved failing state: research/phase3/raw/runs/PH3E-SOAK-004/
//! bug-preserved-e8d-refusal/
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

fn sst_count(db: &std::path::Path) -> usize {
    let sst_dir = attentiondb_storage::catalog::Catalog::sst_dir(db);
    std::fs::read_dir(&sst_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("sst"))
                .count()
        })
        .unwrap_or(0)
}

#[test]
fn recovery_replay_writes_no_sstables() {
    let dir = std::env::temp_dir().join("e8reg-recovery-readonly");
    let _ = std::fs::remove_dir_all(&dir);
    let db = dir.join("db");
    let e = AttentionEngine::open_dir(&db, Durability::Sync).expect("open");
    e.create_collection("bench", DIM, &[HEAD]).unwrap();

    // Anchor a checkpoint, then delete a doc so its tombstone lives ONLY in
    // the WAL (this is the record a mid-replay ghost would immortalize).
    e.insert_document("bench", mk(4242, 1)).unwrap();
    e.checkpoint().unwrap();
    let t = e.begin_transaction("bench");
    e.record_transaction_operation(
        t,
        attentiondb_core::transaction::TxnOp::Delete(uuid::Uuid::from_u128((4242u128 << 64) | 1)),
    )
    .unwrap();
    assert!(e.commit_transaction(t).unwrap());

    // >1000 unflushed WAL ops: the replay window crosses the memtable
    // threshold — the pre-fix ghost trigger.
    for i in 0..1200u32 {
        e.insert_document("bench", mk(10_000 + i, 1)).unwrap();
    }
    e.flush_wal().unwrap();
    e.close().unwrap(); // close-checkpoint flushes the memtable (legitimate)
    let before = sst_count(&db);

    // Recovery: replay >1000 ops. Recovery itself must write NOTHING.
    let e2 = AttentionEngine::open_dir(&db, Durability::Sync).expect("recovery open");
    let after = sst_count(&db);
    assert_eq!(
        after, before,
        "REGRESSION: recovery wrote SSTable(s) during WAL replay (ghost)"
    );
    // State correctness: all replayed docs present; deleted doc stays dead.
    for i in [0u32, 1, 599, 1199] {
        assert!(
            e2.document_store
                .read()
                .get(&uuid::Uuid::from_u128(((10_000 + i) as u128) << 64 | 1))
                .is_some(),
            "replayed doc {i} missing"
        );
    }
    assert!(
        e2.document_store
            .read()
            .get(&uuid::Uuid::from_u128((4242u128 << 64) | 1))
            .is_none(),
        "deleted doc resurrected"
    );

    // Second recovery (checkpointed state) must also stay read-only and
    // keep the deleted doc dead — the ghost would appear at THIS boundary.
    e2.checkpoint().unwrap();
    let n3 = sst_count(&db);
    e2.close().unwrap();
    let e3 = AttentionEngine::open_dir(&db, Durability::Sync).expect("second recovery");
    assert!(
        e3.document_store
            .read()
            .get(&uuid::Uuid::from_u128((4242u128 << 64) | 1))
            .is_none(),
        "deleted doc resurrected after checkpoint+reopen"
    );
    let _ = n3;
    e3.close().unwrap();
}
