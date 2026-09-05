//! Phase 1 integration tests — compaction correctness (TEST 11).
//!
//! Built directly on the storage layer with hand-written SSTables so the file
//! layout is deterministic (DocumentStore::flush auto-compacts with the default
//! config, which would make engine-level file choreography flaky).

mod common;

use common::*;
use std::collections::HashMap;

fn live_record(idx: i64, ts: i64) -> attentiondb_storage::Record {
    let mut r = attentiondb_storage::Record::new(HashMap::new());
    r.id = uuid::Uuid::from_u128(idx as u128);
    r.timestamp = ts;
    r.fields.insert("idx".to_string(), serde_json::json!(idx));
    r.fields
        .insert("body".to_string(), serde_json::json!(format!("body-{idx}")));
    r.k_vecs
        .insert("default".to_string(), one_hot(idx as usize % 8, 8));
    r.tags.push("collection:c".to_string());
    r
}

/// Write an SSTable whose FRAME timestamps are the record's logical timestamp
/// (the timestamp the storage merge actually orders by). Explicit stamps make
/// the file choreography fully deterministic.
fn write_sst(dir: &std::path::Path, name: &str, recs: &[attentiondb_storage::Record]) {
    std::fs::create_dir_all(dir).unwrap();
    let mut w = attentiondb_storage::SSTableWriter::new(&dir.join(name)).unwrap();
    for r in recs {
        w.append_with_timestamp(
            r.id.as_bytes().to_vec(),
            r.to_msgpack().unwrap(),
            r.timestamp,
        )
        .unwrap();
    }
    w.flush().unwrap();
}

fn tombstone_for(idx: i64, ts: i64) -> attentiondb_storage::Record {
    let mut t = attentiondb_storage::Record::new(HashMap::new());
    t.id = uuid::Uuid::from_u128(idx as u128);
    t.timestamp = ts;
    t.tags.push("__TOMBSTONE__".to_string());
    t
}

/// TEST 11 — full compaction: deleted data must never resurrect, and
/// tombstones are reclaimed exactly once when every SST is an input.
#[test]
fn t11_compaction_no_resurrect() {
    let dir = temp_db();
    let sst = attentiondb_storage::Catalog::sst_dir(dir.path());

    // 3 files of 20 live docs + 1 newer file holding 40 tombstones.
    let f1: Vec<_> = (0..20).map(|i| live_record(i, 1_000 + i)).collect();
    let f2: Vec<_> = (20..40).map(|i| live_record(i, 1_000 + i)).collect();
    let f3: Vec<_> = (40..60).map(|i| live_record(i, 1_000 + i)).collect();
    write_sst(&sst, "sstable_0001.sst", &f1);
    write_sst(&sst, "sstable_0002.sst", &f2);
    write_sst(&sst, "sstable_0003.sst", &f3);
    let tombs: Vec<_> = (0..40).map(|i| tombstone_for(i, 9_000 + i)).collect();
    write_sst(&sst, "sstable_0004.sst", &tombs);

    // Before compaction: 60 files' worth of keys, 40 shadowed by tombstones.
    let store = attentiondb_storage::DocumentStore::open_without_wal(sst.clone()).unwrap();
    assert_eq!(store.len(), 20);
    drop(store);

    // Full compaction merges EVERY file → tombstones reclaimed exactly once.
    let result = attentiondb_storage::compact_all(&sst).unwrap();
    assert!(
        result.is_some(),
        "full compaction should run (4 input files)"
    );
    let r = result.unwrap();
    assert_eq!(
        r.tombstones_removed, 40,
        "tombstones reclaimed exactly once"
    );

    // After: single merged file, 20 live docs, deleted stay deleted.
    let store = attentiondb_storage::DocumentStore::open_without_wal(sst.clone()).unwrap();
    assert_eq!(store.len(), 20);
    let all = store.list_all_records();
    for idx in 0..40i64 {
        assert!(
            !all.iter()
                .any(|rec| rec.fields.get("idx").and_then(|v| v.as_i64()) == Some(idx)),
            "deleted doc {idx} resurrected after full compaction"
        );
    }
    assert_eq!(all.len(), 20);
}

/// Incremental compaction retains tombstones (no early GC): merging only the
/// oldest file must NOT reclaim a tombstone living in a newer file, and the
/// delete must still win after the merge (timestamp-aware merge, INV-12).
#[test]
fn t11b_incremental_compaction_keeps_tombstones() {
    let dir = temp_db();
    let sst = attentiondb_storage::Catalog::sst_dir(dir.path());

    let f1: Vec<_> = (0..30).map(|i| live_record(i, 1_000 + i)).collect();
    write_sst(&sst, "sstable_0001.sst", &f1);
    // newer file: tombstone for idx 0 only
    write_sst(&sst, "sstable_0002.sst", &[tombstone_for(0, 9_000)]);

    // Merge ONLY the oldest file (subset): tombstone must survive untouched.
    let cfg = attentiondb_storage::CompactionConfig {
        min_files_to_compact: 2,
        max_files_per_run: 1,
    };
    let result = attentiondb_storage::compact(&sst, &cfg).unwrap();
    assert!(result.is_some(), "incremental compaction should run");
    assert_eq!(
        result.unwrap().tombstones_removed,
        0,
        "partial compaction must NOT reclaim tombstones"
    );

    // Post-merge read: the tombstone in the newer file still shadows idx 0.
    let store = attentiondb_storage::DocumentStore::open_without_wal(sst.clone()).unwrap();
    assert_eq!(store.len(), 29, "no resurrection after incremental merge");
    let all = store.list_all_records();
    assert!(!all
        .iter()
        .any(|rec| rec.fields.get("idx").and_then(|v| v.as_i64()) == Some(0)));
    assert_eq!(all.len(), 29);
}
