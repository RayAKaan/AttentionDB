//! Phase 1 integration tests — transactions (TEST 10) via real subprocess
//! crashes, plus compaction crash behavior (TEST 12) and in-process scenarios.

mod common;

use common::*;

fn helper_bin() -> &'static str {
    env!("CARGO_BIN_EXE_attentiondb-crash-helper")
}

fn run_helper(mode: &str, db: &std::path::Path, n: usize) -> Vec<u8> {
    let out = std::process::Command::new(helper_bin())
        .args([mode, db.to_str().unwrap(), &n.to_string()])
        .output()
        .expect("failed to spawn crash helper");
    out.stdout
}

/// TEST 10 — Transaction atomicity under crash: committed-or-nothing.
#[test]
fn t10_txn_commit_then_crash_is_atomic() {
    let dir = temp_db();
    // helper: create collection, commit txn of 10 inserts, SIGABRT immediately
    let _ = run_helper("txn-commit-abort", dir.path(), 10);
    let e = open_sync(dir.path());
    // The commit is durable → all 10 applied; never a partial set (INV-10).
    assert_eq!(
        e.document_store.read().len(),
        10,
        "committed txn applies fully after crash"
    );
    let coll = e.get_collection("c1").unwrap();
    assert_eq!(coll.total_vectors(), 10);
}

/// TEST 10b — Staged-but-uncommitted transaction: crash discards it entirely.
#[test]
fn t10b_txn_uncommitted_never_applies() {
    let dir = temp_db();
    let _ = run_helper("txn-staged-abort", dir.path(), 10);
    let e = open_sync(&dir_path(&dir));
    assert_eq!(
        e.document_store.read().len(),
        0,
        "uncommitted txn must leave no trace"
    );
    // collection itself was committed pre-abort and must exist
    assert_eq!(e.list_collections(), vec!["c1".to_string()]);
}

fn dir_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
    dir.path().to_path_buf()
}

/// Inserts survive SIGKILL-equivalent abort with Sync durability.
#[test]
fn t10c_insert_then_abort() {
    let dir = temp_db();
    let _ = run_helper("insert-abort", dir.path(), 25);
    let e = open_sync(&dir_path(&dir));
    assert_eq!(e.document_store.read().len(), 25);
    assert_eq!(e.get_collection("c1").unwrap().total_vectors(), 25);
}

/// Checkpoint followed by more writes, then abort: checkpoint state + WAL tail
/// both recover (§13 window coverage).
#[test]
fn t10d_checkpoint_then_abort() {
    let dir = temp_db();
    let _ = run_helper("insert-checkpoint-abort", dir.path(), 15);
    let e = open_sync(&dir_path(&dir));
    assert_eq!(e.document_store.read().len(), 30);
    // deletes and updates after recovery-from-abort
    let ids: Vec<String> = {
        let all = e.document_store.read().list_all_records();
        all.into_iter().map(|r| r.id.to_string()).collect()
    };
    assert!(e.delete_document("c1", &ids[0]).unwrap());
    e.close().unwrap();
    let e = open_sync(&dir_path(&dir));
    assert_eq!(e.document_store.read().len(), 29);
}

/// Delete + update under abort: semantic survival across true process crashes.
#[test]
fn t10e_delete_update_abort() {
    let dir = temp_db();
    let _ = run_helper("insert-abort", dir.path(), 10);
    let _ = run_helper("delete-abort", dir.path(), 3);
    let e = open_sync(&dir_path(&dir));
    assert_eq!(
        e.document_store.read().len(),
        7,
        "3 deleted docs stay deleted"
    );
    let _ = run_helper("update-abort", dir.path(), 4);
    let e2 = open_sync(&dir_path(&dir));
    // 4 docs now carry UPDATED BODY
    let updated = {
        let all = e2.document_store.read().list_all_records();
        all.iter()
            .filter(|r| r.fields.get("body").and_then(|v| v.as_str()) == Some("UPDATED BODY"))
            .count()
    };
    assert_eq!(updated, 4, "updates survived the crash exactly once each");
}

/// TEST 12 — Crash during compaction (file-level windows):
/// (a) a leftover `.tmp` from an interrupted flush is discarded;
/// (b) a completed compacted output coexisting with un-deleted inputs
///     (crash between rename and cleanup) reads back correctly.
#[test]
fn t12_crash_during_compaction() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 8, &["default"]).unwrap();
        for i in 0..40 {
            e.insert_document("c", doc(i, &one_hot(i as usize, 8), "x"))
                .unwrap();
        }
        for i in 0..10 {
            let all = e.document_store.read().list_all_records();
            let u = all
                .iter()
                .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(i))
                .unwrap()
                .id;
            e.delete_document("c", &u.to_string()).unwrap();
        }
        e.checkpoint().unwrap();
        e.close().unwrap();
    }
    // (a) leftover tmp file
    std::fs::write(
        dir.path().join("sst").join("sstable_ interrupted.tmp"),
        b"junk",
    )
    .unwrap();
    {
        let e = open_sync(dir.path());
        assert_eq!(
            e.document_store.read().len(),
            30,
            "tmp ignored, tombstones applied"
        );
    }
    // (b) simulate crash between compaction output install and input cleanup:
    // create an extra copy of an SST with a stale (older) record version.
    let sst_dir = dir.path().join("sst");
    let ssts: Vec<_> = std::fs::read_dir(&sst_dir)
        .unwrap()
        .filter_map(|x| x.ok())
        .map(|x| x.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("sst"))
        .collect();
    assert!(!ssts.is_empty());
    std::fs::copy(&ssts[0], sst_dir.join("sstable_9999999999999999999.sst")).unwrap();
    let e = open_sync(dir.path());
    assert_eq!(
        e.document_store.read().len(),
        30,
        "duplicate/stale SST copies must not resurrect or corrupt state"
    );
}
