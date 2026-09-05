//! THE GOLDEN LIFECYCLE (§36) — the Phase 1 centerpiece.
//!
//! CREATE → INSERT → SEARCH → UPDATE → DELETE → COMMIT → CRASH → RESTART →
//! RECOVER → SEARCH must produce exactly the expected logical state, for both a
//! clean shutdown AND an unclean process termination, and the whole scenario
//! must also survive checkpoint, compaction, backup and restore.

mod common;

use common::*;
use std::collections::HashMap;

fn build_database(dir: &std::path::Path, crash: bool) {
    let e = open_sync(dir);
    // 2. create collection with custom configuration
    let settings = attentiondb_hnsw::CollectionSettings {
        ef_search: 128,
        ..Default::default()
    };
    e.create_collection_with_settings("golden", 16, &["default", "semantic"], settings)
        .unwrap();
    e.create_collection("other", 16, &["default"]).unwrap();

    // 3. insert documents
    for i in 0..40 {
        let mut r = doc(i, &one_hot(i as usize % 16, 16), "golden body");
        r.k_vecs
            .insert("semantic".to_string(), one_hot((i as usize + 5) % 16, 16));
        e.insert_document("golden", r).unwrap();
    }
    for i in 0..10 {
        e.insert_document(
            "other",
            doc(1000 + i, &one_hot(i as usize % 16, 16), "other body"),
        )
        .unwrap();
    }

    // 4. search
    let r = e
        .attend("golden", &["default".into()], &one_hot(7, 16), 3)
        .unwrap();
    assert_eq!(r.len(), 3);

    // 5. update documents
    let all = e.document_store.read().list_all_records();
    let upd_target = all
        .iter()
        .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(10))
        .unwrap()
        .id
        .to_string();
    let mut f = HashMap::new();
    f.insert("idx".to_string(), serde_json::json!(10));
    f.insert("body".to_string(), serde_json::json!("UPDATED GOLDEN"));
    let mut kv = HashMap::new();
    kv.insert("default".to_string(), one_hot(9, 16));
    kv.insert("semantic".to_string(), one_hot(2, 16));
    e.update_document("golden", &upd_target, f, kv).unwrap();

    // 6. delete documents
    for idx in [1i64, 2, 3] {
        let u = all
            .iter()
            .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(idx))
            .unwrap()
            .id
            .to_string();
        assert!(e.delete_document("golden", &u).unwrap());
    }

    // 7. insert more documents
    for i in 40..50 {
        e.insert_document(
            "golden",
            doc(i, &one_hot(i as usize % 16, 16), "late golden"),
        )
        .unwrap();
    }

    // 8. multi-operation transaction: insert 2 + delete 1
    let txn_victim = all
        .iter()
        .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(20))
        .unwrap()
        .id;
    let txn = e.begin_transaction("golden");
    for i in [50i64, 51] {
        e.record_transaction_operation(
            txn,
            attentiondb_core::TxnOp::Insert(doc(i, &one_hot(i as usize % 16, 16), "txn insert")),
        )
        .unwrap();
    }
    e.record_transaction_operation(txn, attentiondb_core::TxnOp::Delete(txn_victim))
        .unwrap();
    assert!(e.commit_transaction(txn).unwrap());

    // 9. checkpoint  10. compact
    e.checkpoint().unwrap();
    let sst = attentiondb_storage::Catalog::sst_dir(dir);
    let _ = attentiondb_storage::compact_all(&sst);

    if crash {
        // 11. UNCLEAN termination: no drop handlers, no flush, no checkpoint.
        std::mem::forget(e);
    } else {
        // 11. clean shutdown
        e.close().unwrap();
    }
}

fn verify_database(dir: &std::path::Path, phase: &str) {
    let e = open_sync(dir);

    // 12/13. search works and finds the updated document (INV-4).
    // NOTE: the updated doc's vector one_hot(9) is exactly shared by docs 9 & 25
    // (3-way tie at similarity 1.0), so top-1 identity is arbitrary — but ALL
    // exact matches must appear in the top-10, deterministically.
    let r = e
        .attend("golden", &["default".into()], &one_hot(9, 16), 10)
        .unwrap();
    assert!(!r.is_empty(), "{phase}: search failed");
    let bodies: Vec<String> = r
        .iter()
        .map(|(id, _)| e.get_document_fields(*id))
        .filter_map(|f| f.get("body").cloned())
        .collect();
    assert!(
        bodies.iter().any(|b| b == "UPDATED GOLDEN"),
        "{phase}: updated doc not in top-10 for its exact vector; got {bodies:?}"
    );

    // 14. verify documents: exact expected count
    // 40 - 3 deleted - 1 txn-deleted + 10 late (40..50) + 2 txn = 48
    let all = e.document_store.read().list_all_records();
    let golden: Vec<_> = all
        .iter()
        .filter(|r| r.tags.contains(&"collection:golden".to_string()))
        .collect();
    assert_eq!(golden.len(), 48, "{phase}: golden collection count");
    let other = all
        .iter()
        .filter(|r| r.tags.contains(&"collection:other".to_string()))
        .count();
    assert_eq!(other, 10, "{phase}: other collection count");

    // 15. deleted documents absent (INV-3)
    for idx in [1i64, 2, 3, 20] {
        assert!(
            !golden
                .iter()
                .any(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(idx)),
            "{phase}: deleted doc {idx} resurrected"
        );
    }
    // late + txn inserts present
    for idx in [40i64, 49, 50, 51] {
        assert!(
            golden
                .iter()
                .any(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(idx)),
            "{phase}: inserted doc {idx} missing"
        );
    }
    // 16. updated documents contain the new state (already checked via search) —
    // and their identity is stable: exactly one doc with idx=10
    assert_eq!(
        golden
            .iter()
            .filter(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(10))
            .count(),
        1,
        "{phase}: update created duplicate"
    );

    // 17. collection configuration survived (INV-7/8)
    let s = e.get_collection("golden").unwrap().settings.read().clone();
    assert_eq!(s.ef_search, 128, "{phase}: settings lost");
    assert_eq!(
        e.get_collection("golden").unwrap().head_count(),
        2,
        "{phase}: heads lost"
    );

    // 18. HNSW: every head fully populated (INV-2)
    let coll = e.get_collection("golden").unwrap();
    for head in coll.list_heads() {
        let idx = coll.head_manager.read().get_head(&head).unwrap();
        // late (40..50) + txn docs only carry the "default" vector:
        // default = 48, semantic = 48 - 12 = 36.
        let expected = if head == "semantic" { 36 } else { 48 };
        assert_eq!(
            idx.read().len(),
            expected,
            "{phase}: head '{head}' incomplete"
        );
    }

    // 19. BM25 works (deterministically rebuilt)
    let bm25_hits = coll.bm25.search("UPDATED", 5);
    assert!(!bm25_hits.is_empty(), "{phase}: BM25 lost updated doc");

    // 20. consistency checker
    let issues = attentiondb_core::checker::check_engine(&e);
    let errors: Vec<_> = issues.iter().filter(|i| i.severity.is_error()).collect();
    assert!(errors.is_empty(), "{phase}: checker errors: {errors:?}");
}

/// 21. backup → 22. restore into a new directory → repeat verification.
#[test]
fn golden_lifecycle_clean_shutdown() {
    let dir = temp_db();
    build_database(dir.path(), false);
    verify_database(dir.path(), "clean/restart");

    let backup = temp_db();
    {
        let e = open_sync(dir.path());
        e.backup_to(backup.path()).unwrap();
    }
    let restored = temp_db();
    attentiondb_core::backup::restore_backup(backup.path(), restored.path()).unwrap();
    verify_database(restored.path(), "clean/restore");
}

/// The same scenario with an unclean termination (SIGKILL-equivalent).
#[test]
fn golden_lifecycle_unclean_termination() {
    let dir = temp_db();
    build_database(dir.path(), true);
    verify_database(dir.path(), "crash/restart");

    // and the recovered database keeps working: mutate, restart, verify again
    {
        let e = open_sync(dir.path());
        e.insert_document("golden", doc(60, &one_hot(12, 16), "post-crash"))
            .unwrap();
        // NOTE: resolve the id FIRST — the document_store read guard must never
        // be held across an engine mutation call (delete takes the write lock;
        // parking_lot is non-reentrant → same-thread read→write deadlocks).
        let u0 = e
            .document_store
            .read()
            .list_all_records()
            .iter()
            .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(0))
            .unwrap()
            .id
            .to_string();
        assert!(e.delete_document("golden", &u0).unwrap());
        e.close().unwrap();
    }
    let e = open_sync(dir.path());
    let all = e.document_store.read().list_all_records();
    assert!(all
        .iter()
        .any(|r| r.fields.get("body").and_then(|v| v.as_str()) == Some("post-crash")));
    assert!(!all
        .iter()
        .any(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(0)));
}
