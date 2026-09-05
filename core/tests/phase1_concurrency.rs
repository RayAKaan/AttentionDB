//! Phase 1 integration tests — concurrency (TEST 17): concurrent writers must
//! leave a consistent, restartable state; conflicting semantics = last committed
//! write wins (docs/consistency-model.md §4).

mod common;

use common::*;

const THREADS: usize = 8;
const OPS: usize = 120;
const TOTAL_EXPECTED: usize = THREADS * OPS;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn t17_concurrent_writers() {
    let dir = temp_db();
    let dim = 8;
    {
        let e = Arc::new(open_sync(dir.path()));
        e.create_collection("c", dim, &["default"]).unwrap();

        let mut handles = Vec::new();
        for t in 0..THREADS {
            let e = e.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..OPS {
                    let idx = (t * OPS + i) as i64;
                    let r = doc(idx, &one_hot(idx as usize % dim, dim), "concurrent");
                    e.insert_document("c", r).unwrap();
                    // Every 10th op: an update + a delete on the doc from another
                    // thread's range → deterministic last-write-wins conflicts.
                    if i % 10 == 0 && idx >= THREADS as i64 {
                        let victim_idx = idx - THREADS as i64;
                        let all = e.document_store.read().list_all_records();
                        if let Some(victim) = all.iter().find(|r| {
                            r.fields.get("idx").and_then(|v| v.as_i64()) == Some(victim_idx)
                        }) {
                            let vid = victim.id.to_string();
                            drop(all);
                            if i % 20 == 0 {
                                let mut f = HashMap::new();
                                f.insert("idx".to_string(), serde_json::json!(victim_idx));
                                f.insert(
                                    "body".to_string(),
                                    serde_json::json!("updated by writer"),
                                );
                                let mut kv = HashMap::new();
                                kv.insert(
                                    "default".to_string(),
                                    one_hot(victim_idx as usize % dim, dim),
                                );
                                let _ = e.update_document("c", &vid, f, kv);
                            } else {
                                let _ = e.delete_document("c", &vid);
                            }
                        }
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("writer thread panicked");
        }
        // engine dropped without close (unclean) — state must still recover
        std::mem::forget(e);
    }

    let e = open_sync(dir.path());
    // restart + consistency checker (INV-1/2/6)
    let issues = attentiondb_core::checker::check_engine(&e);
    let errors: Vec<_> = issues.iter().filter(|i| i.severity.is_error()).collect();
    assert!(
        errors.is_empty(),
        "checker errors after concurrent writes: {errors:?}"
    );

    // every idx appears at most once (INV-5); total count is coherent
    let all = e.document_store.read().list_all_records();
    let mut idxs: Vec<i64> = all
        .iter()
        .filter_map(|r| r.fields.get("idx").and_then(|v| v.as_i64()))
        .collect();
    idxs.sort_unstable();
    let dupes = idxs
        .iter()
        .zip(idxs.iter().skip(1))
        .filter(|(a, b)| a == b)
        .count();
    assert_eq!(dupes, 0, "no duplicate logical documents after concurrency");
    // Each thread executes 6 successful deletes (i ∈ {10,30,50,70,90,110}) on a
    // doc it inserted itself one iteration earlier — so they always hit:
    // 8 × 120 − 8 × 6 = 912. Updates don't change the count (identity remap).
    assert_eq!(all.len(), TOTAL_EXPECTED - THREADS * 6);

    // inserts keep working after the storm + restart
    e.insert_document("c", doc(99999, &one_hot(1, dim), "after"))
        .unwrap();
    e.close().unwrap();
    let e = open_sync(dir.path());
    assert_eq!(
        e.document_store.read().len(),
        TOTAL_EXPECTED - THREADS * 6 + 1
    );
}

/// Readers run concurrently with a writer without corrupting state.
#[test]
fn t17b_readers_during_writes() {
    let dir = temp_db();
    let dim = 8;
    let e = Arc::new(open_sync(dir.path()));
    e.create_collection("c", dim, &["default"]).unwrap();
    for i in 0..50 {
        e.insert_document("c", doc(i, &one_hot(i as usize % dim, dim), "seed"))
            .unwrap();
    }

    let writer = {
        let e = e.clone();
        std::thread::spawn(move || {
            for i in 50..150 {
                e.insert_document("c", doc(i, &one_hot(i as usize % dim, dim), "writer"))
                    .unwrap();
            }
        })
    };
    let reader = {
        let e = e.clone();
        std::thread::spawn(move || {
            for _ in 0..200 {
                let r = e.attend("c", &["default".into()], &one_hot(3, dim), 5);
                assert!(r.is_ok(), "reads must not fail during writes");
            }
        })
    };
    writer.join().unwrap();
    reader.join().unwrap();
    assert_eq!(e.document_store.read().len(), 150);
}
