//! Phase 1 integration tests — deletes, updates, upserts, cache (TEST 3, 4, 5, 16).

mod common;

use common::*;
use std::collections::HashMap;

fn dim() -> usize {
    16
}

/// TEST 3 — Delete persistence: deleted docs can never reappear, in any subsystem.
#[test]
fn t03_delete_persistence() {
    let dir = temp_db();
    let (victim_id, victim_uuid) = {
        let e = open_sync(dir.path());
        e.create_collection("c", dim(), &["default"]).unwrap();
        for i in 0..10 {
            e.insert_document("c", doc(i, &one_hot(i as usize, dim()), "deleteable body"))
                .unwrap();
        }
        let uuid = {
            let all = e.document_store.read().list_all_records();
            all.iter()
                .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(3))
                .unwrap()
                .id
        };
        let nid = e.id_mapper.read().uuid_to_id(&uuid).unwrap();
        assert!(e.delete_document("c", &uuid.to_string()).unwrap());
        // Immediately after delete: not in store, not in search (INV-3).
        assert!(e.document_store.read().get(&uuid).is_none());
        let r = e
            .attend("c", &["default".into()], &one_hot(3, dim()), 10)
            .unwrap();
        assert!(
            r.iter().all(|(id, _)| *id != nid),
            "retired id filtered from results"
        );
        // BM25 no longer returns it.
        let coll = e.get_collection("c").unwrap();
        let bm25_hits = coll.bm25.search("deleteable body", 20);
        assert!(!bm25_hits.iter().any(|(id, _)| *id == nid));
        (nid, uuid)
    };
    // Restart: still gone everywhere.
    let e = open_sync(dir.path());
    assert!(e.document_store.read().get(&victim_uuid).is_none());
    assert!(e.id_mapper.read().uuid_to_id(&victim_uuid).is_none());
    let r = e
        .attend("c", &["default".into()], &one_hot(3, dim()), 10)
        .unwrap();
    assert!(r.iter().all(|(id, _)| *id != victim_id));
    let coll = e.get_collection("c").unwrap();
    let bm25_hits = coll.bm25.search("deleteable body", 20);
    assert!(!bm25_hits.iter().any(|(id, _)| *id == victim_id));
    // checker sees no orphans
    let issues = attentiondb_core::checker::check_engine(&e);
    assert!(
        !issues.iter().any(|i| i.severity.is_error()),
        "checker: {issues:?}"
    );
}

/// TEST 4 — Update persistence: new vector searchable, old vector gone, after restart.
#[test]
fn t04_update_persistence() {
    let dir = temp_db();
    let updated_uuid = {
        let e = open_sync(dir.path());
        e.create_collection("c", dim(), &["default"]).unwrap();
        for i in 0..8 {
            e.insert_document("c", doc(i, &one_hot(i as usize, dim()), "original"))
                .unwrap();
        }
        let uuid = {
            let all = e.document_store.read().list_all_records();
            all.iter()
                .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(2))
                .unwrap()
                .id
        };
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(2));
        fields.insert("body".to_string(), serde_json::json!("UPDATED"));
        let mut kv = HashMap::new();
        kv.insert("default".to_string(), one_hot(15, dim())); // new direction
        e.update_document("c", &uuid.to_string(), fields, kv)
            .unwrap();
        e.close().unwrap();
        uuid
    };
    let e = open_sync(dir.path());
    // The update survived: new vector direction finds the UPDATED body (INV-4).
    // (The updated doc is the ONLY one with direction 15 → deterministic top-1.)
    let r = e
        .attend("c", &["default".into()], &one_hot(15, dim()), 2)
        .unwrap();
    assert!(!r.is_empty());
    let fields = e.get_document_fields(r[0].0);
    assert_eq!(fields.get("body").map(String::as_str), Some("UPDATED"));
    assert_eq!(fields.get("idx").map(String::as_str), Some("2"));
    // The old representation must not surface for that document: a new doc now
    // uniquely owns the old direction, so it wins top-1 deterministically.
    e.insert_document("c", doc(50, &one_hot(2, dim()), "new owner of direction 2"))
        .unwrap();
    let r2 = e
        .attend("c", &["default".into()], &one_hot(2, dim()), 3)
        .unwrap();
    assert!(!r2.is_empty());
    let top_fields = e.get_document_fields(r2[0].0);
    assert_eq!(
        top_fields.get("body").map(String::as_str),
        Some("new owner of direction 2"),
        "stale vector must never outrank the true owner of the direction"
    );
    let _ = updated_uuid;
}

/// TEST 5 — Upsert: exactly one logical document, ever.
#[test]
fn t05_upsert() {
    let dir = temp_db();
    let uuid = uuid::Uuid::new_v4();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", dim(), &["default"]).unwrap();
        // upsert new
        let mut f = HashMap::new();
        f.insert("v".to_string(), serde_json::json!("one"));
        let mut kv = HashMap::new();
        kv.insert("default".to_string(), one_hot(1, dim()));
        e.upsert_document("c", uuid, f, kv).unwrap();
        // upsert same document again
        let mut f2 = HashMap::new();
        f2.insert("v".to_string(), serde_json::json!("two"));
        let mut kv2 = HashMap::new();
        kv2.insert("default".to_string(), one_hot(2, dim()));
        e.upsert_document("c", uuid, f2, kv2).unwrap();
        assert_eq!(
            e.document_store.read().len(),
            1,
            "INV-5: one logical document"
        );
        e.close().unwrap();
    }
    let e = open_sync(dir.path());
    assert_eq!(e.document_store.read().len(), 1);
    let r = e
        .attend("c", &["default".into()], &one_hot(2, dim()), 5)
        .unwrap();
    assert_eq!(r.len(), 1);
    let fields = e.get_document_fields(r[0].0);
    assert_eq!(fields.get("v").map(String::as_str), Some("two"));
    // generated-id upsert (insert semantics) works too
    let mut f3 = HashMap::new();
    f3.insert("v".to_string(), serde_json::json!("three"));
    let mut kv3 = HashMap::new();
    kv3.insert("default".to_string(), one_hot(3, dim()));
    e.upsert_document("c", uuid::Uuid::new_v4(), f3, kv3)
        .unwrap();
    assert_eq!(e.document_store.read().len(), 2);
}

/// TEST 16 — Cache invalidation: cached reads always agree with logical state.
#[test]
fn t16_cache_invalidation() {
    let dir = temp_db();
    let uuid = uuid::Uuid::new_v4();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", dim(), &["default"]).unwrap();
        let mut f = HashMap::new();
        f.insert("body".to_string(), serde_json::json!("v1"));
        let mut kv = HashMap::new();
        kv.insert("default".to_string(), one_hot(1, dim()));
        e.insert_document("c", {
            let mut r = doc(1, &one_hot(1, dim()), "v1");
            r.id = uuid;
            r
        })
        .unwrap();
        // populate cache via a "cold" read path
        let _ = e.document_store.read().get_record(&uuid);
        // update → read (must see v2)
        let mut f2 = HashMap::new();
        f2.insert("body".to_string(), serde_json::json!("v2"));
        let mut kv2 = HashMap::new();
        kv2.insert("default".to_string(), one_hot(2, dim()));
        e.update_document("c", &uuid.to_string(), f2, kv2).unwrap();
        let rec = e.document_store.read().get_record(&uuid).unwrap();
        assert_eq!(rec.fields.get("body"), Some(&serde_json::json!("v2")));
        // delete → read (must be gone)
        assert!(e.delete_document("c", &uuid.to_string()).unwrap());
        assert!(e.document_store.read().get_record(&uuid).is_none());
        e.close().unwrap();
    }
    // restart → read (still gone)
    let e = open_sync(dir.path());
    assert!(e.document_store.read().get_record(&uuid).is_none());
    let _ = uuid; // silence
}
