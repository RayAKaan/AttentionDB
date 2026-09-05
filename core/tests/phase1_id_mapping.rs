//! Phase 1 integration tests — ID mapping durability (TEST 6).

mod common;

use common::*;
use std::collections::HashMap;

/// TEST 6 — ID mapping: insert 1000, restart, insert another 1000, restart,
/// verify all mappings; plus delete→restart→insert must never reuse ids.
#[test]
fn t06_id_mapping_1000() {
    let dir = temp_db();
    let dim = 8;
    {
        let e = open_sync(dir.path());
        e.create_collection("c", dim, &["default"]).unwrap();
        for i in 0i64..1000 {
            e.insert_document("c", doc(i, &one_hot(i as usize, dim), "x"))
                .unwrap();
        }
        let snap = e.id_mapper.read().snapshot();
        assert_eq!(snap.mappings.len(), 1000);
        e.close().unwrap();
    }
    {
        let e = open_sync(dir.path());
        assert_eq!(e.id_mapper.read().len(), 1000, "mappings survive restart");
        for i in 1000i64..2000 {
            e.insert_document("c", doc(i, &one_hot(i as usize, dim), "y"))
                .unwrap();
        }
        assert_eq!(e.id_mapper.read().len(), 2000);
        e.close().unwrap();
    }
    let e = open_sync(dir.path());
    let snap = e.id_mapper.read().snapshot();
    assert_eq!(snap.mappings.len(), 2000, "all 2000 mappings survive");
    // bijection: unique ids
    let mut ids: Vec<u64> = snap.mappings.iter().map(|(_, i)| *i).collect();
    ids.sort_unstable();
    let unique = ids.iter().zip(ids.iter().skip(1)).all(|(a, b)| a != b);
    assert!(unique, "no duplicate numeric ids");
    assert_eq!(ids.first().unwrap(), &1);
    assert_eq!(ids.last().unwrap(), &2000);
    assert_eq!(snap.next_id, 2001);
}

/// Deleted ids are never reused: insert → delete → restart → insert gets fresh ids.
#[test]
fn t06b_no_id_reuse_after_delete_restart() {
    let dir = temp_db();
    let before; // last allocator value seen before the restart
    let dim = 8;
    let retired_ids = {
        let e = open_sync(dir.path());
        e.create_collection("c", dim, &["default"]).unwrap();
        let mut ids = Vec::new();
        for i in 0..10 {
            ids.push(
                e.insert_document("c", doc(i, &one_hot(i as usize, dim), "x"))
                    .unwrap(),
            );
        }
        let mut retired = Vec::new();
        for id in &ids[..5] {
            if e.delete_document("c", id).unwrap() {
                let u = uuid::Uuid::parse_str(id).unwrap();
                retired.push(e.id_mapper.read().uuid_to_id(&u)); // None after retire
            }
        }
        assert!(retired.iter().all(|r| r.is_none()));
        e.close().unwrap();
        // numeric ids 1..=5 are now retired
        let snap = e.id_mapper.read().snapshot();
        snap.retired
    };
    let e = open_sync(dir.path());
    // retired set survived restart
    {
        let mapper = e.id_mapper.read();
        for id in 1..=5u64 {
            assert!(mapper.is_retired(id), "id {id} must stay retired");
        }
        before = mapper.next_id();
    }
    let new_id_str = e
        .insert_document("c", doc(77, &one_hot(7, dim), "fresh"))
        .unwrap();
    let new_id = e
        .id_mapper
        .read()
        .uuid_to_id(&uuid::Uuid::parse_str(&new_id_str).unwrap())
        .unwrap();
    assert!(new_id >= before, "fresh id beyond all previous ids");
    assert!(!retired_ids.is_empty());
}

/// Updates preserve identity: same uuid before/after update, fresh numeric id,
/// old numeric id retired and never reused.
#[test]
fn t06c_update_keeps_identity() {
    let dir = temp_db();
    let dim = 8;
    let e = open_sync(dir.path());
    e.create_collection("c", dim, &["default"]).unwrap();
    let id_str = e
        .insert_document("c", doc(1, &one_hot(1, dim), "a"))
        .unwrap();
    let u = uuid::Uuid::parse_str(&id_str).unwrap();
    let old_numeric = e.id_mapper.read().uuid_to_id(&u).unwrap();

    let mut f = HashMap::new();
    f.insert("v".to_string(), serde_json::json!("b"));
    let mut kv = HashMap::new();
    kv.insert("default".to_string(), one_hot(2, dim));
    e.update_document("c", &id_str, f, kv).unwrap();

    assert_eq!(e.id_mapper.read().uuid_to_id(&u).unwrap(), old_numeric + 1);
    assert!(e.id_mapper.read().is_retired(old_numeric));
    assert_eq!(e.document_store.read().len(), 1, "no duplicate documents");
}
