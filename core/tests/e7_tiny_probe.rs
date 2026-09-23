//! Decisive 3-doc probe: does attend() report doc ids with a systematic
//! offset? Insert e0/e1/e2 with distinct uuids; query each direction; the
//! returned id must be the matching doc's numeric id (verify via mapper).
use attentiondb_core::engine::AttentionEngine;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;

fn rec(idx: u32) -> Record {
    let mut f = HashMap::new();
    f.insert("idx".to_string(), serde_json::json!(idx));
    let mut r = Record::new(f);
    r.id = uuid::Uuid::from_u128(((9000u64 + idx as u64) as u128) << 64 | 1);
    let mut v = vec![0.0f32; 32];
    v[idx as usize] = 1.0;
    r.k_vecs.insert("h".to_string(), v);
    r
}

#[test]
fn attend_ids_match_mapper() {
    let dir = std::env::temp_dir().join("e7-tiny-probe");
    let _ = std::fs::remove_dir_all(&dir);
    let e = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
    e.create_collection("bench", 32, &["h"]).unwrap();
    for i in 0..3u32 {
        e.insert_document("bench", rec(i)).unwrap();
    }
    // mapper ground truth
    for i in 0..3u32 {
        let uuid = uuid::Uuid::from_u128(((9000u64 + i as u64) as u128) << 64 | 1);
        println!("doc idx={i} numeric={:?}", e.id_mapper.read().uuid_to_id(&uuid));
    }
    for qd in 0..3u32 {
        let mut v = vec![0.0f32; 32];
        v[qd as usize] = 1.0;
        let res = e.attend("bench", &["h".to_string()], &v, 3).unwrap();
        println!("query e{qd} -> {:?}", res);
    }
    e.close().unwrap();
}
