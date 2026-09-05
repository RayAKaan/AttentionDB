//! Phase 1 property test — random mutation sequences against a reference model
//! (§27). The model tracks logical document state only; periodically the engine
//! is restarted / checkpointed / compacted and compared against the model.

mod common;

use common::*;
use std::collections::HashMap;

const DIM: usize = 8;
const OPS: usize = 600;

#[test]
fn property_random_ops_match_reference_model() {
    let dir = temp_db();
    let mut rng = Rng::new(0xDEADBEEF);
    // reference model: idx -> (vector, body)
    let mut model: std::collections::HashMap<i64, (Vec<f32>, String)> = Default::default();
    // idx -> uuid string (engine identity)
    let mut uuids: std::collections::HashMap<i64, String> = Default::default();
    let mut next_idx: i64 = 0;

    {
        let e = open_sync(dir.path());
        e.create_collection("c", DIM, &["default"]).unwrap();

        for op_i in 0..OPS {
            // pick an operation
            let roll = rng.below(100);
            if roll < 45 || model.is_empty() {
                // INSERT
                let idx = next_idx;
                next_idx += 1;
                let vec = one_hot(idx as usize % DIM, DIM);
                let body = format!("body-{idx}-{op_i}");
                let r = doc(idx, &vec, &body);
                let id = e.insert_document("c", r).unwrap();
                model.insert(idx, (vec.clone(), body));
                uuids.insert(idx, id);
            } else if roll < 65 {
                // UPDATE existing
                let keys: Vec<i64> = model.keys().copied().collect();
                let idx = keys[rng.below(keys.len() as u64) as usize];
                let vec = one_hot((idx as usize + 3) % DIM, DIM);
                let body = format!("upd-{idx}-{op_i}");
                let mut f = HashMap::new();
                f.insert("idx".to_string(), serde_json::json!(idx));
                f.insert("body".to_string(), serde_json::json!(body));
                let mut kv = HashMap::new();
                kv.insert("default".to_string(), vec.clone());
                e.update_document("c", uuids[&idx].as_str(), f, kv).unwrap();
                model.insert(idx, (vec.clone(), body));
            } else if roll < 80 {
                // UPSERT: half existing, half new
                let (idx, exists) = if model.is_empty() || rng.below(2) == 0 {
                    let idx = next_idx;
                    next_idx += 1;
                    (idx, false)
                } else {
                    let keys: Vec<i64> = model.keys().copied().collect();
                    (keys[rng.below(keys.len() as u64) as usize], true)
                };
                let vec = one_hot((idx as usize + 1) % DIM, DIM);
                let body = format!("upsert-{idx}-{op_i}");
                let uuid = if exists {
                    uuids[&idx].clone()
                } else {
                    uuid::Uuid::new_v4().to_string()
                };
                let mut f = HashMap::new();
                f.insert("idx".to_string(), serde_json::json!(idx));
                f.insert("body".to_string(), serde_json::json!(body));
                let mut kv = HashMap::new();
                kv.insert("default".to_string(), vec.clone());
                let got = e
                    .upsert_document("c", uuid::Uuid::parse_str(&uuid).unwrap(), f, kv)
                    .unwrap();
                if !exists {
                    uuids.insert(idx, got);
                }
                model.insert(idx, (vec.clone(), body));
            } else {
                // DELETE
                let keys: Vec<i64> = model.keys().copied().collect();
                let idx = keys[rng.below(keys.len() as u64) as usize];
                assert!(e.delete_document("c", uuids[&idx].as_str()).unwrap());
                model.remove(&idx);
                uuids.remove(&idx);
            }

            // periodic maintenance + full model comparison
            if op_i % 120 == 119 {
                e.checkpoint().unwrap();
            }
            if op_i % 200 == 199 {
                e.document_store.write().flush().unwrap();
            }
            verify_against_model(&e, &model, &uuids);
        }
        e.close().unwrap();
    }

    // restart: full comparison again (INV-10)
    let e = open_sync(dir.path());
    verify_against_model(&e, &model, &uuids);
    let issues = attentiondb_core::checker::check_engine(&e);
    assert!(
        !issues.iter().any(|i| i.severity.is_error()),
        "checker: {issues:?}"
    );
}

fn verify_against_model(
    e: &attentiondb_core::AttentionEngine,
    model: &std::collections::HashMap<i64, (Vec<f32>, String)>,
    uuids: &std::collections::HashMap<i64, String>,
) {
    // every model doc exists with the exact body; every engine doc is in the model
    let all = e.document_store.read().list_all_records();
    let mut seen: std::collections::HashMap<i64, ()> = Default::default();
    for r in &all {
        let idx = r
            .fields
            .get("idx")
            .and_then(|v| v.as_i64())
            .expect("idx field");
        let (vec, body) = &model[&idx];
        assert_eq!(
            r.fields.get("body").and_then(|v| v.as_str()),
            Some(body.as_str()),
            "doc {idx} body diverged"
        );
        assert_eq!(&r.k_vecs["default"], vec, "doc {idx} vector diverged");
        seen.insert(idx, ());
    }
    assert_eq!(
        seen.len(),
        model.len(),
        "engine doc count diverged from model"
    );
    // deleted docs are truly gone
    for (idx, u) in uuids {
        if !model.contains_key(idx) {
            let uu = uuid::Uuid::parse_str(u).unwrap();
            assert!(
                e.document_store.read().get(&uu).is_none(),
                "doc {idx} was deleted but exists"
            );
        }
    }
}
