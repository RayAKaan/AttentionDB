//! INV-E9-HYGIENE regression: dead-version index retention must be bounded by
//! checkpoint-time hygiene (purge + deterministic rebuild), and the rebuild
//! must preserve retrieval + checker cleanliness (the recovery-path operation).

use attentiondb_core::engine::AttentionEngine;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;

const DIM: usize = 32;
const HEAD: &str = "h";

fn rec(idx: u32, ver: u64, tag: &str) -> Record {
    let mut fields = HashMap::new();
    fields.insert("idx".to_string(), serde_json::json!(idx));
    fields.insert("ver".to_string(), serde_json::json!(ver));
    fields.insert(
        "title".to_string(),
        serde_json::json!(format!("doc-{idx}-v{ver}-{tag}")),
    );
    let mut r = Record::new(fields);
    let mut h: u32 = 0x811C9DC5;
    for b in idx.to_le_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    let mut v = vec![0.0f32; DIM];
    v[(h as usize) % DIM] = 1.0;
    r.k_vecs.insert(HEAD.to_string(), v);
    r
}

fn uuid_for(idx: u32, ver: u64) -> String {
    format!("{:08x}-0000-0000-0000-{:012x}", idx, ver)
}

#[test]
fn e9_hygiene_multi_head_fresh_checkpoint_must_not_rebuild() {
    // E10 SCALE-DEFECT #1 regression: a multi-head collection with ZERO dead
    // entries (vstore == heads x mapped) must NOT trigger the deterministic
    // rebuild at checkpoint — the old head-count-blind ratio forced one
    // (measured 78-157 s per checkpoint at PH3E-SCALE-008/009).
    let dir = std::env::temp_dir().join(format!(
        "e9hygiene-mh-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
    e.create_collection("bench", DIM, &["h", "h2", "h3"])
        .unwrap();
    for i in 0..12_000u32 {
        let mut r = rec(i, 1, "a");
        r.id = uuid::Uuid::parse_str(&uuid_for(i, 1)).unwrap();
        // vector under every head
        let v = r.k_vecs.get(HEAD).cloned().unwrap();
        r.k_vecs.insert("h2".to_string(), v.clone());
        r.k_vecs.insert("h3".to_string(), v);
        e.insert_document("bench", r).unwrap();
    }
    let t0 = std::time::Instant::now();
    e.checkpoint().unwrap();
    let ckpt_ms = t0.elapsed().as_millis();
    let c = e.mem_census();
    let vstore: usize = c.collections.iter().map(|x| x.vector_store_len).sum();
    assert_eq!(vstore, 3 * 12_000, "vstore must equal heads x live");
    assert!(
        ckpt_ms < 30_000,
        "checkpoint with zero dead entries must NOT rebuild (took {ckpt_ms} ms)"
    );
    e.close().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn e9_hygiene_bounds_dead_index_retention_and_preserves_retrieval() {
    let dir = std::env::temp_dir().join(format!(
        "e9hygiene-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    // query = doc 0's own vector (a guaranteed exact match)
    let v = {
        let mut vv = vec![0.0f32; DIM];
        let mut h: u32 = 0x811C9DC5;
        for b in 0u32.to_le_bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(0x01000193);
        }
        vv[(h as usize) % DIM] = 1.0;
        vv
    };
    {
        let e = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        // 900 live docs, then heavy delete/reinsert + update churn WITHOUT any
        // checkpoint: dead index entries accumulate (vstore >> live).
        for i in 0..900u32 {
            let mut r = rec(i, 1, "a");
            r.id = uuid::Uuid::parse_str(&uuid_for(i, 1)).unwrap();
            e.insert_document("bench", r).unwrap();
        }
        {
            let hits0 = e.attend("bench", &[HEAD.to_string()], &v, 5).unwrap();
            let scanned = e.scan_filtered("bench", None, 10_000).unwrap().len();
            eprintln!("after load: attend hits={} scan={}", hits0.len(), scanned);
        }
        for cycle in 0..40u64 {
            for i in 0..900u32 {
                let old = uuid_for(i, cycle + 1);
                e.delete_document("bench", &old).unwrap();
                let mut r = rec(i, cycle + 2, "a");
                r.id = uuid::Uuid::parse_str(&uuid_for(i, cycle + 2)).unwrap();
                e.insert_document("bench", r).unwrap();
            }
        }
        let c_pre = e.mem_census();
        let vstore_pre: usize = c_pre.collections.iter().map(|c| c.vector_store_len).sum();
        assert!(
            vstore_pre > 3 * c_pre.mapper_uuid_to_u64,
            "precondition: dead entries must accumulate before hygiene (vstore={vstore_pre}, live={})",
            c_pre.mapper_uuid_to_u64
        );
        let hits_pre = e.attend("bench", &[HEAD.to_string()], &v, 5).unwrap();
        eprintln!("pre-checkpoint hits: {}", hits_pre.len());
        // checkpoint runs INV-E9-HYGIENE
        e.checkpoint().unwrap();
        let hits_mid = e.attend("bench", &[HEAD.to_string()], &v, 5).unwrap();
        eprintln!(
            "post-checkpoint hits: {} (retired={}, mapped={})",
            hits_mid.len(),
            {
                let c = e.mem_census();
                c.mapper_retired
            },
            {
                let c = e.mem_census();
                c.mapper_uuid_to_u64
            }
        );
        assert!(
            !hits_mid.is_empty(),
            "post-hygiene retrieval must restore the exact match (rebuild ran)"
        );
        let c_post = e.mem_census();
        let vstore_post: usize = c_post.collections.iter().map(|c| c.vector_store_len).sum();
        assert!(
            vstore_post <= c_post.mapper_uuid_to_u64,
            "post-checkpoint vstore ({vstore_post}) must equal mapped live ({})",
            c_post.mapper_uuid_to_u64
        );
        // retrieval still returns the live document for a query near doc 5's vector
        let mut v = vec![0.0f32; DIM];
        v[0] = 1.0;
        let hits = e.attend("bench", &[HEAD.to_string()], &v, 5).unwrap();
        eprintln!("post-hygiene hits: {}", hits.len());
        assert!(
            !hits.is_empty(),
            "retrieval must work after hygiene rebuild"
        );
        // every returned id maps to a live (non-retired) record
        let checker_clean = attentiondb_core::checker::check_engine(&e).is_empty();
        assert!(checker_clean, "checker must be clean after hygiene rebuild");
        e.close().unwrap();
    }
    // fresh-process recovery on the hygiene-managed dir: state intact
    let e2 = AttentionEngine::open_dir(&dir, Durability::Sync).unwrap();
    let scanned2 = e2.scan_filtered("bench", None, 10_000).unwrap().len();
    let hits2 = e2.attend("bench", &[HEAD.to_string()], &v, 5).unwrap();
    let c2 = e2.mem_census();
    eprintln!(
        "reopened: scan={scanned2} attend={} mapped={} retired={} vstore={} tags_of_first={:?}",
        hits2.len(),
        c2.mapper_uuid_to_u64,
        c2.mapper_retired,
        c2.collections
            .iter()
            .map(|c| c.vector_store_len)
            .sum::<usize>(),
        e2.document_store
            .read()
            .list_all_records()
            .first()
            .map(|r| r.tags.clone()),
    );
    assert_eq!(c2.mapper_uuid_to_u64, 900, "all live docs must survive");
    let _ = std::fs::remove_dir_all(&dir);
}
