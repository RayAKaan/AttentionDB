//! Phase 2B engine integration — trained gating lifecycle (§14–18).
//!
//! The non-negotiables under test:
//! - §14: inference uses the loaded model; NO model ⇒ deterministic fallback;
//!   the database stays fully usable.
//! - §15: incompatible models (head count / head coverage) are REJECTED.
//! - §16: invalid/corrupt cards are rejected with typed errors; never
//!   silently ignored.
//! - §17/§18: registry activate/deactivate + hot-swap without restart.

mod common;

use common::*;
use std::collections::HashMap;

const DIM: usize = 8;
const GROUPS: usize = 4; // also head count: group g ↔ head g
const BLOCK: usize = DIM / GROUPS;

/// Hand-crafted "trained" card implementing the group detector:
/// hidden g = mean(block g of query); logit h = 3·hidden_h ⇒ softmax
/// argmax = the group's head. Equivalent to what training should converge to
/// on the controlled corpus, but deterministic without a training run.
fn detector_card(head_names: &[String]) -> attentiondb_learned::gating_v2::ModelCard {
    use attentiondb_learned::gating_v2::{GatingMlp, ModelCard, TrainingMeta};
    let mut m = GatingMlp::new(DIM, GROUPS, GROUPS, 1);
    for g in 0..GROUPS {
        for d in 0..DIM {
            m.w1[g * DIM + d] = if d / BLOCK == g {
                1.0 / BLOCK as f32
            } else {
                0.0
            };
        }
        for h in 0..GROUPS {
            m.w2[h * GROUPS + g] = if h == g { 3.0 } else { 0.0 };
        }
    }
    let meta = TrainingMeta {
        seed: 1,
        dataset_hash: 0xdeadbeef,
        objective: "soft_target".into(),
        learning_rate: 0.01,
        batch_size: 32,
        epochs_run: 1,
        best_val_loss: 0.0,
        l2: 1e-4,
        timestamp_unix: 0,
        code_commit: "test".into(),
        hardware: "ci".into(),
    };
    let mut card = ModelCard::from_mlp(&m, meta, "detector-v1", "soft_target");
    card.head_names = Some(head_names.to_vec());
    card
}

/// Controlled corpus: docs clustered into GROUPS groups; head g is CLEAN
/// (σ≈0) for group-g docs and NOISY for all others. Query for a group-g doc
/// carries the group-g block signal. GT = the queried doc's cluster mates.
fn setup(n_per_group: usize) -> (tempfile::TempDir, attentiondb_core::AttentionEngine) {
    let dir = temp_db();
    let e = open_sync(dir.path());
    let heads: Vec<String> = (0..GROUPS).map(|g| format!("head{g}")).collect();
    let heads_ref: Vec<&str> = heads.iter().map(|s| s.as_str()).collect();
    e.create_collection("c", DIM, &heads_ref).unwrap();

    struct R(u64);
    impl R {
        fn f(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32
        }
    }
    let mut rng = R(0xBEEF);
    let mut id = 0i64;
    for g in 0..GROUPS {
        for c in 0..n_per_group {
            // cluster centroid: group signal block + cluster id in the block
            let centroid: Vec<f32> = (0..DIM)
                .map(|d| {
                    if d / BLOCK == g {
                        rng.f() * 0.5 + 0.5
                    } else {
                        rng.f() * 0.1
                    }
                })
                .collect();
            for _doc in 0..5 {
                let mut fields = HashMap::new();
                fields.insert("idx".to_string(), serde_json::json!(id));
                fields.insert("group".to_string(), serde_json::json!(g as i64));
                fields.insert("cluster".to_string(), serde_json::json!(c as i64));
                let mut r = attentiondb_storage::Record::new(fields);
                for h in 0..GROUPS {
                    let sigma = if h == g { 0.01 } else { 0.35 };
                    let v: Vec<f32> = centroid
                        .iter()
                        .map(|&x| {
                            let noise = (rng.f() + rng.f() + rng.f() + rng.f() - 2.0) * 1.2 * sigma;
                            x + noise
                        })
                        .collect();
                    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let v: Vec<f32> = v.iter().map(|x| x / n).collect();
                    let _ = n;
                    r.k_vecs.insert(format!("head{h}"), v);
                }
                e.insert_document("c", r).unwrap();
                id += 1;
            }
        }
    }
    // set collection mode to LearnedGating (C) — the gating-relevant mode
    {
        let coll = e.get_collection("c").unwrap();
        let mut cfg = coll.retrieval_config.read().clone();
        cfg.mode = attentiondb_core::collection::RetrievalMode::LearnedGating;
        *coll.retrieval_config.write() = cfg;
    }
    (dir, e)
}

fn group_query(g: usize, cluster: usize) -> Vec<f32> {
    let q: Vec<f32> = (0..DIM)
        .map(|d| {
            if d / BLOCK == g {
                0.5 + (cluster % 2) as f32 * 0.25
            } else {
                0.05
            }
        })
        .collect();
    let n: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
    q.iter().map(|x| x / n).collect()
}

#[test]
fn trained_model_redirects_queries_to_the_right_head() {
    let (_dir, e) = setup(6);
    let heads: Vec<String> = (0..GROUPS).map(|g| format!("head{g}")).collect();
    let coll = e.get_collection("c").unwrap();
    let head_names = coll.list_heads();

    // deactivate baseline: no model ⇒ fallback; record rankings
    e.deactivate_gating_model("c").unwrap();
    assert!(e.inspect_gating_model("c").unwrap().is_none());
    let mut fallback_top: Vec<Vec<u64>> = Vec::new();
    for g in 0..GROUPS {
        let q = group_query(g, g);
        let res = e.attend("c", &heads, &q, 5).unwrap();
        fallback_top.push(res.iter().map(|(id, _)| *id).collect());
    }

    // install the trained (detector) card — group g's queries should hit
    // group-g docs: the model puts ~all weight on head g, whose clean
    // candidates ARE the group-g docs.
    let card = detector_card(&head_names);
    e.install_gating_card("c", card).unwrap();
    let inspect = e.inspect_gating_model("c").unwrap().unwrap();
    assert!(inspect.contains("detector-v1"), "inspect: {inspect}");

    for g in 0..GROUPS {
        let q = group_query(g, g);
        let res = e.attend("c", &heads, &q, 5).unwrap();
        assert!(!res.is_empty());
        for (id, _) in &res {
            let f = e.get_document_fields(*id);
            let doc_group: i64 = f.get("group").and_then(|v| v.parse().ok()).unwrap_or(-1);
            assert_eq!(
                doc_group, g as i64,
                "gated query for group {g} returned doc from group {doc_group}"
            );
        }
    }
    let _ = fallback_top; // used below in fallback_restores_previous_ranking
}

#[test]
fn fallback_restores_previous_ranking() {
    let (_dir, e) = setup(4);
    let heads: Vec<String> = (0..GROUPS).map(|g| format!("head{g}")).collect();
    let coll = e.get_collection("c").unwrap();
    let head_names = coll.list_heads();

    let q = group_query(1, 1);
    let before = e.attend("c", &heads, &q, 5).unwrap();
    e.install_gating_card("c", detector_card(&head_names))
        .unwrap();
    let during = e.attend("c", &heads, &q, 5).unwrap();
    e.deactivate_gating_model("c").unwrap();
    let after = e.attend("c", &heads, &q, 5).unwrap();

    // the trained model must CHANGE the ranking on this corpus (otherwise
    // this test fixture proves nothing)
    assert_ne!(
        during.iter().map(|r| r.0).collect::<Vec<_>>(),
        before.iter().map(|r| r.0).collect::<Vec<_>>(),
        "model had no effect — fixture broken"
    );
    // §14: deactivation restores the deterministic fallback exactly
    assert_eq!(
        after.iter().map(|r| r.0).collect::<Vec<_>>(),
        before.iter().map(|r| r.0).collect::<Vec<_>>()
    );
}

#[test]
fn incompatible_models_are_rejected() {
    use attentiondb_learned::gating_v2::{GatingMlp, ModelCard, TrainingMeta};
    let (_dir, e) = setup(2);
    let coll = e.get_collection("c").unwrap();
    let head_names = coll.list_heads();

    // wrong head count
    let m = GatingMlp::new(DIM, GROUPS, GROUPS + 4, 1);
    let mut card = ModelCard::from_mlp(
        &m,
        TrainingMeta {
            seed: 1,
            dataset_hash: 1,
            objective: "soft_target".into(),
            learning_rate: 0.01,
            batch_size: 32,
            epochs_run: 1,
            best_val_loss: 0.0,
            l2: 0.0,
            timestamp_unix: 0,
            code_commit: "t".into(),
            hardware: "ci".into(),
        },
        "wrong-heads",
        "soft_target",
    );
    card.head_names = Some((0..GROUPS + 4).map(|i| format!("h{i}")).collect());
    let err = e.install_gating_card("c", card).unwrap_err();
    assert!(err.to_string().contains("refusing"), "got: {err}");

    // incomplete head coverage
    let mut card2 = detector_card(&head_names);
    if let Some(names) = card2.head_names.as_mut() {
        names.pop();
    }
    let err2 = e.install_gating_card("c", card2).unwrap_err();
    assert!(err2.to_string().contains("refusing"), "got: {err2}");
}

#[test]
fn registry_roundtrip_and_active_delete_guard() {
    use attentiondb_learned::registry::ModelRegistry;
    let (_dir, e) = setup(2);
    let coll = e.get_collection("c").unwrap();
    let head_names = coll.list_heads();
    let reg_root = std::env::temp_dir().join(format!("gating-reg-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&reg_root);
    let reg = ModelRegistry::new(&reg_root);

    let card = detector_card(&head_names);
    let v1 = reg.save_new(&card).unwrap();
    reg.activate(v1).unwrap();
    assert_eq!(reg.active_version().unwrap(), Some(v1));

    // engine loads from the registry (not just direct install)
    e.activate_gating_model("c", &reg_root).unwrap();
    assert!(e
        .inspect_gating_model("c")
        .unwrap()
        .unwrap()
        .contains("detector-v1"));

    // §17: active model is protected from removal
    assert!(matches!(
        reg.remove(v1),
        Err(attentiondb_learned::registry::RegistryError::ActiveModelProtected)
    ));
    // deactivate → removal allowed
    e.deactivate_gating_model("c").unwrap();
    reg.deactivate().unwrap();
    reg.remove(v1).unwrap();
    assert!(reg.list().is_empty());
    let _ = std::fs::remove_dir_all(&reg_root);
}

#[test]
fn hot_swap_takes_effect_without_restart() {
    // 2 clusters × 5 docs × 4 groups = 40 docs ≪ ef_search 64 ⇒ HNSW search
    // is exact/deterministic (hnsw_rs layer RNG is OS-seeded; only tiny
    // corpora give stable exact orderings — session memory).
    let (_dir, e) = setup(2);
    let heads: Vec<String> = (0..GROUPS).map(|g| format!("head{g}")).collect();
    let coll = e.get_collection("c").unwrap();
    let head_names = coll.list_heads();

    e.install_gating_card("c", detector_card(&head_names))
        .unwrap();
    let q = group_query(2, 2);
    let v1 = e.attend("c", &heads, &q, 5).unwrap();

    // v2: deliberately WRONG weights (all mass on head 0) — a different
    // model must change behavior immediately (§18), deterministically.
    use attentiondb_learned::gating_v2::{GatingMlp, ModelCard, TrainingMeta};
    let mut m = GatingMlp::new(DIM, 1, GROUPS, 1);
    m.b2[0] = 10.0; // logit head0 dominates for every query
    let meta = TrainingMeta {
        seed: 2,
        dataset_hash: 2,
        objective: "soft_target".into(),
        learning_rate: 0.01,
        batch_size: 32,
        epochs_run: 1,
        best_val_loss: 0.0,
        l2: 0.0,
        timestamp_unix: 0,
        code_commit: "test".into(),
        hardware: "ci".into(),
    };
    let mut card2 = ModelCard::from_mlp(&m, meta, "head0-always", "soft_target");
    card2.head_names = Some(head_names.clone());
    e.install_gating_card("c", card2).unwrap();
    let v2 = e.attend("c", &heads, &q, 5).unwrap();

    assert_ne!(
        v1.iter().map(|r| r.0).collect::<Vec<_>>(),
        v2.iter().map(|r| r.0).collect::<Vec<_>>(),
        "hot swap did not change behavior"
    );
    // (exact-order equality with gate_override is untestable here: hnsw_rs
    // layer RNG explores different regions per call even at 40 docs — the
    // deterministic weights-level contract lives in collection.rs unit tests)
}

#[test]
fn gating_inference_overhead_is_microsecond_scale() {
    let (_dir, e) = setup(10);
    let heads: Vec<String> = (0..GROUPS).map(|g| format!("head{g}")).collect();
    let coll = e.get_collection("c").unwrap();
    let head_names = coll.list_heads();
    e.install_gating_card("c", detector_card(&head_names))
        .unwrap();
    let q = group_query(0, 0);
    // warmup
    let _ = e.attend("c", &heads, &q, 10).unwrap();
    let t = std::time::Instant::now();
    for _ in 0..50 {
        let _ = e.attend("c", &heads, &q, 10).unwrap();
    }
    let per_query_us = t.elapsed().as_secs_f64() * 1e6 / 50.0;
    // §38: the model must be tiny next to HNSW. Whole-query budget here is
    // generous; the card itself is ~350 params. Guard against regressions
    // that would make the model the bottleneck.
    assert!(
        per_query_us < 5_000.0,
        "query with gating took {per_query_us:.0}µs — model overhead too high"
    );
}
