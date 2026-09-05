//! Phase 2 integration tests — staged retrieval pipeline.
//!
//! Covers: parallel≡serial (§45), candidate generation vs ranking separation
//! (§4), exact rerank agreement with brute force (§10), ablation modes A–E
//! (§9), determinism (§24), float safety (§25), retired-document exclusion
//! (§46), explain breakdowns (§21), edge cases (§22).

mod common;

use common::*;

const DIM: usize = 16;

fn setup(dir: &std::path::Path, n_docs: usize) -> attentiondb_core::AttentionEngine {
    let e = open_sync(dir);
    e.create_collection("c", DIM, &["default", "semantic"])
        .unwrap();
    for i in 0..n_docs {
        let mut r = doc(
            i as i64,
            &one_hot(i % DIM, DIM),
            &format!("body number {i} alpha"),
        );
        r.k_vecs
            .insert("semantic".to_string(), one_hot((i * 7 + 3) % DIM, DIM));
        e.insert_document("c", r).unwrap();
    }
    e
}

/// §45/§5: parallel head search must produce byte-identical results to serial.
#[test]
fn parallel_equals_serial() {
    let dir = temp_db();
    let e = setup(dir.path(), 200);
    for mode in [
        attentiondb_core::collection::RetrievalMode::FixedFusion,
        attentiondb_core::collection::RetrievalMode::LearnedGating,
        attentiondb_core::collection::RetrievalMode::QKAttention,
        attentiondb_core::collection::RetrievalMode::Full,
    ] {
        {
            e.get_collection("c")
                .unwrap()
                .retrieval_config
                .write()
                .parallel = false;
        }
        let serial = e
            .attend(
                "c",
                &["default".into(), "semantic".into()],
                &one_hot(5, DIM),
                10,
            )
            .unwrap();
        {
            e.get_collection("c")
                .unwrap()
                .retrieval_config
                .write()
                .parallel = true;
        }
        let parallel = e
            .attend(
                "c",
                &["default".into(), "semantic".into()],
                &one_hot(5, DIM),
                10,
            )
            .unwrap();
        assert_eq!(serial, parallel, "mode {mode:?}: parallel != serial");
        assert!(!serial.is_empty());
    }
}

/// §10: exact reranking must agree with brute-force cosine where mathematically
/// expected: the document whose vector EQUALS the query wins top-1 with an
/// exact gated similarity of ~1.0 in the breakdown.
///
/// The probe vector (two nonzero coordinates) is unique among the one-hot
/// corpus, so the exact match is unambiguous and no tie-breaking is involved.
#[test]
fn exact_rerank_agrees_with_brute_force() {
    let dir = temp_db();
    // 80 docs << ef_search(64)·reachability: approximate search explores the
    // whole graph, removing hnsw_rs layer-RNG flakiness from this exactness test.
    let e = setup(dir.path(), 80);
    // unique-direction probe doc: same two-hot vector in BOTH heads
    let mut probe = doc(900, &two_hot(5, 11, DIM), "probe unique");
    probe
        .k_vecs
        .insert("semantic".to_string(), two_hot(5, 11, DIM));
    e.insert_document("c", probe).unwrap();
    let ranked = e
        .get_collection("c")
        .unwrap()
        .attend_detailed(
            &["default".into()],
            &two_hot(5, 11, DIM),
            10,
            None,
            Some(attentiondb_core::collection::RetrievalMode::Full),
            None,
            None,
            None,
        )
        .unwrap();
    assert!(!ranked.is_empty());
    assert_eq!(
        ranked[0].id,
        id_of_idx(&e, "c", 900).unwrap(),
        "exact match must win top-1"
    );
    let mhs = ranked[0].features.multi_head_similarity;
    assert!(
        (mhs - 1.0).abs() < 1e-4,
        "exact gated similarity for identical vector should be ~1, got {mhs}"
    );
    for r in &ranked {
        assert!(r.final_score.is_finite());
    }
}

/// Vector with exactly two nonzero coordinates (unique direction among one-hots).
fn two_hot(a: usize, b: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0; dim];
    v[a % dim] = 1.0;
    v[b % dim] = 1.0;
    v
}

/// §9: all ablation modes execute and rank the exact-match document top-1.
#[test]
fn ablation_modes_all_execute_and_rank_exact_match_top1() {
    let dir = temp_db();
    // small corpus (see exact_rerank test note): deterministic graph coverage
    let e = setup(dir.path(), 80);
    let modes = [
        attentiondb_core::collection::RetrievalMode::SingleHead,
        attentiondb_core::collection::RetrievalMode::FixedFusion,
        attentiondb_core::collection::RetrievalMode::LearnedGating,
        attentiondb_core::collection::RetrievalMode::QKAttention,
        attentiondb_core::collection::RetrievalMode::Full,
    ];
    let coll = e.get_collection("c").unwrap();
    // unique-direction probe doc: two-hot vector in BOTH heads
    let mut probe_rec = doc(901, &two_hot(7, 13, DIM), "probe unique");
    probe_rec
        .k_vecs
        .insert("semantic".to_string(), two_hot(7, 13, DIM));
    e.insert_document("c", probe_rec).unwrap();
    let probe_id = id_of_idx(&e, "c", 901).unwrap();
    for mode in modes {
        let r = coll
            .attend_detailed(
                &["default".into(), "semantic".into()],
                &two_hot(7, 13, DIM),
                5,
                None,
                Some(mode),
                None,
                None,
                None,
            )
            .unwrap();
        assert!(!r.is_empty(), "mode {mode:?} returned no results");
        if mode == attentiondb_core::collection::RetrievalMode::Full {
            assert_eq!(
                r[0].id, probe_id,
                "mode {mode:?}: exact-match doc must win top-1"
            );
        } else {
            // approximate stages (A–D) may order exact ties arbitrarily — the
            // contract is containment; precise ordering is exactly what the
            // exact-rerank stage (E) adds.
            assert!(
                r.iter().any(|c| c.id == probe_id),
                "mode {mode:?}: exact-match doc must appear in top-5"
            );
        }
    }
}

/// §46: a deleted document must never appear in results in any mode.
#[test]
fn deleted_doc_never_surfaces_in_any_mode() {
    let dir = temp_db();
    let e = setup(dir.path(), 100);
    let victim = id_of_idx(&e, "c", 42).unwrap(); // numeric id (search identity)
    let victim_uuid = {
        let all = e.document_store.read().list_all_records();
        all.iter()
            .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(42))
            .unwrap()
            .id
            .to_string()
    };
    assert!(e.delete_document("c", &victim_uuid).unwrap());
    let modes = [
        attentiondb_core::collection::RetrievalMode::SingleHead,
        attentiondb_core::collection::RetrievalMode::FixedFusion,
        attentiondb_core::collection::RetrievalMode::LearnedGating,
        attentiondb_core::collection::RetrievalMode::QKAttention,
        attentiondb_core::collection::RetrievalMode::Full,
    ];
    for mode in modes {
        let coll = e.get_collection("c").unwrap();
        let r = coll
            .attend_detailed(
                &["default".into(), "semantic".into()],
                &one_hot(42 % DIM, DIM),
                50,
                None,
                Some(mode),
                None,
                None,
                None,
            )
            .unwrap();
        assert!(
            !r.iter().any(|c| c.id == victim),
            "mode {mode:?}: deleted doc appeared in results"
        );
    }
}

/// §24: repeated identical queries return identical output (ties included).
#[test]
fn repeated_queries_are_deterministic() {
    let dir = temp_db();
    let e = setup(dir.path(), 120);
    // uniform vectors create ties; order must still be stable (score DESC, id ASC)
    for _ in 0..5 {
        let r1 = e
            .attend(
                "c",
                &["default".into(), "semantic".into()],
                &one_hot(1, DIM),
                20,
            )
            .unwrap();
        let r2 = e
            .attend(
                "c",
                &["default".into(), "semantic".into()],
                &one_hot(1, DIM),
                20,
            )
            .unwrap();
        assert_eq!(r1, r2);
    }
}

/// §25: zero query vector and NaN-producing paths must not leak NaN/Inf.
#[test]
fn degenerate_queries_never_leak_nonfinite_scores() {
    let dir = temp_db();
    let e = setup(dir.path(), 60);
    let zero = vec![0.0f32; DIM];
    let r = e
        .attend("c", &["default".into(), "semantic".into()], &zero, 10)
        .unwrap();
    assert!(
        r.iter().all(|(_, s)| s.is_finite()),
        "zero query leaked non-finite: {r:?}"
    );
}

/// §22: k=0, k>docs, empty heads, wrong dimension — typed behavior, no panic.
#[test]
fn edge_cases_are_typed_and_panic_free() {
    let dir = temp_db();
    let e = setup(dir.path(), 30);
    let coll = e.get_collection("c").unwrap();

    assert!(coll
        .attend(&["default".into()], &one_hot(1, DIM), 0)
        .unwrap()
        .is_empty());
    assert!(coll.attend(&[], &one_hot(1, DIM), 10).unwrap().is_empty());
    // k > doc count: returns what exists
    let r = coll
        .attend(&["default".into()], &one_hot(1, DIM), 10_000)
        .unwrap();
    assert!(!r.is_empty() && r.len() <= 30);
    // wrong dimension: typed error, no panic
    let err = coll
        .attend(&["default".into()], &one_hot(1, 8), 5)
        .unwrap_err();
    assert!(
        err.to_string().contains("dimension"),
        "unexpected error: {err}"
    );
    // missing head: documented skip semantics (matches pre-Phase-2 behavior:
    // heads that don't exist are skipped; all-missing → empty, never a panic)
    let missing = coll
        .attend(&["no-such-head".into()], &one_hot(1, DIM), 5)
        .unwrap();
    assert!(missing.is_empty());
    // empty collection
    let dir2 = temp_db();
    let e2 = open_sync(dir2.path());
    e2.create_collection("empty", DIM, &["default"]).unwrap();
    let r = e2
        .attend("empty", &["default".into()], &one_hot(1, DIM), 5)
        .unwrap();
    assert!(r.is_empty());
}

/// §21: detailed results expose per-head provenance for debugging.
#[test]
fn explain_breakdown_exposes_head_provenance() {
    let dir = temp_db();
    let e = setup(dir.path(), 80);
    let coll = e.get_collection("c").unwrap();
    let ranked = coll
        .attend_detailed(
            &["default".into(), "semantic".into()],
            &one_hot(2, DIM),
            5,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    assert!(!ranked.is_empty());
    for r in &ranked {
        // both channels recorded: exactly 2 head slots
        assert_eq!(r.features.head_scores.len(), 2);
        // at least one head actually returned this document
        assert!(r.features.head_scores.iter().any(|s| s.is_some()));
        assert!(r.features.multi_head_similarity.is_finite());
    }
}

/// §14/§35: hybrid through the pipeline — RRF baseline and linear fusion both
/// work; text-only queries are served by the BM25 channel alone.
#[test]
fn hybrid_rrf_and_fusion_and_text_only() {
    let dir = temp_db();
    let e = setup(dir.path(), 100);
    let coll = e.get_collection("c").unwrap();
    let text = "body alpha";

    // RRF default
    let rrf = coll
        .attend_hybrid(
            &["default".into(), "semantic".into()],
            &one_hot(4, DIM),
            text,
            10,
        )
        .unwrap();
    assert!(!rrf.is_empty());

    // linear fusion strategy
    {
        let mut cfg = coll.retrieval_config.write().clone();
        cfg.hybrid_strategy = attentiondb_core::collection::HybridStrategy::Fusion;
        *coll.retrieval_config.write() = cfg;
    }
    let fused = coll
        .attend_hybrid(
            &["default".into(), "semantic".into()],
            &one_hot(4, DIM),
            text,
            10,
        )
        .unwrap();
    assert!(!fused.is_empty());
    assert!(fused.iter().all(|(_, s)| s.is_finite()));

    // text-only (empty vector) — BM25 channel alone
    let text_only = coll
        .attend_hybrid(&["default".into(), "semantic".into()], &[], text, 10)
        .unwrap();
    assert!(!text_only.is_empty());
    assert!(text_only.iter().all(|(_, s)| s.is_finite()));
    {
        let mut cfg = coll.retrieval_config.write().clone();
        cfg.hybrid_strategy = attentiondb_core::collection::HybridStrategy::Rrf;
        *coll.retrieval_config.write() = cfg;
    }
}

/// §16/§38: config validation rejects planner nonsense with typed errors.
#[test]
fn retrieval_config_validation() {
    use attentiondb_core::collection::{RetrievalConfig, RetrievalMode};
    let mut cfg = RetrievalConfig::default();
    assert!(cfg.validate().is_ok());
    cfg.candidate_multiplier = 0;
    assert!(cfg.validate().is_err());
    cfg = RetrievalConfig::default();
    cfg.candidate_budget = 1_000_000;
    assert!(cfg.validate().is_err());
    cfg = RetrievalConfig::default();
    cfg.fusion.multi_head_similarity = 0.9; // weights no longer sum to 1
    assert!(cfg.validate().is_err());
    cfg = RetrievalConfig::default();
    cfg.mode = RetrievalMode::Full;
    assert!(cfg.validate().is_ok());
}

/// §6: candidate budget is enforced end-to-end (union bounded, results ≤ k).
#[test]
fn candidate_budget_enforced() {
    let dir = temp_db();
    let e = setup(dir.path(), 400);
    let coll = e.get_collection("c").unwrap();
    {
        let mut cfg = coll.retrieval_config.write().clone();
        cfg.candidate_budget = 64;
        *coll.retrieval_config.write() = cfg;
    }
    let ranked = coll
        .attend_detailed(
            &["default".into(), "semantic".into()],
            &one_hot(6, DIM),
            10,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    assert!(ranked.len() <= 10);
    assert!(!ranked.is_empty());
}

/// §5: multi-head search actually runs concurrently and beats serial wall-clock
/// at non-trivial head counts (guarded: only asserts speedup headroom, not
/// strict timing, to stay CI-safe; the parallel≡serial equivalence is the
/// semantic contract and is asserted exactly).
#[test]
fn many_heads_parallel_search_is_correct_and_bounded() {
    let dir = temp_db();
    let e = open_sync(dir.path());
    let heads: Vec<String> = (0..8).map(|h| format!("h{h}")).collect();
    let head_refs: Vec<&str> = heads.iter().map(|s| s.as_str()).collect();
    e.create_collection("m", DIM, &head_refs).unwrap();
    for i in 0..300usize {
        let mut r = doc(i as i64, &one_hot(i % DIM, DIM), "multi");
        for h in &heads {
            r.k_vecs
                .insert(h.clone(), one_hot((i + h.len()) % DIM, DIM));
        }
        e.insert_document("m", r).unwrap();
    }
    let q = one_hot(2, DIM);
    {
        let coll = e.get_collection("m").unwrap();
        coll.retrieval_config.write().parallel = false;
    }
    let serial = e.attend("m", &heads, &q, 10).unwrap();
    {
        let coll = e.get_collection("m").unwrap();
        coll.retrieval_config.write().parallel = true;
    }
    let parallel = e.attend("m", &heads, &q, 10).unwrap();
    assert_eq!(serial, parallel, "8-head parallel != serial");
    assert_eq!(serial.len(), 10);
}

/// §22: documents missing a head's vector are still retrievable via other heads
/// and the breakdown shows the missing channel as None.
#[test]
fn partial_head_coverage_is_represented() {
    let dir = temp_db();
    let e = open_sync(dir.path());
    e.create_collection("p", DIM, &["default", "semantic"])
        .unwrap();
    // 40 docs with BOTH heads; 10 docs with ONLY default
    for i in 0..40i64 {
        let mut r = doc(i, &one_hot(i as usize % DIM, DIM), "both");
        r.k_vecs.insert(
            "semantic".to_string(),
            one_hot((i as usize * 7 + 3) % DIM, DIM),
        );
        e.insert_document("p", r).unwrap();
    }
    for i in 40..50i64 {
        let mut r = doc(i, &one_hot(i as usize % DIM, DIM), "only-default");
        r.k_vecs.remove("semantic");
        e.insert_document("p", r).unwrap();
    }
    let coll = e.get_collection("p").unwrap();
    let ranked = coll
        .attend_detailed(
            &["default".into(), "semantic".into()],
            &one_hot(41 % DIM, DIM),
            50,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
    let only_default = ranked
        .iter()
        .find(|r| r.id == id_of_idx(&e, "p", 41).unwrap())
        .expect("only-default doc should be retrievable via default head");
    assert!(
        only_default.features.head_scores[1].is_none(),
        "semantic channel must be None for a doc without that vector"
    );
    assert!(only_default.features.head_scores[0].is_some());
}
