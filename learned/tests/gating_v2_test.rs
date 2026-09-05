//! Phase 2B gating trainer tests: determinism, learnability, safety (§5–7,
//! §15–16, §24, §26).

use attentiondb_learned::eval::{
    evaluate_weighting, fuse_rrf, global_best_head, head_ranking, rank_metrics, uniform_weights,
};
use attentiondb_learned::gating_v2::{
    diagnostics, train_gating, GatingDataset, GatingMlp, HeadExample, ModelCard, Objective,
    QualityTarget, QueryExample, Split, TrainingConfig, TrainingMeta,
};

/// Deterministic xorshift for test fixtures.
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

const H: usize = 4; // heads
const D: usize = 8; // query dim
const G: usize = 4; // query groups = best-head index per group

/// Controlled learnability fixture (§8 miniature): query belongs to group g
/// (encoded as one-hot block g in the query vector); head g retrieves the
/// ground truth, other heads return near-noise candidates. If the trainer
/// cannot learn query→head from THIS, it cannot learn at all.
fn learnable_dataset(n_per_group: usize, seed: u64) -> GatingDataset {
    let mut rng = R(seed);
    let mut queries = Vec::new();
    let mut qid = 0u64;
    for split in [Split::Train, Split::Val, Split::Test] {
        for g in 0..G {
            for _ in 0..n_per_group {
                // query vector: block g carries signal + noise
                let block = D / G;
                let x: Vec<f32> = (0..D)
                    .map(|d| {
                        if d / block == g {
                            0.8 + rng.f() * 0.4
                        } else {
                            rng.f() * 0.2
                        }
                    })
                    .collect();
                // ground truth: 5 ids that only head g finds
                let gt: Vec<u64> = (0..5).map(|i| (g as u64) * 1000 + i).collect();
                let mut heads = Vec::new();
                for h in 0..H {
                    if h == g {
                        // good head: finds all GT at the top
                        let cands: Vec<u64> = gt.clone();
                        let norm: Vec<f32> = (0..5).map(|i| 1.0 - 0.05 * i as f32).collect();
                        heads.push(HeadExample {
                            candidates: cands.clone(),
                            raw_scores: norm.clone(),
                            norm_scores: norm,
                            exact_scores: vec![0.9; 5],
                            recall_at_k: 1.0,
                            ndcg_at_k: 1.0,
                            mrr: 1.0,
                        });
                    } else {
                        // bad head: noise candidates, misses GT entirely
                        let cands: Vec<u64> =
                            (0..20).map(|_| 9000 + (rng.f() * 500.0) as u64).collect();
                        let norm: Vec<f32> = (0..20).map(|i| 1.0 - 0.04 * i as f32).collect();
                        heads.push(HeadExample {
                            candidates: cands,
                            raw_scores: norm.clone(),
                            norm_scores: norm,
                            exact_scores: vec![0.1; 20],
                            recall_at_k: 0.0,
                            ndcg_at_k: 0.0,
                            mrr: 0.0,
                        });
                    }
                }
                queries.push(QueryExample {
                    query_id: qid,
                    query: x,
                    query_group: Some(g as u32),
                    split,
                    ground_truth: gt,
                    heads,
                });
                qid += 1;
            }
        }
    }
    GatingDataset {
        format: "attentiondb-gating-dataset".into(),
        version: 1,
        num_heads: H,
        input_dim: D,
        top_k: 5,
        corpus_desc: "controlled-mini".into(),
        seed,
        queries,
    }
}

#[test]
fn training_is_deterministic_bit_for_bit() {
    let ds = learnable_dataset(8, 7);
    let cfg = TrainingConfig::default();
    let a = train_gating(&ds, &cfg, QualityTarget::Recall);
    let b = train_gating(&ds, &cfg, QualityTarget::Recall);
    assert_eq!(a.model.w1, b.model.w1, "w1 differs across identical runs");
    assert_eq!(a.model.w2, b.model.w2, "w2 differs");
    assert_eq!(a.model.b2, b.model.b2, "b2 differs");
    assert_eq!(a.curves.len(), b.curves.len());
}

#[test]
fn gating_learns_query_conditional_preferences_on_held_out() {
    let ds = learnable_dataset(20, 11);
    let cfg = TrainingConfig::default();
    let out = train_gating(&ds, &cfg, QualityTarget::Recall);

    // §30.1: query-dependent preferences — high weight on the group's head.
    // §36: the model sees only the query vector, never the group label.
    let test: Vec<&QueryExample> = ds
        .queries
        .iter()
        .filter(|q| q.split == Split::Test)
        .collect();
    let mut correct_group = 0usize;
    for q in &test {
        let w = out.model.predict(&q.query);
        let g = q.query_group.unwrap() as usize;
        if w[g] == w.iter().cloned().fold(f32::MIN, f32::max) {
            correct_group += 1;
        }
    }
    assert!(
        correct_group >= test.len() * 9 / 10,
        "gating picked the group's best head on only {correct_group}/{} test queries",
        test.len()
    );

    // §30.2: predicted weight correlates with actual head quality.
    let corr = attentiondb_learned::gating_v2::weight_quality_correlation(
        &ds,
        &out.model,
        Split::Test,
        QualityTarget::Recall,
    );
    assert!(corr > 0.8, "weight-quality correlation too low: {corr}");

    // §30.3: held-out retrieval improves vs uniform.
    let uniform = evaluate_weighting(&ds, Split::Test, &|_| uniform_weights(H));
    let gated = evaluate_weighting(&ds, Split::Test, &|q| out.model.predict(&q.query));
    assert!(
        gated.metrics.recall_at_10 > uniform.metrics.recall_at_10 + 0.3,
        "gated R@10 {:.3} not clearly above uniform {:.3}",
        gated.metrics.recall_at_10,
        uniform.metrics.recall_at_10
    );
}

#[test]
fn pairwise_objective_recovers_quality_ordering() {
    let ds = learnable_dataset(16, 13);
    let cfg = TrainingConfig {
        objective: Objective::Pairwise,
        ..TrainingConfig::default()
    };
    let out = train_gating(&ds, &cfg, QualityTarget::Recall);
    let test: Vec<&QueryExample> = ds
        .queries
        .iter()
        .filter(|q| q.split == Split::Test)
        .collect();
    let mut ordered = 0usize;
    for q in &test {
        let w = out.model.predict(&q.query);
        // good head must outrank every bad head
        let g = q.query_group.unwrap() as usize;
        if (0..H).all(|h| h == g || w[g] > w[h]) {
            ordered += 1;
        }
    }
    assert!(
        ordered >= test.len() * 9 / 10,
        "pairwise ordering on {ordered}/{}",
        test.len()
    );
}

#[test]
fn regression_objective_beats_uniform_on_validation() {
    let ds = learnable_dataset(16, 17);
    let cfg = TrainingConfig {
        objective: Objective::QualityRegression,
        ..TrainingConfig::default()
    };
    let out = train_gating(&ds, &cfg, QualityTarget::Recall);
    let uniform = evaluate_weighting(&ds, Split::Val, &|_| uniform_weights(H));
    let gated = evaluate_weighting(&ds, Split::Val, &|q| out.model.predict(&q.query));
    assert!(gated.metrics.recall_at_10 > uniform.metrics.recall_at_10);
}

#[test]
fn diagnostics_reveal_head_selection_not_collapse() {
    let ds = learnable_dataset(20, 19);
    let out = train_gating(&ds, &TrainingConfig::default(), QualityTarget::Recall);
    let mut ws = Vec::new();
    let mut qs = Vec::new();
    for q in ds.queries.iter().filter(|q| q.split == Split::Test) {
        ws.push(out.model.predict(&q.query));
        qs.push(q.heads.iter().map(|h| h.recall_at_k).collect::<Vec<_>>());
    }
    let d = diagnostics(&ws, &qs);
    // every head should be selected for roughly its share of queries (§7:
    // collapse to one head would show selection_frequency ≈ [1,0,0,0])
    for (i, &s) in d.selection_frequency.iter().enumerate() {
        assert!(
            s > 0.15 && s < 0.85,
            "head {i} selection freq {s} — collapse or degenerate gating"
        );
    }
    assert!(
        d.weight_quality_pearson > 0.8,
        "corr = {}",
        d.weight_quality_pearson
    );
}

#[test]
fn model_card_roundtrip_and_validation() {
    let m = GatingMlp::new(6, 4, 3, 99);
    let meta = TrainingMeta {
        seed: 99,
        dataset_hash: 123,
        objective: "soft_target".into(),
        learning_rate: 0.01,
        batch_size: 32,
        epochs_run: 10,
        best_val_loss: 0.5,
        l2: 1e-4,
        timestamp_unix: 0,
        code_commit: "test".into(),
        hardware: "ci".into(),
    };
    let card = ModelCard::from_mlp(&m, meta.clone(), "model-test", Objective::SoftTarget.name());
    let dir = std::env::temp_dir().join(format!("gating-card-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("m.json");
    card.save(&p).unwrap();
    let loaded = ModelCard::load(&p).unwrap();
    assert_eq!(loaded, card, "roundtrip must be lossless");

    // §16: wrong dims rejected explicitly
    let mut bad = loaded.clone();
    bad.input_dim = 999;
    assert!(bad.validate().is_err());
    // §16: non-finite weights rejected explicitly
    let mut nan = loaded.clone();
    nan.w2[0] = f32::NAN;
    assert!(matches!(
        nan.validate(),
        Err(attentiondb_learned::gating_v2::ModelError::NonFiniteWeight)
    ));
    // §16: corrupt file → typed error, never silent fallback
    std::fs::write(&p, "{ not json").unwrap();
    assert!(matches!(
        ModelCard::load(&p),
        Err(attentiondb_learned::gating_v2::ModelError::Corrupt(_))
    ));
    // wrong format tag
    let card2 = ModelCard::from_mlp(&m, meta, "x", Objective::SoftTarget.name());
    let mut wrong = card2.clone();
    wrong.format = "something-else".into();
    assert!(matches!(
        wrong.validate(),
        Err(attentiondb_learned::gating_v2::ModelError::BadFormat)
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn global_best_head_uses_train_split_only() {
    let ds = learnable_dataset(24, 23);
    // all heads are best for their group ⇒ no dominant global head; the
    // train-split global-best must be a middling head, not oracle-quality.
    let (h, train_recall) = global_best_head(&ds, Split::Train);
    assert!(h < H);
    // every head is good on 1/4 of queries ⇒ global best train recall < 0.5
    assert!(
        train_recall < 0.5,
        "global best head recall {train_recall} — fixture has a dominant head?"
    );
}

#[test]
fn rrf_and_metrics_sanity() {
    let ds = learnable_dataset(4, 29);
    let q = &ds.queries[0];
    let rrf = fuse_rrf(q, 60.0);
    assert!(!rrf.is_empty());
    // good head's candidates get ranks 0..5 from ONE head ⇒ RRF can rank a
    // good candidate top even though bad heads also contribute
    let m = rank_metrics(&rrf, &q.ground_truth, ds.top_k);
    assert!(m.mrr > 0.0);
    let single = head_ranking(q, q.query_group.unwrap() as usize);
    let m2 = rank_metrics(&single, &q.ground_truth, ds.top_k);
    assert!(
        m2.recall_at_10 == 1.0,
        "the group's own head must fully retrieve its GT"
    );
}
