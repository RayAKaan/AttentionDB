//! C8 residual-attention subsystem tests.
//!
//! The two properties that carry the whole experiment are tested here:
//!   1. `lambda = 0` reproduces the baseline score bit-for-bit, so any C8
//!      improvement is attributable to the correction and not to a changed
//!      pipeline;
//!   2. the cached path (arm I) is bit-identical to the uncached path, so a
//!      latency win can never be bought with a ranking change.
//!
//! These live in their own integration-style file because they exercise the
//! subsystem through its public API only.

use attentiondb_attention::cache::{AttentionKVCache, CacheStats};
use attentiondb_attention::config::C8AttentionConfig;
use attentiondb_attention::negative_mining::MIN_HEADS_FOR_DISAGREEMENT;
use attentiondb_attention::projection::ResidualQkvProjection;
use attentiondb_attention::scorer::ResidualScorer;
use attentiondb_attention::{C8AttentionSubsystem, CandidateAttention, RetrievalEvidence};

const D_A: usize = 8;
const HEADS: usize = 3;

fn heads() -> Vec<String> {
    (0..HEADS).map(|h| format!("h{h}")).collect()
}

fn candidate(seed: f32) -> Vec<Vec<f32>> {
    (0..HEADS)
        .map(|h| vec![seed + h as f32 * 0.25; D_A])
        .collect()
}

fn evidence(seed: f32) -> RetrievalEvidence {
    RetrievalEvidence {
        head_sims: (0..HEADS).map(|h| Some(seed + h as f32 * 0.1)).collect(),
        head_ranks: (0..HEADS).map(|h| Some(1.0 / (1.0 + h as f32))).collect(),
        head_present: vec![true; HEADS],
    }
}

fn config(scale: f32) -> C8AttentionConfig {
    C8AttentionConfig::residual(
        D_A,
        4,
        4,
        vec![attentiondb_attention::AlignmentProjection::identity(D_A); HEADS],
        attentiondb_attention::AlignmentProjection::identity(D_A),
        ResidualQkvProjection::random_residual(D_A, 4, 4, 0.1, 99).unwrap(),
        scale,
        false,
        ResidualScorer::new(1.0, 0.0, 0.0),
    )
}

fn query() -> Vec<f32> {
    vec![0.3; D_A]
}

fn inputs() -> (Vec<Vec<Vec<f32>>>, Vec<RetrievalEvidence>, Vec<f32>) {
    let cands = vec![candidate(1.0), candidate(0.5), candidate(-0.25)];
    let ev = vec![evidence(0.9), evidence(0.4), evidence(0.1)];
    let base = vec![0.61f32, 0.42, 0.17];
    (cands, ev, base)
}

fn subsystem(cfg: C8AttentionConfig) -> C8AttentionSubsystem {
    C8AttentionSubsystem::new(cfg, heads()).unwrap()
}

#[test]
fn lambda_zero_reproduces_the_baseline_exactly() {
    let (cands, ev, base) = inputs();
    let out = subsystem(config(0.0))
        .compute(&query(), &cands, &ev, &base)
        .unwrap();

    assert_eq!(out.baseline_scores, base, "baseline must pass through");
    assert_eq!(out.final_scores, base, "lambda=0 must equal the baseline");
    for (i, c) in out.candidates.iter().enumerate() {
        assert_eq!(c.final_score, base[i], "candidate {i}");
        assert_eq!(c.applied_correction, 0.0, "candidate {i} correction");
        assert!(c.final_score.to_bits() == base[i].to_bits(), "bitwise");
    }
    // The attention still ran: this is a control on the machinery, not a
    // bypass, so a degenerate attention would still be observable here.
    assert!(out.attention_deltas.iter().any(|d| *d != 0.0));
    assert!(out.mean_entropy > 0.0);
}

#[test]
fn lambda_scales_the_correction_exactly() {
    let (cands, ev, base) = inputs();
    for scale in [0.05f32, 0.1, 0.5, 1.0, 2.0] {
        let out = subsystem(config(scale))
            .compute(&query(), &cands, &ev, &base)
            .unwrap();
        for (i, c) in out.candidates.iter().enumerate() {
            let expected = base[i] + scale * c.attention_delta;
            assert!(
                (c.final_score - expected).abs() < 1e-6,
                "scale={scale} candidate={i} got {} want {expected}",
                c.final_score
            );
        }
        out.verify_fusion(1e-6).unwrap();
    }
}

#[test]
fn correction_is_deterministic_for_every_lambda() {
    let (cands, ev, base) = inputs();
    let cfg = config(0.25);
    let a = subsystem(cfg.clone())
        .compute(&query(), &cands, &ev, &base)
        .unwrap();
    let b = subsystem(cfg)
        .compute(&query(), &cands, &ev, &base)
        .unwrap();
    assert_eq!(a.final_scores, b.final_scores);
    assert_eq!(a.attention_deltas, b.attention_deltas);
    assert_eq!(a.mean_entropy.to_bits(), b.mean_entropy.to_bits());
    assert_eq!(a.config_fingerprint, b.config_fingerprint);
}

#[test]
fn cached_path_is_bit_identical_to_uncached() {
    let (cands, ev, base) = inputs();
    let sub = subsystem(config(0.2));
    let uncached = sub.compute(&query(), &cands, &ev, &base).unwrap();
    let doc_ids: Vec<u64> = (0..cands.len() as u64).collect();

    // Cold cache: every document is a miss and gets built.
    let mut cache = AttentionKVCache::new(sub.cache_fingerprint());
    let mut stats = CacheStats::default();
    let cold = sub
        .compute_cached(
            &query(),
            &cands,
            &ev,
            &base,
            &doc_ids,
            &mut cache,
            &mut stats,
        )
        .unwrap();
    assert_eq!(
        cold.final_scores, uncached.final_scores,
        "cold cache differs"
    );
    assert_eq!(cold.attention_deltas, uncached.attention_deltas);
    assert_eq!(stats.misses, 3);
    assert_eq!(stats.hits, 0);

    // Warm cache: every document is a hit, and the K/V bucket goes to zero.
    let mut warm_stats = CacheStats::default();
    let warm = sub
        .compute_cached(
            &query(),
            &cands,
            &ev,
            &base,
            &doc_ids,
            &mut cache,
            &mut warm_stats,
        )
        .unwrap();
    assert_eq!(
        warm.final_scores, uncached.final_scores,
        "warm cache differs"
    );
    assert_eq!(warm.timings.kv_projection_us, 0, "warm cache must skip K/V");
    assert_eq!(warm_stats.hits, 3);
    assert_eq!(warm_stats.misses, 0);
    assert!((warm_stats.hit_rate() - 1.0).abs() < 1e-9);
}

#[test]
fn cache_refuses_a_stale_fingerprint() {
    let (cands, ev, base) = inputs();
    let sub = subsystem(config(0.2));
    let doc_ids = vec![0u64, 1, 2];

    // A cache built by a different model must be rejected, not silently used.
    let stale = sub.projection().fingerprint();
    let mut foreign = attentiondb_attention::CacheFingerprint::from_projection(
        sub.fingerprint() ^ 0xFFFF,
        sub.projection(),
        HEADS,
    );
    foreign.projection_fingerprint = stale;
    let mut cache = AttentionKVCache::new(foreign);
    let mut stats = CacheStats::default();
    let err = sub
        .compute_cached(
            &query(),
            &cands,
            &ev,
            &base,
            &doc_ids,
            &mut cache,
            &mut stats,
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("fingerprint"),
        "unexpected error: {err}"
    );
    assert_eq!(stats.lookups(), 0, "must not even consult a stale cache");
}

#[test]
fn evidence_is_ignored_unless_both_switches_are_on() {
    let (cands, ev, base) = inputs();
    let q = query();
    // Deliberately different evidence, so any read of it would be visible.
    let other_ev: Vec<RetrievalEvidence> = ev
        .iter()
        .map(|r| RetrievalEvidence {
            head_sims: vec![Some(9.0); HEADS],
            head_ranks: r.head_ranks.clone(),
            head_present: r.head_present.clone(),
        })
        .collect();

    // Flag on, scorer carries no evidence weight: the flag alone does nothing.
    let unweighted = C8AttentionConfig {
        use_evidence: true,
        scorer: ResidualScorer::new(1.0, 0.0, 0.0),
        ..config(0.3)
    };
    let a = subsystem(unweighted.clone())
        .compute(&q, &cands, &ev, &base)
        .unwrap();
    let b = subsystem(unweighted)
        .compute(&q, &cands, &other_ev, &base)
        .unwrap();
    assert_eq!(
        a.attention_deltas, b.attention_deltas,
        "scorer must ignore evidence"
    );

    // Flag off, scorer weighted: the weight alone does nothing either.
    let weighted = C8AttentionConfig {
        use_evidence: false,
        scorer: ResidualScorer::with_disagreement(1.0, 0.5, 0.25, 0.0),
        ..config(0.3)
    };
    let c = subsystem(weighted.clone())
        .compute(&q, &cands, &ev, &base)
        .unwrap();
    let d = subsystem(weighted)
        .compute(&q, &cands, &other_ev, &base)
        .unwrap();
    assert_eq!(
        c.attention_deltas, d.attention_deltas,
        "flag must gate evidence"
    );

    // Both on: evidence is genuinely read, so the test above is not vacuous.
    let both = C8AttentionConfig {
        use_evidence: true,
        scorer: ResidualScorer::with_disagreement(1.0, 0.5, 0.25, 0.0),
        ..config(0.3)
    };
    let e = subsystem(both.clone())
        .compute(&q, &cands, &ev, &base)
        .unwrap();
    let f = subsystem(both)
        .compute(&q, &cands, &other_ev, &base)
        .unwrap();
    assert_ne!(e.attention_deltas, f.attention_deltas);
}

#[test]
fn arm_f_changes_scores_when_evidence_is_weighted() {
    let (cands, ev, base) = inputs();
    let q = query();
    let plain = subsystem(config(0.3))
        .compute(&q, &cands, &ev, &base)
        .unwrap();
    let with_ev = subsystem(C8AttentionConfig {
        use_evidence: true,
        scorer: ResidualScorer::with_disagreement(1.0, 0.5, 0.25, 0.0),
        ..config(0.3)
    })
    .compute(&q, &cands, &ev, &base)
    .unwrap();
    assert_ne!(
        plain.attention_deltas, with_ev.attention_deltas,
        "evidence/disagreement must change the correction"
    );
    with_ev.verify_fusion(1e-6).unwrap();
}

#[test]
fn residual_projection_at_zero_delta_equals_truncated_identity_attention() {
    // Arm C's machinery: the correction comes from the deterministic base, not
    // from a learned delta, and is non-degenerate.
    let (cands, ev, base) = inputs();
    let cfg = C8AttentionConfig {
        residual_projection: Some(
            ResidualQkvProjection::truncated_identity_residual(D_A, 4, 4, 0.1).unwrap(),
        ),
        ..config(0.5)
    };
    let out = subsystem(cfg)
        .compute(&query(), &cands, &ev, &base)
        .unwrap();
    assert!(out.attention_deltas.iter().any(|d| d.abs() > 0.0));
    // With a zero delta and a zero evidence weight, the materialized projection
    // is the truncated identity itself.
    let p = ResidualQkvProjection::truncated_identity_residual(D_A, 4, 4, 0.1)
        .unwrap()
        .to_qkv();
    assert_eq!(p.w_k, attentiondb_attention::truncated_identity(D_A, 4));
}

#[test]
fn reduced_dimensions_are_honored_end_to_end() {
    for d in [2usize, 4, 8] {
        let (cands, ev, base) = inputs();
        let cfg = C8AttentionConfig::residual(
            D_A,
            d,
            d,
            vec![attentiondb_attention::AlignmentProjection::identity(D_A); HEADS],
            attentiondb_attention::AlignmentProjection::identity(D_A),
            ResidualQkvProjection::random_residual(D_A, d, d, 0.1, 5).unwrap(),
            0.2,
            false,
            ResidualScorer::new(1.0, 0.0, 0.0),
        );
        let out = subsystem(cfg)
            .compute(&query(), &cands, &ev, &base)
            .unwrap();
        for c in &out.candidates {
            assert_eq!(c.output.len(), d, "value dim must follow d_v");
            assert_eq!(c.weights.len(), HEADS, "one weight per head");
            assert_eq!(c.logits.len(), HEADS);
            assert!(c.entropy.is_finite());
        }
        out.verify_fusion(1e-6).unwrap();
    }
}

#[test]
fn attention_weights_are_a_distribution() {
    let (cands, ev, base) = inputs();
    let out = subsystem(config(0.3))
        .compute(&query(), &cands, &ev, &base)
        .unwrap();
    for c in &out.candidates {
        let sum: f32 = c.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights must sum to 1, got {sum}");
        assert!(c.weights.iter().all(|w| *w >= 0.0));
        let h = HEADS as f32;
        assert!(c.entropy >= -1e-6 && c.entropy <= h.ln() + 1e-6);
    }
    let pm: f32 = out.per_head_mean.iter().sum();
    assert!((pm - 1.0).abs() < 1e-5, "per-head mean must sum to 1");
}

#[test]
fn rejects_mismatched_input_lengths() {
    let (cands, ev, base) = inputs();
    let sub = subsystem(config(0.1));
    assert!(sub.compute(&query(), &cands, &ev[..2], &base).is_err());
    assert!(sub.compute(&query(), &cands, &ev, &base[..2]).is_err());
    assert!(sub
        .compute_cached(
            &query(),
            &cands,
            &ev,
            &base,
            &[0, 1],
            &mut { AttentionKVCache::new(sub.cache_fingerprint()) },
            &mut CacheStats::default()
        )
        .is_err());
}

#[test]
fn disabled_config_is_rejected() {
    let (cands, ev, base) = inputs();
    let sub = C8AttentionSubsystem::new(C8AttentionConfig::disabled(), heads()).unwrap();
    assert!(sub.compute(&query(), &cands, &ev, &base).is_err());
}

#[test]
fn zero_candidates_is_a_valid_no_op() {
    let sub = subsystem(config(0.3));
    let out = sub.compute(&query(), &[], &[], &[]).unwrap();
    assert!(out.candidates.is_empty());
    assert!(out.final_scores.is_empty());
    assert_eq!(out.mean_abs_correction(), 0.0);
    assert_eq!(out.mean_entropy, 0.0);
    assert_eq!(out.per_head_mean, vec![0.0; HEADS]);
}

#[test]
fn mean_abs_correction_reports_a_no_op_correction_as_zero() {
    let (cands, ev, base) = inputs();
    let out = subsystem(config(0.0))
        .compute(&query(), &cands, &ev, &base)
        .unwrap();
    assert_eq!(out.mean_abs_correction(), 0.0);
    let out = subsystem(config(0.5))
        .compute(&query(), &cands, &ev, &base)
        .unwrap();
    assert!(out.mean_abs_correction() > 0.0);
}

#[test]
fn disagreement_minimum_head_rule_is_respected() {
    // The subsystem must not silently substitute a variance when fewer than the
    // minimum heads are usable: the disagreement term contributes nothing, so
    // the correction equals the plain attention term.
    let cands = vec![candidate(1.0)];
    let one_head = vec![RetrievalEvidence {
        head_sims: vec![Some(1.0), None, None],
        head_ranks: vec![Some(0.5); HEADS],
        head_present: vec![true, false, false],
    }];
    let two_heads = vec![RetrievalEvidence {
        head_sims: vec![Some(1.0), Some(0.0), None],
        head_ranks: vec![Some(0.5); HEADS],
        head_present: vec![true, true, false],
    }];
    let cfg = || C8AttentionConfig {
        use_evidence: true,
        scorer: ResidualScorer::with_disagreement(1.0, 0.0, 1.0, 0.0),
        ..config(0.5)
    };

    let sparse = subsystem(cfg())
        .compute(&query(), &cands, &one_head, &[0.5])
        .unwrap();
    let dense = subsystem(cfg())
        .compute(&query(), &cands, &two_heads, &[0.5])
        .unwrap();
    assert!(two_heads[0]
        .head_similarity_variance(MIN_HEADS_FOR_DISAGREEMENT)
        .is_some());
    assert_ne!(
        sparse.attention_deltas, dense.attention_deltas,
        "defined vs undefined disagreement must differ"
    );
}

#[test]
fn cache_geometry_is_validated_after_warm() {
    let (cands, ev, base) = inputs();
    let sub = subsystem(config(0.2));
    let doc_ids = vec![0u64, 1, 2];
    let mut cache = AttentionKVCache::new(sub.cache_fingerprint());
    let mut stats = CacheStats::default();
    sub.compute_cached(
        &query(),
        &cands,
        &ev,
        &base,
        &doc_ids,
        &mut cache,
        &mut stats,
    )
    .unwrap();
    cache.validate_geometry().unwrap();
    assert_eq!(cache.len(), 3);
    assert_eq!(stats.entries, 3);
}

/// Bitwise comparison of two attention results, which is the strict parity
/// contract C8 needs (float equality via PartialEq would allow -0.0 == 0.0).
fn same_attention(a: &CandidateAttention, b: &CandidateAttention) -> bool {
    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }
    bits(&a.weights) == bits(&b.weights)
        && bits(&a.output) == bits(&b.output)
        && bits(&a.logits) == bits(&b.logits)
        && a.entropy.to_bits() == b.entropy.to_bits()
}

#[test]
fn engine_attend_and_attend_from_kv_agree_exactly() {
    // Guards the refactor that made the cached path share the engine math.
    let proj = attentiondb_attention::QkvProjection::identity(4);
    let engine = attentiondb_attention::AttentionEngine::new(proj.clone());
    let q_a = vec![0.1, 0.2, 0.3, 0.4];
    let z = vec![vec![1.0, 0.0, -1.0, 0.5], vec![0.2, 0.3, 0.4, 0.5]];
    let direct = engine.attend(&q_a, &z).unwrap();
    let k = proj.project_k(&z).unwrap();
    let v = proj.project_v(&z).unwrap();
    let split = engine.attend_from_kv(&q_a, z.len(), &k, &v).unwrap();
    assert!(same_attention(&direct, &split));
}

#[test]
fn attend_from_kv_rejects_bad_shapes() {
    let proj = attentiondb_attention::QkvProjection::identity(4);
    let engine = attentiondb_attention::AttentionEngine::new(proj);
    let q = vec![1.0; 4];
    assert!(engine.attend_from_kv(&q, 0, &[], &[]).is_err());
    assert!(engine
        .attend_from_kv(&q, 2, &[vec![1.0; 4]], &[vec![1.0; 4]])
        .is_err());
}

#[test]
fn candidate_attention_helpers_are_consistent() {
    // Sanity check on the engine output shape used by the traces.
    let a: CandidateAttention = attentiondb_attention::AttentionEngine::new(
        attentiondb_attention::QkvProjection::identity(4),
    )
    .attend(&[1.0; 4], &[vec![1.0; 4], vec![0.0; 4]])
    .unwrap();
    assert_eq!(a.weights.len(), 2);
    assert_eq!(a.output.len(), 4);
    assert_eq!(a.logits.len(), 2);
    let s: f32 = a.weights.iter().sum();
    assert!((s - 1.0).abs() < 1e-6);
}
