use serde::{Deserialize, Serialize};

/// Default minimum number of usable heads before cross-head variance is defined.
pub const MIN_HEADS_FOR_DISAGREEMENT: usize = 2;

/// Retrieval evidence r_d for a candidate.
/// Contains per-head normalized similarity, rank, and presence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalEvidence {
    /// Per-head normalized similarity scores (missing = None).
    pub head_sims: Vec<Option<f32>>,
    /// Per-head rank features: 1/(1+rank) (missing = None).
    pub head_ranks: Vec<Option<f32>>,
    /// Per-head presence indicator.
    pub head_present: Vec<bool>,
}

impl RetrievalEvidence {
    /// Aggregate evidence into a single scalar (e.g., mean normalized sim).
    pub fn aggregate(&self) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0;
        for &s in &self.head_sims {
            if let Some(v) = s {
                sum += v;
                count += 1;
            }
        }
        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Per-head normalized similarities that are actually usable: the head must
    /// be flagged present *and* carry a finite score.
    ///
    /// A missing head is never imputed as `Some(0.0)`. Imputing zeros would
    /// invent disagreement out of absent data — a candidate that only one head
    /// returned would look maximally "disagreeing" purely because the other
    /// heads were silent.
    pub fn usable_head_sims(&self) -> Vec<f32> {
        let n = self
            .head_sims
            .len()
            .min(self.head_present.len())
            .min(self.head_ranks.len());
        let mut out = Vec::with_capacity(n);
        for h in 0..n {
            if self.head_present[h] {
                if let Some(v) = self.head_sims[h] {
                    if v.is_finite() {
                        out.push(v);
                    }
                }
            }
        }
        out
    }

    /// Number of heads contributing a usable similarity score.
    pub fn usable_head_count(&self) -> usize {
        self.usable_head_sims().len()
    }

    /// Cross-head disagreement `D(d) = Var(r_{d,1}, ..., r_{d,H})` over the
    /// present heads only, using the population variance (C8 decision Q4).
    ///
    /// Returns `None` when fewer than `min_heads` heads are usable, i.e. when
    /// the variance is not defined rather than zero.
    pub fn head_similarity_variance(&self, min_heads: usize) -> Option<f32> {
        let sims = self.usable_head_sims();
        if sims.len() < min_heads.max(2) {
            return None;
        }
        let n = sims.len() as f32;
        let mut sum = 0.0f32;
        for &v in &sims {
            sum += v;
        }
        let mean = sum / n;
        let mut var = 0.0f32;
        for &v in &sims {
            let d = v - mean;
            var += d * d;
        }
        Some(var / n)
    }

    /// Convenience wrapper using the default minimum-count rule.
    pub fn head_disagreement(&self) -> Option<f32> {
        self.head_similarity_variance(MIN_HEADS_FOR_DISAGREEMENT)
    }

    /// Highest per-head normalized similarity across usable heads.
    pub fn max_head_similarity(&self) -> Option<f32> {
        self.usable_head_sims()
            .into_iter()
            .fold(None, |acc: Option<f32>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            })
    }
}

/// Scorer F_θ(q_a, O_d, r_d) that combines attention output and retrieval evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionScorer {
    /// Weight for attention output dot-query term.
    pub w_attn: f32,
    /// Weight for aggregated retrieval evidence term.
    pub w_evidence: f32,
    /// Optional bias.
    pub bias: f32,
}

impl Default for AttentionScorer {
    fn default() -> Self {
        Self {
            w_attn: 1.0,
            w_evidence: 0.0,
            bias: 0.0,
        }
    }
}

impl AttentionScorer {
    /// Create a scorer with given weights.
    pub fn new(w_attn: f32, w_evidence: f32, bias: f32) -> Self {
        Self {
            w_attn,
            w_evidence,
            bias,
        }
    }

    /// Compute final score: s = w_attn * <q_a, O_d> + w_evidence * aggregate(r_d) + bias
    /// For C7-E (no evidence): s = w_attn * <q_a, O_d> + bias
    /// For C7-F (with evidence): s = w_attn * <q_a, O_d> + w_evidence * agg(r_d) + bias
    ///
    /// Note: <q_a, O_d> requires the aligned query, so prefer `score_with_query`.
    pub fn score(
        &self,
        attn: &crate::qkv::CandidateAttention,
        evidence: &RetrievalEvidence,
    ) -> f32 {
        let output_norm = attn.output.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let evidence_term = evidence.aggregate();
        self.w_attn * output_norm + self.w_evidence * evidence_term + self.bias
    }

    /// Score with explicit aligned query: s = w_attn * <q_a, O_d> + w_evidence * agg(r_d) + bias
    pub fn score_with_query(
        &self,
        attn: &crate::qkv::CandidateAttention,
        evidence: &RetrievalEvidence,
        q_a: &[f32],
    ) -> f32 {
        let dot = attn
            .output
            .iter()
            .zip(q_a)
            .map(|(&o, &q)| o * q)
            .sum::<f32>();
        let evidence_term = evidence.aggregate();
        self.w_attn * dot + self.w_evidence * evidence_term + self.bias
    }
}

/// C8 residual scorer: produces the *attention correction only*, never a final
/// score.
///
/// ```text
/// dS_attention = w_attn * <q_a, O_d> + w_evidence * agg(r_d) + w_disagree * D(r_d) + bias
/// S_final     = S_base + lambda * dS_attention
/// ```
///
/// The distinction from [`AttentionScorer`] is the whole point of C8: the
/// baseline score `S_base` is computed by the retrieval pipeline and must
/// survive untouched, so this type is only ever allowed to return a quantity
/// that gets *added* to it. With `lambda = 0` the correction is multiplied out
/// entirely and the baseline is reproduced bit-for-bit, which is C8's central
/// regression test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualScorer {
    /// Weight for the `<q_a, O_d>` attention-output term.
    pub w_attn: f32,
    /// Weight for the mean per-head retrieval evidence.
    pub w_evidence: f32,
    /// Weight for the cross-head disagreement variance (arm F variant).
    pub w_disagree: f32,
    /// Constant bias added to the correction.
    pub bias: f32,
}

impl Default for ResidualScorer {
    fn default() -> Self {
        Self {
            w_attn: 1.0,
            w_evidence: 0.0,
            w_disagree: 0.0,
            bias: 0.0,
        }
    }
}

impl ResidualScorer {
    pub fn new(w_attn: f32, w_evidence: f32, bias: f32) -> Self {
        Self {
            w_attn,
            w_evidence,
            w_disagree: 0.0,
            bias,
        }
    }

    /// Scorer variant that also consumes cross-head disagreement.
    pub fn with_disagreement(w_attn: f32, w_evidence: f32, w_disagree: f32, bias: f32) -> Self {
        Self {
            w_attn,
            w_evidence,
            w_disagree,
            bias,
        }
    }

    /// Whether any evidence term is active (C8-F). When false the evidence is
    /// not read at all, so arm E cannot accidentally depend on it.
    pub fn uses_evidence(&self) -> bool {
        self.w_evidence != 0.0 || self.w_disagree != 0.0
    }

    /// Compute the attention correction for one candidate.
    ///
    /// `q_a` is the aligned query. The disagreement term is skipped entirely
    /// when fewer than two heads are usable — undefined variance contributes
    /// nothing rather than a zero surrogate.
    pub fn attention_delta(
        &self,
        attn: &crate::qkv::CandidateAttention,
        evidence: &RetrievalEvidence,
        q_a: &[f32],
    ) -> f32 {
        let n = attn.output.len().min(q_a.len());
        let mut dot = 0.0f32;
        for i in 0..n {
            dot += q_a[i] * attn.output[i];
        }
        let mut delta = self.w_attn * dot;
        if self.w_evidence != 0.0 {
            delta += self.w_evidence * evidence.aggregate();
        }
        if self.w_disagree != 0.0 {
            if let Some(v) = evidence.head_disagreement() {
                delta += self.w_disagree * v;
            }
        }
        delta + self.bias
    }

    /// Attention-only correction (C8-E): ignores retrieval evidence entirely.
    pub fn attention_delta_without_evidence(
        &self,
        attn: &crate::qkv::CandidateAttention,
        q_a: &[f32],
    ) -> f32 {
        let n = attn.output.len().min(q_a.len());
        let mut dot = 0.0f32;
        for i in 0..n {
            dot += q_a[i] * attn.output[i];
        }
        self.w_attn * dot + self.bias
    }

    /// Combine a baseline score with the correction: `S_final = S_base + lambda * delta`.
    ///
    /// The `lambda == 0.0` branch returns `baseline_score` unchanged rather than
    /// computing `baseline + 0.0 * delta`, so a non-finite or NaN delta cannot
    /// contaminate the baseline. This is what makes the C8-A/B/E/F/G/H/I
    /// parity assertion exact instead of approximate.
    pub fn fuse(baseline_score: f32, delta: f32, residual_scale: f32) -> f32 {
        if residual_scale == 0.0 {
            return baseline_score;
        }
        baseline_score + residual_scale * delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qkv::CandidateAttention;

    fn attn(out: Vec<f32>, entropy: f32) -> CandidateAttention {
        let n = out.len();
        CandidateAttention {
            weights: vec![1.0 / n as f32; n],
            output: out,
            logits: vec![0.0; n],
            entropy,
        }
    }

    fn evidence(sims: &[Option<f32>]) -> RetrievalEvidence {
        RetrievalEvidence {
            head_sims: sims.to_vec(),
            head_ranks: vec![Some(0.5); sims.len()],
            head_present: sims.iter().map(|s| s.is_some()).collect(),
        }
    }

    #[test]
    fn delta_is_the_attention_term_only() {
        let s = ResidualScorer::new(0.5, 0.0, 0.1);
        let d = s.attention_delta(
            &attn(vec![2.0, 4.0], 0.0),
            &evidence(&[Some(1.0)]),
            &[1.0, 1.0],
        );
        assert!((d - (0.5 * 6.0 + 0.1)).abs() < 1e-6);
    }

    #[test]
    fn evidence_term_activates_only_when_weighted() {
        let e = evidence(&[Some(0.2), Some(0.4)]);
        let with = ResidualScorer::with_disagreement(0.0, 1.0, 0.0, 0.0);
        let without = ResidualScorer::with_disagreement(1.0, 0.0, 0.0, 0.0);
        let a = with.attention_delta(&attn(vec![1.0], 0.0), &e, &[1.0]);
        let b = without.attention_delta(&attn(vec![1.0], 0.0), &e, &[1.0]);
        assert!((a - 0.3).abs() < 1e-6);
        assert!((b - 1.0).abs() < 1e-6);
        assert!(with.uses_evidence());
        assert!(!without.uses_evidence());
    }

    #[test]
    fn disagreement_term_uses_variance_over_present_heads() {
        let s = ResidualScorer::with_disagreement(0.0, 0.0, 1.0, 0.0);
        let e = evidence(&[Some(1.0), Some(0.0)]);
        // Var([1, 0]) = 0.25
        let d = s.attention_delta(&attn(vec![0.0], 0.0), &e, &[1.0]);
        assert!((d - 0.25).abs() < 1e-6, "d={d}");
    }

    #[test]
    fn undefined_disagreement_contributes_nothing() {
        let s = ResidualScorer::with_disagreement(1.0, 0.0, 1.0, 0.0);
        let e = evidence(&[Some(0.7)]); // single usable head -> undefined
        let d = s.attention_delta(&attn(vec![2.0], 0.0), &e, &[1.0]);
        assert!((d - 2.0).abs() < 1e-6, "d={d}");
    }

    #[test]
    fn missing_heads_never_counted_as_zero_similarity() {
        let two_present = evidence(&[Some(1.0), Some(0.0)]);
        let one_absent = evidence(&[Some(1.0), None]);
        assert!(two_present.head_disagreement().unwrap() > 0.0);
        assert!(one_absent.head_disagreement().is_none());
        assert_eq!(two_present.usable_head_count(), 2);
        assert_eq!(one_absent.usable_head_count(), 1);
        // And the aggregate is over present heads only.
        assert!((one_absent.aggregate() - 1.0).abs() < 1e-6);
        assert!((two_present.aggregate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn disagreement_is_zero_for_identical_heads() {
        assert!(
            evidence(&[Some(0.4), Some(0.4), Some(0.4)])
                .head_disagreement()
                .unwrap()
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn fuse_with_zero_lambda_returns_baseline_exactly() {
        assert_eq!(ResidualScorer::fuse(0.1234567, f32::NAN, 0.0), 0.1234567);
        assert_eq!(ResidualScorer::fuse(-0.0, 1.0, 0.0), -0.0);
    }

    #[test]
    fn fuse_applies_lambda() {
        assert!((ResidualScorer::fuse(1.0, 2.0, 0.1) - 1.2).abs() < 1e-6);
        assert!((ResidualScorer::fuse(1.0, 2.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((ResidualScorer::fuse(1.0, 2.0, -0.5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn max_head_similarity_ignores_absent_heads() {
        assert!(
            (evidence(&[Some(0.2), None, Some(0.9)])
                .max_head_similarity()
                .unwrap()
                - 0.9)
                .abs()
                < 1e-6
        );
        assert!(evidence(&[None]).max_head_similarity().is_none());
    }
}
