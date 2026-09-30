use serde::{Deserialize, Serialize};

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
