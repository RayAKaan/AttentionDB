use crate::qkv::CandidateAttention;
use serde::{Deserialize, Serialize};

/// Attention diagnostics for a batch of candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDiagnostics {
    /// Mean normalized entropy across candidates: H(A) / log(H)
    pub mean_normalized_entropy: f32,
    /// Per-head mean attention weight: I_h = E[A_h]
    pub per_head_mean: Vec<f32>,
    /// Per-head attention variance: Var[A_h]
    pub per_head_variance: Vec<f32>,
    /// Fraction of candidates where each head has max weight (selection frequency).
    pub selection_frequency: Vec<f32>,
    /// Attention concentration: max_h I_h (0 = uniform, 1 = collapsed).
    pub concentration: f32,
    /// Mean attention entropy (unnormalized).
    pub mean_entropy: f32,
    /// Min entropy observed.
    pub min_entropy: f32,
    /// Max entropy observed.
    pub max_entropy: f32,
    /// Total number of candidates analyzed.
    pub candidate_count: usize,
}

impl AttentionDiagnostics {
    /// Compute diagnostics from a batch of candidate attention results.
    pub fn from_candidates(attentions: &[CandidateAttention]) -> Self {
        let n = attentions.len();
        if n == 0 {
            return Self {
                mean_normalized_entropy: f32::NAN,
                per_head_mean: Vec::new(),
                per_head_variance: Vec::new(),
                selection_frequency: Vec::new(),
                concentration: f32::NAN,
                mean_entropy: f32::NAN,
                min_entropy: f32::NAN,
                max_entropy: f32::NAN,
                candidate_count: 0,
            };
        }

        let h_count = attentions[0].weights.len();
        let mut per_head_sum = vec![0.0f32; h_count];
        let mut per_head_sq_sum = vec![0.0f32; h_count];
        let mut selection_counts = vec![0usize; h_count];
        let mut total_entropy = 0.0f32;
        let mut min_entropy = f32::INFINITY;
        let mut max_entropy = f32::NEG_INFINITY;

        for attn in attentions {
            let weights = &attn.weights;
            for (h, &w) in weights.iter().enumerate() {
                per_head_sum[h] += w;
                per_head_sq_sum[h] += w * w;
            }
            // Selection frequency
            let max_idx = weights
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            selection_counts[max_idx] += 1;

            total_entropy += attn.entropy;
            min_entropy = min_entropy.min(attn.entropy);
            max_entropy = max_entropy.max(attn.entropy);
        }

        let n_f = n as f32;
        let per_head_mean: Vec<f32> = per_head_sum.iter().map(|&s| s / n_f).collect();
        let per_head_variance: Vec<f32> = per_head_sq_sum
            .iter()
            .zip(per_head_mean.iter())
            .map(|(&sq, &m)| (sq / n_f) - m * m)
            .collect();
        let selection_frequency: Vec<f32> =
            selection_counts.iter().map(|&c| c as f32 / n_f).collect();
        let mean_entropy = total_entropy / n_f;
        let concentration = per_head_mean.iter().copied().fold(0.0, f32::max);
        let log_h = (h_count as f32).ln();
        let mean_normalized_entropy = if log_h > 0.0 {
            mean_entropy / log_h
        } else {
            f32::NAN
        };

        Self {
            mean_normalized_entropy,
            per_head_mean,
            per_head_variance,
            selection_frequency,
            concentration,
            mean_entropy,
            min_entropy,
            max_entropy,
            candidate_count: n,
        }
    }

    /// Check if attention is collapsed to uniform (all heads equal).
    pub fn is_uniform(&self, eps: f32) -> bool {
        self.mean_normalized_entropy >= 1.0 - eps
    }

    /// Check if attention is collapsed to single head.
    pub fn is_collapsed(&self, eps: f32) -> bool {
        self.concentration >= 1.0 - eps
    }
}
