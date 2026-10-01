use crate::errors::Result;
use crate::projection::QkvProjection;
use serde::{Deserialize, Serialize};

/// Numerically stable softmax: max-shift then exp then normalize.
pub fn stable_softmax(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - max_logit).exp();
        sum += *l;
    }
    if sum > 0.0 {
        for l in logits.iter_mut() {
            *l /= sum;
        }
    } else {
        // All logits were -inf or NaN — uniform fallback
        let uniform = 1.0 / logits.len() as f32;
        for l in logits.iter_mut() {
            *l = uniform;
        }
    }
}

/// Result of attention for a single candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateAttention {
    /// Attention weights A_d ∈ R^H (sum to 1, non-negative)
    pub weights: Vec<f32>,
    /// Attention output O_d ∈ R^{d_v}
    pub output: Vec<f32>,
    /// Raw logits before softmax (for diagnostics)
    pub logits: Vec<f32>,
    /// Entropy of the attention distribution: H(A) = -Σ A_h log A_h
    pub entropy: f32,
}

/// Core QKV attention engine for a single candidate.
/// Computes: Q = q_a W_Q, K_d = Z_d W_K, V_d = Z_d W_V
/// Then: L_d = Q K_d^T / sqrt(d_k), A_d = softmax(L_d), O_d = A_d V_d
pub struct AttentionEngine {
    qkv: QkvProjection,
    key_dim: usize,
}

impl AttentionEngine {
    /// Create a new attention engine from Q/K/V projections.
    pub fn new(qkv: QkvProjection) -> Self {
        let key_dim = qkv.key_dim;
        Self { qkv, key_dim }
    }

    /// Compute attention for a single candidate.
    ///
    /// Inputs:
    ///   - q_a: aligned query vector ∈ R^{d_a}
    ///   - z_d: aligned candidate representations Z_d ∈ R^{H × d_a}
    ///     (one row per retrieval head)
    ///
    /// Returns: CandidateAttention with weights, output, logits, entropy.
    pub fn attend(&self, q_a: &[f32], z_d: &[Vec<f32>]) -> Result<CandidateAttention> {
        if z_d.is_empty() {
            return Err(crate::errors::AttentionError::EmptyInput(
                "z_d cannot be empty".into(),
            ));
        }
        let q = self.qkv.project_q(q_a)?;
        let k_d = self.qkv.project_k(z_d)?;
        let v_d = self.qkv.project_v(z_d)?;
        self.attend_from_kv(&q, z_d.len(), &k_d, &v_d)
    }

    /// Project the query once so it can be reused across candidates.
    /// This is the only query-dependent step; everything after it is
    /// candidate-local, which is what makes a document-side K/V cache valid.
    pub fn project_query(&self, q_a: &[f32]) -> Result<Vec<f32>> {
        self.qkv.project_q(q_a)
    }

    /// Attention from *already projected* keys and values.
    ///
    /// This is the single implementation of the attention math, shared by the
    /// uncached path (which projects K/V first) and the C8 cached path (which
    /// reuses vectors cached from a previous query). Because both call this
    /// function with identical inputs, the two paths are bit-identical by
    /// construction rather than by agreement between two code paths.
    pub fn attend_from_kv(
        &self,
        q: &[f32],
        head_count: usize,
        k_d: &[Vec<f32>],
        v_d: &[Vec<f32>],
    ) -> Result<CandidateAttention> {
        if head_count == 0 {
            return Err(crate::errors::AttentionError::EmptyInput(
                "z_d cannot be empty".into(),
            ));
        }
        if k_d.len() != head_count || v_d.len() != head_count {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: head_count,
                found: k_d.len().min(v_d.len()),
            });
        }
        let d_k = self.key_dim;
        let value_dim = self.qkv.value_dim;

        // Compute logits: L_d = Q K_d^T / sqrt(d_k)
        // L_d is 1 × H
        let scale = 1.0 / (d_k as f32).sqrt();
        let mut logits = vec![0.0; head_count];
        for h in 0..head_count {
            let mut dot = 0.0;
            let k_h = &k_d[h];
            for d in 0..d_k {
                dot += q[d] * k_h[d];
            }
            logits[h] = dot * scale;
        }

        // Stable softmax
        stable_softmax(&mut logits);

        // Attention output: O_d = A_d V_d
        // A_d is 1 × H, V_d is H × d_v → O_d is 1 × d_v
        let mut output = vec![0.0; value_dim];
        for h in 0..head_count {
            let a_h = logits[h];
            let v_h = &v_d[h];
            for d in 0..value_dim {
                output[d] += a_h * v_h[d];
            }
        }

        // Entropy: H(A) = -Σ A_h log A_h
        let mut entropy = 0.0f32;
        for &a in &logits {
            if a > 0.0 {
                entropy -= a * a.ln();
            }
        }

        Ok(CandidateAttention {
            weights: logits.clone(),
            output,
            logits,
            entropy,
        })
    }

    /// Batch attend for multiple candidates with the same query.
    /// Returns a vector of CandidateAttention, one per candidate.
    pub fn attend_batch(
        &self,
        q_a: &[f32],
        candidates: &[Vec<Vec<f32>>],
    ) -> Result<Vec<CandidateAttention>> {
        candidates.iter().map(|z_d| self.attend(q_a, z_d)).collect()
    }

    /// Get the key dimension.
    pub fn key_dim(&self) -> usize {
        self.key_dim
    }
}
