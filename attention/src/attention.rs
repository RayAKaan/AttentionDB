use crate::config::AttentionConfig;
use crate::errors::{AttentionError, Result};
use crate::qkv::{AttentionEngine, CandidateAttention};
use crate::scorer::RetrievalEvidence;
use serde::{Deserialize, Serialize};

/// Complete attention output for a candidate including diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionOutput {
    /// Per-candidate attention results.
    pub candidates: Vec<CandidateAttention>,
    /// Final attention-based scores s_d for each candidate (same order).
    pub scores: Vec<f32>,
    /// Per-head mean attention weights I_h = E[A_h] (over candidates).
    pub per_head_mean: Vec<f32>,
    /// Mean entropy across candidates.
    pub mean_entropy: f32,
    /// Total attention computation time (microseconds).
    pub compute_time_us: u64,
}

/// Full attention subsystem that combines alignment, QKV, and scoring.
pub struct AttentionSubsystem {
    config: AttentionConfig,
    engine: AttentionEngine,
    head_names: Vec<String>,
}

impl AttentionSubsystem {
    /// Create a new attention subsystem from a validated config.
    pub fn new(config: AttentionConfig, head_names: Vec<String>) -> Result<Self> {
        config.validate()?;
        let engine = AttentionEngine::new(config.qkv_projection.as_ref().unwrap().clone());
        Ok(Self {
            config,
            engine,
            head_names,
        })
    }

    /// Compute attention for all candidates and produce final scores.
    ///
    /// Inputs:
    ///   - query: original query vector (concatenated per-head or canonical)
    ///   - candidates: per-candidate per-head original vectors X_d ∈ R^{H × d_h}
    ///   - retrieval_evidence: per-candidate retrieval evidence r_d
    ///
    /// Returns: AttentionOutput with attention weights, outputs, and final scores.
    pub fn compute(
        &self,
        query: &[f32],
        candidates: &[Vec<Vec<f32>>],
        retrieval_evidence: &[RetrievalEvidence],
    ) -> Result<AttentionOutput> {
        if !self.config.enabled {
            return Err(AttentionError::Config("attention not enabled".into()));
        }
        if candidates.len() != retrieval_evidence.len() {
            return Err(AttentionError::DimensionMismatch {
                expected: candidates.len(),
                found: retrieval_evidence.len(),
            });
        }

        let start = std::time::Instant::now();

        // Align query: q_a = P_q(q)
        let q_a = self
            .config
            .query_alignment
            .as_ref()
            .unwrap()
            .project(query)?;

        // Align each candidate's per-head vectors: Z_d = [P_h(x_{d,h})]_h
        let mut aligned_candidates = Vec::with_capacity(candidates.len());
        for cand in candidates {
            if cand.len() != self.config.head_alignments.len() {
                return Err(AttentionError::DimensionMismatch {
                    expected: self.config.head_alignments.len(),
                    found: cand.len(),
                });
            }
            let mut z_d = Vec::with_capacity(cand.len());
            for (h, x_h) in cand.iter().enumerate() {
                let pa = &self.config.head_alignments[h];
                if x_h.len() != pa.input_dim {
                    return Err(AttentionError::DimensionMismatch {
                        expected: pa.input_dim,
                        found: x_h.len(),
                    });
                }
                z_d.push(pa.project(x_h)?);
            }
            aligned_candidates.push(z_d);
        }

        // Run attention engine batch
        let attention_results = self.engine.attend_batch(&q_a, &aligned_candidates)?;

        // Compute per-head mean attention
        let h = self.config.head_alignments.len();
        let mut per_head_sum = vec![0.0f32; h];
        let mut total_entropy = 0.0f32;
        for attn in &attention_results {
            for (i, &w) in attn.weights.iter().enumerate() {
                per_head_sum[i] += w;
            }
            total_entropy += attn.entropy;
        }
        let n = attention_results.len() as f32;
        let per_head_mean = if n > 0.0 {
            per_head_sum.iter().map(|&s| s / n).collect()
        } else {
            vec![0.0; h]
        };
        let mean_entropy = if n > 0.0 { total_entropy / n } else { 0.0 };

        // Score each candidate
        let mut scores = Vec::with_capacity(attention_results.len());
        for (attn, evidence) in attention_results.iter().zip(retrieval_evidence) {
            let score = self.config.scorer.score_with_query(attn, evidence, &q_a);
            scores.push(score);
        }

        let compute_time_us = start.elapsed().as_micros() as u64;

        Ok(AttentionOutput {
            candidates: attention_results,
            scores,
            per_head_mean,
            mean_entropy,
            compute_time_us,
        })
    }

    /// Get the head names.
    pub fn head_names(&self) -> &[String] {
        &self.head_names
    }
}
