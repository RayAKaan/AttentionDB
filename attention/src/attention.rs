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

// ============================ C8 residual attention ============================

use crate::cache::{AttentionKVCache, CacheFingerprint, CacheStats, CachedCandidateKV};
use crate::config::C8AttentionConfig;
use crate::scorer::ResidualScorer;

/// Per-stage C8 timing breakdown.
///
/// The point of these buckets is to make the latency claim falsifiable. C7's
/// attention was dominated by projecting every candidate through W_K and W_V on
/// every query; the cache arm (I) must show that bucket collapsing while the
/// alignment and fusion buckets stay flat, and the dimension study must show
/// the projection bucket shrinking with d_k. A single total would hide both.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct C8AttentionTimings {
    /// Aligning the query vector.
    pub query_alignment_us: u64,
    /// Aligning all candidate per-head vectors.
    pub candidate_alignment_us: u64,
    /// Projecting Q (once per query).
    pub q_projection_us: u64,
    /// Projecting K/V for candidates. The cacheable bucket: zero on a warm cache.
    pub kv_projection_us: u64,
    /// Time spent looking documents up in the cache.
    pub cache_lookup_us: u64,
    /// Logits + softmax + value aggregation.
    pub attention_us: u64,
    /// Residual scoring and S_base + lambda*dS fusion.
    pub residual_fusion_us: u64,
    /// Whole-subsystem wall time.
    pub total_us: u64,
}

/// One candidate's C8 outcome, keeping baseline and correction separable.
///
/// `baseline_score` and `attention_delta` are both preserved alongside
/// `final_score` so a probe can verify the fusion identity
/// `final = base + scale*delta` per candidate instead of trusting a single
/// aggregate number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct C8CandidateScore {
    /// Baseline S_base exactly as the retrieval pipeline produced it.
    pub baseline_score: f32,
    /// Attention correction dS_attention (pre-scale).
    pub attention_delta: f32,
    /// Residual scale lambda applied to the correction.
    pub residual_scale: f32,
    /// S_final = S_base + lambda * dS_attention.
    pub final_score: f32,
    /// Per-head attention weights A_d.
    pub weights: Vec<f32>,
    /// Attention output O_d.
    pub output: Vec<f32>,
    /// Raw attention logits.
    pub logits: Vec<f32>,
    /// Entropy of the attention distribution.
    pub entropy: f32,
    /// S_final - S_base as actually applied. Exactly zero when lambda is zero.
    pub applied_correction: f32,
}

/// Complete C8 output for one query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C8AttentionOutput {
    /// Per-candidate results, aligned with the candidate list.
    pub candidates: Vec<C8CandidateScore>,
    /// S_final per candidate, same order.
    pub final_scores: Vec<f32>,
    /// S_base per candidate, same order.
    pub baseline_scores: Vec<f32>,
    /// dS_attention per candidate, same order.
    pub attention_deltas: Vec<f32>,
    /// Per-head mean attention weight I_h.
    pub per_head_mean: Vec<f32>,
    /// Mean attention entropy across candidates.
    pub mean_entropy: f32,
    /// Timing breakdown.
    pub timings: C8AttentionTimings,
    /// Cache accounting (arm I).
    pub cache_stats: CacheStats,
    /// Residual scale in force.
    pub residual_scale: f32,
    /// Config fingerprint of the subsystem that produced this output.
    pub config_fingerprint: u64,
}

impl C8AttentionOutput {
    /// Mean absolute applied correction across candidates. Surfaced in the probe
    /// JSON so a correction that does nothing is visible in the artifact rather
    /// than inferred from the metrics.
    pub fn mean_abs_correction(&self) -> f32 {
        if self.candidates.is_empty() {
            return 0.0;
        }
        let s: f32 = self
            .candidates
            .iter()
            .map(|c| c.applied_correction.abs())
            .sum();
        s / self.candidates.len() as f32
    }

    /// Verify the fusion identity for every candidate. Used by the parity tests
    /// and the verifier so a broken fusion cannot hide behind good metrics.
    pub fn verify_fusion(&self, tolerance: f32) -> Result<()> {
        for (i, c) in self.candidates.iter().enumerate() {
            let expected = if c.residual_scale == 0.0 {
                c.baseline_score
            } else {
                c.baseline_score + c.residual_scale * c.attention_delta
            };
            if (c.final_score - expected).abs() > tolerance {
                return Err(crate::errors::AttentionError::Config(format!(
                    "candidate {i}: final {} != base {} + {} * delta {}",
                    c.final_score, c.baseline_score, c.residual_scale, c.attention_delta
                )));
            }
        }
        Ok(())
    }
}

/// C8 residual attention subsystem: attention engine + residual scorer.
///
/// Unlike the C7 subsystem, this one *requires* baseline scores, because the
/// whole contract is that it adds a correction to them and returns both.
pub struct C8AttentionSubsystem {
    config: C8AttentionConfig,
    engine: AttentionEngine,
    projection: crate::projection::QkvProjection,
    head_names: Vec<String>,
    fingerprint: u64,
}

/// Aligned per-head candidate representations: one `H x d_a` matrix per candidate.
type AlignedCandidates = Vec<Vec<Vec<f32>>>;

/// Per-candidate projected keys and values: `(H x d_k, H x d_v)`.
type ProjectedKv = (Vec<Vec<f32>>, Vec<Vec<f32>>);

impl C8AttentionSubsystem {
    /// Build from a validated C8 config.
    pub fn new(config: C8AttentionConfig, head_names: Vec<String>) -> Result<Self> {
        config.validate()?;
        let fingerprint = config.fingerprint();
        // A disabled config carries no dimensions, so it has no projection to
        // materialize. Build an inert 1x1 placeholder: every compute method
        // rejects the subsystem before the engine is ever used, and a disabled
        // subsystem must still be constructible so callers can hold one
        // uniformly without a special case.
        let projection = if config.enabled {
            config.effective_projection()?.to_qkv()
        } else {
            crate::projection::QkvProjection::identity(1)
        };
        let engine = AttentionEngine::new(projection.clone());
        Ok(Self {
            config,
            engine,
            projection,
            head_names,
            fingerprint,
        })
    }

    pub fn config(&self) -> &C8AttentionConfig {
        &self.config
    }

    pub fn head_names(&self) -> &[String] {
        &self.head_names
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The effective (materialized) projection, for cache construction.
    pub fn projection(&self) -> &crate::projection::QkvProjection {
        &self.projection
    }

    /// Fingerprint a cache must carry to be valid for this subsystem.
    pub fn cache_fingerprint(&self) -> CacheFingerprint {
        CacheFingerprint::from_projection(
            self.fingerprint,
            &self.projection,
            self.config.head_alignments.len(),
        )
    }

    /// Align the query and all candidates. Shared by both compute paths so the
    /// cached and uncached arms align identically.
    #[allow(clippy::type_complexity)]
    fn align(
        &self,
        query: &[f32],
        candidates: &[Vec<Vec<f32>>],
        timings: &mut C8AttentionTimings,
    ) -> Result<(Vec<f32>, AlignedCandidates)> {
        let t0 = std::time::Instant::now();
        let q_a = self
            .config
            .query_alignment
            .as_ref()
            .ok_or_else(|| crate::errors::AttentionError::Config("query_alignment missing".into()))?
            .project(query)?;
        timings.query_alignment_us = t0.elapsed().as_micros() as u64;

        let t1 = std::time::Instant::now();
        let mut aligned = Vec::with_capacity(candidates.len());
        for cand in candidates {
            if cand.len() != self.config.head_alignments.len() {
                return Err(crate::errors::AttentionError::DimensionMismatch {
                    expected: self.config.head_alignments.len(),
                    found: cand.len(),
                });
            }
            let mut z_d = Vec::with_capacity(cand.len());
            for (h, x_h) in cand.iter().enumerate() {
                z_d.push(self.config.head_alignments[h].project(x_h)?);
            }
            aligned.push(z_d);
        }
        timings.candidate_alignment_us = t1.elapsed().as_micros() as u64;
        Ok((q_a, aligned))
    }

    /// Fuse baseline scores with attention corrections into C8 candidate scores.
    fn fuse(
        &self,
        q_a: &[f32],
        results: &[CandidateAttention],
        evidence: &[RetrievalEvidence],
        baseline_scores: &[f32],
        timings: &mut C8AttentionTimings,
    ) -> Vec<C8CandidateScore> {
        let t0 = std::time::Instant::now();
        let read_evidence = self.config.use_evidence && self.config.scorer.uses_evidence();
        let mut out = Vec::with_capacity(results.len());
        for ((a, e), &base) in results.iter().zip(evidence).zip(baseline_scores) {
            // Arm E must not even look at the evidence, so the two arms are
            // genuinely different computations, not one with a zero weight.
            let delta = if read_evidence {
                self.config.scorer.attention_delta(a, e, q_a)
            } else {
                self.config.scorer.attention_delta_without_evidence(a, q_a)
            };
            let final_score = ResidualScorer::fuse(base, delta, self.config.residual_scale);
            out.push(C8CandidateScore {
                baseline_score: base,
                attention_delta: delta,
                residual_scale: self.config.residual_scale,
                final_score,
                weights: a.weights.clone(),
                output: a.output.clone(),
                logits: a.logits.clone(),
                entropy: a.entropy,
                applied_correction: final_score - base,
            });
        }
        timings.residual_fusion_us = t0.elapsed().as_micros() as u64;
        out
    }

    /// The input contract shared by both compute paths.
    fn check_inputs(
        &self,
        candidates: &[Vec<Vec<f32>>],
        evidence: &[RetrievalEvidence],
        baseline_scores: &[f32],
    ) -> Result<()> {
        if !self.config.enabled {
            return Err(crate::errors::AttentionError::Config(
                "C8 attention not enabled".into(),
            ));
        }
        if candidates.len() != evidence.len() {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: candidates.len(),
                found: evidence.len(),
            });
        }
        if candidates.len() != baseline_scores.len() {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: candidates.len(),
                found: baseline_scores.len(),
            });
        }
        Ok(())
    }

    fn finish(
        &self,
        fused: Vec<C8CandidateScore>,
        timings: C8AttentionTimings,
        cache_stats: CacheStats,
    ) -> C8AttentionOutput {
        let h = self.config.head_alignments.len();
        let n = fused.len() as f32;
        let mut per_head_sum = vec![0.0f32; h];
        let mut total_entropy = 0.0f32;
        for c in &fused {
            for (i, &w) in c.weights.iter().enumerate() {
                per_head_sum[i] += w;
            }
            total_entropy += c.entropy;
        }
        let per_head_mean = if n > 0.0 {
            per_head_sum.iter().map(|&s| s / n).collect()
        } else {
            vec![0.0; h]
        };
        let mean_entropy = if n > 0.0 { total_entropy / n } else { 0.0 };
        C8AttentionOutput {
            final_scores: fused.iter().map(|c| c.final_score).collect(),
            baseline_scores: fused.iter().map(|c| c.baseline_score).collect(),
            attention_deltas: fused.iter().map(|c| c.attention_delta).collect(),
            candidates: fused,
            per_head_mean,
            mean_entropy,
            timings,
            cache_stats,
            residual_scale: self.config.residual_scale,
            config_fingerprint: self.fingerprint,
        }
    }

    /// Compute the C8 residual correction and the resulting final scores.
    ///
    /// `baseline_scores[i]` is S_base for `candidates[i]` and is passed through
    /// untouched. This is the uncached path (arms C/D/E/F/G/H).
    pub fn compute(
        &self,
        query: &[f32],
        candidates: &[Vec<Vec<f32>>],
        retrieval_evidence: &[RetrievalEvidence],
        baseline_scores: &[f32],
    ) -> Result<C8AttentionOutput> {
        self.check_inputs(candidates, retrieval_evidence, baseline_scores)?;
        let t_start = std::time::Instant::now();
        let mut timings = C8AttentionTimings::default();
        let (q_a, aligned) = self.align(query, candidates, &mut timings)?;

        let t_q = std::time::Instant::now();
        let q = self.engine.project_query(&q_a)?;
        timings.q_projection_us = t_q.elapsed().as_micros() as u64;

        let t_kv = std::time::Instant::now();
        let mut projected: Vec<ProjectedKv> = Vec::with_capacity(aligned.len());
        for z_d in &aligned {
            projected.push((
                self.projection.project_k(z_d)?,
                self.projection.project_v(z_d)?,
            ));
        }
        timings.kv_projection_us = t_kv.elapsed().as_micros() as u64;

        let t_attn = std::time::Instant::now();
        let mut results = Vec::with_capacity(aligned.len());
        for (i, z_d) in aligned.iter().enumerate() {
            results.push(self.engine.attend_from_kv(
                &q,
                z_d.len(),
                &projected[i].0,
                &projected[i].1,
            )?);
        }
        timings.attention_us = t_attn.elapsed().as_micros() as u64;

        let fused = self.fuse(
            &q_a,
            &results,
            retrieval_evidence,
            baseline_scores,
            &mut timings,
        );
        timings.total_us = t_start.elapsed().as_micros() as u64;
        Ok(self.finish(fused, timings, CacheStats::default()))
    }

    /// Compute using the in-memory document-side K/V cache (arm I).
    ///
    /// `doc_ids[i]` identifies `candidates[i]` for cache lookup. Results are
    /// bit-identical to [`Self::compute`]: the cache only skips a pure function
    /// of (document, model), and both paths finish in
    /// [`AttentionEngine::attend_from_kv`].
    #[allow(clippy::too_many_arguments)]
    pub fn compute_cached(
        &self,
        query: &[f32],
        candidates: &[Vec<Vec<f32>>],
        retrieval_evidence: &[RetrievalEvidence],
        baseline_scores: &[f32],
        doc_ids: &[u64],
        cache: &mut AttentionKVCache,
        stats: &mut CacheStats,
    ) -> Result<C8AttentionOutput> {
        self.check_inputs(candidates, retrieval_evidence, baseline_scores)?;
        if candidates.len() != doc_ids.len() {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: candidates.len(),
                found: doc_ids.len(),
            });
        }
        // Refuse to serve stale vectors rather than returning quietly wrong ranks.
        if !cache.is_valid_for(&self.cache_fingerprint()) {
            return Err(crate::errors::AttentionError::Config(
                "C8 cache fingerprint mismatch: refusing to serve stale K/V".into(),
            ));
        }

        let t_start = std::time::Instant::now();
        let mut timings = C8AttentionTimings::default();
        let (q_a, aligned) = self.align(query, candidates, &mut timings)?;

        let t_q = std::time::Instant::now();
        let q = self.engine.project_query(&q_a)?;
        timings.q_projection_us = t_q.elapsed().as_micros() as u64;

        let t_lookup = std::time::Instant::now();
        let t_kv = std::time::Instant::now();
        let mut results = Vec::with_capacity(aligned.len());
        for (i, z_d) in aligned.iter().enumerate() {
            let kv: CachedCandidateKV = match cache.lookup(doc_ids[i], stats) {
                Some(c) => c.clone(),
                None => {
                    let built =
                        CachedCandidateKV::project(z_d, &self.projection).ok_or_else(|| {
                            crate::errors::AttentionError::EmptyInput("empty candidate".into())
                        })?;
                    cache.insert(doc_ids[i], built.clone());
                    built
                }
            };
            results.push(
                self.engine
                    .attend_from_kv(&q, z_d.len(), &kv.keys, &kv.values)?,
            );
        }
        // The two clocks interleave, so report the lookup phase as the total and
        // the projection bucket only for the misses it actually paid for.
        let lookup_total = t_lookup.elapsed().as_micros() as u64;
        timings.cache_lookup_us = lookup_total;
        timings.kv_projection_us = if stats.misses == 0 {
            0
        } else {
            std::cmp::min(t_kv.elapsed().as_micros() as u64, lookup_total)
        };
        let attn_us = t_kv.elapsed().as_micros() as u64;
        timings.attention_us = attn_us.saturating_sub(timings.kv_projection_us);
        cache.seal_stats(stats);

        let fused = self.fuse(
            &q_a,
            &results,
            retrieval_evidence,
            baseline_scores,
            &mut timings,
        );
        timings.total_us = t_start.elapsed().as_micros() as u64;
        Ok(self.finish(fused, timings, *stats))
    }
}
