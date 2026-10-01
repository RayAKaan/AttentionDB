//! C8 residual trainer: real mini-batches, residual regularization, and
//! candidate-set softmax distillation.
//!
//! ```text
//! L = L_contrastive + alpha_r * L_residual + beta * L_distill
//!
//! L_contrastive = InfoNCE over S_final within a query's candidate set
//!                 (positive + mined hard negatives)
//! L_residual    = mean(dW^2)                        [learned correction only]
//! L_distill     = KL(softmax(S_base/tau_d) || softmax(S_final/tau_d))
//! ```
//!
//! Three properties distinguish this trainer from the C7 one, and each exists
//! because of a specific way C7's training was inadequate:
//!
//! 1. **Real mini-batches.** C7 applied one Adam step per example, so
//!    `batch_size` was only a number in the model card. C8 accumulates gradients
//!    over `batch_size` examples and steps once per batch, so the recorded batch
//!    size is the batch size that actually ran.
//!
//! 2. **The baseline is part of the loss.** `S_final = S_base + lambda*dS`, and
//!    `S_base` is a constant, not a free parameter. The contrastive gradient
//!    therefore reaches the QKV weights only through `lambda * dS`, which is
//!    exactly the quantity the residual contract is about.
//!
//! 3. **Distillation as a distribution, not a score.** `KL(T || P)` over
//!    candidate-set softmaxes preserves the baseline's *relative preference
//!    structure*, which min-max MSE destroys. The distillation temperature
//!    `tau_d` is a validated hyperparameter, never a hard-coded constant.
//!
//! Determinism: no RNG at any point. Batches are formed by a fixed stride over a
//! seeded Fisher-Yates permutation (same seed, same order), gradients accumulate
//! in a fixed index order, and Adam is the deterministic variant. The same
//! dataset, seed, and config reproduce bit-identical weights.

use crate::distillation::{candidate_softmax, distillation_gradient, distillation_loss};
use crate::errors::{AttentionError, Result};
use crate::model::{C8ModelCard, C8ResidualMeta, TrainingMeta};
use crate::projection::ResidualQkvProjection;
use crate::scorer::ResidualScorer;
use crate::training::candidate_gradient;
use serde::{Deserialize, Serialize};

/// One C8 training example: a query, its candidate set, the frozen baseline
/// scores for that set, and the evidence each candidate carries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C8QkvExample {
    /// Aligned query `q_a ∈ R^{d_a}`.
    pub q_a: Vec<f32>,
    /// Positive candidate `Z_d^+ ∈ R^{H × d_a}`.
    pub positive: Vec<Vec<f32>>,
    /// Frozen baseline score `S_base` of the positive.
    pub positive_baseline: f32,
    /// Full retrieval evidence of the positive (per-head vectors preserved).
    pub positive_evidence: Option<crate::scorer::RetrievalEvidence>,
    /// Mined negatives `Z_d^- ∈ R^{k × H × d_a}`.
    pub negatives: Vec<Vec<Vec<f32>>>,
    /// Frozen baseline score of each negative, same order.
    pub negatives_baseline: Vec<f32>,
    /// Full retrieval evidence of each negative, same order.
    pub negatives_evidence: Vec<Option<crate::scorer::RetrievalEvidence>>,
    /// Provenance for each mined negative (mirrors `HardNegative`).
    pub negative_sources: Vec<crate::negative_mining::NegSource>,
}

impl C8QkvExample {
    /// Number of candidates in the set (positive + negatives).
    pub fn candidate_count(&self) -> usize {
        1 + self.negatives.len()
    }

    /// Baseline scores of the whole candidate set, positive first.
    pub fn all_baselines(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.candidate_count());
        v.push(self.positive_baseline);
        v.extend_from_slice(&self.negatives_baseline);
        v
    }

    /// Evidence of the whole candidate set, positive first.
    pub fn all_evidence(&self) -> Vec<crate::scorer::RetrievalEvidence> {
        let empty = crate::scorer::RetrievalEvidence {
            head_sims: Vec::new(),
            head_ranks: Vec::new(),
            head_present: Vec::new(),
        };
        let mut v = Vec::with_capacity(self.candidate_count());
        v.push(
            self.positive_evidence
                .clone()
                .unwrap_or_else(|| empty.clone()),
        );
        for e in &self.negatives_evidence {
            v.push(e.clone().unwrap_or_else(|| empty.clone()));
        }
        v
    }

    /// A negative's aggregated evidence scalar, or `None` when absent. This is
    /// what the analytic gradient consumes (the evidence term is scalar).
    pub fn negative_evidence_scalars(&self) -> Vec<Option<f32>> {
        self.negatives_evidence
            .iter()
            .map(|e| e.as_ref().map(crate::scorer::RetrievalEvidence::aggregate))
            .collect()
    }

    /// A negative's cross-head disagreement, or `None` when undefined.
    pub fn negative_disagreements(&self) -> Vec<Option<f32>> {
        self.negatives_evidence
            .iter()
            .map(|e| {
                e.as_ref()
                    .and_then(crate::scorer::RetrievalEvidence::head_disagreement)
            })
            .collect()
    }

    pub fn positive_disagreement(&self) -> Option<f32> {
        self.positive_evidence
            .as_ref()
            .and_then(crate::scorer::RetrievalEvidence::head_disagreement)
    }

    /// Structural validation; dimension checks happen in the builder.
    pub fn validate(&self) -> Result<()> {
        if self.q_a.is_empty() {
            return Err(AttentionError::EmptyInput("C8 q_a empty".into()));
        }
        if self.positive.is_empty() {
            return Err(AttentionError::EmptyInput("C8 positive empty".into()));
        }
        if self.negatives_baseline.len() != self.negatives.len() {
            return Err(AttentionError::DimensionMismatch {
                expected: self.negatives.len(),
                found: self.negatives_baseline.len(),
            });
        }
        if self.negatives_evidence.len() != self.negatives.len() {
            return Err(AttentionError::DimensionMismatch {
                expected: self.negatives.len(),
                found: self.negatives_evidence.len(),
            });
        }
        if self.negative_sources.len() != self.negatives.len() {
            return Err(AttentionError::DimensionMismatch {
                expected: self.negatives.len(),
                found: self.negative_sources.len(),
            });
        }
        if !self.positive_baseline.is_finite() {
            return Err(AttentionError::NonFinite("C8 positive baseline".into()));
        }
        for (i, b) in self.negatives_baseline.iter().enumerate() {
            if !b.is_finite() {
                return Err(AttentionError::NonFinite(format!(
                    "C8 negative {i} baseline"
                )));
            }
        }
        Ok(())
    }
}

/// A C8 training dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C8QkvDataset {
    pub examples: Vec<C8QkvExample>,
    /// Attention dimension d_a of every aligned vector.
    pub attention_dim: usize,
    /// Retrieval head names in order.
    pub head_names: Vec<String>,
    /// Digest of the mined-negative provenance, recorded in the model card so a
    /// model can be traced back to the exact negative set that produced it.
    pub negative_provenance_hash: u64,
}

impl C8QkvDataset {
    /// FNV-1a fingerprint over the serialized dataset.
    pub fn fingerprint(&self) -> u64 {
        let json = serde_json::to_string(self).unwrap_or_default();
        let mut hash = 0xcbf29ce484222325u64;
        for b in json.as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
}

/// Builder that validates every aligned dimension.
#[derive(Debug, Clone)]
pub struct C8QkvDatasetBuilder {
    examples: Vec<C8QkvExample>,
    attention_dim: usize,
    head_names: Vec<String>,
    negative_provenance_hash: u64,
}

impl C8QkvDatasetBuilder {
    pub fn new(attention_dim: usize, head_names: Vec<String>) -> Self {
        Self {
            examples: Vec::new(),
            attention_dim,
            head_names,
            negative_provenance_hash: 0xcbf29ce484222325,
        }
    }

    pub fn push(&mut self, ex: C8QkvExample) -> Result<()> {
        ex.validate()?;
        let heads = self.head_names.len();
        if ex.positive.len() != heads {
            return Err(AttentionError::DimensionMismatch {
                expected: heads,
                found: ex.positive.len(),
            });
        }
        if ex.q_a.len() != self.attention_dim {
            return Err(AttentionError::DimensionMismatch {
                expected: self.attention_dim,
                found: ex.q_a.len(),
            });
        }
        for v in &ex.positive {
            Self::check(v, self.attention_dim)?;
        }
        for neg in &ex.negatives {
            if neg.len() != heads {
                return Err(AttentionError::DimensionMismatch {
                    expected: heads,
                    found: neg.len(),
                });
            }
            for v in neg {
                Self::check(v, self.attention_dim)?;
            }
        }
        // Fold provenance into the builder's running digest: the baseline the
        // negatives were mined against, plus which stratum each came from.
        let mut hash = self.negative_provenance_hash;
        let mut mix = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        };
        for b in ex.all_baselines() {
            mix(&b.to_le_bytes());
        }
        for s in &ex.negative_sources {
            mix(&(s.stratum() as f32).to_le_bytes());
        }
        self.negative_provenance_hash = hash;
        self.examples.push(ex);
        Ok(())
    }

    fn check(v: &[f32], dim: usize) -> Result<()> {
        if v.len() != dim {
            return Err(AttentionError::DimensionMismatch {
                expected: dim,
                found: v.len(),
            });
        }
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return Err(AttentionError::NonFinite(format!("aligned vector[{i}]")));
        }
        Ok(())
    }

    pub fn build(self) -> C8QkvDataset {
        C8QkvDataset {
            examples: self.examples,
            attention_dim: self.attention_dim,
            head_names: self.head_names,
            negative_provenance_hash: self.negative_provenance_hash,
        }
    }
}

/// C8 training hyper-parameters. Every field is a validated hyper-parameter:
/// nothing here is a hard-coded constant that C8 claims to have chosen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C8TrainingConfig {
    pub seed: u64,
    pub learning_rate: f32,
    pub epochs: usize,
    /// Examples per optimizer step. This is the batch size that actually runs.
    pub batch_size: usize,
    /// InfoNCE temperature.
    pub temperature: f32,
    /// Plain L2 on the learned correction (in addition to the residual penalty).
    pub l2: f32,
    /// Residual scale `lambda`, used in the forward pass and the gradient.
    pub residual_scale: f32,
    /// Weight of the residual-size penalty `alpha_r`.
    pub residual_regularization: f32,
    /// Weight of the distillation loss `beta`.
    pub distillation_weight: f32,
    /// Distillation temperature `tau_d` (validated, not fixed).
    pub distillation_temperature: f32,
    /// Include the KL distillation term.
    pub use_distillation: bool,
}

impl Default for C8TrainingConfig {
    fn default() -> Self {
        Self {
            seed: 20260925,
            learning_rate: 1e-2,
            epochs: 5,
            batch_size: 8,
            temperature: 0.07,
            l2: 1e-4,
            residual_scale: 0.1,
            residual_regularization: 0.0,
            distillation_weight: 0.0,
            distillation_temperature: 0.5,
            use_distillation: false,
        }
    }
}

impl C8TrainingConfig {
    fn validate(&self) -> Result<()> {
        for (name, v) in [
            ("learning_rate", self.learning_rate),
            ("temperature", self.temperature),
            ("l2", self.l2),
            ("residual_scale", self.residual_scale),
            ("residual_regularization", self.residual_regularization),
            ("distillation_weight", self.distillation_weight),
            ("distillation_temperature", self.distillation_temperature),
        ] {
            if !v.is_finite() {
                return Err(AttentionError::Config(format!("C8 {name} must be finite")));
            }
        }
        if self.temperature <= 0.0 {
            return Err(AttentionError::Config("C8 temperature must be > 0".into()));
        }
        if self.distillation_temperature <= 0.0 {
            return Err(AttentionError::Config(
                "C8 distillation_temperature must be > 0".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(AttentionError::Config("C8 batch_size must be > 0".into()));
        }
        if self.epochs == 0 {
            return Err(AttentionError::Config("C8 epochs must be > 0".into()));
        }
        if self.use_distillation && self.distillation_weight == 0.0 {
            return Err(AttentionError::Config(
                "use_distillation with zero weight is a no-op; set the weight or disable it".into(),
            ));
        }
        Ok(())
    }
}

/// Loss decomposition for one batch, reported in the training log.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct C8LossReport {
    pub contrastive: f32,
    pub residual: f32,
    pub distill: f32,
    pub total: f32,
}

/// Result of a C8 training run.
#[derive(Debug, Clone)]
pub struct C8TrainedModel {
    /// The learned residual projection.
    pub projection: ResidualQkvProjection,
    /// Total loss per epoch.
    pub loss_history: Vec<f32>,
    /// Contrastive component per epoch.
    pub contrastive_history: Vec<f32>,
    /// Residual penalty per epoch.
    pub residual_history: Vec<f32>,
    /// Distillation component per epoch.
    pub distill_history: Vec<f32>,
    /// Optimizer steps actually taken (epochs * batches, not epochs * examples).
    pub optimizer_steps: usize,
    pub epochs_run: usize,
    pub seed: u64,
    pub config: C8TrainingConfig,
    /// Dataset fingerprint of the run that produced these weights.
    pub dataset_hash: u64,
}

impl C8TrainedModel {
    /// Compose the runtime C8 config for inference.
    #[allow(clippy::too_many_arguments)]
    pub fn to_config(
        &self,
        head_alignments: Vec<crate::alignment::AlignmentProjection>,
        query_alignment: crate::alignment::AlignmentProjection,
        scorer: ResidualScorer,
        use_evidence: bool,
        residual_scale: f32,
        cache_enabled: bool,
    ) -> crate::config::C8AttentionConfig {
        crate::config::C8AttentionConfig {
            enabled: true,
            attention_dim: self.projection.attention_dim,
            key_dim: self.projection.key_dim,
            value_dim: self.projection.value_dim,
            head_alignments,
            query_alignment: Some(query_alignment),
            residual_projection: Some(self.projection.clone()),
            residual_scale,
            use_evidence,
            scorer,
            cache_enabled,
        }
    }

    /// Build the versioned C8 model card.
    #[allow(clippy::too_many_arguments)]
    pub fn to_model_card(
        &self,
        model_id: &str,
        arch: &str,
        head_names: Vec<String>,
        head_alignments: Vec<crate::alignment::AlignmentProjection>,
        query_alignment: crate::alignment::AlignmentProjection,
        scorer: ResidualScorer,
        use_evidence: bool,
        residual_scale: f32,
        negative_provenance_hash: u64,
    ) -> Result<C8ModelCard> {
        C8ModelCard::new(
            model_id,
            arch,
            self.projection.attention_dim,
            self.projection.key_dim,
            self.projection.value_dim,
            head_names,
            head_alignments,
            query_alignment,
            self.projection.clone(),
            scorer,
            use_evidence,
            residual_scale,
            TrainingMeta {
                seed: self.seed,
                dataset_hash: self.dataset_hash,
                objective: "c8-residual-contrastive".into(),
                learning_rate: self.config.learning_rate,
                batch_size: self.config.batch_size,
                epochs_run: self.epochs_run,
                best_val_loss: self.loss_history.last().copied(),
                l2: self.config.l2,
                ..TrainingMeta::default()
            },
            C8ResidualMeta {
                residual_scale: self.config.residual_scale,
                residual_regularization: self.config.residual_regularization,
                distillation_weight: self.config.distillation_weight,
                distillation_temperature: self.config.distillation_temperature,
                use_distillation: self.config.use_distillation,
                delta_sq_norm: self.projection.delta_sq_norm(),
                n_trainable_params: self.projection.trainable_param_count(),
                optimizer_steps: self.optimizer_steps,
                negative_provenance_hash,
            },
            self.dataset_hash,
        )
    }
}

/// Deterministic C8 residual trainer.
pub struct ResidualQkvTrainer {
    cfg: C8TrainingConfig,
    /// Adam state: [Q, K, V] moment vectors.
    adam_m: [Vec<f32>; 3],
    adam_v: [Vec<f32>; 3],
    t: usize,
}

impl ResidualQkvTrainer {
    pub fn new(cfg: C8TrainingConfig) -> Self {
        Self {
            cfg,
            adam_m: [Vec::new(), Vec::new(), Vec::new()],
            adam_v: [Vec::new(), Vec::new(), Vec::new()],
            t: 0,
        }
    }

    pub fn config(&self) -> &C8TrainingConfig {
        &self.cfg
    }

    /// Train the residual projection. Returns the best-seen weights by total
    /// training loss (the C7 selection rule, kept so the comparison is fair).
    pub fn train(
        &mut self,
        init: ResidualQkvProjection,
        dataset: &C8QkvDataset,
        scorer: &ResidualScorer,
    ) -> Result<C8TrainedModel> {
        self.cfg.validate()?;
        if dataset.examples.is_empty() {
            return Err(AttentionError::EmptyInput("C8 empty dataset".into()));
        }
        let lens = (
            init.w_q_delta.len(),
            init.w_k_delta.len(),
            init.w_v_delta.len(),
        );
        self.adam_m = [vec![0.0; lens.0], vec![0.0; lens.1], vec![0.0; lens.2]];
        self.adam_v = [vec![0.0; lens.0], vec![0.0; lens.1], vec![0.0; lens.2]];
        self.t = 0;

        let mut w = init;
        let mut best: Option<(f32, ResidualQkvProjection)> = None;
        let mut total_hist = Vec::new();
        let mut contrastive_hist = Vec::new();
        let mut residual_hist = Vec::new();
        let mut distill_hist = Vec::new();
        let mut steps = 0usize;

        for _epoch in 0..self.cfg.epochs {
            let order = permutation(dataset.examples.len(), self.cfg.seed);
            let mut epoch = C8LossReport::default();
            let mut batches = 0usize;
            for batch in order.chunks(self.cfg.batch_size) {
                let report = self.step_batch(&mut w, batch, dataset, scorer)?;
                epoch.contrastive += report.contrastive;
                epoch.residual += report.residual;
                epoch.distill += report.distill;
                epoch.total += report.total;
                batches += 1;
                steps += 1;
            }
            let n = batches as f32;
            let avg = C8LossReport {
                contrastive: epoch.contrastive / n,
                residual: epoch.residual / n,
                distill: epoch.distill / n,
                total: epoch.total / n,
            };
            total_hist.push(avg.total);
            contrastive_hist.push(avg.contrastive);
            residual_hist.push(avg.residual);
            distill_hist.push(avg.distill);
            if best.as_ref().map(|(l, _)| avg.total < *l).unwrap_or(true) {
                best = Some((avg.total, w.clone()));
            }
        }

        let (_, best_w) = best.ok_or_else(|| AttentionError::EmptyInput("no training".into()))?;
        Ok(C8TrainedModel {
            projection: best_w,
            loss_history: total_hist,
            contrastive_history: contrastive_hist,
            residual_history: residual_hist,
            distill_history: distill_hist,
            optimizer_steps: steps,
            epochs_run: self.cfg.epochs,
            seed: self.cfg.seed,
            config: self.cfg.clone(),
            dataset_hash: dataset.fingerprint(),
        })
    }

    /// One optimizer step over a batch of examples: accumulate gradients across
    /// the whole batch, then apply a single Adam update.
    fn step_batch(
        &mut self,
        w: &mut ResidualQkvProjection,
        batch: &[usize],
        dataset: &C8QkvDataset,
        scorer: &ResidualScorer,
    ) -> Result<C8LossReport> {
        // Forward in the effective (materialized) space, then map the gradient
        // back onto the learned correction.
        let effective = w.to_qkv();
        let lambda = self.cfg.residual_scale;
        let n_ex = batch.len() as f32;

        let mut g_q = vec![0.0f32; effective.w_q.len()];
        let mut g_k = vec![0.0f32; effective.w_k.len()];
        let mut g_v = vec![0.0f32; effective.w_v.len()];
        let mut report = C8LossReport::default();

        // Regularization on the learned correction. Only the reported value is
        // computed here; its gradient is added after batch averaging so it
        // enters the update at the same scale as the data gradient.
        let alpha_r = self.cfg.residual_regularization;
        if alpha_r != 0.0 {
            let sq = w.delta_sq_norm();
            let n_params = w.trainable_param_count().max(1) as f32;
            report.residual = alpha_r * sq / n_params;
        }

        for &ei in batch {
            let ex = &dataset.examples[ei];
            let back = self.example_backward(&effective, ex, scorer, lambda)?;
            report.contrastive += back.contrastive;
            report.distill += back.distill;
            for i in 0..g_q.len() {
                g_q[i] += back.g_q[i];
            }
            for i in 0..g_k.len() {
                g_k[i] += back.g_k[i];
            }
            for i in 0..g_v.len() {
                g_v[i] += back.g_v[i];
            }
        }

        // Average over the batch (the real mini-batch step).
        for g in [&mut g_q, &mut g_k, &mut g_v] {
            for v in g.iter_mut() {
                *v /= n_ex;
            }
        }
        report.contrastive /= n_ex;
        report.residual /= n_ex;
        report.distill /= n_ex;

        // Residual-size penalty gradient, computed after averaging so it enters
        // the update at the same scale as the data gradient.
        if alpha_r != 0.0 {
            let n_params = w.trainable_param_count().max(1) as f32;
            let scale = 2.0 * alpha_r / n_params;
            for (dst, src) in [
                (&mut g_q, &w.w_q_delta),
                (&mut g_k, &w.w_k_delta),
                (&mut g_v, &w.w_v_delta),
            ] {
                for i in 0..src.len() {
                    dst[i] += scale * src[i];
                }
            }
        }

        // Transfer the effective-space gradient onto the learned correction and
        // apply the chain rule factor alpha once.
        let d_q = w.delta_grad_from_effective(&g_q);
        let d_k = w.delta_grad_from_effective(&g_k);
        let d_v = w.delta_grad_from_effective(&g_v);

        self.t += 1;
        let lr = self.cfg.learning_rate;
        let l2 = self.cfg.l2;
        // The three updates are independent, so take the optimizer config once
        // and update the parameter/moment arrays in place. Calling a
        // `&mut self` method three times would alias `self`.
        let updates: [(&mut Vec<f32>, &[f32], usize); 3] = [
            (&mut w.w_q_delta, &d_q, 0),
            (&mut w.w_k_delta, &d_k, 1),
            (&mut w.w_v_delta, &d_v, 2),
        ];
        for (params, grads, slot) in updates {
            Self::adam_update(
                params,
                grads,
                &mut self.adam_m[slot],
                &mut self.adam_v[slot],
                self.t,
                lr,
                l2,
            )?;
        }

        report.total = report.contrastive + report.residual + report.distill;
        Ok(report)
    }

    /// Backward pass for one example, in effective projection space.
    ///
    /// The contrastive term runs on `S_final = S_base + lambda * dS`. Because
    /// `S_base` is constant, `dL/d(dS_c) = dL/dS_final_c`, and `dS_c` is linear
    /// in the projection weights through the attention term. `dS_c` is built
    /// here from the same `ResidualScorer` the runtime subsystem uses, so the
    /// training objective and the deployed score cannot drift apart.
    ///
    /// Only the attention term has weight dependence. The evidence and
    /// disagreement terms are constants with respect to `W`, so they shift the
    /// score (and therefore the loss) without contributing gradient — they are
    /// included in `dS` but not in the returned `g_*`.
    fn example_backward(
        &self,
        effective: &crate::projection::QkvProjection,
        ex: &C8QkvExample,
        scorer: &ResidualScorer,
        lambda: f32,
    ) -> Result<C8ExampleBackward> {
        let n = ex.candidate_count();
        let neg_evidence = ex.negative_evidence_scalars();
        let neg_disagreement = ex.negative_disagreements();
        let read_evidence = scorer.uses_evidence();
        // The scorer's constant bias is lambda-scaled like every other term, so
        // it is added to the score even though it carries no gradient.
        let w_attn = lambda * scorer.w_attn;
        let w_evidence = if read_evidence {
            lambda * scorer.w_evidence
        } else {
            0.0
        };
        let w_disagree = if read_evidence {
            lambda * scorer.w_disagree
        } else {
            0.0
        };

        // Candidate list: positive first, then mined negatives.
        let mut grads = Vec::with_capacity(n);
        let mut deltas = Vec::with_capacity(n);
        for c in 0..n {
            let (z, evidence, disagreement) = if c == 0 {
                (
                    ex.positive.clone(),
                    ex.positive_evidence.as_ref().map(|e| e.aggregate()),
                    ex.positive_disagreement(),
                )
            } else {
                let i = c - 1;
                (
                    ex.negatives[i].clone(),
                    neg_evidence[i],
                    neg_disagreement[i],
                )
            };
            let g = candidate_gradient(effective, &ex.q_a, &z, w_attn, evidence, w_evidence)?;
            // `g.score` covers the attention and evidence terms. Add the
            // disagreement term and the bias explicitly: both are constants in W,
            // so they change the loss but not the gradient.
            let mut delta = g.score + lambda * scorer.bias;
            if w_disagree != 0.0 {
                if let Some(dis) = disagreement {
                    delta += w_disagree * dis;
                }
                // Undefined disagreement (fewer than two usable heads) is
                // skipped entirely rather than treated as zero.
            }
            deltas.push(delta);
            grads.push(g);
        }

        // --- S_final = S_base + lambda * dS ---
        let base = ex.all_baselines();
        let final_scores: Vec<f32> = base
            .iter()
            .zip(deltas.iter())
            .map(|(&b, &d)| b + d)
            .collect();

        // --- InfoNCE over S_final ---
        let tau = self.cfg.temperature;
        let probs = candidate_softmax(&final_scores, tau)?;
        // Numerically stable log of the positive's probability.
        let contrastive = -probs[0].max(1e-30f32).ln();
        // dL/dS_final_c = (p_c - 1{c = pos}) / tau
        let mut d_loss_d_final: Vec<f32> = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (p - if i == 0 { 1.0 } else { 0.0 }) / tau)
            .collect();

        // --- Distillation: KL(softmax(S_base/tau_d) || softmax(S_final/tau_d)) ---
        let beta = if self.cfg.use_distillation {
            self.cfg.distillation_weight
        } else {
            0.0
        };
        let mut distill = 0.0f32;
        if beta != 0.0 {
            let td = self.cfg.distillation_temperature;
            let kl = distillation_loss(&base, &final_scores, td)?;
            let d_kl = distillation_gradient(&base, &final_scores, td)?;
            for (d, g) in d_loss_d_final.iter_mut().zip(d_kl.iter()) {
                *d += beta * g;
            }
            distill = beta * kl;
        }

        // --- Chain rule: the analytic g_* is already d(delta)/dW. ---
        let mut g_q = vec![0.0f32; grads[0].g_q.len()];
        let mut g_k = vec![0.0f32; grads[0].g_k.len()];
        let mut g_v = vec![0.0f32; grads[0].g_v.len()];
        for (c, g) in grads.iter().enumerate() {
            let coef = d_loss_d_final[c];
            if coef == 0.0 {
                continue;
            }
            for i in 0..g_q.len() {
                g_q[i] += coef * g.g_q[i];
            }
            for i in 0..g_k.len() {
                g_k[i] += coef * g.g_k[i];
            }
            for i in 0..g_v.len() {
                g_v[i] += coef * g.g_v[i];
            }
        }

        if !contrastive.is_finite() || !distill.is_finite() {
            return Err(AttentionError::NonFinite("C8 example loss".into()));
        }
        Ok(C8ExampleBackward {
            contrastive,
            distill,
            g_q,
            g_k,
            g_v,
        })
    }

    /// One deterministic Adam update. Static so the caller can update the three
    /// parameter matrices without aliasing `self`.
    #[allow(clippy::too_many_arguments)]
    fn adam_update(
        params: &mut [f32],
        grads: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        t: usize,
        lr: f32,
        l2: f32,
    ) -> Result<()> {
        let (beta1, beta2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let b1t = beta1.powi(t as i32);
        let b2t = beta2.powi(t as i32);
        for i in 0..params.len() {
            // L2 is applied as part of the gradient (decoupled from Adam's
            // moment normalization), matching C7's convention.
            let g = grads[i] + l2 * params[i];
            if !g.is_finite() {
                return Err(AttentionError::NonFinite("C8 training gradient".into()));
            }
            let mi = beta1 * m[i] + (1.0 - beta1) * g;
            let vi = beta2 * v[i] + (1.0 - beta2) * g * g;
            m[i] = mi;
            v[i] = vi;
            let m_hat = mi / (1.0 - b1t);
            let v_hat = vi / (1.0 - b2t);
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            if !params[i].is_finite() {
                return Err(AttentionError::NonFinite("C8 parameter".into()));
            }
        }
        Ok(())
    }
}

/// Per-example backward result: the loss split for reporting plus the gradient
/// contribution in effective projection space.
struct C8ExampleBackward {
    contrastive: f32,
    /// Already multiplied by the distillation weight `beta`.
    distill: f32,
    g_q: Vec<f32>,
    g_k: Vec<f32>,
    g_v: Vec<f32>,
}

/// Seeded Fisher-Yates permutation. Deterministic, and reproducible across
/// processes because it uses the crate's own `DetRng` rather than the platform RNG.
fn permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = crate::alignment::DetRng::new(seed);
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::negative_mining::NegSource;

    const D: usize = 4;
    const HEADS: usize = 2;

    /// Per-head aligned vector. Each head must see a *different* vector:
    /// if every head is identical then `d score/dA_h` is constant across `h`,
    /// the softmax shift-invariance cancels it, and the Q/K gradients are
    /// provably zero. A fixture that hides that is not testing the trainer.
    fn vec4(a: f32, head: usize) -> Vec<f32> {
        let h = head as f32;
        vec![a, 0.5 - 0.1 * h, -0.25, 0.125 + 0.2 * h]
    }

    fn heads_of(a: f32) -> Vec<Vec<f32>> {
        (0..HEADS).map(|h| vec4(a, h)).collect()
    }

    fn ev(sims: [Option<f32>; HEADS]) -> crate::scorer::RetrievalEvidence {
        crate::scorer::RetrievalEvidence {
            head_sims: sims.to_vec(),
            head_ranks: vec![Some(0.5); HEADS],
            head_present: sims.iter().map(|s| s.is_some()).collect(),
        }
    }

    fn example(q: f32, pos: f32, base_pos: f32, negs: &[(f32, f32)]) -> C8QkvExample {
        C8QkvExample {
            q_a: vec4(q, 0),
            positive: heads_of(pos),
            positive_baseline: base_pos,
            positive_evidence: Some(ev([Some(0.9), Some(0.7)])),
            negatives: negs.iter().map(|(v, _)| heads_of(*v)).collect(),
            negatives_baseline: negs.iter().map(|(_, b)| *b).collect(),
            negatives_evidence: negs
                .iter()
                .map(|_| Some(ev([Some(0.3), Some(0.1)])))
                .collect(),
            negative_sources: negs.iter().map(|_| NegSource::HighBaseline).collect(),
        }
    }

    fn dataset(n: usize) -> C8QkvDataset {
        let mut b = C8QkvDatasetBuilder::new(D, (0..HEADS).map(|h| format!("h{h}")).collect());
        for i in 0..n {
            let q = 1.0 + i as f32 * 0.05;
            b.push(example(q, 1.0, 0.8, &[(0.4, 0.5), (0.2, 0.3), (-0.3, 0.1)]))
                .unwrap();
        }
        b.build()
    }

    fn cfg() -> C8TrainingConfig {
        C8TrainingConfig {
            seed: 20260925,
            learning_rate: 0.05,
            epochs: 30,
            batch_size: 4,
            temperature: 0.07,
            l2: 1e-4,
            residual_scale: 0.2,
            residual_regularization: 0.0,
            distillation_weight: 0.0,
            distillation_temperature: 0.5,
            use_distillation: false,
        }
    }

    fn init() -> ResidualQkvProjection {
        ResidualQkvProjection::truncated_identity_residual(D, D, D, 0.1).unwrap()
    }

    fn scorer() -> ResidualScorer {
        ResidualScorer::new(1.0, 0.0, 0.0)
    }

    #[test]
    fn permutation_is_deterministic_and_a_permutation() {
        let a = permutation(50, 20260925);
        let b = permutation(50, 20260925);
        assert_eq!(a, b);
        assert_ne!(a, permutation(50, 1));
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<usize>>());
    }

    #[test]
    fn real_minibatches_mean_multiple_examples_per_step() {
        let ds = dataset(12);
        let mut t = ResidualQkvTrainer::new(C8TrainingConfig { epochs: 3, ..cfg() });
        let m = t.train(init(), &ds, &scorer()).unwrap();
        // 3 epochs * ceil(12 / 4) batches = 9 steps. The C7 trainer would have
        // reported 36 (one per example), which is the bug this fixes.
        assert_eq!(
            m.optimizer_steps, 9,
            "steps must be per batch, not per example"
        );
        assert_eq!(m.config.batch_size, 4);
    }

    #[test]
    fn training_is_bit_deterministic() {
        let ds = dataset(10);
        let c = cfg();
        let mut t1 = ResidualQkvTrainer::new(c.clone());
        let m1 = t1.train(init(), &ds, &scorer()).unwrap();
        let mut t2 = ResidualQkvTrainer::new(c);
        let m2 = t2.train(init(), &ds, &scorer()).unwrap();
        assert_eq!(m1.projection.w_q_delta, m2.projection.w_q_delta);
        assert_eq!(m1.projection.w_k_delta, m2.projection.w_k_delta);
        assert_eq!(m1.projection.w_v_delta, m2.projection.w_v_delta);
        assert_eq!(m1.loss_history, m2.loss_history);
        assert_eq!(m1.dataset_hash, m2.dataset_hash);
    }

    #[test]
    fn training_reduces_loss() {
        let ds = dataset(16);
        let mut t = ResidualQkvTrainer::new(cfg());
        let m = t.train(init(), &ds, &scorer()).unwrap();
        let first = m.loss_history.first().copied().unwrap();
        let last = m.loss_history.last().copied().unwrap();
        assert!(last < first, "{first} -> {last}");
    }

    #[test]
    fn identical_heads_make_qk_gradients_vanish() {
        // Documents the identifiability boundary: when every head sees the same
        // vector, `d score/dA_h` is the same for all `h`, so the softmax's
        // shift-invariance cancels the attention-weight gradient exactly. Only
        // the value path keeps a gradient. This is a property of single-vector
        // attention under `d score = w * <q_a, O>`, not a trainer defect, and it
        // is why real multi-head evidence (distinct aligned head subspaces) is
        // necessary for Q/K to be learnable.
        let mut ex = example(1.0, 1.0, 0.8, &[(0.4, 0.5), (0.2, 0.3)]);
        let flat = vec![1.0, 0.5, -0.25, 0.125];
        ex.positive = vec![flat.clone(), flat.clone()];
        for n in &mut ex.negatives {
            *n = vec![flat.clone(), flat.clone()];
        }
        let g = crate::training::candidate_gradient(
            &init().to_qkv(),
            &ex.q_a,
            &ex.positive,
            1.0,
            None,
            0.0,
        )
        .unwrap();
        assert!(g.g_q.iter().all(|v| *v == 0.0), "Q gradient must cancel");
        assert!(g.g_k.iter().all(|v| *v == 0.0), "K gradient must cancel");
        assert!(
            g.g_v.iter().any(|v| *v != 0.0),
            "the value path is still identifiable"
        );
    }

    #[test]
    fn distinct_heads_give_nonzero_qk_gradients() {
        // The converse of the test above: the C8 fixture must actually exercise
        // the Q/K path, otherwise a broken Q/K gradient would go unnoticed.
        let ex = example(1.0, 1.0, 0.8, &[(0.4, 0.5), (0.2, 0.3)]);
        let g = crate::training::candidate_gradient(
            &init().to_qkv(),
            &ex.q_a,
            &ex.positive,
            1.0,
            None,
            0.0,
        )
        .unwrap();
        assert!(g.g_q.iter().any(|v| *v != 0.0));
        assert!(g.g_k.iter().any(|v| *v != 0.0));
        assert!(g.g_v.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn frozen_base_is_never_updated() {
        let ds = dataset(8);
        let start = init();
        let base = start.w_q_base.clone();
        let mut t = ResidualQkvTrainer::new(cfg());
        let m = t.train(start, &ds, &scorer()).unwrap();
        assert_eq!(m.projection.w_q_base, base, "W0 must stay frozen");
        let moved = [
            &m.projection.w_q_delta,
            &m.projection.w_k_delta,
            &m.projection.w_v_delta,
        ];
        for (name, d) in ["Q", "K", "V"].iter().zip(moved.iter()) {
            assert!(
                d.iter().any(|v| *v != 0.0),
                "the {name} correction must actually move"
            );
        }
    }

    #[test]
    fn residual_regularization_shrinks_the_correction() {
        let ds = dataset(16);
        let run = |reg: f32| {
            let mut t = ResidualQkvTrainer::new(C8TrainingConfig {
                residual_regularization: reg,
                ..cfg()
            });
            t.train(init(), &ds, &scorer()).unwrap()
        };
        let free = run(0.0);
        let penalized = run(1.0);
        assert!(
            penalized.projection.delta_sq_norm() < free.projection.delta_sq_norm(),
            "reg {} vs free {}",
            penalized.projection.delta_sq_norm(),
            free.projection.delta_sq_norm()
        );
        assert!(penalized.residual_history.iter().all(|r| *r >= 0.0));
    }

    #[test]
    fn distillation_adds_a_non_negative_kl_term() {
        let ds = dataset(12);
        let mut t = ResidualQkvTrainer::new(C8TrainingConfig {
            use_distillation: true,
            distillation_weight: 0.5,
            distillation_temperature: 0.5,
            ..cfg()
        });
        let m = t.train(init(), &ds, &scorer()).unwrap();
        assert!(m.distill_history.iter().all(|d| *d >= 0.0));
        assert!(
            m.distill_history.iter().any(|d| *d > 0.0),
            "a trained model must diverge from the teacher at least once"
        );
    }

    #[test]
    fn zero_residual_scale_freezes_the_delta() {
        // lambda = 0 kills the gradient into the weights, so the correction
        // must stay exactly zero. This is the training-side twin of the
        // lambda=0 inference parity test.
        let ds = dataset(8);
        let mut t = ResidualQkvTrainer::new(C8TrainingConfig {
            residual_scale: 0.0,
            ..cfg()
        });
        let m = t.train(init(), &ds, &scorer()).unwrap();
        assert!(m.projection.w_q_delta.iter().all(|v| v.abs() < 1e-12));
        assert!(m.projection.w_k_delta.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn rejects_inconsistent_configs() {
        let ds = dataset(4);
        let bad = C8TrainingConfig {
            temperature: 0.0,
            ..cfg()
        };
        assert!(ResidualQkvTrainer::new(bad)
            .train(init(), &ds, &scorer())
            .is_err());
        let bad = C8TrainingConfig {
            batch_size: 0,
            ..cfg()
        };
        assert!(ResidualQkvTrainer::new(bad)
            .train(init(), &ds, &scorer())
            .is_err());
        let bad = C8TrainingConfig {
            use_distillation: true,
            distillation_weight: 0.0,
            ..cfg()
        };
        assert!(ResidualQkvTrainer::new(bad)
            .train(init(), &ds, &scorer())
            .is_err());
        let bad = C8TrainingConfig {
            distillation_temperature: f32::NAN,
            ..cfg()
        };
        assert!(ResidualQkvTrainer::new(bad)
            .train(init(), &ds, &scorer())
            .is_err());
    }

    #[test]
    fn dataset_builder_validates_shapes() {
        let mut b = C8QkvDatasetBuilder::new(D, vec!["h0".into(), "h1".into()]);
        assert!(b.push(example(1.0, 1.0, 0.8, &[(0.4, 0.5)])).is_ok());

        let mut bad = example(1.0, 1.0, 0.8, &[(0.4, 0.5)]);
        bad.negatives_evidence.pop();
        assert!(b.push(bad).is_err());

        let mut bad = example(1.0, 1.0, 0.8, &[(0.4, 0.5)]);
        bad.q_a = vec![1.0; D + 1];
        assert!(b.push(bad).is_err());

        let mut bad = example(1.0, 1.0, 0.8, &[(0.4, 0.5)]);
        bad.positive = vec![vec![1.0; D]];
        assert!(b.push(bad).is_err(), "head count mismatch");
    }

    #[test]
    fn provenance_hash_changes_with_negative_sources() {
        let mut a = C8QkvDatasetBuilder::new(D, vec!["h0".into(), "h1".into()]);
        a.push(example(1.0, 1.0, 0.8, &[(0.4, 0.5)])).unwrap();
        let da = a.build();

        let mut b = C8QkvDatasetBuilder::new(D, vec!["h0".into(), "h1".into()]);
        let mut ex = example(1.0, 1.0, 0.8, &[(0.4, 0.5)]);
        ex.negative_sources = vec![NegSource::HeadDisagreement];
        b.push(ex).unwrap();
        let db = b.build();

        assert_ne!(
            da.negative_provenance_hash, db.negative_provenance_hash,
            "a different negative provenance must be visible in the model card"
        );
    }

    #[test]
    fn example_helpers_agree_in_length_and_order() {
        let ex = example(1.0, 1.0, 0.8, &[(0.4, 0.5), (0.2, 0.3)]);
        assert_eq!(ex.candidate_count(), 3);
        assert_eq!(ex.all_baselines().len(), 3);
        assert_eq!(ex.all_evidence().len(), 3);
        assert_eq!(ex.all_baselines()[0], 0.8, "positive comes first");
        assert_eq!(ex.all_baselines()[1], 0.5);
    }

    #[test]
    fn unrestricted_projection_also_trains() {
        // Arm D trains the same code path with no base.
        let ds = dataset(12);
        let mut t = ResidualQkvTrainer::new(cfg());
        let m = t
            .train(
                ResidualQkvProjection::unrestricted(D, D, D, 7).unwrap(),
                &ds,
                &scorer(),
            )
            .unwrap();
        assert!(!m.projection.use_residual);
        assert!(m.loss_history.last() < m.loss_history.first());
    }

    #[test]
    fn reduced_dimension_training_runs() {
        let ds = dataset(8);
        let mut t = ResidualQkvTrainer::new(cfg());
        let m = t
            .train(
                ResidualQkvProjection::truncated_identity_residual(D, 2, 2, 0.1).unwrap(),
                &ds,
                &scorer(),
            )
            .unwrap();
        assert_eq!(m.projection.key_dim, 2);
        assert!(m.projection.w_k_delta.iter().all(|v| v.is_finite()));
    }
}
