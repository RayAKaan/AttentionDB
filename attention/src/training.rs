use crate::errors::{AttentionError, Result};
use crate::model::TrainingMeta;
use crate::projection::QkvProjection;
use serde::{Deserialize, Serialize};

/// One training example: an aligned query, an aligned positive candidate
/// (per-head vectors `Vec<Vec<f32>>` = Z_d), and aligned hard negatives
/// (each also `Vec<Vec<f32>>`). Optional retrieval evidence per candidate
/// is carried alongside so the C7-F scorer (attention + evidence) is
/// reproduced during training. All vectors must already be ALIGNED
/// (q_a via query alignment, candidates via per-head alignments).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QkvExample {
    /// Aligned query vector q_a ∈ R^{d_a}.
    pub q_a: Vec<f32>,
    /// Positive candidate Z_d^+ ∈ R^{H × d_a}.
    pub positive: Vec<Vec<f32>>,
    /// Positive candidate retrieval evidence (C7-F; `None` for C7-E).
    pub positive_evidence: Option<f32>,
    /// Hard negatives Z_d^- ∈ R^{k × H × d_a}.
    pub negatives: Vec<Vec<Vec<f32>>>,
    /// Per-negative retrieval evidence.
    pub negatives_evidence: Vec<Option<f32>>,
}

impl QkvExample {
    /// Sanity check: query and positive must be non-empty with consistent
    /// head count. Full dimension checks happen in the dataset builder.
    pub fn validate(&self) -> Result<()> {
        if self.q_a.is_empty() {
            return Err(AttentionError::EmptyInput("q_a empty".into()));
        }
        if self.positive.is_empty() {
            return Err(AttentionError::EmptyInput("positive empty".into()));
        }
        Ok(())
    }
}

/// A training dataset: aligned queries with their positive and hard-negative
/// candidate sets. Deterministic fingerprint over the serialized examples is
/// used for reproducibility (spec §C7: dataset hash in the model card).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QkvDataset {
    pub examples: Vec<QkvExample>,
    /// Per-head dimension d_a the aligned vectors use.
    pub attention_dim: usize,
    /// Retrieval head names in order (for the model card).
    pub head_names: Vec<String>,
}

impl QkvDataset {
    /// FNV-1a fingerprint over the canonical serialized examples.
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

/// Builder for [`QkvDataset`] that validates all aligned dimensions.
#[derive(Debug, Clone)]
pub struct QkvDatasetBuilder {
    examples: Vec<QkvExample>,
    attention_dim: usize,
    head_names: Vec<String>,
}

impl QkvDatasetBuilder {
    pub fn new(attention_dim: usize, head_names: Vec<String>) -> Self {
        Self {
            examples: Vec::new(),
            attention_dim,
            head_names,
        }
    }

    /// Push one example; verifies every aligned vector in q_a, positive, and
    /// each negative has length `attention_dim` and matches head count.
    pub fn push(&mut self, ex: QkvExample) -> Result<()> {
        let heads = self.present_heads();
        if ex.positive.len() != heads {
            return Err(AttentionError::DimensionMismatch {
                expected: heads,
                found: ex.positive.len(),
            });
        }
        for v in ex.positive.iter() {
            Self::check_vec(v, self.attention_dim)?;
        }
        if ex.q_a.len() != self.attention_dim {
            return Err(AttentionError::DimensionMismatch {
                expected: self.attention_dim,
                found: ex.q_a.len(),
            });
        }
        for neg in ex.negatives.iter() {
            if neg.len() != heads {
                return Err(AttentionError::DimensionMismatch {
                    expected: heads,
                    found: neg.len(),
                });
            }
            for v in neg.iter() {
                Self::check_vec(v, self.attention_dim)?;
            }
        }
        if ex.negatives_evidence.len() != ex.negatives.len() {
            return Err(AttentionError::DimensionMismatch {
                expected: ex.negatives.len(),
                found: ex.negatives_evidence.len(),
            });
        }
        self.examples.push(ex);
        Ok(())
    }

    fn present_heads(&self) -> usize {
        self.head_names.len()
    }

    fn check_vec(v: &[f32], dim: usize) -> Result<()> {
        if v.len() != dim {
            return Err(AttentionError::DimensionMismatch {
                expected: dim,
                found: v.len(),
            });
        }
        Ok(())
    }

    pub fn build(self) -> QkvDataset {
        QkvDataset {
            examples: self.examples,
            attention_dim: self.attention_dim,
            head_names: self.head_names,
        }
    }
}

/// Hyper-parameters for the deterministic contrastive QKV trainer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub seed: u64,
    pub learning_rate: f32,
    pub epochs: usize,
    /// Number of candidates per step (positive + negatives) recorded for the
    /// model card's `batch_size`; training itself is per-example (batch of 1
    /// query with its hard negatives), fully deterministic.
    pub batch_size: usize,
    pub temperature: f32,
    pub l2: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            seed: 20260925,
            learning_rate: 1e-2,
            epochs: 5,
            batch_size: 8,
            temperature: 0.07,
            l2: 1e-4,
        }
    }
}

/// Per-candidate gradient of the attention score s = w_attn·⟨q_a, O_d⟩
/// (+ evidence term, which does not depend on the weights) with respect to
/// the Q/K/V projection weights. Computed analytically through the forward
/// pass so training is exact, fast, and deterministic (no autodiff, no RNG).
#[derive(Debug, Clone)]
pub struct QkvGradient {
    /// ∂loss/∂W_Q ∈ R^{d_a × d_k} (row-major).
    pub g_q: Vec<f32>,
    /// ∂loss/∂W_K ∈ R^{d_a × d_k}.
    pub g_k: Vec<f32>,
    /// ∂loss/∂W_V ∈ R^{d_a × d_v}.
    pub g_v: Vec<f32>,
    /// Candidate score (for diagnostics).
    pub score: f32,
}

/// Forward + analytic backward for one candidate.
/// Returns (embedding output O_d, attention weights, QKV gradients).
/// Only the loss-weighted gradient is returned; the caller accumulates.
pub(crate) fn candidate_gradient(
    qkv: &QkvProjection,
    q_a: &[f32],
    z_d: &[Vec<f32>],
    w_attn: f32,
    evidence: Option<f32>,
    w_evidence: f32,
) -> Result<QkvGradient> {
    let h_count = z_d.len();
    let d_a = qkv.attention_dim;
    let d_k = qkv.key_dim;
    let d_v = qkv.value_dim;

    if q_a.len() != d_a {
        return Err(AttentionError::DimensionMismatch {
            expected: d_a,
            found: q_a.len(),
        });
    }
    for v in z_d {
        if v.len() != d_a {
            return Err(AttentionError::DimensionMismatch {
                expected: d_a,
                found: v.len(),
            });
        }
    }

    let scale = 1.0 / (d_k as f32).sqrt();

    // Q = q_a W_Q  (len d_k)
    let mut q = vec![0.0f32; d_k];
    for j in 0..d_k {
        let mut acc = 0.0;
        for i in 0..d_a {
            acc += q_a[i] * qkv.w_q[i * d_k + j];
        }
        q[j] = acc;
    }

    // K = z W_K (H x d_k), V = z W_V (H x d_v)
    let mut k = vec![vec![0.0f32; d_k]; h_count];
    let mut v = vec![vec![0.0f32; d_v]; h_count];
    for h in 0..h_count {
        for m in 0..d_k {
            let mut acc = 0.0;
            for i in 0..d_a {
                acc += z_d[h][i] * qkv.w_k[i * d_k + m];
            }
            k[h][m] = acc;
        }
        for c in 0..d_v {
            let mut acc = 0.0;
            for i in 0..d_a {
                acc += z_d[h][i] * qkv.w_v[i * d_v + c];
            }
            v[h][c] = acc;
        }
    }

    // logits L_h = (Q·K_h) * scale  → stable softmax A_h
    let mut logits = vec![0.0f32; h_count];
    for h in 0..h_count {
        let mut dot = 0.0;
        for m in 0..d_k {
            dot += q[m] * k[h][m];
        }
        logits[h] = dot * scale;
    }
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut a = vec![0.0f32; h_count];
    let mut sum_e = 0.0f32;
    for h in 0..h_count {
        let e = (logits[h] - max_l).exp();
        a[h] = e;
        sum_e += e;
    }
    if sum_e > 0.0 {
        for h in 0..h_count {
            a[h] /= sum_e;
        }
    } else {
        let u = 1.0 / h_count as f32;
        for h in 0..h_count {
            a[h] = u;
        }
    }

    // O_d[c] = Σ_h A_h V_d[h][c]
    let mut out = vec![0.0f32; d_v];
    for h in 0..h_count {
        for c in 0..d_v {
            out[c] += a[h] * v[h][c];
        }
    }

    // score = w_attn * <q_a, O_d> + w_evidence * evidence + bias(0 in training)
    let mut score = 0.0f32;
    for c in 0..d_v {
        score += q_a[c] * out[c];
    }
    score = w_attn * score + w_evidence * evidence.unwrap_or(0.0);

    // --- Backward ---
    // d score / dV_h[c] = w_attn * q_a[c] * A_h
    // d score / dA_h    = w_attn * Σ_c q_a[c] V_h[c]
    let mut d_score_da = vec![0.0f32; h_count];
    let mut g_v = vec![0.0f32; d_a * d_v];
    for h in 0..h_count {
        let mut dot_qa_v = 0.0;
        for c in 0..d_v {
            let dv = w_attn * q_a[c] * a[h];
            dot_qa_v += q_a[c] * v[h][c];
            // g_v[i][c] += dv * z[h][i]
            if dv != 0.0 {
                for i in 0..d_a {
                    g_v[i * d_v + c] += dv * z_d[h][i];
                }
            }
        }
        d_score_da[h] = w_attn * dot_qa_v;
    }

    // G = Σ_h (d score/dA_h) A_h ;  d score/dL_k = A_k (d score/dA_k - G)
    let mut gsum = 0.0f32;
    for h in 0..h_count {
        gsum += d_score_da[h] * a[h];
    }
    let mut d_score_dl = vec![0.0f32; h_count];
    for k in 0..h_count {
        d_score_dl[k] = a[k] * (d_score_da[k] - gsum);
    }

    // d score/dt_k = d score/dL_k * scale ;  t_k = Q·K_k
    // d score/dQ_m = Σ_k d score/dt_k * K_k[m]
    // d score/dK_k[m] = d score/dt_k * Q_m
    let mut d_score_dq = vec![0.0f32; d_k];
    for hh in 0..h_count {
        let dt = d_score_dl[hh] * scale;
        if dt == 0.0 {
            continue;
        }
        for m in 0..d_k {
            d_score_dq[m] += dt * k[hh][m];
        }
    }

    // grad W_Q[m][j] = d score/dQ_j * q_a[m]
    let mut g_q = vec![0.0f32; d_a * d_k];
    for m in 0..d_a {
        let qm = q_a[m];
        if qm == 0.0 {
            continue;
        }
        for j in 0..d_k {
            g_q[m * d_k + j] = d_score_dq[j] * qm;
        }
    }

    // grad W_K[i][m] = Σ_k d score/dt_k * Q_m * z_d[k][i]
    let mut g_k = vec![0.0f32; d_a * d_k];
    for k in 0..h_count {
        let dt = d_score_dl[k] * scale;
        if dt == 0.0 {
            continue;
        }
        for m in 0..d_k {
            let qm = q[m] * dt;
            if qm == 0.0 {
                continue;
            }
            for i in 0..d_a {
                g_k[i * d_k + m] += qm * z_d[k][i];
            }
        }
    }

    Ok(QkvGradient {
        g_q,
        g_k,
        g_v,
        score,
    })
}

/// Deterministic contrastive trainer for the C7 Q/K/V projections.
///
/// Loss: InfoNCE over the per-candidate attention scores within each
/// query's candidate set (positive + hard negatives):
///   loss = -ln( softmax( s_pos / τ )_pos )
/// Weight updates are Adam (deterministic — no stochastic sampling anywhere),
/// so the same seed, dataset, and config reproduce bit-identical weights.
///
/// This module provides training ONLY; it never touches test data. Callers
/// must pass train-split aligned examples (spec C7 constraints).
#[derive(Debug, Clone)]
pub struct ContrastiveQkvTrainer {
    cfg: TrainingConfig,
    /// Adam state per matrix (m_q, v_q), (m_k, v_k), (m_v, v_v).
    adam_m: [Vec<f32>; 3],
    adam_v: [Vec<f32>; 3],
    t: usize,
}

impl ContrastiveQkvTrainer {
    pub fn new(cfg: TrainingConfig) -> Self {
        Self {
            cfg,
            adam_m: [Vec::new(), Vec::new(), Vec::new()],
            adam_v: [Vec::new(), Vec::new(), Vec::new()],
            t: 0,
        }
    }

    /// Train the given projection for `cfg.epochs` epochs.
    /// Returns the best-seen weights (lowest average train loss) and a report.
    pub fn train(
        &mut self,
        init: QkvProjection,
        dataset: &QkvDataset,
        w_attn: f32,
        w_evidence: f32,
    ) -> Result<TrainedModel> {
        let (nq, nk, nv) = (init.w_q.len(), init.w_k.len(), init.w_v.len());
        self.adam_m = [vec![0.0; nq], vec![0.0; nk], vec![0.0; nv]];
        self.adam_v = [vec![0.0; nq], vec![0.0; nk], vec![0.0; nv]];
        self.t = 0;

        let mut best: Option<(f32, QkvProjection)> = None;
        let mut history = Vec::new();

        let mut w = init;
        for _epoch in 0..self.cfg.epochs {
            let mut epoch_loss = 0.0;
            if dataset.examples.is_empty() {
                return Err(AttentionError::EmptyInput("empty dataset".into()));
            }
            for ex in &dataset.examples {
                let loss = self.step(&mut w, ex, w_attn, w_evidence)?;
                epoch_loss += loss;
            }
            let avg = epoch_loss / dataset.examples.len() as f32;
            history.push(avg);
            if best.as_ref().map(|(l, _)| avg < *l).unwrap_or(true) {
                best = Some((avg, w.clone()));
            }
        }

        let (_, best_w) = best.ok_or_else(|| AttentionError::EmptyInput("no training".into()))?;
        Ok(TrainedModel {
            projection: best_w,
            loss_history: history,
            epochs_run: self.cfg.epochs,
            seed: self.cfg.seed,
            config: self.cfg.clone(),
        })
    }

    /// One deterministic Adam update for one example. Returns the loss.
    fn step(
        &mut self,
        w: &mut QkvProjection,
        ex: &QkvExample,
        w_attn: f32,
        w_evidence: f32,
    ) -> Result<f32> {
        // Scores + gradients for positive and negatives.
        let pos_grad = candidate_gradient(
            w,
            &ex.q_a,
            &ex.positive,
            w_attn,
            ex.positive_evidence,
            w_evidence,
        )?;
        let mut scores = vec![pos_grad.score];
        let mut neg_grads = Vec::with_capacity(ex.negatives.len());
        for (i, neg) in ex.negatives.iter().enumerate() {
            let g = candidate_gradient(
                w,
                &ex.q_a,
                neg,
                w_attn,
                ex.negatives_evidence.get(i).copied().flatten(),
                w_evidence,
            )?;
            scores.push(g.score);
            neg_grads.push(g);
        }

        // Stable softmax over scores / temperature.
        let tau = self.cfg.temperature;
        let scaled: Vec<f32> = scores.iter().map(|s| s / tau).collect();
        let max_s = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = Vec::with_capacity(scaled.len());
        let mut sum = 0.0f32;
        for s in &scaled {
            let e = (s - max_s).exp();
            exps.push(e);
            sum += e;
        }
        let loss = if sum > 0.0 {
            let lp = (exps[0] / sum).ln().max(-80.0);
            -lp
        } else {
            f32::INFINITY
        };

        // ∂loss/∂s_c = softmax_c - δ_{c=pos}
        let mut p = Vec::with_capacity(exps.len());
        for e in &exps {
            p.push(e / sum);
        }
        let d_loss_ds: Vec<f32> = p
            .iter()
            .enumerate()
            .map(|(i, &pi)| (pi - if i == 0 { 1.0 } else { 0.0 }) / tau)
            .collect();

        // Accumulate weighted gradients per matrix.
        let mut g_q = vec![0.0f32; pos_grad.g_q.len()];
        let mut g_k = vec![0.0f32; pos_grad.g_k.len()];
        let mut g_v = vec![0.0f32; pos_grad.g_v.len()];
        {
            let c0 = d_loss_ds[0];
            if c0 != 0.0 {
                for i in 0..g_q.len() {
                    g_q[i] += c0 * pos_grad.g_q[i];
                }
                for i in 0..g_k.len() {
                    g_k[i] += c0 * pos_grad.g_k[i];
                }
                for i in 0..g_v.len() {
                    g_v[i] += c0 * pos_grad.g_v[i];
                }
            }
            for (gi, g) in neg_grads.iter().enumerate() {
                let c = d_loss_ds[gi + 1];
                if c == 0.0 {
                    continue;
                }
                for i in 0..g_q.len() {
                    g_q[i] += c * g.g_q[i];
                }
                for i in 0..g_k.len() {
                    g_k[i] += c * g.g_k[i];
                }
                for i in 0..g_v.len() {
                    g_v[i] += c * g.g_v[i];
                }
            }
        }

        // L2 + Adam update per matrix.
        let lr = self.cfg.learning_rate;
        let l2 = self.cfg.l2;
        self.t += 1;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let b1t = beta1.powi(self.t as i32);
        let b2t = beta2.powi(self.t as i32);

        let upd =
            |params: &mut [f32], grads: &[f32], m: &mut Vec<f32>, v: &mut Vec<f32>| -> Result<()> {
                for i in 0..params.len() {
                    let g = grads[i] + l2 * params[i];
                    if !g.is_finite() {
                        return Err(AttentionError::NonFinite("training gradient".into()));
                    }
                    let mi = beta1 * m[i] + (1.0 - beta1) * g;
                    let vi = beta2 * v[i] + (1.0 - beta2) * g * g;
                    m[i] = mi;
                    v[i] = vi;
                    let m_hat = mi / (1.0 - b1t);
                    let v_hat = vi / (1.0 - b2t);
                    params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
                Ok(())
            };
        upd(&mut w.w_q, &g_q, &mut self.adam_m[0], &mut self.adam_v[0])?;
        upd(&mut w.w_k, &g_k, &mut self.adam_m[1], &mut self.adam_v[1])?;
        upd(&mut w.w_v, &g_v, &mut self.adam_m[2], &mut self.adam_v[2])?;

        Ok(loss)
    }
}

/// Result of [`ContrastiveQkvTrainer::train`]: best weights + a report for the
/// model card.
#[derive(Debug, Clone)]
pub struct TrainedModel {
    pub projection: QkvProjection,
    pub loss_history: Vec<f32>,
    pub epochs_run: usize,
    pub seed: u64,
    pub config: TrainingConfig,
}

impl TrainedModel {
    /// Compose a runtime `AttentionConfig` (with the scorer) for inference.
    pub fn to_config(
        &self,
        alignments: Vec<crate::alignment::AlignmentProjection>,
        query_alignment: crate::alignment::AlignmentProjection,
        scorer: crate::scorer::AttentionScorer,
        use_evidence: bool,
    ) -> crate::config::AttentionConfig {
        crate::config::AttentionConfig::learned(
            alignments.len(),
            self.projection.attention_dim,
            self.projection.key_dim,
            self.projection.value_dim,
            alignments,
            query_alignment,
            self.projection.clone(),
            use_evidence,
            scorer,
        )
    }

    /// Fill a `TrainingMeta` from this training run.
    pub fn to_meta(&self, dataset_hash: u64) -> TrainingMeta {
        TrainingMeta {
            seed: self.seed,
            dataset_hash,
            objective: "contrastive".into(),
            learning_rate: self.config.learning_rate,
            batch_size: self.config.batch_size,
            epochs_run: self.epochs_run,
            best_val_loss: self.loss_history.last().copied(),
            l2: self.config.l2,
            timestamp_unix: crate::model::TrainingMeta::default().timestamp_unix,
            code_commit: String::new(),
            hardware: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::C7ModelCard;
    use crate::scorer::AttentionScorer;

    fn proj(d: usize) -> QkvProjection {
        QkvProjection::identity(d)
    }

    fn aligned_vec(d: usize, active: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; d];
        v[0] = active;
        v[1] = 0.5;
        v
    }

    fn toy_dataset(d: usize, heads: usize, examples: usize) -> QkvDataset {
        let mut b = QkvDatasetBuilder::new(d, (0..heads).map(|h| format!("h{h}")).collect());
        for e in 0..examples {
            let q = aligned_vec(d, 1.0 + e as f32 * 0.1);
            let pos = vec![aligned_vec(d, 1.0); heads];
            let hard_neg = vec![aligned_vec(d, -1.0); heads];
            let easy_neg = vec![aligned_vec(d, -0.9); heads];
            b.push(QkvExample {
                q_a: q,
                positive: pos,
                positive_evidence: None,
                negatives: vec![hard_neg, easy_neg],
                negatives_evidence: vec![None, None],
            })
            .unwrap();
        }
        b.build()
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let d = 4;
        let qkv = proj(d);
        let q_a = vec![0.8, 0.1, -0.3, 0.4];
        let z = vec![vec![0.5, -0.2, 0.9, 0.1], vec![0.1, 0.7, -0.4, 0.2]];
        let g = candidate_gradient(&qkv, &q_a, &z, 1.0, None, 0.0).unwrap();

        // Score as a function of a single weight; numeric derivative.
        let eps = 1e-3f32;
        let check = |w: &QkvProjection, score_fn: &dyn Fn(&QkvProjection) -> f32| -> Vec<f32> {
            // Perturb each weight in [w_q, w_k, w_v] in order.
            let mut num = Vec::new();
            for which in 0..3usize {
                let len = [w.w_q.len(), w.w_k.len(), w.w_v.len()][which];
                for i in 0..len {
                    let mut up_w = w.clone();
                    let mut down_w = w.clone();
                    match which {
                        0 => {
                            up_w.w_q[i] += eps;
                            down_w.w_q[i] -= eps;
                        }
                        1 => {
                            up_w.w_k[i] += eps;
                            down_w.w_k[i] -= eps;
                        }
                        _ => {
                            up_w.w_v[i] += eps;
                            down_w.w_v[i] -= eps;
                        }
                    }
                    let up = score_fn(&up_w);
                    let down = score_fn(&down_w);
                    num.push((up - down) / (2.0 * eps));
                }
            }
            num
        };
        let score_fn = |w: &QkvProjection| {
            candidate_gradient(w, &q_a, &z, 1.0, None, 0.0)
                .unwrap()
                .score
        };
        let num = check(&qkv, &score_fn);

        let ana = {
            let mut acc = g.g_q.clone();
            acc.extend_from_slice(&g.g_k);
            acc.extend_from_slice(&g.g_v);
            acc
        };
        for i in 0..ana.len() {
            let diff = (ana[i] - num[i]).abs();
            let tol = 1e-2 * (1.0 + ana[i].abs());
            assert!(
                diff <= tol,
                "grad[{i}] analytic={} numeric={} diff={diff}",
                ana[i],
                num[i]
            );
        }
    }

    #[test]
    fn training_reduces_loss_and_is_deterministic() {
        let d = 4;
        let heads = 2;
        let ds = toy_dataset(d, heads, 6);
        let cfg = TrainingConfig {
            seed: 20260925,
            learning_rate: 0.05,
            epochs: 40,
            batch_size: 8,
            temperature: 0.07,
            l2: 1e-4,
        };
        let mut t1 = ContrastiveQkvTrainer::new(cfg.clone());
        let m1 = t1
            .train(QkvProjection::random(d, d, d, 42), &ds, 1.0, 0.0)
            .unwrap();
        let mut t2 = ContrastiveQkvTrainer::new(cfg.clone());
        let m2 = t2
            .train(QkvProjection::random(d, d, d, 42), &ds, 1.0, 0.0)
            .unwrap();

        // Determinism: bit-identical weights and losses.
        assert_eq!(m1.projection.w_q, m2.projection.w_q);
        assert_eq!(m1.projection.w_k, m2.projection.w_k);
        assert_eq!(m1.projection.w_v, m2.projection.w_v);
        assert_eq!(m1.loss_history, m2.loss_history);

        // Loss decreases overall.
        let first = m1.loss_history.first().copied().unwrap();
        let last = m1.loss_history.last().copied().unwrap();
        assert!(last < first, "loss must decrease: {first} -> {last}");

        // Composable to a valid model card.
        let scorer = AttentionScorer::default();
        let align_proj = crate::alignment::AlignmentProjection::identity(d);
        let cfg_out = m1.to_config(
            vec![align_proj.clone(), align_proj.clone()],
            align_proj.clone(),
            scorer.clone(),
            false,
        );
        assert!(cfg_out.enabled);
        let card = C7ModelCard::new(
            "c7-toy",
            "contrastive-qkv",
            d,
            d,
            d,
            (0..heads).map(|h| format!("h{h}")).collect(),
            vec![align_proj.clone(), align_proj.clone()],
            align_proj.clone(),
            m1.projection.clone(),
            scorer,
            m1.to_meta(ds.fingerprint()),
            ds.fingerprint(),
        )
        .unwrap();
        assert!(card.validate().is_ok());
    }

    #[test]
    fn dataset_builder_rejects_bad_dims() {
        let d = 4;
        let mut b = QkvDatasetBuilder::new(d, vec!["h0".into(), "h1".into()]);
        assert!(b
            .push(QkvExample {
                q_a: vec![1.0; d],
                positive: vec![vec![1.0; d], vec![1.0; d]],
                positive_evidence: None,
                negatives: vec![vec![vec![0.5; d], vec![0.5; d]]],
                negatives_evidence: vec![None],
            })
            .is_ok());
        assert!(b
            .push(QkvExample {
                q_a: vec![1.0; d],
                positive: vec![vec![1.0; d]], // 1 head, but dataset has 2
                positive_evidence: None,
                negatives: vec![],
                negatives_evidence: vec![],
            })
            .is_err());
    }

    #[test]
    fn evidence_affects_training_score() {
        let d = 4;
        let qkv = proj(d);
        let q_a = vec![1.0, 0.0, 0.0, 0.0];
        let z = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let no_ev = candidate_gradient(&qkv, &q_a, &z, 1.0, None, 1.0).unwrap();
        let with_ev = candidate_gradient(&qkv, &q_a, &z, 1.0, Some(0.5), 1.0).unwrap();
        assert!((with_ev.score - no_ev.score - 0.5).abs() < 1e-6);
    }
}
