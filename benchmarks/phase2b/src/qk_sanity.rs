//! PH2C-QK-001 — candidate-level QK attention SANITY dataset (Phase 2C §6).
//!
//! Purpose: a synthetic dataset that ISOLATES candidate-level query–candidate
//! interaction from head selection, in the sense required by Phase 2C §6:
//!   (a) head selection is EASY — head 1 carries no signal (pure noise),
//!       so a good gating learns to weight head 0;
//!   (b) candidate ORDERING requires query-dependent interaction — the
//!       relevant candidate is constructed to rank LAST under the query
//!       cosine in the signal head, for every query (cos ≤ ~0.14 vs
//!       distractor cos ≈ 0.30);
//!   (c) therefore ANY per-head weighting of per-head cosine scores (the
//!       entire gating function class, including an oracle over weights)
//!       achieves at-most-chance R@1, while a learned bilinear
//!       score_i = (W_Q q)·(W_K x_i)/√d separates the relevant candidate
//!       exactly (up to the 0.1 noise).
//!
//! Why this isolates candidate-level attention: the gating class computes
//! s_i = Σ_h w_h(q)·ŝ_h(x_i) — a per-candidate score that is a FIXED
//! (query-independent) functional of per-head cosine features combined with
//! query-dependent mixture weights. On this dataset ŝ_signal ranks the
//! relevant candidate last for ALL queries, so no mixture weight vector can
//! place it first. The only separating signal is qᵀ M x_i with
//! M = v0⊗u0 + v1⊗u1 (M ≠ scalar·I), which only a query–candidate bilinear
//! form (QK) can express. Gating's failure here is NOT a head-selection
//! failure: we log its learned weights to document that selection is easy
//! (weight concentrated on the signal head) while ordering is impossible.
//!
//! Consequence for the phase gate (Phase 2C §6/§41): if the QK machinery
//! cannot learn THIS dataset, candidate-level QK cannot be expected to learn
//! the real corpora, the sample-efficiency study (§19) is skipped, and the
//! finding is recorded as a negative result.

use attentiondb_learned::eval::{rank_metrics, RankMetrics};
use attentiondb_learned::gating_v2::{
    train_gating, DetRng, GatingDataset, GatingMlp, HeadExample, ModelCard, Objective,
    QualityTarget, QueryExample, Split, TrainOutcome, TrainingConfig, TrainingMeta,
};
use std::fmt::Write as _;

// ---------------------------------------------------------------- constants
const DIM: usize = 8; // per-head vector dimension
const N_HEADS: usize = 2;
const POOL: usize = 10; // candidates per query (1 relevant + 9 distractors)
const N_TRAIN: usize = 600;
const N_VAL: usize = 150;
const N_TEST: usize = 250;
const NOISE: f32 = 0.03; // direction jitter for query + relevant vectors.
                         // Chosen so the anti-cosine invariant is EXACT: the relevant candidate's
                         // head-0 cosine to the query has std ≈ NOISE·√2 ≈ 0.042, while every
                         // distractor sits at cos ≈ DISTRACTOR_COS/√(1+DISTRACTOR_COS²) ≈ 0.286
                         // (>6σ above the relevant's worst case). Query family separation stays
                         // natural: cos(q, u_f) ≈ 0.9995.
const DISTRACTOR_COS: f32 = 0.3; // distractor cosine to the family axis u_f
const TOP_K: usize = 1; // 1 relevant doc per query; ordering is the task
const DATA_SEED: u64 = 0x5A17;

const LR_GRID: [f32; 2] = [0.05, 0.01];
const T_GRID: [f32; 4] = [0.25, 0.5, 1.0, 2.0];
const MAX_EPOCHS: usize = 300;
const PATIENCE: usize = 30;
const BATCH: usize = 32;
const DQ: usize = 8; // QK inner dimension

// ------------------------------------------------------------------- data
/// Per-query content sidecar (candidate vectors are NOT part of the
/// historical GatingDataset schema; candidate-level QK needs them).
#[derive(Clone)]
pub struct QueryContent {
    /// Concatenated head vectors (N_HEADS * DIM).
    pub q: Vec<f32>,
    /// Per candidate, same layout, pool order (candidate 0 = relevant).
    pub cands: Vec<Vec<f32>>,
}

#[derive(Clone)]
pub struct SanityData {
    pub ds: GatingDataset,
    pub content: Vec<QueryContent>,
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in v.iter_mut() {
        *x /= n;
    }
}

fn unit_noise(rng: &mut DetRng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
    normalize(&mut v);
    v
}

/// axis + NOISE·(unit noise), normalized: cos(v, axis) ≈ 1 - O(NOISE²).
fn near_axis(rng: &mut DetRng, dim: usize, axis: usize, noise: f32) -> Vec<f32> {
    let n = unit_noise(rng, dim);
    let mut v = vec![0.0f32; dim];
    for k in 0..dim {
        v[k] = if k == axis { 1.0 } else { 0.0 } + noise * n[k];
    }
    normalize(&mut v);
    v
}

/// Generate the sanity dataset. Family axes: u0 = e0, u1 = e1 (query
/// direction); relevance axes: v0 = e2, v1 = e3. Distractor noise is
/// projected orthogonal to BOTH u_f and v_f before mixing in DISTRACTOR_COS.
pub fn generate(seed: u64) -> SanityData {
    let mut rng = DetRng::new(seed);
    let n = N_TRAIN + N_VAL + N_TEST;
    let mut queries = Vec::with_capacity(n);
    let mut content = Vec::with_capacity(n);
    for qi in 0..n {
        let split = if qi < N_TRAIN {
            Split::Train
        } else if qi < N_TRAIN + N_VAL {
            Split::Val
        } else {
            Split::Test
        };
        let f = qi % 2; // balanced families, deterministic
        let (uf, vf) = (f, 2 + f); // coordinate indices of e_{u_f}, e_{v_f}

        // ---- query: head0 near u_f, head1 pure noise ----
        let q0 = near_axis(&mut rng, DIM, uf, NOISE);
        let q1 = unit_noise(&mut rng, DIM);
        let mut qcat = q0.clone();
        qcat.extend_from_slice(&q1);

        // ---- relevant: head0 near v_f (⊥ query axis), head1 noise ----
        let r0 = near_axis(&mut rng, DIM, vf, NOISE);
        let r1 = unit_noise(&mut rng, DIM);
        let mut rcat = r0.clone();
        rcat.extend_from_slice(&r1);

        // ---- distractors: cos(u_f) = DISTRACTOR_COS, ⊥ v_f ----
        let mut distractors = Vec::with_capacity(POOL - 1);
        for _ in 0..(POOL - 1) {
            let mut w = unit_noise(&mut rng, DIM);
            w[uf] = 0.0;
            w[vf] = 0.0;
            normalize(&mut w);
            let mut d0 = vec![0.0f32; DIM];
            for k in 0..DIM {
                d0[k] = DISTRACTOR_COS * (if k == uf { 1.0 } else { 0.0 }) + w[k];
            }
            normalize(&mut d0);
            let d1 = unit_noise(&mut rng, DIM);
            let mut dcat = d0.clone();
            dcat.extend_from_slice(&d1);
            distractors.push(dcat);
        }

        // pool order: relevant first, then distractors; ids unique per query
        let base = (qi as u64) * 100;
        let cands_cat: Vec<Vec<f32>> = std::iter::once(rcat)
            .chain(distractors.iter().cloned())
            .collect();
        let ids: Vec<u64> = (0..POOL).map(|j| base + j as u64).collect();
        let gt = vec![base];

        // per-head cosines (raw), minmax-normalized per pool
        let mut heads = Vec::with_capacity(N_HEADS);
        for h in 0..N_HEADS {
            let raw: Vec<f32> = cands_cat
                .iter()
                .map(|c| (0..DIM).map(|k| qcat[h * DIM + k] * c[h * DIM + k]).sum())
                .collect();
            let mut norm = raw.clone();
            minmax(&mut norm);
            let mut ranked: Vec<(u64, f32)> =
                ids.iter().copied().zip(raw.iter().copied()).collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            let m = rank_metrics(&ranked, &gt, TOP_K);
            heads.push(HeadExample {
                candidates: ids.clone(),
                raw_scores: raw.clone(),
                norm_scores: norm,
                exact_scores: raw, // sanity world: exact ≡ head cosine (moved)
                recall_at_k: m.recall_at_1 as f32, // top_k = 1
                ndcg_at_k: m.ndcg_at_10 as f32,
                mrr: m.mrr as f32,
            });
        }

        queries.push(QueryExample {
            query_id: qi as u64,
            query: qcat.clone(),
            query_group: Some(f as u32), // dataset metadata only, never a model input (§36)
            split,
            ground_truth: gt,
            heads,
        });
        content.push(QueryContent {
            q: qcat,
            cands: cands_cat,
        });
    }

    let ds = GatingDataset {
        format: "attentiondb-gating-dataset".into(),
        version: 1,
        num_heads: N_HEADS,
        input_dim: N_HEADS * DIM,
        top_k: TOP_K,
        corpus_desc: "qk-sanity: anti-cosine paired-family construction (PH2C-QK-001)".into(),
        seed,
        queries,
    };
    SanityData { ds, content }
}

fn minmax(v: &mut [f32]) {
    let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1e-12);
    for x in v.iter_mut() {
        *x = (*x - lo) / span;
    }
}

// --------------------------------------------------------------- QK model
/// Linear candidate-level QK (Phase 2C §3): Q = W_Q·q, K_i = W_K·x_i,
/// score_i = Q·K_i/√d. No hidden layers — the simplest valid model (§4).
pub struct QkModel {
    wq: Vec<f32>, // DQ x DIN
    wk: Vec<f32>, // DQ x DIN
    din: usize,
}

impl QkModel {
    pub fn new(din: usize, seed: u64) -> Self {
        let mut rng = DetRng::new(seed ^ 0xA17);
        QkModel {
            wq: (0..DQ * din).map(|_| rng.gauss() * 0.1).collect(),
            wk: (0..DQ * din).map(|_| rng.gauss() * 0.1).collect(),
            din,
        }
    }

    fn q_proj(&self, q: &[f32]) -> Vec<f32> {
        (0..DQ)
            .map(|i| {
                (0..self.din)
                    .map(|j| self.wq[i * self.din + j] * q[j])
                    .sum()
            })
            .collect()
    }

    fn k_proj(&self, x: &[f32]) -> Vec<f32> {
        (0..DQ)
            .map(|i| {
                (0..self.din)
                    .map(|j| self.wk[i * self.din + j] * x[j])
                    .sum()
            })
            .collect()
    }

    /// Scores for one query's candidate pool (pool order preserved).
    pub fn scores(&self, q: &[f32], cands: &[Vec<f32>]) -> Vec<f32> {
        let qp = self.q_proj(q);
        let scale = (DQ as f32).sqrt();
        cands
            .iter()
            .map(|x| {
                let kp = self.k_proj(x);
                qp.iter().zip(kp.iter()).map(|(a, b)| a * b).sum::<f32>() / scale
            })
            .collect()
    }

    fn snapshot(&self) -> Vec<f32> {
        let mut v = self.wq.clone();
        v.extend_from_slice(&self.wk);
        v
    }

    fn restore(&mut self, snap: &[f32]) {
        let n = DQ * self.din;
        self.wq.copy_from_slice(&snap[..n]);
        self.wk.copy_from_slice(&snap[n..]);
    }
}

// ----------------------------------------------------------------- Adam
/// Mirrors gating_v2::Adam exactly (that impl is crate-private).
struct Adam {
    lr: f32,
    t: u64,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    fn new(n: usize, lr: f32) -> Self {
        Adam {
            lr,
            t: 0,
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }
    fn step(&mut self, w: &mut [f32], g: &[f32]) {
        self.t += 1;
        let c1 = 1.0 - 0.9f32.powi(self.t as i32);
        let c2 = 1.0 - 0.999f32.powi(self.t as i32);
        for i in 0..w.len() {
            self.m[i] = 0.9 * self.m[i] + 0.1 * g[i];
            self.v[i] = 0.999 * self.v[i] + 0.001 * g[i] * g[i];
            let mh = self.m[i] / c1;
            let vh = self.v[i] / c2;
            w[i] -= self.lr * mh / (vh.sqrt() + 1e-8);
        }
    }
}

fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}

/// dlogits = p ∘ (dw − ⟨p, dw⟩).
fn softmax_backward(p: &[f32], dw: &[f32]) -> Vec<f32> {
    let dot: f32 = p.iter().zip(dw.iter()).map(|(a, b)| a * b).sum();
    p.iter()
        .zip(dw.iter())
        .map(|(pi, gi)| pi * (gi - dot))
        .collect()
}

// ------------------------------------------------------------- QK trainer
pub struct TrainCurve {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_r1: f64,
}

/// InfoNCE (softmax-CE with 1 positive) at temperature T; Adam, minibatch.
/// Selects the snapshot with best validation R@1 (early stop, Phase 2C §4).
pub fn train_qk(data: &SanityData, lr: f32, t: f32, seed: u64) -> (QkModel, Vec<TrainCurve>) {
    let din = data.ds.input_dim;
    let mut model = QkModel::new(din, seed);
    let mut opt_wq = Adam::new(model.wq.len(), lr);
    let mut opt_wk = Adam::new(model.wk.len(), lr);
    let mut rng = DetRng::new(seed ^ 0x7D2B);
    let train: Vec<usize> = (0..data.ds.queries.len())
        .filter(|&i| data.ds.queries[i].split == Split::Train)
        .collect();
    let val: Vec<usize> = (0..data.ds.queries.len())
        .filter(|&i| data.ds.queries[i].split == Split::Val)
        .collect();

    let mut curves = Vec::new();
    let mut best_r1 = -1.0f64;
    let mut best_snap = model.snapshot();
    let mut stall = 0usize;
    let scale = (DQ as f32).sqrt();

    for epoch in 0..MAX_EPOCHS {
        let mut order = train.clone();
        rng.shuffle(&mut order);
        let mut loss_sum = 0.0f32;
        let mut gwq = vec![0.0f32; model.wq.len()];
        let mut gwk = vec![0.0f32; model.wk.len()];
        let mut in_batch = 0usize;
        for &qi in &order {
            let qc = &data.content[qi];
            let qp = model.q_proj(&qc.q);
            let kps: Vec<Vec<f32>> = qc.cands.iter().map(|x| model.k_proj(x)).collect();
            let scores: Vec<f32> = kps
                .iter()
                .map(|kp| qp.iter().zip(kp.iter()).map(|(a, b)| a * b).sum::<f32>() / scale)
                .collect();
            let logits_t: Vec<f32> = scores.iter().map(|s| s / t).collect();
            let p = softmax(&logits_t);
            loss_sum += -p[0].ln(); // candidate 0 is the relevant one
            let ds: Vec<f32> = p
                .iter()
                .enumerate()
                .map(|(i, pi)| (pi - if i == 0 { 1.0 } else { 0.0 }) / t)
                .collect();
            let mut dqp = [0.0f32; DQ];
            for (i, kp) in kps.iter().enumerate() {
                for d in 0..DQ {
                    dqp[d] += ds[i] * kp[d] / scale;
                    for j in 0..din {
                        gwk[d * din + j] += ds[i] * qp[d] * qc.cands[i][j] / scale;
                    }
                }
            }
            for d in 0..DQ {
                for j in 0..din {
                    gwq[d * din + j] += dqp[d] * qc.q[j];
                }
            }
            in_batch += 1;
            if in_batch == BATCH {
                for g in gwq.iter_mut() {
                    *g /= BATCH as f32;
                }
                for g in gwk.iter_mut() {
                    *g /= BATCH as f32;
                }
                opt_wq.step(&mut model.wq, &gwq);
                opt_wk.step(&mut model.wk, &gwk);
                gwq.iter_mut().for_each(|g| *g = 0.0);
                gwk.iter_mut().for_each(|g| *g = 0.0);
                in_batch = 0;
            }
        }
        if in_batch > 0 {
            let nf = in_batch as f32;
            for g in gwq.iter_mut() {
                *g /= nf;
            }
            for g in gwk.iter_mut() {
                *g /= nf;
            }
            opt_wq.step(&mut model.wq, &gwq);
            opt_wk.step(&mut model.wk, &gwk);
        }

        let r1 = eval_qk_r1(&model, data, &val);
        curves.push(TrainCurve {
            epoch,
            train_loss: loss_sum / train.len() as f32,
            val_r1: r1,
        });
        if r1 > best_r1 + 1e-6 {
            best_r1 = r1;
            best_snap = model.snapshot();
            stall = 0;
        } else {
            stall += 1;
            if stall >= PATIENCE {
                break;
            }
        }
    }
    model.restore(&best_snap);
    (model, curves)
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] + 1e-12 || (v[i] - v[best]).abs() <= 1e-12 && i < best {
            best = i;
        }
    }
    best
}

fn eval_qk_r1(model: &QkModel, data: &SanityData, idx: &[usize]) -> f64 {
    let mut hits = 0usize;
    for &i in idx {
        let s = model.scores(&data.content[i].q, &data.content[i].cands);
        hits += (argmax(&s) == 0) as usize;
    }
    hits as f64 / idx.len().max(1) as f64
}

pub fn eval_qk(model: &QkModel, data: &SanityData, split: Split) -> RankMetrics {
    let rs: Vec<RankMetrics> = data
        .ds
        .queries
        .iter()
        .filter(|q| q.split == split)
        .map(|q| {
            let s = model.scores(
                &data.content[q.query_id as usize].q,
                &data.content[q.query_id as usize].cands,
            );
            let mut ranked: Vec<(u64, f32)> = q.heads[0]
                .candidates
                .iter()
                .copied()
                .zip(s.iter().copied())
                .collect();
            // rank_metrics consumes slice ORDER as the ranking — sort here.
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            rank_metrics(&ranked, &q.ground_truth, data.ds.top_k)
        })
        .collect();
    avg(&rs)
}

fn avg(rs: &[RankMetrics]) -> RankMetrics {
    let n = rs.len().max(1) as f64;
    RankMetrics {
        recall_at_1: rs.iter().map(|r| r.recall_at_1).sum::<f64>() / n,
        recall_at_5: rs.iter().map(|r| r.recall_at_5).sum::<f64>() / n,
        recall_at_10: rs.iter().map(|r| r.recall_at_10).sum::<f64>() / n,
        recall_at_50: rs.iter().map(|r| r.recall_at_50).sum::<f64>() / n,
        ndcg_at_10: rs.iter().map(|r| r.ndcg_at_10).sum::<f64>() / n,
        mrr: rs.iter().map(|r| r.mrr).sum::<f64>() / n,
    }
}

// --------------------------------------------- gating variant B (InfoNCE)
/// Gating trained with the QK objective (fairness control, Phase 2C §6):
/// softmax head weights from an MLP over the query, fused score
/// s_i = Σ_h w_h·ŝ_h(x_i), trained with the same InfoNCE-at-T loss. If THIS
/// arm also fails, the isolation is robust to the gating objective — the
/// failure is the function class, not the loss.
struct GateB {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    din: usize,
    hid: usize,
}

impl GateB {
    fn new(din: usize, hid: usize, seed: u64) -> Self {
        let mut rng = DetRng::new(seed ^ 0xB21);
        GateB {
            w1: (0..hid * din).map(|_| rng.gauss() * 0.1).collect(),
            b1: vec![0.0; hid],
            w2: (0..N_HEADS * hid).map(|_| rng.gauss() * 0.1).collect(),
            b2: vec![0.0; N_HEADS],
            din,
            hid,
        }
    }

    fn weights(&self, q: &[f32]) -> Vec<f32> {
        let h: Vec<f32> = (0..self.hid)
            .map(|i| {
                (self.b1[i]
                    + (0..self.din)
                        .map(|j| self.w1[i * self.din + j] * q[j])
                        .sum::<f32>())
                .tanh()
            })
            .collect();
        let lg: Vec<f32> = (0..N_HEADS)
            .map(|i| {
                self.b2[i]
                    + (0..self.hid)
                        .map(|j| self.w2[i * self.hid + j] * h[j])
                        .sum::<f32>()
            })
            .collect();
        softmax(&lg)
    }

    fn snapshot(&self) -> Vec<f32> {
        let mut v = self.w1.clone();
        v.extend_from_slice(&self.b1);
        v.extend_from_slice(&self.w2);
        v.extend_from_slice(&self.b2);
        v
    }

    fn restore(&mut self, snap: &[f32]) {
        let (din, hid) = (self.din, self.hid);
        let mut o = 0;
        self.w1.copy_from_slice(&snap[o..o + hid * din]);
        o += hid * din;
        self.b1.copy_from_slice(&snap[o..o + hid]);
        o += hid;
        self.w2.copy_from_slice(&snap[o..o + N_HEADS * hid]);
        o += N_HEADS * hid;
        self.b2.copy_from_slice(&snap[o..o + N_HEADS]);
    }
}

fn fused_scores(q: &QueryExample, w: &[f32]) -> Vec<f32> {
    (0..POOL)
        .map(|i| {
            q.heads
                .iter()
                .enumerate()
                .map(|(h, he)| w[h] * he.norm_scores[i])
                .sum::<f32>()
        })
        .collect()
}

fn train_gate_b(data: &SanityData, lr: f32, t: f32, seed: u64) -> (GateB, f32) {
    let din = data.ds.input_dim;
    let hid = 16;
    let mut m = GateB::new(din, hid, seed);
    let mut opts = [
        Adam::new(m.w1.len(), lr),
        Adam::new(m.b1.len(), lr),
        Adam::new(m.w2.len(), lr),
        Adam::new(m.b2.len(), lr),
    ];
    let mut rng = DetRng::new(seed ^ 0x9E37);
    let train: Vec<usize> = (0..data.ds.queries.len())
        .filter(|&i| data.ds.queries[i].split == Split::Train)
        .collect();
    let val: Vec<usize> = (0..data.ds.queries.len())
        .filter(|&i| data.ds.queries[i].split == Split::Val)
        .collect();

    let mut best_r1 = -1.0f64;
    let mut best = m.snapshot();
    let mut best_loss = 0.0f32;
    let mut stall = 0usize;

    for _epoch in 0..MAX_EPOCHS {
        let mut order = train.clone();
        rng.shuffle(&mut order);
        let mut loss_sum = 0.0f32;
        let mut g = [
            vec![0.0f32; m.w1.len()],
            vec![0.0f32; m.b1.len()],
            vec![0.0f32; m.w2.len()],
            vec![0.0f32; m.b2.len()],
        ];
        let mut in_batch = 0usize;
        for &qi in &order {
            let q = &data.ds.queries[qi];
            let w = m.weights(&q.query);
            let scores = fused_scores(q, &w);
            let logits_t: Vec<f32> = scores.iter().map(|s| s / t).collect();
            let p = softmax(&logits_t);
            loss_sum += -p[0].ln();
            let ds: Vec<f32> = p
                .iter()
                .enumerate()
                .map(|(i, pi)| (pi - if i == 0 { 1.0 } else { 0.0 }) / t)
                .collect();
            // dw_h = Σ_i ds_i·norm_{h,i}; then through the softmax jacobian
            let mut dw = [0.0f32; N_HEADS];
            for (h, he) in q.heads.iter().enumerate() {
                for (i, di) in ds.iter().enumerate() {
                    dw[h] += di * he.norm_scores[i];
                }
            }
            let dlog = softmax_backward(&w, &dw);
            // MLP backward
            let hpre: Vec<f32> = (0..hid)
                .map(|i| {
                    m.b1[i]
                        + (0..din)
                            .map(|j| m.w1[i * din + j] * q.query[j])
                            .sum::<f32>()
                })
                .collect();
            let hact: Vec<f32> = hpre.iter().map(|x| x.tanh()).collect();
            for i in 0..N_HEADS {
                g[3][i] += dlog[i];
                for j in 0..hid {
                    g[2][i * hid + j] += dlog[i] * hact[j];
                    let d = dlog[i] * m.w2[i * hid + j] * (1.0 - hact[j] * hact[j]);
                    g[1][j] += d;
                    for k in 0..din {
                        g[0][j * din + k] += d * q.query[k];
                    }
                }
            }
            in_batch += 1;
            if in_batch == BATCH {
                let nf = BATCH as f32;
                for gi in g.iter_mut() {
                    gi.iter_mut().for_each(|x| *x /= nf);
                }
                step_gate_b(&mut m, &mut opts, &g);
                for gi in g.iter_mut() {
                    gi.iter_mut().for_each(|x| *x = 0.0);
                }
                in_batch = 0;
            }
        }
        if in_batch > 0 {
            let nf = in_batch as f32;
            for gi in g.iter_mut() {
                gi.iter_mut().for_each(|x| *x /= nf);
            }
            step_gate_b(&mut m, &mut opts, &g);
        }

        let r1 = eval_gate_b_r1(&m, data, &val);
        if r1 > best_r1 + 1e-6 {
            best_r1 = r1;
            best = m.snapshot();
            best_loss = loss_sum / train.len() as f32;
            stall = 0;
        } else {
            stall += 1;
            if stall >= PATIENCE {
                break;
            }
        }
    }
    m.restore(&best);
    (m, best_loss)
}

fn step_gate_b(m: &mut GateB, opts: &mut [Adam; 4], g: &[Vec<f32>; 4]) {
    let (w1, b1, w2, b2) = (&mut m.w1, &mut m.b1, &mut m.w2, &mut m.b2);
    opts[0].step(w1, &g[0]);
    opts[1].step(b1, &g[1]);
    opts[2].step(w2, &g[2]);
    opts[3].step(b2, &g[3]);
}

fn eval_gate_b_r1(m: &GateB, data: &SanityData, idx: &[usize]) -> f64 {
    let mut hits = 0usize;
    for &i in idx {
        let w = m.weights(&data.ds.queries[i].query);
        let scores = fused_scores(&data.ds.queries[i], &w);
        hits += (argmax(&scores) == 0) as usize;
    }
    hits as f64 / idx.len().max(1) as f64
}

fn eval_gate_b(m: &GateB, data: &SanityData, split: Split) -> RankMetrics {
    let rs: Vec<RankMetrics> = data
        .ds
        .queries
        .iter()
        .filter(|q| q.split == split)
        .map(|q| {
            let w = m.weights(&q.query);
            let mut acc: Vec<(u64, f32)> =
                q.heads[0].candidates.iter().map(|&c| (c, 0.0f32)).collect();
            for (h, he) in q.heads.iter().enumerate() {
                for (i, a) in acc.iter_mut().enumerate() {
                    a.1 += w[h] * he.norm_scores[i];
                }
            }
            acc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            rank_metrics(&acc, &q.ground_truth, data.ds.top_k)
        })
        .collect();
    avg(&rs)
}

// --------------------------------------------- gating variant A (as shipped)
/// The shipped Phase 2B recipe: quality-regression-family objective
/// (SoftTarget on per-head recall — the Phase 2B default), model selected on
/// validation loss by train_gating itself.
fn train_gate_a(data: &SanityData, seed: u64) -> (GatingMlp, TrainOutcome) {
    let cfg = TrainingConfig {
        objective: Objective::SoftTarget, // shipped default (F1–F10 recipe)
        lr: 0.01,
        batch_size: BATCH,
        max_epochs: MAX_EPOCHS,
        patience: PATIENCE,
        hidden: 16,
        seed,
        ..TrainingConfig::default()
    };
    let outcome = train_gating(&data.ds, &cfg, QualityTarget::Recall);
    (outcome.model.clone(), outcome)
}

fn eval_gate_a(data: &SanityData, mlp: &GatingMlp, t: f32, split: Split) -> RankMetrics {
    let rs: Vec<RankMetrics> = data
        .ds
        .queries
        .iter()
        .filter(|q| q.split == split)
        .map(|q| {
            let lg = mlp.logits(&q.query);
            let w = softmax(&lg.iter().map(|x| x / t).collect::<Vec<_>>());
            let mut acc: Vec<(u64, f32)> =
                q.heads[0].candidates.iter().map(|&c| (c, 0.0f32)).collect();
            for (h, he) in q.heads.iter().enumerate() {
                for (i, a) in acc.iter_mut().enumerate() {
                    a.1 += w[h] * he.norm_scores[i];
                }
            }
            acc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            rank_metrics(&acc, &q.ground_truth, data.ds.top_k)
        })
        .collect();
    avg(&rs)
}

fn fit_temperature_gate_a(data: &SanityData, mlp: &GatingMlp) -> f32 {
    let mut best = (T_GRID[0], -1.0f64);
    for &t in T_GRID.iter() {
        let r = eval_gate_a(data, mlp, t, Split::Val);
        if r.recall_at_1 > best.1 {
            best = (t, r.recall_at_1);
        }
    }
    best.0
}

// ---------------------------------------------------------------- oracle
/// Upper bound by construction: rank the relevant candidate first.
fn oracle_metrics(data: &SanityData, split: Split) -> RankMetrics {
    let rs: Vec<RankMetrics> = data
        .ds
        .queries
        .iter()
        .filter(|q| q.split == split)
        .map(|q| {
            let mut ranked: Vec<(u64, f32)> = q.heads[0]
                .candidates
                .iter()
                .map(|&c| {
                    (
                        c,
                        if q.ground_truth.contains(&c) {
                            1.0
                        } else {
                            0.0
                        },
                    )
                })
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            rank_metrics(&ranked, &q.ground_truth, data.ds.top_k)
        })
        .collect();
    avg(&rs)
}

fn eval_fixed_weights(data: &SanityData, split: Split, w: &[f32]) -> RankMetrics {
    let rs: Vec<RankMetrics> = data
        .ds
        .queries
        .iter()
        .filter(|q| q.split == split)
        .map(|q| {
            let mut acc: Vec<(u64, f32)> =
                q.heads[0].candidates.iter().map(|&c| (c, 0.0f32)).collect();
            for (h, he) in q.heads.iter().enumerate() {
                for (i, a) in acc.iter_mut().enumerate() {
                    a.1 += w[h] * he.norm_scores[i];
                }
            }
            acc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            rank_metrics(&acc, &q.ground_truth, data.ds.top_k)
        })
        .collect();
    avg(&rs)
}

fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let m = xs.iter().sum::<f64>() / n;
    let sd = (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n).sqrt();
    (m, sd)
}

// ------------------------------------------------------------- orchestration
struct ArmRow {
    arm: &'static str,
    seed: String,
    m: RankMetrics,
}

pub fn run(out_dir: &str, seeds: &[u64]) -> String {
    std::fs::create_dir_all(format!("{out_dir}/models")).unwrap();
    let data = generate(DATA_SEED);
    let ds_hash = data.ds.content_hash();

    // ---------- val grids (seed 42 only; selection on VAL only, §9/§10) ----
    let mut grid_rows = String::from("arm,lr,T,val_r1\n");
    let mut best_qk = (LR_GRID[0], T_GRID[0], -1.0f64);
    for &lr in LR_GRID.iter() {
        for &t in T_GRID.iter() {
            let (m, _c) = train_qk(&data, lr, t, 42);
            let val: Vec<usize> = (0..data.ds.queries.len())
                .filter(|&i| data.ds.queries[i].split == Split::Val)
                .collect();
            let r1 = eval_qk_r1(&m, &data, &val);
            grid_rows.push_str(&format!("qk,{lr},{t},{r1:.4}\n"));
            if r1 > best_qk.2 {
                best_qk = (lr, t, r1);
            }
        }
    }
    let mut best_gb = (LR_GRID[0], T_GRID[0], -1.0f64);
    for &lr in LR_GRID.iter() {
        for &t in T_GRID.iter() {
            let (m, _loss) = train_gate_b(&data, lr, t, 42);
            let val: Vec<usize> = (0..data.ds.queries.len())
                .filter(|&i| data.ds.queries[i].split == Split::Val)
                .collect();
            let r1 = eval_gate_b_r1(&m, &data, &val);
            grid_rows.push_str(&format!("gating_infnce,{lr},{t},{r1:.4}\n"));
            if r1 > best_gb.2 {
                best_gb = (lr, t, r1);
            }
        }
    }
    std::fs::write(format!("{out_dir}/val_grid.csv"), &grid_rows).unwrap();

    // ---------- frozen configs → multiseed ----------
    let (qk_lr, qk_t) = (best_qk.0, best_qk.1);
    let (gb_lr, gb_t) = (best_gb.0, best_gb.1);
    let mut rows: Vec<ArmRow> = Vec::new();

    // fixed references
    rows.push(ArmRow {
        arm: "uniform",
        seed: "agg".into(),
        m: eval_fixed_weights(&data, Split::Test, &[0.5, 0.5]),
    });
    rows.push(ArmRow {
        arm: "global_best_head0",
        seed: "agg".into(),
        m: eval_fixed_weights(&data, Split::Test, &[1.0, 0.0]),
    });
    rows.push(ArmRow {
        arm: "oracle",
        seed: "agg".into(),
        m: oracle_metrics(&data, Split::Test),
    });

    // gating A (shipped objective) — multiseed
    let mut ga: Vec<RankMetrics> = Vec::new();
    for &s in seeds {
        let (mlp, outcome) = train_gate_a(&data, s);
        let t = fit_temperature_gate_a(&data, &mlp);
        let m = eval_gate_a(&data, &mlp, t, Split::Test);
        ga.push(m);
        let meta = TrainingMeta {
            seed: s,
            dataset_hash: ds_hash,
            objective: format!("soft_target(recall)+valT={t}"),
            learning_rate: 0.01,
            batch_size: BATCH,
            epochs_run: outcome.curves.len(),
            best_val_loss: outcome.best_val_loss,
            l2: 1e-4,
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            code_commit: option_env!("GIT_HASH").unwrap_or("unknown").to_string(),
            hardware: format!(
                "cpus={}",
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            ),
        };
        let card = ModelCard::from_mlp(&mlp, meta, &format!("qk1-gateA-s{s}"), "soft_target");
        card.save(std::path::Path::new(&format!(
            "{out_dir}/models/gateA_s{s}.json"
        )))
        .unwrap();
    }
    rows.push(ArmRow {
        arm: "gating_qualityreg",
        seed: "agg".into(),
        m: avg(&ga),
    });

    // gating B (InfoNCE on fused) — multiseed with frozen (lr, T)
    let mut gb: Vec<RankMetrics> = Vec::new();
    for &s in seeds {
        let (m, _l) = train_gate_b(&data, gb_lr, gb_t, s);
        let r = eval_gate_b(&m, &data, Split::Test);
        gb.push(r);
    }
    rows.push(ArmRow {
        arm: "gating_infnce",
        seed: "agg".into(),
        m: avg(&gb),
    });

    // QK untrained (random init, Q0) + trained — multiseed with frozen (lr, T)
    let q0 = QkModel::new(data.ds.input_dim, 999);
    rows.push(ArmRow {
        arm: "qk_untrained",
        seed: "agg".into(),
        m: eval_qk(&q0, &data, Split::Test),
    });
    let mut qk: Vec<RankMetrics> = Vec::new();
    for &s in seeds {
        let (m, curves) = train_qk(&data, qk_lr, qk_t, s);
        let r = eval_qk(&m, &data, Split::Test);
        qk.push(r);
        if s == seeds[0] {
            let mut csv = String::from("epoch,train_loss,val_r1\n");
            for c in &curves {
                let _ = writeln!(csv, "{},{:.6},{:.4}", c.epoch, c.train_loss, c.val_r1);
            }
            std::fs::write(format!("{out_dir}/trainlog_qk.csv"), &csv).unwrap();
        }
        save_qk_model(
            &m,
            &format!("{out_dir}/models/qk_s{s}.json"),
            s,
            qk_lr,
            qk_t,
        );
    }
    rows.push(ArmRow {
        arm: "qk_trained",
        seed: "agg".into(),
        m: avg(&qk),
    });

    // ---------- eval_test.csv ----------
    let mut csv = String::from("arm,seed,R@1,R@5,R@10,NDCG@10,MRR\n");
    for r in &rows {
        let _ = writeln!(
            csv,
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
            r.arm,
            r.seed,
            r.m.recall_at_1,
            r.m.recall_at_5,
            r.m.recall_at_10,
            r.m.ndcg_at_10,
            r.m.mrr
        );
    }
    std::fs::write(format!("{out_dir}/eval_test.csv"), &csv).unwrap();

    // ---------- per-seed variability summary ----------
    let mut variability = String::from("arm,metric,mean,std,min,max\n");
    for (name, ms) in [
        ("qk_trained", &qk),
        ("gating_qualityreg", &ga),
        ("gating_infnce", &gb),
    ] {
        for (metric, vals) in [
            ("R@1", ms.iter().map(|m| m.recall_at_1).collect::<Vec<_>>()),
            (
                "NDCG@10",
                ms.iter().map(|m| m.ndcg_at_10).collect::<Vec<_>>(),
            ),
            ("MRR", ms.iter().map(|m| m.mrr).collect::<Vec<_>>()),
        ] {
            let (mu, sd) = mean_std(&vals);
            let _ = writeln!(
                variability,
                "{name},{metric},{mu:.4},{sd:.4},{:.4},{:.4}",
                vals.iter().cloned().fold(f64::INFINITY, f64::min),
                vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            );
        }
    }
    std::fs::write(format!("{out_dir}/variability.csv"), &variability).unwrap();

    // ---------- gating weight diagnostics (selection vs ordering) ----------
    let mut wdiag = String::from("arm,mean_w0,mean_w1,mean_entropy\n");
    {
        let (mlp, _o) = train_gate_a(&data, seeds[0]);
        let t = fit_temperature_gate_a(&data, &mlp);
        let (mut s0, mut s1, mut ent, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for q in data.ds.queries.iter().filter(|q| q.split == Split::Test) {
            let lg = mlp.logits(&q.query);
            let w = softmax(&lg.iter().map(|x| x / t).collect::<Vec<_>>());
            s0 += w[0] as f64;
            s1 += w[1] as f64;
            ent += -(w[0].max(1e-9) as f64 * (w[0].max(1e-9) as f64).ln()
                + w[1].max(1e-9) as f64 * (w[1].max(1e-9) as f64).ln());
            n += 1.0;
        }
        let _ = writeln!(
            wdiag,
            "gating_qualityreg,{:.4},{:.4},{:.4}",
            s0 / n,
            s1 / n,
            ent / n
        );
    }
    {
        let (m, _l) = train_gate_b(&data, gb_lr, gb_t, seeds[0]);
        let (mut s0, mut s1, mut ent, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for q in data.ds.queries.iter().filter(|q| q.split == Split::Test) {
            let w = m.weights(&q.query);
            s0 += w[0] as f64;
            s1 += w[1] as f64;
            ent += -(w[0].max(1e-9) as f64 * (w[0].max(1e-9) as f64).ln()
                + w[1].max(1e-9) as f64 * (w[1].max(1e-9) as f64).ln());
            n += 1.0;
        }
        let _ = writeln!(
            wdiag,
            "gating_infnce,{:.4},{:.4},{:.4}",
            s0 / n,
            s1 / n,
            ent / n
        );
    }
    std::fs::write(format!("{out_dir}/gating_weights_test.csv"), &wdiag).unwrap();

    // ---------- config ----------
    let config = format!(
        "experiment=PH2C-QK-001\ndataset_seed={DATA_SEED:#x}\nn_heads={N_HEADS}\nhead_dim={DIM}\n\
         pool={POOL}\nn_train={N_TRAIN}\nn_val={N_VAL}\nn_test={N_TEST}\nnoise={NOISE}\n\
         distractor_cos={DISTRACTOR_COS}\ntop_k={TOP_K}\nqk:linear_no_hidden\n\
         frozen_qk_lr={qk_lr}\nfrozen_qk_T={qk_t}\nfrozen_gateB_lr={gb_lr}\nfrozen_gateB_T={gb_t}\n\
         max_epochs={MAX_EPOCHS}\npatience={PATIENCE}\nbatch={BATCH}\nl2=0\n\
         objective_qk=infonce(softmax_ce,1_positive)\nobjective_gateB=infonce_on_fused\n\
         objective_gateA=soft_target_recall(shipped)+val_temperature\nseeds={seeds:?}\n\
         selection=val_only(lr,T per arm at seed 42, then frozen)\n"
    );
    std::fs::write(format!("{out_dir}/config.txt"), &config).unwrap();

    // ---------- dataset.json + qk_content.json sidecar ----------
    let ds_json = serde_json::to_string(&data.ds).unwrap();
    std::fs::write(format!("{out_dir}/dataset.json"), &ds_json).unwrap();
    let mut cj = String::from("{\"format\":\"qk-content-sidecar\",\"v\":1,\"queries\":[");
    for (i, qc) in data.content.iter().enumerate() {
        if i > 0 {
            cj.push(',');
        }
        cj.push_str(&format!("{{\"q\":{:?},\"cands\":{:?}}}", qc.q, qc.cands));
    }
    cj.push_str("]}");
    std::fs::write(format!("{out_dir}/qk_content.json"), &cj).unwrap();

    // ---------- metrics.json ----------
    let summary = serde_json::json!({
        "experiment": "PH2C-QK-001",
        "dataset_hash": format!("{ds_hash:#x}"),
        "frozen_qk": {"lr": qk_lr, "T": qk_t, "val_r1": best_qk.2},
        "frozen_gating_infnce": {"lr": gb_lr, "T": gb_t, "val_r1": best_gb.2},
        "arms": rows.iter().map(|r| serde_json::json!({
            "arm": r.arm, "seed": r.seed,
            "R@1": format!("{:.4}", r.m.recall_at_1),
            "R@5": format!("{:.4}", r.m.recall_at_5),
            "R@10": format!("{:.4}", r.m.recall_at_10),
            "NDCG@10": format!("{:.4}", r.m.ndcg_at_10),
            "MRR": format!("{:.4}", r.m.mrr),
        })).collect::<Vec<_>>(),
    });
    std::fs::write(
        format!("{out_dir}/metrics.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .unwrap();

    let qa = avg(&qk);
    let ga_m = rows
        .iter()
        .find(|r| r.arm == "gating_qualityreg")
        .unwrap()
        .m;
    let gb_m = avg(&gb);
    let un = rows.iter().find(|r| r.arm == "uniform").unwrap().m;
    let orc = rows.iter().find(|r| r.arm == "oracle").unwrap().m;
    format!(
        "PH2C-QK-001 done (qk lr={qk_lr} T={qk_t} | gateB lr={gb_lr} T={gb_t})\n\
         test R@1: qk_trained={:.4}  gating_qualityreg={:.4}  gating_infnce={:.4}  \
         qk_untrained={:.4}  uniform={:.4}  oracle={:.4}",
        qa.recall_at_1,
        ga_m.recall_at_1,
        gb_m.recall_at_1,
        rows.iter()
            .find(|r| r.arm == "qk_untrained")
            .unwrap()
            .m
            .recall_at_1,
        un.recall_at_1,
        orc.recall_at_1
    )
}

fn save_qk_model(m: &QkModel, path: &str, seed: u64, lr: f32, t: f32) {
    let j = serde_json::json!({
        "model_id": path.rsplit('/').next().unwrap_or(path),
        "kind": "linear_qk_wq_wk",
        "note": "Q=W_Q.q, K=W_K.x, score=Q.K/sqrt(8); PH2C-QK-001 sanity",
        "seed": seed, "lr": lr, "T": t,
        "wq": m.wq, "wk": m.wk,
    });
    std::fs::write(path, serde_json::to_string_pretty(&j).unwrap()).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_geometry_anti_cosine() {
        let d = generate(DATA_SEED);
        // (a) relevant is NEVER top under the signal-head cosine
        for q in &d.ds.queries {
            let raw = &q.heads[0].raw_scores;
            let rel = raw[0];
            assert!(
                raw.iter().skip(1).all(|&s| s > rel),
                "relevant must rank last on head-0 cosine"
            );
        }
        // (b) family separation: cos(q, u_f) high, cos(relevant, q) ~ 0
        let mut n_rel = 0.0f64;
        let mut n_dis = 0.0f64;
        let mut cnt = 0.0f64;
        for qc in d.content.iter().step_by(97) {
            let (q0, r0, d0) = (&qc.q[..8], &qc.cands[0][..8], &qc.cands[1][..8]);
            n_rel += q0.iter().zip(r0).map(|(a, b)| (a * b) as f64).sum::<f64>();
            n_dis += q0.iter().zip(d0).map(|(a, b)| (a * b) as f64).sum::<f64>();
            cnt += 1.0;
        }
        assert!(
            n_rel / cnt < 0.15,
            "relevant must be near-orthogonal to query"
        );
        assert!(n_dis / cnt > 0.25, "distractors must be mildly attractive");
    }

    #[test]
    fn sanity_head_quality_targets_reflect_true_head_ranking() {
        // HC-5 regression: head quality metrics must come from the SORTED
        // head ranking. Signal head: relevant always last -> recall@1 == 0.
        let d = generate(DATA_SEED);
        for q in &d.ds.queries {
            assert_eq!(q.heads[0].recall_at_k, 0.0, "head-0 recall@1 must be 0");
        }
    }

    #[test]
    fn sanity_qk_learns_gating_cannot() {
        let d = generate(DATA_SEED);
        // QK: full grid entry (lr=0.05, T=0.25) must reach ~1.0 val R@1.
        let (m, _c) = train_qk(&d, 0.05, 0.25, 42);
        let val: Vec<usize> = (0..d.ds.queries.len())
            .filter(|&i| d.ds.queries[i].split == Split::Val)
            .collect();
        assert!(
            eval_qk_r1(&m, &d, &val) > 0.9,
            "QK must learn the sanity set"
        );
        // Untrained QK stays at chance.
        let q0 = QkModel::new(d.ds.input_dim, 999);
        assert!(
            eval_qk_r1(&q0, &d, &val) < 0.35,
            "untrained QK must be ~chance"
        );
    }
}
