//! PH2C-QK-002 — real-corpus trained candidate-level QK vs trained gating
//! (Phase 2C main evaluation, experiment spec §2–§18).
//!
//! Paired protocol on the CACHED Phase 2B datasets (§3): every arm sees the
//! identical per-head candidate pools (§6), identical splits, identical GT.
//! QK content comes from the `qk-cache` sidecar (doc content matrix +
//! per-head candidate doc indices aligned with cached pool order). Training
//! and evaluation never touch HNSW (§4 discipline, same as Phase 2B).
//!
//! Arms (§5): uniform / global-best / trained gating (shipped Phase 2B
//! protocol) / trained linear QK (PH2C-QK-001-validated: Q=W_Q·q, K=W_K·x,
//! score=Q·K/√d, multi-positive InfoNCE at temperature T) / gating+QK
//! (RRF k=60 of the two rankings — no new learned machinery) / RRF k=60 /
//! oracle. Plus candidate budgets (§9), per-group breakdown (§10),
//! multiseed (§11), QK sanity checks incl. ranking movement (§14),
//! exact-rerank diagnostics (§15), latency + params (§16–§18).

use attentiondb_learned::eval::{
    fuse_rrf, fuse_weighted, global_best_head, rank_metrics, uniform_weights, RankMetrics,
};
use attentiondb_learned::gating_v2::{
    train_gating, DetRng, GatingDataset, GatingMlp, ModelCard, Objective, QueryExample, Split,
    TrainOutcome, TrainingConfig, TrainingMeta,
};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::rc::Rc;

const DQ: usize = 8;
const BATCH: usize = 32;
const MAX_EPOCHS: usize = 200;
const PATIENCE: usize = 20;
const LR_GRID: [f32; 2] = [0.05, 0.01];
const T_GRID: [f32; 4] = [0.25, 0.5, 1.0, 2.0];
const BUDGETS: [usize; 4] = [10, 25, 50, 100];
const RRF_K: f32 = 60.0;
const RERANK_TOP: usize = 50;

// ------------------------------------------------------------------ views
#[derive(Deserialize)]
struct Sidecar {
    docs: Vec<Vec<f32>>,
    docidx_to_engine: Vec<u64>,
    queries: Vec<Vec<Vec<u32>>>,
}

#[derive(Clone)]
struct Docs {
    rows: Rc<Vec<Vec<f32>>>,
    idx2eng: Rc<Vec<u64>>,
    eng2idx: Rc<HashMap<u64, usize>>,
}

#[derive(Clone)]
struct View {
    ds: GatingDataset,
    unions: Vec<Vec<usize>>,
    docs: Docs,
}

impl View {
    fn full(ds: GatingDataset, sc: Sidecar) -> View {
        let eng2idx: HashMap<u64, usize> = sc
            .docidx_to_engine
            .iter()
            .enumerate()
            .map(|(i, &e)| (e, i))
            .collect();
        let unions: Vec<Vec<usize>> = sc
            .queries
            .iter()
            .map(|heads| {
                let mut seen = HashSet::new();
                let mut u = Vec::new();
                for hs in heads {
                    for &c in hs {
                        if seen.insert(c) {
                            u.push(c as usize);
                        }
                    }
                }
                u
            })
            .collect();
        View {
            ds,
            unions,
            docs: Docs {
                rows: Rc::new(sc.docs),
                idx2eng: Rc::new(sc.docidx_to_engine),
                eng2idx: Rc::new(eng2idx),
            },
        }
    }

    /// Truncate every head pool to its top-K (pools are engine-rank ordered —
    /// validated by validate_datasets.py), recompute unions. Models are
    /// reused unchanged: the budget varies candidate generation only (§9).
    fn truncated(&self, k: usize) -> View {
        let mut ds = self.ds.clone();
        let mut unions = Vec::with_capacity(ds.queries.len());
        for q in ds.queries.iter_mut() {
            let mut seen = HashSet::new();
            let mut u = Vec::new();
            for h in q.heads.iter_mut() {
                let n = k.min(h.candidates.len());
                h.candidates.truncate(n);
                h.raw_scores.truncate(n);
                h.norm_scores.truncate(n);
                h.exact_scores.truncate(n);
                for &c in &h.candidates {
                    if seen.insert(c) {
                        u.push(self.docs.eng2idx[&c]);
                    }
                }
            }
            unions.push(u);
        }
        View {
            ds,
            unions,
            docs: self.docs.clone(),
        }
    }

    fn gt_positions(&self, qi: usize) -> Vec<usize> {
        self.ds.queries[qi]
            .ground_truth
            .iter()
            .filter_map(|&e| self.docs.eng2idx.get(&e).copied())
            .collect()
    }
}

/// rank_metrics consumes slice ORDER as the ranking (HC-5): always sort by
/// score desc, id asc, through this helper.
fn rank_sorted(ranked: &[(u64, f32)], gt: &[u64], k: usize) -> RankMetrics {
    let mut v = ranked.to_vec();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    rank_metrics(&v, gt, k)
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

fn test_queries(v: &View) -> Vec<usize> {
    (0..v.ds.queries.len())
        .filter(|&i| v.ds.queries[i].split == Split::Test)
        .collect()
}

// --------------------------------------------------------------- QK model
#[derive(Clone)]
struct QkModel {
    wq: Vec<f32>,
    wk: Vec<f32>,
    din: usize,
}

impl QkModel {
    fn new(din: usize, seed: u64) -> Self {
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
                    .sum::<f32>()
            })
            .collect()
    }
    fn k_dot(&self, d: usize, docs: &Docs) -> [f32; DQ] {
        let x = &docs.rows[d];
        let mut out = [0.0f32; DQ];
        for (i, o) in out.iter_mut().enumerate() {
            *o = x
                .iter()
                .zip(self.wk[i * self.din..(i + 1) * self.din].iter())
                .map(|(a, b)| a * b)
                .sum();
        }
        out
    }
    /// QK scores over a union (doc indices).
    fn scores(&self, qp: &[f32], union: &[usize], docs: &Docs) -> Vec<f32> {
        let scale = (DQ as f32).sqrt();
        union
            .iter()
            .map(|&d| {
                let k = self.k_dot(d, docs);
                qp.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / scale
            })
            .collect()
    }
    fn snapshot(&self) -> Vec<f32> {
        let mut v = self.wq.clone();
        v.extend_from_slice(&self.wk);
        v
    }
    fn restore(&mut self, s: &[f32]) {
        let n = DQ * self.din;
        self.wq.copy_from_slice(&s[..n]);
        self.wk.copy_from_slice(&s[n..]);
    }
}

// ----------------------------------------------------------------- Adam
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
            w[i] -= self.lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}

/// Multi-positive InfoNCE at temperature T over the union:
/// p = softmax(s/T), loss = −ln(Σ_{i∈P} p_i); dL/ds_j = p_j − δ_{j∈P}·p_j/Z.
/// Reduces to the PH2C-QK-001 single-positive InfoNCE when |P| = 1.
fn infonce(scores: &[f32], pos: &HashSet<usize>, t: f32) -> (f32, Vec<f32>) {
    let logits: Vec<f32> = scores.iter().map(|s| s / t).collect();
    let p = softmax(&logits);
    let z: f32 = pos.iter().map(|&i| p[i]).sum::<f32>().max(1e-12);
    let grad = p
        .iter()
        .enumerate()
        .map(|(j, &pj)| pj - if pos.contains(&j) { pj / z } else { 0.0 })
        .collect();
    (-z.ln(), grad)
}

struct QkTrainOut {
    model: QkModel,
    curves: Vec<(usize, f32, f64)>,
    param_delta: f32,
    init: Vec<f32>,
}

fn train_qk(v: &View, lr: f32, t: f32, seed: u64) -> QkTrainOut {
    let din = v.ds.input_dim;
    let mut model = QkModel::new(din, seed);
    let init = model.snapshot();
    let mut opt_wq = Adam::new(model.wq.len(), lr);
    let mut opt_wk = Adam::new(model.wk.len(), lr);
    let mut rng = DetRng::new(seed ^ 0x7D2B);
    let train: Vec<usize> = (0..v.ds.queries.len())
        .filter(|&i| v.ds.queries[i].split == Split::Train)
        .collect();
    let val: Vec<usize> = (0..v.ds.queries.len())
        .filter(|&i| v.ds.queries[i].split == Split::Val)
        .collect();
    let mut curves = Vec::new();
    let mut best_r10 = -1.0f64;
    let mut best_snap = model.snapshot();
    let mut stall = 0usize;
    let scale = (DQ as f32).sqrt();

    for epoch in 0..MAX_EPOCHS {
        let mut order = train.clone();
        rng.shuffle(&mut order);
        let mut loss_sum = 0.0f32;
        let (mut gwq, mut gwk) = (vec![0.0f32; model.wq.len()], vec![0.0f32; model.wk.len()]);
        let mut in_batch = 0usize;
        for &qi in &order {
            let q = &v.ds.queries[qi];
            let qp = model.q_proj(&q.query);
            let scores = model.scores(&qp, &v.unions[qi], &v.docs);
            // GT doc indices → UNION POSITIONS (GT docs absent from the
            // union are skipped; multiview gt-frac = 0.9975 so this is rare)
            let union = &v.unions[qi];
            let pos: HashSet<usize> = v
                .gt_positions(qi)
                .iter()
                .filter_map(|g| union.iter().position(|&d| d == *g))
                .collect();
            if pos.is_empty() {
                continue;
            }
            let (loss, dsc) = infonce(&scores, &pos, t);
            loss_sum += loss;
            let mut dqp = [0.0f32; DQ];
            for (u, &d) in union.iter().enumerate() {
                let x = &v.docs.rows[d];
                for di in 0..DQ {
                    let kdot = x
                        .iter()
                        .zip(model.wk[di * din..(di + 1) * din].iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                    dqp[di] += dsc[u] * kdot / scale;
                    let g = dsc[u] * qp[di] / scale;
                    let row = &mut gwk[di * din..(di + 1) * din];
                    for (gr, xv) in row.iter_mut().zip(x.iter()) {
                        *gr += g * xv;
                    }
                }
            }
            for di in 0..DQ {
                let row = &mut gwq[di * din..(di + 1) * din];
                for (gr, qv) in row.iter_mut().zip(q.query.iter()) {
                    *gr += dqp[di] * qv;
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
        let val_r10 = eval_qk_idx(&model, v, &val).recall_at_10;
        curves.push((epoch, loss_sum / train.len() as f32, val_r10));
        if val_r10 > best_r10 + 1e-6 {
            best_r10 = val_r10;
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
    let param_delta = init
        .iter()
        .zip(model.snapshot().iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    QkTrainOut {
        model,
        curves,
        param_delta,
        init,
    }
}

fn eval_qk_idx(m: &QkModel, v: &View, idx: &[usize]) -> RankMetrics {
    let rs: Vec<RankMetrics> = idx
        .iter()
        .map(|&qi| {
            let q = &v.ds.queries[qi];
            let qp = m.q_proj(&q.query);
            let s = m.scores(&qp, &v.unions[qi], &v.docs);
            let ranked: Vec<(u64, f32)> = v.unions[qi]
                .iter()
                .map(|&d| v.docs.idx2eng[d])
                .zip(s.iter().copied())
                .collect();
            rank_sorted(&ranked, &q.ground_truth, v.ds.top_k)
        })
        .collect();
    avg(&rs)
}

fn eval_qk(m: &QkModel, v: &View, split: Split) -> RankMetrics {
    let idx: Vec<usize> = (0..v.ds.queries.len())
        .filter(|&i| v.ds.queries[i].split == split)
        .collect();
    eval_qk_idx(m, v, &idx)
}

// ------------------------------------------------------------- fusion arms
fn eval_weighted<F: Fn(&QueryExample) -> Vec<f32>>(v: &View, w: &F, split: Split) -> RankMetrics {
    let rs: Vec<RankMetrics> =
        v.ds.queries
            .iter()
            .filter(|q| q.split == split)
            .map(|q| rank_sorted(&fuse_weighted(q, &w(q)), &q.ground_truth, v.ds.top_k))
            .collect();
    avg(&rs)
}

fn eval_rrf(v: &View) -> RankMetrics {
    let rs: Vec<RankMetrics> =
        v.ds.queries
            .iter()
            .filter(|q| q.split == Split::Test)
            .map(|q| rank_sorted(&fuse_rrf(q, RRF_K), &q.ground_truth, v.ds.top_k))
            .collect();
    avg(&rs)
}

fn oracle_weighted(v: &View) -> RankMetrics {
    let rs: Vec<RankMetrics> =
        v.ds.queries
            .iter()
            .filter(|q| q.split == Split::Test)
            .map(|q| {
                let mut best = (0usize, -1.0f32);
                for (h, he) in q.heads.iter().enumerate() {
                    if he.recall_at_k > best.1 {
                        best = (h, he.recall_at_k);
                    }
                }
                let mut w = vec![0.0f32; q.heads.len()];
                w[best.0] = 1.0;
                rank_sorted(&fuse_weighted(q, &w), &q.ground_truth, v.ds.top_k)
            })
            .collect();
    avg(&rs)
}

fn rrf_combine(a: &[(u64, f32)], b: &[(u64, f32)]) -> Vec<(u64, f32)> {
    let ranks = |v: &[(u64, f32)]| -> HashMap<u64, usize> {
        let mut s = v.to_vec();
        s.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
        s.iter()
            .enumerate()
            .map(|(i, (id, _))| (*id, i + 1))
            .collect()
    };
    let (ra, rb) = (ranks(a), ranks(b));
    let ids: HashSet<u64> = ra.keys().chain(rb.keys()).copied().collect();
    let mut out: Vec<(u64, f32)> = ids
        .into_iter()
        .map(|id| {
            let da = *ra.get(&id).unwrap_or(&(ra.len() + 1)) as f32;
            let db = *rb.get(&id).unwrap_or(&(rb.len() + 1)) as f32;
            (id, 1.0 / (RRF_K + da) + 1.0 / (RRF_K + db))
        })
        .collect();
    out.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
    out
}

fn eval_gqk(v: &View, g: &GatingMlp, t: f32, qm: &QkModel) -> RankMetrics {
    let rs: Vec<RankMetrics> =
        v.ds.queries
            .iter()
            .filter(|q| q.split == Split::Test)
            .map(|q| {
                let lg = g.logits(&q.query);
                let w = softmax(&lg.iter().map(|x| x / t).collect::<Vec<_>>());
                let gr = fuse_weighted(q, &w);
                let qp = qm.q_proj(&q.query);
                let s = qm.scores(&qp, &v.unions[qi_of(q)], &v.docs);
                let qr: Vec<(u64, f32)> = v.unions[qi_of(q)]
                    .iter()
                    .map(|&d| v.docs.idx2eng[d])
                    .zip(s.iter().copied())
                    .collect();
                rank_sorted(&rrf_combine(&gr, &qr), &q.ground_truth, v.ds.top_k)
            })
            .collect();
    avg(&rs)
}

fn qi_of(q: &QueryExample) -> usize {
    q.query_id as usize
}

// ------------------------------------------------------------- gating arm
fn gating_weights(m: &GatingMlp, q: &QueryExample, t: f32) -> Vec<f32> {
    let lg = m.logits(&q.query);
    softmax(&lg.iter().map(|x| x / t).collect::<Vec<_>>())
}

fn eval_gating(m: &GatingMlp, v: &View, t: f32, split: Split) -> RankMetrics {
    eval_weighted(v, &|q| gating_weights(m, q, t), split)
}

fn fit_temperature(m: &GatingMlp, v: &View) -> f32 {
    let mut best = (T_GRID[0], -1.0f64);
    for &t in T_GRID.iter() {
        let r = eval_gating(m, v, t, Split::Val).recall_at_10;
        if r > best.1 {
            best = (t, r);
        }
    }
    best.0
}

fn train_gating_all_objectives(
    v: &View,
    seed: u64,
) -> Vec<(Objective, GatingMlp, f32, f32, usize, TrainOutcome)> {
    let lrs = [0.01f32, 0.003];
    let hiddens = [32usize, 64];
    let mut out = Vec::new();
    for &obj in [
        Objective::QualityRegression,
        Objective::SoftTarget,
        Objective::Pairwise,
    ]
    .iter()
    {
        for &lr in lrs.iter() {
            for &hidden in hiddens.iter() {
                let cfg = TrainingConfig {
                    objective: obj,
                    max_epochs: 300,
                    patience: 15,
                    lr,
                    hidden,
                    seed,
                    ..TrainingConfig::default()
                };
                let tr = train_gating(
                    &v.ds,
                    &cfg,
                    attentiondb_learned::gating_v2::QualityTarget::Recall,
                );
                let t = fit_temperature(&tr.model, v);
                out.push((obj, tr.model.clone(), t, lr, hidden, tr));
            }
        }
    }
    out
}

// ------------------------------------------------------- exact rerank arms
/// exact-weighted fusion: s(c) = Σ_{h: c∈pool_h} w_h·exact_h(c).
fn exact_weighted(v: &View, wf: &dyn Fn(&QueryExample) -> Vec<f32>) -> RankMetrics {
    let rs: Vec<RankMetrics> =
        v.ds.queries
            .iter()
            .filter(|q| q.split == Split::Test)
            .map(|q| {
                let w = wf(q);
                let mut acc: HashMap<u64, f32> = HashMap::new();
                for (h, he) in q.heads.iter().enumerate() {
                    for (i, &c) in he.candidates.iter().enumerate() {
                        *acc.entry(c).or_insert(0.0) += w[h] * he.exact_scores[i];
                    }
                }
                let ranked: Vec<(u64, f32)> = acc.into_iter().collect();
                rank_sorted(&ranked, &q.ground_truth, v.ds.top_k)
            })
            .collect();
    avg(&rs)
}

// -------------------------------------------------------------- percentile
fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((p / 100.0) * (s.len() as f64 - 1.0)).round() as usize;
    s[i.min(s.len() - 1)]
}

fn kendall_tau(a: &[(u64, f32)], b: &[(u64, f32)]) -> f64 {
    let rk = |v: &[(u64, f32)]| -> HashMap<u64, usize> {
        let mut s = v.to_vec();
        s.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
        s.iter().enumerate().map(|(i, (id, _))| (*id, i)).collect()
    };
    let (ra, rb) = (rk(a), rk(b));
    let ids: Vec<u64> = ra.keys().copied().collect();
    let n = ids.len();
    if n < 2 {
        return 0.0;
    }
    let (mut conc, mut disc) = (0.0f64, 0.0f64);
    for i in 0..n {
        for j in (i + 1)..n {
            let (a1, a2) = (ra[&ids[i]], ra[&ids[j]]);
            let (b1, b2) = (rb[&ids[i]], rb[&ids[j]]);
            if (a1 < a2) == (b1 < b2) {
                conc += 1.0;
            } else {
                disc += 1.0;
            }
        }
    }
    (conc - disc) / (conc + disc)
}

// ------------------------------------------------------------- entry point
pub fn run(
    corpus_name: &str,
    dataset_path: &str,
    content_path: &str,
    out_dir: &str,
    seeds: &[u64],
) -> String {
    std::fs::create_dir_all(format!("{out_dir}/models")).unwrap();
    let t_start = std::time::Instant::now();
    let ds = GatingDataset::parse(&std::fs::read_to_string(dataset_path).unwrap()).unwrap();
    let sc: Sidecar =
        serde_json::from_str(&std::fs::read_to_string(content_path).unwrap()).unwrap();
    let v = View::full(ds, sc);
    let ds_hash = v.ds.content_hash();
    let mut report = String::new();

    // ---------- §12 QK validation grid (seed 42 only) ----------
    let mut grid_csv = String::from("lr,T,val_r10\n");
    let mut best = (LR_GRID[0], T_GRID[0], -1.0f64, None::<QkTrainOut>);
    for &lr in LR_GRID.iter() {
        for &t in T_GRID.iter() {
            let out = train_qk(&v, lr, t, 42);
            let _ = writeln!(
                grid_csv,
                "{lr},{t},{:.4}",
                out.curves.last().map(|c| c.2).unwrap_or(0.0)
            );
            let r = out.curves.iter().map(|c| c.2).fold(-1.0f64, f64::max);
            if r > best.2 {
                best = (lr, t, r, Some(out));
            }
        }
    }
    std::fs::write(format!("{out_dir}/val_grid_qk.csv"), &grid_csv).unwrap();
    let (qk_lr, qk_t) = (best.0, best.1);
    let qk42_box = best.3.unwrap();
    let qk42 = qk42_box.model.clone();
    let _ = writeln!(
        report,
        "frozen QK: lr={qk_lr} T={qk_t} (best val R@10={:.4})",
        best.2
    );

    // ---------- gating: shipped protocol (3 objectives, val-select) ----------
    let g_all = train_gating_all_objectives(&v, 42);
    let (sel_obj, g42, sel_t, sel_lr, sel_hidden, _g42_out) = g_all
        .iter()
        .max_by(|a, b| {
            eval_gating(&a.1, &v, a.2, Split::Val)
                .recall_at_10
                .total_cmp(&eval_gating(&b.1, &v, b.2, Split::Val).recall_at_10)
        })
        .map(|(o, m, t, lr, hidden, out)| (*o, m.clone(), *t, lr, hidden, out.clone()))
        .unwrap();
    let _ = writeln!(report, "frozen gating: objective={sel_obj:?} T={sel_t} lr={sel_lr} hidden={sel_hidden} (val-selected)");

    // ---------- §11 multiseed ----------
    let mut per_seed: Vec<(String, u64, RankMetrics)> = Vec::new();
    for (si, &seed) in seeds.iter().enumerate() {
        let cands = if si == 0 {
            g_all.clone()
        } else {
            train_gating_all_objectives(&v, seed)
        };
        let (obj, gm, gt_fit, _lr, _hid, _o) = cands
            .iter()
            .max_by(|a, b| {
                eval_gating(&a.1, &v, a.2, Split::Val)
                    .recall_at_10
                    .total_cmp(&eval_gating(&b.1, &v, b.2, Split::Val).recall_at_10)
            })
            .map(|(o, m, t, lr, hidden, out)| (*o, m.clone(), *t, lr, hidden, out.clone()))
            .unwrap();
        let qout = if si == 0 {
            QkTrainOut {
                model: qk42.clone(),
                curves: qk42_box.curves.clone(),
                param_delta: qk42_box.param_delta,
                init: qk42_box.init.clone(),
            }
        } else {
            train_qk(&v, qk_lr, qk_t, seed)
        };
        let g_res = eval_gating(&gm, &v, gt_fit, Split::Test);
        let q_res = eval_qk(&qout.model, &v, Split::Test);
        let gq_res = eval_gqk(&v, &gm, gt_fit, &qout.model);
        per_seed.push((format!("gating({obj:?})"), seed, g_res));
        per_seed.push(("qk".into(), seed, q_res));
        per_seed.push(("gating_qk".into(), seed, gq_res));
        if si == 0 {
            let mut csv = String::from("epoch,train_loss,val_r10\n");
            for (e, l, r) in &qout.curves {
                let _ = writeln!(csv, "{e},{l:.6},{r:.4}");
            }
            let _ = writeln!(
                csv,
                "param_delta_l2,{:.6},{}",
                qout.param_delta, q_res.recall_at_10
            );
            std::fs::write(format!("{out_dir}/trainlog_qk.csv"), &csv).unwrap();
            // model cards
            let meta = TrainingMeta {
                seed,
                dataset_hash: ds_hash,
                objective: format!("{obj:?}+valT={gt_fit}"),
                learning_rate: 0.01,
                batch_size: BATCH,
                epochs_run: _o.curves.len(),
                best_val_loss: _o.best_val_loss,
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
            let card =
                ModelCard::from_mlp(&gm, meta, &format!("qk2-gating-s{seed}"), "gating_main");
            card.save(std::path::Path::new(&format!(
                "{out_dir}/models/gating_s{seed}.json"
            )))
            .unwrap();
            let qj = serde_json::json!({"model_id": format!("qk2-qk-s{seed}"), "kind": "linear_qk",
                "lr": qk_lr, "T": qk_t, "seed": seed, "wq": qout.model.wq, "wk": qout.model.wk});
            std::fs::write(
                format!("{out_dir}/models/qk_s{seed}.json"),
                serde_json::to_string_pretty(&qj).unwrap(),
            )
            .unwrap();
        }
    }

    // ---------- results.csv (agg + per-seed) ----------
    let mut rows: Vec<(String, String, RankMetrics)> = Vec::new();
    // fixed reference arms (deterministic)
    rows.push((
        "uniform".into(),
        "agg".into(),
        eval_weighted(&v, &|_q| uniform_weights(v.ds.num_heads), Split::Test),
    ));
    let (gb_head, _) = global_best_head(&v.ds, Split::Train);
    {
        let mut w = vec![0.0f32; v.ds.num_heads];
        w[gb_head] = 1.0;
        rows.push((
            "global_best".into(),
            "agg".into(),
            eval_weighted(&v, &move |_q| w.clone(), Split::Test),
        ));
    }
    rows.push(("rrf_k60".into(), "agg".into(), eval_rrf(&v)));
    // Oracle (§5-G, analysis-only): per-query all-weight on the head with the
    // best CACHED head recall vs GT — information no deployable model gets.
    rows.push(("oracle".into(), "agg".into(), oracle_weighted(&v)));
    for arm in ["gating", "qk", "gating_qk"] {
        let ms: Vec<RankMetrics> = per_seed
            .iter()
            .filter(|(a, _, _)| a.split('(').next() == Some(arm))
            .map(|(_, _, m)| *m)
            .collect();
        assert_eq!(
            ms.len(),
            seeds.len(),
            "agg filter must match exactly the per-seed rows of {arm}"
        );
        rows.push((arm.to_string(), "agg".into(), avg(&ms)));
    }
    for (a, s, m) in &per_seed {
        rows.push((a.split('(').next().unwrap().to_string(), format!("{s}"), *m));
    }
    let mut csv = String::from("arm,seed,R@1,R@5,R@10,NDCG@10,MRR\n");
    for (arm, seed, m) in &rows {
        let _ = writeln!(
            csv,
            "{arm},{seed},{:.4},{:.4},{:.4},{:.4},{:.4}",
            m.recall_at_1, m.recall_at_5, m.recall_at_10, m.ndcg_at_10, m.mrr
        );
    }
    std::fs::write(format!("{out_dir}/results.csv"), &csv).unwrap();

    // ---------- candidate recall (§8) ----------
    let cr = |vv: &View| -> (f64, f64) {
        let n = vv.ds.queries.len() as f64;
        let (sum, ge1) = (0..vv.ds.queries.len())
            .map(|qi| {
                let gt = vv.gt_positions(qi);
                let hits = gt.iter().filter(|g| vv.unions[qi].contains(g)).count();
                (
                    hits as f64 / gt.len().max(1) as f64,
                    (hits > 0) as u8 as f64,
                )
            })
            .fold((0.0, 0.0), |a, x| (a.0 + x.0, a.1 + x.1));
        (sum / n, 100.0 * ge1 / n)
    };
    let (crf, crp) = cr(&v);
    std::fs::write(
        format!("{out_dir}/candidate_recall.csv"),
        format!("pool,K,candidate_recall_gt_frac,pct_queries_ge1\nunion,100,{crf:.4},{crp:.2}\n"),
    )
    .unwrap();
    let _ = writeln!(
        report,
        "candidate recall (full pool): gt-frac={crf:.4}, queries≥1={crp:.1}%"
    );

    // ---------- §9 budgets (frozen seed-42 models) ----------
    let mut budget_csv =
        String::from("K,cand_recall_gt_frac,pct_ge1,uniform_r10,gating_r10,qk_r10,gating_qk_r10\n");
    for &k in BUDGETS.iter() {
        let tv = v.truncated(k);
        let (f, g1) = cr(&tv);
        let un =
            eval_weighted(&tv, &|_q| uniform_weights(tv.ds.num_heads), Split::Test).recall_at_10;
        let ga = eval_gating(&g42, &tv, sel_t, Split::Test).recall_at_10;
        let qk = eval_qk(&qk42, &tv, Split::Test).recall_at_10;
        let gq = eval_gqk(&tv, &g42, sel_t, &qk42).recall_at_10;
        let _ = writeln!(
            budget_csv,
            "{k},{f:.4},{g1:.1},{un:.4},{ga:.4},{qk:.4},{gq:.4}"
        );
    }
    std::fs::write(format!("{out_dir}/budgets.csv"), &budget_csv).unwrap();

    // ---------- §10 per-group (seed-42 models, test split) ----------
    let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
    for &qi in test_queries(&v).iter() {
        if let Some(g) = v.ds.queries[qi].query_group {
            groups.entry(g).or_default().push(qi);
        }
    }
    let mut grp_csv = String::from("group,n,uniform_r10,gating_r10,qk_r10,gating_qk_r10,oracle_r10,gating_mean_w,qk_mean_topgap\n");
    for (g, mut idxs) in groups {
        idxs.sort();
        let m = |r: RankMetrics| r.recall_at_10;
        let un = m(eval_qk_arm_generic(&v, &idxs, ArmKind::Uniform));
        let ga = m(eval_qk_arm_generic(&v, &idxs, ArmKind::Gating(&g42, sel_t)));
        let qk = m(eval_qk_arm_generic(&v, &idxs, ArmKind::Qk(&qk42)));
        let gq = m(eval_qk_arm_generic(
            &v,
            &idxs,
            ArmKind::Gqk(&g42, sel_t, &qk42),
        ));
        let orc = m(eval_qk_arm_generic(&v, &idxs, ArmKind::OraclePq));
        let (mw, gap) = diagnostics_for(&v, &idxs, &g42, sel_t, &qk42);
        let _ = writeln!(
            grp_csv,
            "{g},{},{un:.4},{ga:.4},{qk:.4},{gq:.4},{orc:.4},{mw:.4},{gap:.4}",
            idxs.len()
        );
    }
    std::fs::write(format!("{out_dir}/by_query_type.csv"), &grp_csv).unwrap();

    // ---------- §14 QK sanity checks ----------
    let q0 = QkModel::new(v.ds.input_dim, 999);
    let untrained = eval_qk(&q0, &v, Split::Test).recall_at_10;
    let tq = test_queries(&v);
    let (score_std_mean, score_std_min) = {
        let mut stds = Vec::new();
        for &qi in &tq {
            let qp = qk42.q_proj(&v.ds.queries[qi].query);
            let s = qk42.scores(&qp, &v.unions[qi], &v.docs);
            let mu = s.iter().sum::<f32>() / s.len() as f32;
            stds.push((s.iter().map(|x| (x - mu).powi(2)).sum::<f32>() / s.len() as f32).sqrt());
        }
        (
            stds.iter().sum::<f32>() / stds.len() as f32,
            stds.iter().cloned().fold(f32::INFINITY, f32::min),
        )
    };
    let tau = {
        let sample: Vec<usize> = tq
            .iter()
            .step_by((tq.len() / 100).max(1))
            .copied()
            .collect();
        let mut acc = 0.0f64;
        for &qi in &sample {
            let q = &v.ds.queries[qi];
            let base = fuse_weighted(q, &uniform_weights(v.ds.num_heads));
            let qp = qk42.q_proj(&q.query);
            let s = qk42.scores(&qp, &v.unions[qi], &v.docs);
            let qkr: Vec<(u64, f32)> = v.unions[qi]
                .iter()
                .map(|&d| v.docs.idx2eng[d])
                .zip(s.iter().copied())
                .collect();
            acc += kendall_tau(&base, &qkr);
        }
        acc / sample.len() as f64
    };
    std::fs::write(
        format!("{out_dir}/qk_sanity_checks.csv"),
        format!("check,value\nuntrained_qk_r10,{untrained:.4}\nparam_delta_l2,{:.4}\nscore_std_mean,{score_std_mean:.6}\nscore_std_min,{score_std_min:.6}\nkendall_tau_qk_vs_uniform,{tau:.4}\n", qk42_box.param_delta),
    ).unwrap();
    let _ = writeln!(report, "QK sanity: untrained R@10={untrained:.4}, ‖ΔW‖={:.3}, score-std mean={score_std_mean:.4} min={score_std_min:.4}, τ(qk,uniform)={tau:.4}", qk42_box.param_delta);

    // ---------- §15 exact-rerank diagnostics (seed-42 models) ----------
    let gm42 = g42.clone();
    let diag = |name: &str, r: RankMetrics, lat: f64| {
        format!(
            "{name},{:.4},{:.4},{:.4},{lat:.2}",
            r.recall_at_10, r.ndcg_at_10, r.mrr
        )
    };
    let mut rd = String::from("config,R@10,NDCG@10,MRR,p50_us\n");
    // exact-weighted configs (weights defined per config; formulas in methodology)
    let t_diag = std::time::Instant::now();
    let _ = writeln!(
        rd,
        "{}",
        diag(
            "exact_unif",
            exact_weighted(&v, &|_q| uniform_weights(v.ds.num_heads)),
            0.0
        )
    );
    let unif_us = t_diag.elapsed().as_secs_f64() * 1e6;
    {
        let mut w = vec![0.0f32; v.ds.num_heads];
        w[gb_head] = 1.0;
        let _ = writeln!(
            rd,
            "{}",
            diag("exact_gbest", exact_weighted(&v, &move |_q| w.clone()), 0.0)
        );
    }
    let _ = writeln!(
        rd,
        "{}",
        diag(
            "exact_gating",
            exact_weighted(&v, &|q| gating_weights(&gm42, q, sel_t)),
            0.0
        )
    );
    let _ = writeln!(
        rd,
        "{}",
        diag(
            "exact_top1",
            exact_weighted(&v, &|q| {
                let w = gating_weights(&gm42, q, sel_t);
                let (bi, _) = w
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap();
                let mut x = vec![0.0f32; w.len()];
                x[bi] = 1.0;
                x
            }),
            0.0
        )
    );
    let _ = writeln!(
        rd,
        "{}",
        diag(
            "exact_top2",
            exact_weighted(&v, &|q| {
                let mut w = gating_weights(&gm42, q, sel_t);
                let mut order: Vec<usize> = (0..w.len()).collect();
                order.sort_by(|a, b| w[*b].total_cmp(&w[*a]));
                for i in 2..w.len() {
                    w[order[i]] = 0.0;
                }
                let s: f32 = w.iter().sum();
                w.iter_mut().for_each(|x| *x /= s);
                w
            }),
            0.0
        )
    );
    let qk_main_row = rows
        .iter()
        .find(|(a, s, _)| a == "qk" && s == "agg")
        .unwrap()
        .2;
    let _ = writeln!(rd, "{}", diag("qk_no_exact", qk_main_row, 0.0));
    let _ = writeln!(
        rd,
        "{}",
        diag(
            "qk_then_exact_gating",
            rerank_two_stage(&v, RerankBase::Qk(&qk42), &g42, sel_t),
            0.0
        )
    );
    let _ = writeln!(
        rd,
        "{}",
        diag(
            "gatingqk_then_exact_gating",
            rerank_two_stage(&v, RerankBase::Gqk(&g42, sel_t, &qk42), &g42, sel_t),
            0.0
        )
    );
    let _ = unif_us;
    std::fs::write(format!("{out_dir}/rerank_diagnostics.csv"), &rd).unwrap();

    // ---------- §16 latency micro-bench (warm, this binary) ----------
    let lat = latency_bench(&v, &g42, sel_t, &qk42);
    std::fs::write(format!("{out_dir}/latency.csv"), &lat.0).unwrap();

    // ---------- §18 params ----------
    let gparams = g42.w1.len() + g42.b1.len() + g42.w2.len() + g42.b2.len();
    let qparams = 2 * DQ * v.ds.input_dim;
    std::fs::write(
        format!("{out_dir}/params.csv"),
        format!(
            "component,params,serialized_bytes\ngating,{gparams},{}\nqk,{qparams},{}\n",
            std::fs::metadata(format!("{out_dir}/models/gating_s{}.json", seeds[0]))
                .map(|m| m.len())
                .unwrap_or(0),
            std::fs::metadata(format!("{out_dir}/models/qk_s{}.json", seeds[0]))
                .map(|m| m.len())
                .unwrap_or(0)
        ),
    )
    .unwrap();

    // ---------- variability ----------
    let mut var = String::from("arm,metric,mean,std,min,max\n");
    for arm in ["gating", "qk", "gating_qk"] {
        for (metric, vals) in [
            (
                "R@10",
                per_seed
                    .iter()
                    .filter(|(a, _, _)| a.split('(').next() == Some(arm))
                    .map(|(_, _, m)| m.recall_at_10)
                    .collect::<Vec<_>>(),
            ),
            (
                "NDCG@10",
                per_seed
                    .iter()
                    .filter(|(a, _, _)| a.split('(').next() == Some(arm))
                    .map(|(_, _, m)| m.ndcg_at_10)
                    .collect::<Vec<_>>(),
            ),
            (
                "MRR",
                per_seed
                    .iter()
                    .filter(|(a, _, _)| a.split('(').next() == Some(arm))
                    .map(|(_, _, m)| m.mrr)
                    .collect::<Vec<_>>(),
            ),
        ] {
            let n = vals.len() as f64;
            let mu = vals.iter().sum::<f64>() / n;
            let sd = (vals.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / n).sqrt();
            let _ = writeln!(
                var,
                "{arm},{metric},{mu:.4},{sd:.4},{:.4},{:.4}",
                vals.iter().cloned().fold(f64::INFINITY, f64::min),
                vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            );
        }
    }
    std::fs::write(format!("{out_dir}/variability.csv"), &var).unwrap();

    // ---------- config/metrics ----------
    let cfg = serde_json::json!({
        "experiment": format!("PH2C-QK-002-{}", corpus_name.to_uppercase()),
        "corpus": corpus_name,
        "dataset_path": dataset_path,
        "content_path": content_path,
        "dataset_hash": format!("{ds_hash:#x}"),
        "seeds": seeds,
        "qk": {"kind": "linear Wq/Wk", "dq": DQ, "frozen_lr": qk_lr, "frozen_T": qk_t,
                "objective": "multi_positive_infonce", "grid": "lr×T val-only (seed 42)"},
        "gating": {"protocol": "3 objectives val-select + val temperature (shipped Phase 2B)", "frozen_objective": format!("{sel_obj:?}"), "frozen_T": sel_t},
        "gating_qk": {"fusion": "RRF k=60 of gating and QK rankings"},
        "budgets": BUDGETS, "rerank_top_m": RERANK_TOP,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch": BATCH,
    });
    std::fs::write(
        format!("{out_dir}/config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .unwrap();

    let g_agg = rows
        .iter()
        .find(|(a, s, _)| a == "gating" && s == "agg")
        .unwrap()
        .2;
    let q_agg = rows
        .iter()
        .find(|(a, s, _)| a == "qk" && s == "agg")
        .unwrap()
        .2;
    let gq_agg = rows
        .iter()
        .find(|(a, s, _)| a == "gating_qk" && s == "agg")
        .unwrap()
        .2;
    let _ = writeln!(report,
        "TEST aggregates: gating R@10={:.4} | qk R@10={:.4} (Δ={:+.4}) | gating+qk R@10={:.4} | uniform R@10={:.4} | oracle R@10={:.4}",
        g_agg.recall_at_10, q_agg.recall_at_10, q_agg.recall_at_10 - g_agg.recall_at_10,
        gq_agg.recall_at_10,
        rows.iter().find(|(a, _, _)| a == "uniform").unwrap().2.recall_at_10,
        rows.iter().find(|(a, _, _)| a == "oracle").unwrap().2.recall_at_10);
    let _ = writeln!(
        report,
        "latency p50 µs: gating={:.3} qk={:.3} (elapsed total {:?})",
        lat.1,
        lat.2,
        t_start.elapsed()
    );
    std::fs::write(format!("{out_dir}/report.txt"), &report).unwrap();
    format!("PH2C-QK-002-{corpus_name}: done in {:?}", t_start.elapsed())
}

enum ArmKind<'a> {
    Uniform,
    Gating(&'a GatingMlp, f32),
    Qk(&'a QkModel),
    Gqk(&'a GatingMlp, f32, &'a QkModel),
    OraclePq,
}

fn eval_qk_arm_generic(v: &View, idx: &[usize], kind: ArmKind) -> RankMetrics {
    let rs: Vec<RankMetrics> = idx
        .iter()
        .map(|&qi| {
            let q = &v.ds.queries[qi];
            let ranked = match kind {
                ArmKind::Uniform => fuse_weighted(q, &uniform_weights(v.ds.num_heads)),
                ArmKind::Gating(m, t) => fuse_weighted(q, &gating_weights(m, q, t)),
                ArmKind::Qk(m) => {
                    let qp = m.q_proj(&q.query);
                    let s = m.scores(&qp, &v.unions[qi], &v.docs);
                    v.unions[qi]
                        .iter()
                        .map(|&d| v.docs.idx2eng[d])
                        .zip(s.iter().copied())
                        .collect()
                }
                ArmKind::Gqk(m, t, qm) => {
                    let gr = fuse_weighted(q, &gating_weights(m, q, t));
                    let qp = qm.q_proj(&q.query);
                    let s = qm.scores(&qp, &v.unions[qi], &v.docs);
                    let qr: Vec<(u64, f32)> = v.unions[qi]
                        .iter()
                        .map(|&d| v.docs.idx2eng[d])
                        .zip(s.iter().copied())
                        .collect();
                    rrf_combine(&gr, &qr)
                }
                ArmKind::OraclePq => {
                    let mut best = (0usize, -1.0f32);
                    for (h, he) in q.heads.iter().enumerate() {
                        if he.recall_at_k > best.1 {
                            best = (h, he.recall_at_k);
                        }
                    }
                    let mut w = vec![0.0f32; q.heads.len()];
                    w[best.0] = 1.0;
                    fuse_weighted(q, &w)
                }
            };
            rank_sorted(&ranked, &q.ground_truth, v.ds.top_k)
        })
        .collect();
    avg(&rs)
}

fn diagnostics_for(v: &View, idx: &[usize], g: &GatingMlp, t: f32, qm: &QkModel) -> (f64, f64) {
    let (mut top1_share, mut gap) = (0.0f64, 0.0f64);
    for &qi in idx {
        let q = &v.ds.queries[qi];
        let w = gating_weights(g, q, t);
        top1_share += w.iter().cloned().fold(0.0f32, f32::max) as f64;
        let qp = qm.q_proj(&q.query);
        let mut s = qm.scores(&qp, &v.unions[qi], &v.docs);
        s.sort_by(|a, b| b.partial_cmp(a).unwrap());
        gap += (s[0] - s[1]) as f64;
    }
    let n = idx.len().max(1) as f64;
    (top1_share / n, gap / n)
}

enum RerankBase<'a> {
    Qk(&'a QkModel),
    Gqk(&'a GatingMlp, f32, &'a QkModel),
}

/// Two-stage rerank: rerank the top-RERANK_TOP of the base ranking by
/// exact-weighted(gating) scores; remaining candidates keep base order below.
fn rerank_two_stage(v: &View, base: RerankBase, g: &GatingMlp, t: f32) -> RankMetrics {
    let rs: Vec<RankMetrics> =
        v.ds.queries
            .iter()
            .filter(|q| q.split == Split::Test)
            .map(|q| {
                let b: Vec<(u64, f32)> = match base {
                    RerankBase::Qk(m) => {
                        let qp = m.q_proj(&q.query);
                        let s = m.scores(&qp, &v.unions[qi_of(q)], &v.docs);
                        v.unions[qi_of(q)]
                            .iter()
                            .map(|&d| v.docs.idx2eng[d])
                            .zip(s.iter().copied())
                            .collect()
                    }
                    RerankBase::Gqk(m, mt, qm) => {
                        let gr = fuse_weighted(q, &gating_weights(m, q, mt));
                        let qp = qm.q_proj(&q.query);
                        let s = qm.scores(&qp, &v.unions[qi_of(q)], &v.docs);
                        let qr: Vec<(u64, f32)> = v.unions[qi_of(q)]
                            .iter()
                            .map(|&d| v.docs.idx2eng[d])
                            .zip(s.iter().copied())
                            .collect();
                        rrf_combine(&gr, &qr)
                    }
                };
                let mut bs = b.clone();
                bs.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
                let head: HashSet<u64> = bs.iter().take(RERANK_TOP).map(|x| x.0).collect();
                let w = gating_weights(g, q, t);
                let mut acc: HashMap<u64, f32> = HashMap::new();
                for (h, he) in q.heads.iter().enumerate() {
                    for (i, &c) in he.candidates.iter().enumerate() {
                        if head.contains(&c) {
                            *acc.entry(c).or_insert(0.0) += w[h] * he.exact_scores[i];
                        }
                    }
                }
                let mut rr: Vec<(u64, f32)> = bs
                    .iter()
                    .take(RERANK_TOP)
                    .map(|&(c, _)| (c, *acc.get(&c).unwrap_or(&0.0)))
                    .collect();
                rr.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
                let mut final_ranked = rr;
                final_ranked.extend(bs.into_iter().skip(RERANK_TOP));
                rank_sorted(&final_ranked, &q.ground_truth, v.ds.top_k)
            })
            .collect();
    avg(&rs)
}

fn latency_bench(v: &View, g: &GatingMlp, t: f32, qm: &QkModel) -> (String, f64, f64) {
    let tq = test_queries(v);
    let reps = 20;
    let mut t_gating = Vec::new();
    let mut t_qk = Vec::new();
    let mut t_fuse = Vec::new();
    let mut t_rrf = Vec::new();
    let mut t_exact = Vec::new();
    // warmup
    for _ in 0..3 {
        for &qi in &tq {
            let q = &v.ds.queries[qi];
            let _ = gating_weights(g, q, t);
            let qp = qm.q_proj(&q.query);
            let _ = qm.scores(&qp, &v.unions[qi], &v.docs);
            let _ = fuse_weighted(q, &uniform_weights(v.ds.num_heads));
            let _ = fuse_rrf(q, RRF_K);
        }
    }
    for _ in 0..reps {
        for &qi in &tq {
            let q = &v.ds.queries[qi];
            let t0 = std::time::Instant::now();
            let _ = gating_weights(g, q, t);
            t_gating.push(t0.elapsed().as_secs_f64() * 1e6);
            let t1 = std::time::Instant::now();
            let qp = qm.q_proj(&q.query);
            let _ = qm.scores(&qp, &v.unions[qi], &v.docs);
            t_qk.push(t1.elapsed().as_secs_f64() * 1e6);
            let t2 = std::time::Instant::now();
            let _ = fuse_weighted(q, &uniform_weights(v.ds.num_heads));
            t_fuse.push(t2.elapsed().as_secs_f64() * 1e6);
            let t3 = std::time::Instant::now();
            let _ = fuse_rrf(q, RRF_K);
            t_rrf.push(t3.elapsed().as_secs_f64() * 1e6);
            let t4 = std::time::Instant::now();
            let mut acc = 0.0f32;
            for he in &q.heads {
                for &e in &he.exact_scores {
                    acc += e;
                }
            }
            let _ = acc;
            t_exact.push(t4.elapsed().as_secs_f64() * 1e6);
        }
    }
    let row = |name: &str, v: &[f64]| {
        format!(
            "{name},{:.3},{:.3},{:.3},{:.0}\n",
            pct(v, 50.0),
            pct(v, 95.0),
            pct(v, 99.0),
            1e6 / pct(v, 50.0)
        )
    };
    let mut csv = String::from("component,p50_us,p95_us,p99_us,qps\n");
    csv.push_str(&row("gating_forward", &t_gating));
    csv.push_str(&row("qk_forward_union", &t_qk));
    csv.push_str(&row("fuse_weighted_pools", &t_fuse));
    csv.push_str(&row("fuse_rrf_pools", &t_rrf));
    csv.push_str(&row("exact_sum_pools", &t_exact));
    let pg: Vec<f64> = t_gating
        .iter()
        .zip(t_fuse.iter())
        .map(|(a, b)| a + b)
        .collect();
    let pq: Vec<f64> = t_qk.iter().zip(t_rrf.iter()).map(|(a, b)| a + b).collect();
    csv.push_str(&row("pipeline_gating_fwd+fuse", &pg));
    csv.push_str(&row("pipeline_qk_fwd+rankfuse", &pq));
    (csv, pct(&t_gating, 50.0), pct(&t_qk, 50.0))
}
