//! Phase 2B — offline retrieval evaluation over cached datasets (§9, §10,
//! §28, §29, §33). Everything here works from `GatingDataset` so evaluation
//! never re-runs HNSW.

use crate::gating_v2::{GatingDataset, QueryExample, Split};
use serde::Serialize;
use std::collections::HashMap;

/// Weighted fusion: score(c) = Σ_h w_h · norm_h(c) over heads where the
/// candidate appears. This mirrors the pipeline's mhs component.
pub fn fuse_weighted(q: &QueryExample, weights: &[f32]) -> Vec<(u64, f32)> {
    let mut acc: HashMap<u64, f32> = HashMap::new();
    for (h, head) in q.heads.iter().enumerate() {
        let w = weights.get(h).copied().unwrap_or(0.0);
        for (i, &c) in head.candidates.iter().enumerate() {
            *acc.entry(c).or_insert(0.0) += w * head.norm_scores[i];
        }
    }
    let mut v: Vec<(u64, f32)> = acc.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    v
}

/// RRF baseline (§33): score(c) = Σ_h 1/(k + rank_h(c)); deterministic.
pub fn fuse_rrf(q: &QueryExample, k: f32) -> Vec<(u64, f32)> {
    let mut acc: HashMap<u64, f32> = HashMap::new();
    for head in &q.heads {
        for (rank, &c) in head.candidates.iter().enumerate() {
            *acc.entry(c).or_insert(0.0) += 1.0 / (k + rank as f32);
        }
    }
    let mut v: Vec<(u64, f32)> = acc.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    v
}

/// Single-head ranking (no fusion).
pub fn head_ranking(q: &QueryExample, head: usize) -> Vec<(u64, f32)> {
    q.heads[head]
        .candidates
        .iter()
        .zip(q.heads[head].norm_scores.iter())
        .map(|(&c, &s)| (c, s))
        .collect()
}

/// Ranking metrics against ordered ground truth (§28).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct RankMetrics {
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub recall_at_50: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
}

pub fn rank_metrics(ranked: &[(u64, f32)], gt: &[u64], k_gt: usize) -> RankMetrics {
    // relevant set = GT top-k_gt ids; GT is ordered (best first) for NDCG.
    let rel: std::collections::HashSet<u64> = gt.iter().take(k_gt).copied().collect();
    let idcg: f64 = (0..k_gt).map(|i| 1.0 / ((i + 2) as f64).log2()).sum();
    let at = |k: usize| -> f64 {
        let hits = ranked
            .iter()
            .take(k)
            .filter(|(id, _)| rel.contains(id))
            .count();
        hits as f64 / k_gt as f64
    };
    let ndcg: f64 = ranked
        .iter()
        .take(10)
        .enumerate()
        .filter(|(_, (id, _))| rel.contains(id))
        .map(|(i, _)| 1.0 / ((i + 2) as f64).log2())
        .sum::<f64>()
        / idcg;
    let mrr = ranked
        .iter()
        .position(|(id, _)| rel.contains(id))
        .map(|p| 1.0 / (p + 1) as f64)
        .unwrap_or(0.0);
    RankMetrics {
        recall_at_1: at(1),
        recall_at_5: at(5),
        recall_at_10: at(10),
        recall_at_50: at(50),
        ndcg_at_10: ndcg,
        mrr,
    }
}

impl RankMetrics {
    pub fn get_r1(&self) -> f64 {
        self.recall_at_1
    }
    pub fn get_r5(&self) -> f64 {
        self.recall_at_5
    }
    pub fn get_r10(&self) -> f64 {
        self.recall_at_10
    }
    pub fn get_ndcg(&self) -> f64 {
        self.ndcg_at_10
    }
    pub fn get_mrr(&self) -> f64 {
        self.mrr
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        f64::NAN
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Latency numbers are recorded by the live engine benchmarks; this struct
/// carries only retrieval-quality aggregates from the cached dataset plus
/// head-count context (§28: candidate/head counts are dataset properties).
#[derive(Debug, Clone, Default, Serialize)]
pub struct EvalReport {
    pub n_queries: usize,
    pub metrics: RankMetrics,
    pub avg_candidates_per_query: f64,
    pub num_heads: usize,
}

/// Evaluate ANY weighting function on a split. `weights_for(query) -> per-head w`.
pub fn evaluate_weighting<F>(ds: &GatingDataset, split: Split, weights_for: &F) -> EvalReport
where
    F: Fn(&QueryExample) -> Vec<f32>,
{
    let (mut r1, mut r5, mut r10, mut r50, mut nd, mut mrr) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut cands = Vec::new();
    for q in &ds.queries {
        if q.split != split {
            continue;
        }
        let ranked = fuse_weighted(q, &weights_for(q));
        let m = rank_metrics(&ranked, &q.ground_truth, ds.top_k);
        r1.push(m.recall_at_1);
        r5.push(m.recall_at_5);
        r10.push(m.recall_at_10);
        r50.push(m.recall_at_50);
        nd.push(m.ndcg_at_10);
        mrr.push(m.mrr);
        cands.push(q.heads.iter().map(|h| h.candidates.len()).sum::<usize>() as f64);
    }
    EvalReport {
        n_queries: r10.len(),
        metrics: RankMetrics {
            recall_at_1: mean(&r1),
            recall_at_5: mean(&r5),
            recall_at_10: mean(&r10),
            recall_at_50: mean(&r50),
            ndcg_at_10: mean(&nd),
            mrr: mean(&mrr),
        },
        avg_candidates_per_query: mean(&cands),
        num_heads: ds.num_heads,
    }
}

/// Uniform baseline (§29): equal weights over all heads.
pub fn uniform_weights(n: usize) -> Vec<f32> {
    vec![1.0 / n as f32; n]
}

/// Global-best single head: chosen on TRAIN split quality only (never test —
/// otherwise it would be an oracle). Returns (head_index, mean train recall).
pub fn global_best_head(ds: &GatingDataset, split: Split) -> (usize, f64) {
    let n = ds.num_heads;
    let mut sums = vec![0.0f64; n];
    let mut count = 0usize;
    for q in &ds.queries {
        if q.split != split {
            continue;
        }
        count += 1;
        for (h, head) in q.heads.iter().enumerate() {
            sums[h] += head.recall_at_k as f64;
        }
    }
    let mut best = 0usize;
    for h in 1..n {
        if sums[h] > sums[best] {
            best = h;
        }
    }
    (
        best,
        if count == 0 {
            f64::NAN
        } else {
            sums[best] / count as f64
        },
    )
}

/// ORACLE per-query head (§10): argmax per-head recall on the query itself.
/// Upper bound only — uses ground truth at query time; never shippable.
pub fn oracle_weights(q: &QueryExample) -> Vec<f32> {
    let mut best = 0usize;
    for h in 1..q.heads.len() {
        if q.heads[h].recall_at_k > q.heads[best].recall_at_k {
            best = h;
        }
    }
    let mut w = vec![0.0f32; q.heads.len()];
    w[best] = 1.0;
    w
}

/// ORACLE for single-head evaluation of a fused weighting (§29): evaluate the
/// oracle's fused report too (usually ≈ best head's own ranking).
pub fn evaluate_single_head(ds: &GatingDataset, split: Split, head: usize) -> EvalReport {
    let (mut r1, mut r5, mut r10, mut r50, mut nd, mut mrr) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for q in &ds.queries {
        if q.split != split {
            continue;
        }
        let ranked = head_ranking(q, head);
        let m = rank_metrics(&ranked, &q.ground_truth, ds.top_k);
        r1.push(m.recall_at_1);
        r5.push(m.recall_at_5);
        r10.push(m.recall_at_10);
        r50.push(m.recall_at_50);
        nd.push(m.ndcg_at_10);
        mrr.push(m.mrr);
    }
    let n = r10.len();
    EvalReport {
        n_queries: n,
        metrics: RankMetrics {
            recall_at_1: mean(&r1),
            recall_at_5: mean(&r5),
            recall_at_10: mean(&r10),
            recall_at_50: mean(&r50),
            ndcg_at_10: mean(&nd),
            mrr: mean(&mrr),
        },
        avg_candidates_per_query: f64::NAN,
        num_heads: 1,
    }
}

/// RRF evaluation on a split (§33).
pub fn evaluate_rrf(ds: &GatingDataset, split: Split, k: f32) -> EvalReport {
    let (mut r1, mut r5, mut r10, mut r50, mut nd, mut mrr) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for q in &ds.queries {
        if q.split != split {
            continue;
        }
        let ranked = fuse_rrf(q, k);
        let m = rank_metrics(&ranked, &q.ground_truth, ds.top_k);
        r1.push(m.recall_at_1);
        r5.push(m.recall_at_5);
        r10.push(m.recall_at_10);
        r50.push(m.recall_at_50);
        nd.push(m.ndcg_at_10);
        mrr.push(m.mrr);
    }
    EvalReport {
        n_queries: r10.len(),
        metrics: RankMetrics {
            recall_at_1: mean(&r1),
            recall_at_5: mean(&r5),
            recall_at_10: mean(&r10),
            recall_at_50: mean(&r50),
            ndcg_at_10: mean(&nd),
            mrr: mean(&mrr),
        },
        avg_candidates_per_query: f64::NAN,
        num_heads: ds.num_heads,
    }
}

/// Head diversity (§21/§22): pairwise top-K overlap (Jaccard@K) between heads.
#[derive(Debug, Clone, Serialize)]
pub struct PairOverlap {
    pub head_a: usize,
    pub head_b: usize,
    pub jaccard_at_k: f64,
}

pub fn head_diversity(ds: &GatingDataset, split: Split, k: usize) -> Vec<PairOverlap> {
    let n = ds.num_heads;
    let mut acc = vec![0.0f64; n * n];
    let mut count = 0usize;
    for q in &ds.queries {
        if q.split != split {
            continue;
        }
        count += 1;
        let tops: Vec<std::collections::HashSet<u64>> = (0..n)
            .map(|h| q.heads[h].candidates.iter().take(k).copied().collect())
            .collect();
        for a in 0..n {
            for b in 0..n {
                let inter = tops[a].intersection(&tops[b]).count();
                let union = tops[a].union(&tops[b]).count();
                acc[a * n + b] += if union == 0 {
                    0.0
                } else {
                    inter as f64 / union as f64
                };
            }
        }
    }
    let mut out = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            out.push(PairOverlap {
                head_a: a,
                head_b: b,
                jaccard_at_k: if count == 0 {
                    f64::NAN
                } else {
                    acc[a * n + b] / count as f64
                },
            });
        }
    }
    out
}

/// MinMax normalize scores in place (degenerate range → zeros); matches the
/// pipeline's stage-3 normalization for cached datasets.
pub fn minmax(scores: &mut [f32]) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for s in scores.iter() {
        if s.is_finite() {
            lo = lo.min(*s);
            hi = hi.max(*s);
        }
    }
    let range = hi - lo;
    if !range.is_finite() || range < 1e-6 {
        for s in scores.iter_mut() {
            *s = 0.0;
        }
        return;
    }
    for s in scores.iter_mut() {
        *s = if s.is_finite() {
            (*s - lo) / range
        } else {
            0.0
        };
    }
}

/// Fusion with an arbitrary per-head score selector (§32): lets the rerank
/// study fuse EXACT per-head similarities with the same weighted-sum code
/// path as normalized fusion, on identical cached candidates.
pub fn fuse_weighted_scores<F>(q: &QueryExample, weights: &[f32], score_of: F) -> Vec<(u64, f32)>
where
    F: Fn(&crate::gating_v2::HeadExample, usize) -> f32,
{
    let mut acc: HashMap<u64, f32> = HashMap::new();
    for (h, head) in q.heads.iter().enumerate() {
        let w = weights.get(h).copied().unwrap_or(0.0);
        for (i, &c) in head.candidates.iter().enumerate() {
            *acc.entry(c).or_insert(0.0) += w * score_of(head, i);
        }
    }
    let mut v: Vec<(u64, f32)> = acc.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    v
}

/// Mean of a ranking metric list helper for studies.
pub fn mean_metric(ranked: &[(u64, f32)], gt: &[u64], k: usize) -> RankMetrics {
    rank_metrics(ranked, gt, k)
}
