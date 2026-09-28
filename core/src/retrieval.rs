//! Staged retrieval pipeline (Phase 2).
//!
//! Canonical stages, each independently testable and benchmarkable:
//!
//! ```text
//! CandidateSet (per-head HNSW hits: id, raw score, rank)
//!   → candidate_union (bounded, provenance-preserving)
//!   → score normalization (MinMax | ZScore | Softmax | Rank; NaN-safe)
//!   → head gating (query → head weights; uniform or learned softmax)
//!   → candidate attention (Q·K/√d, retrieval-oriented; see docs/retrieval/attention.md)
//!   → fusion  S = α·attention + β·multi_head_similarity + γ·bm25  (α+β+γ=1)
//!   → exact rerank (optional; exact similarity from stored vectors)
//!   → top-K (deterministic: score DESC, id ASC)
//! ```
//!
//! HNSW = candidate generation. Attention = candidate interaction/adaptive
//! scoring. Exact rerank = precise final ordering. These are NOT blurred:
//! approximate graph scores never reach the output when exact reranking runs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Candidate generation types
// ---------------------------------------------------------------------------

/// One hit from one head's candidate generator (HNSW), pre-normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadHit {
    pub id: u64,
    /// Raw generator score as returned by the head (higher = better).
    pub raw_score: f32,
    /// 0-based rank within the head's result list.
    pub rank: usize,
}

/// Raw candidate generation output: per-head hit lists in head order.
#[derive(Debug, Clone, Default)]
pub struct CandidateSet {
    pub source_heads: Vec<String>,
    pub per_head: Vec<Vec<HeadHit>>,
}

impl CandidateSet {
    pub fn new(source_heads: Vec<String>) -> Self {
        let n = source_heads.len();
        Self {
            source_heads,
            per_head: Vec::with_capacity(n),
        }
    }

    pub fn push_head(&mut self, hits: Vec<HeadHit>) {
        self.per_head.push(hits);
    }
}

/// A candidate after union: one entry per document with full provenance.
#[derive(Debug, Clone)]
pub struct UnionCandidate {
    pub id: u64,
    /// Aligned with `CandidateSet::source_heads`; `None` = head did not return it.
    pub head_scores: Vec<Option<f32>>,
    /// Aligned ranks; `None` = not returned by that head.
    pub head_ranks: Vec<Option<usize>>,
}

/// Union of per-head candidate lists, preserving provenance.
///
/// Membership order is deterministic (first-seen in head order, then rank);
/// the returned list is bounded to `budget` by best normalized head score
/// (ties: id ASC). Missing heads are recorded as `None`, never fabricated.
pub fn candidate_union(set: &CandidateSet, budget: usize) -> Vec<UnionCandidate> {
    let mut order: Vec<u64> = Vec::new();
    let mut index: HashMap<u64, usize> = HashMap::new();
    let n_heads = set.source_heads.len();

    for hits in &set.per_head {
        for hit in hits {
            if !hit.raw_score.is_finite() {
                continue; // non-finite generator output is never a candidate
            }
            let slot = match index.get(&hit.id) {
                Some(&i) => i,
                None => {
                    let i = order.len();
                    order.push(hit.id);
                    index.insert(hit.id, i);
                    i
                }
            };
            let _ = slot;
        }
    }

    let mut union: Vec<UnionCandidate> = order
        .iter()
        .map(|&id| UnionCandidate {
            id,
            head_scores: vec![None; n_heads],
            head_ranks: vec![None; n_heads],
        })
        .collect();

    for (h, hits) in set.per_head.iter().enumerate() {
        for hit in hits {
            if !hit.raw_score.is_finite() {
                continue;
            }
            let i = index[&hit.id];
            union[i].head_scores[h] = Some(hit.raw_score);
            union[i].head_ranks[h] = Some(hit.rank);
        }
    }

    if union.len() > budget {
        // Bound by best head score (raw; normalization is per-head and monotonic
        // for the default, so raw max is a stable budget key), ties by id ASC.
        union.sort_by(|a, b| {
            let sa = a.head_scores.iter().fold(f32::NEG_INFINITY, |m, x| {
                m.max(x.unwrap_or(f32::NEG_INFINITY))
            });
            let sb = b.head_scores.iter().fold(f32::NEG_INFINITY, |m, x| {
                m.max(x.unwrap_or(f32::NEG_INFINITY))
            });
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        union.truncate(budget);
    }
    union
}

// ---------------------------------------------------------------------------
// Cross-head interaction (C5: research-only, additive, default-off)
// ---------------------------------------------------------------------------
//
// Legacy scoring-only cross-head behavior (union + fusion) does not change
// candidate GENERATION. C5 adds an optional one-step interaction that does:
// after the first per-head search, each head H receives a refined query built
// from the "cross-head surprise" set S_H = ids that OTHER heads round-1-revealed
// but H did not. H re-searches with q'_H (round 2), and its round-1 ∪ round-2
// list is capped back to the per-head budget. NO scoring/config change happens
// after union — stages 3..9 of the pipeline are identical to the no-interaction
// path, so any difference in output is attributable to the candidate SET.
//
// Effort rule (frozen in C5 protocol §4): round-1 runs at ef/2 and round-2 at
// ef - ef/2, so the total HNSW ef work of C5-C equals C5-B's single-search ef.
// λ = 0.0 is the glue arm and MUST reproduce the control candidate set.

// ---------------------------------------------------------------------------
// Adaptive retrieval allocation (C6)
// ---------------------------------------------------------------------------

/// Adaptive retrieval policy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptivePolicyType {
    /// C6-C: Static equal split across heads.
    StaticEqual,
    /// C6-D: Query-adaptive via head centroids.
    QueryAdaptive,
    /// C6-E: Interaction-guided (two-stage with redistribution).
    InteractionGuided,
    /// Negative control: randomized allocation.
    RandomizedControl,
}

impl Default for AdaptivePolicyType {
    fn default() -> Self { Self::StaticEqual }
}

/// Adaptive retrieval configuration (C6).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRetrievalConfig {
    pub policy_type: AdaptivePolicyType,
    /// Fraction of budget for stage 1 (C6-E).
    pub stage1_fraction: f32,
    /// Redistribution trigger (C6-E).
    pub overlap_threshold: f32,
    pub entropy_threshold: f32,
    /// Randomized control seed.
    pub randomized_seed: u64,
}

impl Default for AdaptiveRetrievalConfig {
    fn default() -> Self {
        Self {
            policy_type: AdaptivePolicyType::StaticEqual,
            stage1_fraction: 0.3,
            overlap_threshold: 0.3,
            entropy_threshold: 1.0,
            randomized_seed: 20260925,
        }
    }
}

/// Interaction parameters (validation-gated). Off by default (`None`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CrossRefineConfig {
    /// Blend of the query with the cross-head centroid (0.0 = pure query).
    pub lambda: f32,
}

/// Per-query causal ledger for the interaction cell (C5 protocol §6). Records
/// exactly what the interaction changed so a claim can be falsified or shown
/// inert. Lists are in HNSW rank order (best first) unless noted.
#[derive(Debug, Clone, Default)]
pub struct CrossHeadTrace {
    /// Heads that contributed (present) candidates, in head order.
    pub heads: Vec<String>,
    /// Round-1 per-head candidate ids (engine numeric ids), in rank order.
    pub round1: Vec<Vec<u64>>,
    /// S_H = ids returned by OTHER heads in round-1 but not by this head.
    pub surprise: Vec<Vec<u64>>,
    /// Whether H's round-2 query was refined (true) or = round-1 query (false).
    pub refined: Vec<bool>,
    /// Round-2 ids that were NOT in H's round-1 list (interaction additions).
    pub round2_added: Vec<Vec<u64>>,
    /// ids that reach the FINAL union only because of the interaction.
    pub interaction_union_additions: Vec<u64>,
    /// per-head list length after round-1 ∪ round-2 cap (== per_head_k when
    /// the head had that many).
    pub per_head_capped: Vec<usize>,
    /// union size of round-1 only.
    pub union_pre: usize,
    /// union size after interaction (== candidate_union budget cap applied).
    pub union_post: usize,
    /// ef used for round-1 and round-2 (sum == control ef).
    pub ef_r1: usize,
    pub ef_r2: usize,
}

/// L2-normalize in place; a zero/empty/non-finite vector is left as-is so the
/// caller can fall back to the original query (deterministic, NaN-safe).
pub fn l2_normalize(v: &mut [f32]) {
    let norm2: f64 = v.iter().map(|&x| x as f64 * x as f64).sum();
    if norm2.is_finite() && norm2 > 0.0 {
        let inv = 1.0 / norm2.sqrt();
        for x in v.iter_mut() {
            *x = (*x as f64 * inv) as f32;
        }
    }
}

/// Refined query for one head: `q' = normalize(q + λ·centroid)`.
/// λ == 0 returns the input query unchanged (glue arm, documented). A
/// degenerate centroid (empty) returns the query unchanged. Always finite.
pub fn cross_refine_query(q: &[f32], centroid: &[f32], lambda: f32) -> Vec<f32> {
    if lambda == 0.0 || centroid.is_empty() {
        return q.to_vec();
    }
    let mut out: Vec<f32> = q
        .iter()
        .zip(centroid.iter())
        .map(|(a, c)| {
            let v = *a + lambda * c;
            if v.is_finite() {
                v
            } else {
                *a
            }
        })
        .collect();
    // q and centroid are same-dim (both head embedding space); if lengths
    // differ (should not happen: gated by caller), keep q's tail unchanged.
    if q.len() > centroid.len() {
        out.extend_from_slice(&q[centroid.len()..]);
    }
    l2_normalize(&mut out);
    out
}

/// Mean vector of `vectors` (same dim), in id-ascending input order so the
/// float sum is deterministic. Empty/zero-norm → empty (caller falls back).
pub fn cross_head_centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    let Some(first_len) = vectors.first().map(|v| v.len()) else {
        return Vec::new();
    };
    // deterministic canonical order (ids already sorted by caller)
    let mut sum = vec![0.0f64; first_len];
    let mut n = 0usize;
    for v in vectors {
        if v.len() != first_len {
            continue;
        }
        for (s, &x) in sum.iter_mut().zip(v.iter()) {
            if x.is_finite() {
                *s += x as f64;
            }
        }
        n += 1;
    }
    if n == 0 || !sum.iter().all(|s| s.is_finite()) {
        return Vec::new();
    }
    sum.iter().map(|s| (*s / n as f64) as f32).collect()
}

// ---------------------------------------------------------------------------
// Score normalization (§7, §25 — float safety)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreNormalization {
    /// (x − min) / (max − min); degenerate (max == min) → all 0.5.
    MinMax,
    /// (x − μ) / σ; zero variance → all 0.0.
    ZScore,
    /// Numerically stable softmax (max-subtracted); set-relative.
    Softmax,
    /// 1 − rank/n (top = 1.0, last = 1/n); only order matters.
    Rank,
}

/// Sanitize a score slice: non-finite values become finite and sort last.
/// NaN/±Inf must never propagate into fused scores (§25).
fn sanitize(scores: &mut [f32]) {
    for s in scores.iter_mut() {
        if s.is_nan() {
            *s = f32::NEG_INFINITY; // NaN = unusable → sorts last
        } else if *s == f32::INFINITY {
            *s = f32::MAX; // +Inf is a valid "best"; clamp to keep math finite
        }
        // -Inf already sorts last and is handled per-method below.
    }
}

/// Normalize one head's scores in place. Input order is the head's rank order.
pub fn normalize_scores(scores: &mut [f32], method: ScoreNormalization) {
    sanitize(scores);
    if scores.is_empty() {
        return;
    }
    match method {
        ScoreNormalization::MinMax => {
            // Range computed over FINITE values only; sanitized (non-finite)
            // entries are fixed at 0.0 and never participate in (s−min)/(max−min),
            // which would otherwise produce inf/inf = NaN.
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &s in scores.iter() {
                if s.is_finite() {
                    if s < min {
                        min = s;
                    }
                    if s > max {
                        max = s;
                    }
                }
            }
            if !max.is_finite() {
                for s in scores.iter_mut() {
                    *s = 0.0;
                }
                return;
            }
            if max - min < 1e-6 {
                for s in scores.iter_mut() {
                    *s = if s.is_finite() { 0.5 } else { 0.0 };
                }
                return;
            }
            for s in scores.iter_mut() {
                *s = if s.is_finite() {
                    (*s - min) / (max - min)
                } else {
                    0.0
                };
            }
        }
        ScoreNormalization::ZScore => {
            let finite: Vec<f32> = scores
                .iter()
                .copied()
                .filter(|s| *s != f32::NEG_INFINITY)
                .collect();
            if finite.is_empty() {
                for s in scores.iter_mut() {
                    *s = 0.0;
                }
                return;
            }
            let n = finite.len() as f32;
            let mean = finite.iter().sum::<f32>() / n;
            let var = finite.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n;
            let std = var.sqrt();
            for s in scores.iter_mut() {
                *s = if *s == f32::NEG_INFINITY || std < f32::EPSILON {
                    0.0
                } else {
                    (*s - mean) / std
                };
            }
        }
        ScoreNormalization::Softmax => {
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max == f32::NEG_INFINITY {
                for s in scores.iter_mut() {
                    *s = 0.0;
                }
                return;
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                let e = (*s - max).exp(); // max-subtracted: no overflow (§25)
                *s = e;
                sum += e;
            }
            if sum <= 0.0 || !sum.is_finite() {
                for s in scores.iter_mut() {
                    *s = 0.0;
                }
                return;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }
        ScoreNormalization::Rank => {
            let n = scores.len();
            // Ranks computed on sanitized values; NEG_INFINITY (non-finite) sorts last.
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut out = vec![0.0f32; n];
            for (position, &i) in idx.iter().enumerate() {
                out[i] = 1.0 - (position as f32) / (n as f32);
            }
            scores.copy_from_slice(&out);
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic ordering (§24)
// ---------------------------------------------------------------------------

/// Final ordering comparator: final_score DESC, id ASC. Stable and total for
/// finite scores (scores are sanitized before reaching this point).
pub fn rank_comparator(a: &(u64, f32), b: &(u64, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.0.cmp(&b.0))
}

/// Deterministic top-K: dedupe ids (best score wins), drop non-finite scores,
/// sort by (score DESC, id ASC), truncate. Duplicate candidates can never
/// produce duplicate results (§46 regression protection).
pub fn deterministic_top_k(scored: Vec<(u64, f32)>, k: usize) -> Vec<(u64, f32)> {
    let mut best: HashMap<u64, f32> = HashMap::new();
    for (id, s) in scored {
        if !s.is_finite() {
            continue;
        }
        let slot = best.entry(id).or_insert(f32::NEG_INFINITY);
        if s > *slot {
            *slot = s;
        }
    }
    let mut out: Vec<(u64, f32)> = best.into_iter().collect();
    out.sort_by(rank_comparator);
    out.truncate(k);
    out
}

// ---------------------------------------------------------------------------
// RRF baseline (§15) — standard 1/(k + rank)
// ---------------------------------------------------------------------------

/// Reciprocal Rank Fusion with configurable k (literature default 60).
/// Inputs are per-source ranked lists (best first). Ties: id ASC.
pub fn rrf_fuse(lists: &[&[(u64, f32)]], k: f32, top_k: usize) -> Vec<(u64, f32)> {
    let mut acc: HashMap<u64, f32> = HashMap::new();
    for list in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            *acc.entry(*id).or_default() += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<(u64, f32)> = acc.into_iter().collect();
    out.sort_by(rank_comparator);
    out.truncate(top_k);
    out
}

// ---------------------------------------------------------------------------
// Candidate-level Q/K attention (§3) — mathematically defined, retrieval-first
// ---------------------------------------------------------------------------

/// Q/K attention scorer over candidates.
///
/// Definition (full derivation in docs/retrieval/attention.md):
/// - Query profile `q ∈ R^{H}` (H = heads): the head-gate distribution, i.e.
///   the query's "interest profile" over heads (uniform or learned softmax).
/// - Candidate features `x_i ∈ R^{F}`: normalized per-head similarities and
///   per-head rank features (missing head → 0).
/// - `Q = W_q · q`, `K_i = W_k · x_i` (both in R^{D}); init is exact identity
///   on the shared subspace so an untrained scorer reduces to the gated
///   multi-head similarity (MODE D starts AT MODE C — no fake gains).
/// - `attention_logit_i = (Q · K_i) / sqrt(D)`, bounded by tanh to keep the
///   score scale-compatible with similarities:
///   `attention_score_i = 0.5 * (tanh(logit_i) + 1) ∈ [0, 1]`.
///
/// Why tanh instead of softmax-over-candidates: a database must return scores
/// that are comparable across pages and stable under candidate-set changes;
/// softmax makes scores set-relative (removing one candidate rescales all
/// others). Complexity: O(C·(F + D)) for C candidates — never O(C²).
pub struct AttentionScorer {
    heads: usize,
    features: usize,
    attn_dim: usize,
    wq: Vec<f32>, // heads    × attn_dim, row-major
    wk: Vec<f32>, // features × attn_dim, row-major
}

impl AttentionScorer {
    /// Identity-initialized scorer: `W_q` selects the head profile,
    /// `W_k` selects the similarity features (rank features weighted 0).
    pub fn new(heads: usize, rank_feature_weight: f32) -> Self {
        let attn_dim = heads.max(1);
        let features = heads * 2; // [sims | ranks]
        let mut wq = vec![0.0; heads * attn_dim];
        for h in 0..heads.min(attn_dim) {
            wq[h * attn_dim + h] = 1.0;
        }
        let mut wk = vec![0.0; features * attn_dim];
        for h in 0..heads.min(attn_dim) {
            wk[h * attn_dim + h] = 1.0;
            // rank features contribute a fixed small amount, columns after sims
            wk[(heads + h) * attn_dim + h] = rank_feature_weight;
        }
        Self {
            heads,
            features,
            attn_dim,
            wq,
            wk,
        }
    }

    fn project_q(&self, profile: &[f32]) -> Vec<f32> {
        let mut q = vec![0.0; self.attn_dim];
        for (h, &v) in profile.iter().take(self.heads).enumerate() {
            if v.is_finite() {
                for (d, qd) in q.iter_mut().enumerate() {
                    *qd += v * self.wq[h * self.attn_dim + d];
                }
            }
        }
        q
    }

    fn project_k(&self, features: &[f32]) -> Vec<f32> {
        let mut k = vec![0.0; self.attn_dim];
        for (f, &v) in features.iter().take(self.features).enumerate() {
            if v.is_finite() {
                for (d, kd) in k.iter_mut().enumerate() {
                    *kd += v * self.wk[f * self.attn_dim + d];
                }
            }
        }
        k
    }

    /// Score all candidates. `profiles_per_candidate[i]` = candidate features.
    /// Returns attention scores in [0, 1]; non-finite inputs are ignored
    /// (treated as 0 features), so output is always finite.
    pub fn score(&self, profile: &[f32], features_per_candidate: &[Vec<f32>]) -> Vec<f32> {
        let q = self.project_q(profile);
        let scale = 1.0 / (self.attn_dim as f32).sqrt();
        features_per_candidate
            .iter()
            .map(|x| {
                let k = self.project_k(x);
                let logit: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
                if logit.is_finite() {
                    0.5 * (logit.tanh() + 1.0)
                } else {
                    0.5 // neutral for degenerate logits
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Fusion (§8) — explicit, normalized equation
// ---------------------------------------------------------------------------

/// Fusion weights; must sum to ~1 (validated).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionWeights {
    pub attention: f32,
    pub multi_head_similarity: f32,
    pub bm25: f32,
}

impl FusionWeights {
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("attention", self.attention),
            ("multi_head_similarity", self.multi_head_similarity),
            ("bm25", self.bm25),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(format!("fusion weight {name} must be finite and >= 0"));
            }
        }
        let sum = self.attention + self.multi_head_similarity + self.bm25;
        if (sum - 1.0).abs() > 1e-3 {
            return Err(format!("fusion weights must sum to 1, got {sum}"));
        }
        Ok(())
    }
}

/// Per-candidate feature bundle flowing through the pipeline.
#[derive(Debug, Clone, Default)]
pub struct CandidateFeatures {
    /// Normalized per-head similarity (aligned with head order).
    pub head_scores: Vec<Option<f32>>,
    /// Gated multi-head similarity: Σ gate_h · norm_sim_h / Σ gate_h.
    pub multi_head_similarity: f32,
    /// Normalized BM25 score (None when the query has no text component).
    pub bm25: Option<f32>,
    /// Candidate-level attention score (None when mode < D).
    pub attention: Option<f32>,
}

/// Final scoring equation (documented, no hidden multipliers):
/// `S = w.attention·attn + w.mhs·mhs + w.bm25·bm25` over the components
/// present; present weights are re-normalized to sum to 1. Output is always
/// finite (any non-finite component is dropped from the sum).
pub fn fuse_candidate(f: &CandidateFeatures, w: &FusionWeights) -> f32 {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    if let Some(a) = f.attention {
        if a.is_finite() {
            num += w.attention * a;
            den += w.attention;
        }
    }
    if f.multi_head_similarity.is_finite() {
        num += w.multi_head_similarity * f.multi_head_similarity;
        den += w.multi_head_similarity;
    }
    if let Some(b) = f.bm25 {
        if b.is_finite() {
            num += w.bm25 * b;
            den += w.bm25;
        }
    }
    if den > f32::EPSILON {
        num / den
    } else {
        0.0
    }
}

/// A fully scored candidate with its breakdown (§21 result explainability).
#[derive(Debug, Clone)]
pub struct RankedCandidate {
    pub id: u64,
    pub final_score: f32,
    pub features: CandidateFeatures,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hits(ids: &[u64], base: f32) -> Vec<HeadHit> {
        ids.iter()
            .enumerate()
            .map(|(r, &id)| HeadHit {
                id,
                raw_score: base - r as f32 * 0.1,
                rank: r,
            })
            .collect()
    }

    #[test]
    fn union_preserves_provenance_and_is_deterministic() {
        let mut set = CandidateSet::new(vec!["a".into(), "b".into()]);
        set.push_head(hits(&[1, 2, 3], 0.9));
        set.push_head(hits(&[2, 4], 0.8));
        let union = candidate_union(&set, 100);
        assert_eq!(union.len(), 4);
        let u2 = union.iter().find(|u| u.id == 2).unwrap();
        let s = |x: Option<f32>| x.unwrap_or(f32::NAN);
        assert!((s(u2.head_scores[0]) - 0.8).abs() < 1e-5);
        assert!((s(u2.head_scores[1]) - 0.8).abs() < 1e-5);
        assert_eq!(u2.head_ranks, vec![Some(1), Some(0)]);
        let u4 = union.iter().find(|u| u.id == 4).unwrap();
        assert_eq!(u4.head_scores[0], None, "head a did not return id 4");
        // order deterministic: first-seen
        let ids: Vec<u64> = union.iter().map(|u| u.id).collect();
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn union_budget_bounds_and_prefers_high_scores() {
        let mut set = CandidateSet::new(vec!["a".into()]);
        set.push_head(hits(&[10, 11, 12, 13], 0.9));
        let union = candidate_union(&set, 2);
        assert_eq!(union.len(), 2);
        assert_eq!(union[0].id, 10);
        assert_eq!(union[1].id, 11);
    }

    #[test]
    fn nonfinite_generator_output_never_enters_union() {
        let mut set = CandidateSet::new(vec!["a".into()]);
        set.push_head(vec![
            HeadHit {
                id: 1,
                raw_score: 0.9,
                rank: 0,
            },
            HeadHit {
                id: 2,
                raw_score: f32::NAN,
                rank: 1,
            },
            HeadHit {
                id: 3,
                raw_score: f32::INFINITY,
                rank: 2,
            },
        ]);
        let union = candidate_union(&set, 10);
        assert_eq!(union.len(), 1);
        assert_eq!(union[0].id, 1);
    }

    #[test]
    fn normalization_edge_cases_are_safe() {
        // empty
        let mut v: Vec<f32> = vec![];
        normalize_scores(&mut v, ScoreNormalization::MinMax);
        assert!(v.is_empty());

        // one candidate → MinMax 1.0 (it is both min and max → degenerate 0.5)
        let mut v = vec![0.7f32];
        normalize_scores(&mut v, ScoreNormalization::MinMax);
        assert!((v[0] - 0.5).abs() < 1e-6);

        // all equal → MinMax 0.5, ZScore 0, Softmax uniform
        let mut v = vec![0.4f32, 0.4, 0.4];
        normalize_scores(&mut v, ScoreNormalization::MinMax);
        assert!(v.iter().all(|&x| (x - 0.5).abs() < 1e-6));
        let mut v = vec![0.4f32, 0.4, 0.4];
        normalize_scores(&mut v, ScoreNormalization::ZScore);
        assert!(v.iter().all(|&x| x == 0.0));
        let mut v = vec![0.4f32, 0.4, 0.4];
        normalize_scores(&mut v, ScoreNormalization::Softmax);
        assert!(v.iter().all(|&x| (x - 1.0 / 3.0).abs() < 1e-6));

        // NaN and infinity never escape
        for method in [
            ScoreNormalization::MinMax,
            ScoreNormalization::ZScore,
            ScoreNormalization::Softmax,
            ScoreNormalization::Rank,
        ] {
            let mut v = vec![0.9f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.1];
            normalize_scores(&mut v, method);
            assert!(v.iter().all(|x| x.is_finite()), "{method:?} → {v:?}");
        }

        // negative scores normalize correctly
        let mut v = vec![-2.0f32, -1.0, 0.0];
        normalize_scores(&mut v, ScoreNormalization::MinMax);
        assert!(
            (v[0] - 0.0).abs() < 1e-6 && (v[1] - 0.5).abs() < 1e-6 && (v[2] - 1.0).abs() < 1e-6
        );

        // huge logits must not overflow softmax (max subtraction)
        let mut v = vec![1e30f32, 1e30 - 1.0, 0.0];
        normalize_scores(&mut v, ScoreNormalization::Softmax);
        assert!(v.iter().all(|x| x.is_finite()));
        assert!(
            (v[0] - v[1]).abs() < 1e-6,
            "near-equal logits stay near-equal"
        );
    }

    #[test]
    fn top_k_is_deterministic_on_ties() {
        let scored = vec![(7, 0.5f32), (3, 0.9), (7, 0.9), (1, 0.9)];
        let mut seen = std::collections::HashSet::new();
        let out = deterministic_top_k(scored.clone(), 10);
        for w in out.windows(2) {
            assert!(w[0].1 > w[1].1 || (w[0].1 == w[1].1 && w[0].0 < w[1].0));
        }
        // run twice: identical output
        let out2 = deterministic_top_k(scored, 10);
        for (a, b) in out.iter().zip(out2.iter()) {
            assert_eq!(a, b);
            assert!(seen.insert(a.0));
        }
    }

    #[test]
    fn rrf_uses_k_offset() {
        let l1 = [(1u64, 1.0f32), (2, 0.9)];
        let l2 = [(2, 1.0f32)];
        let out = rrf_fuse(&[&l1, &l2], 60.0, 10);
        // id 1: 1/61 ; id 2: 1/62 + 1/61 → id 2 wins
        assert_eq!(out[0].0, 2);
        assert!((out[0].1 - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-9);
    }

    #[test]
    fn attention_identity_init_matches_gated_similarity() {
        // With rank weight 0 and identity projections, MODE D degenerates to
        // the gated multi-head similarity — the honest baseline.
        let scorer = AttentionScorer::new(2, 0.0);
        let profile = [0.7, 0.3];
        // candidate fully matches head 0: features = [1.0, 0.0 | ranks ignored]
        let scores = scorer.score(&profile, &[vec![1.0, 0.0, 0.0, 0.0]]);
        let s_full = scores[0];
        // candidate matches both heads uniformly: 0.7·1 + 0.3·1 = 1.0 logit
        let scores2 = scorer.score(&profile, &[vec![1.0, 1.0, 0.0, 0.0]]);
        let s_uniform = scores2[0];
        assert!(s_uniform > s_full, "1.0 logit must outscore 0.7 logit");
        assert!((0.0..=1.0).contains(&s_full));
        assert!(s_full.is_finite());
    }

    #[test]
    fn attention_handles_nonfinite_features() {
        let scorer = AttentionScorer::new(2, 0.1);
        let scores = scorer.score(&[0.5, 0.5], &[vec![f32::NAN, 1.0, f32::INFINITY, 0.0]]);
        assert!(scores[0].is_finite());
        let scores = scorer.score(&[f32::NAN, 0.5], &[vec![1.0, 1.0, 0.0, 0.0]]);
        assert!(scores[0].is_finite());
    }

    #[test]
    fn fusion_renormalizes_present_components() {
        let w = FusionWeights {
            attention: 0.5,
            multi_head_similarity: 0.3,
            bm25: 0.2,
        };
        w.validate().unwrap();
        // no bm25 present → weights renormalize over attention+mhs
        let f = CandidateFeatures {
            head_scores: vec![Some(1.0)],
            multi_head_similarity: 1.0,
            bm25: None,
            attention: Some(0.0),
        };
        let s = fuse_candidate(&f, &w);
        // (0.5·0 + 0.3·1) / 0.8 = 0.375
        assert!((s - 0.375).abs() < 1e-6);
    }

    #[test]
    fn fusion_rejects_bad_weights() {
        let w = FusionWeights {
            attention: 0.5,
            multi_head_similarity: 0.5,
            bm25: 0.5,
        };
        assert!(w.validate().is_err());
        let w = FusionWeights {
            attention: f32::NAN,
            multi_head_similarity: 0.5,
            bm25: 0.5,
        };
        assert!(w.validate().is_err());
    }

    #[test]
    fn cross_refine_lambda_zero_is_glue() {
        // λ=0 must reproduce the query exactly: q' = normalize(q + 0·centroid),
        // and for cosine (scale-invariant) ranking this is the control arm.
        let q = vec![0.3, -0.5, 0.9];
        let c = vec![1.0, 1.0, 1.0];
        let q2 = cross_refine_query(&q, &c, 0.0);
        // (q + 0·c) = q; normalization preserves direction → same cosine order
        assert_eq!(q2, q, "λ=0 must return the input query unchanged");
    }

    #[test]
    fn cross_refine_empties_and_degenerate_are_safe() {
        let q = vec![0.1, 0.8];
        // empty centroid → fall back to query (never panic, always finite)
        assert_eq!(cross_refine_query(&q, &[], 0.5), q);
        // NaN in the centroid must not poison a finite query
        let out = cross_refine_query(&q, &[1.0, f32::NAN], 0.5);
        assert!(out.iter().all(|x| x.is_finite()), "output must be finite");
        // centroid longer than query still yields a query-length vector
        let long = cross_refine_query(&q, &[1.0, 0.0, 9.9], 0.5);
        assert_eq!(long.len(), q.len());
        assert!(long.iter().all(|x| x.is_finite()));
        // query longer than centroid keeps the query's unmatched tail
        let long_q = cross_refine_query(&[0.1, 0.8, 0.3], &[1.0, 0.0], 0.5);
        assert_eq!(long_q.len(), 3);
    }

    #[test]
    fn cross_refine_is_normalized_and_direction_sensitive() {
        let q = vec![1.0, 0.0];
        let c = vec![0.0, 1.0];
        let q2 = cross_refine_query(&q, &c, 1.0);
        let norm: f32 = q2.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "q' must be L2-normalized");
        // both components active (0.5, 0.5) — direction points into quadrant
        assert!(q2[0] > 0.0 && q2[1] > 0.0);
        // λ=0.1 must stay closer to the query than λ=1.0
        let q2_small = cross_refine_query(&q, &c, 0.1);
        assert!(
            q2_small[0] > q2[0],
            "smaller λ keeps q' closer to the original query"
        );
    }

    #[test]
    fn cross_head_centroid_is_deterministic_mean() {
        let v = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        // input order irrelevant → reordered input gives the same mean
        let mut v2 = v.clone();
        v2.reverse();
        let a = cross_head_centroid(&v);
        let b = cross_head_centroid(&v2);
        assert_eq!(a, b, "centroid must be order-independent");
        assert_eq!(a, vec![3.0, 4.0], "mean of the three vectors");
    }

    #[test]
    fn cross_head_centroid_skips_mismatched_and_empty_inputs() {
        assert_eq!(cross_head_centroid(&[]), Vec::<f32>::new());
        // mismatched dims are skipped; single-vector input is the mean
        let v = vec![vec![1.0, 2.0], vec![5.0, 6.0, 7.0]];
        assert_eq!(cross_head_centroid(&v), vec![1.0, 2.0]);
    }
}
