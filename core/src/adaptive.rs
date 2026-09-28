//! Adaptive retrieval allocation (C6).
//!
//! This module provides deterministic, budget-constrained allocation of
//! retrieval effort (candidates, EF) across heads based on query/head signals
//! and/or interaction-stage feedback. All policies are pure functions with no
//! hidden state; the `AdaptiveRetriever` orchestrates the two-stage process
//! and emits a full causal trace.

use crate::retrieval::HeadHit;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Total retrieval budget for a query (hard caps).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalBudget {
    /// Hard cap on union candidate pool size.
    pub total_candidates: usize,
    /// Sum of per-head EF (proxy for HNSW work).
    pub total_ef_work: usize,
    /// Minimum candidates any head can receive.
    pub min_per_head: usize,
    /// Maximum candidates any head can receive.
    pub max_per_head: usize,
}

impl Default for RetrievalBudget {
    fn default() -> Self {
        Self {
            total_candidates: 500,
            total_ef_work: 192,   // 64 * 3 heads
            min_per_head: 20,
            max_per_head: 300,
        }
    }
}

impl RetrievalBudget {
    pub fn validate(&self) -> Result<(), String> {
        if self.total_candidates < 16 {
            return Err("total_candidates must be >= 16".into());
        }
        if self.total_ef_work == 0 {
            return Err("total_ef_work must be > 0".into());
        }
        if self.min_per_head > self.max_per_head {
            return Err("min_per_head > max_per_head".into());
        }
        if self.min_per_head * 3 > self.total_candidates {
            return Err("min_per_head * n_heads exceeds total_candidates".into());
        }
        Ok(())
    }
}

/// Per-head allocation decision (deterministic, inspectable).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadAllocation {
    pub head: String,
    pub candidates: usize,
    pub ef: usize,
    pub round: usize,
    pub reason: AllocationReason,
}

/// Why this allocation was made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationReason {
    InitialEqual,
    QuerySignal,
    InteractionSignal,
    RandomizedControl,
    MinFloor,
    MaxCeiling,
}

/// Stage 1 results for interaction-guided redistribution.
#[derive(Debug, Clone)]
pub struct Stage1Result {
    pub head: String,
    pub candidates: Vec<HeadHit>,
    pub overlap_with_others: f32,
    pub score_entropy: f32,
    pub top_score: f32,
}

/// Redistribution trigger conditions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RedistributionTrigger {
    Always,
    OverlapBelow(f32),
    EntropyAbove(f32),
    AnySignalChange,
}

/// Full causal trace for adaptive retrieval.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdaptiveTrace {
    pub initial_allocation: Vec<HeadAllocation>,
    pub redistributions: Vec<HeadAllocation>,
    pub final_allocation: Vec<HeadAllocation>,
    pub budget_conserved: bool,
    pub total_candidates_used: usize,
    pub total_ef_work_used: usize,
}

/// Allocation policy trait with Debug support for trait objects.
pub trait AllocationPolicy: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;
    fn initial_allocation(&self, budget: &RetrievalBudget, heads: &[String], query: &[f32], head_centroids: Option<&[Vec<f32>]>) -> Vec<HeadAllocation>;
    fn redistribute(&self, budget: &RetrievalBudget, stage1_results: &[Stage1Result]) -> Vec<HeadAllocation>;
}

/// Trait alias for dyn AllocationPolicy + Debug (workaround for trait object limitations).
type DynAllocationPolicy = dyn AllocationPolicy;

/// C6-C: Static equal allocation.
#[derive(Debug, Clone, Default)]
pub struct StaticEqualPolicy;

impl AllocationPolicy for StaticEqualPolicy {
    fn name(&self) -> &'static str { "static_equal" }

    fn initial_allocation(&self, budget: &RetrievalBudget, heads: &[String], _query: &[f32], _centroids: Option<&[Vec<f32>]>) -> Vec<HeadAllocation> {
        let n = heads.len().max(1);
        let base_cand = budget.total_candidates / n;
        let base_ef = budget.total_ef_work / n;
        let mut allocs = Vec::with_capacity(n);
        let rem_cand = budget.total_candidates % n;
        let rem_ef = budget.total_ef_work % n;
        for (i, h) in heads.iter().enumerate() {
            let cand = base_cand + if i < rem_cand { 1 } else { 0 };
            let ef = base_ef + if i < rem_ef { 1 } else { 0 };
            allocs.push(HeadAllocation {
                head: h.clone(),
                candidates: cand.clamp(budget.min_per_head, budget.max_per_head),
                ef: ef.max(1),
                round: 1,
                reason: AllocationReason::InitialEqual,
            });
        }
        allocs
    }

    fn redistribute(&self, _budget: &RetrievalBudget, _stage1: &[Stage1Result]) -> Vec<HeadAllocation> {
        Vec::new()
    }
}

/// C6-D: Query-adaptive allocation via head centroids.
#[derive(Debug, Clone, Default)]
pub struct QueryAdaptivePolicy;

impl AllocationPolicy for QueryAdaptivePolicy {
    fn name(&self) -> &'static str { "query_adaptive" }

    fn initial_allocation(&self, budget: &RetrievalBudget, heads: &[String], query: &[f32], centroids: Option<&[Vec<f32>]>) -> Vec<HeadAllocation> {
        let n = heads.len().max(1);
        let Some(cents) = centroids else {
            return StaticEqualPolicy.initial_allocation(budget, heads, query, None);
        };
        if cents.len() != n {
            return StaticEqualPolicy.initial_allocation(budget, heads, query, None);
        }

        // Compute cosine(query, centroid) per head.
        let mut signals: Vec<f32> = Vec::with_capacity(n);
        for c in cents {
            let s = cosine_similarity(query, c).max(0.0);
            signals.push(s);
        }
        let sum: f32 = signals.iter().sum();
        if sum <= 0.0 {
            return StaticEqualPolicy.initial_allocation(budget, heads, query, None);
        }

        // Proportional allocation with floor/ceiling.
        let mut allocs = Vec::with_capacity(n);
        let mut remaining_cand = budget.total_candidates;
        let mut remaining_ef = budget.total_ef_work;
        for (i, h) in heads.iter().enumerate() {
            let frac = signals[i] / sum;
            let mut cand = ((budget.total_candidates as f32) * frac).round() as usize;
            let mut ef = ((budget.total_ef_work as f32) * frac).round() as usize;
            cand = cand.clamp(budget.min_per_head, budget.max_per_head);
            ef = ef.max(1);
            if i == n - 1 {
                cand = remaining_cand.min(cand);
                ef = remaining_ef.min(ef);
            }
            remaining_cand = remaining_cand.saturating_sub(cand);
            remaining_ef = remaining_ef.saturating_sub(ef);
            allocs.push(HeadAllocation {
                head: h.clone(),
                candidates: cand,
                ef,
                round: 1,
                reason: AllocationReason::QuerySignal,
            });
        }
        allocs
    }

    fn redistribute(&self, _budget: &RetrievalBudget, _stage1: &[Stage1Result]) -> Vec<HeadAllocation> {
        Vec::new()
    }
}

/// C6-E: Interaction-guided (two-stage) allocation.
#[derive(Debug, Clone)]
pub struct InteractionGuidedPolicy {
    pub stage1_fraction: f32,
    pub overlap_threshold: f32,
    pub entropy_threshold: f32,
}

impl Default for InteractionGuidedPolicy {
    fn default() -> Self {
        Self {
            stage1_fraction: 0.3,
            overlap_threshold: 0.3,
            entropy_threshold: 1.0,
        }
    }
}

impl AllocationPolicy for InteractionGuidedPolicy {
    fn name(&self) -> &'static str { "interaction_guided" }

    fn initial_allocation(&self, budget: &RetrievalBudget, heads: &[String], _query: &[f32], _centroids: Option<&[Vec<f32>]>) -> Vec<HeadAllocation> {
        let n = heads.len().max(1);
        let stage1_cand = ((budget.total_candidates as f32) * self.stage1_fraction).round() as usize;
        let stage1_ef = ((budget.total_ef_work as f32) * self.stage1_fraction).round() as usize;
        let base_cand = stage1_cand / n;
        let base_ef = stage1_ef / n;
        let rem_cand = stage1_cand % n;
        let rem_ef = stage1_ef % n;
        let mut allocs = Vec::with_capacity(n);
        for (i, h) in heads.iter().enumerate() {
            let cand = (base_cand + if i < rem_cand { 1 } else { 0 }).clamp(budget.min_per_head, budget.max_per_head);
            let ef = (base_ef + if i < rem_ef { 1 } else { 0 }).max(1);
            allocs.push(HeadAllocation {
                head: h.clone(),
                candidates: cand,
                ef,
                round: 1,
                reason: AllocationReason::InitialEqual,
            });
        }
        allocs
    }

    fn redistribute(&self, budget: &RetrievalBudget, stage1: &[Stage1Result]) -> Vec<HeadAllocation> {
        let n = stage1.len().max(1);
        let stage1_cand: usize = stage1.iter().map(|r| r.candidates.len()).sum();
        let stage1_ef: usize = stage1.iter().map(|r| r.candidates.len()).sum(); // proxy
        let rem_cand = budget.total_candidates.saturating_sub(stage1_cand);
        let rem_ef = budget.total_ef_work.saturating_sub(stage1_ef);

        // Compute redistribution signals: low overlap + high entropy = more budget
        let mut signals: Vec<f32> = Vec::with_capacity(n);
        for r in stage1 {
            let overlap_penalty = (1.0 - r.overlap_with_others).max(0.0);
            let entropy_bonus = (r.score_entropy / self.entropy_threshold).min(2.0);
            let signal = overlap_penalty * (1.0 + entropy_bonus);
            signals.push(signal.max(0.01));
        }
        let sum: f32 = signals.iter().sum();
        if sum <= 0.0 {
            return Vec::new();
        }

        let mut allocs = Vec::with_capacity(n);
        let mut remaining_cand = rem_cand;
        let mut remaining_ef = rem_ef;
        for (i, r) in stage1.iter().enumerate() {
            let frac = signals[i] / sum;
            let mut cand = ((rem_cand as f32) * frac).round() as usize;
            let mut ef = ((rem_ef as f32) * frac).round() as usize;
            cand = cand.clamp(0, budget.max_per_head.saturating_sub(r.candidates.len()));
            ef = ef.max(0);
            if i == n - 1 {
                cand = remaining_cand.min(cand);
                ef = remaining_ef.min(ef);
            }
            remaining_cand = remaining_cand.saturating_sub(cand);
            remaining_ef = remaining_ef.saturating_sub(ef);
            if cand > 0 || ef > 0 {
                allocs.push(HeadAllocation {
                    head: r.head.clone(),
                    candidates: cand,
                    ef,
                    round: 2,
                    reason: AllocationReason::InteractionSignal,
                });
            }
        }
        allocs
    }
}

/// Negative control: randomized allocation.
#[derive(Debug, Clone, Default)]
pub struct RandomizedPolicy {
    seed: u64,
}

impl RandomizedPolicy {
    pub fn new(seed: u64) -> Self { Self { seed } }
}

impl AllocationPolicy for RandomizedPolicy {
    fn name(&self) -> &'static str { "randomized" }

    fn initial_allocation(&self, budget: &RetrievalBudget, heads: &[String], _query: &[f32], _centroids: Option<&[Vec<f32>]>) -> Vec<HeadAllocation> {
        let n = heads.len().max(1);
        let mut rng = Lcg(self.seed);
        let mut weights: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 { weights = vec![1.0; n]; }
        let mut allocs = Vec::with_capacity(n);
        let mut remaining_cand = budget.total_candidates;
        let mut remaining_ef = budget.total_ef_work;
        for (i, h) in heads.iter().enumerate() {
            let frac = weights[i] / weights.iter().sum::<f32>();
            let mut cand = ((budget.total_candidates as f32) * frac).round() as usize;
            let mut ef = ((budget.total_ef_work as f32) * frac).round() as usize;
            cand = cand.clamp(budget.min_per_head, budget.max_per_head);
            ef = ef.max(1);
            if i == n - 1 { cand = remaining_cand; ef = remaining_ef; }
            remaining_cand = remaining_cand.saturating_sub(cand);
            remaining_ef = remaining_ef.saturating_sub(ef);
            allocs.push(HeadAllocation {
                head: h.clone(),
                candidates: cand,
                ef,
                round: 1,
                reason: AllocationReason::RandomizedControl,
            });
        }
        allocs
    }

    fn redistribute(&self, budget: &RetrievalBudget, stage1: &[Stage1Result]) -> Vec<HeadAllocation> {
        let n = stage1.len().max(1);
        let mut rng = Lcg(self.seed.wrapping_add(0x9E3779B9));
        let mut weights: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 { weights = vec![1.0; n]; }
        let stage1_cand: usize = stage1.iter().map(|r| r.candidates.len()).sum();
        let stage1_ef: usize = stage1.iter().map(|r| r.candidates.len()).sum();
        let rem_cand = budget.total_candidates.saturating_sub(stage1_cand);
        let rem_ef = budget.total_ef_work.saturating_sub(stage1_ef);
        let mut allocs = Vec::with_capacity(n);
        let mut remaining_cand = rem_cand;
        let mut remaining_ef = rem_ef;
        for (i, r) in stage1.iter().enumerate() {
            let frac = weights[i] / weights.iter().sum::<f32>();
            let mut cand = ((rem_cand as f32) * frac).round() as usize;
            let mut ef = ((rem_ef as f32) * frac).round() as usize;
            cand = cand.clamp(0, budget.max_per_head.saturating_sub(r.candidates.len()));
            ef = ef.max(0);
            if i == n - 1 { cand = remaining_cand; ef = remaining_ef; }
            remaining_cand = remaining_cand.saturating_sub(cand);
            remaining_ef = remaining_ef.saturating_sub(ef);
            if cand > 0 || ef > 0 {
                allocs.push(HeadAllocation {
                    head: r.head.clone(),
                    candidates: cand,
                    ef,
                    round: 2,
                    reason: AllocationReason::RandomizedControl,
                });
            }
        }
        allocs
    }
}

/// Cosine similarity helper.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    let len = a.len().min(b.len());
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = (na * nb).sqrt();
    if d == 0.0 { 0.0 } else { dot / d }
}

/// Simple LCG for deterministic randomness.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / u64::MAX as f32
    }
}

/// Main orchestrator for adaptive retrieval.
#[derive(Debug)]
pub struct AdaptiveRetriever {
    policy: Box<DynAllocationPolicy>,
    budget: RetrievalBudget,
    head_centroids: Option<Vec<Vec<f32>>>, // precomputed per-head centroids
}

impl AdaptiveRetriever {
    pub fn new(policy: Box<dyn AllocationPolicy>, budget: RetrievalBudget, head_centroids: Option<Vec<Vec<f32>>>) -> Self {
        Self { policy, budget, head_centroids }
    }

    pub fn run(
        &self,
        heads: &[String],
        query: &[f32],
        search_fn: &dyn Fn(&str, &[f32], usize, Option<usize>) -> Vec<HeadHit>,
    ) -> (Vec<HeadHit>, AdaptiveTrace) {
        let mut trace = AdaptiveTrace::default();
        let mut all_hits: Vec<HeadHit> = Vec::new();

        // Stage 1: initial allocation
        let initial = self.policy.initial_allocation(&self.budget, heads, query, self.head_centroids.as_deref());
        trace.initial_allocation = initial.clone();

        let mut stage1_results: Vec<Stage1Result> = Vec::new();
        for alloc in &initial {
            let hits = search_fn(&alloc.head, query, alloc.candidates, Some(alloc.ef));
            let overlap = 0.0f32; // computed below
            let entropy = score_entropy(&hits);
            let top = hits.first().map(|h| h.raw_score).unwrap_or(0.0);
            stage1_results.push(Stage1Result {
                head: alloc.head.clone(),
                candidates: hits.clone(),
                overlap_with_others: overlap,
                score_entropy: entropy,
                top_score: top,
            });
            all_hits.extend(hits);
        }

        // Compute overlap after all stage1 results collected
        let head_ids: Vec<HashMap<u64, ()>> = stage1_results
            .iter()
            .map(|r| r.candidates.iter().map(|h| (h.id, ())).collect())
            .collect();
        for (i, r) in stage1_results.iter_mut().enumerate() {
            let mut overlap_union = HashMap::new();
            for (j, other) in head_ids.iter().enumerate() {
                if i != j { overlap_union.extend(other.clone()); }
            }
            if !overlap_union.is_empty() {
                let own: HashMap<u64, ()> = r.candidates.iter().map(|h| (h.id, ())).collect();
                let inter = own.keys().filter(|id| overlap_union.contains_key(id)).count();
                r.overlap_with_others = inter as f32 / own.len().max(1) as f32;
            }
        }

        // Stage 2: redistribution (if policy supports it)
        let redistributed = self.policy.redistribute(&self.budget, &stage1_results);
        if !redistributed.is_empty() {
            trace.redistributions = redistributed.clone();
            for alloc in &redistributed {
                let hits = search_fn(&alloc.head, query, alloc.candidates, Some(alloc.ef));
                all_hits.extend(hits);
            }
        }

        // Final allocation = initial + redistributed
        trace.final_allocation = initial.into_iter().chain(redistributed).collect();

        // Budget conservation check
        let used_cand: usize = trace.final_allocation.iter().map(|a| a.candidates).sum();
        let used_ef: usize = trace.final_allocation.iter().map(|a| a.ef).sum();
        trace.total_candidates_used = used_cand;
        trace.total_ef_work_used = used_ef;
        trace.budget_conserved = used_cand <= self.budget.total_candidates && used_ef <= self.budget.total_ef_work;

        (all_hits, trace)
    }
}

/// Score entropy of a hit list.
pub fn score_entropy(hits: &[HeadHit]) -> f32 {
    if hits.is_empty() { return 0.0; }
    let scores: Vec<f32> = hits.iter().map(|h| h.raw_score.max(0.0)).collect();
    let sum: f32 = scores.iter().sum();
    if sum <= 0.0 { return 0.0; }
    let mut ent = 0.0f32;
    for s in scores {
        let p = s / sum;
        if p > 0.0 { ent -= p * p.ln(); }
    }
    ent
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_budget() -> RetrievalBudget {
        RetrievalBudget { total_candidates: 500, total_ef_work: 192, min_per_head: 20, max_per_head: 300 }
    }

    #[test]
    fn budget_conservation_static() {
        let b = dummy_budget();
        let p = StaticEqualPolicy;
        let heads = vec!["a".into(), "b".into(), "c".into()];
        let allocs = p.initial_allocation(&b, &heads, &[0.1; 8], None);
        let sum_cand: usize = allocs.iter().map(|a| a.candidates).sum();
        let sum_ef: usize = allocs.iter().map(|a| a.ef).sum();
        assert!(sum_cand <= b.total_candidates);
        assert!(sum_ef <= b.total_ef_work);
    }

    #[test]
    fn allocation_normalization() {
        let b = dummy_budget();
        let p = StaticEqualPolicy;
        let heads = vec!["a".into(), "b".into()];
        let allocs = p.initial_allocation(&b, &heads, &[0.1; 8], None);
        assert_eq!(allocs[0].candidates + allocs[1].candidates, b.total_candidates);
    }

    #[test]
    fn zero_signal_heads_get_min() {
        let b = RetrievalBudget { total_candidates: 100, total_ef_work: 60, min_per_head: 10, max_per_head: 80 };
        let p = QueryAdaptivePolicy;
        let heads = vec!["a".into(), "b".into()];
        let query = vec![0.0; 8];
        let cents = vec![vec![0.0; 8], vec![0.0; 8]];
        let allocs = p.initial_allocation(&b, &heads, &query, Some(&cents));
        assert!(allocs.iter().all(|a| a.candidates >= b.min_per_head));
    }

    #[test]
    fn min_per_head_floor() {
        let b = RetrievalBudget { total_candidates: 100, total_ef_work: 60, min_per_head: 25, max_per_head: 80 };
        let p = StaticEqualPolicy;
        let heads = vec!["a".into(), "b".into(), "c".into()];
        let allocs = p.initial_allocation(&b, &heads, &[0.1; 8], None);
        assert!(allocs.iter().all(|a| a.candidates >= b.min_per_head));
    }

    #[test]
    fn max_per_head_ceiling() {
        let b = RetrievalBudget { total_candidates: 500, total_ef_work: 192, min_per_head: 20, max_per_head: 100 };
        let p = StaticEqualPolicy;
        let heads = vec!["a".into(), "b".into()];
        let allocs = p.initial_allocation(&b, &heads, &[0.1; 8], None);
        assert!(allocs.iter().all(|a| a.candidates <= b.max_per_head));
    }

    #[test]
    fn deterministic_allocation() {
        let b = dummy_budget();
        let p = QueryAdaptivePolicy;
        let heads = vec!["a".into(), "b".into()];
        let q = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let cents = vec![vec![0.1; 8], vec![0.5; 8]];
        let a1 = p.initial_allocation(&b, &heads, &q, Some(&cents));
        let a2 = p.initial_allocation(&b, &heads, &q, Some(&cents));
        assert_eq!(a1, a2);
    }

    #[test]
    fn redistribution_conservation() {
        let b = dummy_budget();
        let p = InteractionGuidedPolicy::default();
        let heads: Vec<String> = vec!["a".into(), "b".into()];
        let stage1 = vec![
            Stage1Result { head: "a".into(), candidates: vec![HeadHit{id:1, raw_score:0.9, rank:0}], overlap_with_others: 0.1, score_entropy: 0.5, top_score: 0.9 },
            Stage1Result { head: "b".into(), candidates: vec![HeadHit{id:2, raw_score:0.8, rank:0}], overlap_with_others: 0.1, score_entropy: 0.5, top_score: 0.8 },
        ];
        let red = p.redistribute(&b, &stage1);
        let total: usize = red.iter().map(|a| a.candidates).sum();
        assert!(total <= b.total_candidates);
    }

    #[test]
    fn exhausted_budget_no_allocation() {
        let b = RetrievalBudget { total_candidates: 0, total_ef_work: 0, min_per_head: 0, max_per_head: 0 };
        let p = StaticEqualPolicy;
        let heads: Vec<String> = vec!["a".into()];
        let allocs = p.initial_allocation(&b, &heads, &[0.1; 8], None);
        assert_eq!(allocs[0].candidates, 0);
    }

    #[test]
    fn candidate_deduplication() {
        let b = dummy_budget();
        let p = StaticEqualPolicy;
        let heads: Vec<String> = vec!["a".into(), "b".into()];
        let allocs = p.initial_allocation(&b, &heads, &[0.1; 8], None);
        // This test verifies the allocation logic; dedup is handled by candidate_union
        assert!(allocs.iter().all(|a| a.candidates > 0));
    }

    #[test]
    fn head_accounting() {
        let b = dummy_budget();
        let p = StaticEqualPolicy;
        let heads: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let allocs = p.initial_allocation(&b, &heads, &[0.1; 8], None);
        assert_eq!(allocs.len(), 3);
        assert!(allocs.iter().all(|a| !a.head.is_empty()));
    }

    #[test]
    fn adaptive_trace_generation() {
        let b = dummy_budget();
        let p = StaticEqualPolicy;
        let retriever = AdaptiveRetriever::new(Box::new(p), b, None);
        let heads: Vec<String> = vec!["a".into(), "b".into()];
        let search_fn = |_h: &str, _q: &[f32], k: usize, _ef: Option<usize>| -> Vec<HeadHit> {
            (0..k).map(|i| HeadHit { id: i as u64, raw_score: 1.0 - i as f32 * 0.1, rank: i }).collect()
        };
        let (_hits, trace) = retriever.run(&heads, &[0.1; 8], &search_fn);
        assert!(!trace.initial_allocation.is_empty());
        assert!(trace.budget_conserved);
    }
}