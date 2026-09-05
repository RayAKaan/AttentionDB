//! Query planner (Phase 2 §16/§17/§36) — deterministic heuristics over
//! collection metadata, producing a `RetrievalPlan` BEFORE execution.
//!
//! The planner is deliberately not cost-based: it makes bounded, explainable
//! decisions from explicit inputs (requested heads, top_k, filter presence,
//! collection facts). Every decision it makes is recorded in the plan so
//! EXPLAIN shows what actually happened (§17).

use crate::collection::{HybridStrategy, RetrievalConfig, RetrievalMode};
use crate::error::CoreError;
use crate::retrieval::ScoreNormalization;
use attentiondb_query::filter::FilterExpr;
use std::fmt;

/// Hard query limits (§18/§42) — enforced before any allocation.
#[derive(Debug, Clone, Copy)]
pub struct QueryLimits {
    pub max_top_k: usize,
    pub max_query_text_len: usize,
    pub max_query_vector_dim: usize,
    pub max_heads: usize,
    pub max_deadline_ms: u64,
}

impl Default for QueryLimits {
    fn default() -> Self {
        Self {
            max_top_k: 1_000,
            max_query_text_len: 4_096,
            max_query_vector_dim: 4_096,
            max_heads: 32,
            max_deadline_ms: 60_000,
        }
    }
}

impl QueryLimits {
    pub fn validate_top_k(&self, top_k: usize) -> Result<(), CoreError> {
        if top_k == 0 {
            return Err(CoreError::InvalidArgument("top_k must be >= 1".into()));
        }
        if top_k > self.max_top_k {
            return Err(CoreError::InvalidArgument(format!(
                "top_k {} exceeds limit {}",
                top_k, self.max_top_k
            )));
        }
        Ok(())
    }

    pub fn validate_heads(&self, heads: &[String]) -> Result<(), CoreError> {
        if heads.len() > self.max_heads {
            return Err(CoreError::InvalidArgument(format!(
                "{} heads requested, limit is {}",
                heads.len(),
                self.max_heads
            )));
        }
        Ok(())
    }

    pub fn validate_text(&self, text: &str) -> Result<(), CoreError> {
        if text.len() > self.max_query_text_len {
            return Err(CoreError::InvalidArgument(format!(
                "query text {} bytes exceeds limit {}",
                text.len(),
                self.max_query_text_len
            )));
        }
        Ok(())
    }

    pub fn validate_vector_dim(&self, dim: usize) -> Result<(), CoreError> {
        if dim > self.max_query_vector_dim {
            return Err(CoreError::InvalidArgument(format!(
                "query vector dim {dim} exceeds limit {}",
                self.max_query_vector_dim
            )));
        }
        Ok(())
    }

    pub fn validate_deadline_ms(&self, ms: u64) -> Result<(), CoreError> {
        if ms == 0 || ms > self.max_deadline_ms {
            return Err(CoreError::InvalidArgument(format!(
                "timeout_ms must be in 1..={}",
                self.max_deadline_ms
            )));
        }
        Ok(())
    }
}

/// Filter execution placement (§12). Phase 2 ships PostFilter (correctness
/// first: post-filtering can never leak a non-matching document) with bounded
/// candidate expansion. PreFilter (filtered HNSW scan) is reserved for the
/// benchmarked planner upgrade — see docs/retrieval/filtering.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterExecution {
    PostFilterWithExpansion,
    /// Filter-only scan (no vector ranking).
    ScanOnly,
}

/// How the text channel is combined with the vector channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HybridExecution {
    None,
    Rrf { k: f32 },
    LinearFusion,
}

/// The planned execution, produced before running anything (§16).
#[derive(Debug, Clone)]
pub struct RetrievalPlan {
    pub collection: String,
    pub heads: Vec<String>,
    /// Heads that were requested but do not exist (skipped, reported).
    pub missing_heads: Vec<String>,
    pub top_k: usize,
    pub per_head_candidates: usize,
    pub candidate_budget: usize,
    pub mode: RetrievalMode,
    pub normalization: ScoreNormalization,
    pub parallel: bool,
    pub ef_search: usize,
    pub filter_execution: FilterExecution,
    pub hybrid: HybridExecution,
    pub exact_rerank: bool,
    /// Expansion rounds the filter path may use (0 = no filter).
    pub max_expansion_rounds: usize,
}

impl RetrievalPlan {
    /// EXPLAIN text (§17). Human-readable, one stage per block.
    pub fn explain_text(&self) -> String {
        let mut s = String::new();
        s.push_str("QUERY\n");
        s.push_str(&format!(
            "  collection: {}\n  top_k: {}\n",
            self.collection, self.top_k
        ));
        if !self.missing_heads.is_empty() {
            s.push_str(&format!(
                "  skipped missing heads: {}\n",
                self.missing_heads.join(", ")
            ));
        }
        s.push_str(&format!(
            "VECTOR SEARCH\n  heads: {}\n  ef_search: {}\n  candidates_per_head: {}\n  parallel: {}\n",
            if self.heads.is_empty() { "<none>".to_string() } else { self.heads.join(", ") },
            self.ef_search,
            self.per_head_candidates,
            self.parallel,
        ));
        s.push_str(&format!(
            "CANDIDATE UNION\n  budget: {}\n",
            self.candidate_budget
        ));
        s.push_str(&format!(
            "RERANK\n  mode: {:?}\n  normalization: {:?}\n  exact_similarity: {}\n",
            self.mode,
            self.normalization,
            if self.exact_rerank {
                "on (metric-consistent)"
            } else {
                "off"
            },
        ));
        match self.hybrid {
            HybridExecution::None => {}
            HybridExecution::Rrf { k } => {
                s.push_str(&format!("HYBRID\n  strategy: rrf (k={k})\n"));
            }
            HybridExecution::LinearFusion => {
                s.push_str("HYBRID\n  strategy: linear fusion (attention + mhs + bm25)\n");
            }
        }
        match self.filter_execution {
            FilterExecution::ScanOnly => {
                s.push_str("FILTER\n  strategy: scan-only (no vector ranking)\n")
            }
            FilterExecution::PostFilterWithExpansion => {
                s.push_str(&format!(
                    "FILTER\n  strategy: post-filter with candidate expansion (up to {} extra rounds)\n",
                    self.max_expansion_rounds
                ));
            }
        }
        s.push_str(&format!("FINAL TOP K\n  {}\n", self.top_k));
        s
    }
}

impl fmt::Display for RetrievalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.explain_text())
    }
}

/// Planner inputs gathered from the collection (facts, not guesses).
pub struct PlanFacts {
    pub existing_heads: Vec<String>,
    pub ef_search: usize,
}

/// Plan a single-collection vector (+filter, +text) retrieval.
/// Deterministic: identical inputs produce an identical plan.
#[allow(clippy::too_many_arguments)]
pub fn plan_retrieval(
    collection: &str,
    requested_heads: &[String],
    top_k: usize,
    filter: Option<&FilterExpr>,
    has_text: bool,
    facts: &PlanFacts,
    cfg: &RetrievalConfig,
    limits: &QueryLimits,
) -> Result<RetrievalPlan, CoreError> {
    limits.validate_top_k(top_k)?;
    limits.validate_heads(requested_heads)?;

    let mut heads = Vec::new();
    let mut missing = Vec::new();
    for h in requested_heads {
        if facts.existing_heads.iter().any(|x| x == h) {
            heads.push(h.clone());
        } else {
            missing.push(h.clone());
        }
    }
    if heads.is_empty() && !requested_heads.is_empty() {
        return Err(CoreError::CollectionNotFound(collection.to_string()));
    }
    // No explicit heads → all existing heads.
    if requested_heads.is_empty() {
        heads = facts.existing_heads.clone();
    }

    let per_head_candidates = (top_k * cfg.candidate_multiplier).max(cfg.min_candidates_per_head);
    let filter_execution = if filter.is_some() {
        FilterExecution::PostFilterWithExpansion
    } else {
        FilterExecution::ScanOnly
    };
    let hybrid = if has_text {
        match cfg.hybrid_strategy {
            HybridStrategy::Rrf => HybridExecution::Rrf { k: cfg.rrf_k },
            HybridStrategy::Fusion => HybridExecution::LinearFusion,
        }
    } else {
        HybridExecution::None
    };

    Ok(RetrievalPlan {
        collection: collection.to_string(),
        heads,
        missing_heads: missing,
        top_k,
        per_head_candidates,
        candidate_budget: cfg.candidate_budget,
        mode: cfg.mode,
        normalization: cfg.normalization,
        parallel: cfg.parallel,
        ef_search: facts.ef_search,
        filter_execution,
        hybrid,
        exact_rerank: cfg.mode == RetrievalMode::Full,
        max_expansion_rounds: if filter.is_some() { 2 } else { 0 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> PlanFacts {
        PlanFacts {
            existing_heads: vec!["default".into(), "semantic".into()],
            ef_search: 64,
        }
    }

    #[test]
    fn plans_are_deterministic_and_complete() {
        let cfg = RetrievalConfig::default();
        let p1 = plan_retrieval(
            "c",
            &["default".into()],
            10,
            None,
            false,
            &facts(),
            &cfg,
            &QueryLimits::default(),
        )
        .unwrap();
        let p2 = plan_retrieval(
            "c",
            &["default".into()],
            10,
            None,
            false,
            &facts(),
            &cfg,
            &QueryLimits::default(),
        )
        .unwrap();
        assert_eq!(p1.explain_text(), p2.explain_text());
        assert_eq!(p1.heads, vec!["default"]);
        assert_eq!(p1.per_head_candidates, 50);
        assert!(p1.exact_rerank);
        assert_eq!(p1.filter_execution, FilterExecution::ScanOnly);
    }

    #[test]
    fn missing_heads_are_reported_not_fatal() {
        let cfg = RetrievalConfig::default();
        let p = plan_retrieval(
            "c",
            &["default".into(), "ghost".into()],
            10,
            None,
            false,
            &facts(),
            &cfg,
            &QueryLimits::default(),
        )
        .unwrap();
        assert_eq!(p.heads, vec!["default"]);
        assert_eq!(p.missing_heads, vec!["ghost"]);
        // all-missing → error
        assert!(plan_retrieval(
            "c",
            &["ghost".into()],
            10,
            None,
            false,
            &facts(),
            &cfg,
            &QueryLimits::default()
        )
        .is_err());
    }

    #[test]
    fn limits_reject_abuse() {
        let limits = QueryLimits::default();
        assert!(limits.validate_top_k(10_000_000).is_err());
        assert!(limits.validate_top_k(10).is_ok());
        assert!(limits.validate_top_k(0).is_err());
        assert!(limits.validate_heads(&vec!["h".to_string(); 64]).is_err());
        assert!(limits.validate_text(&"x".repeat(10_000)).is_err());
        assert!(limits.validate_deadline_ms(0).is_err());
        assert!(limits.validate_deadline_ms(1_000_000).is_err());
    }

    #[test]
    fn explain_text_contains_all_stages() {
        let cfg = RetrievalConfig::default();
        let filter = FilterExpr::Comparison {
            field: "category".into(),
            op: FilterOp::Eq,
            value: attentiondb_query::filter::FilterValue::Str("book".into()),
        };
        let p = plan_retrieval(
            "c",
            &["default".into(), "semantic".into()],
            10,
            Some(&filter),
            true,
            &facts(),
            &cfg,
            &QueryLimits::default(),
        )
        .unwrap();
        let text = p.explain_text();
        for stage in [
            "QUERY",
            "VECTOR SEARCH",
            "CANDIDATE UNION",
            "RERANK",
            "HYBRID",
            "FILTER",
            "FINAL TOP K",
        ] {
            assert!(text.contains(stage), "explain missing '{stage}':\n{text}");
        }
        assert!(text.contains("post-filter"));
        assert!(text.contains("rrf"));
    }

    use attentiondb_query::filter::FilterOp;
}
