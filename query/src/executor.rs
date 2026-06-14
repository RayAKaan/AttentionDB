use crate::planner::PhysicalPlan;
use crate::error::QueryError;

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub ids: Vec<u64>,
    pub scores: Vec<f32>,
    pub latency_ms: f64,
    pub plan: String,
}

pub struct QueryExecutor;

impl QueryExecutor {
    /// Execute a physical plan against a query vector.
    /// In production this calls into Phase 2 HNSW indexes.
    pub fn execute(plan: &PhysicalPlan, query_vector: &[f32]) -> Result<QueryResult, QueryError> {
        let start = std::time::Instant::now();

        if query_vector.is_empty() {
            return Err(QueryError::Execution("Empty query vector".into()));
        }

        // Stage 1: Multi-head HNSW search (placeholder — would call Phase 2)
        let head_count = plan.hnsw_search.heads.len();
        if head_count == 0 {
            return Err(QueryError::Execution("No heads specified in plan".into()));
        }

        // Stage 2: Fusion + filter (placeholder)
        let candidate_count = plan.hnsw_search.k * head_count;
        let ids: Vec<u64> = (0..candidate_count as u64).collect();
        let scores: Vec<f32> = (0..candidate_count)
            .map(|i| 1.0 - (i as f32 / candidate_count as f32) * 0.5)
            .collect();

        // Apply min_weight threshold
        let mut filtered: Vec<(u64, f32)> = ids.into_iter()
            .zip(scores.into_iter())
            .filter(|(_, s)| *s >= plan.min_weight)
            .take(plan.top_k)
            .collect();

        // Stage 3: Exact rerank (placeholder)
        if let Some(ref _rerank) = plan.exact_rerank {
            filtered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            filtered.truncate(plan.top_k);
        }

        let latency = start.elapsed().as_secs_f64() * 1000.0;

        let (final_ids, final_scores): (Vec<_>, Vec<_>) = filtered.into_iter().unzip();

        Ok(QueryResult {
            ids: final_ids,
            scores: final_scores,
            latency_ms: latency,
            plan: format!("HNSW(ef={},k={}) + Filter(min={}) + Rerank",
                         plan.hnsw_search.ef, plan.hnsw_search.k, plan.min_weight),
        })
    }

    /// Execute a query with a rich result description
    pub fn execute_with_status(plan: &PhysicalPlan, query_vector: &[f32]) -> Result<(QueryResult, String), QueryError> {
        let result = Self::execute(plan, query_vector)?;
        let status = format!(
            "Executed plan: {} | Heads: {} | Top-K: {} | Results: {} | Latency: {:.3}ms",
            result.plan,
            plan.hnsw_search.heads.len(),
            plan.top_k,
            result.ids.len(),
            result.latency_ms,
        );
        Ok((result, status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::*;

    #[test]
    fn test_execute_empty_vector() {
        let plan = PhysicalPlan {
            hnsw_search: HNSWSearchStep { heads: vec![("semantic".into(), 1.0)], ef: 64, k: 30 },
            exact_rerank: None,
            filter_steps: vec![],
            top_k: 10,
            min_weight: 0.01,
        };
        let result = QueryExecutor::execute(&plan, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_no_heads() {
        let plan = PhysicalPlan {
            hnsw_search: HNSWSearchStep { heads: vec![], ef: 64, k: 30 },
            exact_rerank: None,
            filter_steps: vec![],
            top_k: 10,
            min_weight: 0.01,
        };
        let result = QueryExecutor::execute(&plan, &[0.1; 256]);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_basic() {
        let plan = PhysicalPlan {
            hnsw_search: HNSWSearchStep { heads: vec![("semantic".into(), 1.0)], ef: 64, k: 30 },
            exact_rerank: Some(ExactRerankStep { top_candidates: 30 }),
            filter_steps: vec![],
            top_k: 5,
            min_weight: 0.01,
        };
        let result = QueryExecutor::execute(&plan, &[0.1; 256]).unwrap();
        assert_eq!(result.ids.len(), 5);
        assert!(result.latency_ms >= 0.0);
    }

    #[test]
    fn test_execute_with_status() {
        let plan = PhysicalPlan {
            hnsw_search: HNSWSearchStep { heads: vec![("a".into(), 1.0), ("b".into(), 0.5)], ef: 64, k: 30 },
            exact_rerank: None,
            filter_steps: vec![],
            top_k: 10,
            min_weight: 0.05,
        };
        let (_result, status) = QueryExecutor::execute_with_status(&plan, &[0.1; 256]).unwrap();
        assert!(status.contains("Heads: 2"));
        assert!(status.contains("Top-K: 10"));
    }
}
