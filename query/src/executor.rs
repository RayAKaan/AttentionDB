use crate::planner::PhysicalPlan;
use crate::parser::AQLStatement;
use crate::error::QueryError;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub ids: Vec<u64>,
    pub scores: Vec<f32>,
    pub latency_ms: f64,
    pub plan: String,
}

#[derive(Debug, Clone)]
pub enum ExecuteResult {
    QueryResult(QueryResult),
    DdlResult { collection: String, message: String },
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

        let head_count = plan.hnsw_search.heads.len();
        if head_count == 0 {
            return Err(QueryError::Execution("No heads specified in plan".into()));
        }

        let candidate_count = plan.hnsw_search.k * head_count;
        let ids: Vec<u64> = (0..candidate_count as u64).collect();
        let scores: Vec<f32> = (0..candidate_count)
            .map(|i| 1.0 - (i as f32 / candidate_count as f32) * 0.5)
            .collect();

        let mut filtered: Vec<(u64, f32)> = ids.into_iter()
            .zip(scores.into_iter())
            .filter(|(_, s)| *s >= plan.min_weight)
            .take(plan.top_k)
            .collect();

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

    /// Execute a physical plan against a real HNSW index.
    /// Unlike `execute()`, this calls `index.search()` instead of returning placeholder results.
    pub fn execute_on_index(
        plan: &PhysicalPlan,
        index: &attentiondb_hnsw::HNSWIndex,
        query_vector: &[f32],
    ) -> Result<QueryResult, QueryError> {
        let start = std::time::Instant::now();

        if query_vector.is_empty() {
            return Err(QueryError::Execution("Empty query vector".into()));
        }

        let results = index.search(query_vector, plan.top_k, Some(plan.hnsw_search.ef))
            .map_err(|e| QueryError::Execution(e.to_string()))?;

        let (ids, scores): (Vec<u64>, Vec<f32>) = results.into_iter().unzip();
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        Ok(QueryResult {
            ids,
            scores,
            latency_ms: latency,
            plan: format!("RealHNSW(ef={},k={})", plan.hnsw_search.ef, plan.top_k),
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

/// Execute a parsed AQL statement (query or DDL).
///
/// If `indexes` is provided and the collection exists, the query path will
/// call `execute_on_index` for real HNSW search instead of placeholder results.
pub fn execute_statement(
    statement: &AQLStatement,
    indexes: Option<&HashMap<String, attentiondb_hnsw::HNSWIndex>>,
    query_vector: Option<&[f32]>,
) -> Result<ExecuteResult, QueryError> {
    match statement {
        AQLStatement::Query(query) => {
            // Build a plan from the parsed query (simplified — real planner is more complex)
            use crate::planner::*;
            let heads: Vec<(String, f32)> = query.heads.iter().map(|h| (h.clone(), 1.0)).collect();
            let plan = PhysicalPlan {
                hnsw_search: HNSWSearchStep {
                    heads,
                    ef: 64,
                    k: query.top_k * 3,
                },
                exact_rerank: Some(ExactRerankStep { top_candidates: query.top_k * 3 }),
                filter_steps: vec![],
                top_k: query.top_k,
                min_weight: query.min_weight,
            };
            let vec = query_vector.ok_or_else(|| QueryError::Execution("Query vector required".into()))?;

            // Use real search if an index is available, otherwise placeholder fallback
            let result = if let Some(indexes) = indexes {
                if let Some(index) = indexes.get(&query.collection) {
                    QueryExecutor::execute_on_index(&plan, index, vec)?
                } else {
                    QueryExecutor::execute(&plan, vec)?
                }
            } else {
                QueryExecutor::execute(&plan, vec)?
            };

            Ok(ExecuteResult::QueryResult(result))
        }
        AQLStatement::CreateCollection(coll) => {
            let mut msg = format!("Created collection '{}'", coll.collection);
            if !coll.fields.is_empty() {
                let field_str: Vec<String> = coll.fields.iter()
                    .map(|(n, t)| format!("{}: {}", n, t))
                    .collect();
                msg.push_str(&format!(" with fields [{}]", field_str.join(", ")));
            }
            msg.push_str(&format!(
                " with settings (ef_search={}, ef_construction={}, max_connections={}, similarity={}, exact_rerank={})",
                coll.settings.ef_search,
                coll.settings.ef_construction,
                coll.settings.max_nb_connection,
                coll.settings.similarity_metric,
                coll.settings.enable_exact_reranking,
            ));
            msg.push_str(".");
            Ok(ExecuteResult::DdlResult {
                collection: coll.collection.clone(),
                message: msg,
            })
        }
        AQLStatement::AlterCollection(alter) => {
            let msg = format!(
                "Altered collection '{}' settings to (ef_search={}, ef_construction={}, max_connections={}, similarity={}, exact_rerank={})",
                alter.collection,
                alter.settings.ef_search,
                alter.settings.ef_construction,
                alter.settings.max_nb_connection,
                alter.settings.similarity_metric,
                alter.settings.enable_exact_reranking,
            );
            Ok(ExecuteResult::DdlResult {
                collection: alter.collection.clone(),
                message: msg,
            })
        }
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

    #[test]
    fn test_execute_ddl() {
        use crate::parser::parse_aql;
        let aql = r#"CREATE COLLECTION papers (title TEXT) WITH (ef_search = 256, similarity = "cosine")"#;
        let stmt = parse_aql(aql).unwrap();
        let result = execute_statement(&stmt, None, None).unwrap();
        match result {
            ExecuteResult::DdlResult { collection, message } => {
                assert_eq!(collection, "papers");
                assert!(message.contains("ef_search=256"));
                assert!(message.contains("similarity=cosine"));
            }
            _ => panic!("Expected DdlResult"),
        }
    }

    #[test]
    fn test_execute_query_statement() {
        use crate::parser::parse_aql;
        let aql = r#"ATTEND TO docs WHERE QUERY "test" TOP_K 5"#;
        let stmt = parse_aql(aql).unwrap();
        let result = execute_statement(&stmt, None, Some(&[0.1; 256])).unwrap();
        match result {
            ExecuteResult::QueryResult(r) => {
                assert_eq!(r.ids.len(), 5);
            }
            _ => panic!("Expected QueryResult"),
        }
    }
}
