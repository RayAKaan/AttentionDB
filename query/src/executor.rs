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
    /// Execute a physical plan against a real HNSW index.
    /// This is the only query execution path — no placeholder fallback.
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
}

/// Execute a parsed AQL statement (query or DDL).
///
/// The `indexes` parameter must point to the collection's head index map.
/// DDL-only callers (CREATE, ALTER) should pass an empty HashMap.
pub fn execute_statement(
    statement: &AQLStatement,
    indexes: &HashMap<String, attentiondb_hnsw::HNSWIndex>,
    query_vector: Option<&[f32]>,
) -> Result<ExecuteResult, QueryError> {
    match statement {
        AQLStatement::Query(query) => {
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

            let index = indexes.get(&query.collection)
                .ok_or_else(|| QueryError::Execution(format!("Collection '{}' not found", query.collection)))?;

            let result = QueryExecutor::execute_on_index(&plan, index, vec)?;
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
    use attentiondb_hnsw::HNSWConfig;

    fn create_test_index() -> attentiondb_hnsw::HNSWIndex {
        let mut index = attentiondb_hnsw::HNSWIndex::new("test", 4, HNSWConfig::default());
        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index
    }

    #[test]
    fn test_execute_on_index_basic() {
        let plan = PhysicalPlan {
            hnsw_search: HNSWSearchStep { heads: vec![("semantic".into(), 1.0)], ef: 64, k: 30 },
            exact_rerank: None,
            filter_steps: vec![],
            top_k: 5,
            min_weight: 0.01,
        };
        let index = create_test_index();
        let result = QueryExecutor::execute_on_index(&plan, &index, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.latency_ms >= 0.0);
    }

    #[test]
    fn test_execute_on_index_empty_vector() {
        let plan = PhysicalPlan {
            hnsw_search: HNSWSearchStep { heads: vec![("semantic".into(), 1.0)], ef: 64, k: 30 },
            exact_rerank: None,
            filter_steps: vec![],
            top_k: 10,
            min_weight: 0.01,
        };
        let index = create_test_index();
        let result = QueryExecutor::execute_on_index(&plan, &index, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_ddl() {
        use crate::parser::parse_aql;
        let aql = r#"CREATE COLLECTION papers (title TEXT) WITH (ef_search = 256, similarity = "cosine")"#;
        let stmt = parse_aql(aql).unwrap();
        let empty_indexes = HashMap::new();
        let result = execute_statement(&stmt, &empty_indexes, None).unwrap();
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

        let mut indexes = HashMap::new();
        let index = create_test_index();
        indexes.insert("docs".to_string(), index);

        let result = execute_statement(&stmt, &indexes, Some(&[1.0, 0.0, 0.0, 0.0])).unwrap();
        match result {
            ExecuteResult::QueryResult(r) => {
                assert!(!r.ids.is_empty());
            }
            _ => panic!("Expected QueryResult"),
        }
    }

    #[test]
    fn test_execute_query_collection_not_found() {
        use crate::parser::parse_aql;
        let aql = r#"ATTEND TO nonexistent WHERE QUERY "test" TOP_K 5"#;
        let stmt = parse_aql(aql).unwrap();
        let empty_indexes = HashMap::new();
        let result = execute_statement(&stmt, &empty_indexes, Some(&[1.0, 0.0, 0.0, 0.0]));
        assert!(result.is_err());
    }
}
