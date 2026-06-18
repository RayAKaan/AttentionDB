use crate::planner::PhysicalPlan;
use crate::parser::AQLStatement;
use crate::error::QueryError;
use attentiondb_hnsw::HNSWIndex;
use attentiondb_multihead::MultiHeadManager;
use attentiondb_distributed::shard::ShardManager;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub ids: Vec<u64>,
    pub scores: Vec<f32>,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct ExecuteResult {
    pub success: bool,
    pub message: String,
    pub affected_collection: Option<String>,
}

pub struct QueryExecutor;

impl QueryExecutor {
    pub fn execute(
        plan: &PhysicalPlan,
        index: &HNSWIndex,
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

        Ok(QueryResult { ids, scores, latency_ms: latency })
    }

    pub fn execute_distributed<F>(
        plan: &PhysicalPlan,
        shard_manager: &ShardManager,
        query_vector: &[f32],
        mut dispatch_to_shard: F,
    ) -> Result<QueryResult, QueryError>
    where
        F: FnMut(u32, &PhysicalPlan, &[f32]) -> Result<Vec<(u64, f32)>, QueryError>,
    {
        let start = std::time::Instant::now();
        if query_vector.is_empty() {
            return Err(QueryError::Execution("Empty query vector in distributed search".into()));
        }

        let shard_ids = shard_manager.list_shards();
        if shard_ids.is_empty() {
            return Err(QueryError::Execution("Active ShardManager contains no quorum shards".into()));
        }

        let mut gathered_results: Vec<(String, Vec<(u64, f32)>)> = Vec::with_capacity(shard_ids.len());

        for shard_id in shard_ids {
            let shard_hits = dispatch_to_shard(shard_id, plan, query_vector)?;
            let head_key = format!("remote_shard_{}", shard_id);
            gathered_results.push((head_key, shard_hits));
        }

        let gate_weights = vec![1.0; gathered_results.len()];
        let mut global_fused = attentiondb_multihead::fusion::fuse_scores(&gathered_results, &gate_weights);

        global_fused.truncate(plan.top_k);
        let (ids, scores): (Vec<u64>, Vec<f32>) = global_fused.into_iter().unzip();
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        Ok(QueryResult { ids, scores, latency_ms: latency })
    }

    pub fn execute_statement(
        statement: &AQLStatement,
        indexes: &mut HashMap<String, HNSWIndex>,
        head_managers: &mut HashMap<String, MultiHeadManager>,
        query_vector: Option<&[f32]>,
    ) -> Result<ExecuteResult, QueryError> {
        match statement {
            AQLStatement::Query(q) => {
                let vec = query_vector.ok_or_else(|| {
                    QueryError::Execution("Query vector required for ATTEND".into())
                })?;

                if let Some(manager) = head_managers.get(&q.collection) {
                    let heads = if q.heads.is_empty() || q.heads == ["default"] {
                        manager.list_heads()
                    } else {
                        q.heads.clone()
                    };

                    let mut head_results: Vec<(String, Vec<(u64, f32)>)> = Vec::new();
                    for head_name in &heads {
                        if let Ok(index) = manager.create_hnsw_index_for_head(
                            head_name,
                            vec.len(),
                            attentiondb_hnsw::HNSWConfig::default(),
                        ) {
                            if let Ok(results) = index.search(vec, q.top_k, None) {
                                head_results.push((head_name.clone(), results));
                            }
                        }
                    }

                    if head_results.is_empty() {
                        return Err(QueryError::Execution(format!(
                            "No heads available for collection '{}'", q.collection
                        )));
                    }

                    let gate_weights = vec![1.0; head_results.len()];
                    let mut fused = attentiondb_multihead::fusion::fuse_scores(&head_results, &gate_weights);
                    fused.truncate(q.top_k);

                    let count = fused.len();

                    return Ok(ExecuteResult {
                        success: true,
                        message: format!(
                            "Query executed on '{}' via {} head(s), {} results",
                            q.collection, head_results.len(), count
                        ),
                        affected_collection: Some(q.collection.clone()),
                    });
                }

                let index = indexes.get(&q.collection).ok_or_else(|| {
                    QueryError::Execution(format!("Collection '{}' not found", q.collection))
                })?;

                let plan = PhysicalPlan {
                    hnsw_search: crate::planner::HNSWSearchStep {
                        heads: q.heads.iter().map(|h| (h.clone(), 1.0)).collect(),
                        ef: 128,
                        k: q.top_k * 3,
                    },
                    exact_rerank: None,
                    filter_steps: vec![],
                    top_k: q.top_k,
                    min_weight: q.min_weight,
                };

                let result = Self::execute(&plan, index, vec)?;

                Ok(ExecuteResult {
                    success: true,
                    message: format!(
                        "Query executed on '{}', {} results, {:.2}ms",
                        q.collection, result.ids.len(), result.latency_ms
                    ),
                    affected_collection: Some(q.collection.clone()),
                })
            }

            AQLStatement::CreateCollection(c) => {
                let config = attentiondb_hnsw::HNSWConfig {
                    max_nb_connection: c.settings.max_nb_connection,
                    ef_construction: c.settings.ef_construction,
                    ef_search: c.settings.ef_search,
                    store_vectors: true,
                    max_elements: 1_000_000,
                };

                let index = attentiondb_hnsw::HNSWIndex::with_settings(
                    &c.collection,
                    256,
                    config,
                    c.settings.clone(),
                )
                .map_err(|e| QueryError::Execution(e.to_string()))?;

                indexes.insert(c.collection.clone(), index);

                let mut msg = format!("Created collection '{}'", c.collection);
                if !c.fields.is_empty() {
                    let field_str: Vec<String> = c.fields.iter()
                        .map(|(n, t)| format!("{}: {}", n, t))
                        .collect();
                    msg.push_str(&format!(" with fields [{}]", field_str.join(", ")));
                }
                msg.push_str(&format!(
                    " with settings (ef_search={}, ef_construction={}, max_connections={}, similarity={}, exact_rerank={})",
                    c.settings.ef_search,
                    c.settings.ef_construction,
                    c.settings.max_nb_connection,
                    c.settings.similarity_metric,
                    c.settings.enable_exact_reranking,
                ));
                if !c.head_settings.is_empty() {
                    let head_str: Vec<String> = c.head_settings.iter()
                        .map(|(name, s)| format!("{}: (ef_search={})", name, s.ef_search))
                        .collect();
                    msg.push_str(&format!(". Per-head settings: [{}]", head_str.join(", ")));
                }

                Ok(ExecuteResult {
                    success: true,
                    message: msg,
                    affected_collection: Some(c.collection.clone()),
                })
            }

            AQLStatement::AlterCollection(a) => {
                if let Some(index) = indexes.get_mut(&a.collection) {
                    index.update_settings(a.settings.clone())
                        .map_err(|e| QueryError::Execution(e.to_string()))?;

                    let mut msg = format!(
                        "Altered collection '{}' settings to (ef_search={}, ef_construction={}, max_connections={}, similarity={}, exact_rerank={})",
                        a.collection,
                        a.settings.ef_search,
                        a.settings.ef_construction,
                        a.settings.max_nb_connection,
                        a.settings.similarity_metric,
                        a.settings.enable_exact_reranking,
                    );
                    if !a.head_settings.is_empty() {
                        let head_str: Vec<String> = a.head_settings.iter()
                            .map(|(name, s)| format!("{}: (ef_search={})", name, s.ef_search))
                            .collect();
                        msg.push_str(&format!(". Per-head settings: [{}]", head_str.join(", ")));
                    }

                    Ok(ExecuteResult {
                        success: true,
                        message: msg,
                        affected_collection: Some(a.collection.clone()),
                    })
                } else {
                    Err(QueryError::Execution(format!("Collection '{}' not found", a.collection)))
                }
            }
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
    fn test_execute_basic() {
        let plan = PhysicalPlan {
            hnsw_search: HNSWSearchStep { heads: vec![("semantic".into(), 1.0)], ef: 64, k: 30 },
            exact_rerank: None,
            filter_steps: vec![],
            top_k: 5,
            min_weight: 0.01,
        };
        let index = create_test_index();
        let result = QueryExecutor::execute(&plan, &index, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.latency_ms >= 0.0);
    }

    #[test]
    fn test_execute_empty_vector() {
        let plan = PhysicalPlan {
            hnsw_search: HNSWSearchStep { heads: vec![("semantic".into(), 1.0)], ef: 64, k: 30 },
            exact_rerank: None,
            filter_steps: vec![],
            top_k: 10,
            min_weight: 0.01,
        };
        let index = create_test_index();
        let result = QueryExecutor::execute(&plan, &index, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_ddl() {
        use crate::parser::parse_aql;
        let aql = r#"CREATE COLLECTION papers (title TEXT) WITH (ef_search = 256, similarity = "cosine")"#;
        let stmt = parse_aql(aql).unwrap();
        let mut empty_indexes = HashMap::new();
        let mut empty_managers = HashMap::new();
        let result = QueryExecutor::execute_statement(&stmt, &mut empty_indexes, &mut empty_managers, None).unwrap();
        assert!(result.success);
        assert_eq!(result.affected_collection.as_deref(), Some("papers"));
        assert!(result.message.contains("ef_search=256"));
        assert!(result.message.contains("similarity=cosine"));

        let index = empty_indexes.get("papers");
        assert!(index.is_some());
    }

    #[test]
    fn test_execute_query_statement() {
        use crate::parser::parse_aql;
        let aql = r#"ATTEND TO docs WHERE QUERY "test" TOP_K 5"#;
        let stmt = parse_aql(aql).unwrap();

        let mut indexes = HashMap::new();
        let mut managers = HashMap::new();
        let index = create_test_index();
        indexes.insert("docs".to_string(), index);

        let result = QueryExecutor::execute_statement(&stmt, &mut indexes, &mut managers, Some(&[1.0, 0.0, 0.0, 0.0])).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_execute_distributed_scatter_gather() {
        let mut sm = ShardManager::with_virtual_nodes(10);
        sm.add_shard(attentiondb_distributed::shard::Shard::new(1, vec!["semantic".into()], "addr1"));
        sm.add_shard(attentiondb_distributed::shard::Shard::new(2, vec!["semantic".into()], "addr2"));

        let plan = PhysicalPlan {
            hnsw_search: HNSWSearchStep { heads: vec![("semantic".into(), 1.0)], ef: 64, k: 10 },
            exact_rerank: None,
            filter_steps: vec![],
            top_k: 5,
            min_weight: 0.0,
        };

        let result = QueryExecutor::execute_distributed(
            &plan,
            &sm,
            &[1.0, 0.0, 0.0, 0.0],
            |shard_id, _, _| {
                if shard_id == 1 {
                    Ok(vec![(101, 0.95), (102, 0.80)])
                } else {
                    Ok(vec![(201, 0.90), (202, 0.85)])
                }
            }
        ).unwrap();

        assert_eq!(result.ids.len(), 4);
        assert_eq!(result.ids[0], 101);
        assert_eq!(result.ids[1], 201);
        assert_eq!(result.ids[2], 202);
        assert_eq!(result.ids[3], 102);
    }

    #[test]
    fn test_execute_query_collection_not_found() {
        use crate::parser::parse_aql;
        let aql = r#"ATTEND TO nonexistent WHERE QUERY "test" TOP_K 5"#;
        let stmt = parse_aql(aql).unwrap();
        let mut empty_indexes = HashMap::new();
        let mut empty_managers = HashMap::new();
        let result = QueryExecutor::execute_statement(&stmt, &mut empty_indexes, &mut empty_managers, Some(&[1.0, 0.0, 0.0, 0.0]));
        assert!(result.is_err());
    }
}
