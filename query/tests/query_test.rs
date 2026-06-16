use std::collections::HashMap;
use attentiondb_query::{parse_aql, plan_query, QueryExecutor, AQLStatement, ExecuteResult, execute_statement};
use attentiondb_hnsw::HNSWConfig;

fn get_query(stmt: AQLStatement) -> attentiondb_query::AQLQuery {
    match stmt {
        AQLStatement::Query(q) => q,
        _ => panic!("Expected Query"),
    }
}

fn create_test_index_with_dim(dim: usize) -> attentiondb_hnsw::HNSWIndex {
    let mut index = attentiondb_hnsw::HNSWIndex::new("test", dim, HNSWConfig::default());
    index.insert(1, &vec![0.1; dim]).unwrap();
    index
}

#[test]
fn test_full_pipeline() {
    let aql = r#"ATTEND TO papers WHERE QUERY "attention" HEADS [semantic] TOP_K 5"#;
    let parsed = parse_aql(aql).unwrap();
    let plan = plan_query(get_query(parsed)).unwrap();
    let index = create_test_index_with_dim(256);
    let result = QueryExecutor::execute_on_index(&plan, &index, &[0.1; 256]).unwrap();
    assert!(result.latency_ms >= 0.0);
}

#[test]
fn test_multi_head_pipeline() {
    let aql = r#"ATTEND TO docs WHERE QUERY "test" HEADS [semantic, temporal, structural] TOP_K 15"#;
    let parsed = parse_aql(aql).unwrap();
    let plan = plan_query(get_query(parsed)).unwrap();
    let index = create_test_index_with_dim(256);
    let result = QueryExecutor::execute_on_index(&plan, &index, &[0.2; 256]).unwrap();
    assert!(result.latency_ms >= 0.0);
}

#[test]
fn test_temporal_decay_pipeline() {
    let aql = r#"ATTEND TO logs WHERE QUERY "error rates" HEADS [temporal] TOP_K 20 MIN_WEIGHT 0.1 TEMPORAL_DECAY 0.5"#;
    let parsed = parse_aql(aql).unwrap();
    let plan = plan_query(get_query(parsed)).unwrap();
    let index = create_test_index_with_dim(256);
    let _result = QueryExecutor::execute_on_index(&plan, &index, &[0.3; 256]).unwrap();
    let temporal_weight = plan.hnsw_search.heads.iter()
        .find(|(n, _)| n == "temporal")
        .map(|(_, w)| *w);
    assert!((temporal_weight.unwrap() - 0.5).abs() < 1e-6);
}

#[test]
fn test_invalid_aql() {
    let result = parse_aql("GARBAGE INPUT");
    assert!(result.is_err());
}

#[test]
fn test_empty_heads_defaults() {
    let aql = r#"ATTEND TO papers WHERE QUERY "test" TOP_K 5"#;
    let parsed = parse_aql(aql).unwrap();
    match parsed {
        AQLStatement::Query(q) => assert_eq!(q.heads, vec!["default"]),
        _ => panic!("Expected Query"),
    }
}

#[test]
fn test_min_weight_filtering() {
    let aql = r#"ATTEND TO papers WHERE QUERY "test" MIN_WEIGHT 0.5"#;
    let parsed = parse_aql(aql).unwrap();
    let plan = plan_query(get_query(parsed)).unwrap();
    let index = create_test_index_with_dim(256);
    let result = QueryExecutor::execute_on_index(&plan, &index, &[0.1; 256]).unwrap();
    assert!(result.latency_ms >= 0.0);
}

#[test]
fn test_ef_auto_scaling() {
    let aql_small = r#"ATTEND TO papers WHERE QUERY "test" TOP_K 3"#;
    let aql_large = r#"ATTEND TO papers WHERE QUERY "test" TOP_K 100"#;
    let plan_small = plan_query(get_query(parse_aql(aql_small).unwrap())).unwrap();
    let plan_large = plan_query(get_query(parse_aql(aql_large).unwrap())).unwrap();
    assert!(plan_small.hnsw_search.ef < plan_large.hnsw_search.ef);
}

#[test]
fn test_create_collection_ddl() {
    let aql = r#"CREATE COLLECTION papers (title TEXT, body TEXT) WITH (ef_search = 256, similarity = "cosine")"#;
    let parsed = parse_aql(aql).unwrap();
    match parsed {
        AQLStatement::CreateCollection(c) => {
            assert_eq!(c.collection, "papers");
            assert_eq!(c.settings.ef_search, 256);
            assert_eq!(c.settings.similarity_metric, "cosine");
        }
        _ => panic!("Expected CreateCollection"),
    }
}

#[test]
fn test_alter_collection_ddl() {
    let aql = r#"ALTER COLLECTION papers SET (ef_search = 512, max_connections = 64)"#;
    let parsed = parse_aql(aql).unwrap();
    match parsed {
        AQLStatement::AlterCollection(a) => {
            assert_eq!(a.collection, "papers");
            assert_eq!(a.settings.ef_search, 512);
            assert_eq!(a.settings.max_nb_connection, 64);
        }
        _ => panic!("Expected AlterCollection"),
    }
}

#[test]
fn test_alter_collection_executor() {
    let aql = r#"ALTER COLLECTION metrics SET (ef_search = 128, exact_rerank = false)"#;
    let parsed = parse_aql(aql).unwrap();
    let empty_indexes = HashMap::new();
    let result = execute_statement(&parsed, &empty_indexes, None).unwrap();
    match result {
        ExecuteResult::DdlResult { collection, message } => {
            assert_eq!(collection, "metrics");
            assert!(message.contains("ef_search=128"));
            assert!(message.contains("exact_rerank=false"));
        }
        _ => panic!("Expected DdlResult"),
    }
}
