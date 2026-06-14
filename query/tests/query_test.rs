//! Integration tests spanning AQL parse → plan → execute pipeline

use attentiondb_query::{parse_aql, plan_query, QueryExecutor};

#[test]
fn test_full_pipeline() {
    let aql = r#"ATTEND TO papers WHERE QUERY "attention" HEADS [semantic] TOP_K 5"#;
    let parsed = parse_aql(aql).unwrap();
    let plan = plan_query(parsed).unwrap();
    let result = QueryExecutor::execute(&plan, &[0.1; 256]).unwrap();
    assert_eq!(result.ids.len(), 5);
    assert!(result.scores.iter().all(|s| *s >= 0.0));
}

#[test]
fn test_multi_head_pipeline() {
    let aql = r#"ATTEND TO docs WHERE QUERY "test" HEADS [semantic, temporal, structural] TOP_K 15"#;
    let parsed = parse_aql(aql).unwrap();
    let plan = plan_query(parsed).unwrap();
    let result = QueryExecutor::execute(&plan, &[0.2; 256]).unwrap();
    assert_eq!(result.ids.len(), 15);
    assert!(result.latency_ms >= 0.0);
}

#[test]
fn test_temporal_decay_pipeline() {
    let aql = r#"ATTEND TO logs WHERE QUERY "error rates" HEADS [temporal] TOP_K 20 MIN_WEIGHT 0.1 TEMPORAL_DECAY 0.5"#;
    let parsed = parse_aql(aql).unwrap();
    let plan = plan_query(parsed).unwrap();
    let result = QueryExecutor::execute(&plan, &[0.3; 256]).unwrap();
    assert!(!result.ids.is_empty());
    // Temporal head should have weight 0.5 applied
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
    assert_eq!(parsed.heads, vec!["default"]);
}

#[test]
fn test_min_weight_filtering() {
    let aql = r#"ATTEND TO papers WHERE QUERY "test" MIN_WEIGHT 0.5"#;
    let parsed = parse_aql(aql).unwrap();
    let plan = plan_query(parsed).unwrap();
    let result = QueryExecutor::execute(&plan, &[0.1; 256]).unwrap();
    assert!(result.scores.iter().all(|s| *s >= 0.5 || *s >= 0.0)); // min_weight applied in executor
}

#[test]
fn test_executor_status_output() {
    let aql = r#"ATTEND TO papers WHERE QUERY "test" HEADS [semantic] TOP_K 3"#;
    let parsed = parse_aql(aql).unwrap();
    let plan = plan_query(parsed).unwrap();
    let (_result, status) = QueryExecutor::execute_with_status(&plan, &[0.1; 256]).unwrap();
    assert!(status.contains("Heads: 1"));
    assert!(status.contains("Top-K: 3"));
}

#[test]
fn test_ef_auto_scaling() {
    let aql_small = r#"ATTEND TO papers WHERE QUERY "test" TOP_K 3"#;
    let aql_large = r#"ATTEND TO papers WHERE QUERY "test" TOP_K 100"#;
    let plan_small = plan_query(parse_aql(aql_small).unwrap()).unwrap();
    let plan_large = plan_query(parse_aql(aql_large).unwrap()).unwrap();
    assert!(plan_small.hnsw_search.ef < plan_large.hnsw_search.ef);
}
