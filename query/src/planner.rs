use crate::parser::AQLQuery;
use crate::error::QueryError;

#[derive(Debug, Clone)]
pub struct LogicalPlan {
    pub collection: String,
    pub query_text: String,
    pub heads: Vec<String>,
    pub top_k: usize,
    pub min_weight: f32,
    pub temporal_decay: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct PhysicalPlan {
    pub hnsw_search: HNSWSearchStep,
    pub exact_rerank: Option<ExactRerankStep>,
    pub filter_steps: Vec<FilterStep>,
    pub top_k: usize,
    pub min_weight: f32,
}

#[derive(Debug, Clone)]
pub struct HNSWSearchStep {
    pub heads: Vec<(String, f32)>, // (head_name, weight)
    pub ef: usize,
    pub k: usize,
}

#[derive(Debug, Clone)]
pub struct ExactRerankStep {
    pub top_candidates: usize,
}

#[derive(Debug, Clone)]
pub struct FilterStep {
    pub filter_type: String,
    pub expression: String,
}

/// Convert parsed AQL into a logical plan
pub fn build_logical(query: AQLQuery) -> LogicalPlan {
    LogicalPlan {
        collection: query.collection,
        query_text: query.query_text,
        heads: query.heads,
        top_k: query.top_k,
        min_weight: query.min_weight,
        temporal_decay: query.temporal_decay,
    }
}

/// Optimize logical plan into a physical plan
pub fn plan_query(query: AQLQuery) -> Result<PhysicalPlan, QueryError> {
    let logical = build_logical(query);

    if logical.collection.is_empty() {
        return Err(QueryError::Planning("Collection name is required".into()));
    }

    // Default ef based on top_k
    let ef = match logical.top_k {
        0..=5 => 32,
        6..=20 => 64,
        _ => 128,
    };

    let overfetch = logical.top_k * 3;

    // Assign weights — temporal head gets decay weight if present
    let heads: Vec<(String, f32)> = if let Some(decay) = logical.temporal_decay {
        logical.heads.iter().map(|h| {
            let w = if h == "temporal" { decay } else { 1.0 };
            (h.clone(), w)
        }).collect()
    } else {
        logical.heads.iter().map(|h| (h.clone(), 1.0)).collect()
    };

    Ok(PhysicalPlan {
        hnsw_search: HNSWSearchStep {
            heads,
            ef,
            k: overfetch,
        },
        exact_rerank: Some(ExactRerankStep {
            top_candidates: overfetch,
        }),
        filter_steps: vec![],
        top_k: logical.top_k,
        min_weight: logical.min_weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_aql;

    #[test]
    fn test_basic_planning() {
        let aql = r#"ATTEND TO papers WHERE QUERY "attention" HEADS [semantic] TOP_K 5"#;
        let parsed = parse_aql(aql).unwrap();
        let plan = plan_query(parsed).unwrap();
        assert_eq!(plan.top_k, 5);
        assert_eq!(plan.hnsw_search.ef, 32);
        assert!(plan.exact_rerank.is_some());
    }

    #[test]
    fn test_plan_validates_collection() {
        let parsed = AQLQuery {
            collection: String::new(),
            query_text: "test".into(),
            heads: vec!["semantic".into()],
            top_k: 10,
            min_weight: 0.01,
            temporal_decay: None,
            exact_filters: vec![],
        };
        let result = plan_query(parsed);
        assert!(result.is_err());
    }

    #[test]
    fn test_temporal_decay_weight() {
        let aql = r#"ATTEND TO papers WHERE QUERY "test" HEADS [semantic, temporal] TEMPORAL_DECAY 0.3"#;
        let parsed = parse_aql(aql).unwrap();
        let plan = plan_query(parsed).unwrap();
        let temporal_weight = plan.hnsw_search.heads.iter()
            .find(|(n, _)| n == "temporal")
            .map(|(_, w)| *w);
        assert!((temporal_weight.unwrap() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_ef_scales_with_top_k() {
        let aql_small = r#"ATTEND TO papers WHERE QUERY "test" TOP_K 3"#;
        let aql_med = r#"ATTEND TO papers WHERE QUERY "test" TOP_K 15"#;
        let aql_large = r#"ATTEND TO papers WHERE QUERY "test" TOP_K 50"#;

        assert_eq!(plan_query(parse_aql(aql_small).unwrap()).unwrap().hnsw_search.ef, 32);
        assert_eq!(plan_query(parse_aql(aql_med).unwrap()).unwrap().hnsw_search.ef, 64);
        assert_eq!(plan_query(parse_aql(aql_large).unwrap()).unwrap().hnsw_search.ef, 128);
    }
}
