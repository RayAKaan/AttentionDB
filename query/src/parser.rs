use pest::Parser;
use pest_derive::Parser;
use crate::error::QueryError;

#[derive(Parser)]
#[grammar = "aql.pest"]
pub struct AQLParser;

#[derive(Debug, Clone)]
pub struct AQLQuery {
    pub collection: String,
    pub query_text: String,
    pub heads: Vec<String>,
    pub top_k: usize,
    pub min_weight: f32,
    pub temporal_decay: Option<f32>,
    pub exact_filters: Vec<String>,
}

pub fn parse_aql(input: &str) -> Result<AQLQuery, QueryError> {
    let mut pairs = AQLParser::parse(Rule::query, input)
        .map_err(|e| QueryError::Parse(e.to_string()))?;

    let mut query = AQLQuery {
        collection: String::new(),
        query_text: String::new(),
        heads: vec![],
        top_k: 10,
        min_weight: 0.01,
        temporal_decay: None,
        exact_filters: vec![],
    };

    // The top-level Pairs iterator yields the query rule pair; iterate its inner pairs.
    if let Some(query_pair) = pairs.next() {
        for pair in query_pair.into_inner() {
            match pair.as_rule() {
                Rule::identifier => {
                    if query.collection.is_empty() {
                        query.collection = pair.as_str().to_string();
                    }
                }
                Rule::query_text => {
                    let s = pair.as_str();
                    query.query_text = s[1..s.len()-1].to_string();
                }
                Rule::heads => {
                    for head in pair.into_inner() {
                        if head.as_rule() == Rule::identifier {
                            query.heads.push(head.as_str().to_string());
                        }
                    }
                }
                Rule::top_k => query.top_k = pair.as_str().parse().unwrap_or(10),
                Rule::min_weight => query.min_weight = pair.as_str().parse().unwrap_or(0.01),
                Rule::temporal_decay => query.temporal_decay = pair.as_str().parse().ok(),
                _ => {}
            }
        }
    }

    if query.heads.is_empty() {
        query.heads = vec!["default".to_string()];
    }

    if query.collection.is_empty() {
        return Err(QueryError::Parse("Missing collection name".into()));
    }

    Ok(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let aql = r#"ATTEND TO papers WHERE QUERY "attention" HEADS [semantic] TOP_K 5"#;
        let q = parse_aql(aql).unwrap();
        assert_eq!(q.collection, "papers");
        assert_eq!(q.query_text, "attention");
        assert_eq!(q.heads, vec!["semantic"]);
        assert_eq!(q.top_k, 5);
    }

    #[test]
    fn test_parse_multi_head() {
        let aql = r#"ATTEND TO docs WHERE QUERY "test" HEADS [semantic, temporal] TOP_K 10"#;
        let q = parse_aql(aql).unwrap();
        assert_eq!(q.heads, vec!["semantic", "temporal"]);
    }

    #[test]
    fn test_parse_default_head() {
        let aql = r#"ATTEND TO docs WHERE QUERY "hello" TOP_K 3"#;
        let q = parse_aql(aql).unwrap();
        assert_eq!(q.heads, vec!["default"]);
    }

    #[test]
    fn test_parse_min_weight() {
        let aql = r#"ATTEND TO docs WHERE QUERY "x" MIN_WEIGHT 0.05"#;
        let q = parse_aql(aql).unwrap();
        assert!((q.min_weight - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_parse_temporal_decay() {
        let aql = r#"ATTEND TO docs WHERE QUERY "x" TEMPORAL_DECAY 0.3"#;
        let q = parse_aql(aql).unwrap();
        assert!((q.temporal_decay.unwrap() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_parse_missing_collection() {
        let aql = r#"WHERE QUERY "x""#;
        let result = parse_aql(aql);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_full_query() {
        let aql = r#"ATTEND TO papers WHERE QUERY "attention mechanisms in transformers" HEADS [semantic, temporal, structural] TOP_K 20 MIN_WEIGHT 0.05 TEMPORAL_DECAY 0.3"#;
        let q = parse_aql(aql).unwrap();
        assert_eq!(q.collection, "papers");
        assert_eq!(q.query_text, "attention mechanisms in transformers");
        assert_eq!(q.heads.len(), 3);
        assert_eq!(q.top_k, 20);
        assert!((q.min_weight - 0.05).abs() < 1e-6);
        assert!((q.temporal_decay.unwrap() - 0.3).abs() < 1e-6);
    }
}

