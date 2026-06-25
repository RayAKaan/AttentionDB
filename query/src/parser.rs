use crate::error::QueryError;
use pest::Parser;
use pest_derive::Parser;
use std::collections::HashMap;

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

#[derive(Debug, Clone)]
pub struct CreateCollection {
    pub collection: String,
    pub fields: Vec<(String, String)>,
    pub settings: attentiondb_hnsw::CollectionSettings,
    pub head_settings: HashMap<String, attentiondb_hnsw::CollectionSettings>,
}

#[derive(Debug, Clone)]
pub struct AlterCollection {
    pub collection: String,
    pub settings: attentiondb_hnsw::CollectionSettings,
    pub head_settings: HashMap<String, attentiondb_hnsw::CollectionSettings>,
}

#[derive(Debug, Clone)]
pub enum AQLStatement {
    Query(AQLQuery),
    CreateCollection(CreateCollection),
    AlterCollection(AlterCollection),
}

fn parse_collection_settings(
    raw: &[(&str, String)],
) -> Result<attentiondb_hnsw::CollectionSettings, QueryError> {
    let mut settings = attentiondb_hnsw::CollectionSettings::default();

    for (key, val) in raw {
        match *key {
            "ef_search" => {
                settings.ef_search = val
                    .parse()
                    .map_err(|_| QueryError::Parse(format!("Invalid ef_search value: {}", val)))?;
            }
            "ef_construction" => {
                settings.ef_construction = val.parse().map_err(|_| {
                    QueryError::Parse(format!("Invalid ef_construction value: {}", val))
                })?;
            }
            "max_connections" => {
                settings.max_nb_connection = val.parse().map_err(|_| {
                    QueryError::Parse(format!("Invalid max_connections value: {}", val))
                })?;
            }
            "similarity" => {
                settings.similarity_metric = val.clone();
            }
            "exact_rerank" => {
                settings.enable_exact_reranking = val.parse().map_err(|_| {
                    QueryError::Parse(format!("Invalid exact_rerank value: {}", val))
                })?;
            }
            "enable_gpu_fusion" => {
                settings.enable_gpu_fusion = val.parse().map_err(|_| {
                    QueryError::Parse(format!("Invalid enable_gpu_fusion value: {}", val))
                })?;
            }
            _ => {
                return Err(QueryError::Parse(format!("Unknown setting: {}", key)));
            }
        }
    }

    settings
        .validate()
        .map_err(|e| QueryError::Parse(format!("Invalid collection settings: {}", e)))?;

    Ok(settings)
}

fn parse_head_settings(
    raw: &[(&str, String)],
) -> Result<HashMap<String, attentiondb_hnsw::CollectionSettings>, QueryError> {
    let mut head_map: HashMap<String, Vec<(&str, String)>> = HashMap::new();
    for (key, val) in raw {
        let dot = key.find('.').ok_or_else(|| {
            QueryError::Parse(format!(
                "Expected head-qualified setting key (head_name.key), got: {}",
                key
            ))
        })?;
        let head_name = &key[..dot];
        let setting_key = &key[dot + 1..];
        head_map
            .entry(head_name.to_string())
            .or_default()
            .push((setting_key, val.clone()));
    }

    let mut result = HashMap::new();
    for (head_name, head_raw) in head_map {
        let settings = parse_collection_settings(&head_raw)?;
        result.insert(head_name, settings);
    }
    Ok(result)
}

pub fn parse_aql(input: &str) -> Result<AQLStatement, QueryError> {
    let mut pairs =
        AQLParser::parse(Rule::query, input).map_err(|e| QueryError::Parse(e.to_string()))?;

    let query_pair = pairs
        .next()
        .ok_or_else(|| QueryError::Parse("Empty input".into()))?;
    let inner = query_pair
        .into_inner()
        .next()
        .ok_or_else(|| QueryError::Parse("No statement found".into()))?;

    match inner.as_rule() {
        Rule::attend => Ok(AQLStatement::Query(parse_attend(inner))),
        Rule::create_collection => Ok(AQLStatement::CreateCollection(parse_create_collection(
            inner,
        )?)),
        Rule::alter_collection => Ok(AQLStatement::AlterCollection(parse_alter_collection(
            inner,
        )?)),
        _ => Err(QueryError::Parse(format!(
            "Unexpected rule: {:?}",
            inner.as_rule()
        ))),
    }
}

fn parse_attend(pair: pest::iterators::Pair<Rule>) -> AQLQuery {
    let mut query = AQLQuery {
        collection: String::new(),
        query_text: String::new(),
        heads: vec![],
        top_k: 10,
        min_weight: 0.01,
        temporal_decay: None,
        exact_filters: vec![],
    };

    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::identifier => {
                if query.collection.is_empty() {
                    query.collection = pair.as_str().to_string();
                }
            }
            Rule::query_text => {
                let s = pair.as_str();
                query.query_text = s[1..s.len() - 1].to_string();
            }
            Rule::heads => {
                for head in pair.into_inner() {
                    if head.as_rule() == Rule::identifier {
                        query.heads.push(head.as_str().to_string());
                    }
                }
            }
            Rule::top_k_value => query.top_k = pair.as_str().parse().unwrap_or(10),
            Rule::min_weight_value => query.min_weight = pair.as_str().parse().unwrap_or(0.01),
            Rule::temporal_decay_value => query.temporal_decay = pair.as_str().parse().ok(),
            _ => {}
        }
    }

    if query.heads.is_empty() {
        query.heads = vec!["default".to_string()];
    }

    query
}

fn parse_create_collection(
    pair: pest::iterators::Pair<Rule>,
) -> Result<CreateCollection, QueryError> {
    let mut collection = String::new();
    let mut fields = Vec::new();
    let mut raw_settings: Vec<(String, String)> = Vec::new();

    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::identifier => {
                if collection.is_empty() {
                    collection = pair.as_str().to_string();
                }
            }
            Rule::field_list => {
                for field_def in pair.into_inner() {
                    if field_def.as_rule() == Rule::field_def {
                        let mut parts = field_def.into_inner();
                        let name = parts
                            .next()
                            .map(|p| p.as_str().to_string())
                            .unwrap_or_default();
                        let ty = parts
                            .next()
                            .map(|p| p.as_str().to_string())
                            .unwrap_or_default();
                        fields.push((name, ty));
                    }
                }
            }
            Rule::setting_pair => {
                let mut parts = pair.into_inner();
                let key = parts
                    .next()
                    .map(|p| p.as_str().trim().to_string())
                    .unwrap_or_default();
                let raw_val = parts.next().map(|p| p.as_str()).unwrap_or_default();
                let val = if raw_val.starts_with('"') && raw_val.ends_with('"') {
                    raw_val[1..raw_val.len() - 1].to_string()
                } else {
                    raw_val.to_string()
                };
                raw_settings.push((key, val));
            }
            _ => {}
        }
    }

    let (head_settings, collection_settings): (Vec<_>, Vec<_>) = raw_settings
        .into_iter()
        .partition(|(key, _)| key.contains('.'));

    let settings = parse_collection_settings(
        &collection_settings
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect::<Vec<_>>(),
    )?;
    let head_settings = parse_head_settings(
        &head_settings
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect::<Vec<_>>(),
    )?;

    Ok(CreateCollection {
        collection,
        fields,
        settings,
        head_settings,
    })
}

fn parse_alter_collection(
    pair: pest::iterators::Pair<Rule>,
) -> Result<AlterCollection, QueryError> {
    let mut collection = String::new();
    let mut raw_settings: Vec<(String, String)> = Vec::new();

    for pair in pair.into_inner() {
        match pair.as_rule() {
            Rule::identifier => {
                if collection.is_empty() {
                    collection = pair.as_str().to_string();
                }
            }
            Rule::setting_pair => {
                let mut parts = pair.into_inner();
                let key = parts
                    .next()
                    .map(|p| p.as_str().trim().to_string())
                    .unwrap_or_default();
                let raw_val = parts.next().map(|p| p.as_str()).unwrap_or_default();
                let val = if raw_val.starts_with('"') && raw_val.ends_with('"') {
                    raw_val[1..raw_val.len() - 1].to_string()
                } else {
                    raw_val.to_string()
                };
                raw_settings.push((key, val));
            }
            _ => {}
        }
    }

    let (head_settings, collection_settings): (Vec<_>, Vec<_>) = raw_settings
        .into_iter()
        .partition(|(key, _)| key.contains('.'));

    let settings = parse_collection_settings(
        &collection_settings
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect::<Vec<_>>(),
    )?;
    let head_settings = parse_head_settings(
        &head_settings
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect::<Vec<_>>(),
    )?;

    Ok(AlterCollection {
        collection,
        settings,
        head_settings,
    })
}

/// Extract per-head settings from a parsed AQL statement.
/// Returns a map of head_name -> CollectionSettings.
pub fn get_per_head_settings(
    statement: &AQLStatement,
) -> HashMap<String, attentiondb_hnsw::CollectionSettings> {
    match statement {
        AQLStatement::CreateCollection(c) => c.head_settings.clone(),
        AQLStatement::AlterCollection(a) => a.head_settings.clone(),
        _ => HashMap::new(),
    }
}

/// Check whether a parsed AQL statement contains any per-head settings.
pub fn has_per_head_settings(statement: &AQLStatement) -> bool {
    match statement {
        AQLStatement::CreateCollection(c) => !c.head_settings.is_empty(),
        AQLStatement::AlterCollection(a) => !a.head_settings.is_empty(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let aql = r#"ATTEND TO papers WHERE QUERY "attention" HEADS [semantic] TOP_K 5"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::Query(q) => {
                assert_eq!(q.collection, "papers");
                assert_eq!(q.query_text, "attention");
                assert_eq!(q.heads, vec!["semantic"]);
                assert_eq!(q.top_k, 5);
            }
            _ => panic!("Expected Query"),
        }
    }

    #[test]
    fn test_parse_multi_head() {
        let aql = r#"ATTEND TO docs WHERE QUERY "test" HEADS [semantic, temporal] TOP_K 10"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::Query(q) => assert_eq!(q.heads, vec!["semantic", "temporal"]),
            _ => panic!("Expected Query"),
        }
    }

    #[test]
    fn test_parse_default_head() {
        let aql = r#"ATTEND TO docs WHERE QUERY "hello" TOP_K 3"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::Query(q) => assert_eq!(q.heads, vec!["default"]),
            _ => panic!("Expected Query"),
        }
    }

    #[test]
    fn test_parse_min_weight() {
        let aql = r#"ATTEND TO docs WHERE QUERY "x" MIN_WEIGHT 0.05"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::Query(q) => assert!((q.min_weight - 0.05).abs() < 1e-6),
            _ => panic!("Expected Query"),
        }
    }

    #[test]
    fn test_parse_temporal_decay() {
        let aql = r#"ATTEND TO docs WHERE QUERY "x" TEMPORAL_DECAY 0.3"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::Query(q) => assert!((q.temporal_decay.unwrap() - 0.3).abs() < 1e-6),
            _ => panic!("Expected Query"),
        }
    }

    #[test]
    fn test_parse_full_query() {
        let aql = r#"ATTEND TO papers WHERE QUERY "attention mechanisms" HEADS [semantic, temporal, structural] TOP_K 20 MIN_WEIGHT 0.05 TEMPORAL_DECAY 0.3"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::Query(q) => {
                assert_eq!(q.collection, "papers");
                assert_eq!(q.query_text, "attention mechanisms");
                assert_eq!(q.heads.len(), 3);
                assert_eq!(q.top_k, 20);
                assert!((q.min_weight - 0.05).abs() < 1e-6);
                assert!((q.temporal_decay.unwrap() - 0.3).abs() < 1e-6);
            }
            _ => panic!("Expected Query"),
        }
    }

    #[test]
    fn test_create_collection_basic() {
        let aql = r#"CREATE COLLECTION papers (title TEXT, body TEXT)"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::CreateCollection(c) => {
                assert_eq!(c.collection, "papers");
                assert_eq!(c.fields.len(), 2);
                assert_eq!(c.fields[0], ("title".to_string(), "TEXT".to_string()));
                assert_eq!(c.fields[1], ("body".to_string(), "TEXT".to_string()));
                assert_eq!(c.settings.ef_search, 64);
            }
            _ => panic!("Expected CreateCollection"),
        }
    }

    #[test]
    fn test_create_collection_with_settings() {
        let aql = r#"CREATE COLLECTION papers (title TEXT) WITH (ef_search = 256, similarity = "cosine")"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::CreateCollection(c) => {
                assert_eq!(c.collection, "papers");
                assert_eq!(c.fields.len(), 1);
                assert_eq!(c.settings.ef_search, 256);
                assert_eq!(c.settings.similarity_metric, "cosine");
            }
            _ => panic!("Expected CreateCollection"),
        }
    }

    #[test]
    fn test_create_collection_empty_fields() {
        let aql = r#"CREATE COLLECTION metrics () WITH (ef_search = 128)"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::CreateCollection(c) => {
                assert_eq!(c.collection, "metrics");
                assert!(c.fields.is_empty());
                assert_eq!(c.settings.ef_search, 128);
            }
            _ => panic!("Expected CreateCollection"),
        }
    }

    #[test]
    fn test_alter_collection_basic() {
        let aql = r#"ALTER COLLECTION papers SET (ef_search = 256)"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::AlterCollection(a) => {
                assert_eq!(a.collection, "papers");
                assert_eq!(a.settings.ef_search, 256);
            }
            _ => panic!("Expected AlterCollection"),
        }
    }

    #[test]
    fn test_alter_collection_multiple_settings() {
        let aql = r#"ALTER COLLECTION metrics SET (ef_search = 128, ef_construction = 600, max_connections = 32)"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::AlterCollection(a) => {
                assert_eq!(a.collection, "metrics");
                assert_eq!(a.settings.ef_search, 128);
                assert_eq!(a.settings.ef_construction, 600);
                assert_eq!(a.settings.max_nb_connection, 32);
            }
            _ => panic!("Expected AlterCollection"),
        }
    }

    #[test]
    fn test_alter_collection_with_string_settings() {
        let aql =
            r#"ALTER COLLECTION papers SET (similarity = "dot_product", exact_rerank = true)"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::AlterCollection(a) => {
                assert_eq!(a.collection, "papers");
                assert_eq!(a.settings.similarity_metric, "dot_product");
                assert_eq!(a.settings.enable_exact_reranking, true);
            }
            _ => panic!("Expected AlterCollection"),
        }
    }

    #[test]
    fn test_create_collection_with_per_head_settings() {
        let aql = r#"CREATE COLLECTION papers (title TEXT) WITH (ef_search = 256, semantic.ef_search = 128, temporal.ef_search = 64)"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::CreateCollection(c) => {
                assert_eq!(c.collection, "papers");
                assert_eq!(c.settings.ef_search, 256);
                assert_eq!(c.settings.similarity_metric, "cosine");
                assert_eq!(c.head_settings.len(), 2);
                assert_eq!(c.head_settings.get("semantic").unwrap().ef_search, 128);
                assert_eq!(c.head_settings.get("temporal").unwrap().ef_search, 64);
            }
            _ => panic!("Expected CreateCollection"),
        }
    }

    #[test]
    fn test_alter_collection_with_per_head_settings() {
        let aql = r#"ALTER COLLECTION papers SET (ef_search = 512, semantic.ef_search = 256, temporal.ef_search = 128)"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::AlterCollection(a) => {
                assert_eq!(a.collection, "papers");
                assert_eq!(a.settings.ef_search, 512);
                assert_eq!(a.head_settings.len(), 2);
                assert_eq!(a.head_settings.get("semantic").unwrap().ef_search, 256);
                assert_eq!(a.head_settings.get("temporal").unwrap().ef_search, 128);
            }
            _ => panic!("Expected AlterCollection"),
        }
    }

    #[test]
    fn test_per_head_settings_only() {
        let aql = r#"CREATE COLLECTION papers (title TEXT) WITH (semantic.ef_search = 128, temporal.ef_search = 64)"#;
        let stmt = parse_aql(aql).unwrap();
        match stmt {
            AQLStatement::CreateCollection(c) => {
                assert_eq!(c.collection, "papers");
                assert_eq!(c.settings.ef_search, 64); // default
                assert_eq!(c.head_settings.len(), 2);
                assert_eq!(c.head_settings.get("semantic").unwrap().ef_search, 128);
                assert_eq!(c.head_settings.get("temporal").unwrap().ef_search, 64);
            }
            _ => panic!("Expected CreateCollection"),
        }
    }
}
