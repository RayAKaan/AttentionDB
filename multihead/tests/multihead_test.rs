use attentiondb_multihead::{
    fuse_scores, normalize_scores, HeadConfig, HeadType, MultiHeadManager,
};

fn make_test_manager() -> MultiHeadManager {
    let mut m = MultiHeadManager::new(8, 3);
    m.add_head(HeadConfig::new("semantic", HeadType::Semantic, 8).with_weight(1.0));
    m.add_head(HeadConfig::new("temporal", HeadType::Temporal, 8).with_weight(0.8));
    m.add_head(HeadConfig::new("structural", HeadType::Structural, 8).with_weight(0.6));
    m
}

#[test]
fn test_add_and_list_heads() {
    let mut manager = MultiHeadManager::new(8, 2);
    assert_eq!(manager.head_count(), 0);
    manager.add_head(HeadConfig::new("a", HeadType::Semantic, 8));
    manager.add_head(HeadConfig::new("b", HeadType::Temporal, 8));
    assert_eq!(manager.head_count(), 2);
    let mut heads = manager.list_heads();
    heads.sort();
    assert_eq!(heads, vec!["a", "b"]);
}

#[test]
fn test_get_head() {
    let mut manager = MultiHeadManager::new(8, 2);
    manager.add_head(HeadConfig::new("test", HeadType::Semantic, 8));
    let head = manager.get_head("test").unwrap();
    assert_eq!(head.head_type, HeadType::Semantic);
}

#[test]
fn test_get_head_not_found() {
    let manager = MultiHeadManager::new(8, 2);
    assert!(manager.get_head("nonexistent").is_err());
}

#[test]
fn test_fuse_basic() {
    let manager = make_test_manager();
    let query_emb = vec![0.1; 8];
    let head_results = vec![
        ("semantic".to_string(), vec![(1, 0.9)]),
        ("temporal".to_string(), vec![(2, 0.8)]),
        ("structural".to_string(), vec![(3, 0.7)]),
    ];
    let fused = manager.fuse(&query_emb, &head_results).unwrap();
    assert!(!fused.is_empty());
}

#[test]
fn test_fuse_dimension_mismatch() {
    let manager = make_test_manager();
    let bad_query = vec![0.1; 4]; // wrong dim
    let head_results = vec![("semantic".to_string(), vec![(1, 0.9)])];
    let result = manager.fuse(&bad_query, &head_results);
    assert!(result.is_err());
}

#[test]
fn test_fuse_scores_aggregates() {
    let results = vec![
        ("a".to_string(), vec![(1, 1.0), (2, 0.5)]),
        ("b".to_string(), vec![(2, 0.5), (3, 1.0)]),
    ];
    let gates = vec![0.5, 0.5];
    let fused = fuse_scores(&results, &gates);
    // ID 2: 0.5*0.5 + 0.5*0.5 = 0.5
    let id2 = fused.iter().find(|(id, _)| *id == 2).unwrap();
    assert!((id2.1 - 0.5).abs() < 1e-5);
}

#[test]
fn test_normalize_scores_empty() {
    let mut scores: Vec<(u64, f32)> = vec![];
    normalize_scores(&mut scores);
    assert!(scores.is_empty());
}

#[test]
fn test_normalize_scores_single() {
    let mut scores = vec![(1, 0.5)];
    normalize_scores(&mut scores);
    assert!((scores[0].1 - 1.0).abs() < 1e-5);
}

#[test]
fn test_head_config_builder() {
    let config = HeadConfig::new("custom", HeadType::Custom("test".into()), 64)
        .with_weight(0.5)
        .with_fields(vec!["field1".into()]);
    assert_eq!(config.name, "custom");
    assert!((config.weight - 0.5).abs() < 1e-5);
    assert_eq!(config.fields, vec!["field1"]);
}

#[test]
fn test_get_head_weights() {
    let manager = make_test_manager();
    let weights = manager.get_head_weights();
    assert!((weights["semantic"] - 1.0).abs() < 1e-5);
    assert!((weights["temporal"] - 0.8).abs() < 1e-5);
}

#[test]
fn test_fuse_weighted() {
    let manager = make_test_manager();
    let head_results = vec![
        ("semantic".to_string(), vec![(1, 0.9)]),
        ("temporal".to_string(), vec![(2, 0.8)]),
    ];
    let explicit = vec![
        ("semantic".to_string(), 1.0),
        ("temporal".to_string(), 0.0), // zero out temporal
    ];
    let fused = manager.fuse_weighted(&head_results, &explicit);
    // First result should be semantic (score 0.9*1.0=0.9)
    assert_eq!(fused[0].0, 1);
}

#[test]
fn test_multiple_ids_same_head() {
    let results = vec![("semantic".to_string(), vec![(1, 0.9), (2, 0.8), (3, 0.7)])];
    let gates = vec![1.0];
    let fused = fuse_scores(&results, &gates);
    assert_eq!(fused.len(), 3);
    assert_eq!(fused[0].0, 1); // highest score first
}
