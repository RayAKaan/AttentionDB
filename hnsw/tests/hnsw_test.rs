use attentiondb_hnsw::{HNSWIndex, HeadIndexManager, HNSWConfig};

#[test]
fn test_insert_basic() {
    let mut index = HNSWIndex::new("test", 8, HNSWConfig::default());
    let vec = vec![0.1; 8];
    index.insert(1, &vec).unwrap();
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
}

#[test]
fn test_search_returns_results() {
    let mut index = HNSWIndex::new("test", 8, HNSWConfig::default());
    for i in 0..100 {
        let vec: Vec<f32> = (0..8).map(|x| ((i + x) as f32) * 0.1).collect();
        index.insert(i, &vec).unwrap();
    }
    let query: Vec<f32> = (0..8).map(|x| (x as f32) * 0.1).collect();
    let results = index.search(&query, 5, Some(32)).unwrap();
    assert!(!results.is_empty());
    assert!(results.len() <= 5);
}

#[test]
fn test_search_empty_index() {
    let index = HNSWIndex::new("test", 8, HNSWConfig::default());
    let query = vec![0.0; 8];
    let result = index.search(&query, 5, None);
    assert!(result.is_err());
}

#[test]
fn test_rerank_exact() {
    let mut index = HNSWIndex::new("test", 4, HNSWConfig::default());
    index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = index.rerank_exact(&query, &[1, 2], 2).unwrap();
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_search_with_rerank() {
    let mut index = HNSWIndex::new("test", 8, HNSWConfig::default());
    for i in 0..50 {
        let vec: Vec<f32> = (0..8).map(|x| ((i + x) as f32) * 0.1).collect();
        index.insert(i, &vec).unwrap();
    }
    let query: Vec<f32> = (0..8).map(|x| (x as f32) * 0.1).collect();
    let results = index.search_with_rerank(&query, 5, Some(32)).unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn test_multi_head_fusion() {
    let manager = HeadIndexManager::new(8);
    manager.add_head("a");
    manager.add_head("b");

    for i in 0..50 {
        let vec: Vec<f32> = (0..8).map(|x| ((i + x) as f32) * 0.05).collect();
        manager.insert("a", i, &vec).unwrap();
        manager.insert("b", i + 100, &vec).unwrap();
    }

    let query: Vec<f32> = (0..8).map(|x| (x as f32) * 0.05).collect();
    let results = manager.search_multi(&["a", "b"], &query, 10, None).unwrap();
    assert!(!results.is_empty());
    assert!(results.len() <= 10);
}

#[test]
fn test_multi_head_weighted() {
    let manager = HeadIndexManager::new(8);
    manager.add_head("a");
    manager.add_head("b");

    for i in 0..30 {
        let vec: Vec<f32> = (0..8).map(|x| ((i + x) as f32) * 0.05).collect();
        manager.insert("a", i, &vec).unwrap();
        manager.insert("b", i + 100, &vec).unwrap();
    }

    let query: Vec<f32> = (0..8).map(|x| (x as f32) * 0.05).collect();
    let results = manager.search_multi_weighted(
        &[("a", 1.0), ("b", 0.5)], &query, 5, None,
    ).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_dimension_mismatch_error() {
    let mut index = HNSWIndex::new("test", 8, HNSWConfig::default());
    let bad_vec = vec![0.1; 4];
    let result = index.insert(1, &bad_vec);
    assert!(result.is_err());
}

#[test]
fn test_insert_and_search_tracking() {
    let mut index = HNSWIndex::new("test", 8, HNSWConfig::default());
    for i in 0..20 {
        let vec: Vec<f32> = (0..8).map(|x| ((i + x) as f32) * 0.1).collect();
        index.insert(i, &vec).unwrap();
    }
    assert!(index.len() > 0);
    let query = vec![0.1; 8];
    let results = index.search(&query, 5, None).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_save_load() {
    let mut index = HNSWIndex::new("test", 8, HNSWConfig::default());
    for i in 0..30 {
        let vec: Vec<f32> = (0..8).map(|x| ((i + x) as f32) * 0.1).collect();
        index.insert(i, &vec).unwrap();
    }

    let dir = std::env::temp_dir().join("test_hnsw_roundtrip");
    index.save(&dir).unwrap();

    let loaded = HNSWIndex::load(&dir).unwrap();
    assert_eq!(loaded.len(), 30);
    assert!(loaded.is_built);
}

#[test]
fn test_get_vector() {
    let mut index = HNSWIndex::new("test", 4, HNSWConfig::default());
    let vec = vec![0.1, 0.2, 0.3, 0.4];
    index.insert(42, &vec).unwrap();

    let retrieved = index.get_vector(42);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), &vec);
}

#[test]
fn test_list_and_count_heads() {
    let manager = HeadIndexManager::new(8);
    assert_eq!(manager.head_count(), 0);

    manager.add_head("a");
    manager.add_head("b");
    manager.add_head("c");

    assert_eq!(manager.head_count(), 3);
    assert_eq!(manager.list_heads().len(), 3);
}

#[test]
fn test_remove_head() {
    let manager = HeadIndexManager::new(8);
    manager.add_head("a");
    manager.add_head("b");
    manager.remove_head("a").unwrap();
    assert_eq!(manager.head_count(), 1);
    assert!(manager.get_head("a").is_err());
}

#[test]
fn test_insert_batch() {
    let mut index = HNSWIndex::new("test", 4, HNSWConfig::default());
    let batch: Vec<(u64, Vec<f32>)> = (0..10)
        .map(|i| (i, vec![i as f32 * 0.1; 4]))
        .collect();

    index.insert_batch(&batch).unwrap();
    assert_eq!(index.len(), 10);
}

#[test]
fn test_rerank_basic() {
    let mut index = HNSWIndex::new("test", 4, HNSWConfig::default());
    index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = index.rerank_exact(&query, &[1, 2], 2).unwrap();
    assert_eq!(results[0].0, 1);
}

#[test]
fn test_rerank_multi_candidate() {
    let mut index = HNSWIndex::new("test", 4, HNSWConfig::default());
    index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
    index.insert(3, &[0.0, 0.0, 1.0, 0.0]).unwrap();

    let query = vec![0.0, 0.0, 1.0, 0.0];
    let results = index.rerank_exact(&query, &[1, 2, 3], 3).unwrap();
    assert_eq!(results[0].0, 3);
    assert_eq!(results.len(), 3);
}

#[test]
fn test_rerank_empty_candidates() {
    let mut index = HNSWIndex::new("test", 4, HNSWConfig::default());
    index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = index.rerank_exact(&query, &[], 5).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_rerank_cpubackend_matches_cpu() {
    use attentiondb_hnsw::gpu::{CpuBackend, GpuBackend};
    let backend = CpuBackend;
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let candidates = vec![(1, vec![1.0, 0.0, 0.0, 0.0]), (2, vec![0.0, 1.0, 0.0, 0.0])];
    let results = backend.rerank_exact(&query, &candidates, 2).unwrap();
    assert_eq!(results[0].0, 1);
    assert_eq!(results[1].0, 2);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_gpu_cpu_backend_always_available() {
    let config = HNSWConfig::default();
    let mut index = HNSWIndex::new("test", 4, config);
    let query = vec![1.0, 0.0, 0.0, 0.0];
    index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    let results = index.rerank_exact(&query, &[1], 1).unwrap();
    assert_eq!(results.len(), 1);
}
