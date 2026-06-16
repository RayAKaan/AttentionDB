use attentiondb_hnsw::{HNSWIndex, HNSWConfig};
use tempfile::tempdir;

fn create_test_index(head_name: &str, dim: usize, count: usize) -> HNSWIndex {
    let config = HNSWConfig {
        max_nb_connection: 8,
        ef_construction: 100,
        ef_search: 32,
        store_vectors: true,
        max_elements: 10_000,
    };

    let mut index = HNSWIndex::new(head_name, dim, config);

    for i in 0..count {
        let vec: Vec<f32> = (0..dim).map(|x| ((i + x) as f32).sin() * 0.1).collect();
        let _ = index.insert(i as u64, &vec);
    }

    index
}

#[test]
fn test_load_with_progress_reports() {
    let dir = tempdir().unwrap();
    let index = create_test_index("progress_test", 8, 2500);
    index.save(dir.path()).unwrap();

    let mut progress_calls = Vec::new();

    let loaded = HNSWIndex::load_with_progress(dir.path(), |p| {
        progress_calls.push((p.loaded_vectors, p.total_vectors));
    }).unwrap();

    assert_eq!(loaded.len(), 2500);
    assert!(!progress_calls.is_empty());

    let last = progress_calls.last().unwrap();
    assert_eq!(last.1, 2500);
}

#[test]
fn test_version_migration_v1_to_v2() {
    let dir = tempdir().unwrap();
    let index = create_test_index("migration_test", 8, 100);
    index.save(dir.path()).unwrap();

    let meta_path = dir.path().join("metadata.json");
    let mut meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();

    meta["version"] = serde_json::json!(1);
    if meta.get("checksum").is_none() {
        meta["checksum"] = serde_json::json!("legacy");
    }

    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap()).unwrap();

    let loaded = HNSWIndex::load(dir.path());
    assert!(loaded.is_ok());

    let loaded_index = loaded.unwrap();
    assert_eq!(loaded_index.len(), 100);
}

#[test]
fn test_load_with_progress_multiple_callbacks() {
    let dir = tempdir().unwrap();
    let index = create_test_index("multi_progress", 8, 5000);
    index.save(dir.path()).unwrap();

    let mut first_callback = Vec::new();
    let mut second_callback = Vec::new();

    let _ = HNSWIndex::load_with_progress(dir.path(), |p| {
        first_callback.push(p.loaded_vectors);
    });

    let _ = HNSWIndex::load_with_progress(dir.path(), |p| {
        second_callback.push(p.loaded_vectors);
    });

    assert!(!first_callback.is_empty());
    assert!(!second_callback.is_empty());
}
