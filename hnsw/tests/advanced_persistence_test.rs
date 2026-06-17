use attentiondb_hnsw::{HNSWIndex, HNSWConfig};
use attentiondb_hnsw::persistence::{
    async_persistence::save_index_async,
    compaction::compact_index,
    backup::{create_backup, list_backups},
};
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

#[tokio::test]
async fn test_async_save() {
    let dir = tempdir().unwrap();
    let index = create_test_index("async_test", 8, 50);

    let result = save_index_async(&index, dir.path()).await;
    assert!(result.is_ok());

    assert!(dir.path().join("metadata.json").exists());
    assert!(dir.path().join("vectors.bin").exists());
}

#[test]
fn test_compaction() {
    let dir = tempdir().unwrap();
    let index = create_test_index("compact_test", 8, 100);
    index.save(dir.path()).unwrap();

    let result = compact_index(dir.path());
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 100);
}

#[test]
fn test_backup_creation() {
    let dir = tempdir().unwrap();
    let index = create_test_index("backup_test", 8, 40);
    index.save(dir.path()).unwrap();

    let backup_path = create_backup(dir.path());
    assert!(backup_path.is_ok());

    let backup_dir = backup_path.unwrap();
    assert!(backup_dir.exists());
    assert!(backup_dir.join("metadata.json").exists());
    assert!(backup_dir.join("vectors.bin").exists());
}

#[test]
fn test_list_backups() {
    let dir = tempdir().unwrap();
    let index = create_test_index("backup_list_test", 8, 30);
    index.save(dir.path()).unwrap();

    let _ = create_backup(dir.path());
    std::thread::sleep(std::time::Duration::from_secs(1));
    let _ = create_backup(dir.path());

    let backups = list_backups(dir.path());
    assert!(backups.is_ok());
    assert!(backups.unwrap().len() >= 2);
}
