use attentiondb_hnsw::persistence::{
    async_compaction::compact_index_async,
    remote_backup::{download_backup, upload_backup},
};
use attentiondb_hnsw::{HNSWConfig, HNSWIndex};
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
async fn test_async_compaction() {
    let dir = tempdir().unwrap();
    let index = create_test_index("async_compact_test", 8, 80);
    index.save(dir.path()).unwrap();

    let result = compact_index_async(dir.path()).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 80);
}

#[tokio::test]
async fn test_remote_backup_placeholder() {
    let dir = tempdir().unwrap();
    let index = create_test_index("remote_backup_test", 8, 30);
    index.save(dir.path()).unwrap();

    let result = upload_backup(dir.path(), "http://localhost:9999/backup").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_remote_download_placeholder() {
    let dir = tempdir().unwrap();

    let result = download_backup("http://localhost:9999/backup", dir.path()).await;
    assert!(result.is_err());
}
