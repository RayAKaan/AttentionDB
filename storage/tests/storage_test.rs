//! Basic storage layer tests (Phase 1: authoritative WAL, catalog, DocumentStore)

use attentiondb_storage::{Catalog, DocumentStore, Durability, Record, RecordKind, Wal, WalRecord};
use std::collections::HashMap;
use tempfile::tempdir;

#[test]
fn test_record_serialization() {
    let mut fields = HashMap::new();
    fields.insert("name".to_string(), serde_json::json!("Rayyan"));

    let record = Record::new(fields);
    let bytes = record.to_msgpack().unwrap();
    let restored = Record::from_msgpack(&bytes).unwrap();

    assert_eq!(record.id, restored.id);
    assert_eq!(record.fields.get("name"), restored.fields.get("name"));
}

#[test]
fn test_document_store_crud() {
    let mut store = DocumentStore::new();

    let mut fields = HashMap::new();
    fields.insert("name".to_string(), serde_json::json!("Test User"));

    let record = Record::new(fields);
    let id = record.id;

    store.insert(record.clone()).unwrap();
    assert_eq!(store.len(), 1);

    let fetched = store.get(&id).unwrap();
    assert_eq!(fetched.fields.get("name"), record.fields.get("name"));

    store.delete(&id).unwrap();
    assert_eq!(store.len(), 0);
}

#[test]
fn test_wal_open_and_append() {
    let dir = tempdir().unwrap();
    let wal_dir = dir.path().join("WAL");
    let mut wal = Wal::open(&wal_dir, Durability::Sync, 1 << 20).unwrap();
    let mut r = WalRecord::new(0, RecordKind::InsertDocument);
    r.payload = b"hello".to_vec();
    wal.append(r).unwrap();
}

#[test]
fn test_catalog_roundtrip_in_db_dir() {
    let dir = tempdir().unwrap();
    Catalog::ensure_dirs(dir.path()).unwrap();
    let cat = Catalog::fresh();
    cat.save(dir.path()).unwrap();
    let (loaded, fallback) = Catalog::load(dir.path()).unwrap();
    assert!(!fallback);
    assert_eq!(loaded.database_format_version, cat.database_format_version);
}

#[test]
fn test_document_store_flush_and_reload() {
    let dir = tempdir().unwrap();
    let sst_dir = dir.path().join("sst");
    {
        let mut store = DocumentStore::open_without_wal(sst_dir.clone()).unwrap();
        for i in 0..10 {
            let mut fields = HashMap::new();
            fields.insert("i".to_string(), serde_json::json!(i));
            store.insert(Record::new(fields)).unwrap();
        }
        store.flush().unwrap();
    }
    let store = DocumentStore::open_without_wal(sst_dir).unwrap();
    assert_eq!(store.len(), 10);
}
