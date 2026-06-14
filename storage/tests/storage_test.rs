//! Basic storage layer tests for Phase 1

use attentiondb_storage::{Record, DocumentStore, Wal};
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
fn test_wal_creation() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let _wal = Wal::new(&wal_path).unwrap();
}
