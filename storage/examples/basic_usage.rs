//! Basic example of using the Phase 1 storage layer

use attentiondb_storage::{DocumentStore, Record};
use std::collections::HashMap;

fn main() {
    println!("AttentionDB Phase 1 - Basic Usage Example");

    let mut store = DocumentStore::new();

    let mut fields = HashMap::new();
    fields.insert("name".to_string(), serde_json::json!("Rayyan"));
    fields.insert("role".to_string(), serde_json::json!("Researcher"));

    let record = Record::new(fields);
    println!("Created record: {}", record.id);

    let id = store.insert(record).unwrap();
    println!("Inserted record. Total records: {}", store.len());

    if let Some(fetched) = store.get(&id) {
        println!("Fetched: {:?}", fetched.fields.get("name"));
    }
}
