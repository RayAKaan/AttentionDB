//! Phase 1 integration tests — backup & restore (TEST 18).

mod common;

use common::*;
use std::collections::HashMap;

/// TEST 18 — Backup/restore: exact logical state reconstruction, including
/// deletes staying deleted.
#[test]
fn t18_backup_restore_roundtrip() {
    let dir = temp_db();
    let backup = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("products", 8, &["default"]).unwrap();
        e.create_collection("users", 8, &["default"]).unwrap();
        for i in 0..30 {
            e.insert_document(
                "products",
                doc(i, &one_hot(i as usize % 8, 8), "product body"),
            )
            .unwrap();
        }
        for i in 100..110 {
            e.insert_document("users", doc(i, &one_hot(i as usize % 8, 8), "user body"))
                .unwrap();
        }
        // update one, delete two
        let all = e.document_store.read().list_all_records();
        let upd = all
            .iter()
            .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(5))
            .unwrap()
            .id
            .to_string();
        let mut f = HashMap::new();
        f.insert("body".to_string(), serde_json::json!("product UPDATED"));
        let mut kv = HashMap::new();
        kv.insert("default".to_string(), one_hot(7, 8));
        e.update_document("products", &upd, f, kv).unwrap();
        for idx in [1i64, 2] {
            let u = all
                .iter()
                .find(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(idx))
                .unwrap()
                .id
                .to_string();
            e.delete_document("products", &u).unwrap();
        }
        // collection settings
        let s = attentiondb_hnsw::CollectionSettings {
            ef_search: 96,
            ..Default::default()
        };
        e.alter_collection_settings("products", s).unwrap();

        e.backup_to(backup.path()).unwrap();
        e.close().unwrap();
    }

    // "Destroy" the database and restore elsewhere.
    let restored = temp_db();
    attentiondb_core::backup::restore_backup(backup.path(), restored.path()).unwrap();
    let e = open_sync(restored.path());
    // collections + settings
    let mut cols = e.list_collections();
    cols.sort();
    assert_eq!(cols, vec!["products".to_string(), "users".to_string()]);
    let s = e
        .get_collection("products")
        .unwrap()
        .settings
        .read()
        .clone();
    assert_eq!(s.ef_search, 96);
    // documents: 30 - 2 deleted (updated one still counts, 28 live)
    let products = e.document_store.read().list_all_records();
    let prod_count = products
        .iter()
        .filter(|r| r.tags.contains(&"collection:products".to_string()))
        .count();
    assert_eq!(prod_count, 28, "28 live product docs (30 - 2 deleted)");
    // updated doc has the new body
    assert!(products
        .iter()
        .any(|r| r.fields.get("body").and_then(|v| v.as_str()) == Some("product UPDATED")));
    // deleted docs stay deleted
    assert!(!products
        .iter()
        .any(|r| r.fields.get("idx").and_then(|v| v.as_i64()) == Some(1)
            || r.fields.get("idx").and_then(|v| v.as_i64()) == Some(2)));
    // search equivalence: the updated doc's exact vector (one_hot(7)) is shared
    // by docs 7/15/23 too (4-way tie at sim 1.0), so per-process hnsw_rs layer
    // RNG makes top-1 identity arbitrary — but ALL exact matches appear in the
    // top-10 deterministically, both before and after restore.
    let q = one_hot(7, 8);
    let r = e.attend("products", &["default".into()], &q, 10).unwrap();
    let bodies: Vec<String> = r
        .iter()
        .map(|(id, _)| e.get_document_fields(*id))
        .filter_map(|f| f.get("body").cloned())
        .collect();
    assert!(
        bodies.iter().any(|b| b == "product UPDATED"),
        "updated doc not in top-10 after restore; got {bodies:?}"
    );
    // checker passes
    let issues = attentiondb_core::checker::check_engine(&e);
    assert!(
        !issues.iter().any(|i| i.severity.is_error()),
        "checker: {issues:?}"
    );
}

/// Restoring into a non-empty directory is refused (never mix states).
#[test]
fn t18b_restore_refuses_nonempty_destination() {
    let dir = temp_db();
    let backup = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 8, &["default"]).unwrap();
        e.insert_document("c", doc(1, &one_hot(1, 8), "x")).unwrap();
        e.backup_to(backup.path()).unwrap();
    }
    let dest = temp_db();
    std::fs::write(dest.path().join("existing-file"), b"data").unwrap();
    let result = attentiondb_core::backup::restore_backup(backup.path(), dest.path());
    assert!(result.is_err(), "restore must refuse non-empty destination");
}
