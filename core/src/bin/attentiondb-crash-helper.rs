//! `attentiondb-crash-helper` — subprocess crash-test helper (§15).
//!
//! Used by Phase 1 crash tests: the test spawns this binary, which performs a
//! sequence of mutations and then kills its own process with SIGABRT
//! (`std::process::abort`) — Rust destructors do NOT run, closely simulating
//! `kill -9`. The test then reopens the database and verifies the documented
//! post-crash state.
//!
//! Modes:
//!   insert-abort <db_dir> <n>          open/create, insert n docs, abort
//!   insert-checkpoint-abort <db> <n>   insert n, checkpoint, insert n more, abort
//!   txn-commit-abort <db_dir> <n>      commit txn of n inserts, abort
//!   txn-staged-abort <db_dir> <n>      stage txn of n inserts (no commit), abort
//!   delete-abort <db_dir> <k>          delete first k docs by id, abort
//!   update-abort <db_dir> <n>          update first n docs' vectors, abort

use attentiondb_core::AttentionEngine;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;

fn open(db_dir: &str) -> AttentionEngine {
    match AttentionEngine::open_dir(
        std::path::Path::new(db_dir),
        Durability::Sync, // Sync so every acknowledged write is fsync-durable
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("crash-helper: open failed: {e}");
            std::process::exit(3);
        }
    }
}

fn ensure_collection(engine: &AttentionEngine, name: &str, dim: usize, heads: &[&str]) {
    if engine.get_collection(name).is_err() {
        engine
            .create_collection(name, dim, heads)
            .expect("create collection");
    }
}

fn make_record(i: usize, dim: usize) -> Record {
    let mut fields = HashMap::new();
    fields.insert("idx".to_string(), serde_json::json!(i));
    fields.insert(
        "body".to_string(),
        serde_json::json!(format!("document body number {i}")),
    );
    let mut rec = Record::new(fields);
    rec.k_vecs
        .insert("default".to_string(), vec![i as f32; dim]);
    rec
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: attentiondb-crash-helper <mode> <db_dir> [n]");
        std::process::exit(2);
    }
    let mode = args[1].as_str();
    let db_dir = args[2].as_str();
    let n: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(10);
    let dim = 8;

    match mode {
        "insert-abort" => {
            let engine = open(db_dir);
            ensure_collection(&engine, "c1", dim, &["default"]);
            for i in 0..n {
                engine
                    .insert_document("c1", make_record(i, dim))
                    .expect("insert");
            }
            eprintln!("crash-helper: inserted {n}, aborting");
            std::process::abort();
        }
        "insert-checkpoint-abort" => {
            let engine = open(db_dir);
            ensure_collection(&engine, "c1", dim, &["default"]);
            for i in 0..n {
                engine
                    .insert_document("c1", make_record(i, dim))
                    .expect("insert");
            }
            engine.checkpoint().expect("checkpoint");
            for i in n..2 * n {
                engine
                    .insert_document("c1", make_record(i, dim))
                    .expect("insert");
            }
            eprintln!("crash-helper: checkpointed at {n}, inserted {n} more, aborting");
            std::process::abort();
        }
        "txn-commit-abort" => {
            let engine = open(db_dir);
            ensure_collection(&engine, "c1", dim, &["default"]);
            let txn = engine.begin_transaction("c1");
            for i in 0..n {
                engine
                    .record_transaction_operation(
                        txn,
                        attentiondb_core::TxnOp::Insert(make_record(i, dim)),
                    )
                    .expect("stage");
            }
            engine.commit_transaction(txn).expect("commit");
            eprintln!("crash-helper: committed txn of {n}, aborting before/around apply");
            std::process::abort();
        }
        "txn-staged-abort" => {
            let engine = open(db_dir);
            ensure_collection(&engine, "c1", dim, &["default"]);
            let txn = engine.begin_transaction("c1");
            for i in 0..n {
                engine
                    .record_transaction_operation(
                        txn,
                        attentiondb_core::TxnOp::Insert(make_record(i, dim)),
                    )
                    .expect("stage");
            }
            eprintln!("crash-helper: staged (never committed) txn of {n}, aborting");
            std::process::abort();
        }
        "delete-abort" => {
            let engine = open(db_dir);
            // Snapshot FIRST and drop the read guard: delete_document takes the
            // document_store write lock; parking_lot is non-reentrant and a
            // same-thread read→write upgrade deadlocks (see the concurrency
            // contract on AttentionEngine::document_store).
            let records: Vec<_> = engine.document_store.read().list_all_records();
            let mut deleted = 0;
            for rec in records {
                if deleted >= n {
                    break;
                }
                if rec.tags.contains(&"collection:c1".to_string()) {
                    let id = rec.id.to_string();
                    if engine.delete_document("c1", &id).expect("delete") {
                        deleted += 1;
                    }
                }
            }
            eprintln!("crash-helper: deleted {deleted}, aborting");
            std::process::abort();
        }
        "update-abort" => {
            let engine = open(db_dir);
            let mut updated = 0;
            let records: Vec<_> = engine.document_store.read().list_all_records();
            for rec in records {
                if updated >= n {
                    break;
                }
                if !rec.tags.contains(&"collection:c1".to_string()) {
                    continue;
                }
                let mut fields = HashMap::new();
                fields.insert("idx".to_string(), serde_json::json!(updated));
                fields.insert("body".to_string(), serde_json::json!("UPDATED BODY"));
                let mut k_vecs = HashMap::new();
                k_vecs.insert("default".to_string(), vec![-1.0; dim]);
                engine
                    .update_document("c1", &rec.id.to_string(), fields, k_vecs)
                    .expect("update");
                updated += 1;
            }
            eprintln!("crash-helper: updated {updated}, aborting");
            std::process::abort();
        }
        _ => {
            eprintln!("unknown mode {mode}");
            std::process::exit(2);
        }
    }
}
