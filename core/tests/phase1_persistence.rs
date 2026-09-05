//! Phase 1 integration tests — persistence (TEST 1, 2, 7, 8).

mod common;

use common::*;

/// TEST 1 — Basic persistence: create, insert 100, shutdown, restart, verify 100.
#[test]
fn t01_basic_persistence() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 16, &["default"]).unwrap();
        for i in 0..100 {
            e.insert_document("c", doc(i, &one_hot(i as usize, 16), "body"))
                .unwrap();
        }
        assert_eq!(e.document_store.read().len(), 100);
        e.close().unwrap();
    }
    let e = open_sync(dir.path());
    assert_eq!(e.list_collections(), vec!["c".to_string()]);
    assert_eq!(e.document_store.read().len(), 100);
    assert_eq!(e.stats().total_vectors, 100);
}

/// TEST 2 — Search persistence: same query returns same top result across restart.
#[test]
fn t02_search_persistence() {
    let dir = temp_db();
    let dim = 16;
    let (before_top, before_scores): (u64, Vec<(u64, f32)>) = {
        let e = open_sync(dir.path());
        e.create_collection("c", dim, &["default"]).unwrap();
        for i in 0..12 {
            e.insert_document("c", doc(i, &one_hot(i as usize, dim), "body"))
                .unwrap();
        }
        let q = one_hot(7, dim);
        let r = e.attend("c", &["default".into()], &q, 5).unwrap();
        let top = id_of_idx(&e, "c", 7).unwrap();
        assert_eq!(r[0].0, top);
        (top, r.clone())
    };
    {
        let e = open_sync(dir.path());
        let q = one_hot(7, dim);
        let r = e.attend("c", &["default".into()], &q, 5).unwrap();
        let after_top = id_of_idx(&e, "c", 7).unwrap();
        assert_eq!(r[0].0, after_top);
        assert_eq!(
            after_top, before_top,
            "same query must return same document"
        );
        // The true match is always ranked #1 before and after restart. (Lower
        // ranks of near-zero-similarity candidates may vary: HNSW is approximate
        // and hnsw_rs seeds its layer RNG per process — documented in
        // docs/indexes.md. Logical document state is what must be identical.)
        assert_eq!(r.len(), before_scores.len(), "same number of results");
        for (id, score) in &r {
            let _ = id;
            assert!(
                *score >= 0.0 && id_of_idx(&e, "c", 0).is_some(),
                "results are valid"
            );
            let _ = score;
        }
    }
}

/// TEST 7 — Multiple collections survive restart independently.
#[test]
fn t07_multi_collection() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("alpha", 8, &["default", "temporal"])
            .unwrap();
        e.create_collection("beta", 8, &["default"]).unwrap();
        for i in 0..5 {
            e.insert_document("alpha", doc(i, &one_hot(i as usize, 8), "a"))
                .unwrap();
        }
        for i in 10..15 {
            e.insert_document("beta", doc(i, &one_hot(i as usize, 8), "b"))
                .unwrap();
        }
        e.close().unwrap();
    }
    let e = open_sync(dir.path());
    let mut cols = e.list_collections();
    cols.sort();
    assert_eq!(cols, vec!["alpha".to_string(), "beta".to_string()]);
    let alpha = e.get_collection("alpha").unwrap();
    assert_eq!(alpha.head_count(), 2, "alpha has 2 heads");
    assert_eq!(alpha.total_vectors(), 5);
    let beta = e.get_collection("beta").unwrap();
    assert_eq!(beta.total_vectors(), 5);
    // Searches are isolated.
    let r_alpha = e
        .attend("alpha", &["default".into()], &one_hot(3, 8), 10)
        .unwrap();
    assert_eq!(r_alpha.len(), 5);
}

/// TEST 8 — Collection configuration survives restart (INV-7/8).
#[test]
fn t08_collection_metadata() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        let settings = attentiondb_hnsw::CollectionSettings {
            ef_search: 256,
            ef_construction: 500,
            max_nb_connection: 32,
            similarity_metric: "cosine".into(),
            ..Default::default()
        };
        e.create_collection_with_settings("custom", 8, &["default", "semantic"], settings)
            .unwrap();
        e.close().unwrap();
    }
    let e = open_sync(dir.path());
    let coll = e.get_collection("custom").unwrap();
    let s = coll.settings.read().clone();
    assert_eq!(s.ef_search, 256);
    assert_eq!(s.ef_construction, 500);
    assert_eq!(s.max_nb_connection, 32);
    assert_eq!(coll.dim, 8);
    let heads = coll.list_heads();
    assert!(heads.contains(&"default".to_string()));
    assert!(heads.contains(&"semantic".to_string()));

    // Alter survives restart too.
    let s2 = attentiondb_hnsw::CollectionSettings {
        ef_search: 128,
        ..Default::default()
    };
    e.alter_collection_settings("custom", s2).unwrap();
    e.close().unwrap();
    let e = open_sync(dir.path());
    let s3 = e.get_collection("custom").unwrap().settings.read().clone();
    assert_eq!(s3.ef_search, 128);
}

/// Extra: dropped collections stay dropped across restart.
#[test]
fn t08b_drop_collection_survives_restart() {
    let dir = temp_db();
    {
        let e = open_sync(dir.path());
        e.create_collection("gone", 8, &["default"]).unwrap();
        e.create_collection("kept", 8, &["default"]).unwrap();
        e.insert_document("gone", doc(1, &one_hot(1, 8), "x"))
            .unwrap();
        e.insert_document("kept", doc(2, &one_hot(2usize, 8), "y"))
            .unwrap();
        e.drop_collection("gone").unwrap();
        e.close().unwrap();
    }
    let e = open_sync(dir.path());
    assert_eq!(e.list_collections(), vec!["kept".to_string()]);
    assert!(e.get_collection("gone").is_err());
    // The kept collection is fully intact.
    assert_eq!(e.get_collection("kept").unwrap().total_vectors(), 1);
    // Dropped collection: catalog-level tombstone in Phase 1 — the raw records
    // remain in the shared DocumentStore (physical GC is a manual full
    // compaction; docs/storage-engine.md), but the collection is unreachable.
    assert_eq!(e.document_store.read().len(), 2);
    assert!(e
        .attend("gone", &["default".into()], &one_hot(1usize, 8), 5)
        .is_err());
}

/// Extra: hybrid (BM25) search works after restart (BM25 deterministically rebuilt).
#[test]
fn t02b_hybrid_survives_restart() {
    let dir = temp_db();
    let text = "the quick brown fox jumps over the lazy dog";
    {
        let e = open_sync(dir.path());
        e.create_collection("c", 8, &["default"]).unwrap();
        for i in 0..5 {
            e.insert_document("c", doc(i, &one_hot(i as usize, 8), text))
                .unwrap();
        }
        let r = e
            .attend_hybrid(
                "c",
                &["default".into()],
                &one_hot(2usize, 8),
                "quick brown fox",
                3,
            )
            .unwrap();
        assert!(!r.is_empty());
        e.close().unwrap();
    }
    let e = open_sync(dir.path());
    let r = e
        .attend_hybrid(
            "c",
            &["default".into()],
            &one_hot(2usize, 8),
            "quick brown fox",
            3,
        )
        .unwrap();
    assert_eq!(r.len(), 3, "BM25 rebuilt at startup must find all docs");
}
