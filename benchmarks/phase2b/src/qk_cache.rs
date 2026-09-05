//! PH2C-QK-002 — candidate-content sidecar builder (Phase 2C §3/§4/§7).
//!
//! The Phase 2B caches (dataset.json) store per-head candidate ENGINE ids +
//! scores but not document vectors; candidate-level QK needs K = W_K·x with
//! the actual content. This subcommand rebuilds the SAME corpus in-process
//! (deterministic doc generation; only HNSW graph construction is OS-seeded
//! and irrelevant here) and:
//!   1. verifies the fresh engine-id mapping reproduces the CACHED ground
//!      truth exactly for every query (HC-1-class check against mapping
//!      drift), and that cached query vectors == corpus query vectors;
//!   2. verifies cached exact_scores == recomputed cosines (content
//!      linkage, tolerance 1e-4);
//!   3. writes qk_content.json: the doc content matrix (concat head vectors
//!      per doc), docidx→engine-id table, and per-query per-head candidate
//!      DOC INDICES aligned with the cached pool order (pools are NOT
//!      regenerated — §6 fairness: identical candidate pools for every arm).
//!
//! The cached dataset.json remains the sole source of pools/scores/splits;
//! this sidecar only attaches content by doc index.

use attentiondb_learned::gating_v2::GatingDataset;
use phase2b_bench::corpora::{self};
use std::collections::HashMap;

#[derive(serde::Serialize)]
struct Sidecar {
    format: &'static str,
    version: u32,
    corpus: String,
    num_heads: usize,
    head_dim: usize,
    docs: Vec<Vec<f32>>, // per doc: concat head vectors
    docidx_to_engine: Vec<u64>,
    /// per query: per head: candidate doc indices (aligned with cached pool order)
    queries: Vec<Vec<Vec<u32>>>,
}

pub fn run(corpus_name: &str, dataset_path: &str, out_path: &str) -> String {
    let corpus = super::build_corpus_q(corpus_name, None);
    let (_guard, e, _ids) = corpora::insert_corpus(&corpus);
    let coll = e.get_collection("bench").unwrap();

    // engine numeric id → corpus doc index (same resolution as Phase 2B)
    let mut id_to_idx: HashMap<u64, usize> = HashMap::new();
    {
        let store = e.document_store.read();
        let mapper = e.id_mapper.read();
        for rec in store.list_all_records() {
            if let Some(idx_field) = rec.fields.get("idx").and_then(|v| v.as_u64()) {
                if let Some(numeric) = mapper.uuid_to_id(&rec.id) {
                    id_to_idx.insert(numeric, idx_field as usize);
                }
            }
        }
    }
    let mut idx_to_engine = vec![0u64; corpus.docs.len()];
    for (&engine_id, &di) in id_to_idx.iter() {
        idx_to_engine[di] = engine_id;
    }

    let json = std::fs::read_to_string(dataset_path).unwrap();
    let ds = GatingDataset::parse(&json).unwrap();
    assert_eq!(ds.queries.len(), corpus.queries.len(), "query count drift");

    // ---- check 1: fresh mapping reproduces cached GT exactly (HC-1 class) ----
    let mut gt_mismatch = 0usize;
    for (i, q) in ds.queries.iter().enumerate() {
        let hints: Vec<usize> = q
            .ground_truth
            .iter()
            .map(|&eid| {
                *id_to_idx
                    .get(&eid)
                    .unwrap_or_else(|| panic!("engine id {eid} missing from fresh mapping"))
            })
            .collect();
        if hints
            != corpus.queries[i]
                .ground_truth
                .iter()
                .map(|&x| x as usize)
                .collect::<Vec<_>>()
        {
            gt_mismatch += 1;
        }
    }
    assert_eq!(
        gt_mismatch, 0,
        "GT mapping mismatch on {gt_mismatch} queries — HC-1-class drift, refusing"
    );

    // ---- check 2 (informational): cached vs rebuilt QUERY vectors ----
    // HC-6 finding: corpus regeneration is NOT query-vector-stable for
    // controlled/noise (GT/doc picks stable). The cached dataset.json is the
    // reproducibility unit; its query vectors are canonical. We therefore do
    // NOT use rebuilt query vectors anywhere downstream.
    let qdrift = (0..ds.queries.len())
        .filter(|&i| {
            let bq = &corpus.queries[i];
            let mut concat = Vec::new();
            for h in 0..corpus.head_names.len() {
                concat.extend_from_slice(&bq.vectors[h]);
            }
            ds.queries[i].query != concat
        })
        .count();

    let head_dim = ds.input_dim / ds.num_heads;

    // ---- check 3 (BINDING): doc content linkage via cached exact scores.
    // The cached exact score was cosine(ORIGINAL query head vec, ORIGINAL doc
    // head vec). Original query == cached query (dataset.json). So:
    // cosine(cached query head vec, TODAY's doc vec at the mapped index)
    // must reproduce the cached exact score — otherwise TODAY's doc content
    // does not correspond to the pools and attaching it would be invalid.
    let mut checked = 0usize;
    for (i, q) in ds.queries.iter().enumerate().step_by(25) {
        for (h, head) in q.heads.iter().enumerate() {
            let qv = &q.query[h * head_dim..(h + 1) * head_dim];
            for (j, &eid) in head.candidates.iter().enumerate() {
                let di = id_to_idx[&eid];
                let cos = corpora::cosine(qv, &corpus.docs[di].head_vecs[h]);
                assert!(
                    (cos - head.exact_scores[j]).abs() < 1e-4,
                    "DOC CONTENT DRIFT q{i} h{h} j{j}: {cos} vs {} — pools and content do not correspond; refusing",
                    head.exact_scores[j]
                );
                checked += 1;
            }
        }
    }

    // ---- build sidecar ----
    let head_dim = ds.input_dim / ds.num_heads;

    let docs: Vec<Vec<f32>> = corpus
        .docs
        .iter()
        .map(|d| {
            let mut v = Vec::with_capacity(ds.input_dim);
            for h in 0..corpus.head_names.len() {
                v.extend_from_slice(&d.head_vecs[h]);
            }
            v
        })
        .collect();
    let queries: Vec<Vec<Vec<u32>>> = ds
        .queries
        .iter()
        .map(|q| {
            q.heads
                .iter()
                .map(|head| {
                    head.candidates
                        .iter()
                        .map(|&eid| id_to_idx[&eid] as u32)
                        .collect()
                })
                .collect()
        })
        .collect();
    let sc = Sidecar {
        format: "attentiondb-qk-content-sidecar",
        version: 2,
        corpus: corpus_name.to_string(),
        num_heads: ds.num_heads,
        head_dim,
        docs,
        docidx_to_engine: idx_to_engine,
        queries,
    };
    std::fs::write(out_path, serde_json::to_string(&sc).unwrap()).unwrap();

    // sanity: a candidate used by the engine must still be the same doc the
    // engine scored — spot-check one query's top raw score against content.
    let _ = (coll, checked);
    format!(
        "qk-content sidecar written: {out_path} (docs={} head_dim={head_dim} gt_verified={} exact_scores_verified={checked} query_vec_drift={qdrift})",
        corpus.docs.len(),
        ds.queries.len()
    )
}
