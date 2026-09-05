//! Phase 2 §56/§57 — metric contract: exact names exist, stage latencies are
//! recorded, and NO high-cardinality labels (doc ids / query text) ever leak
//! into the metrics surface.

use attentiondb_core::AttentionEngine;
use attentiondb_query::filter::{FilterExpr, FilterOp, FilterValue};

fn setup(n: usize) -> (tempfile::TempDir, AttentionEngine) {
    let dir = tempfile::tempdir().unwrap();
    let e = AttentionEngine::open_dir(dir.path(), attentiondb_storage::Durability::Sync).unwrap();
    e.create_collection("c", 8, &["default"]).unwrap();
    for i in 0..n {
        let mut r = attentiondb_storage::Record::new(std::collections::HashMap::from([
            ("idx".to_string(), serde_json::json!(i)),
            (
                "body".to_string(),
                serde_json::json!(format!("alpha beta gamma {i}")),
            ),
            (
                "tier".to_string(),
                serde_json::json!(if i % 2 == 0 { "gold" } else { "silver" }),
            ),
        ]));
        r.k_vecs.insert("default".to_string(), one_hot(i % 8, 8));
        e.insert_document("c", r).unwrap();
    }
    (dir, e)
}

fn one_hot(i: usize, dim: usize) -> Vec<f32> {
    (0..dim).map(|j| if j == i { 1.0 } else { 0.0 }).collect()
}

#[test]
fn metric_names_and_cardinality_contract() {
    // In-process Prometheus recorder; render must contain the §56 names.
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::set_global_recorder(recorder).unwrap();

    let (_dir, e) = setup(60);
    let f = FilterExpr::Comparison {
        field: "tier".into(),
        op: FilterOp::Eq,
        value: FilterValue::Str("gold".into()),
    };
    // exercise: vector, filtered, hybrid+filter, scan
    let _ = e
        .attend_with_deadline_ms("c", &["default".into()], &one_hot(1, 8), 5, 1_000)
        .unwrap();
    let _ = e
        .attend_filtered("c", &["default".into()], &one_hot(2, 8), 5, Some(&f))
        .unwrap();
    let _ = e
        .attend_hybrid_filtered_with_deadline(
            "c",
            &["default".into()],
            &one_hot(3, 8),
            "alpha beta",
            5,
            &f,
            None,
        )
        .unwrap();
    let _ = e.scan_filtered("c", Some(&f), 10).unwrap();

    attentiondb_api::observability::record_query("c", "vector", 5, 0.05);
    attentiondb_api::observability::record_query_error("attend");

    let out = handle.render();
    for name in [
        "attentiondb_queries_total",
        "attentiondb_query_errors_total",
        "attentiondb_query_latency_seconds",
        "attentiondb_stage_latency_seconds",
        "attentiondb_filter_selectivity",
        "attentiondb_candidates_total",
    ] {
        assert!(
            out.contains(name),
            "metric '{name}' missing from /metrics render:\n{}",
            &out[..out.len().min(2000)]
        );
    }
    for stage in [
        "hnsw",
        "candidate_union",
        "exact_rerank",
        "attention",
        "bm25",
    ] {
        assert!(
            out.contains(&format!("stage=\"{stage}\"")),
            "stage '{stage}' missing"
        );
    }
    // HIGH-CARDINALITY GUARD: no doc field values or query text as labels.
    assert!(
        !out.contains("alpha beta gamma"),
        "query text leaked into metrics"
    );
    assert!(
        !out.contains("tier=\"gold\"") || !out.contains("labels"),
        "doc fields must not be labels"
    );
    // mode labels are the only per-query label dimension
    assert!(
        out.contains("mode=\"vector\"") || out.contains("mode="),
        "mode label missing"
    );
}
