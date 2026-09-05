//! Phase 2 integration tests — planner, EXPLAIN, limits, timeout (§16–19).

mod common;

use common::*;

const DIM: usize = 8;

fn setup(n: usize) -> (tempfile::TempDir, attentiondb_core::AttentionEngine) {
    let dir = temp_db();
    let e = open_sync(dir.path());
    e.create_collection("c", DIM, &["default", "semantic"])
        .unwrap();
    for i in 0..n {
        let mut r = doc(i as i64, &one_hot(i % DIM, DIM), "x");
        r.k_vecs
            .insert("semantic".to_string(), one_hot((i * 3) % DIM, DIM));
        e.insert_document("c", r).unwrap();
    }
    (dir, e)
}

/// §17: EXPLAIN renders the real plan with all stages and skips missing heads.
#[test]
fn explain_reflects_actual_plan() {
    let (_dir, e) = setup(60);
    let text = e
        .explain("c", &["default".into(), "ghost".into()], 10, None, false)
        .unwrap();
    for stage in [
        "QUERY",
        "VECTOR SEARCH",
        "CANDIDATE UNION",
        "RERANK",
        "FINAL TOP K",
    ] {
        assert!(text.contains(stage), "missing '{stage}' in:\n{text}");
    }
    assert!(text.contains("ghost"), "missing heads must be reported");
    assert!(
        text.contains("exact_similarity: on"),
        "MODE E default must show exact rerank"
    );
    // deterministic
    let text2 = e
        .explain("c", &["default".into(), "ghost".into()], 10, None, false)
        .unwrap();
    assert_eq!(text, text2);
}

/// §17: explain with a filter shows the filter strategy without running it.
#[test]
fn explain_shows_filter_strategy() {
    let (_dir, e) = setup(40);
    let f = attentiondb_query::filter::FilterExpr::Comparison {
        field: "idx".into(),
        op: attentiondb_query::filter::FilterOp::Eq,
        value: attentiondb_query::filter::FilterValue::Int(1),
    };
    let text = e
        .explain("c", &["default".into()], 10, Some(&f), false)
        .unwrap();
    assert!(
        text.contains("post-filter"),
        "filter strategy missing:\n{text}"
    );
    // explain does not execute: works on an empty collection too
    let dir2 = temp_db();
    let e2 = open_sync(dir2.path());
    e2.create_collection("empty", DIM, &["default"]).unwrap();
    let t2 = e2
        .explain("empty", &["default".into()], 10, None, false)
        .unwrap();
    assert!(t2.contains("VECTOR SEARCH"));
}

/// §19: a deadline that cannot possibly be met surfaces CoreError::Timeout —
/// never a partial or unfiltered result.
#[test]
fn deadline_surfaces_timeout_error() {
    let (_dir, e) = setup(300);
    // deadline already in the past → first stage check fires
    let past = std::time::Instant::now() - std::time::Duration::from_secs(1);
    let coll = e.get_collection("c").unwrap();
    let err = coll
        .attend_detailed(
            &["default".into(), "semantic".into()],
            &one_hot(1, DIM),
            10,
            None,
            None,
            None,
            None,
            Some(past),
        )
        .unwrap_err();
    assert!(
        matches!(err, attentiondb_core::CoreError::Timeout(_)),
        "expected Timeout, got {err:?}"
    );
    // engine wrapper: timeout_ms=0 and huge values are rejected by limits
    assert!(e
        .attend_with_deadline_ms("c", &["default".into()], &one_hot(1, DIM), 10, 0)
        .is_err());
    assert!(e
        .attend_with_deadline_ms("c", &["default".into()], &one_hot(1, DIM), 10, 10_000_000)
        .is_err());
}

/// §18/§42: abusive queries are rejected before allocation.
#[test]
fn query_limits_reject_abuse_before_execution() {
    let (_dir, e) = setup(10);
    assert!(e
        .attend_with_deadline_ms("c", &["default".into()], &one_hot(1, DIM), 5_000_000, 1000)
        .is_err());
    // too many heads
    let heads: Vec<String> = (0..64).map(|i| format!("h{i}")).collect();
    assert!(e
        .attend_with_deadline_ms("c", &heads, &one_hot(1, DIM), 10, 1000)
        .is_err());
    // dimension over limit
    let huge = vec![0.0f32; 5000];
    assert!(e
        .attend_with_deadline_ms("c", &["default".into()], &huge, 10, 1000)
        .is_err());
    // a normal query still works after rejections
    let r = e
        .attend_with_deadline_ms("c", &["default".into()], &one_hot(1, DIM), 5, 1000)
        .unwrap();
    assert!(!r.is_empty());
}

/// §19: a realistic deadline still completes normal queries (no false timeouts).
#[test]
fn normal_queries_meet_their_deadline() {
    let (_dir, e) = setup(500);
    for _ in 0..10 {
        let r = e
            .attend_with_deadline_ms(
                "c",
                &["default".into(), "semantic".into()],
                &one_hot(3, DIM),
                10,
                5_000,
            )
            .unwrap();
        assert!(!r.is_empty());
    }
}
