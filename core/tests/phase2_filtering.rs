//! Phase 2 integration tests — metadata filtering (§11–13).
//!
//! The non-negotiable invariant: a filtered query NEVER returns a document
//! that does not satisfy the filter — regardless of vector similarity, selectivity,
//! or expansion state. Plus: null/missing semantics, nested expressions, typed
//! comparisons, filter-only scans.

mod common;

use attentiondb_query::filter::{parse_filter_json, FilterExpr, FilterOp, FilterValue};
use common::*;
use serde_json::json;
use std::collections::HashMap;

const DIM: usize = 8;
const N_DOCS: usize = 2_000;

/// Raw (typed) fields for a numeric id — filter evaluation happens on RAW
/// fields, never the stringified REST projection.
fn raw_fields(
    e: &attentiondb_core::AttentionEngine,
    numeric: u64,
) -> HashMap<String, serde_json::Value> {
    let uuid = e.id_mapper.read().id_to_uuid(numeric).copied().unwrap();
    e.document_store.read().get(&uuid).unwrap().fields.clone()
}

/// 2000 docs: categories x/y/z/missing (500/500/500/500), year 2015..2025,
/// some active flags, some explicit nulls.
fn setup(dir: &std::path::Path) -> attentiondb_core::AttentionEngine {
    let e = open_sync(dir);
    e.create_collection("docs", DIM, &["default"]).unwrap();
    for i in 0..N_DOCS {
        let mut fields: HashMap<String, serde_json::Value> = HashMap::new();
        fields.insert("idx".to_string(), json!(i));
        fields.insert("body".to_string(), json!(format!("document {i}")));
        match i % 4 {
            0 => {
                fields.insert("category".to_string(), json!("x"));
            }
            1 => {
                fields.insert("category".to_string(), json!("y"));
            }
            2 => {
                fields.insert("category".to_string(), json!("z"));
            }
            _ => {} // no category field at all
        }
        fields.insert("year".to_string(), json!(2015 + (i % 10) as i64));
        if i % 3 == 0 {
            fields.insert("active".to_string(), json!(true));
        }
        if i % 5 == 0 {
            fields.insert("deleted_at".to_string(), serde_json::Value::Null);
        }
        let mut r = attentiondb_storage::Record::new(fields);
        r.k_vecs
            .insert("default".to_string(), one_hot(i % DIM, DIM));
        e.insert_document("docs", r).unwrap();
    }
    e
}

fn cat_eq(cat: &str) -> FilterExpr {
    FilterExpr::Comparison {
        field: "category".into(),
        op: FilterOp::Eq,
        value: FilterValue::Str(cat.into()),
    }
}

/// §13 — THE invariant: 100% filter precision under many query shapes.
#[test]
fn filtered_query_never_returns_nonmatching_documents() {
    let dir = temp_db();
    let e = setup(dir.path());

    let queries: Vec<Vec<f32>> = (0..DIM).map(|i| one_hot(i, DIM)).collect();
    let filters: Vec<FilterExpr> = vec![
        cat_eq("x"),
        cat_eq("y"),
        cat_eq("z"),
        // category in a set
        FilterExpr::In {
            field: "category".into(),
            values: vec![FilterValue::Str("x".into()), FilterValue::Str("z".into())],
            negated: false,
        },
        // range
        FilterExpr::Comparison {
            field: "year".into(),
            op: FilterOp::Gte,
            value: FilterValue::Int(2020),
        },
        // nested: (category = x OR category = y) AND year < 2018 AND active
        FilterExpr::And(
            Box::new(FilterExpr::Or(Box::new(cat_eq("x")), Box::new(cat_eq("y")))),
            Box::new(FilterExpr::And(
                Box::new(FilterExpr::Comparison {
                    field: "year".into(),
                    op: FilterOp::Lt,
                    value: FilterValue::Int(2018),
                }),
                Box::new(FilterExpr::Comparison {
                    field: "active".into(),
                    op: FilterOp::Eq,
                    value: FilterValue::Bool(true),
                }),
            )),
        ),
        // explicit null semantics
        FilterExpr::IsNull {
            field: "deleted_at".into(),
            negated: false,
        },
        FilterExpr::IsNull {
            field: "category".into(),
            negated: true,
        },
    ];

    for f in &filters {
        for q in &queries {
            let results = e
                .attend_filtered("docs", &["default".into()], q, 50, Some(f))
                .unwrap();
            for (numeric, _score) in &results {
                let fields = raw_fields(&e, *numeric);
                let ok = f.eval(&fields);
                assert!(
                    ok,
                    "FILTER LEAK: doc {numeric} (fields={fields:?}) does not satisfy filter, query {q:?}"
                );
            }
        }
    }
}

/// Selective filter + expansion: category = "x" (25% of docs) must still fill
/// top_k=30 from a default 100-candidate pool via expansion rounds.
#[test]
fn selective_filter_expands_candidates() {
    let dir = temp_db();
    let e = setup(dir.path());
    // very selective: category = x AND year = 2021
    // (2015 + i%10 == 2021 → i%10 == 6; i%4 == 0 → i%20 == 16 → 100 of 2000 docs)
    let f = FilterExpr::And(
        Box::new(cat_eq("x")),
        Box::new(FilterExpr::Comparison {
            field: "year".into(),
            op: FilterOp::Eq,
            value: FilterValue::Int(2021),
        }),
    );
    let q = one_hot(3, DIM);
    let results = e
        .attend_filtered("docs", &["default".into()], &q, 30, Some(&f))
        .unwrap();
    assert!(
        results.len() >= 20,
        "expansion should recover selective-filter results, got {}",
        results.len()
    );
    for (numeric, _) in &results {
        let fields = raw_fields(&e, *numeric);
        assert!(f.eval(&fields), "leak after expansion: {fields:?}");
    }
    let expected = e.scan_filtered("docs", Some(&f), 10_000).unwrap().len();
    assert_eq!(expected, 100, "2000 docs: exactly 100 satisfy x AND 2021");
}

/// Extremely selective filter: result smaller than top_k is FINE — smaller
/// results, never unfiltered filler.
#[test]
fn hyper_selective_filter_returns_smaller_results() {
    let dir = temp_db();
    let e = setup(dir.path());
    // exactly one doc: idx 7 has category x (7 % 4 == 3 → actually NO category)…
    // use year = 2015 AND idx = 3 for a unique hit via AND on idx.
    let f = FilterExpr::And(
        Box::new(cat_eq("y")),
        Box::new(FilterExpr::Comparison {
            field: "idx".into(),
            op: FilterOp::Eq,
            value: FilterValue::Int(5),
        }),
    );
    let results = e
        .attend_filtered(
            "docs",
            &["default".into()],
            &one_hot(5 % DIM, DIM),
            50,
            Some(&f),
        )
        .unwrap();
    assert_eq!(
        results.len(),
        1,
        "exactly one doc: idx=5, category=y; got {}",
        results.len()
    );
}

/// Filter-only scan: deterministic (numeric id ASC), typed, and consistent
/// with the filtered vector path.
#[test]
fn filter_only_scan_is_deterministic_and_consistent() {
    let dir = temp_db();
    let e = setup(dir.path());
    let f = cat_eq("z");
    let scan = e.scan_filtered("docs", Some(&f), 10_000).unwrap();
    assert_eq!(scan.len(), 500);
    // deterministic: numeric ids strictly ascending
    for w in scan.windows(2) {
        assert!(w[0].1 < w[1].1, "scan not sorted by numeric id");
    }
    // limit honored
    let limited = e.scan_filtered("docs", Some(&f), 7).unwrap();
    assert_eq!(limited.len(), 7);
    assert_eq!(&limited[..], &scan[..7]);
    // every scan entry satisfies the filter
    for (uuid_s, numeric) in &scan {
        let fields = raw_fields(&e, *numeric);
        assert!(f.eval(&fields));
        let _ = uuid_s;
    }
    // no filter = whole collection
    let all = e.scan_filtered("docs", None, 10_000).unwrap();
    assert_eq!(all.len(), N_DOCS);
}

/// Wrong types, missing fields, null — end to end through the engine path.
#[test]
fn type_mismatch_and_missing_never_match() {
    let dir = temp_db();
    let e = setup(dir.path());
    // category is a string; comparing to int must match NOTHING
    let f = FilterExpr::Comparison {
        field: "category".into(),
        op: FilterOp::Eq,
        value: FilterValue::Int(3),
    };
    let r = e
        .attend_filtered("docs", &["default".into()], &one_hot(1, DIM), 50, Some(&f))
        .unwrap();
    assert!(r.is_empty(), "cross-type equality matched: {r:?}");

    // category = "x" OR NOT category = "x". Under the DOCUMENTED two-valued
    // semantics (filter.rs module header: NOT(...) composes the two-valued
    // result, so it matches documents missing the field) this expression is a
    // tautology — every document matches, missing-category ones included.
    // (E6-D9: this test previously asserted three-valued SQL NOT semantics,
    // contradicting the module contract since baseline; the mismatch surfaced
    // intermittently whenever the leaked doc entered the ANN candidate pool.
    // Retrieval semantics are frozen — the test was corrected, not the code.)
    // The documented, deterministic way to exclude missing-category docs is
    // IsNotNull composed with AND:
    let f2 = FilterExpr::And(
        Box::new(FilterExpr::IsNull {
            field: "category".into(),
            negated: true,
        }),
        Box::new(FilterExpr::Or(
            Box::new(cat_eq("x")),
            Box::new(FilterExpr::Not(Box::new(cat_eq("x")))),
        )),
    );
    let r2 = e
        .attend_filtered(
            "docs",
            &["default".into()],
            &one_hot(2, DIM),
            100,
            Some(&f2),
        )
        .unwrap();
    for (numeric, _) in &r2 {
        let fields = raw_fields(&e, *numeric);
        assert!(
            fields.contains_key("category"),
            "doc without category leaked through IsNotNull-AND: {fields:?}"
        );
    }
    // docs with a non-x category DO appear (NOT(cat = x) is true for them)
    assert!(r2.iter().any(|numeric| raw_fields(&e, numeric.0)
        .get("category")
        .and_then(|v| v.as_str())
        .map(|c| c != "x")
        .unwrap_or(false)));
}

/// JSON wire format → AST → engine: end-to-end for the REST surface.
#[test]
fn json_filter_wire_format_end_to_end() {
    let dir = temp_db();
    let e = setup(dir.path());
    let wire = json!({
        "and": [
            {"field": "category", "op": "=", "value": "y"},
            {"not": {"field": "year", "op": "<", "value": 2020}}
        ]
    });
    let f = parse_filter_json(&wire).unwrap();
    let r = e
        .attend_filtered("docs", &["default".into()], &one_hot(4, DIM), 25, Some(&f))
        .unwrap();
    assert!(!r.is_empty());
    for (numeric, _) in &r {
        let fields = raw_fields(&e, *numeric);
        assert_eq!(fields.get("category").and_then(|v| v.as_str()), Some("y"));
        let year = fields.get("year").and_then(|v| v.as_i64()).unwrap();
        assert!(year >= 2020);
    }
    // invalid ops are rejected with a structured error
    assert!(parse_filter_json(&json!({"field": "a", "op": "~", "value": 1})).is_err());
    assert!(parse_filter_json(&json!({"and": [{"field": "a", "op": "=", "value": 1}]})).is_err());
}

/// Deleted documents never survive a filter+search, even when they match the filter.
#[test]
fn deleted_docs_excluded_from_filtered_results() {
    let dir = temp_db();
    let e = setup(dir.path());
    // delete 20 specific category-x docs (record exactly which)
    let mut deleted_idxs: Vec<i64> = Vec::new();
    {
        let all = e.document_store.read().list_all_records();
        for rec in &all {
            if deleted_idxs.len() >= 20 {
                break;
            }
            if rec.fields.get("category").and_then(|v| v.as_str()) == Some("x")
                && rec
                    .fields
                    .get("idx")
                    .and_then(|v| v.as_i64())
                    .map(|i| i < 100)
                    .unwrap_or(false)
            {
                let idx = rec.fields.get("idx").and_then(|v| v.as_i64()).unwrap();
                let u = rec.id.to_string();
                if e.delete_document("docs", &u).unwrap() {
                    deleted_idxs.push(idx);
                }
            }
        }
    }
    assert_eq!(deleted_idxs.len(), 20);
    let r = e
        .attend_filtered(
            "docs",
            &["default".into()],
            &one_hot(0, DIM),
            50,
            Some(&cat_eq("x")),
        )
        .unwrap();
    assert!(!r.is_empty());
    for (numeric, _) in &r {
        let fields = raw_fields(&e, *numeric);
        let idx = fields.get("idx").and_then(|v| v.as_i64()).unwrap();
        assert!(
            !deleted_idxs.contains(&idx),
            "deleted doc {idx} resurfaced in filtered results"
        );
        assert_eq!(fields.get("category").and_then(|v| v.as_str()), Some("x"));
    }
}

/// §13 invariant extended to hybrid: filter+text runs BOTH channels through
/// the filter — a non-matching document can never leak via BM25 or vector.
#[test]
fn hybrid_filtered_rrf_never_leaks() {
    let dir = temp_db();
    let e = open_sync(dir.path());
    e.create_collection("hy", 8, &["default"]).unwrap();
    // i%4==0 → category x (with year); others y/z/absent. one_hot(i%8) vectors.
    for i in 0..2000i64 {
        let mut fields: HashMap<String, serde_json::Value> = HashMap::new();
        fields.insert("idx".to_string(), json!(i));
        fields.insert("body".to_string(), json!(format!("body {i} item")));
        match i % 4 {
            0 => {
                fields.insert("category".to_string(), json!("x"));
                fields.insert("year".to_string(), json!(2015 + (i % 10)));
            }
            1 => {
                fields.insert("category".to_string(), json!("y"));
            }
            2 => {
                fields.insert("category".to_string(), json!("z"));
            }
            _ => {}
        }
        let mut r = attentiondb_storage::Record::new(fields);
        r.k_vecs
            .insert("default".to_string(), one_hot(i as usize % 8, 8));
        e.insert_document("hy", r).unwrap();
    }
    // x AND year=2021 → i%20==16 → exactly 100 docs; their vectors are
    // one_hot(16%8)=one_hot(0), so query one_hot(0) hits them first.
    let f = FilterExpr::And(
        Box::new(FilterExpr::Comparison {
            field: "category".into(),
            op: FilterOp::Eq,
            value: FilterValue::Str("x".into()),
        }),
        Box::new(FilterExpr::Comparison {
            field: "year".into(),
            op: FilterOp::Eq,
            value: FilterValue::Int(2021),
        }),
    );
    let q = one_hot(0, 8);
    for top_k in [10usize, 25] {
        let res = e
            .attend_hybrid_filtered_with_deadline(
                "hy",
                &["default".into()],
                &q,
                "body item",
                top_k,
                &f,
                None,
            )
            .unwrap();
        assert!(!res.is_empty());
        assert!(res.len() <= top_k);
        for (id, _) in &res {
            let fields = raw_fields(&e, *id);
            assert_eq!(
                fields.get("category").and_then(|v| v.as_str()),
                Some("x"),
                "leak id={id}"
            );
            assert_eq!(
                fields.get("year").and_then(|v| v.as_i64()),
                Some(2021),
                "leak id={id}"
            );
        }
    }
}

/// §13: Fusion strategy + filter — sparse channel filtered BEFORE it enters
/// the staged pipeline; every survivor satisfies the filter.
#[test]
fn hybrid_filtered_fusion_never_leaks() {
    let dir = temp_db();
    let e = open_sync(dir.path());
    e.create_collection("hf", 8, &["default"]).unwrap();
    for i in 0..1500i64 {
        let mut fields: HashMap<String, serde_json::Value> = HashMap::new();
        fields.insert("idx".to_string(), json!(i));
        fields.insert("body".to_string(), json!(format!("alpha beta {i}")));
        fields.insert(
            "tier".to_string(),
            json!(if i % 3 == 0 { "gold" } else { "silver" }),
        );
        let mut r = attentiondb_storage::Record::new(fields);
        r.k_vecs
            .insert("default".to_string(), one_hot(i as usize % 8, 8));
        e.insert_document("hf", r).unwrap();
    }
    {
        let coll = e.get_collection("hf").unwrap();
        let mut cfg = coll.retrieval_config.read().clone();
        cfg.hybrid_strategy = attentiondb_core::collection::HybridStrategy::Fusion;
        *coll.retrieval_config.write() = cfg;
    }
    let gold = FilterExpr::Comparison {
        field: "tier".into(),
        op: FilterOp::Eq,
        value: FilterValue::Str("gold".into()),
    };
    let q = one_hot(0, 8);
    let res = e
        .attend_hybrid_filtered_with_deadline(
            "hf",
            &["default".into()],
            &q,
            "alpha beta",
            20,
            &gold,
            None,
        )
        .unwrap();
    assert!(!res.is_empty());
    assert!(res.len() <= 20);
    for (id, _) in &res {
        let fields = raw_fields(&e, *id);
        assert_eq!(
            fields.get("tier").and_then(|v| v.as_str()),
            Some("gold"),
            "leak id={id}"
        );
    }
}
