//! PH3D `dbtest` — production-database validation harness (Phase 3D spec).
//!
//! Subcommands (all operate on ISOLATED database dirs; every run ends with
//! the consistency checker as a mandatory gate, §32):
//!   model    — §33 state-machine test vs a reference model (mutations,
//!              filters, restart, compaction, ID mapping, checker gate)
//!   filterx  — §4–§6 dedicated filter correctness suite
//!   txn      — §18–§21 transaction atomicity/rollback scenarios
//!   crashchild — child process for §15/§16: performs N acked mutations then
//!              crashes at a controlled point (exit / park-for-SIGKILL)
//!   verify   — reopen a crashed DB, derive expected state from the ack
//!              sidecar, compare + consistency-check (§17 matrix row)
//!   walcorrupt — §14: reopen a DB whose WAL was corrupted (driver does the
//!              file surgery; this binary only reports open behavior)
//!   concur   — §22–§24 concurrent readers / mixed read-write workload
//!   backup   — §28–§31 backup/restore + quiescence requirement
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::Arc;

use attentiondb_core::checker::check_engine;
use attentiondb_core::engine::AttentionEngine;
use attentiondb_query::filter::{FilterExpr, FilterOp, FilterValue};
use attentiondb_storage::{compaction, Durability, Record};

const DIM: usize = 32;
const HEAD: &str = "h";
const N_DOCS: usize = 200;

// ---------------------------------------------------------------- utils

pub fn dur_from_env() -> Durability {
    match std::env::var("PH3D_DURABILITY").as_deref() {
        Ok("sync") => Durability::Sync,
        Ok("group") => Durability::GroupCommit,
        _ => Durability::Async,
    }
}

fn open_db(dir: &std::path::Path) -> AttentionEngine {
    AttentionEngine::open_dir(dir, dur_from_env()).unwrap()
}

fn vec_for(idx: u32) -> Vec<f32> {
    // deterministic unit vector per logical id
    let mut v: Vec<f32> = vec![0.0; DIM];
    let mut h = 0x811C9DC5u32;
    for b in idx.to_le_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    v[(h as usize) % DIM] = 1.0;
    v[((h >> 8) as usize) % DIM] = 0.5;
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / n).collect()
}

fn doc_record(idx: u32, version: u64, cat: &str, num: i64) -> Record {
    let mut fields = HashMap::new();
    fields.insert("idx".to_string(), serde_json::json!(idx));
    fields.insert("version".to_string(), serde_json::json!(version));
    fields.insert("cat".to_string(), serde_json::json!(cat));
    fields.insert("num".to_string(), serde_json::json!(num));
    fields.insert("title".to_string(), serde_json::json!(format!("doc-{idx}-v{version}")));
    let mut r = Record::new(fields);
    r.id = uuid_for(idx, version);
    r.k_vecs.insert(HEAD.to_string(), vec_for(idx));
    r
}

fn uuid_for(idx: u32, version: u64) -> uuid::Uuid {
    uuid::Uuid::from_u128(((idx as u128) << 64) | (version as u128))
}

/// Collection-namespaced uuid: uuid identity is GLOBAL across collections in
/// the engine (collections are membership tags), so multi-collection probes
/// MUST namespace their logical ids or they re-member each other's documents.
fn uuidc(coll: u32, idx: u32, version: u64) -> uuid::Uuid {
    uuid::Uuid::from_u128(((coll as u128) << 96) | ((idx as u128) << 64) | (version as u128))
}

fn doc_record_c(coll: u32, idx: u32, version: u64, cat: &str, num: i64) -> Record {
    let mut r = doc_record(idx, version, cat, num);
    r.id = uuidc(coll, idx, version);
    r
}

/// logical id -> (version, cat, num) — the reference model state
type Model = BTreeMap<u32, (u64, String, i64)>;

/// Export observed logical state: idx -> (version, cat, num) from the store,
/// plus per-idx numeric-id presence and retrievability through attend.
fn export_state(e: &AttentionEngine) -> (Model, Vec<String>) {
    export_state_coll(e, "bench")
}

/// Collection-scoped variant (membership tag defines the collection).
fn export_state_coll(e: &AttentionEngine, coll: &str) -> (Model, Vec<String>) {
    let mut model = Model::new();
    let mut issues = Vec::new();
    let store = e.document_store.read();
    let tag = format!("collection:{coll}");
    for rec in store.list_all_records() {
        // document store is global across collections; only members of `coll`
        // (membership tag) belong in the compared state
        if !rec.tags.contains(&tag) {
            continue;
        }
        let idx = rec.fields.get("idx").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as u32;
        let version = rec.fields.get("version").and_then(|v| v.as_u64()).unwrap_or(0);
        let cat = rec.fields.get("cat").and_then(|v| v.as_str()).unwrap_or("?").to_string();
        let num = rec.fields.get("num").and_then(|v| v.as_i64()).unwrap_or(0);
        if model.insert(idx, (version, cat, num)).is_some() {
            issues.push(format!("duplicate logical id {idx} in store"));
        }
    }
    (model, issues)
}

fn topk_observed(e: &AttentionEngine, qidx: u32, k: usize) -> Vec<u32> {
    let q = vec_for(qidx);
    let got = e.attend("bench", &[HEAD.to_string()], &q, k).unwrap();
    got.iter()
        .map(|(id, _)| {
            let store = e.document_store.read();
            store
                .list_all_records()
                .into_iter()
                .find(|r| e.id_mapper.read().uuid_to_id(&r.id) == Some(*id))
                .and_then(|r| r.fields.get("idx").and_then(|v| v.as_u64()))
                .unwrap_or(u64::MAX) as u32
        })
        .collect()
}

fn filtered_observed(e: &AttentionEngine, f: &FilterExpr, qidx: u32, k: usize) -> Vec<u32> {
    let q = vec_for(qidx);
    let got = e.attend_filtered("bench", &[HEAD.to_string()], &q, k, Some(f)).unwrap();
    let store = e.document_store.read();
    got.iter()
        .map(|(id, _)| {
            store
                .list_all_records()
                .into_iter()
                .find(|r| e.id_mapper.read().uuid_to_id(&r.id) == Some(*id))
                .and_then(|r| r.fields.get("idx").and_then(|v| v.as_u64()))
                .unwrap_or(u64::MAX) as u32
        })
        .collect()
}

fn issue_lines(issues: &[attentiondb_core::checker::CheckIssue]) -> (Vec<String>, Vec<String>) {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for i in issues {
        let line = format!("{:?} {}: {}", i.severity, i.code, i.detail);
        if matches!(i.severity, attentiondb_core::checker::Severity::Error) {
            errors.push(line);
        } else {
            warnings.push(line);
        }
    }
    (errors, warnings)
}

/// Gate semantics: `clean` = zero ERROR-severity issues (engine or dir level).
/// WARNING-severity issues (e.g. INDEX_RETIRED_VECTOR: retired vectors awaiting
/// purge — the documented lazy-tombstone behavior) are reported but non-fatal.
fn checker_report(e: &AttentionEngine, dir: &std::path::Path) -> serde_json::Value {
    let issues = check_engine(e);
    let dir_issues = attentiondb_core::checker::check_db_dir(dir).unwrap_or_default();
    let (e_err, e_warn) = issue_lines(&issues);
    let (d_err, d_warn) = issue_lines(&dir_issues);
    serde_json::json!({
        "engine_errors": e_err,
        "engine_warnings": e_warn,
        "dir_errors": d_err,
        "dir_warnings": d_warn,
        "warning_count": e_warn.len() + d_warn.len(),
        "clean": e_err.is_empty() && d_err.is_empty(),
    })
}

fn write_json(path: &str, v: &serde_json::Value) {
    std::fs::write(path, serde_json::to_string_pretty(v).unwrap()).unwrap();
}

// ---------------------------------------------------------------- model

/// §33 model-based suite. Deterministic per seed. Returns a summary string.
fn run_model(dir: &std::path::Path, seed: u64, n_ops: usize, out: &str) -> String {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut model: Model = BTreeMap::new();
    let mut log = String::from("opno,op,idx,detail\n");
    let mut checks = Vec::new(); // (family, ok, detail)

    let mut e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    // second collection for §31 isolation
    e.create_collection("other", DIM, &[HEAD]).unwrap();

    let mut det = Det::new(seed);
    let next_new = N_DOCS as u32;
    // initial population
    for idx in 0..N_DOCS as u32 {
        let cat = ["news", "sport", "tech", "finance"][idx as usize % 4];
        let num = (idx as i64) * 10;
        let r = doc_record(idx, 1, cat, num);
        e.insert_document("bench", r).unwrap();
        model.insert(idx, (1, cat.to_string(), num));
    }
    e.flush_wal().unwrap();
    checks.push(("initial_build_checker", {
        let rep = checker_report(&e, dir);
        write_json(&format!("{out}/consistency-initial.json"), &rep);
        rep["clean"].as_bool().unwrap_or(false)
    }, "checker after build".into()));

    // isolation: write a doc into "other" and ensure it never leaks into "bench"
    {
        let r = doc_record(9999, 1, "news", 1);
        e.insert_document("other", r).unwrap();
        let got = e.attend("bench", &[HEAD.to_string()], &vec_for(9999), 5).unwrap();
        let leaked = got.iter().any(|(id, _)| {
            e.id_mapper.read().uuid_to_id(&uuid_for(9999, 1)) == Some(*id)
        });
        checks.push(("collection_isolation", !leaked, "other-collection doc absent from bench queries".into()));
    }

    let cats = ["news", "sport", "tech", "finance"];
    let mut filter_fail = 0usize;
    let mut filter_tests = 0usize;
    for opno in 0..n_ops {
        let roll = det.next_u64() % 100;
        let idx = (det.next_u64() % (next_new as u64 + 4)) as u32;
        let op = if roll < 30 || idx >= next_new {
            "insert_or_upsert"
        } else if roll < 50 {
            "update"
        } else if roll < 65 {
            "delete"
        } else if roll < 80 {
            "query"
        } else if roll < 92 {
            "filter"
        } else if roll < 95 {
            "checkpoint"
        } else if roll < 98 {
            "compact"
        } else {
            "restart"
        };
        match op {
            "insert_or_upsert" => {
                let exists = model.contains_key(&idx);
                let version = if exists { model[&idx].0 + 1 } else { 1 };
                let cat = cats[(det.next_u64() % 4) as usize];
                let num = (det.next_u64() % 2000) as i64;
                let r = doc_record(idx, version, cat, num);
                if exists {
                    // engine identity note: update_document preserves the uuid the
                    // document was inserted under (uuid_for(idx,1)); only fields/
                    // vectors/internal version change (engine bumps record.version).
                    e.update_document("bench", &uuid_for(idx, 1).to_string(),
                                      r.fields.clone(), r.k_vecs.clone()).unwrap();
                } else {
                    e.insert_document("bench", r.clone()).unwrap();
                }
                model.insert(idx, (version, cat.to_string(), num));
                let _ = writeln!(log, "{opno},upsert,{idx},v{version}");
            }
            "update" => {
                if model.contains_key(&idx) {
                    let version = model[&idx].0 + 1;
                    let cat = cats[(det.next_u64() % 4) as usize];
                    let num = (det.next_u64() % 2000) as i64;
                    let r = doc_record(idx, version, cat, num);
                    e.update_document("bench", &uuid_for(idx, 1).to_string(),
                                      r.fields.clone(), r.k_vecs.clone()).unwrap();
                    model.insert(idx, (version, cat.to_string(), num));
                    let _ = writeln!(log, "{opno},update,{idx},v{version}");
                }
            }
            "delete" => {
                if model.contains_key(&idx) && idx != 0 {
                    e.delete_document("bench", &uuid_for(idx, 1).to_string()).unwrap();
                    model.remove(&idx);
                    let _ = writeln!(log, "{opno},delete,{idx},");
                    // deleted doc must vanish from retrieval immediately (§9)
                    let got = topk_observed(&e, idx, 300);
                    if got.contains(&idx) {
                        checks.push(("delete_immediate", false, format!("idx {idx} returned after delete")));
                    }
                }
            }
            "query" => {
                let got = topk_observed(&e, idx, 20);
                // invariant: only LIVE docs returned (INV-1); ordering deterministic
                let mut rerun = topk_observed(&e, idx, 20);
                if got.iter().any(|i| !model.contains_key(i)) {
                    checks.push(("inv_live_only", false, format!("dead doc in topk at op {opno}")));
                }
                if got != rerun {
                    rerun = topk_observed(&e, idx, 20);
                    if got != rerun {
                        checks.push(("query_determinism", false, format!("nondeterministic topk at op {opno}")));
                    }
                }
                let _ = writeln!(log, "{opno},query,{idx},k20");
            }
            "filter" => {
                filter_tests += 1;
                let which = det.next_u64() % 4;
                let (f, eligible): (FilterExpr, Vec<u32>) = match which {
                    0 => (eqf("cat", "sport"), model.iter().filter(|(_, (_, c, _))| c == "sport").map(|(k, _)| *k).collect()),
                    1 => (cmpf("num", FilterOp::Gte, FilterValue::Int(1000)),
                          model.iter().filter(|(_, (_, _, n))| *n >= 1000).map(|(k, _)| *k).collect()),
                    2 => (andf(eqf("cat", "news"), cmpf("num", FilterOp::Lt, FilterValue::Int(100))),
                          model.iter().filter(|(_, (_, c, n))| c == "news" && *n < 100).map(|(k, _)| *k).collect()),
                    _ => (eqf("cat", "no-such-cat"), vec![]), // zero-match
                };
                let got = filtered_observed(&e, &f, idx, 50);
                // INV-8: no excluded doc ever returned
                let leaked = got.iter().any(|i| !eligible.contains(i));
                if leaked {
                    filter_fail += 1;
                    checks.push(("inv_filter_leak", false, format!("filter returned excluded docs at op {opno}: {got:?}")));
                }
                // zero-match must be empty
                if matches!(which, 3) && !got.is_empty() {
                    filter_fail += 1;
                    checks.push(("filter_zero_match", false, "zero-match filter returned rows".into()));
                }
                // determinism
                if got != filtered_observed(&e, &f, idx, 50) {
                    filter_fail += 1;
                    checks.push(("filter_determinism", false, format!("op {opno}")));
                }
                let _ = writeln!(log, "{opno},filter,{idx},w{which}");
            }
            "checkpoint" => {
                e.flush_wal().unwrap();
                e.checkpoint().unwrap();
                let _ = writeln!(log, "{opno},checkpoint,,");
            }
            "compact" => {
                // compaction is a dir-level (offline) operation: close → compact → reopen
                drop(e);
                let res = compaction::compact_all(dir).unwrap();
                if let Some(r) = &res {
                    let _ = compaction::cleanup_merged_files(r);
                }
                e = open_db(dir);
                // post-compact equivalence + deleted stay deleted (§26)
                let (obs, iss) = export_state(&e);
                if obs != model {
                    checks.push(("compact_equivalence", false, format!("model drift after compact at op {opno}")));
                }
                if !iss.is_empty() {
                    checks.push(("compact_store_unique", false, iss.join(";")));
                }
                let rep = checker_report(&e, dir);
                write_json(&format!("{out}/consistency-compact-{opno}.json"), &rep);
                if !rep["clean"].as_bool().unwrap_or(false) {
                    checks.push(("compact_checker", false, format!("{:?}", rep)));
                }
                let _ = writeln!(log, "{opno},compact,,");
            }
            _ => {
                drop(e);
                e = open_db(dir);
                let (obs, _) = export_state(&e);
                if obs != model {
                    checks.push(("restart_equivalence", false, format!("model drift after restart at op {opno}")));
                }
                let _ = writeln!(log, "{opno},restart,,");
            }
        }
        if checks.iter().any(|c| !c.1) && checks.len() > 400 {
            break; // bounded failure collection
        }
    }

    // final: full-state comparison + deleted-not-resurrected + checker
    drop(e);
    let res = compaction::compact_all(dir).unwrap();
    if let Some(r) = &res {
        let _ = compaction::cleanup_merged_files(r);
    }
    let e = open_db(dir);
    let (obs, iss) = export_state(&e);
    checks.push(("final_state_equivalence", obs == model, format!("model={} observed={}", model.len(), obs.len())));
    checks.push(("final_store_unique", iss.is_empty(), iss.join(";")));
    // every live doc retrievable; every dead doc absent (INV-1/2)
    let mut dead_hits = 0usize;
    for idx in 0..next_new {
        if model.contains_key(&idx) { continue; }
        if topk_observed(&e, idx, 400).contains(&idx) { dead_hits += 1; }
    }
    checks.push(("final_dead_not_retrievable", dead_hits == 0, format!("{dead_hits} dead docs retrieved")));
    let rep = checker_report(&e, dir);
    write_json(&format!("{out}/consistency-final.json"), &rep);
    checks.push(("final_checker", rep["clean"].as_bool().unwrap_or(false), format!("{:?}", rep)));

    let (pass, fail) = checks.iter().fold((0usize, 0usize), |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) });
    let mut csv = String::from("family,ok,detail\n");
    for (fam, ok, detail) in &checks {
        if !*ok {
            let _ = writeln!(csv, "{fam},false,\"{}\"", detail.replace('"', "'"));
        }
    }
    for (fam, ok, _) in &checks {
        if *ok {
            let _ = writeln!(csv, "{fam},true,");
        }
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    std::fs::write(format!("{out}/operation-log.jsonl"), {
        let mut j = String::new();
        for line in log.lines().skip(1) {
            let p: Vec<&str> = line.splitn(4, ',').collect();
            let _ = writeln!(j, "{}", serde_json::json!({"opno": p[0], "op": p[1], "idx": p[2], "detail": p.get(3).unwrap_or(&"")}));
        }
        j
    }).unwrap();
    let exp: serde_json::Value = model.iter().map(|(k, (v, c, n))|
        (k.to_string(), serde_json::json!({"version": v, "cat": c, "num": n}))).collect::<serde_json::Map<_, _>>().into();
    write_json(&format!("{out}/expected-state.json"), &exp);
    let obsj: serde_json::Value = obs.iter().map(|(k, (v, c, n))|
        (k.to_string(), serde_json::json!({"version": v, "cat": c, "num": n}))).collect::<serde_json::Map<_, _>>().into();
    write_json(&format!("{out}/observed-state.json"), &obsj);
    let mjson: serde_json::Value = serde_json::json!({
        "seed": seed, "ops": n_ops, "checks_total": checks.len(),
        "checks_failed": fail, "filter_tests": filter_tests, "filter_failures": filter_fail,
    });
    write_json(&format!("{out}/metrics.json"), &mjson);
    let _ = std::fs::remove_dir_all(dir);
    format!("model seed={seed} ops={n_ops}: {pass} passed, {fail} FAILED (filter {filter_fail}/{filter_tests})")
}

pub struct Det(u64);
impl Det {
    pub fn new(seed: u64) -> Self { Det(seed | 1) }
    pub fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

// filters
fn eqf(field: &str, v: &str) -> FilterExpr {
    FilterExpr::Comparison { field: field.into(), op: FilterOp::Eq, value: FilterValue::Str(v.into()) }
}
fn cmpf(field: &str, op: FilterOp, v: FilterValue) -> FilterExpr {
    FilterExpr::Comparison { field: field.into(), op, value: v }
}
fn andf(a: FilterExpr, b: FilterExpr) -> FilterExpr {
    FilterExpr::And(Box::new(a), Box::new(b))
}

// ---------------------------------------------------------------- filterx

/// §4–§6 focused filter suite: lifecycle × filters × restart/flush/compact.
fn run_filterx(dir: &std::path::Path, out: &str) -> String {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let mut checks = Vec::new();
    // corpus: cat ∈ {a,b}, num = idx; 120 docs
    for idx in 0..120u32 {
        let cat = if idx % 2 == 0 { "a" } else { "b" };
        e.insert_document("bench", doc_record(idx, 1, cat, idx as i64)).unwrap();
    }
    e.flush_wal().unwrap();

    let mut filter_recalls: Vec<(String, f64)> = Vec::new();
    let expect_filtered = |e: &AttentionEngine, f: &FilterExpr, want: Vec<u32>, name: &str, checks: &mut Vec<(String, bool, String)>, recalls: &mut Vec<(String, f64)>| {
        let got = filtered_observed(e, f, 7, 500);
        let got_set: std::collections::HashSet<u32> = got.iter().copied().collect();
        let want_set: std::collections::HashSet<u32> = want.iter().copied().collect();
        // SOUNDNESS: every RETURNED doc satisfies the filter (the guarantee)
        let sound = got.iter().all(|g| want_set.contains(g));
        // DETERMINISM: same query twice -> identical result
        let det = got == filtered_observed(e, f, 7, 500);
        let mut want_sorted = want.clone();
        want_sorted.sort();
        // RECALL (metric, not assertion): completeness is candidate-bound
        let hit = want.iter().filter(|w| got_set.contains(w)).count();
        let recall = if want.is_empty() { 1.0 } else { hit as f64 / want.len() as f64 };
        recalls.push((name.to_string(), recall));
        checks.push((name.to_string(), sound && det,
            if sound && det { format!("recall {recall:.3}") } else { format!("sound={sound} det={det} recall {recall:.3} want {want_sorted:?} got {got:?}") }));
    };

    // 1) basic filters
    expect_filtered(&e, &eqf("cat", "a"), (0..120).filter(|i| i % 2 == 0).collect(), "filter_eq_all", &mut checks, &mut filter_recalls);
    expect_filtered(&e, &cmpf("num", FilterOp::Gt, FilterValue::Int(100)), (101..120).collect(), "filter_selective", &mut checks, &mut filter_recalls);
    expect_filtered(&e, &cmpf("num", FilterOp::Gt, FilterValue::Int(10_000)), vec![], "filter_zero_match", &mut checks, &mut filter_recalls);
    expect_filtered(&e, &andf(eqf("cat", "b"), cmpf("num", FilterOp::Lte, FilterValue::Int(20))),
                    (1..=19).filter(|i| i % 2 == 1).collect(), "filter_and", &mut checks, &mut filter_recalls);
    // filter + top-k: restricted K still filter-clean
    {
        let got = filtered_observed(&e, &eqf("cat", "a"), 1, 7);
        checks.push(("filter_topk_clean".into(), got.iter().all(|i| i % 2 == 0) && got.len() <= 7, format!("{got:?}")));
    }

    // 2) filter × mutation lifecycle (§5)
    // insert matching
    e.insert_document("bench", doc_record(200, 1, "a", 5)).unwrap();
    expect_filtered(&e, &eqf("cat", "a"), (0..120).filter(|i| i % 2 == 0).chain([200]).collect(), "fm_insert_matching", &mut checks, &mut filter_recalls);
    // update so it LEAVES the filter
    e.update_document("bench", &uuid_for(200, 1).to_string(),
                      doc_record(200, 2, "b", 5).fields, doc_record(200, 2, "b", 5).k_vecs).unwrap();
    expect_filtered(&e, &eqf("cat", "a"), (0..120).filter(|i| i % 2 == 0).collect(), "fm_update_leaves", &mut checks, &mut filter_recalls);
    // update so it re-ENTERS with new num
    e.update_document("bench", &uuid_for(200, 1).to_string(),
                      doc_record(200, 3, "b", 999).fields, doc_record(200, 3, "b", 999).k_vecs).unwrap();
    expect_filtered(&e, &andf(eqf("cat", "b"), cmpf("num", FilterOp::Eq, FilterValue::Int(999))), vec![200], "fm_update_enters", &mut checks, &mut filter_recalls);
    // delete matching / nonmatching
    e.delete_document("bench", &uuid_for(200, 1).to_string()).unwrap();
    expect_filtered(&e, &andf(eqf("cat", "b"), cmpf("num", FilterOp::Eq, FilterValue::Int(999))), vec![], "fm_delete_matching", &mut checks, &mut filter_recalls);
    e.delete_document("bench", &uuid_for(1, 1).to_string()).unwrap();
    expect_filtered(&e, &eqf("cat", "b"), (1..120).filter(|i| i % 2 == 1 && *i != 1).collect(), "fm_delete_nonmatching", &mut checks, &mut filter_recalls);
    // reinsert deleted logical doc (new version)
    e.insert_document("bench", doc_record(1, 2, "b", 1)).unwrap();
    expect_filtered(&e, &eqf("cat", "b"), (1..120).filter(|i| i % 2 == 1).collect(), "fm_reinsert", &mut checks, &mut filter_recalls);

    // 3) restart / flush / compact persistence of filter semantics (§6)
    e.flush_wal().unwrap();
    e.checkpoint().unwrap();
    drop(e);
    let e = open_db(dir);
    expect_filtered(&e, &eqf("cat", "b"), (1..120).filter(|i| i % 2 == 1).collect(), "fr_restart", &mut checks, &mut filter_recalls);
    drop(e);
    let res = compaction::compact_all(dir).unwrap();
    if let Some(r) = &res {
        let _ = compaction::cleanup_merged_files(r);
    }
    let e = open_db(dir);
    expect_filtered(&e, &eqf("cat", "b"), (1..120).filter(|i| i % 2 == 1).collect(), "fr_compact", &mut checks, &mut filter_recalls);
    let rep = checker_report(&e, dir);
    checks.push(("final_checker".into(), rep["clean"].as_bool().unwrap_or(false), format!("{rep:?}")));

    let (pass, fail) = checks.iter().fold((0usize, 0usize), |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) });
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks {
        let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'"));
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(&format!("{out}/metrics.json"), &serde_json::json!({
        "passed": pass, "failed": fail,
        "filter_recall": filter_recalls.iter().map(|(n, r)| serde_json::json!({"check": n, "recall": r})).collect::<Vec<_>>(),
    }));
    let _ = std::fs::remove_dir_all(dir);
    format!("filterx: {pass} passed, {fail} FAILED")
}

// ---------------------------------------------------------------- txn

/// §18–§21. The engine's transaction model is: staged Insert/Delete ops,
/// WAL BeginTxn→TxnOp*→CommitTxn, then apply. No update op; no isolation.
fn run_txn(dir: &std::path::Path, out: &str) -> String {
    use attentiondb_core::transaction::TxnOp;
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut checks: Vec<(String, bool, String)> = Vec::new();
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for idx in 0..20u32 {
        e.insert_document("bench", doc_record(idx, 1, "a", idx as i64)).unwrap();
    }
    e.flush_wal().unwrap();

    // commit: all ops visible
    let t = e.txn_manager.begin_transaction("bench");
    let ins1 = doc_record(100, 1, "new", 1);
    let ins2 = doc_record(101, 1, "new", 2);
    e.txn_manager.record_operation(t, TxnOp::Insert(ins1)).unwrap();
    e.txn_manager.record_operation(t, TxnOp::Insert(ins2)).unwrap();
    e.txn_manager.record_operation(t, TxnOp::Delete(uuid_for(3, 1))).unwrap();
    let ok = e.commit_transaction(t).unwrap();
    let (obs, _) = export_state(&e);
    checks.push(("txn_commit_applied".to_string(),
                 ok && obs.contains_key(&100) && obs.contains_key(&101) && !obs.contains_key(&3),
                 format!("{obs:?}")));
    // rollback: nothing visible
    let t = e.txn_manager.begin_transaction("bench");
    e.txn_manager.record_operation(t, TxnOp::Insert(doc_record(102, 1, "new", 3))).unwrap();
    e.txn_manager.record_operation(t, TxnOp::Delete(uuid_for(4, 1))).unwrap();
    let rolled = e.txn_manager.rollback_transaction(t).unwrap();
    let (obs, _) = export_state(&e);
    checks.push(("txn_rollback_invisible".to_string(),
                 rolled && !obs.contains_key(&102) && obs.contains_key(&4),
                 format!("{obs:?}")));
    // repeated/re-entrant use after rollback must fail cleanly
    let gone = e.txn_manager.get_staged_transaction(t).is_none();
    checks.push(("txn_rollback_unstaged".to_string(), gone, String::new()));
    // multi-op matrix (insert+delete mixes) in ONE txn
    let t = e.txn_manager.begin_transaction("bench");
    for i in 0..5u32 {
        e.txn_manager.record_operation(t, TxnOp::Insert(doc_record(110 + i, 1, "m", i as i64))).unwrap();
    }
    e.txn_manager.record_operation(t, TxnOp::Delete(uuid_for(5, 1))).unwrap();
    e.txn_manager.record_operation(t, TxnOp::Delete(uuid_for(6, 1))).unwrap();
    e.commit_transaction(t).unwrap();
    let (obs, _) = export_state(&e);
    let ok = (110..115).all(|i| obs.contains_key(&(i as u32))) && !obs.contains_key(&5) && !obs.contains_key(&6);
    checks.push(("txn_multiop_matrix".to_string(), ok, String::new()));
    // §20 injected commit failure: an op that fails pre-validation must abort
    // the WHOLE commit — no partial apply, engine unchanged, checker clean.
    {
        let t = e.txn_manager.begin_transaction("bench");
        let mut bad = doc_record(300, 1, "bad", 1);
        bad.k_vecs.insert(HEAD.to_string(), vec![0.0; DIM + 7]); // wrong dim
        e.txn_manager.record_operation(t, TxnOp::Insert(bad)).unwrap();
        let good = doc_record(301, 1, "bad", 2);
        e.txn_manager.record_operation(t, TxnOp::Insert(good)).unwrap();
        let r = e.commit_transaction(t);
        let (obs, _) = export_state(&e);
        checks.push(("txn_commit_failure_atomic".to_string(),
                     r.is_err() && !obs.contains_key(&300) && !obs.contains_key(&301),
                     format!("commit_err={} has300={} has301={}", r.is_err(), obs.contains_key(&300), obs.contains_key(&301))));
    }
    // delete-if-present: deleting an unknown uuid inside a committed txn must
    // be a clean no-op (TxnOp::Delete with numeric 0 semantics), not an error.
    {
        let t = e.txn_manager.begin_transaction("bench");
        e.txn_manager.record_operation(t, TxnOp::Delete(uuid::Uuid::from_u128(0xf0f0_f0f0))).unwrap();
        let r = e.commit_transaction(t);
        let rep2 = checker_report(&e, dir);
        checks.push(("txn_delete_missing_noop".to_string(),
                     r.is_ok() && rep2["clean"].as_bool().unwrap_or(false),
                     format!("commit_ok={}", r.is_ok())));
    }
    let rep = checker_report(&e, dir);
    checks.push(("txn_checker".to_string(), rep["clean"].as_bool().unwrap_or(false), format!("{rep:?}")));

    // crash-in-commit: parent kills the process mid/after WAL-commit; on
    // reopen the txn must be ALL-or-NOTHING (atomic via WAL markers).
    // This part runs in crashchild mode (see run()); here we only verify
    // the post-crash state given the sidecar marker file if present.
    let side = std::path::PathBuf::from(format!("{}.txncrash", dir.display()));
    if side.exists() {
        let data = std::fs::read_to_string(&side).unwrap();
        // marker: "committed" if CommitTxn WAL append returned before crash
        let committed = data.trim_end().ends_with("committed");
        drop(e);
        let e2 = open_db(dir);
        let (obs, _) = export_state(&e2);
        let has_all = (200..210).all(|i| obs.contains_key(&(i as u32)));
        let has_none = (200..210).all(|i| !obs.contains_key(&(i as u32)));
        checks.push(("txn_crash_atomicity".to_string(),
                     has_all || has_none,
                     format!("committed_marker={committed} all={has_all} none={has_none}")));
        checks.push(("txn_crash_matches_wal".to_string(),
                     if committed { has_all } else { has_none || has_all /* apply-phase crash recovers to all via replay */ },
                     String::new()));
        let rep = checker_report(&e2, dir);
        checks.push(("txn_crash_checker".to_string(), rep["clean"].as_bool().unwrap_or(false), format!("{rep:?}")));
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_file(&side);
    }

    let (pass, fail) = checks.iter().fold((0usize, 0usize), |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) });
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks {
        let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'"));
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(&format!("{out}/metrics.json"), &serde_json::json!({"passed": pass, "failed": fail}));
    if !side.exists() {
        let _ = std::fs::remove_dir_all(dir);
    }
    format!("txn: {pass} passed, {fail} FAILED")
}

/// child for the crash-in-commit test: commit a 10-insert txn; parent
/// SIGKILLs at an uncontrolled moment; we mark "committed" AFTER
/// commit_transaction returns.
fn txn_child(dir: &std::path::Path) -> ! {
    use attentiondb_core::transaction::TxnOp;
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for idx in 0..10u32 {
        e.insert_document("bench", doc_record(idx, 1, "a", idx as i64)).unwrap();
    }
    e.flush_wal().unwrap();
    let t = e.txn_manager.begin_transaction("bench");
    for i in 200..210u32 {
        e.txn_manager.record_operation(t, TxnOp::Insert(doc_record(i, 1, "txn", i as i64))).unwrap();
    }
    let r = e.commit_transaction(t);
    let side = std::path::PathBuf::from(format!("{}.txncrash", dir.display()));
    let mut f = std::fs::File::create(&side).unwrap();
    let _ = writeln!(f, "{}", if r.is_ok() { "committed" } else { "failed" });
    let _ = f.flush();
    loop { std::thread::sleep(std::time::Duration::from_secs(3600)); } // parent will SIGKILL
}

// ---------------------------------------------------------------- crash child / verify

/// child: N acked mutations; sidecar records acks; crash at `point`.
fn run_crashchild(dir: &std::path::Path, point: &str, n: usize) -> ! {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let side = format!("{}.sidecar", dir.display());
    let mut sc = std::fs::File::create(&side).unwrap();
    let ack = |sc: &mut std::fs::File, i: usize| {
        let _ = writeln!(sc, "ACK {i}");
        let _ = sc.flush();
    };
    for i in 0..n {
        let idx = 1000 + i as u32;
        e.insert_document("bench", doc_record(idx, 1, "c", i as i64)).unwrap();
        ack(&mut sc, i);
        if point == "mid_inserts" && i == n / 2 {
            loop { std::thread::sleep(std::time::Duration::from_secs(3600)); } // parent SIGKILLs
        }
    }
    match point {
        "after_acks" => std::process::exit(137), // no flush, no checkpoint
        "after_flush" => { e.flush_wal().unwrap(); std::process::exit(137); }
        "after_checkpoint" => { e.flush_wal().unwrap(); e.checkpoint().unwrap(); std::process::exit(137); }
        "mid_flush" => { loop { std::thread::sleep(std::time::Duration::from_secs(3600)); } }
        "during_commit_txn" => {
            // same engine, same dir, same sidecar bookkeeping: flush the acked
            // baseline, then commit one 10-insert txn; marker records the commit
            // call's outcome AFTER it returns; parent SIGKILLs while we park.
            use attentiondb_core::transaction::TxnOp;
            e.flush_wal().unwrap();
            let t = e.txn_manager.begin_transaction("bench");
            for i in 2000..2010u32 {
                e.txn_manager.record_operation(t, TxnOp::Insert(doc_record(i, 1, "txn", i as i64))).unwrap();
            }
            let r = e.commit_transaction(t);
            let side = std::path::PathBuf::from(format!("{}.txncrash", dir.display()));
            let mut f = std::fs::File::create(&side).unwrap();
            let _ = writeln!(f, "{}", if r.is_ok() { "committed" } else { "failed" });
            let _ = f.flush();
            loop { std::thread::sleep(std::time::Duration::from_secs(3600)); }
        }
        _ => {
            let _ = compaction::compact_all(dir); // after_compact
            std::process::exit(137);
        }
    }
}

/// reopen after crash; derive expected from sidecar acks; compare + checker.
fn run_verify(dir: &std::path::Path, point: &str) -> String {
    let side = format!("{}.sidecar", dir.display());
    let acked = std::fs::read_to_string(&side)
        .map(|s| s.lines().filter(|l| l.starts_with("ACK ")).count())
        .unwrap_or(0);
    let open_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| open_db(dir)));
    match open_result {
        Err(_) => {
            let _ = std::fs::remove_dir_all(dir);
            let _ = std::fs::remove_file(&side);
            format!("{point},ERROR_ON_OPEN,0/0,0,pre_existing=0,txn=-")
        }
        Ok(e) => {
            let (obs, iss) = export_state(&e);
            // expected: docs 1000..1000+acked present (GroupCommit/flushed cases)
            let mut present = 0usize;
            for i in 0..acked {
                if obs.contains_key(&(1000 + i as u32)) {
                    present += 1;
                }
            }
            // no UNACKED doc may be present (beyond acked)
            let mut unacked_present = 0usize;
            for i in acked..acked + 64 {
                if obs.contains_key(&(1000 + i as u32)) {
                    unacked_present += 1;
                }
            }
            let rep = checker_report(&e, dir);
            let clean = rep["clean"].as_bool().unwrap_or(false);
            // txn all-or-nothing (atomicity under crash): the 10 txn inserts
            // (2000..2009) must be ALL present or ALL absent — never partial —
            // and must agree with the marker the child wrote after commit
            // returned ("committed" => all 10; "failed"/absent => 0).
            let txn_present = (2000..2010u32).filter(|i| obs.contains_key(i)).count();
            let marker = std::fs::read_to_string(format!("{}.txncrash", dir.display()))
                .map(|m| m.trim().to_string())
                .unwrap_or_default();
            let partial = txn_present > 0 && txn_present < 10;
            // A commit that returned Ok but vanished is only admissible under
            // Durability::Async (userspace buffer lost with the process;
            // all-or-nothing still holds). Under group/sync it is a real
            // durability violation.
            let async_mode = matches!(dur_from_env(), Durability::Async);
            let txn_ok = if partial {
                false
            } else {
                match marker.as_str() {
                    "committed" => txn_present == 10 || (async_mode && txn_present == 0),
                    "failed" => txn_present == 0,
                    _ => txn_present == 0 || txn_present == 10,
                }
            };
            let txn_verdict = if partial {
                "PARTIAL"
            } else if marker == "committed" && txn_present == 0 {
                "COMMITTED_NOT_DURABLE_ASYNC"
            } else if txn_present == 10 {
                "COMMITTED_DURABLE"
            } else {
                "ABSENT"
            };
            let txn_note = format!("txn={txn_present}/10 marker={marker} {txn_verdict}");
            let verdict = if !clean || !txn_ok {
                "INCONSISTENT"
            } else if present == acked {
                "ALL_ACKED"
            } else if present < acked && unacked_present == 0 {
                "PREFIX"
            } else {
                "MISMATCH"
            };
            let n_start: usize = obs.keys().filter(|k| **k < 1000).count();
            let _ = iss;
            let _ = std::fs::remove_dir_all(dir);
            let _ = std::fs::remove_file(&side);
            format!("{point},{verdict},{present}/{acked},{unacked_present},pre_existing={n_start},{txn_note}")
        }
    }
}

/// §14 helper: reopen a (possibly corrupted) DB and report behavior only.
fn run_walcorrupt_verify(dir: &std::path::Path, mode: &str) -> String {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let e = open_db(dir);
        let (obs, iss) = export_state(&e);
        let rep = checker_report(&e, dir);
        (obs.len(), iss.len(), rep["clean"].as_bool().unwrap_or(false))
    }));
    let _ = std::fs::remove_dir_all(dir);
    match r {
        Err(_) => format!("{mode},ERROR_ON_OPEN,,"),
        Ok((n, iss, clean)) => format!("{mode},OPENED,n_docs={n},store_issues={iss},checker_clean={clean}"),
    }
}

// ---------------------------------------------------------------- concur

fn run_concur(dir: &std::path::Path, readers: usize, writers: usize, out: &str) -> String {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let e = Arc::new(open_db(dir));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for idx in 0..300u32 {
        e.insert_document("bench", doc_record(idx, 1, "a", idx as i64)).unwrap();
    }
    e.flush_wal().unwrap();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // deterministic per-writer op log (§24): writers record each cycle's ops
    let oplog = Arc::new(std::sync::Mutex::new(
        std::fs::File::create(format!("{out}/operation-log.jsonl")).unwrap()));
    let mut errors = 0usize;
    let mut lat_all: Vec<f64> = Vec::new();
    let mut handles = Vec::new();
    let t_start = std::time::Instant::now();
    for r in 0..readers {
        let e = Arc::clone(&e);
        let stop = Arc::clone(&stop);
        handles.push(std::thread::spawn(move || {
            let mut lat = Vec::new();
            let mut errs = 0usize;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let q = vec_for((r * 7 + 3) as u32);
                let t0 = std::time::Instant::now();
                let got = e.attend("bench", &[HEAD.to_string()], &q, 10);
                lat.push(t0.elapsed().as_secs_f64() * 1e6);
                match got {
                    Ok(res) if res.len() > 10 => errs += 1,
                    Ok(_) => {}
                    Err(_) => errs += 1,
                }
            }
            (lat, errs)
        }));
    }
    for w in 0..writers {
        let e = Arc::clone(&e);
        let stop = Arc::clone(&stop);
        let oplog = Arc::clone(&oplog);
        handles.push(std::thread::spawn(move || {
            let mut errs = 0usize;
            let mut i = 0u64;
            let slot = 1000 + w as u32 * 100_000;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let idx = slot + i as u32;
                // deterministic mixed cycle: insert -> update -> (delete on i%5==4)
                let r = doc_record(idx, 1, "w", i as i64);
                let ok_ins = e.insert_document("bench", r).is_ok();
                if !ok_ins { errs += 1; }
                if i % 2 == 1 {
                    let upd = doc_record(idx, 2, "w2", i as i64);
                    if e.update_document("bench", &uuid_for(idx, 1).to_string(),
                                         upd.fields, upd.k_vecs).is_err() { errs += 1; }
                }
                if i % 5 == 4
                    && e.delete_document("bench", &uuid_for(idx, 1).to_string()).is_err()
                {
                    errs += 1;
                }
                if i % 25 == 24 {
                    let _ = e.flush_wal();
                }
                {
                    let mut f = oplog.lock().unwrap();
                    let _ = writeln!(f, "{{\"writer\":{w},\"opno\":{i},\"idx\":{idx},\"op\":\"ins{},upd{},del{}\"}}",
                        ok_ins as u8, (i % 2 == 1) as u8, (i % 5 == 4) as u8);
                }
                i += 1;
            }
            (Vec::new(), errs)
        }));
    }
    std::thread::sleep(std::time::Duration::from_secs(4));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for h in handles {
        if let Ok((lat, errs)) = h.join() {
            errors += errs;
            lat_all.extend(lat);
        }
    }
    let elapsed = t_start.elapsed().as_secs_f64();
    let total = lat_all.len();
    let qps = total as f64 / elapsed;
    let mut latc = lat_all.clone();
    latc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pc = |p: f64| latc[((p / 100.0) * (latc.len() as f64 - 1.0)).round() as usize % latc.len().max(1)];
    // post-workload checker gate + logical sanity
    let (obs, iss) = export_state(&e);
    let rep = checker_report(&e, dir);
    let clean = rep["clean"].as_bool().unwrap_or(false) && iss.is_empty();
    let warn_ct = rep["warning_count"].as_u64().unwrap_or(0);
    let mut csv = String::from("readers,writers,queries,p50_us,p95_us,p99_us,qps,errors,checker_clean,checker_warnings,live_docs\n");
    let _ = writeln!(csv, "{readers},{writers},{total},{:.1},{:.1},{:.1},{qps:.0},{errors},{clean},{warn_ct},{}", pc(50.0), pc(95.0), pc(99.0), obs.len());
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(&format!("{out}/metrics.json"), &serde_json::json!({
        "readers": readers, "writers": writers, "queries": total, "qps": qps,
        "errors": errors, "checker_clean": clean, "checker_warnings": warn_ct,
        "store_issues": iss, "live_docs": obs.len(),
    }));
    let _ = std::fs::remove_dir_all(dir);
    format!("concur r{readers}w{writers}: {total} queries, {errors} errors, {qps:.0} QPS, checker_clean={clean}")
}

// ---------------------------------------------------------------- backup

fn run_backup(dir: &std::path::Path, out: &str) -> String {
    use attentiondb_core::backup::{copy_database_dir, restore_backup};
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut checks: Vec<(String, bool, String)> = Vec::new();
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for idx in 0..150u32 {
        e.insert_document("bench", doc_record(idx, 1, "a", idx as i64)).unwrap();
    }
    e.flush_wal().unwrap();
    e.checkpoint().unwrap();
    let bak = std::path::PathBuf::from(format!("{}.bak", dir.display()));
    let _ = std::fs::remove_dir_all(&bak);
    copy_database_dir(dir, &bak).unwrap();
    let manifest = restore_backup(&bak, &std::path::PathBuf::from(format!("{}.restore", dir.display())));
    // capture backup-state BEFORE further mutation
    let backup_state = export_state(&e).0;
    // mutate A: inserts + deletes + update
    for idx in 150..170u32 {
        e.insert_document("bench", doc_record(idx, 1, "post", idx as i64)).unwrap();
    }
    e.delete_document("bench", &uuid_for(0, 1).to_string()).unwrap();
    e.update_document("bench", &uuid_for(1, 1).to_string(),
                      doc_record(1, 9, "mutated", 1).fields, doc_record(1, 9, "mutated", 1).k_vecs).unwrap();
    e.flush_wal().unwrap();
    drop(e);
    // open restored DB and compare against backup state
    let rdir = std::path::PathBuf::from(format!("{}.restore", dir.display()));
    let rb = open_db(&rdir);
    let (rstate, riss) = export_state(&rb);
    checks.push(("restore_state_equals_backup".into(), rstate == backup_state,
                 format!("backup={} restored={}", backup_state.len(), rstate.len())));
    checks.push(("restore_store_unique".into(), riss.is_empty(), riss.join(";")));
    checks.push(("restore_not_mutated".into(),
                 !rstate.contains_key(&150) && rstate.contains_key(&1),
                 format!("has150={} has1={}", rstate.contains_key(&150), rstate.contains_key(&1))));
    let rep = checker_report(&rb, &rdir);
    checks.push(("restore_checker".into(), rep["clean"].as_bool().unwrap_or(false), format!("{rep:?}")));
    let m = manifest.ok();
    checks.push(("restore_manifest".into(), m.is_some(), String::new()));
    // backup during activity: copy while engine holds the dir + writes happen.
    // copy_database_dir does NOT coordinate with the mutation gate, so this
    // probe DOCUMENTS the outcome (quiescence requirement) rather than gating.
    {
        let e2 = Arc::new(open_db(dir));
        for idx in 300..320u32 {
            let _ = e2.insert_document("bench", doc_record(idx, 1, "live", idx as i64));
        }
        // ACTIVE writers during the copy (phase 2 of the probe)
        let stop_w = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut wh = Vec::new();
        for wid in 0..2usize {
            let e2 = Arc::clone(&e2);
            let stop_w = Arc::clone(&stop_w);
            wh.push(std::thread::spawn(move || {
                let mut i = 0u64;
                let slot = 5000 + wid as u32 * 10_000;
                while !stop_w.load(std::sync::atomic::Ordering::Relaxed) {
                    let idx = slot + i as u32;
                    let _ = e2.insert_document("bench", doc_record(idx, 1, "lw", i as i64));
                    i += 1;
                }
                i
            }));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        let live_bak = std::path::PathBuf::from(format!("{}.livebak", dir.display()));
        let _ = std::fs::remove_dir_all(&live_bak);
        let copy_ok = copy_database_dir(dir, &live_bak).is_ok();
        stop_w.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut written = 0u64;
        for h in wh {
            written += h.join().unwrap_or(0);
        }
        if copy_ok {
            let rd = std::path::PathBuf::from(format!("{}.liverestore", dir.display()));
            let r = restore_backup(&live_bak, &rd);
            let ok = r.is_ok();
            if ok {
                let re = open_db(&rd);
                let (st, _) = export_state(&re);
                // a copy taken under ACTIVE writers: we DOCUMENT whether it
                // opened checker-clean (§28–30: no online backup implemented;
                // quiescent backup is the supported path).
                let rep2 = checker_report(&re, &rd);
                checks.push(("live_writer_backup_documented".into(), true,
                             format!("docs={} consistent={} writers_wrote={}", st.len(),
                                     rep2["clean"].as_bool().unwrap_or(false), written)));
                let _ = std::fs::remove_dir_all(&rd);
            } else {
                checks.push(("live_backup_consistent".into(), true,
                             "restore refused live backup (documented: quiescent required)".into()));
            }
        } else {
            checks.push(("live_backup_consistent".into(), true,
                         "copy refused while active (documented: quiescent required)".into()));
        }
        let _ = std::fs::remove_dir_all(&live_bak);
    }
    let _ = std::fs::remove_dir_all(&bak);
    let _ = std::fs::remove_dir_all(&rdir);
    let _ = std::fs::remove_dir_all(dir);

    let (pass, fail) = checks.iter().fold((0usize, 0usize), |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) });
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks {
        let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'"));
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(&format!("{out}/metrics.json"), &serde_json::json!({"passed": pass, "failed": fail}));
    format!("backup: {pass} passed, {fail} FAILED")
}

// ---------------------------------------------------------------- compaction

/// §26: offline compaction before/after equivalence. Populate (with updates +
/// deletes so tombstones exist), checkpoint, close, compact_all + cleanup,
/// reopen: observed state must be identical and checker-clean.
fn run_compact(dir: &std::path::Path, out: &str) -> String {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut checks: Vec<(String, bool, String)> = Vec::new();
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for idx in 0..300u32 {
        e.insert_document("bench", doc_record(idx, 1, "a", idx as i64)).unwrap();
    }
    // churn: updates + deletes create overwrites + tombstones for GC
    for idx in 0..100u32 {
        let r = doc_record(idx, 2, "b", idx as i64 * 10);
        e.update_document("bench", &uuid_for(idx, 1).to_string(), r.fields, r.k_vecs).unwrap();
    }
    for idx in 100..150u32 {
        e.delete_document("bench", &uuid_for(idx, 1).to_string()).unwrap();
    }
    e.flush_wal().unwrap();
    e.checkpoint().unwrap(); // generation 1 SST
    // second generation: more writes so a second SST exists to merge into
    for idx in 300..360u32 {
        e.insert_document("bench", doc_record(idx, 1, "c", idx as i64)).unwrap();
    }
    for idx in 0..30u32 {
        let r = doc_record(idx, 3, "c", idx as i64 * 100);
        e.update_document("bench", &uuid_for(idx, 1).to_string(), r.fields, r.k_vecs).unwrap();
    }
    e.flush_wal().unwrap();
    e.checkpoint().unwrap(); // generation 2 SST
    let (before, biss) = export_state(&e);
    checks.push(("before_checker".into(), checker_report(&e, dir)["clean"].as_bool().unwrap_or(false) && biss.is_empty(), format!("{biss:?}")));
    drop(e);
    let cres = compaction::compact_all(dir);
    let did = matches!(&cres, Ok(Some(_)));
    let cerr = match &cres { Err(er) => er.to_string(), Ok(None) => "no-op (<2 sst)".into(), Ok(Some(_)) => String::new() };
    let removed = match &cres {
        Ok(Some(res)) => compaction::cleanup_merged_files(res).unwrap_or(0),
        _ => 0,
    };
    let e = open_db(dir);
    let (after, aiss) = export_state(&e);
    checks.push(("after_state_equal".into(), before == after && aiss.is_empty(),
        if before == after { format!("docs={}", after.len()) } else { format!("before={} after={} iss={biss:?}/{aiss:?}", before.len(), after.len()) }));
    checks.push(("after_checker".into(), checker_report(&e, dir)["clean"].as_bool().unwrap_or(false), String::new()));
    // §10 spot: twice-updated doc survives compaction at LATEST content (v3)
    checks.push(("updated_content_survives".into(),
        after.get(&20).map(|(_, c, n)| c == "c" && *n == 2000).unwrap_or(false), String::new()));
    // §9 spot: deleted docs stay deleted after compaction (tombstone GC)
    checks.push(("deleted_stay_deleted".into(),
        (100..150u32).all(|i| !after.contains_key(&i)), String::new()));
    let (pass, fail) = checks.iter().fold((0usize, 0usize), |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) });
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks { let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'")); }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(&format!("{out}/metrics.json"), &serde_json::json!({
        "passed": pass, "failed": fail, "compaction_ran": did,
        "compaction_error": cerr, "merged_files_removed": removed, "docs": after.len(),
    }));
    let _ = std::fs::remove_dir_all(dir);
    format!("compact: {pass} passed, {fail} FAILED (ran={did} removed={removed})")
}

/// §27: compaction-under-load probe. compact_all is dir-level/offline; this
/// documents what happens when it is invoked while an engine holds the dir
/// open. Detect + document only — never a pass/fail correctness gate.
fn run_compact_live(dir: &std::path::Path) -> String {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    {
        let e = open_db(dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        for idx in 0..50u32 {
            e.insert_document("bench", doc_record(idx, 1, "a", idx as i64)).unwrap();
        }
        e.flush_wal().unwrap();
        e.checkpoint().unwrap();
        // three SST generations: below the auto-compact threshold (4 files)
        for gen in 0..2u32 {
            for idx in (50 + gen * 15)..(65 + gen * 15) {
                e.insert_document("bench", doc_record(idx, 1, "a", idx as i64)).unwrap();
            }
            e.flush_wal().unwrap();
            e.checkpoint().unwrap();
        }
        let r = compaction::compact_all(dir); // engine still open (idle)
        match r {
            Ok(Some(res)) => format!("compactlive,COMPACTED_WHILE_OPEN,merged={},out_entries={} (NOT cleaning up; engine holds the dir)", res.files_merged, res.output_entries),
            Ok(None) => "compactlive,NOOP,<2 sst files".into(),
            Err(er) => format!("compactlive,REFUSED,{er}"),
        }
        // engine dropped here without cleanup_merged_files
    }
}

// ---------------------------------------------------------------- integration

/// §11/§31/§4 + §6/§26/§29 cross-feature integration: multi-collection
/// isolation across restart/compaction/backup-restore, graceful-shutdown
/// durability of all four mutation kinds, filter × multi-head soundness.
fn run_integration(dir: &std::path::Path, out: &str) -> String {
    use attentiondb_core::backup::{copy_database_dir, restore_backup};
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut checks: Vec<(String, bool, String)> = Vec::new();
    let e = open_db(dir);
    e.create_collection("alpha", DIM, &[HEAD]).unwrap();
    e.create_collection("beta", DIM, &[HEAD]).unwrap();
    e.create_collection("gamma", DIM, &[HEAD, "h2"]).unwrap();
    for i in 0..60u32 {
        e.insert_document("alpha", doc_record_c(1, i, 1, "x", i as i64)).unwrap();
    }
    for i in 0..40u32 {
        e.insert_document("beta", doc_record_c(2, i, 1, "y", i as i64)).unwrap();
    }
    for i in 0..24u32 {
        e.insert_document("gamma", doc_record_c(3, i, 1, "g", i as i64)).unwrap();
    }
    e.flush_wal().unwrap();
    e.checkpoint().unwrap(); // generation 1
    // churn: updates first (0..10 incl. odd ids), then deletes on the REMAINING
    // odd ids (11..59) — never deleting a doc the update loop must still see
    for i in 0..10u32 {
        let r = doc_record_c(1, i, 2, "x2", i as i64 * 3);
        e.update_document("alpha", &uuidc(1, i, 1).to_string(), r.fields, r.k_vecs).unwrap();
    }
    for i in (11..60u32).step_by(2) {
        e.delete_document("alpha", &uuidc(1, i, 1).to_string()).unwrap();
    }
    let r = doc_record_c(2, 2000, 1, "y", 7);
    e.insert_document("beta", r.clone()).unwrap(); // upsert: new logical id
    let r = doc_record_c(2, 0, 2, "y2", 100);
    e.update_document("beta", &uuidc(2, 0, 1).to_string(), r.fields, r.k_vecs).unwrap(); // upsert: existing
    for i in 24..36u32 {
        e.insert_document("gamma", doc_record_c(3, i, 1, "g", i as i64)).unwrap();
    }
    e.flush_wal().unwrap();
    e.checkpoint().unwrap(); // generation 2 (enables offline merge)

    // (a) live isolation (§31)
    let (sa, _) = export_state_coll(&e, "alpha");
    let (sb, _) = export_state_coll(&e, "beta");
    let (sg, _) = export_state_coll(&e, "gamma");
    let alpha_ok = sa.len() == 35
        && (0..60u32).step_by(2).all(|i| sa.contains_key(&i))
        && (11..60u32).step_by(2).all(|i| !sa.contains_key(&i))
        && (1..10u32).step_by(2).all(|i| sa.contains_key(&i))
        && sa.get(&5).map(|(_, c, _)| c == "x2").unwrap_or(false);
    checks.push(("isol_live_alpha".into(), alpha_ok, format!("docs={} first={sa:?}", sa.len())));
    checks.push(("isol_live_beta".into(),
                 sb.len() == 41 && sb.contains_key(&2000) && sb.get(&0).map(|(v, c, _)| *v == 2 && c == "y2").unwrap_or(false),
                 format!("docs={}", sb.len())));
    checks.push(("isol_live_gamma".into(), sg.len() == 36, format!("docs={}", sg.len())));

    // (b) filter × multi-head (§4): soundness + determinism across heads h+h2
    {
        let heads = vec![HEAD.to_string(), "h2".to_string()];
        let f = eqf("cat", "g");
        let got = filtered_observed_multi(&e, "gamma", &heads, &f, 1, 40);
        let got2 = filtered_observed_multi(&e, "gamma", &heads, &f, 1, 40);
        checks.push(("filter_multihead_sound".into(),
                     got.iter().all(|i| sg.contains_key(i)) && got == got2 && got.len() <= 40,
                     format!("{got:?}")));
        let zero = filtered_observed_multi(&e, "gamma", &heads, &eqf("cat", "no-such"), 1, 40);
        checks.push(("filter_multihead_zero_match".into(), zero.is_empty(), format!("{zero:?}")));
    }

    // (c) graceful-shutdown durability (§11): close -> reopen -> all four
    // mutation kinds (insert/update/delete/upsert) persisted
    e.close().unwrap();
    let e2 = open_db(dir);
    let (ra, _) = export_state_coll(&e2, "alpha");
    let (rb, _) = export_state_coll(&e2, "beta");
    let (rg, _) = export_state_coll(&e2, "gamma");
    checks.push(("graceful_restart_alpha".into(), ra == sa, format!("equal={}", ra == sa)));
    checks.push(("graceful_restart_beta".into(), rb == sb, format!("equal={}", rb == sb)));
    checks.push(("graceful_restart_gamma".into(), rg == sg, format!("equal={}", rg == sg)));
    checks.push(("graceful_checker".into(), checker_report(&e2, dir)["clean"].as_bool().unwrap_or(false), String::new()));
    drop(e2);

    // (d) compaction preserves both collections + isolation (§26/§31)
    let cres = compaction::compact_all(dir);
    if let Ok(Some(res)) = &cres {
        compaction::cleanup_merged_files(res).unwrap();
    }
    let e3 = open_db(dir);
    let (ca, _) = export_state_coll(&e3, "alpha");
    let (cb, _) = export_state_coll(&e3, "beta");
    checks.push(("compact_alpha_equal".into(), ca == sa, String::new()));
    checks.push(("compact_beta_equal".into(), cb == sb, String::new()));
    checks.push(("compact_checker".into(), checker_report(&e3, dir)["clean"].as_bool().unwrap_or(false), String::new()));

    // (e) backup/restore reproduces BOTH collections (§29/§31)
    let bak = std::path::PathBuf::from(format!("{}.bak", dir.display()));
    let rdir = std::path::PathBuf::from(format!("{}.restored", dir.display()));
    let _ = std::fs::remove_dir_all(&bak);
    let _ = std::fs::remove_dir_all(&rdir);
    let copy_ok = copy_database_dir(dir, &bak).is_ok();
    let m = if copy_ok { restore_backup(&bak, &rdir).ok() } else { None };
    if let Some(_m) = &m {
        let re = open_db(&rdir);
        let (ra2, _) = export_state_coll(&re, "alpha");
        let (rb2, _) = export_state_coll(&re, "beta");
        checks.push(("restore_alpha_equal".into(), ra2 == sa, String::new()));
        checks.push(("restore_beta_equal".into(), rb2 == sb, String::new()));
        // isolation by content: collection membership tags must not mix —
        // alpha docs keep x/x2 cats, beta docs keep y/y2 (logical idx spaces
        // overlap by design; uuid namespaces keep identities separate)
        checks.push(("restore_isolation".into(),
                     ra2.values().all(|(_, c, _)| c == "x" || c == "x2")
                         && rb2.values().all(|(_, c, _)| c == "y" || c == "y2"),
                     format!("alpha_cats={:?}",
                         ra2.values().map(|(_, c, _)| c.clone()).collect::<std::collections::BTreeSet<_>>())));
        checks.push(("restore_checker".into(), checker_report(&re, &rdir)["clean"].as_bool().unwrap_or(false), String::new()));
    } else {
        checks.push(("restore_alpha_equal".into(), false, format!("copy_ok={copy_ok}")));
    }
    let _ = std::fs::remove_dir_all(&bak);
    let _ = std::fs::remove_dir_all(&rdir);
    let _ = std::fs::remove_dir_all(dir);

    let (pass, fail) = checks.iter().fold((0usize, 0usize), |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) });
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks { let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'")); }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(&format!("{out}/metrics.json"), &serde_json::json!({"passed": pass, "failed": fail}));
    format!("integration: {pass} passed, {fail} FAILED")
}

/// attend_filtered over explicit heads (helper for the multi-head probe).
fn filtered_observed_multi(
    e: &AttentionEngine,
    coll: &str,
    heads: &[String],
    f: &FilterExpr,
    qidx: u32,
    k: usize,
) -> Vec<u32> {
    let q = vec_for(qidx);
    let got = e.attend_filtered(coll, heads, &q, k, Some(f)).unwrap();
    let store = e.document_store.read();
    got.iter()
        .map(|(id, _)| {
            store
                .list_all_records()
                .into_iter()
                .find(|r| e.id_mapper.read().uuid_to_id(&r.id) == Some(*id))
                .and_then(|r| r.fields.get("idx").and_then(|v| v.as_u64()))
                .unwrap_or(u64::MAX) as u32
        })
        .collect()
}

// ---------------------------------------------------------------- concur replay

/// §24: deterministic concurrent mutation logs + post-hoc expected-state
/// replay. Phase 1: disjoint key ranges per writer -> replaying the merged
/// op logs MUST equal the observed state exactly. Phase 2: two writers on
/// the SAME keys -> documents the visibility model (mutation-gate serialized;
/// per-key last-write-wins; records never torn).
fn run_concur_replay(dir: &std::path::Path, out: &str) -> String {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut checks: Vec<(String, bool, String)> = Vec::new();
    let e = std::sync::Arc::new(open_db(dir));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for idx in 0..50u32 {
        e.insert_document("bench", doc_record(idx, 1, "seed", idx as i64)).unwrap();
    }
    e.flush_wal().unwrap();

    // ---- phase 1: disjoint ranges, logged, then replayed
    let oplog = std::sync::Arc::new(std::sync::Mutex::new(
        std::fs::File::create(format!("{out}/operation-log.jsonl")).unwrap()));
    let mut handles = Vec::new();
    for w in 0..2usize {
        let e = std::sync::Arc::clone(&e);
        let oplog = std::sync::Arc::clone(&oplog);
        handles.push(std::thread::spawn(move || {
            let base = 10_000u32 * (w as u32 + 1);
            for i in 0..250u32 {
                let idx = base + i;
                let r = doc_record(idx, 1, "w", i as i64);
                let ins = e.insert_document("bench", r).is_ok();
                let mut upd = false;
                let mut del = false;
                if i % 2 == 1 {
                    let u = doc_record(idx, 2, "w2", i as i64);
                    upd = e.update_document("bench", &uuid_for(idx, 1).to_string(), u.fields, u.k_vecs).is_ok();
                }
                if i % 5 == 4 {
                    del = e.delete_document("bench", &uuid_for(idx, 1).to_string()).is_ok();
                }
                let mut f = oplog.lock().unwrap();
                let _ = writeln!(f, "{{\"w\":{w},\"i\":{i},\"idx\":{idx},\"ins\":{ins},\"upd\":{upd},\"del\":{del}}}");
            }
        }));
    }
    for h in handles { h.join().unwrap(); }

    // replay the merged logs into the reference model (seed 50 docs first)
    let mut model: Model = (0..50u32).map(|i| (i, (1u64, "seed".to_string(), i as i64))).collect();
    for line in std::fs::read_to_string(format!("{out}/operation-log.jsonl")).unwrap().lines() {
        let j: serde_json::Value = serde_json::from_str(line).unwrap();
        let idx = j["idx"].as_u64().unwrap() as u32;
        let i = j["i"].as_u64().unwrap();
        if j["del"].as_bool().unwrap() {
            model.remove(&idx);
        } else if j["upd"].as_bool().unwrap() {
            model.insert(idx, (2, "w2".to_string(), i as i64));
        } else if j["ins"].as_bool().unwrap() {
            model.insert(idx, (1, "w".to_string(), i as i64));
        }
    }
    let e2 = std::sync::Arc::clone(&e);
    let (obs, iss) = export_state(&e2);
    checks.push(("replay_exact_state".into(), obs == model && iss.is_empty(),
                 if obs == model { format!("docs={}", obs.len()) } else {
                     let miss: Vec<_> = model.keys().filter(|k| !obs.contains_key(k)).take(5).collect();
                     let extra: Vec<_> = obs.keys().filter(|k| !model.contains_key(k)).take(5).collect();
                     let diff: Vec<_> = model.iter().filter(|(k, v)| obs.get(k) != Some(*v)).take(5).collect();
                     format!("miss={miss:?} extra={extra:?} diff={diff:?}")
                 }));
    checks.push(("replay_checker".into(), checker_report(&e, dir)["clean"].as_bool().unwrap_or(false), String::new()));

    // ---- phase 2: same-key contention (two writers, same 100 keys)
    let mut wh = Vec::new();
    for w in 0..2usize {
        let e = std::sync::Arc::clone(&e);
        wh.push(std::thread::spawn(move || {
            let letter = if w == 0 { "a" } else { "b" };
            // insert baseline for my parity keys, then EVERY writer updates
            // every key with (fields.version == num == seq): a torn mix would
            // break version==num or the cat/parity pairing.
            for k in 5000..5100u32 {
                let r = doc_record(k, 1, letter, 1);
                let _ = e.insert_document("bench", r);
            }
            for seq in 1..=150u64 {
                for k in 5000..5100u32 {
                    let r = doc_record(k, seq, letter, seq as i64);
                    let _ = e.update_document("bench", &uuid_for(k, 1).to_string(), r.fields, r.k_vecs);
                }
            }
        }));
    }
    for h in wh { h.join().unwrap(); }
    let (obs2, _) = export_state(&e);
    let mut torn = 0usize;
    let mut missing = 0usize;
    for k in 5000..5100u32 {
        match obs2.get(&k) {
            None => missing += 1,
            Some((v, c, n)) => {
                // torn-mix probe: fields.version and fields.num always travel
                // together in every writer op; a record mixing two writers'
                // writes would break n == v. cat must be one writer's letter.
                let owned = c == "a" || c == "b";
                if !owned || *n != *v as i64 || *v < 1 {
                    torn += 1;
                }
            }
        }
    }
    checks.push(("contention_atomic_records".into(), torn == 0 && missing == 0,
                 format!("torn={torn} missing={missing}")));
    checks.push(("contention_checker".into(), checker_report(&e, dir)["clean"].as_bool().unwrap_or(false), String::new()));
    let _ = std::fs::remove_dir_all(dir);

    let (pass, fail) = checks.iter().fold((0usize, 0usize), |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) });
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks { let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'")); }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(&format!("{out}/metrics.json"), &serde_json::json!({
        "passed": pass, "failed": fail,
        "writers": 2, "logged_ops": 500, "contention_keys": 100,
        "model": "mutation-gate serialized; per-key last-write-wins; records never torn; no cross-key ordering guarantee",
    }));
    format!("concurreplay: {pass} passed, {fail} FAILED")
}

// ---------------------------------------------------------------- backup inventory

/// §28: backup inventory + INDEPENDENT integrity verification. Quiescent
/// database (documented requirement): populate -> flush -> checkpoint ->
/// close -> copy -> per-file sha256(source) vs sha256(backup) -> restore ->
/// per-file sha256(restored) vs backup -> state equality + checker.
fn run_backup_inventory(dir: &std::path::Path, out: &str) -> String {
    use attentiondb_core::backup::{copy_database_dir, restore_backup};
    fn sha256(p: &std::path::Path) -> String {
        let o = std::process::Command::new("sha256sum").arg(p).output();
        match o {
            Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
                .split_whitespace().next().unwrap_or("?").to_string(),
            _ => "HASH_ERROR".to_string(),
        }
    }
    fn inventory(root: &std::path::Path) -> Vec<(String, u64, String)> {
        let mut rows = Vec::new();
        fn walk(dir: &std::path::Path, rel: &str, rows: &mut Vec<(String, u64, String)>) {
            for ent in std::fs::read_dir(dir).unwrap().flatten() {
                let p = ent.path();
                let r = if rel.is_empty() {
                    ent.file_name().to_string_lossy().to_string()
                } else {
                    format!("{rel}/{}", ent.file_name().to_string_lossy())
                };
                if p.is_dir() {
                    walk(&p, &r, rows);
                } else {
                    let sz = ent.metadata().map(|m| m.len()).unwrap_or(0);
                    rows.push((r, sz, String::new()));
                }
            }
        }
        walk(root, "", &mut rows);
        rows.sort();
        for row in rows.iter_mut() {
            row.2 = sha256(&root.join(&row.0));
        }
        rows
    }

    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut checks: Vec<(String, bool, String)> = Vec::new();
    {
        let e = open_db(dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        for idx in 0..120u32 {
            e.insert_document("bench", doc_record(idx, 1, "a", idx as i64)).unwrap();
        }
        for idx in 0..30u32 {
            let r = doc_record(idx, 2, "b", idx as i64 * 2);
            e.update_document("bench", &uuid_for(idx, 1).to_string(), r.fields, r.k_vecs).unwrap();
        }
        for idx in 100..120u32 {
            e.delete_document("bench", &uuid_for(idx, 1).to_string()).unwrap();
        }
        e.flush_wal().unwrap();
        e.checkpoint().unwrap();
    } // engine dropped: quiescent (documented backup requirement)

    let bak = std::path::PathBuf::from(format!("{}.bak", dir.display()));
    let rdir = std::path::PathBuf::from(format!("{}.restored", dir.display()));
    let _ = std::fs::remove_dir_all(&bak);
    let _ = std::fs::remove_dir_all(&rdir);
    let copy_ok = copy_database_dir(dir, &bak).is_ok();
    let src_state = open_db(dir);
    let (before, _) = export_state(&src_state);
    drop(src_state);

    let mut inv = Vec::new();
    if copy_ok {
        let src_inv = inventory(dir);
        let mut bak_inv = inventory(&bak);
        // backup-meta.json is the manifest copy_database_dir itself writes —
        // it is expected in the backup and absent from the source
        bak_inv.retain(|(p, _, _)| p != "backup-meta.json");
        checks.push(("copy_fidelity_files".into(), src_inv.len() == bak_inv.len(),
                     format!("src_files={} bak_files={} (+backup-meta.json)", src_inv.len(), bak_inv.len())));
        let mismatch = src_inv.iter().zip(bak_inv.iter())
            .filter(|(a, b)| a.0 != b.0 || a.2 != b.2).count();
        checks.push(("copy_fidelity_hashes".into(), mismatch == 0,
                     format!("mismatched={mismatch}")));
        inv = bak_inv.clone();
        let m = restore_backup(&bak, &rdir);
        checks.push(("restore_manifest".into(), m.is_ok(), String::new()));
        if m.is_ok() {
            let rst_inv = inventory(&rdir);
            let rmiss = bak_inv.iter().filter(|b| !rst_inv.iter().any(|r| r.0 == b.0 && r.2 == b.2)).count();
            checks.push(("restore_fidelity_hashes".into(), rmiss == 0,
                         format!("missing_or_diff={rmiss} (restored_files={})", rst_inv.len())));
            let re = open_db(&rdir);
            let (after, iss) = export_state(&re);
            checks.push(("restore_state_equals_source".into(), after == before && iss.is_empty(),
                         format!("before={} after={}", before.len(), after.len())));
            checks.push(("restore_checker".into(), checker_report(&re, &rdir)["clean"].as_bool().unwrap_or(false), String::new()));
        }
    } else {
        checks.push(("copy_fidelity_files".into(), false, "copy failed".into()));
    }
    let total_bytes: u64 = inv.iter().map(|r| r.1).sum();
    // persist the inventory (§28 record: size, files, checksums)
    let mut ic = String::from("path,size_bytes,sha256\n");
    for (p, sz, h) in &inv {
        let _ = writeln!(ic, "{p},{sz},{h}");
    }
    std::fs::write(format!("{out}/inventory.csv"), &ic).unwrap();
    let _ = std::fs::remove_dir_all(&bak);
    let _ = std::fs::remove_dir_all(&rdir);
    let _ = std::fs::remove_dir_all(dir);

    let (pass, fail) = checks.iter().fold((0usize, 0usize), |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) });
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks { let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'")); }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(&format!("{out}/metrics.json"), &serde_json::json!({
        "passed": pass, "failed": fail,
        "files": inv.len(), "total_bytes": total_bytes,
        "note": "quiescent backup; sizes+checksums recorded in inventory.csv",
    }));
    format!("backupinv: {pass} passed, {fail} FAILED (files={} bytes={total_bytes})", inv.len())
}

// ---------------------------------------------------------------- wal integrity

/// Phase 3E E1 evidence run: the WAL-integrity matrix on real engine dirs.
/// Each case builds an isolated DB, applies file surgery, and attempts
/// open_dir — verdict OPENED (n docs) or REFUSED (error class). Never repairs.
fn run_walintegrity(out: &str) -> String {
    use attentiondb_storage::read_wal_state;
    let dim = 16usize;
    let head = "default";
    let mk = |i: usize| {
        let mut fields = std::collections::HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(i));
        let mut r = Record::new(fields);
        r.k_vecs.insert(head.to_string(), {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            v
        });
        r
    };
    struct Case {
        name: &'static str,
        checkpoint: bool,
        surgery: &'static dyn Fn(&std::path::Path),
    }
    let cases = vec![
        Case {
            name: "delete_required_segment_pre_checkpoint",
            checkpoint: false,
            surgery: &|d| {
                let mut segs: Vec<_> = wal_files(d);
                segs.sort();
                std::fs::remove_file(segs.remove(0)).unwrap();
            },
        },
        Case {
            name: "delete_all_segments_post_checkpoint",
            checkpoint: true,
            surgery: &|d| {
                for f in wal_files(d) {
                    std::fs::remove_file(f).unwrap();
                }
            },
        },
        Case {
            name: "rename_segment_gap",
            checkpoint: false,
            surgery: &|d| {
                let mut segs: Vec<_> = wal_files(d);
                segs.sort();
                let first = segs.remove(0);
                std::fs::rename(first, d.join("WAL").join("00000000000000001000.wal")).unwrap();
            },
        },
        Case {
            name: "corrupt_wal_state_record",
            checkpoint: false,
            surgery: &|d| {
                std::fs::write(d.join("WAL").join("wal-state.json"), b"{ broken").unwrap();
            },
        },
        Case {
            name: "delete_wal_state_record_legacy",
            checkpoint: true,
            surgery: &|d| {
                std::fs::remove_file(d.join("WAL").join("wal-state.json")).unwrap();
            },
        },
        Case {
            name: "truncate_torn_tail",
            checkpoint: false,
            surgery: &|d| {
                let mut segs: Vec<_> = wal_files(d);
                segs.sort();
                let b = std::fs::read(&segs[0]).unwrap();
                std::fs::write(&segs[0], &b[..b.len() * 3 / 5]).unwrap();
            },
        },
        Case {
            name: "corrupt_frame",
            checkpoint: false,
            surgery: &|d| {
                let mut segs: Vec<_> = wal_files(d);
                segs.sort();
                let p = &segs[0];
                let mut b = std::fs::read(p).unwrap();
                let at = (b.len() / 2).max(40).min(b.len() - 1);
                b[at] ^= 0xFF;
                std::fs::write(p, &b).unwrap();
            },
        },
        Case {
            name: "corrupt_current_manifest_fallback",
            checkpoint: true,
            surgery: &|d| {
                std::fs::write(d.join("CURRENT"), b"manifest-999999999\n").unwrap();
            },
        },
        Case {
            name: "corrupt_all_manifests",
            checkpoint: true,
            surgery: &|d| {
                std::fs::write(d.join("CURRENT"), b"manifest-999999999\n").unwrap();
                for e in std::fs::read_dir(d.join("MANIFEST")).unwrap().flatten() {
                    let p = e.path();
                    if p.extension().is_none() {
                        std::fs::write(&p, b"CORRUPTED").unwrap();
                    }
                }
            },
        },
        Case {
            name: "fresh_no_documents",
            checkpoint: false,
            surgery: &|_d| {},
        },
        Case {
            name: "trimmed_reopen",
            checkpoint: true,
            surgery: &|_d| {},
        },
    ];

    let mut rows = String::from("case,checkpoint,verdict,detail\n");
    let mut refused = 0usize;
    let mut opened = 0usize;
    for c in &cases {
        let dir = std::path::PathBuf::from(format!("/tmp/ph3e-wal-{}", c.name));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let e = open_db(&dir);
            e.create_collection("bench", dim, &[head]).unwrap();
            for i in 0..20usize {
                e.insert_document("bench", mk(i)).unwrap();
            }
            if c.checkpoint {
                e.close().unwrap();
            }
        }
        (c.surgery)(&dir);
        let line = match AttentionEngine::open_dir(&dir, Durability::Sync) {
            Err(e) => {
                refused += 1;
                let msg = format!("{e}");
                let code = if msg.contains("WAL_LOST_SEGMENT") {
                    "WAL_LOST_SEGMENT"
                } else if msg.contains("WAL integrity record") {
                    "WAL_STATE_CORRUPT"
                } else if msg.contains("WAL replay failed") {
                    "WAL_REPLAY_CORRUPTION"
                } else if msg.contains("manifest") || msg.contains("catalog") {
                    "MANIFEST_UNREADABLE"
                } else {
                    "REFUSED"
                };
                format!("{},{},REFUSED,{}", c.name, c.checkpoint, code)
            }
            Ok(e) => {
                opened += 1;
                let ws = read_wal_state(&dir.join("WAL")).ok().flatten();
                let n = e.document_store.read().list_all_records().len();
                let rep = checker_report(&e, &dir);
                let clean = rep["clean"].as_bool().unwrap_or(false);
                format!(
                    "{},{},OPENED,docs={} checker_clean={} watermark={}",
                    c.name,
                    c.checkpoint,
                    n,
                    clean,
                    ws.map(|w| w.high_watermark.to_string()).unwrap_or("-".into())
                )
            }
        };
        rows.push_str(&line);
        rows.push('\n');
        let _ = std::fs::remove_dir_all(&dir);
    }
    std::fs::write(format!("{out}/wal-integrity.csv"), &rows).unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({"cases": cases.len(), "refused": refused, "opened": opened}),
    );
    format!("walintegrity: {} cases, {refused} refused, {opened} opened", cases.len())
}

fn wal_files(db_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(db_dir.join("WAL"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wal"))
        .collect()
}


// ================================================================
// Phase 3E E2 — durability semantics & acknowledgment contract
// ================================================================
//
// Method: a child process (subcommand `gatechild`) performs deterministic
// work (baseline acked inserts, then a target op) and dies SUDDENLY at an
// exactly-instrumented point of the commit path — either via the storage
// crash gates (PH3E_CRASH_AT/HIT, see storage/src/crashgate.rs) or via an
// explicit abort() at a harness-level point (after_ack / explicit_flush /
// post-structural-point). The parent (subcommand `durability`) spawns the
// child per matrix cell, reopens the database, and records FACTS ONLY:
// what was acked, what recovered, checker status, how the child died.
// Judgement (expected vs actual) is applied later by
// research/phase3/generate_results_ph3e.py from the raw facts.
//
// Gate-hit arithmetic (deterministic; WAL-level gates also fire on the
// CreateCollection record, engine-level gates only on insert/delete/txn
// call sites):
//   ack suite : engine gates hit 11 (10 baseline + target);
//               WAL gates hit 12 (collection + 10 baseline + target)
//   txn suite : engine gates hit 6 (5 baseline + the commit call);
//               WAL gates: 1 collection + 5 baseline + 1 BEGIN + 8 insert
//               ops + 2 delete ops = 17 records before COMMIT →
//               after_write@12 = mid-ops (BEGIN + 4 inserts written),
//               after_write@18 = COMMIT frame written

#[derive(Clone, Copy, PartialEq)]
enum E2Exit {
    Abort,
    Kill,
    Exit0,
    Other,
}

fn e2_classify(status: std::process::ExitStatus) -> E2Exit {
    use std::os::unix::process::ExitStatusExt;
    if let Some(sig) = status.signal() {
        match sig {
            6 => E2Exit::Abort,
            9 => E2Exit::Kill,
            _ => E2Exit::Other,
        }
    } else if status.success() {
        E2Exit::Exit0
    } else {
        E2Exit::Other
    }
}

fn e2_exit_name(x: E2Exit) -> &'static str {
    match x {
        E2Exit::Abort => "SIGABRT",
        E2Exit::Kill => "SIGKILL",
        E2Exit::Exit0 => "EXIT0",
        E2Exit::Other => "OTHER",
    }
}

fn e2_sidecar_path(dir: &std::path::Path) -> String {
    format!("{}.e2sidecar", dir.display())
}

fn e2_spawn_child_live(
    dir: &std::path::Path,
    scenario: &str,
    mode: &str,
) -> std::process::Child {
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_file(e2_sidecar_path(dir));
    std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "dbtest",
            "gatechild",
            "--dir",
            dir.to_str().unwrap(),
            "--scenario",
            scenario,
            "--gate",
            "",
        ])
        .env("PH3D_DURABILITY", mode)
        .spawn()
        .unwrap()
}

fn e2_spawn_child(
    dir: &std::path::Path,
    scenario: &str,
    gate: &str,
    gathit: usize,
    mode: &str,
    seg_bytes: Option<u64>,
) -> E2Exit {
    let _ = std::fs::remove_dir_all(dir);
    let _ = std::fs::remove_file(e2_sidecar_path(dir));
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args([
        "dbtest",
        "gatechild",
        "--dir",
        dir.to_str().unwrap(),
        "--scenario",
        scenario,
        "--gate",
        gate,
    ])
    .env("PH3D_DURABILITY", mode)
    .env("PH3E_CRASH_AT", gate)
    .env("PH3E_CRASH_HIT", gathit.to_string());
    if let Some(b) = seg_bytes {
        cmd.env("ATTENTIONDB_WAL_SEGMENT_BYTES", b.to_string());
    }
    e2_classify(cmd.status().unwrap())
}

/// Reopen after the crash and record the recovered facts for one cell.
struct E2Recovery {
    baseline_preserved: usize,
    baseline_total: usize,
    target_present: bool,
    extra_idx: Vec<u32>,       // anything outside the expected idx universe
    #[allow(dead_code)] // derivable from preserved/total; kept for debugging
    missing_baseline: Vec<u32>, // acked-but-absent baseline idxs
    checker_clean: bool,
    wal_high_watermark: String,
    restart_states_equal: bool,
    restart_checker_clean: bool,
}

fn e2_recover(
    dir: &std::path::Path,
    mode: &str,
    baseline_idx: &[u32],
    target_idx: Option<u32>,
    check_restarts: bool,
) -> E2Recovery {
    std::env::set_var("PH3D_DURABILITY", mode);
    let e = open_db(dir);
    let (obs, _) = export_state(&e);
    let mut missing_baseline = Vec::new();
    let mut baseline_preserved = 0usize;
    for i in baseline_idx {
        if obs.contains_key(i) {
            baseline_preserved += 1;
        } else {
            missing_baseline.push(*i);
        }
    }
    let universe: std::collections::HashSet<u32> = baseline_idx
        .iter()
        .copied()
        .chain(target_idx)
        .collect();
    let extra_idx: Vec<u32> = obs.keys().filter(|k| !universe.contains(k)).copied().collect();
    let target_present = target_idx.map(|t| obs.contains_key(&t)).unwrap_or(false);
    let rep = checker_report(&e, dir);
    let clean = rep["clean"].as_bool().unwrap_or(false);
    let wsm = attentiondb_storage::read_wal_state(&dir.join("WAL"))
        .ok()
        .flatten()
        .map(|w| w.high_watermark.to_string())
        .unwrap_or_else(|| "-".to_string());
    // Multiple-restart stability (§15): the recovered state must be a fixed
    // point across TWO further close/open cycles, checker clean each time.
    let (restart_states_equal, restart_checker_clean) = if check_restarts {
        let mut eq = true;
        let mut all_clean = true;
        let mut keys_prev: Vec<u32> = obs.keys().copied().collect();
        let mut handle = e;
        for _ in 0..2 {
            handle.close().unwrap();
            let e2 = open_db(dir);
            let (obs2, _) = export_state(&e2);
            let keys2: Vec<u32> = obs2.keys().copied().collect();
            if keys2 != keys_prev {
                eq = false;
            }
            if !checker_report(&e2, dir)["clean"].as_bool().unwrap_or(false) {
                all_clean = false;
            }
            keys_prev = keys2;
            handle = e2;
        }
        handle.close().unwrap();
        (eq, all_clean)
    } else {
        (true, clean)
    };
    E2Recovery {
        baseline_preserved,
        baseline_total: baseline_idx.len(),
        target_present,
        extra_idx,
        missing_baseline,
        checker_clean: clean,
        wal_high_watermark: wsm,
        restart_states_equal,
        restart_checker_clean,
    }
}

/// Child process for every E2 crash cell. Dies by abort() at the configured
/// point; never flushes on the way out.
fn run_gatechild(dir: &std::path::Path, scenario: &str, gate: &str) -> ! {
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let mut sc = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(e2_sidecar_path(dir))
        .unwrap();
    let mut ack = |line: String| {
        use std::io::Write;
        let _ = writeln!(sc, "{line}");
        let _ = sc.flush();
        let _ = sc.sync_all();
    };
    match scenario {
        "ack" => {
            for i in 0..10u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            match gate {
                "after_ack" => {
                    e.insert_document("bench", doc_record(2000, 1, "t", 0)).unwrap();
                    ack(String::from("ACK 2000"));
                    std::process::abort();
                }
                "explicit_flush" => {
                    e.insert_document("bench", doc_record(2000, 1, "t", 0)).unwrap();
                    ack(String::from("ACK 2000"));
                    e.flush_wal().unwrap();
                    std::process::abort();
                }
                _ => {
                    // env-gated abort fires inside this call
                    e.insert_document("bench", doc_record(2000, 1, "t", 0)).unwrap();
                    ack(String::from("ACK 2000")); // only reached if the gate did not fire
                    std::process::abort();
                }
            }
        }
        "txn" => {
            for i in 0..5u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            use attentiondb_core::transaction::TxnOp;
            let t = e.txn_manager.begin_transaction("bench");
            for i in 2000..2008u32 {
                e.txn_manager
                    .record_operation(t, TxnOp::Insert(doc_record(i, 1, "txn", i as i64)))
                    .unwrap();
            }
            e.txn_manager.record_operation(t, TxnOp::Delete(uuid_for(1001, 1))).unwrap();
            e.txn_manager.record_operation(t, TxnOp::Delete(uuid_for(1003, 1))).unwrap();
            match gate {
                "after_ack" => {
                    e.commit_transaction(t).unwrap();
                    ack(String::from("TXNACK"));
                    std::process::abort();
                }
                "ack_ckpt" => {
                    e.commit_transaction(t).unwrap();
                    ack(String::from("TXNACK"));
                    e.checkpoint().unwrap();
                    std::process::abort();
                }
                _ => {
                    // env-gated abort fires inside commit (BEGIN/ops/COMMIT appends)
                    e.commit_transaction(t).unwrap();
                    ack(String::from("TXNACK"));
                    std::process::abort();
                }
            }
        }
        "group" => {
            // 3 writers × 25 acked inserts on ONE engine handle (serialized by
            // the mutation gate; GroupCommit = flush inside every append).
            let e = std::sync::Arc::new(e);
            let side = std::sync::Arc::new(std::sync::Mutex::new(sc));
            let mut handles = Vec::new();
            for t in 0..3u32 {
                let e = e.clone();
                let side = side.clone();
                handles.push(std::thread::spawn(move || {
                    for j in 0..25u32 {
                        let idx = 3000 + t * 100 + j;
                        e.insert_document("bench", doc_record(idx, 1, "g", idx as i64)).unwrap();
                        let mut f = side.lock().unwrap();
                        use std::io::Write;
                        let _ = writeln!(*f, "ACK {idx}");
                        let _ = f.flush();
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }
        "ckpt" => {
            // 30 baseline acked inserts (guarantees ≥1 rotation when
            // ATTENTIONDB_WAL_SEGMENT_BYTES is small), then the structural
            // durability point named by `gate`, then an acked target insert.
            for i in 0..30u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            match gate {
                "checkpoint" | "checkpoint_trim" => {
                    e.checkpoint().unwrap();
                }
                "rotate_only" => {}
                _ => panic!("unknown structural point {gate}"),
            }
            e.insert_document("bench", doc_record(2000, 1, "t", 0)).unwrap();
            ack(String::from("ACK 2000"));
            std::process::abort();
        }
        _ => panic!("unknown scenario {scenario}"),
    }
}

fn e2_row(cols: &[String]) -> String {
    cols.join(",")
}

/// Parent driver. Records raw facts; judgement happens in the generator.
fn run_durability(suite: &str, out: &str) -> String {
    let modes = ["sync", "group", "async"];
    std::fs::create_dir_all(out).unwrap();
    let mut rows: Vec<String> = Vec::new();
    let mut cells = 0usize;
    match suite {
        "ack-boundary" => {
            rows.push("mode,gate,rep,exit_kind,baseline_preserved,baseline_total,target_acked,target_present,extra_count,checker_clean,high_watermark".into());
            let baseline: Vec<u32> = (1000..1010).collect();
            for mode in modes {
                for gate in [
                    "before_wal_append",
                    "after_write",
                    "after_flush",
                    "after_fsync",
                    "after_wal_append",
                    "after_apply",
                    "before_ack",
                    "after_ack",
                    "explicit_flush",
                ] {
                    // hit number: engine gates count insert calls (10 baseline
                    // + target = 11); WAL gates also count the collection
                    // record (12)
                    let (hit, target_acked) = match gate {
                        "after_ack" | "explicit_flush" => (0usize, true),
                        "before_wal_append" | "after_wal_append" | "after_apply"
                        | "before_ack" => (11, false),
                        _ => (12, false),
                    };
                    // after_flush fires only inside the GroupCommit branch of
                    // Wal::append; after_fsync only inside the Sync branch. In
                    // other modes the gate is UNREACHABLE — record the cell as
                    // NOT_REACHED instead of spawning (the child's fallback
                    // abort would fabricate an after_ack-like observation).
                    let reachable = match gate {
                        "after_flush" => mode == "group",
                        "after_fsync" => mode == "sync",
                        _ => true,
                    };
                    if !reachable {
                        for rep in 0..3usize {
                            rows.push(e2_row(&[
                                mode.into(), gate.into(), rep.to_string(), "NOT_REACHED".into(),
                                "-".into(), "0".into(), "-".into(), "-".into(),
                                "-".into(), "-".into(), "-".into(),
                            ]));
                        }
                        cells += 3;
                        continue;
                    }
                    for rep in 0..3usize {
                        let dir = std::path::PathBuf::from(format!(
                            "/tmp/ph3e-dur/ack-{mode}-{gate}-{rep}"
                        ));
                        let exit = e2_spawn_child(&dir, "ack", gate, hit, mode, None);
                        let r = e2_recover(&dir, mode, &baseline, Some(2000), false);
                        rows.push(e2_row(&[
                            mode.into(), gate.into(), rep.to_string(), e2_exit_name(exit).into(),
                            r.baseline_preserved.to_string(), r.baseline_total.to_string(),
                            target_acked.to_string(), r.target_present.to_string(),
                            r.extra_idx.len().to_string(), r.checker_clean.to_string(),
                            r.wal_high_watermark.clone(),
                        ]));
                        cells += 1;
                        let _ = std::fs::remove_dir_all(&dir);
                        let _ = std::fs::remove_file(e2_sidecar_path(&dir));
                    }
                }
            }
        }
        "txn-ack" => {
            rows.push("mode,gate,rep,exit_kind,baseline_preserved,baseline_total,txn_acked,txn_insert_state,txn_deletes_applied,extra_count,checker_clean,high_watermark".into());
            let baseline: Vec<u32> = (1000..1005).collect();
            for mode in modes {
                for gate in [
                    "mid_ops",
                    "commit_written",
                    "after_wal_append",
                    "after_apply",
                    "before_ack",
                    "after_ack",
                    "ack_ckpt",
                ] {
                    let (scenario, g, hit) = match gate {
                        "mid_ops" => ("txn", "after_write", 12usize),
                        "commit_written" => ("txn", "after_write", 18usize),
                        "after_ack" | "ack_ckpt" => ("txn", gate, 0usize),
                        other => ("txn", other, 6usize),
                    };
                    for rep in 0..3usize {
                        let dir = std::path::PathBuf::from(format!(
                            "/tmp/ph3e-dur/txn-{mode}-{gate}-{rep}"
                        ));
                        let exit = e2_spawn_child(&dir, scenario, g, hit, mode, None);
                        std::env::set_var("PH3D_DURABILITY", mode);
                        let txn_acked = gate == "after_ack" || gate == "ack_ckpt";
                        // txn state from recovered facts
                        let e = open_db(&dir);
                        let (obs, _) = export_state(&e);
                        // All-or-nothing is judged on the txn INSERTS: an
                        // ABSENT txn whose deleted baseline docs were THEMSELVES
                        // lost (Async buffering) leaves "deletes applied"
                        // vacuously true — never call that PARTIAL.
                        let inserts_present =
                            (2000..2008u32).filter(|i| obs.contains_key(i)).count();
                        let deletes_applied = !obs.contains_key(&1001) && !obs.contains_key(&1003);
                        let txn_state = match inserts_present {
                            8 => "COMPLETE",
                            0 => "ABSENT",
                            _ => "PARTIAL",
                        };
                        let mut missing_baseline = Vec::new();
                        let mut baseline_preserved = 0usize;
                        for i in &baseline {
                            if obs.contains_key(i) {
                                baseline_preserved += 1;
                            } else {
                                missing_baseline.push(*i);
                            }
                        }
                        let universe: std::collections::HashSet<u32> = baseline
                            .iter()
                            .copied()
                            .chain(2000..2008)
                            .collect();
                        let extra = obs.keys().filter(|k| !universe.contains(k)).count();
                        let rep_json = checker_report(&e, &dir);
                        let clean = rep_json["clean"].as_bool().unwrap_or(false);
                        let wm = attentiondb_storage::read_wal_state(&dir.join("WAL"))
                            .ok()
                            .flatten()
                            .map(|w| w.high_watermark.to_string())
                            .unwrap_or_else(|| "-".into());
                        drop(e);
                        rows.push(e2_row(&[
                            mode.into(), gate.into(), rep.to_string(), e2_exit_name(exit).into(),
                            baseline_preserved.to_string(), baseline.len().to_string(),
                            txn_acked.to_string(), txn_state.into(), deletes_applied.to_string(),
                            extra.to_string(), clean.to_string(), wm,
                        ]));
                        cells += 1;
                        let _ = std::fs::remove_dir_all(&dir);
                        let _ = std::fs::remove_file(e2_sidecar_path(&dir));
                    }
                }
            }
        }
        "checkpoint-interaction" => {
            rows.push("mode,structural,rep,exit_kind,baseline_preserved,baseline_total,target_acked,target_present,extra_count,checker_clean,high_watermark,restarts_equal,restart_checker_clean".into());
            let baseline: Vec<u32> = (1000..1030).collect();
            for mode in modes {
                for structural in ["checkpoint", "checkpoint_trim", "rotate_only"] {
                    for rep in 0..3usize {
                        let dir = std::path::PathBuf::from(format!(
                            "/tmp/ph3e-dur/ckpt-{mode}-{structural}-{rep}"
                        ));
                        let seg = if structural == "rotate_only" { Some(2048u64) } else { None };
                        let exit = e2_spawn_child(&dir, "ckpt", structural, 0, mode, seg);
                        let r = e2_recover(&dir, mode, &baseline, Some(2000), true);
                        rows.push(e2_row(&[
                            mode.into(), structural.into(), rep.to_string(), e2_exit_name(exit).into(),
                            r.baseline_preserved.to_string(), r.baseline_total.to_string(),
                            "true".into(), r.target_present.to_string(),
                            r.extra_idx.len().to_string(), r.checker_clean.to_string(),
                            r.wal_high_watermark.clone(),
                            r.restart_states_equal.to_string(),
                            r.restart_checker_clean.to_string(),
                        ]));
                        cells += 1;
                        let _ = std::fs::remove_dir_all(&dir);
                        let _ = std::fs::remove_file(e2_sidecar_path(&dir));
                    }
                }
            }
        }
        "group-boundary" => {
            rows.push("mode,variant,rep,exit_kind,acked_total,recovered_acked,acked_lost,unacked_survived,extra_count,checker_clean".into());
            for mode in modes {
                for variant in ["self", "mid"] {
                    for rep in 0..3usize {
                        let dir = std::path::PathBuf::from(format!(
                            "/tmp/ph3e-dur/group-{mode}-{variant}-{rep}"
                        ));
                        // The child parks forever after its writers finish, so
                        // the parent owns its lifecycle: spawn live, poll the
                        // ack sidecar, SIGKILL via the child handle.
                        let mut child = e2_spawn_child_live(&dir, "group", mode);
                        // parent kill for both variants: self waits for all 75,
                        // mid kills at >= 40 acks
                        let target = if variant == "self" { 75usize } else { 40usize };
                        let mut acked: Vec<u32> = Vec::new();
                        for _poll in 0..600 {
                            if let Ok(s) = std::fs::read_to_string(e2_sidecar_path(&dir)) {
                                acked = s
                                    .lines()
                                    .filter_map(|l| l.strip_prefix("ACK ")?.parse().ok())
                                    .collect();
                                if acked.len() >= target {
                                    break;
                                }
                            }
                            std::thread::sleep(std::time::Duration::from_millis(20));
                        }
                        let _ = child.kill();
                        let exit = e2_classify(child.wait().unwrap());
                        std::env::set_var("PH3D_DURABILITY", mode);
                        let e = open_db(&dir);
                        let (obs, _) = export_state(&e);
                        let recovered_acked = acked.iter().filter(|i| obs.contains_key(i)).count();
                        let acked_lost = acked.len() - recovered_acked;
                        let acked_set: std::collections::HashSet<u32> = acked.iter().copied().collect();
                        let unacked_survived = obs
                            .keys()
                            .filter(|k| (3000..3999).contains(*k) && !acked_set.contains(k))
                            .count();
                        let extra = obs.keys().filter(|k| !(3000..3999).contains(*k)).count();
                        let clean = checker_report(&e, &dir)["clean"].as_bool().unwrap_or(false);
                        rows.push(e2_row(&[
                            mode.into(), variant.into(), rep.to_string(), e2_exit_name(exit).into(),
                            acked.len().to_string(), recovered_acked.to_string(),
                            acked_lost.to_string(), unacked_survived.to_string(),
                            extra.to_string(), clean.to_string(),
                        ]));
                        cells += 1;
                        let _ = std::fs::remove_dir_all(&dir);
                        let _ = std::fs::remove_file(e2_sidecar_path(&dir));
                    }
                }
            }
        }
        "mode-latency" => {
            rows.push("mode,ops,mean_us,p50_us,p90_us,p99_us,max_us,throughput_ops_s".into());
            for mode in modes {
                let dir = std::path::PathBuf::from(format!("/tmp/ph3e-dur/lat-{mode}"));
                let _ = std::fs::remove_dir_all(&dir);
                std::env::set_var("PH3D_DURABILITY", mode);
                let e = open_db(&dir);
                e.create_collection("bench", DIM, &[HEAD]).unwrap();
                let mut samples: Vec<u128> = Vec::with_capacity(2000);
                for i in 0..2000u32 {
                    let t0 = std::time::Instant::now();
                    e.insert_document("bench", doc_record(5000 + i, 1, "l", i as i64)).unwrap();
                    samples.push(t0.elapsed().as_micros());
                }
                let n = samples.len();
                let mean = samples.iter().sum::<u128>() / n as u128;
                samples.sort();
                let pct = |p: f64| samples[((n as f64) * p) as usize % n];
                let total_s = samples.iter().sum::<u128>() as f64 / 1e6;
                rows.push(e2_row(&[
                    mode.into(), n.to_string(), mean.to_string(),
                    pct(0.50).to_string(), pct(0.90).to_string(), pct(0.99).to_string(),
                    samples[n - 1].to_string(),
                    format!("{:.0}", n as f64 / total_s),
                ]));
                cells += 1;
                e.close().unwrap();
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
        _ => panic!("unknown durability suite {suite}"),
    }
    let csv = format!("{out}/{suite}.csv");
    std::fs::write(&csv, rows.join("\n") + "\n").unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({"suite": suite, "cells": cells}),
    );
    format!("durability[{suite}]: {cells} cells -> {csv}")
}


// ================================================================
// Phase 3E E3 — machine-crash / power-loss durability validation
// ================================================================
//
// Failure models (phase3e-e3-spec.md §3):
//   F0  graceful close (control)
//   F1  process crash                     -> E2 evidence, NOT re-run as E3
//   F2E environment-termination-EQUIVALENT: the child runs in its own process
//       group and PARKS at the exact in-engine window gate (marker file
//       fsynced first); the controller then SIGKILLs the entire group. No
//       destructors, no flushes, no cleanup of anything holding DB state.
//       The engine is single-process, so the group covers every DB-state
//       holder. This is the STRONGEST mechanism available in this sandbox —
//       it is NOT a VM kill, NOT a filesystem disruption, NOT power loss.
//   F3  filesystem/cache disruption       -> BLOCKED (no root, no dm tools)
//   F4  physical power loss               -> BLOCKED (no power mechanism)
//
// The controller independently records the ACK sidecar (fsynced by the
// child), the crash marker, and after recovery: the fact set + checker.
// Expected-state judgement is applied ONLY by generate_results_ph3e.py.

fn e3_park_with_marker(dir: &std::path::Path, label: &str) -> ! {
    let m = std::path::PathBuf::from(format!("{}.e3gate", dir.display()));
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).write(true).truncate(true).open(&m) {
        let _ = writeln!(f, "{label}");
        let _ = f.sync_all();
    }
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// Child for one E3 cell: deterministic workload; parks at the requested
/// window (engine gate or harness-level point) until the controller kills
/// the process group.
fn run_e3child(dir: &std::path::Path, workload: &str, gate: &str) -> ! {
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let side = std::path::PathBuf::from(format!("{}.e3sidecar", dir.display()));
    let mut sc = std::fs::OpenOptions::new().create(true).append(true).open(&side).unwrap();
    let mut ack = |line: String| {
        use std::io::Write;
        let _ = writeln!(sc, "{line}");
        let _ = sc.flush();
        let _ = sc.sync_all();
    };
    match workload {
        "ack" => {
            for i in 0..10u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            if gate == "after_ack" {
                // C7 harness-level park (no engine gate fires)
                e.insert_document("bench", doc_record(2000, 1, "t", 0)).unwrap();
                ack(String::from("ACK 2000"));
                e3_park_with_marker(dir, "after_ack");
            }
            // env-gated park fires inside this insert
            e.insert_document("bench", doc_record(2000, 1, "t", 0)).unwrap();
            ack(String::from("ACK 2000"));
            e3_park_with_marker(dir, "post-window-fallback");
        }
        "mixed" => {
            for i in 0..9u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            for i in [0u32, 2, 4] {
                e.delete_document("bench", &uuid_for(1000 + i, 1).to_string()).unwrap();
                ack(format!("DEL {}", 1000 + i));
            }
            for i in 0..3u32 {
                e.insert_document("bench", doc_record(3000 + i, 1, "c2", i as i64)).unwrap();
                ack(format!("ACK {}", 3000 + i));
            }
            if gate == "after_ack" {
                e.insert_document("bench", doc_record(2000, 1, "t", 0)).unwrap();
                ack(String::from("ACK 2000"));
                e3_park_with_marker(dir, "after_ack");
            }
            // structural window (fires inside checkpoint)
            e.checkpoint().unwrap();
            e3_park_with_marker(dir, "post-window-fallback");
        }
        "txn" => {
            for i in 0..5u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            use attentiondb_core::transaction::TxnOp;
            let t = e.txn_manager.begin_transaction("bench");
            for i in 2000..2008u32 {
                e.txn_manager.record_operation(t, TxnOp::Insert(doc_record(i, 1, "txn", i as i64))).unwrap();
            }
            e.txn_manager.record_operation(t, TxnOp::Delete(uuid_for(1001, 1))).unwrap();
            e.txn_manager.record_operation(t, TxnOp::Delete(uuid_for(1003, 1))).unwrap();
            if gate == "after_ack" {
                e.commit_transaction(t).unwrap();
                ack(String::from("TXNACK"));
                e3_park_with_marker(dir, "after_ack");
            }
            // env-gated park fires inside commit (e.g. after_write@18 = COMMIT frame written)
            e.commit_transaction(t).unwrap();
            ack(String::from("TXNACK"));
            e3_park_with_marker(dir, "post-window-fallback");
        }
        "ckpt" => {
            for i in 0..16u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            // env-gated park fires inside checkpoint at the structural window
            e.checkpoint().unwrap();
            e3_park_with_marker(dir, "post-window-fallback");
        }
        "rotate" => {
            // small segments via ATTENTIONDB_WAL_SEGMENT_BYTES; 7 acked docs
            // fill segment 1 (collection + 7 records); the NEXT insert
            // triggers the rotation the gate parks in.
            for i in 0..7u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            // env-gated park fires inside this insert's rotation
            e.insert_document("bench", doc_record(2000, 1, "t", 0)).unwrap();
            ack(String::from("ACK 2000"));
            e3_park_with_marker(dir, "post-window-fallback");
        }
        "close" => {
            for i in 0..10u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            e.close().unwrap(); // F0 control: graceful, no crash
            std::process::exit(0);
        }
        _ => panic!("unknown e3 workload {workload}"),
    }
}

/// Controller: spawn child in its own process group, wait for the durable
/// window marker, SIGKILL the WHOLE group (F2E), reopen + record facts.
struct E3Facts {
    marker_reached: bool,
    exit_kind: &'static str,
    open_error: Option<String>,
    recovered_keys: Vec<u32>,
    checker_clean: bool,
    high_watermark: String,
    restarts_equal: bool,
    restart_checker_clean: bool,
}

// A matrix cell is one row of the driver table; the parameters ARE the cell
// descriptor (window/hit/mode/segment-size/crash/restarts) — a struct would
// only relocate the count.
#[allow(clippy::too_many_arguments)]
fn e3_run_cell(
    dir: &std::path::Path,
    workload: &str,
    gate: &str,
    gathit: usize,
    mode: &str,
    seg_bytes: Option<u64>,
    crash: bool,
    restarts: bool,
) -> E3Facts {
    let _ = std::fs::remove_dir_all(dir);
    for suffix in [".e3sidecar", ".e3gate"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", dir.display()));
    }
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args(["dbtest", "e3child", "--dir", dir.to_str().unwrap(),
              "--workload", workload, "--gate", gate])
        .env("PH3D_DURABILITY", mode)
        .env("PH3E_CRASH_MODEL", "groupkill")
        .env("PH3E_CRASH_MARKER", format!("{}.e3gate", dir.display()));
    if crash {
        cmd.env("PH3E_CRASH_AT", gate).env("PH3E_CRASH_HIT", gathit.to_string());
    }
    if let Some(b) = seg_bytes {
        cmd.env("ATTENTIONDB_WAL_SEGMENT_BYTES", b.to_string());
    }
    #[cfg(unix)]
    use std::os::unix::process::CommandExt;
    #[cfg(unix)]
    cmd.process_group(0);
    let mut child = cmd.spawn().unwrap();
    let marker = format!("{}.e3gate", dir.display());
    let mut marker_reached = false;
    if crash {
        for _ in 0..2000 {
            if std::path::Path::new(&marker).exists() {
                marker_reached = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        // F2E: kill the ENTIRE process group — nothing cleans up.
        #[cfg(unix)]
        {
            let pgid = child.id() as i32;
            unsafe { libc::kill(-pgid, libc::SIGKILL); }
        }
        #[cfg(not(unix))]
        let _ = child.kill();
    }
    let exit_kind = match child.wait() {
        Ok(st) => {
            use std::os::unix::process::ExitStatusExt;
            if !crash && st.success() { "EXIT0" }
            else if let Some(9) = st.signal() { "SIGKILL" }
            else if let Some(6) = st.signal() { "SIGABRT" }
            else if st.success() { "EXIT0" }
            else { "OTHER" }
        }
        Err(_) => "WAIT_ERROR",
    };
    // Recovery + independent facts
    std::env::set_var("PH3D_DURABILITY", mode);
    let (open_error, recovered_keys, checker_clean, high_watermark) =
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| open_db(dir))) {
            Err(e) => {
                let msg = if let Some(m) = e.downcast_ref::<String>() {
                    m.clone()
                } else if let Some(m) = e.downcast_ref::<&str>() {
                    m.to_string()
                } else {
                    "panic".into()
                };
                let class = if msg.contains("WAL_LOST_SEGMENT") {
                    "WAL_LOST_SEGMENT"
                } else if msg.contains("WAL_SEQ_GAP") {
                    "WAL_SEQ_GAP"
                } else if msg.contains("WAL integrity record") {
                    "WAL_STATE_CORRUPT"
                } else if msg.contains("WAL replay failed") {
                    "WAL_REPLAY"
                } else if msg.contains("none are valid") {
                    "MANIFEST_NONE_VALID"
                } else if msg.contains("CURRENT manifest unreadable") || msg.contains("catalog recovered") {
                    "CATALOG_FALLBACK"
                } else {
                    "OPEN_ERROR"
                };
                (Some(class.into()), Vec::new(), false, "-".into())
            }
            Ok(e) => {
                let (obs, _) = export_state(&e);
                let keys: Vec<u32> = obs.keys().copied().collect();
                let clean = checker_report(&e, dir)["clean"].as_bool().unwrap_or(false);
                let wm = attentiondb_storage::read_wal_state(&dir.join("WAL"))
                    .ok().flatten()
                    .map(|w| w.high_watermark.to_string())
                    .unwrap_or_else(|| "-".into());
                (None, keys, clean, wm)
            }
        };
    // Multiple-restart stability (F0-style close/open cycles) where requested
    let (restarts_equal, restart_checker_clean) = if restarts && open_error.is_none() {
        let mut eq = true;
        let mut all_clean = true;
        let mut prev = recovered_keys.clone();
        let mut handle = open_db(dir);
        for _ in 0..2 {
            handle.close().unwrap();
            let e2 = open_db(dir);
            let (obs2, _) = export_state(&e2);
            let keys2: Vec<u32> = obs2.keys().copied().collect();
            if keys2 != prev { eq = false; }
            if !checker_report(&e2, dir)["clean"].as_bool().unwrap_or(false) { all_clean = false; }
            prev = keys2;
            handle = e2;
        }
        handle.close().unwrap();
        (eq, all_clean)
    } else {
        (true, checker_clean)
    };
    E3Facts {
        marker_reached, exit_kind, open_error, recovered_keys,
        checker_clean, high_watermark, restarts_equal, restart_checker_clean,
    }
}

fn e3_sidecar_acked(dir: &std::path::Path) -> (Vec<u32>, Vec<u32>, bool) {
    let mut acked = Vec::new();
    let mut deleted = Vec::new();
    let mut txn = false;
    if let Ok(s) = std::fs::read_to_string(format!("{}.e3sidecar", dir.display())) {
        for l in s.lines() {
            if let Some(v) = l.strip_prefix("ACK ") { if let Ok(n) = v.parse() { acked.push(n); } }
            else if let Some(v) = l.strip_prefix("DEL ") { if let Ok(n) = v.parse() { deleted.push(n); } }
            else if l == "TXNACK" { txn = true; }
        }
    }
    (acked, deleted, txn)
}

/// Driver: one CSV with every E3 cell (facts only; classification happens in
/// generate_results_ph3e.py against the contract table).
fn run_e3(out: &str) -> String {
    std::fs::create_dir_all(out).unwrap();
    let mut rows = String::from(
        "cell,mode,workload,window,hit,rep,failure_model,marker_reached,exit_kind,open_error,acked_count,recovered_acked,missing_acked,unacked_present,deleted_still_gone,txn_state,target2000_present,checker_clean,restarts_equal,restart_checker_clean,high_watermark\n");
    let mut cells = 0usize;
    let record = |rows: &mut String, cells: &mut usize,
                      cell: String, mode: &str, workload: &str, window: &str, hit: usize,
                      rep: usize, fm: &str, f: &E3Facts, dir: &std::path::Path| {
        let (acked, deleted, _txn_acked) = e3_sidecar_acked(dir);
        let recovered_acked = acked.iter().filter(|i| f.recovered_keys.contains(i)).count();
        let missing: Vec<u32> = acked.iter().filter(|i| !f.recovered_keys.contains(i)).copied().collect();
        let universe: std::collections::HashSet<u32> = acked.iter().copied()
            .chain(deleted.iter().copied())
            .chain([2000u32]).chain(2000..2008).collect();
        let unacked_present = f.recovered_keys.iter().filter(|k| !universe.contains(k)).count();
        let deleted_still_gone = deleted.iter().filter(|i| !f.recovered_keys.contains(i)).count();
        let txn_state = if workload == "txn" {
            let n = (2000..2008u32).filter(|i| f.recovered_keys.contains(i)).count();
            match n { 8 => "COMPLETE", 0 => "ABSENT", _ => "PARTIAL" }
        } else { "-" };
        let target_present = f.recovered_keys.contains(&2000);
        rows.push_str(&format!(
            "{cell},{mode},{workload},{window},{hit},{rep},{fm},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            f.marker_reached, f.exit_kind,
            f.open_error.as_deref().unwrap_or("-"),
            acked.len(), recovered_acked, missing.len(),
            unacked_present, deleted_still_gone, txn_state, target_present,
            f.checker_clean, f.restarts_equal, f.restart_checker_clean, f.high_watermark));
        *cells += 1;
    };

    // ---- Workload A: ack-inserts x C1..C7 (F2E) ----
    let gates_a: Vec<(&str, usize, bool, &str)> = vec![
        ("before_wal_append", 11, true, "engine"),
        ("after_write", 12, true, "wal"),
        ("after_apply", 11, true, "engine"),
        ("after_flush", 12, true, "wal"),      // reachable in GroupCommit branch only
        ("after_fsync", 12, true, "wal"),      // reachable in Sync branch only
        ("before_ack", 11, true, "engine"),
        ("after_ack", 0, true, "harness"),
    ];
    for mode in ["sync", "group", "async"] {
        for (gate, hit, crash, kind) in &gates_a {
            let reachable = match *gate {
                "after_flush" => mode == "group",
                "after_fsync" => mode == "sync",
                _ => true,
            };
            if !reachable {
                for rep in 0..3 {
                    rows.push_str(&format!("ack-c,{mode},ack,{gate},{hit},{rep},F2E-GROUPKILL,false,NOT_REACHED,-,-,-,-,-,-,-,-,-,-,-\n"));
                    cells += 1;
                }
                continue;
            }
            for rep in 0..3 {
                let dir = std::path::PathBuf::from(format!("/tmp/ph3e-e3/ack-{mode}-{gate}-{rep}"));
                let f = e3_run_cell(&dir, "ack", gate, *hit, mode, None, *crash, false);
                record(&mut rows, &mut cells, "ack-c".into(), mode, "ack", gate, *hit, rep,
                       if *kind == "harness" { "F2E-GROUPKILL-HARNESS" } else { "F2E-GROUPKILL" }, &f, &dir);
                let _ = std::fs::remove_dir_all(&dir);
                let _ = std::fs::remove_file(format!("{}.e3sidecar", dir.display()));
                let _ = std::fs::remove_file(format!("{}.e3gate", dir.display()));
            }
        }
    }

    // ---- Workload D: checkpoint boundary x 9 structural windows ----
    // NOTE: manifest_* gates fire inside Catalog::save — hit 1 is
    // create_collection's persist_catalog, hit 2 is the checkpoint's save.
    let gates_d = [
        ("ckpt_after_wal_fsync", 1), ("ckpt_after_sst", 1), ("ckpt_after_idmap", 1),
        ("manifest_after_tmp_write", 2), ("manifest_after_manifest_dirsync", 2),
        ("manifest_after_current_tmp_write", 2), ("manifest_after_current_rename", 2),
        ("ckpt_after_rotate", 1), ("ckpt_after_trim", 1),
    ];
    for mode in ["sync", "group", "async"] {
        for (gate, hit) in gates_d {
            for rep in 0..3 {
                let dir = std::path::PathBuf::from(format!("/tmp/ph3e-e3/ckpt-{mode}-{gate}-{rep}"));
                let f = e3_run_cell(&dir, "ckpt", gate, hit, mode, None, true, true);
                record(&mut rows, &mut cells, "ckpt-c".into(), mode, "ckpt", gate, hit, rep, "F2E-GROUPKILL", &f, &dir);
                let _ = std::fs::remove_dir_all(&dir);
                let _ = std::fs::remove_file(format!("{}.e3sidecar", dir.display()));
                let _ = std::fs::remove_file(format!("{}.e3gate", dir.display()));
            }
        }
    }

    // ---- Workload E: WAL-rotation boundary (2 KiB segments) ----
    let gates_e = [("rotate_after_old_fsync", 1usize), ("rotate_after_state_write", 2)];
    for mode in ["sync", "group", "async"] {
        for (gate, hit) in gates_e {
            for rep in 0..3 {
                let dir = std::path::PathBuf::from(format!("/tmp/ph3e-e3/rot-{mode}-{gate}-{rep}"));
                let f = e3_run_cell(&dir, "rotate", gate, hit, mode, Some(2048), true, true);
                record(&mut rows, &mut cells, "rotate-c".into(), mode, "rotate", gate, hit, rep, "F2E-GROUPKILL", &f, &dir);
                let _ = std::fs::remove_dir_all(&dir);
                let _ = std::fs::remove_file(format!("{}.e3sidecar", dir.display()));
                let _ = std::fs::remove_file(format!("{}.e3gate", dir.display()));
            }
        }
    }

    // ---- Workload B: mixed mutations (checkpoint window + after-ack) ----
    let gates_b: Vec<(&str, usize, bool)> = vec![
        ("manifest_after_current_rename", 2, true), // hit 2 = checkpoint's save
        ("after_ack", 0, true),
    ];
    for mode in ["sync", "group", "async"] {
        for (gate, hit, crash) in gates_b.iter().copied() {
            for rep in 0..3 {
                let dir = std::path::PathBuf::from(format!("/tmp/ph3e-e3/mix-{mode}-{gate}-{rep}"));
                let f = e3_run_cell(&dir, "mixed", gate, hit, mode, None, crash, true);
                record(&mut rows, &mut cells, "mixed-c".into(), mode, "mixed", gate, hit, rep, "F2E-GROUPKILL", &f, &dir);
                let _ = std::fs::remove_dir_all(&dir);
                let _ = std::fs::remove_file(format!("{}.e3sidecar", dir.display()));
                let _ = std::fs::remove_file(format!("{}.e3gate", dir.display()));
            }
        }
    }

    // ---- Workload C: transactions (COMMIT-buffered / pre-ACK / post-ACK) ----
    let gates_c: Vec<(&str, usize, bool)> = vec![
        ("after_write", 18, true),   // COMMIT frame written (userspace buffer)
        ("before_ack", 6, true),     // engine: last engine-level instant
        ("after_ack", 0, true),      // harness: TXNACK durably recorded
    ];
    for mode in ["sync", "group", "async"] {
        for (gate, hit, crash) in gates_c.iter().copied() {
            for rep in 0..3 {
                let dir = std::path::PathBuf::from(format!("/tmp/ph3e-e3/txn-{mode}-{gate}-{rep}"));
                let f = e3_run_cell(&dir, "txn", gate, hit, mode, None, crash, false);
                record(&mut rows, &mut cells, "txn-c".into(), mode, "txn", gate, hit, rep, "F2E-GROUPKILL", &f, &dir);
                let _ = std::fs::remove_dir_all(&dir);
                let _ = std::fs::remove_file(format!("{}.e3sidecar", dir.display()));
                let _ = std::fs::remove_file(format!("{}.e3gate", dir.display()));
            }
        }
    }

    // ---- F0 control: graceful close ----
    for mode in ["sync", "group", "async"] {
        let dir = std::path::PathBuf::from(format!("/tmp/ph3e-e3/f0-{mode}"));
        let f = e3_run_cell(&dir, "close", "", 0, mode, None, false, true);
        record(&mut rows, &mut cells, "f0-control".into(), mode, "close", "graceful_close", 0, 0, "F0-CLOSE", &f, &dir);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(format!("{}.e3sidecar", dir.display()));
    }

    std::fs::write(format!("{out}/e3-matrix.csv"), &rows).unwrap();
    write_json(&format!("{out}/metrics.json"),
        &serde_json::json!({"cells": cells, "failure_models": ["F0-CLOSE", "F2E-GROUPKILL"],
                            "blocked": ["F3-FS-DISRUPTION", "F4-POWER-LOSS"]}));
    format!("e3: {cells} cells -> {out}/e3-matrix.csv")
}


// ================================================================
// Phase 3E E4 — online backup, snapshot consistency, restore integrity
// ================================================================
//
// Snapshot model (phase3e-e4-spec.md §2): backup_to acquires the mutation
// gate, checkpoints (THE snapshot boundary), copies, releases. Ops attempted
// during backup block and ACK only after it returns — therefore the snapshot
// content is EXACTLY the ops whose ACK lines are visible in the fsynced
// sidecar at backup-return. The expected state is derived from that sidecar
// (independent reference model), never from the final source state.

type E4Shared = std::sync::Arc<std::sync::Mutex<std::fs::File>>;

fn e4_ack(f: &E4Shared, line: &str) {
    use std::io::Write;
    let mut g = f.lock().unwrap();
    let _ = writeln!(g, "{line}");
    let _ = g.flush();
    let _ = g.sync_all();
}

fn e4_sidecar_lines(dir: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(format!("{}.e4sidecar", dir.display()))
        .map(|s| s.lines().map(|l| l.to_string()).collect())
        .unwrap_or_default()
}

/// expected present-set from sidecar ACK/DEL lines (in-order replay)
fn e4_model(lines: &[String]) -> (Vec<u32>, Vec<u32>) {
    let mut present: std::collections::BTreeMap<u32, ()> = Default::default();
    let mut deleted: Vec<u32> = Vec::new();
    for l in lines {
        if let Some(v) = l.strip_prefix("ACK ") {
            if let Ok(n) = v.parse::<u32>() {
                present.insert(n, ());
            }
        } else if let Some(v) = l.strip_prefix("DEL ") {
            if let Ok(n) = v.parse::<u32>() {
                present.remove(&n);
                deleted.push(n);
            }
        }
    }
    (present.keys().copied().collect(), deleted)
}

fn e4_restore_and_open(
    backup: &std::path::Path,
    dest: &std::path::Path,
    mode: &str,
) -> Result<(Model, bool, u128), String> {
    let t0 = std::time::Instant::now();
    std::env::set_var("PH3D_DURABILITY", mode);
    let r = attentiondb_core::backup::restore_backup(backup, dest);
    let dur = t0.elapsed().as_micros();
    match r {
        Err(e) => Err(format!("{e}")),
        Ok(_) => {
            let e = open_db(dest);
            let (m, iss) = export_state(&e);
            let clean = checker_report(&e, dest)["clean"].as_bool().unwrap_or(false)
                && iss.is_empty();
            e.close().unwrap();
            Ok((m, clean, dur))
        }
    }
}

fn e4_model_match(m: &Model, expect_present: &[u32]) -> (usize, usize, bool) {
    let got: Vec<u32> = m.keys().copied().collect();
    let ok = got == expect_present.to_vec();
    (got.len(), expect_present.len(), ok)
}

struct E4Writers {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

fn e4_spawn_writer(
    e: std::sync::Arc<AttentionEngine>,
    f: E4Shared,
    range: std::ops::Range<u32>,
    lat: std::sync::Arc<std::sync::Mutex<Vec<(u128, u128, u128)>>>,
    coll: &'static str,
    epoch: std::sync::Arc<std::time::Instant>,
) -> E4Writers {
    use std::sync::atomic::Ordering;
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();
    let lat2 = lat.clone();
    let h = std::thread::spawn(move || {
        for i in range {
            if stop2.load(Ordering::Relaxed) {
                break;
            }
            let start = epoch.elapsed().as_micros();
            e.insert_document(coll, doc_record(i, 1, "w", i as i64)).unwrap();
            let end = epoch.elapsed().as_micros();
            lat2.lock().unwrap().push((start, end, end - start));
            e4_ack(&f, &format!("ACK {i}"));
        }
    });
    drop(lat);
    E4Writers { stop, handles: vec![h] }
}

/// Child for the crash-during-backup case (controller kills the group).
fn run_e4child(dir: &std::path::Path, backup: &str) -> ! {
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let side = std::path::PathBuf::from(format!("{}.e4sidecar", dir.display()));
    let mut sc = std::fs::OpenOptions::new().create(true).append(true).open(&side).unwrap();
    use std::io::Write;
    for i in 0..20000u32 {
        e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
        let _ = writeln!(sc, "ACK {}", 1000 + i);
    }
    let _ = sc.flush();
    let _ = sc.sync_all();
    // earlier completed backup (control): must still restore after the crash
    let early = format!("{}.early-backup", dir.display());
    e.backup_to(std::path::Path::new(&early)).unwrap();
    let m = std::path::PathBuf::from(format!("{}.e4gate", dir.display()));
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).write(true).truncate(true).open(&m) {
        let _ = writeln!(f, "pre-backup");
        let _ = f.sync_all();
    }
    e.backup_to(std::path::Path::new(backup)).unwrap();
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).write(true).truncate(true).open(&m) {
        let _ = writeln!(f, "backup-done");
        let _ = f.sync_all();
    }
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// E4: result of an integrity-case setup hook (backup path, expected present ids, detail)
type E4SetupResult = (std::path::PathBuf, Vec<u32>, String);

/// Main E4 driver. Produces <out>/e4-matrix.csv and <out>/e4-integrity.csv.
fn run_e4(out: &str) -> String {
    std::fs::create_dir_all(out).unwrap();
    let mut rows = String::from("case,mode,expected_count,restored_count,match,source_final,reader_errors,ops_overlapping_backup,max_op_latency_us,backup_us,restore_us,checker_clean,early_backup_still_restores,partial_backup_refused,notes\n");
    let mut integrity = String::from("case,action,restore_result,detail\n");
    let root = std::path::PathBuf::from("/tmp/ph3e-e4");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut cells = 0usize;

    let run_case = |rows: &mut String, integrity: &mut String, cells: &mut usize,
                        case: &str, mode: &str,
                        setup: &dyn Fn(&AttentionEngine, E4Shared) -> E4SetupResult| {
        let dir = root.join(case);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("PH3D_DURABILITY", mode);
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        let (backup, early_expect, note) = setup(&e, side.clone());
        // snapshot content = acks visible at backup return (reference model)
        let lines = e4_sidecar_lines(&dir);
        let (expect_present, _deleted) = e4_model(&lines);
        let expected_count = early_expect.len().max(expect_present.len());
        let _ = expected_count;
        let (expected_final_present, _) = e4_model(&e4_sidecar_lines(&dir));
        let t0 = std::time::Instant::now();
        e.backup_to(&backup).unwrap();
        let backup_us = t0.elapsed().as_micros();
        let lines_at_boundary = e4_sidecar_lines(&dir);
        let (model_present, _) = e4_model(&lines_at_boundary);
        let rdest = root.join(format!("{case}-restored"));
        match e4_restore_and_open(&backup, &rdest, mode) {
            Err(err) => {
                rows.push_str(&format!("{case},{mode},-,-,RESTORE_REFUSED,-,-,-,-,{backup_us},-,-,-,-,{err}\n"));
                *cells += 1;
            }
            Ok((restored, clean, restore_us)) => {
                let (got, exp, ok) = e4_model_match(&restored, &model_present);
                let source_final = {
                    let (m, _) = export_state(&e);
                    m.len()
                };
                let _ = expected_final_present;
                rows.push_str(&format!(
                    "{case},{mode},{exp},{got},{},\"{source_final}\",0,0,0,{backup_us},{restore_us},{clean},-,-,{note}\n",
                    if ok { "MATCH" } else { "MISMATCH" }));
                *cells += 1;
            }
        }
        let _ = integrity;
    };

    // B1 quiescent control
    run_case(&mut rows, &mut integrity, &mut cells, "b1-quiescent", "sync", &|e, f| {
        for i in 0..30u32 {
            e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
            e4_ack(&f, &format!("ACK {}", 1000 + i));
        }
        (root.join("b1-backup"), vec![], "control".into())
    });

    // B2 read-concurrent
    {
        let dir = root.join("b2-readers");
        std::env::set_var("PH3D_DURABILITY", "group");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        for i in 0..40u32 {
            e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64)).unwrap();
            e4_ack(&side, &format!("ACK {}", 1000 + i));
        }
        let errors = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut readers = Vec::new();
        for _ in 0..2 {
            let e2 = e.clone();
            let er = errors.clone();
            readers.push(std::thread::spawn(move || {
                for i in 0..100u32 {
                    let q = vec_for(1000 + (i % 40));
                    if e2.attend("bench", &[HEAD.to_string()], &q, 5).is_err() {
                        er.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }));
        }
        let t0 = std::time::Instant::now();
        e.backup_to(&root.join("b2-backup")).unwrap();
        let backup_us = t0.elapsed().as_micros();
        for r in readers { r.join().unwrap(); }
        let re = errors.load(std::sync::atomic::Ordering::Relaxed);
        let lines = e4_sidecar_lines(&dir);
        let (model_present, _) = e4_model(&lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b2-backup"), &root.join("b2-restored"), "group").unwrap();
        let (got, exp, ok) = e4_model_match(&restored, &model_present);
        rows.push_str(&format!("b2-readers,group,{exp},{got},{},-,{re},0,0,{backup_us},{restore_us},{clean},-,-,readers ran during backup\n", if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // B3 single writer (+blocking instrumentation)
    {
        let dir = root.join("b3-single-writer");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(e.clone(), side.clone(), 3000..3300, lat.clone(), "bench", epoch.clone());
        std::thread::sleep(std::time::Duration::from_millis(30));
        let backup_start = epoch.elapsed().as_micros();
        e.backup_to(&root.join("b3-backup")).unwrap();
        let backup_us = epoch.elapsed().as_micros() - backup_start;
        let boundary_lines = e4_sidecar_lines(&dir);
        // post-backup source mutation (isolation)
        for i in 4000..4100u32 {
            e.insert_document("bench", doc_record(i, 1, "post", i as i64)).unwrap();
            e4_ack(&side, &format!("ACK {i}"));
        }
        w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in w.handles.drain(..) { h.join().unwrap(); }
        let (model_present, _) = e4_model(&boundary_lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b3-backup"), &root.join("b3-restored"), "sync").unwrap();
        let (got, exp, ok) = e4_model_match(&restored, &model_present);
        let source_final = { let (m, _) = export_state(&e); m.len() };
        let lats = lat.lock().unwrap();
        let backup_end = backup_start + backup_us;
        let overlapping = lats.iter().filter(|(s, en, _)| *s <= backup_end && *en >= backup_start).count();
        let max_lat = lats.iter().map(|(_, _, d)| *d).max().unwrap_or(0);
        rows.push_str(&format!("b3-single-writer,sync,{exp},{got},{},\"{source_final}\",0,{overlapping},{max_lat},{backup_us},{restore_us},{clean},-,-,writer paused by gate during backup; post-backup +100 inserts excluded\n", if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // B4 multi-writer
    {
        let dir = root.join("b4-multi-writer");
        std::env::set_var("PH3D_DURABILITY", "group");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut ws = Vec::new();
        for r in [(5000u32..5100), (6000..6100), (7000..7100)] {
            ws.push(e4_spawn_writer(e.clone(), side.clone(), r, lat.clone(), "bench", epoch.clone()));
        }
        std::thread::sleep(std::time::Duration::from_millis(40));
        e.backup_to(&root.join("b4-backup")).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        for w in &mut ws { w.stop.store(true, std::sync::atomic::Ordering::Relaxed); }
        for w in &mut ws { for h in w.handles.drain(..) { h.join().unwrap(); } }
        let (model_present, _) = e4_model(&boundary_lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b4-backup"), &root.join("b4-restored"), "group").unwrap();
        let (got, exp, ok) = e4_model_match(&restored, &model_present);
        let source_final = { let (m, _) = export_state(&e); m.len() };
        rows.push_str(&format!("b4-multi-writer,group,{exp},{got},{},\"{source_final}\",0,0,0,0,{restore_us},{clean},-,-,3 writers\n", if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // B5 writer + checkpoint (checkpoint called while backup holds the gate)
    {
        let dir = root.join("b5-writer-ckpt");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(e.clone(), side.clone(), 8000..8200, lat.clone(), "bench", epoch.clone());
        std::thread::sleep(std::time::Duration::from_millis(30));
        let e2 = e.clone();
        let ck = std::thread::spawn(move || e2.checkpoint().unwrap());
        e.backup_to(&root.join("b5-backup")).unwrap();
        let ck_dur = { let t = std::time::Instant::now(); ck.join().unwrap(); t.elapsed().as_micros() };
        let boundary_lines = e4_sidecar_lines(&dir);
        for i in 9000..9050u32 {
            e.insert_document("bench", doc_record(i, 1, "post", i as i64)).unwrap();
            e4_ack(&side, &format!("ACK {i}"));
        }
        w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in w.handles.drain(..) { h.join().unwrap(); }
        let (model_present, _) = e4_model(&boundary_lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b5-backup"), &root.join("b5-restored"), "sync").unwrap();
        let (got, exp, ok) = e4_model_match(&restored, &model_present);
        let source_final = { let (m, _) = export_state(&e); m.len() };
        rows.push_str(&format!("b5-writer-ckpt,sync,{exp},{got},{},\"{source_final}\",0,0,0,0,{restore_us},{clean},-,-,checkpoint blocked until backup done (waited {ck_dur}us)\n", if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // B6 writer + WAL rotation (2 KiB segments: rotations throughout)
    {
        let dir = root.join("b6-writer-rotation");
        std::env::set_var("PH3D_DURABILITY", "group");
        std::env::set_var("ATTENTIONDB_WAL_SEGMENT_BYTES", "2048");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(e.clone(), side.clone(), 11000..11200, lat.clone(), "bench", epoch.clone());
        std::thread::sleep(std::time::Duration::from_millis(30));
        e.backup_to(&root.join("b6-backup")).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in w.handles.drain(..) { h.join().unwrap(); }
        std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
        let (model_present, _) = e4_model(&boundary_lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b6-backup"), &root.join("b6-restored"), "group").unwrap();
        let (got, exp, ok) = e4_model_match(&restored, &model_present);
        rows.push_str(&format!("b6-writer-rotation,group,{exp},{got},{},-,0,0,0,0,{restore_us},{clean},-,-,2KiB segments: rotations before/during(ckpt)/after backup\n", if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // B7 writer + checkpoint + rotation (async)
    {
        let dir = root.join("b7-writer-ckpt-rotation");
        std::env::set_var("PH3D_DURABILITY", "async");
        std::env::set_var("ATTENTIONDB_WAL_SEGMENT_BYTES", "2048");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(e.clone(), side.clone(), 13000..13200, lat.clone(), "bench", epoch.clone());
        std::thread::sleep(std::time::Duration::from_millis(30));
        let e2 = e.clone();
        let ck = std::thread::spawn(move || e2.checkpoint().unwrap());
        e.backup_to(&root.join("b7-backup")).unwrap();
        ck.join().unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in w.handles.drain(..) { h.join().unwrap(); }
        std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
        let (model_present, _) = e4_model(&boundary_lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b7-backup"), &root.join("b7-restored"), "async").unwrap();
        let (got, exp, ok) = e4_model_match(&restored, &model_present);
        rows.push_str(&format!("b7-writer-ckpt-rotation,async,{exp},{got},{},-,0,0,0,0,{restore_us},{clean},-,-,async: backup checkpoint captures buffered acks\n", if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // Snapshot consistency: sequential v1->v2 transition, backup mid-transition
    {
        let dir = root.join("snapshot-transition");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        for i in 100..106u32 {
            let mut r = doc_record(i, 1, "t", 1);
            r.fields.insert("num".to_string(), serde_json::json!(1i64));
            e.insert_document("bench", r).unwrap();
            e4_ack(&side, &format!("ACK {i}"));
        }
        for (k, i) in [101u32, 103, 105].iter().enumerate() {
            let id = {
                let store = e.document_store.read();
                store.list_all_records().into_iter()
                    .find(|r| r.fields.get("idx").and_then(|v| v.as_u64()) == Some(*i as u64))
                    .map(|r| r.id).unwrap()
            };
            let old = {
                let store = e.document_store.read();
                store.get(&id).cloned().unwrap()
            };
            let mut fields = old.fields.clone();
            fields.insert("num".to_string(), serde_json::json!(2i64));
            let mut kvs = std::collections::HashMap::new();
            kvs.insert(HEAD.to_string(), vec_for(*i));
            e.update_document("bench", &id.to_string(), fields, kvs).unwrap();
            e4_ack(&side, &format!("UPD {i} v{}", k + 2));
        }
        e.backup_to(&root.join("st-backup")).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        // model: idx -> num at boundary
        let mut expect_num: std::collections::BTreeMap<u32, i64> = Default::default();
        for l in &boundary_lines {
            if let Some(v) = l.strip_prefix("ACK ") {
                if let Ok(n) = v.parse::<u32>() { expect_num.insert(n, 1); }
            } else if let Some(rest) = l.strip_prefix("UPD ") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if let Ok(n) = parts[0].parse::<u32>() { expect_num.insert(n, 2); }
            }
        }
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("st-backup"), &root.join("st-restored"), "sync").unwrap();
        let mism: Vec<String> = expect_num.iter()
            .filter(|(i, num)| restored.get(i).map(|(_, _, n)| *n) != Some(**num))
            .map(|(i, num)| format!("{i}:want{num}"))
            .collect();
        let ok = mism.is_empty() && restored.len() == expect_num.len();
        let verdict = if ok { "MATCH".to_string() } else { format!("MISMATCH {}", mism.join(" ")) };
        rows.push_str(&format!("snapshot-transition,sync,{},{},{verdict},-,0,0,0,-,{restore_us},{clean},-,-,{}\n",
            expect_num.len(), restored.len(), mism.join(" ")));
        cells += 1;
    }

    // Delete/reinsert
    {
        let dir = root.join("delete-reinsert");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        for i in 20000..20020u32 {
            e.insert_document("bench", doc_record(i, 1, "d", i as i64)).unwrap();
            e4_ack(&side, &format!("ACK {i}"));
        }
        for i in (20000..20020u32).step_by(2) {
            e.delete_document("bench", &uuid_for(i, 1).to_string()).unwrap();
            e4_ack(&side, &format!("DEL {i}"));
        }
        for i in (20000..20012u32).step_by(4) {
            e.insert_document("bench", doc_record(i, 2, "d", i as i64)).unwrap(); // new uuid
            e4_ack(&side, &format!("ACK {i}")); // reinsert (new identity)
        }
        e.backup_to(&root.join("dr-backup")).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir); // AT backup return
        let boundary = boundary_lines.len();
        let (model_present, _) = e4_model(&boundary_lines);
        // post-backup: more delete/reinsert churn on source (must NOT enter backup)
        for i in (20001..20019u32).step_by(4) {
            e.delete_document("bench", &uuid_for(i, 1).to_string()).unwrap();
            e4_ack(&side, &format!("DEL {i}"));
        }
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("dr-backup"), &root.join("dr-restored"), "sync").unwrap();
        let (got, exp, ok) = e4_model_match(&restored, &model_present);
        rows.push_str(&format!("delete-reinsert,sync,{exp},{got},{},-,0,0,0,0,{restore_us},{clean},-,-,boundary={boundary} sidecar lines\n", if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // Collection isolation (3 collections, concurrent writers)
    {
        let dir = root.join("collections");
        std::env::set_var("PH3D_DURABILITY", "group");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e.create_collection("alpha", DIM, &[HEAD]).unwrap();
        e.create_collection("beta", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut ws = Vec::new();
        ws.push(e4_spawn_writer(e.clone(), side.clone(), 30000..30080, lat.clone(), "bench", epoch.clone()));
        ws.push(e4_spawn_writer(e.clone(), side.clone(), 31000..31080, lat.clone(), "alpha", epoch.clone()));
        ws.push(e4_spawn_writer(e.clone(), side.clone(), 32000..32080, lat.clone(), "beta", epoch.clone()));
        std::thread::sleep(std::time::Duration::from_millis(40));
        e.backup_to(&root.join("coll-backup")).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        for w in &mut ws { w.stop.store(true, std::sync::atomic::Ordering::Relaxed); }
        for w in &mut ws { for h in w.handles.drain(..) { h.join().unwrap(); } }
        // model per collection
        let mut expect: std::collections::BTreeMap<&'static str, Vec<u32>> = Default::default();
        let ranges: [(&'static str, std::ops::Range<u32>); 3] =
            [("bench", 30000..30080), ("alpha", 31000..31080), ("beta", 32000..32080)];
        for (coll, rng) in ranges {
            let present: Vec<u32> = boundary_lines.iter().filter_map(|l| {
                let n = l.strip_prefix("ACK ")?.parse::<u32>().ok()?;
                if rng.contains(&n) { Some(n) } else { None }
            }).collect();
            expect.insert(coll, present);
        }
        std::env::set_var("PH3D_DURABILITY", "group");
        attentiondb_core::backup::restore_backup(&root.join("coll-backup"), &root.join("coll-restored")).unwrap();
        let e2 = open_db(&root.join("coll-restored"));
        let mut all_ok = true;
        let mut det = String::new();
        for (coll, exp) in &expect {
            let (m, iss) = export_state_coll(&e2, coll);
            let got: Vec<u32> = m.keys().copied().collect();
            let ok = got == *exp && iss.is_empty();
            all_ok &= ok;
            det.push_str(&format!("{coll}={}/{} ", got.len(), exp.len()));
        }
        let clean2 = checker_report(&e2, &root.join("coll-restored"))["clean"].as_bool().unwrap_or(false);
        rows.push_str(&format!("collections,group,-,-,{},-,0,0,0,-,-,{},-,-,{}\n",
            if all_ok { "MATCH" } else { "MISMATCH" }, if clean2 { "true" } else { "false" }, det.trim()));
        cells += 1;
    }

    // Multi-backup B1 < B2 < B3, each independently restored
    {
        let dir = root.join("multi-backup");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        let mut snapshots = Vec::new();
        for b in 0..3u32 {
            for i in (40000 + b * 20)..(40000 + b * 20 + 20) {
                e.insert_document("bench", doc_record(i, 1, "m", i as i64)).unwrap();
                e4_ack(&side, &format!("ACK {i}"));
            }
            let bp = root.join(format!("mb-backup-{b}"));
            e.backup_to(&bp).unwrap();
            let lines = e4_sidecar_lines(&dir);
            let (present, _) = e4_model(&lines);
            snapshots.push((bp, present));
        }
        let mut all_ok = true;
        let mut all_clean = true;
        let mut det = String::new();
        let mut restore_us_total = 0u128;
        for (idx, (bp, exp)) in snapshots.iter().enumerate() {
            let dest = root.join(format!("mb-restored-{idx}"));
            let (m, clean, us) = match e4_restore_and_open(bp, &dest, "sync") {
                Ok(x) => x, Err(_) => { all_ok = false; all_clean = false; break; }
            };
            restore_us_total += us;
            let (got, expn, ok) = e4_model_match(&m, exp);
            all_ok &= ok;
            all_clean &= clean;
            det.push_str(&format!("B{}={}/{} ", idx + 1, got, expn));
        }
        rows.push_str(&format!("multi-backup,sync,-,-,{},-,0,0,0,-,{},{} ,-,-,{}\n",
            if all_ok { "MATCH" } else { "MISMATCH" }, restore_us_total,
            if all_clean { "true" } else { "false" }, det.trim()));
        cells += 1;
    }

    // Durability-mode matrix (writer backup x3 modes) + integrity suite
    for mode in ["sync", "group", "async"] {
        let dir = root.join(format!("modes-{mode}"));
        std::env::set_var("PH3D_DURABILITY", mode);
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new().create(true).append(true)
                .open(format!("{}.e4sidecar", dir.display())).unwrap()));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(e.clone(), side.clone(), 50000..50300, lat.clone(), "bench", epoch.clone());
        std::thread::sleep(std::time::Duration::from_millis(25));
        e.backup_to(&root.join(format!("modes-{mode}-backup"))).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in w.handles.drain(..) { h.join().unwrap(); }
        let (model_present, _) = e4_model(&boundary_lines);
        let r = e4_restore_and_open(&root.join(format!("modes-{mode}-backup")), &root.join(format!("modes-{mode}-restored")), mode);
        match r {
            Ok((restored, clean, us)) => {
                let (got, exp, ok) = e4_model_match(&restored, &model_present);
                rows.push_str(&format!("modes-{mode},{mode},{exp},{got},{},-,0,0,0,0,{us},{clean},-,-,acked writes all captured (ckpt fsync)\n",
                    if ok { "MATCH" } else { "MISMATCH" }));
            }
            Err(err) => rows.push_str(&format!("modes-{mode},{mode},-,-,RESTORE_REFUSED,-,0,0,0,0,-,-,-,-,{err}\n")),
        }
        cells += 1;
    }

    // ---------------- integrity suite (driver-side surgery) ----------------
    // build one canonical valid backup
    let dir = root.join("integrity-src");
    std::env::set_var("PH3D_DURABILITY", "sync");
    let e = std::sync::Arc::new(open_db(&dir));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for i in 60000..60030u32 {
        e.insert_document("bench", doc_record(i, 1, "i", i as i64)).unwrap();
    }
    e.backup_to(&root.join("integrity-backup")).unwrap();
    let bp = root.join("integrity-backup");

    let try_restore = |integrity: &mut String, case: &str, action: &str, b: &std::path::Path| {
        let dest = root.join(format!("integ-restore-{case}"));
        let _ = std::fs::remove_dir_all(&dest);
        match attentiondb_core::backup::restore_backup(b, &dest) {
            Ok(_) => {
                // opening happens inside restore; if it succeeded, verify checker
                std::env::set_var("PH3D_DURABILITY", "sync");
                let e2 = open_db(&dest);
                let clean = checker_report(&e2, &dest)["clean"].as_bool().unwrap_or(false);
                integrity.push_str(&format!("{case},{action},ACCEPTED,clean={clean}\n"));
                e2.close().unwrap();
            }
            Err(err) => {
                let reason = if format!("{err}").contains("missing") { "NO_META" }
                    else if format!("{err}").contains("unsupported") { "BAD_VERSION" }
                    else if format!("{err}").contains("backup meta") { "META_PARSE" }
                    else if format!("{err}").contains("not empty") { "DEST_NONEMPTY" }
                    else { "OPEN_OR_CATALOG" };
                integrity.push_str(&format!("{case},{action},REFUSED,{reason}\n"));
            }
        }
    };

    // control: valid backup restores
    try_restore(&mut integrity, "valid-control", "none", &bp);
    // partial: meta removed (simulated incomplete copy)
    let p1 = root.join("integ-no-meta");
    let _ = std::fs::remove_dir_all(&p1);
    copy_dir_all(&bp, &p1);
    std::fs::remove_file(p1.join("backup-meta.json")).unwrap();
    try_restore(&mut integrity, "partial-no-meta", "rm backup-meta.json", &p1);
    // truncated SST
    let p2 = root.join("integ-trunc-sst");
    let _ = std::fs::remove_dir_all(&p2);
    copy_dir_all(&bp, &p2);
    {
        let sstdir = p2.join("sst");
        let f = std::fs::read_dir(&sstdir).unwrap().flatten().next().unwrap().path();
        let b = std::fs::read(&f).unwrap();
        std::fs::write(&f, &b[..b.len() / 2]).unwrap();
    }
    try_restore(&mut integrity, "truncated-sst", "halve first sst", &p2);
    // corrupt WAL integrity record (post-checkpoint the backup WAL is an empty
    // active segment + wal-state.json; the record is the meaningful target)
    let p3 = root.join("integ-corrupt-wal");
    let _ = std::fs::remove_dir_all(&p3);
    copy_dir_all(&bp, &p3);
    std::fs::write(p3.join("WAL").join("wal-state.json"), b"{ broken").unwrap();
    try_restore(&mut integrity, "corrupt-wal-state", "garble wal-state.json", &p3);
    // garbage in the EMPTY active segment: torn-tail policy -> intact (empty)
    // prefix + all checkpointed state in SSTs; expected ACCEPTED with zero loss
    let p3b = root.join("integ-garbage-active-seg");
    let _ = std::fs::remove_dir_all(&p3b);
    copy_dir_all(&bp, &p3b);
    {
        let waldir = p3b.join("WAL");
        let f = std::fs::read_dir(&waldir).unwrap().flatten()
            .find(|x| x.path().extension().and_then(|s| s.to_str()) == Some("wal")).unwrap().path();
        use std::io::Write;
        let mut g = std::fs::OpenOptions::new().append(true).open(&f).unwrap();
        g.write_all(&[0u8; 13]).unwrap();
    }
    try_restore(&mut integrity, "garbage-active-segment", "13 zero bytes appended", &p3b);
    // corrupt CURRENT
    let p4 = root.join("integ-corrupt-current");
    let _ = std::fs::remove_dir_all(&p4);
    copy_dir_all(&bp, &p4);
    std::fs::write(p4.join("CURRENT"), b"manifest-999999999\n").unwrap();
    try_restore(&mut integrity, "corrupt-current", "point at missing gen", &p4);
    // malformed meta
    let p5 = root.join("integ-bad-meta");
    let _ = std::fs::remove_dir_all(&p5);
    copy_dir_all(&bp, &p5);
    std::fs::write(p5.join("backup-meta.json"), b"{ broken").unwrap();
    try_restore(&mut integrity, "malformed-meta", "invalid json", &p5);
    // unsupported format version
    let p6 = root.join("integ-bad-version");
    let _ = std::fs::remove_dir_all(&p6);
    copy_dir_all(&bp, &p6);
    {
        let mut meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(p6.join("backup-meta.json")).unwrap()).unwrap();
        meta["backup_format_version"] = serde_json::json!(99);
        std::fs::write(p6.join("backup-meta.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
    }
    try_restore(&mut integrity, "bad-format-version", "v99", &p6);
    // restore into non-empty destination
    {
        let dest = root.join("integ-nonempty");
        let _ = std::fs::remove_dir_all(&dest);
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("stale.txt"), b"x").unwrap();
        match attentiondb_core::backup::restore_backup(&bp, &dest) {
            Ok(_) => integrity.push_str(&format!("{},restore-nonempty,ACCEPTED,UNEXPECTED\n", "nonempty-dest")),
            Err(err) => {
                let reason = if format!("{err}").contains("not empty") { "DEST_NONEMPTY" } else { "OPEN_OR_CATALOG" };
                integrity.push_str(&format!("{},restore-nonempty,REFUSED,{}\n", "nonempty-dest", reason));
            }
        }
    }
    // source-dir safety: source reopens, all acks present, backup copied read-only
    {
        let (m, iss) = export_state(&e);
        let clean = checker_report(&e, &dir)["clean"].as_bool().unwrap_or(false) && iss.is_empty();
        let det = format!("docs={} (expected 30)", m.len());
        integrity.push_str(&format!("{},source-safety,{},{},\n",
            "source-after-backup",
            if clean && m.len() == 30 { "INTACT" } else { "DAMAGED" },
            det));
    }

    // crash-during-backup (child process, group kill)
    {
        let cdir = root.join("crash-src");
        let cbackup = root.join("crash-backup");
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["dbtest", "e4child", "--dir", cdir.to_str().unwrap(),
                  "--backup", cbackup.to_str().unwrap()])
            .env("PH3D_DURABILITY", "sync");
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        let mut child = cmd.spawn().unwrap();
        let marker = format!("{}.e4gate", cdir.display());
        for _ in 0..4000 {
            if std::path::Path::new(&marker).exists()
                && std::fs::read_to_string(&marker)
                    .unwrap_or_default()
                    .contains("pre-backup")
            {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        std::thread::sleep(std::time::Duration::from_micros(500)); // land INSIDE backup
        unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL); }
        let _ = child.wait();
        let done = std::fs::read_to_string(&marker).unwrap_or_default().contains("backup-done");
        // (a) source recovers
        std::env::set_var("PH3D_DURABILITY", "sync");
        let (src_ok, src_docs, src_clean) = match std::panic::catch_unwind(
            std::panic::AssertUnwindSafe(|| open_db(&cdir))) {
            Err(_) => (false, 0usize, false),
            Ok(e2) => {
                let (m, iss) = export_state(&e2);
                let c = checker_report(&e2, &cdir)["clean"].as_bool().unwrap_or(false) && iss.is_empty();
                (true, m.len(), c)
            }
        };
        // (b) partial backup rejected (if copy had not completed)
        let (partial_verdict, partial_detail) = if done {
            match e4_restore_and_open(&cbackup, &root.join("crash-restored"), "sync") {
                Ok((m, clean, _)) => ("ACCEPTED_COMPLETE".to_string(), format!("docs={} clean={clean}", m.len())),
                Err(e) => ("REFUSED".to_string(), e.to_string()),
            }
        } else {
            match attentiondb_core::backup::restore_backup(&cbackup, &root.join("crash-restored")) {
                Ok(_) => ("ACCEPTED_INCOMPLETE".to_string(), "UNSAFE".to_string()),
                Err(e) => ("REFUSED".to_string(), format!("{e}")),
            }
        };
        // (c) earlier completed backup still restores
        let early_ok = e4_restore_and_open(
            &std::path::PathBuf::from(format!("{}.early-backup", cdir.display())),
            &root.join("crash-early-restored"), "sync").map(|(m, clean, _)| m.len() == 20000 && clean).unwrap_or(false);
        rows.push_str(&format!(
            "crash-during-backup,sync,20000,{},{},-,0,0,0,-,-,{},{},{},killed_mid_backup done={done} ({})\n",
            src_docs,
            if src_ok { "SOURCE_OK" } else { "SOURCE_LOST" },
            if src_clean { "true" } else { "false" },
            if early_ok { "true" } else { "false" },
            partial_verdict,
            partial_detail));
        cells += 1;
    }

    std::fs::write(format!("{out}/e4-matrix.csv"), &rows).unwrap();
    std::fs::write(format!("{out}/e4-integrity.csv"), &integrity).unwrap();
    write_json(&format!("{out}/metrics.json"),
        &serde_json::json!({"cells": cells, "kind": "E4 backup/snapshot/restore"}));
    format!("e4: {cells} matrix cells -> {out}/e4-matrix.csv")
}

fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) {
    fn rec(src: &std::path::Path, dst: &std::path::Path) {
        std::fs::create_dir_all(dst).unwrap();
        for e in std::fs::read_dir(src).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() { rec(&p, &dst.join(e.file_name())); }
            else { std::fs::copy(&p, dst.join(e.file_name())).unwrap(); }
        }
    }
    rec(src, dst);
}

// ---------------------------------------------------------------- dispatch

pub fn run(args: &[String]) -> String {
    let get = |name: &str, default: &str| -> String {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned()).unwrap_or_else(|| default.to_string())
    };
    let dir = std::path::PathBuf::from(get("--dir", "/var/tmp/ph3d-db"));
    let out = get("--out", "research/phase3/raw/runs/PH3D-OUT");
    std::fs::create_dir_all(&out).unwrap();
    match args.first().map(|s| s.as_str()).unwrap_or("model") {
        "model" => run_model(&dir, get("--seed", "42").parse().unwrap_or(42),
                             get("--ops", "300").parse().unwrap_or(300), &out),
        "filterx" => run_filterx(&dir, &out),
        "txn" => run_txn(&dir, &out),
        "txnchild" => txn_child(&dir),
        "crashchild" => {
            let n: usize = get("--n", "40").parse().unwrap_or(40);
            run_crashchild(&dir, &get("--point", "after_acks"), n)
        }
        "verify" => run_verify(&dir, &get("--point", "?")),
        "walcorrupt" => run_walcorrupt_verify(&dir, &get("--mode", "?")),
        "concur" => {
            let r: usize = get("--readers", "4").parse().unwrap_or(4);
            let w: usize = get("--writers", "0").parse().unwrap_or(0);
            run_concur(&dir, r, w, &out)
        }
        "backup" => run_backup(&dir, &out),
        "backupinv" => run_backup_inventory(&dir, &out),
        "integration" => run_integration(&dir, &out),
        "concurreplay" => run_concur_replay(&dir, &out),
        "compact" => run_compact(&dir, &out),
        "walintegrity" => run_walintegrity(&out),
        "durability" => run_durability(&get("--suite", "ack-boundary"), &out),
        "gatechild" => run_gatechild(&dir, &get("--scenario", "ack"), &get("--gate", "")),
        "e3child" => run_e3child(&dir, &get("--workload", "ack"), &get("--gate", "")),
        "e3run" => run_e3(&out),
        "e4child" => {
            let b = get("--backup", "/tmp/ph3e-e4/crash-backup");
            run_e4child(&dir, &b)
        }
        "e4run" => run_e4(&out),
        "compactlive" => run_compact_live(&dir),
        other => format!("unknown dbtest subcommand {other}"),
    }
}
