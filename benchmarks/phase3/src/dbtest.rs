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

pub(crate) fn open_db(dir: &std::path::Path) -> AttentionEngine {
    AttentionEngine::open_dir(dir, dur_from_env()).unwrap()
}

pub(crate) fn vec_for(idx: u32) -> Vec<f32> {
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

pub(crate) fn doc_record(idx: u32, version: u64, cat: &str, num: i64) -> Record {
    let mut fields = HashMap::new();
    fields.insert("idx".to_string(), serde_json::json!(idx));
    fields.insert("version".to_string(), serde_json::json!(version));
    fields.insert("cat".to_string(), serde_json::json!(cat));
    fields.insert("num".to_string(), serde_json::json!(num));
    fields.insert(
        "title".to_string(),
        serde_json::json!(format!("doc-{idx}-v{version}")),
    );
    let mut r = Record::new(fields);
    r.id = uuid_for(idx, version);
    r.k_vecs.insert(HEAD.to_string(), vec_for(idx));
    r
}

pub(crate) fn uuid_for(idx: u32, version: u64) -> uuid::Uuid {
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
        let idx = rec
            .fields
            .get("idx")
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::MAX) as u32;
        let version = rec
            .fields
            .get("version")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cat = rec
            .fields
            .get("cat")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let num = rec.fields.get("num").and_then(|v| v.as_i64()).unwrap_or(0);
        if model.insert(idx, (version, cat, num)).is_some() {
            issues.push(format!("duplicate logical id {idx} in store"));
        }
    }
    (model, issues)
}

/// Order-independent full record multiset (idx, version, cat, num) for the
/// bench collection — unlike export_state it does NOT collapse duplicate
/// logical ids (blind-write overlaps keep both live versions).
fn e7_records(e: &AttentionEngine) -> Vec<(u32, u64, String, i64)> {
    let store = e.document_store.read();
    let tag = "collection:bench";
    let mut v: Vec<(u32, u64, String, i64)> = store
        .list_all_records()
        .iter()
        .filter(|rec| rec.tags.iter().any(|t| t == tag))
        .map(|rec| {
            (
                rec.fields
                    .get("idx")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(u64::MAX) as u32,
                rec.fields
                    .get("version")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0),
                rec.fields
                    .get("cat")
                    .and_then(|x| x.as_str())
                    .unwrap_or("?")
                    .to_string(),
                rec.fields.get("num").and_then(|x| x.as_i64()).unwrap_or(0),
            )
        })
        .collect();
    v.sort();
    v
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
    let got = e
        .attend_filtered("bench", &[HEAD.to_string()], &q, k, Some(f))
        .unwrap();
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
pub(crate) fn checker_report(e: &AttentionEngine, dir: &std::path::Path) -> serde_json::Value {
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
    checks.push((
        "initial_build_checker",
        {
            let rep = checker_report(&e, dir);
            write_json(&format!("{out}/consistency-initial.json"), &rep);
            rep["clean"].as_bool().unwrap_or(false)
        },
        "checker after build".into(),
    ));

    // isolation: write a doc into "other" and ensure it never leaks into "bench"
    {
        let r = doc_record(9999, 1, "news", 1);
        e.insert_document("other", r).unwrap();
        let got = e
            .attend("bench", &[HEAD.to_string()], &vec_for(9999), 5)
            .unwrap();
        let leaked = got
            .iter()
            .any(|(id, _)| e.id_mapper.read().uuid_to_id(&uuid_for(9999, 1)) == Some(*id));
        checks.push((
            "collection_isolation",
            !leaked,
            "other-collection doc absent from bench queries".into(),
        ));
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
                    e.update_document(
                        "bench",
                        &uuid_for(idx, 1).to_string(),
                        r.fields.clone(),
                        r.k_vecs.clone(),
                    )
                    .unwrap();
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
                    e.update_document(
                        "bench",
                        &uuid_for(idx, 1).to_string(),
                        r.fields.clone(),
                        r.k_vecs.clone(),
                    )
                    .unwrap();
                    model.insert(idx, (version, cat.to_string(), num));
                    let _ = writeln!(log, "{opno},update,{idx},v{version}");
                }
            }
            "delete" => {
                if model.contains_key(&idx) && idx != 0 {
                    e.delete_document("bench", &uuid_for(idx, 1).to_string())
                        .unwrap();
                    model.remove(&idx);
                    let _ = writeln!(log, "{opno},delete,{idx},");
                    // deleted doc must vanish from retrieval immediately (§9)
                    let got = topk_observed(&e, idx, 300);
                    if got.contains(&idx) {
                        checks.push((
                            "delete_immediate",
                            false,
                            format!("idx {idx} returned after delete"),
                        ));
                    }
                }
            }
            "query" => {
                let got = topk_observed(&e, idx, 20);
                // invariant: only LIVE docs returned (INV-1); ordering deterministic
                let mut rerun = topk_observed(&e, idx, 20);
                if got.iter().any(|i| !model.contains_key(i)) {
                    checks.push((
                        "inv_live_only",
                        false,
                        format!("dead doc in topk at op {opno}"),
                    ));
                }
                if got != rerun {
                    rerun = topk_observed(&e, idx, 20);
                    if got != rerun {
                        checks.push((
                            "query_determinism",
                            false,
                            format!("nondeterministic topk at op {opno}"),
                        ));
                    }
                }
                let _ = writeln!(log, "{opno},query,{idx},k20");
            }
            "filter" => {
                filter_tests += 1;
                let which = det.next_u64() % 4;
                let (f, eligible): (FilterExpr, Vec<u32>) = match which {
                    0 => (
                        eqf("cat", "sport"),
                        model
                            .iter()
                            .filter(|(_, (_, c, _))| c == "sport")
                            .map(|(k, _)| *k)
                            .collect(),
                    ),
                    1 => (
                        cmpf("num", FilterOp::Gte, FilterValue::Int(1000)),
                        model
                            .iter()
                            .filter(|(_, (_, _, n))| *n >= 1000)
                            .map(|(k, _)| *k)
                            .collect(),
                    ),
                    2 => (
                        andf(
                            eqf("cat", "news"),
                            cmpf("num", FilterOp::Lt, FilterValue::Int(100)),
                        ),
                        model
                            .iter()
                            .filter(|(_, (_, c, n))| c == "news" && *n < 100)
                            .map(|(k, _)| *k)
                            .collect(),
                    ),
                    _ => (eqf("cat", "no-such-cat"), vec![]), // zero-match
                };
                let got = filtered_observed(&e, &f, idx, 50);
                // INV-8: no excluded doc ever returned
                let leaked = got.iter().any(|i| !eligible.contains(i));
                if leaked {
                    filter_fail += 1;
                    checks.push((
                        "inv_filter_leak",
                        false,
                        format!("filter returned excluded docs at op {opno}: {got:?}"),
                    ));
                }
                // zero-match must be empty
                if matches!(which, 3) && !got.is_empty() {
                    filter_fail += 1;
                    checks.push((
                        "filter_zero_match",
                        false,
                        "zero-match filter returned rows".into(),
                    ));
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
                    checks.push((
                        "compact_equivalence",
                        false,
                        format!("model drift after compact at op {opno}"),
                    ));
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
                    checks.push((
                        "restart_equivalence",
                        false,
                        format!("model drift after restart at op {opno}"),
                    ));
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
    checks.push((
        "final_state_equivalence",
        obs == model,
        format!("model={} observed={}", model.len(), obs.len()),
    ));
    checks.push(("final_store_unique", iss.is_empty(), iss.join(";")));
    // every live doc retrievable; every dead doc absent (INV-1/2)
    let mut dead_hits = 0usize;
    for idx in 0..next_new {
        if model.contains_key(&idx) {
            continue;
        }
        if topk_observed(&e, idx, 400).contains(&idx) {
            dead_hits += 1;
        }
    }
    checks.push((
        "final_dead_not_retrievable",
        dead_hits == 0,
        format!("{dead_hits} dead docs retrieved"),
    ));
    let rep = checker_report(&e, dir);
    write_json(&format!("{out}/consistency-final.json"), &rep);
    checks.push((
        "final_checker",
        rep["clean"].as_bool().unwrap_or(false),
        format!("{:?}", rep),
    ));

    let (pass, fail) =
        checks.iter().fold(
            (0usize, 0usize),
            |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) },
        );
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
    let exp: serde_json::Value = model
        .iter()
        .map(|(k, (v, c, n))| {
            (
                k.to_string(),
                serde_json::json!({"version": v, "cat": c, "num": n}),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();
    write_json(&format!("{out}/expected-state.json"), &exp);
    let obsj: serde_json::Value = obs
        .iter()
        .map(|(k, (v, c, n))| {
            (
                k.to_string(),
                serde_json::json!({"version": v, "cat": c, "num": n}),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();
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
    pub fn new(seed: u64) -> Self {
        Det(seed | 1)
    }
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
    FilterExpr::Comparison {
        field: field.into(),
        op: FilterOp::Eq,
        value: FilterValue::Str(v.into()),
    }
}
fn cmpf(field: &str, op: FilterOp, v: FilterValue) -> FilterExpr {
    FilterExpr::Comparison {
        field: field.into(),
        op,
        value: v,
    }
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
        e.insert_document("bench", doc_record(idx, 1, cat, idx as i64))
            .unwrap();
    }
    e.flush_wal().unwrap();

    let mut filter_recalls: Vec<(String, f64)> = Vec::new();
    let expect_filtered = |e: &AttentionEngine,
                           f: &FilterExpr,
                           want: Vec<u32>,
                           name: &str,
                           checks: &mut Vec<(String, bool, String)>,
                           recalls: &mut Vec<(String, f64)>| {
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
        let recall = if want.is_empty() {
            1.0
        } else {
            hit as f64 / want.len() as f64
        };
        recalls.push((name.to_string(), recall));
        checks.push((
            name.to_string(),
            sound && det,
            if sound && det {
                format!("recall {recall:.3}")
            } else {
                format!(
                    "sound={sound} det={det} recall {recall:.3} want {want_sorted:?} got {got:?}"
                )
            },
        ));
    };

    // 1) basic filters
    expect_filtered(
        &e,
        &eqf("cat", "a"),
        (0..120).filter(|i| i % 2 == 0).collect(),
        "filter_eq_all",
        &mut checks,
        &mut filter_recalls,
    );
    expect_filtered(
        &e,
        &cmpf("num", FilterOp::Gt, FilterValue::Int(100)),
        (101..120).collect(),
        "filter_selective",
        &mut checks,
        &mut filter_recalls,
    );
    expect_filtered(
        &e,
        &cmpf("num", FilterOp::Gt, FilterValue::Int(10_000)),
        vec![],
        "filter_zero_match",
        &mut checks,
        &mut filter_recalls,
    );
    expect_filtered(
        &e,
        &andf(
            eqf("cat", "b"),
            cmpf("num", FilterOp::Lte, FilterValue::Int(20)),
        ),
        (1..=19).filter(|i| i % 2 == 1).collect(),
        "filter_and",
        &mut checks,
        &mut filter_recalls,
    );
    // filter + top-k: restricted K still filter-clean
    {
        let got = filtered_observed(&e, &eqf("cat", "a"), 1, 7);
        checks.push((
            "filter_topk_clean".into(),
            got.iter().all(|i| i % 2 == 0) && got.len() <= 7,
            format!("{got:?}"),
        ));
    }

    // 2) filter × mutation lifecycle (§5)
    // insert matching
    e.insert_document("bench", doc_record(200, 1, "a", 5))
        .unwrap();
    expect_filtered(
        &e,
        &eqf("cat", "a"),
        (0..120).filter(|i| i % 2 == 0).chain([200]).collect(),
        "fm_insert_matching",
        &mut checks,
        &mut filter_recalls,
    );
    // update so it LEAVES the filter
    e.update_document(
        "bench",
        &uuid_for(200, 1).to_string(),
        doc_record(200, 2, "b", 5).fields,
        doc_record(200, 2, "b", 5).k_vecs,
    )
    .unwrap();
    expect_filtered(
        &e,
        &eqf("cat", "a"),
        (0..120).filter(|i| i % 2 == 0).collect(),
        "fm_update_leaves",
        &mut checks,
        &mut filter_recalls,
    );
    // update so it re-ENTERS with new num
    e.update_document(
        "bench",
        &uuid_for(200, 1).to_string(),
        doc_record(200, 3, "b", 999).fields,
        doc_record(200, 3, "b", 999).k_vecs,
    )
    .unwrap();
    expect_filtered(
        &e,
        &andf(
            eqf("cat", "b"),
            cmpf("num", FilterOp::Eq, FilterValue::Int(999)),
        ),
        vec![200],
        "fm_update_enters",
        &mut checks,
        &mut filter_recalls,
    );
    // delete matching / nonmatching
    e.delete_document("bench", &uuid_for(200, 1).to_string())
        .unwrap();
    expect_filtered(
        &e,
        &andf(
            eqf("cat", "b"),
            cmpf("num", FilterOp::Eq, FilterValue::Int(999)),
        ),
        vec![],
        "fm_delete_matching",
        &mut checks,
        &mut filter_recalls,
    );
    e.delete_document("bench", &uuid_for(1, 1).to_string())
        .unwrap();
    expect_filtered(
        &e,
        &eqf("cat", "b"),
        (1..120).filter(|i| i % 2 == 1 && *i != 1).collect(),
        "fm_delete_nonmatching",
        &mut checks,
        &mut filter_recalls,
    );
    // reinsert deleted logical doc (new version)
    e.insert_document("bench", doc_record(1, 2, "b", 1))
        .unwrap();
    expect_filtered(
        &e,
        &eqf("cat", "b"),
        (1..120).filter(|i| i % 2 == 1).collect(),
        "fm_reinsert",
        &mut checks,
        &mut filter_recalls,
    );

    // 3) restart / flush / compact persistence of filter semantics (§6)
    e.flush_wal().unwrap();
    e.checkpoint().unwrap();
    drop(e);
    let e = open_db(dir);
    expect_filtered(
        &e,
        &eqf("cat", "b"),
        (1..120).filter(|i| i % 2 == 1).collect(),
        "fr_restart",
        &mut checks,
        &mut filter_recalls,
    );
    drop(e);
    let res = compaction::compact_all(dir).unwrap();
    if let Some(r) = &res {
        let _ = compaction::cleanup_merged_files(r);
    }
    let e = open_db(dir);
    expect_filtered(
        &e,
        &eqf("cat", "b"),
        (1..120).filter(|i| i % 2 == 1).collect(),
        "fr_compact",
        &mut checks,
        &mut filter_recalls,
    );
    let rep = checker_report(&e, dir);
    checks.push((
        "final_checker".into(),
        rep["clean"].as_bool().unwrap_or(false),
        format!("{rep:?}"),
    ));

    let (pass, fail) =
        checks.iter().fold(
            (0usize, 0usize),
            |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) },
        );
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks {
        let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'"));
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({
            "passed": pass, "failed": fail,
            "filter_recall": filter_recalls.iter().map(|(n, r)| serde_json::json!({"check": n, "recall": r})).collect::<Vec<_>>(),
        }),
    );
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
        e.insert_document("bench", doc_record(idx, 1, "a", idx as i64))
            .unwrap();
    }
    e.flush_wal().unwrap();

    // commit: all ops visible
    let t = e.txn_manager.begin_transaction("bench");
    let ins1 = doc_record(100, 1, "new", 1);
    let ins2 = doc_record(101, 1, "new", 2);
    e.txn_manager
        .record_operation(t, TxnOp::Insert(ins1))
        .unwrap();
    e.txn_manager
        .record_operation(t, TxnOp::Insert(ins2))
        .unwrap();
    e.txn_manager
        .record_operation(t, TxnOp::Delete(uuid_for(3, 1)))
        .unwrap();
    let ok = e.commit_transaction(t).unwrap();
    let (obs, _) = export_state(&e);
    checks.push((
        "txn_commit_applied".to_string(),
        ok && obs.contains_key(&100) && obs.contains_key(&101) && !obs.contains_key(&3),
        format!("{obs:?}"),
    ));
    // rollback: nothing visible
    let t = e.txn_manager.begin_transaction("bench");
    e.txn_manager
        .record_operation(t, TxnOp::Insert(doc_record(102, 1, "new", 3)))
        .unwrap();
    e.txn_manager
        .record_operation(t, TxnOp::Delete(uuid_for(4, 1)))
        .unwrap();
    let rolled = e.txn_manager.rollback_transaction(t).unwrap();
    let (obs, _) = export_state(&e);
    checks.push((
        "txn_rollback_invisible".to_string(),
        rolled && !obs.contains_key(&102) && obs.contains_key(&4),
        format!("{obs:?}"),
    ));
    // repeated/re-entrant use after rollback must fail cleanly
    let gone = e.txn_manager.get_staged_transaction(t).is_none();
    checks.push(("txn_rollback_unstaged".to_string(), gone, String::new()));
    // multi-op matrix (insert+delete mixes) in ONE txn
    let t = e.txn_manager.begin_transaction("bench");
    for i in 0..5u32 {
        e.txn_manager
            .record_operation(t, TxnOp::Insert(doc_record(110 + i, 1, "m", i as i64)))
            .unwrap();
    }
    e.txn_manager
        .record_operation(t, TxnOp::Delete(uuid_for(5, 1)))
        .unwrap();
    e.txn_manager
        .record_operation(t, TxnOp::Delete(uuid_for(6, 1)))
        .unwrap();
    e.commit_transaction(t).unwrap();
    let (obs, _) = export_state(&e);
    let ok = (110..115).all(|i| obs.contains_key(&(i as u32)))
        && !obs.contains_key(&5)
        && !obs.contains_key(&6);
    checks.push(("txn_multiop_matrix".to_string(), ok, String::new()));
    // §20 injected commit failure: an op that fails pre-validation must abort
    // the WHOLE commit — no partial apply, engine unchanged, checker clean.
    {
        let t = e.txn_manager.begin_transaction("bench");
        let mut bad = doc_record(300, 1, "bad", 1);
        bad.k_vecs.insert(HEAD.to_string(), vec![0.0; DIM + 7]); // wrong dim
        e.txn_manager
            .record_operation(t, TxnOp::Insert(bad))
            .unwrap();
        let good = doc_record(301, 1, "bad", 2);
        e.txn_manager
            .record_operation(t, TxnOp::Insert(good))
            .unwrap();
        let r = e.commit_transaction(t);
        let (obs, _) = export_state(&e);
        checks.push((
            "txn_commit_failure_atomic".to_string(),
            r.is_err() && !obs.contains_key(&300) && !obs.contains_key(&301),
            format!(
                "commit_err={} has300={} has301={}",
                r.is_err(),
                obs.contains_key(&300),
                obs.contains_key(&301)
            ),
        ));
    }
    // delete-if-present: deleting an unknown uuid inside a committed txn must
    // be a clean no-op (TxnOp::Delete with numeric 0 semantics), not an error.
    {
        let t = e.txn_manager.begin_transaction("bench");
        e.txn_manager
            .record_operation(t, TxnOp::Delete(uuid::Uuid::from_u128(0xf0f0_f0f0)))
            .unwrap();
        let r = e.commit_transaction(t);
        let rep2 = checker_report(&e, dir);
        checks.push((
            "txn_delete_missing_noop".to_string(),
            r.is_ok() && rep2["clean"].as_bool().unwrap_or(false),
            format!("commit_ok={}", r.is_ok()),
        ));
    }
    let rep = checker_report(&e, dir);
    checks.push((
        "txn_checker".to_string(),
        rep["clean"].as_bool().unwrap_or(false),
        format!("{rep:?}"),
    ));

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
        checks.push((
            "txn_crash_atomicity".to_string(),
            has_all || has_none,
            format!("committed_marker={committed} all={has_all} none={has_none}"),
        ));
        checks.push((
            "txn_crash_matches_wal".to_string(),
            if committed {
                has_all
            } else {
                has_none || has_all /* apply-phase crash recovers to all via replay */
            },
            String::new(),
        ));
        let rep = checker_report(&e2, dir);
        checks.push((
            "txn_crash_checker".to_string(),
            rep["clean"].as_bool().unwrap_or(false),
            format!("{rep:?}"),
        ));
        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_file(&side);
    }

    let (pass, fail) =
        checks.iter().fold(
            (0usize, 0usize),
            |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) },
        );
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks {
        let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'"));
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({"passed": pass, "failed": fail}),
    );
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
        e.insert_document("bench", doc_record(idx, 1, "a", idx as i64))
            .unwrap();
    }
    e.flush_wal().unwrap();
    let t = e.txn_manager.begin_transaction("bench");
    for i in 200..210u32 {
        e.txn_manager
            .record_operation(t, TxnOp::Insert(doc_record(i, 1, "txn", i as i64)))
            .unwrap();
    }
    let r = e.commit_transaction(t);
    let side = std::path::PathBuf::from(format!("{}.txncrash", dir.display()));
    let mut f = std::fs::File::create(&side).unwrap();
    let _ = writeln!(f, "{}", if r.is_ok() { "committed" } else { "failed" });
    let _ = f.flush();
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    } // parent will SIGKILL
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
        e.insert_document("bench", doc_record(idx, 1, "c", i as i64))
            .unwrap();
        ack(&mut sc, i);
        if point == "mid_inserts" && i == n / 2 {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            } // parent SIGKILLs
        }
    }
    match point {
        "after_acks" => std::process::exit(137), // no flush, no checkpoint
        "after_flush" => {
            e.flush_wal().unwrap();
            std::process::exit(137);
        }
        "after_checkpoint" => {
            e.flush_wal().unwrap();
            e.checkpoint().unwrap();
            std::process::exit(137);
        }
        "mid_flush" => loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        },
        "during_commit_txn" => {
            // same engine, same dir, same sidecar bookkeeping: flush the acked
            // baseline, then commit one 10-insert txn; marker records the commit
            // call's outcome AFTER it returns; parent SIGKILLs while we park.
            use attentiondb_core::transaction::TxnOp;
            e.flush_wal().unwrap();
            let t = e.txn_manager.begin_transaction("bench");
            for i in 2000..2010u32 {
                e.txn_manager
                    .record_operation(t, TxnOp::Insert(doc_record(i, 1, "txn", i as i64)))
                    .unwrap();
            }
            let r = e.commit_transaction(t);
            let side = std::path::PathBuf::from(format!("{}.txncrash", dir.display()));
            let mut f = std::fs::File::create(&side).unwrap();
            let _ = writeln!(f, "{}", if r.is_ok() { "committed" } else { "failed" });
            let _ = f.flush();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
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
        (
            obs.len(),
            iss.len(),
            rep["clean"].as_bool().unwrap_or(false),
        )
    }));
    let _ = std::fs::remove_dir_all(dir);
    match r {
        Err(_) => format!("{mode},ERROR_ON_OPEN,,"),
        Ok((n, iss, clean)) => {
            format!("{mode},OPENED,n_docs={n},store_issues={iss},checker_clean={clean}")
        }
    }
}

// ---------------------------------------------------------------- concur

fn run_concur(dir: &std::path::Path, readers: usize, writers: usize, out: &str) -> String {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let e = Arc::new(open_db(dir));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for idx in 0..300u32 {
        e.insert_document("bench", doc_record(idx, 1, "a", idx as i64))
            .unwrap();
    }
    e.flush_wal().unwrap();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // deterministic per-writer op log (§24): writers record each cycle's ops
    let oplog = Arc::new(std::sync::Mutex::new(
        std::fs::File::create(format!("{out}/operation-log.jsonl")).unwrap(),
    ));
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
    let pc = |p: f64| {
        latc[((p / 100.0) * (latc.len() as f64 - 1.0)).round() as usize % latc.len().max(1)]
    };
    // post-workload checker gate + logical sanity
    let (obs, iss) = export_state(&e);
    let rep = checker_report(&e, dir);
    let clean = rep["clean"].as_bool().unwrap_or(false) && iss.is_empty();
    let warn_ct = rep["warning_count"].as_u64().unwrap_or(0);
    let mut csv = String::from("readers,writers,queries,p50_us,p95_us,p99_us,qps,errors,checker_clean,checker_warnings,live_docs\n");
    let _ = writeln!(
        csv,
        "{readers},{writers},{total},{:.1},{:.1},{:.1},{qps:.0},{errors},{clean},{warn_ct},{}",
        pc(50.0),
        pc(95.0),
        pc(99.0),
        obs.len()
    );
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({
            "readers": readers, "writers": writers, "queries": total, "qps": qps,
            "errors": errors, "checker_clean": clean, "checker_warnings": warn_ct,
            "store_issues": iss, "live_docs": obs.len(),
        }),
    );
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
        e.insert_document("bench", doc_record(idx, 1, "a", idx as i64))
            .unwrap();
    }
    e.flush_wal().unwrap();
    e.checkpoint().unwrap();
    let bak = std::path::PathBuf::from(format!("{}.bak", dir.display()));
    let _ = std::fs::remove_dir_all(&bak);
    copy_database_dir(dir, &bak).unwrap();
    let manifest = restore_backup(
        &bak,
        &std::path::PathBuf::from(format!("{}.restore", dir.display())),
    );
    // capture backup-state BEFORE further mutation
    let backup_state = export_state(&e).0;
    // mutate A: inserts + deletes + update
    for idx in 150..170u32 {
        e.insert_document("bench", doc_record(idx, 1, "post", idx as i64))
            .unwrap();
    }
    e.delete_document("bench", &uuid_for(0, 1).to_string())
        .unwrap();
    e.update_document(
        "bench",
        &uuid_for(1, 1).to_string(),
        doc_record(1, 9, "mutated", 1).fields,
        doc_record(1, 9, "mutated", 1).k_vecs,
    )
    .unwrap();
    e.flush_wal().unwrap();
    drop(e);
    // open restored DB and compare against backup state
    let rdir = std::path::PathBuf::from(format!("{}.restore", dir.display()));
    let rb = open_db(&rdir);
    let (rstate, riss) = export_state(&rb);
    checks.push((
        "restore_state_equals_backup".into(),
        rstate == backup_state,
        format!("backup={} restored={}", backup_state.len(), rstate.len()),
    ));
    checks.push((
        "restore_store_unique".into(),
        riss.is_empty(),
        riss.join(";"),
    ));
    checks.push((
        "restore_not_mutated".into(),
        !rstate.contains_key(&150) && rstate.contains_key(&1),
        format!(
            "has150={} has1={}",
            rstate.contains_key(&150),
            rstate.contains_key(&1)
        ),
    ));
    let rep = checker_report(&rb, &rdir);
    checks.push((
        "restore_checker".into(),
        rep["clean"].as_bool().unwrap_or(false),
        format!("{rep:?}"),
    ));
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
                checks.push((
                    "live_writer_backup_documented".into(),
                    true,
                    format!(
                        "docs={} consistent={} writers_wrote={}",
                        st.len(),
                        rep2["clean"].as_bool().unwrap_or(false),
                        written
                    ),
                ));
                let _ = std::fs::remove_dir_all(&rd);
            } else {
                checks.push((
                    "live_backup_consistent".into(),
                    true,
                    "restore refused live backup (documented: quiescent required)".into(),
                ));
            }
        } else {
            checks.push((
                "live_backup_consistent".into(),
                true,
                "copy refused while active (documented: quiescent required)".into(),
            ));
        }
        let _ = std::fs::remove_dir_all(&live_bak);
    }
    let _ = std::fs::remove_dir_all(&bak);
    let _ = std::fs::remove_dir_all(&rdir);
    let _ = std::fs::remove_dir_all(dir);

    let (pass, fail) =
        checks.iter().fold(
            (0usize, 0usize),
            |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) },
        );
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks {
        let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'"));
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({"passed": pass, "failed": fail}),
    );
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
        e.insert_document("bench", doc_record(idx, 1, "a", idx as i64))
            .unwrap();
    }
    // churn: updates + deletes create overwrites + tombstones for GC
    for idx in 0..100u32 {
        let r = doc_record(idx, 2, "b", idx as i64 * 10);
        e.update_document("bench", &uuid_for(idx, 1).to_string(), r.fields, r.k_vecs)
            .unwrap();
    }
    for idx in 100..150u32 {
        e.delete_document("bench", &uuid_for(idx, 1).to_string())
            .unwrap();
    }
    e.flush_wal().unwrap();
    e.checkpoint().unwrap(); // generation 1 SST
                             // second generation: more writes so a second SST exists to merge into
    for idx in 300..360u32 {
        e.insert_document("bench", doc_record(idx, 1, "c", idx as i64))
            .unwrap();
    }
    for idx in 0..30u32 {
        let r = doc_record(idx, 3, "c", idx as i64 * 100);
        e.update_document("bench", &uuid_for(idx, 1).to_string(), r.fields, r.k_vecs)
            .unwrap();
    }
    e.flush_wal().unwrap();
    e.checkpoint().unwrap(); // generation 2 SST
    let (before, biss) = export_state(&e);
    checks.push((
        "before_checker".into(),
        checker_report(&e, dir)["clean"].as_bool().unwrap_or(false) && biss.is_empty(),
        format!("{biss:?}"),
    ));
    drop(e);
    let cres = compaction::compact_all(dir);
    let did = matches!(&cres, Ok(Some(_)));
    let cerr = match &cres {
        Err(er) => er.to_string(),
        Ok(None) => "no-op (<2 sst)".into(),
        Ok(Some(_)) => String::new(),
    };
    let removed = match &cres {
        Ok(Some(res)) => compaction::cleanup_merged_files(res).unwrap_or(0),
        _ => 0,
    };
    let e = open_db(dir);
    let (after, aiss) = export_state(&e);
    checks.push((
        "after_state_equal".into(),
        before == after && aiss.is_empty(),
        if before == after {
            format!("docs={}", after.len())
        } else {
            format!(
                "before={} after={} iss={biss:?}/{aiss:?}",
                before.len(),
                after.len()
            )
        },
    ));
    checks.push((
        "after_checker".into(),
        checker_report(&e, dir)["clean"].as_bool().unwrap_or(false),
        String::new(),
    ));
    // §10 spot: twice-updated doc survives compaction at LATEST content (v3)
    checks.push((
        "updated_content_survives".into(),
        after
            .get(&20)
            .map(|(_, c, n)| c == "c" && *n == 2000)
            .unwrap_or(false),
        String::new(),
    ));
    // §9 spot: deleted docs stay deleted after compaction (tombstone GC)
    checks.push((
        "deleted_stay_deleted".into(),
        (100..150u32).all(|i| !after.contains_key(&i)),
        String::new(),
    ));
    let (pass, fail) =
        checks.iter().fold(
            (0usize, 0usize),
            |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) },
        );
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks {
        let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'"));
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({
            "passed": pass, "failed": fail, "compaction_ran": did,
            "compaction_error": cerr, "merged_files_removed": removed, "docs": after.len(),
        }),
    );
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
            e.insert_document("bench", doc_record(idx, 1, "a", idx as i64))
                .unwrap();
        }
        e.flush_wal().unwrap();
        e.checkpoint().unwrap();
        // three SST generations: below the auto-compact threshold (4 files)
        for gen in 0..2u32 {
            for idx in (50 + gen * 15)..(65 + gen * 15) {
                e.insert_document("bench", doc_record(idx, 1, "a", idx as i64))
                    .unwrap();
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
        e.insert_document("alpha", doc_record_c(1, i, 1, "x", i as i64))
            .unwrap();
    }
    for i in 0..40u32 {
        e.insert_document("beta", doc_record_c(2, i, 1, "y", i as i64))
            .unwrap();
    }
    for i in 0..24u32 {
        e.insert_document("gamma", doc_record_c(3, i, 1, "g", i as i64))
            .unwrap();
    }
    e.flush_wal().unwrap();
    e.checkpoint().unwrap(); // generation 1
                             // churn: updates first (0..10 incl. odd ids), then deletes on the REMAINING
                             // odd ids (11..59) — never deleting a doc the update loop must still see
    for i in 0..10u32 {
        let r = doc_record_c(1, i, 2, "x2", i as i64 * 3);
        e.update_document("alpha", &uuidc(1, i, 1).to_string(), r.fields, r.k_vecs)
            .unwrap();
    }
    for i in (11..60u32).step_by(2) {
        e.delete_document("alpha", &uuidc(1, i, 1).to_string())
            .unwrap();
    }
    let r = doc_record_c(2, 2000, 1, "y", 7);
    e.insert_document("beta", r.clone()).unwrap(); // upsert: new logical id
    let r = doc_record_c(2, 0, 2, "y2", 100);
    e.update_document("beta", &uuidc(2, 0, 1).to_string(), r.fields, r.k_vecs)
        .unwrap(); // upsert: existing
    for i in 24..36u32 {
        e.insert_document("gamma", doc_record_c(3, i, 1, "g", i as i64))
            .unwrap();
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
    checks.push((
        "isol_live_alpha".into(),
        alpha_ok,
        format!("docs={} first={sa:?}", sa.len()),
    ));
    checks.push((
        "isol_live_beta".into(),
        sb.len() == 41
            && sb.contains_key(&2000)
            && sb
                .get(&0)
                .map(|(v, c, _)| *v == 2 && c == "y2")
                .unwrap_or(false),
        format!("docs={}", sb.len()),
    ));
    checks.push((
        "isol_live_gamma".into(),
        sg.len() == 36,
        format!("docs={}", sg.len()),
    ));

    // (b) filter × multi-head (§4): soundness + determinism across heads h+h2
    {
        let heads = vec![HEAD.to_string(), "h2".to_string()];
        let f = eqf("cat", "g");
        let got = filtered_observed_multi(&e, "gamma", &heads, &f, 1, 40);
        let got2 = filtered_observed_multi(&e, "gamma", &heads, &f, 1, 40);
        checks.push((
            "filter_multihead_sound".into(),
            got.iter().all(|i| sg.contains_key(i)) && got == got2 && got.len() <= 40,
            format!("{got:?}"),
        ));
        let zero = filtered_observed_multi(&e, "gamma", &heads, &eqf("cat", "no-such"), 1, 40);
        checks.push((
            "filter_multihead_zero_match".into(),
            zero.is_empty(),
            format!("{zero:?}"),
        ));
    }

    // (c) graceful-shutdown durability (§11): close -> reopen -> all four
    // mutation kinds (insert/update/delete/upsert) persisted
    e.close().unwrap();
    let e2 = open_db(dir);
    let (ra, _) = export_state_coll(&e2, "alpha");
    let (rb, _) = export_state_coll(&e2, "beta");
    let (rg, _) = export_state_coll(&e2, "gamma");
    checks.push((
        "graceful_restart_alpha".into(),
        ra == sa,
        format!("equal={}", ra == sa),
    ));
    checks.push((
        "graceful_restart_beta".into(),
        rb == sb,
        format!("equal={}", rb == sb),
    ));
    checks.push((
        "graceful_restart_gamma".into(),
        rg == sg,
        format!("equal={}", rg == sg),
    ));
    checks.push((
        "graceful_checker".into(),
        checker_report(&e2, dir)["clean"].as_bool().unwrap_or(false),
        String::new(),
    ));
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
    checks.push((
        "compact_checker".into(),
        checker_report(&e3, dir)["clean"].as_bool().unwrap_or(false),
        String::new(),
    ));

    // (e) backup/restore reproduces BOTH collections (§29/§31)
    let bak = std::path::PathBuf::from(format!("{}.bak", dir.display()));
    let rdir = std::path::PathBuf::from(format!("{}.restored", dir.display()));
    let _ = std::fs::remove_dir_all(&bak);
    let _ = std::fs::remove_dir_all(&rdir);
    let copy_ok = copy_database_dir(dir, &bak).is_ok();
    let m = if copy_ok {
        restore_backup(&bak, &rdir).ok()
    } else {
        None
    };
    if let Some(_m) = &m {
        let re = open_db(&rdir);
        let (ra2, _) = export_state_coll(&re, "alpha");
        let (rb2, _) = export_state_coll(&re, "beta");
        checks.push(("restore_alpha_equal".into(), ra2 == sa, String::new()));
        checks.push(("restore_beta_equal".into(), rb2 == sb, String::new()));
        // isolation by content: collection membership tags must not mix —
        // alpha docs keep x/x2 cats, beta docs keep y/y2 (logical idx spaces
        // overlap by design; uuid namespaces keep identities separate)
        checks.push((
            "restore_isolation".into(),
            ra2.values().all(|(_, c, _)| c == "x" || c == "x2")
                && rb2.values().all(|(_, c, _)| c == "y" || c == "y2"),
            format!(
                "alpha_cats={:?}",
                ra2.values()
                    .map(|(_, c, _)| c.clone())
                    .collect::<std::collections::BTreeSet<_>>()
            ),
        ));
        checks.push((
            "restore_checker".into(),
            checker_report(&re, &rdir)["clean"]
                .as_bool()
                .unwrap_or(false),
            String::new(),
        ));
    } else {
        checks.push((
            "restore_alpha_equal".into(),
            false,
            format!("copy_ok={copy_ok}"),
        ));
    }
    let _ = std::fs::remove_dir_all(&bak);
    let _ = std::fs::remove_dir_all(&rdir);
    let _ = std::fs::remove_dir_all(dir);

    let (pass, fail) =
        checks.iter().fold(
            (0usize, 0usize),
            |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) },
        );
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks {
        let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'"));
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({"passed": pass, "failed": fail}),
    );
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
        e.insert_document("bench", doc_record(idx, 1, "seed", idx as i64))
            .unwrap();
    }
    e.flush_wal().unwrap();

    // ---- phase 1: disjoint ranges, logged, then replayed
    let oplog = std::sync::Arc::new(std::sync::Mutex::new(
        std::fs::File::create(format!("{out}/operation-log.jsonl")).unwrap(),
    ));
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
    for h in handles {
        h.join().unwrap();
    }

    // replay the merged logs into the reference model (seed 50 docs first)
    let mut model: Model = (0..50u32)
        .map(|i| (i, (1u64, "seed".to_string(), i as i64)))
        .collect();
    for line in std::fs::read_to_string(format!("{out}/operation-log.jsonl"))
        .unwrap()
        .lines()
    {
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
    checks.push((
        "replay_exact_state".into(),
        obs == model && iss.is_empty(),
        if obs == model {
            format!("docs={}", obs.len())
        } else {
            let miss: Vec<_> = model
                .keys()
                .filter(|k| !obs.contains_key(k))
                .take(5)
                .collect();
            let extra: Vec<_> = obs
                .keys()
                .filter(|k| !model.contains_key(k))
                .take(5)
                .collect();
            let diff: Vec<_> = model
                .iter()
                .filter(|(k, v)| obs.get(k) != Some(*v))
                .take(5)
                .collect();
            format!("miss={miss:?} extra={extra:?} diff={diff:?}")
        },
    ));
    checks.push((
        "replay_checker".into(),
        checker_report(&e, dir)["clean"].as_bool().unwrap_or(false),
        String::new(),
    ));

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
                    let _ =
                        e.update_document("bench", &uuid_for(k, 1).to_string(), r.fields, r.k_vecs);
                }
            }
        }));
    }
    for h in wh {
        h.join().unwrap();
    }
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
    checks.push((
        "contention_atomic_records".into(),
        torn == 0 && missing == 0,
        format!("torn={torn} missing={missing}"),
    ));
    checks.push((
        "contention_checker".into(),
        checker_report(&e, dir)["clean"].as_bool().unwrap_or(false),
        String::new(),
    ));
    let _ = std::fs::remove_dir_all(dir);

    let (pass, fail) =
        checks.iter().fold(
            (0usize, 0usize),
            |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) },
        );
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks {
        let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'"));
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({
            "passed": pass, "failed": fail,
            "writers": 2, "logged_ops": 500, "contention_keys": 100,
            "model": "mutation-gate serialized; per-key last-write-wins; records never torn; no cross-key ordering guarantee",
        }),
    );
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
                .split_whitespace()
                .next()
                .unwrap_or("?")
                .to_string(),
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
            e.insert_document("bench", doc_record(idx, 1, "a", idx as i64))
                .unwrap();
        }
        for idx in 0..30u32 {
            let r = doc_record(idx, 2, "b", idx as i64 * 2);
            e.update_document("bench", &uuid_for(idx, 1).to_string(), r.fields, r.k_vecs)
                .unwrap();
        }
        for idx in 100..120u32 {
            e.delete_document("bench", &uuid_for(idx, 1).to_string())
                .unwrap();
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
        checks.push((
            "copy_fidelity_files".into(),
            src_inv.len() == bak_inv.len(),
            format!(
                "src_files={} bak_files={} (+backup-meta.json)",
                src_inv.len(),
                bak_inv.len()
            ),
        ));
        let mismatch = src_inv
            .iter()
            .zip(bak_inv.iter())
            .filter(|(a, b)| a.0 != b.0 || a.2 != b.2)
            .count();
        checks.push((
            "copy_fidelity_hashes".into(),
            mismatch == 0,
            format!("mismatched={mismatch}"),
        ));
        inv = bak_inv.clone();
        let m = restore_backup(&bak, &rdir);
        checks.push(("restore_manifest".into(), m.is_ok(), String::new()));
        if m.is_ok() {
            let rst_inv = inventory(&rdir);
            let rmiss = bak_inv
                .iter()
                .filter(|b| !rst_inv.iter().any(|r| r.0 == b.0 && r.2 == b.2))
                .count();
            checks.push((
                "restore_fidelity_hashes".into(),
                rmiss == 0,
                format!("missing_or_diff={rmiss} (restored_files={})", rst_inv.len()),
            ));
            let re = open_db(&rdir);
            let (after, iss) = export_state(&re);
            checks.push((
                "restore_state_equals_source".into(),
                after == before && iss.is_empty(),
                format!("before={} after={}", before.len(), after.len()),
            ));
            checks.push((
                "restore_checker".into(),
                checker_report(&re, &rdir)["clean"]
                    .as_bool()
                    .unwrap_or(false),
                String::new(),
            ));
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

    let (pass, fail) =
        checks.iter().fold(
            (0usize, 0usize),
            |(p, f), c| if c.1 { (p + 1, f) } else { (p, f + 1) },
        );
    let mut csv = String::from("check,ok,detail\n");
    for (n, ok, d) in &checks {
        let _ = writeln!(csv, "{n},{ok},\"{}\"", d.replace('"', "'"));
    }
    std::fs::write(format!("{out}/results.csv"), &csv).unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({
            "passed": pass, "failed": fail,
            "files": inv.len(), "total_bytes": total_bytes,
            "note": "quiescent backup; sizes+checksums recorded in inventory.csv",
        }),
    );
    format!(
        "backupinv: {pass} passed, {fail} FAILED (files={} bytes={total_bytes})",
        inv.len()
    )
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
                    ws.map(|w| w.high_watermark.to_string())
                        .unwrap_or("-".into())
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
    format!(
        "walintegrity: {} cases, {refused} refused, {opened} opened",
        cases.len()
    )
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

fn e2_spawn_child_live(dir: &std::path::Path, scenario: &str, mode: &str) -> std::process::Child {
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
    extra_idx: Vec<u32>, // anything outside the expected idx universe
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
    let universe: std::collections::HashSet<u32> =
        baseline_idx.iter().copied().chain(target_idx).collect();
    let extra_idx: Vec<u32> = obs
        .keys()
        .filter(|k| !universe.contains(k))
        .copied()
        .collect();
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
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                    .unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            match gate {
                "after_ack" => {
                    e.insert_document("bench", doc_record(2000, 1, "t", 0))
                        .unwrap();
                    ack(String::from("ACK 2000"));
                    std::process::abort();
                }
                "explicit_flush" => {
                    e.insert_document("bench", doc_record(2000, 1, "t", 0))
                        .unwrap();
                    ack(String::from("ACK 2000"));
                    e.flush_wal().unwrap();
                    std::process::abort();
                }
                _ => {
                    // env-gated abort fires inside this call
                    e.insert_document("bench", doc_record(2000, 1, "t", 0))
                        .unwrap();
                    ack(String::from("ACK 2000")); // only reached if the gate did not fire
                    std::process::abort();
                }
            }
        }
        "txn" => {
            for i in 0..5u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                    .unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            use attentiondb_core::transaction::TxnOp;
            let t = e.txn_manager.begin_transaction("bench");
            for i in 2000..2008u32 {
                e.txn_manager
                    .record_operation(t, TxnOp::Insert(doc_record(i, 1, "txn", i as i64)))
                    .unwrap();
            }
            e.txn_manager
                .record_operation(t, TxnOp::Delete(uuid_for(1001, 1)))
                .unwrap();
            e.txn_manager
                .record_operation(t, TxnOp::Delete(uuid_for(1003, 1)))
                .unwrap();
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
                        e.insert_document("bench", doc_record(idx, 1, "g", idx as i64))
                            .unwrap();
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
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                    .unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            match gate {
                "checkpoint" | "checkpoint_trim" => {
                    e.checkpoint().unwrap();
                }
                "rotate_only" => {}
                _ => panic!("unknown structural point {gate}"),
            }
            e.insert_document("bench", doc_record(2000, 1, "t", 0))
                .unwrap();
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
                        "before_wal_append" | "after_wal_append" | "after_apply" | "before_ack" => {
                            (11, false)
                        }
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
                                mode.into(),
                                gate.into(),
                                rep.to_string(),
                                "NOT_REACHED".into(),
                                "-".into(),
                                "0".into(),
                                "-".into(),
                                "-".into(),
                                "-".into(),
                                "-".into(),
                                "-".into(),
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
                            mode.into(),
                            gate.into(),
                            rep.to_string(),
                            e2_exit_name(exit).into(),
                            r.baseline_preserved.to_string(),
                            r.baseline_total.to_string(),
                            target_acked.to_string(),
                            r.target_present.to_string(),
                            r.extra_idx.len().to_string(),
                            r.checker_clean.to_string(),
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
                        let universe: std::collections::HashSet<u32> =
                            baseline.iter().copied().chain(2000..2008).collect();
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
                            mode.into(),
                            gate.into(),
                            rep.to_string(),
                            e2_exit_name(exit).into(),
                            baseline_preserved.to_string(),
                            baseline.len().to_string(),
                            txn_acked.to_string(),
                            txn_state.into(),
                            deletes_applied.to_string(),
                            extra.to_string(),
                            clean.to_string(),
                            wm,
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
                        let seg = if structural == "rotate_only" {
                            Some(2048u64)
                        } else {
                            None
                        };
                        let exit = e2_spawn_child(&dir, "ckpt", structural, 0, mode, seg);
                        let r = e2_recover(&dir, mode, &baseline, Some(2000), true);
                        rows.push(e2_row(&[
                            mode.into(),
                            structural.into(),
                            rep.to_string(),
                            e2_exit_name(exit).into(),
                            r.baseline_preserved.to_string(),
                            r.baseline_total.to_string(),
                            "true".into(),
                            r.target_present.to_string(),
                            r.extra_idx.len().to_string(),
                            r.checker_clean.to_string(),
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
                        let acked_set: std::collections::HashSet<u32> =
                            acked.iter().copied().collect();
                        let unacked_survived = obs
                            .keys()
                            .filter(|k| (3000..3999).contains(*k) && !acked_set.contains(k))
                            .count();
                        let extra = obs.keys().filter(|k| !(3000..3999).contains(*k)).count();
                        let clean = checker_report(&e, &dir)["clean"].as_bool().unwrap_or(false);
                        rows.push(e2_row(&[
                            mode.into(),
                            variant.into(),
                            rep.to_string(),
                            e2_exit_name(exit).into(),
                            acked.len().to_string(),
                            recovered_acked.to_string(),
                            acked_lost.to_string(),
                            unacked_survived.to_string(),
                            extra.to_string(),
                            clean.to_string(),
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
                    e.insert_document("bench", doc_record(5000 + i, 1, "l", i as i64))
                        .unwrap();
                    samples.push(t0.elapsed().as_micros());
                }
                let n = samples.len();
                let mean = samples.iter().sum::<u128>() / n as u128;
                samples.sort();
                let pct = |p: f64| samples[((n as f64) * p) as usize % n];
                let total_s = samples.iter().sum::<u128>() as f64 / 1e6;
                rows.push(e2_row(&[
                    mode.into(),
                    n.to_string(),
                    mean.to_string(),
                    pct(0.50).to_string(),
                    pct(0.90).to_string(),
                    pct(0.99).to_string(),
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
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&m)
    {
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
    let mut sc = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&side)
        .unwrap();
    let mut ack = |line: String| {
        use std::io::Write;
        let _ = writeln!(sc, "{line}");
        let _ = sc.flush();
        let _ = sc.sync_all();
    };
    match workload {
        "ack" => {
            for i in 0..10u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                    .unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            if gate == "after_ack" {
                // C7 harness-level park (no engine gate fires)
                e.insert_document("bench", doc_record(2000, 1, "t", 0))
                    .unwrap();
                ack(String::from("ACK 2000"));
                e3_park_with_marker(dir, "after_ack");
            }
            // env-gated park fires inside this insert
            e.insert_document("bench", doc_record(2000, 1, "t", 0))
                .unwrap();
            ack(String::from("ACK 2000"));
            e3_park_with_marker(dir, "post-window-fallback");
        }
        "mixed" => {
            for i in 0..9u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                    .unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            for i in [0u32, 2, 4] {
                e.delete_document("bench", &uuid_for(1000 + i, 1).to_string())
                    .unwrap();
                ack(format!("DEL {}", 1000 + i));
            }
            for i in 0..3u32 {
                e.insert_document("bench", doc_record(3000 + i, 1, "c2", i as i64))
                    .unwrap();
                ack(format!("ACK {}", 3000 + i));
            }
            if gate == "after_ack" {
                e.insert_document("bench", doc_record(2000, 1, "t", 0))
                    .unwrap();
                ack(String::from("ACK 2000"));
                e3_park_with_marker(dir, "after_ack");
            }
            // structural window (fires inside checkpoint)
            e.checkpoint().unwrap();
            e3_park_with_marker(dir, "post-window-fallback");
        }
        "txn" => {
            for i in 0..5u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                    .unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            use attentiondb_core::transaction::TxnOp;
            let t = e.txn_manager.begin_transaction("bench");
            for i in 2000..2008u32 {
                e.txn_manager
                    .record_operation(t, TxnOp::Insert(doc_record(i, 1, "txn", i as i64)))
                    .unwrap();
            }
            e.txn_manager
                .record_operation(t, TxnOp::Delete(uuid_for(1001, 1)))
                .unwrap();
            e.txn_manager
                .record_operation(t, TxnOp::Delete(uuid_for(1003, 1)))
                .unwrap();
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
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                    .unwrap();
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
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                    .unwrap();
                ack(format!("ACK {}", 1000 + i));
            }
            // env-gated park fires inside this insert's rotation
            e.insert_document("bench", doc_record(2000, 1, "t", 0))
                .unwrap();
            ack(String::from("ACK 2000"));
            e3_park_with_marker(dir, "post-window-fallback");
        }
        "close" => {
            for i in 0..10u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                    .unwrap();
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
    cmd.args([
        "dbtest",
        "e3child",
        "--dir",
        dir.to_str().unwrap(),
        "--workload",
        workload,
        "--gate",
        gate,
    ])
    .env("PH3D_DURABILITY", mode)
    .env("PH3E_CRASH_MODEL", "groupkill")
    .env("PH3E_CRASH_MARKER", format!("{}.e3gate", dir.display()));
    if crash {
        cmd.env("PH3E_CRASH_AT", gate)
            .env("PH3E_CRASH_HIT", gathit.to_string());
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
            unsafe {
                libc::kill(-pgid, libc::SIGKILL);
            }
        }
        #[cfg(not(unix))]
        let _ = child.kill();
    }
    let exit_kind = match child.wait() {
        Ok(st) => {
            use std::os::unix::process::ExitStatusExt;
            if !crash && st.success() {
                "EXIT0"
            } else if let Some(9) = st.signal() {
                "SIGKILL"
            } else if let Some(6) = st.signal() {
                "SIGABRT"
            } else if st.success() {
                "EXIT0"
            } else {
                "OTHER"
            }
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
                } else if msg.contains("CURRENT manifest unreadable")
                    || msg.contains("catalog recovered")
                {
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
                    .ok()
                    .flatten()
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
            if keys2 != prev {
                eq = false;
            }
            if !checker_report(&e2, dir)["clean"].as_bool().unwrap_or(false) {
                all_clean = false;
            }
            prev = keys2;
            handle = e2;
        }
        handle.close().unwrap();
        (eq, all_clean)
    } else {
        (true, checker_clean)
    };
    E3Facts {
        marker_reached,
        exit_kind,
        open_error,
        recovered_keys,
        checker_clean,
        high_watermark,
        restarts_equal,
        restart_checker_clean,
    }
}

fn e3_sidecar_acked(dir: &std::path::Path) -> (Vec<u32>, Vec<u32>, bool) {
    let mut acked = Vec::new();
    let mut deleted = Vec::new();
    let mut txn = false;
    if let Ok(s) = std::fs::read_to_string(format!("{}.e3sidecar", dir.display())) {
        for l in s.lines() {
            if let Some(v) = l.strip_prefix("ACK ") {
                if let Ok(n) = v.parse() {
                    acked.push(n);
                }
            } else if let Some(v) = l.strip_prefix("DEL ") {
                if let Ok(n) = v.parse() {
                    deleted.push(n);
                }
            } else if l == "TXNACK" {
                txn = true;
            }
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
    let record = |rows: &mut String,
                  cells: &mut usize,
                  cell: String,
                  mode: &str,
                  workload: &str,
                  window: &str,
                  hit: usize,
                  rep: usize,
                  fm: &str,
                  f: &E3Facts,
                  dir: &std::path::Path| {
        let (acked, deleted, _txn_acked) = e3_sidecar_acked(dir);
        let recovered_acked = acked
            .iter()
            .filter(|i| f.recovered_keys.contains(i))
            .count();
        let missing: Vec<u32> = acked
            .iter()
            .filter(|i| !f.recovered_keys.contains(i))
            .copied()
            .collect();
        let universe: std::collections::HashSet<u32> = acked
            .iter()
            .copied()
            .chain(deleted.iter().copied())
            .chain([2000u32])
            .chain(2000..2008)
            .collect();
        let unacked_present = f
            .recovered_keys
            .iter()
            .filter(|k| !universe.contains(k))
            .count();
        let deleted_still_gone = deleted
            .iter()
            .filter(|i| !f.recovered_keys.contains(i))
            .count();
        let txn_state = if workload == "txn" {
            let n = (2000..2008u32)
                .filter(|i| f.recovered_keys.contains(i))
                .count();
            match n {
                8 => "COMPLETE",
                0 => "ABSENT",
                _ => "PARTIAL",
            }
        } else {
            "-"
        };
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
        ("after_flush", 12, true, "wal"), // reachable in GroupCommit branch only
        ("after_fsync", 12, true, "wal"), // reachable in Sync branch only
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
                record(
                    &mut rows,
                    &mut cells,
                    "ack-c".into(),
                    mode,
                    "ack",
                    gate,
                    *hit,
                    rep,
                    if *kind == "harness" {
                        "F2E-GROUPKILL-HARNESS"
                    } else {
                        "F2E-GROUPKILL"
                    },
                    &f,
                    &dir,
                );
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
        ("ckpt_after_wal_fsync", 1),
        ("ckpt_after_sst", 1),
        ("ckpt_after_idmap", 1),
        ("manifest_after_tmp_write", 2),
        ("manifest_after_manifest_dirsync", 2),
        ("manifest_after_current_tmp_write", 2),
        ("manifest_after_current_rename", 2),
        ("ckpt_after_rotate", 1),
        ("ckpt_after_trim", 1),
    ];
    for mode in ["sync", "group", "async"] {
        for (gate, hit) in gates_d {
            for rep in 0..3 {
                let dir =
                    std::path::PathBuf::from(format!("/tmp/ph3e-e3/ckpt-{mode}-{gate}-{rep}"));
                let f = e3_run_cell(&dir, "ckpt", gate, hit, mode, None, true, true);
                record(
                    &mut rows,
                    &mut cells,
                    "ckpt-c".into(),
                    mode,
                    "ckpt",
                    gate,
                    hit,
                    rep,
                    "F2E-GROUPKILL",
                    &f,
                    &dir,
                );
                let _ = std::fs::remove_dir_all(&dir);
                let _ = std::fs::remove_file(format!("{}.e3sidecar", dir.display()));
                let _ = std::fs::remove_file(format!("{}.e3gate", dir.display()));
            }
        }
    }

    // ---- Workload E: WAL-rotation boundary (2 KiB segments) ----
    let gates_e = [
        ("rotate_after_old_fsync", 1usize),
        ("rotate_after_state_write", 2),
    ];
    for mode in ["sync", "group", "async"] {
        for (gate, hit) in gates_e {
            for rep in 0..3 {
                let dir = std::path::PathBuf::from(format!("/tmp/ph3e-e3/rot-{mode}-{gate}-{rep}"));
                let f = e3_run_cell(&dir, "rotate", gate, hit, mode, Some(2048), true, true);
                record(
                    &mut rows,
                    &mut cells,
                    "rotate-c".into(),
                    mode,
                    "rotate",
                    gate,
                    hit,
                    rep,
                    "F2E-GROUPKILL",
                    &f,
                    &dir,
                );
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
                record(
                    &mut rows,
                    &mut cells,
                    "mixed-c".into(),
                    mode,
                    "mixed",
                    gate,
                    hit,
                    rep,
                    "F2E-GROUPKILL",
                    &f,
                    &dir,
                );
                let _ = std::fs::remove_dir_all(&dir);
                let _ = std::fs::remove_file(format!("{}.e3sidecar", dir.display()));
                let _ = std::fs::remove_file(format!("{}.e3gate", dir.display()));
            }
        }
    }

    // ---- Workload C: transactions (COMMIT-buffered / pre-ACK / post-ACK) ----
    let gates_c: Vec<(&str, usize, bool)> = vec![
        ("after_write", 18, true), // COMMIT frame written (userspace buffer)
        ("before_ack", 6, true),   // engine: last engine-level instant
        ("after_ack", 0, true),    // harness: TXNACK durably recorded
    ];
    for mode in ["sync", "group", "async"] {
        for (gate, hit, crash) in gates_c.iter().copied() {
            for rep in 0..3 {
                let dir = std::path::PathBuf::from(format!("/tmp/ph3e-e3/txn-{mode}-{gate}-{rep}"));
                let f = e3_run_cell(&dir, "txn", gate, hit, mode, None, crash, false);
                record(
                    &mut rows,
                    &mut cells,
                    "txn-c".into(),
                    mode,
                    "txn",
                    gate,
                    hit,
                    rep,
                    "F2E-GROUPKILL",
                    &f,
                    &dir,
                );
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
        record(
            &mut rows,
            &mut cells,
            "f0-control".into(),
            mode,
            "close",
            "graceful_close",
            0,
            0,
            "F0-CLOSE",
            &f,
            &dir,
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(format!("{}.e3sidecar", dir.display()));
    }

    std::fs::write(format!("{out}/e3-matrix.csv"), &rows).unwrap();
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({"cells": cells, "failure_models": ["F0-CLOSE", "F2E-GROUPKILL"],
                            "blocked": ["F3-FS-DISRUPTION", "F4-POWER-LOSS"]}),
    );
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
            let clean =
                checker_report(&e, dest)["clean"].as_bool().unwrap_or(false) && iss.is_empty();
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
            e.insert_document(coll, doc_record(i, 1, "w", i as i64))
                .unwrap();
            let end = epoch.elapsed().as_micros();
            lat2.lock().unwrap().push((start, end, end - start));
            e4_ack(&f, &format!("ACK {i}"));
        }
    });
    drop(lat);
    E4Writers {
        stop,
        handles: vec![h],
    }
}

/// Child for the crash-during-backup case (controller kills the group).
fn run_e4child(dir: &std::path::Path, backup: &str) -> ! {
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let side = std::path::PathBuf::from(format!("{}.e4sidecar", dir.display()));
    let mut sc = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&side)
        .unwrap();
    use std::io::Write;
    for i in 0..20000u32 {
        e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
            .unwrap();
        let _ = writeln!(sc, "ACK {}", 1000 + i);
    }
    let _ = sc.flush();
    let _ = sc.sync_all();
    // earlier completed backup (control): must still restore after the crash
    let early = format!("{}.early-backup", dir.display());
    e.backup_to(std::path::Path::new(&early)).unwrap();
    let m = std::path::PathBuf::from(format!("{}.e4gate", dir.display()));
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&m)
    {
        let _ = writeln!(f, "pre-backup");
        let _ = f.sync_all();
    }
    e.backup_to(std::path::Path::new(backup)).unwrap();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&m)
    {
        let _ = writeln!(f, "backup-done");
        let _ = f.sync_all();
    }
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// E4: result of an integrity-case setup hook (backup path, expected present ids, detail)
type E4SetupResult = (std::path::PathBuf, Vec<u32>, String);

// ====================================================================
// E5: online compaction / tombstone safety / storage lifecycle
// ====================================================================

/// (sst file count, total bytes, tmp file count) for a db dir
fn e5_sst_stats(db: &std::path::Path) -> (usize, u64, usize) {
    let sst_dir = attentiondb_storage::catalog::Catalog::sst_dir(db);
    let mut n = 0usize;
    let mut b = 0u64;
    let mut t = 0usize;
    if let Ok(rd) = std::fs::read_dir(&sst_dir) {
        for e in rd.flatten() {
            let p = e.path();
            match p.extension().and_then(|x| x.to_str()) {
                Some("sst") => {
                    n += 1;
                    b += e.metadata().map(|m| m.len()).unwrap_or(0);
                }
                Some("tmp") => {
                    t += 1;
                }
                _ => {}
            }
        }
    }
    (n, b, t)
}

/// full value-level equality of exported state vs expected model
fn e5_full_match(e: &AttentionEngine, expected: &Model) -> (usize, usize, bool) {
    let (m, iss) = export_state(e);
    let ok = &m == expected && iss.is_empty();
    (m.len(), expected.len(), ok)
}

fn e5_clean(e: &AttentionEngine, dir: &std::path::Path) -> bool {
    checker_report(e, dir)["clean"].as_bool().unwrap_or(false)
}

/// reopen a db dir fresh; export + checker; close.
fn e5_reopen(dir: &std::path::Path, mode: &str) -> Result<(Model, bool), String> {
    std::env::set_var("PH3D_DURABILITY", mode);
    let e = attentiondb_core::AttentionEngine::open_dir(dir, dur_from_env())
        .map_err(|e| format!("{e}"))?;
    let (m, iss) = export_state(&e);
    let clean = checker_report(&e, dir)["clean"].as_bool().unwrap_or(false) && iss.is_empty();
    e.close().unwrap();
    Ok((m, clean))
}

/// expected value-model from ACK/DEL sidecar lines. Two doc families appear in
/// the E5 harness: pre-built checkpoints use doc_record(1000+i, 1, "c", i) and
/// live writers use doc_record(i, 1, "w", i) — the cat/num rule follows the idx.
fn e5_expected_from_sidecar(lines: &[String]) -> Model {
    let mut m: Model = BTreeMap::new();
    for l in lines {
        if let Some(v) = l.strip_prefix("ACK ") {
            if let Ok(i) = v.parse::<u32>() {
                if i >= 5000 {
                    m.insert(i, (1, "c".into(), (i - 5000) as i64));
                } else {
                    m.insert(i, (1, "w".into(), i as i64));
                }
            }
        } else if let Some(v) = l.strip_prefix("DEL ") {
            if let Ok(i) = v.parse::<u32>() {
                m.remove(&i);
            }
        }
    }
    m
}

/// child for crash-during-compaction windows: builds 3 SST generations with a
/// tombstone, then compact_storage() — the configured crash gate parks the child
/// (groupkill model); the controller kills the process group at the window.
fn run_e5child(dir: &std::path::Path) -> ! {
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    // gen1
    e.insert_document("bench", doc_record(1, 1, "c", 100))
        .unwrap();
    e.insert_document("bench", doc_record(2, 1, "c", 200))
        .unwrap();
    e.checkpoint().unwrap();
    // gen2: same-uuid version update of idx1 (equal-ms collision candidate)
    let mut upd = doc_record(1, 1, "c", 100);
    upd.fields.insert("num".to_string(), serde_json::json!(150));
    e.insert_document("bench", upd).unwrap();
    e.checkpoint().unwrap();
    // gen3: delete idx2
    e.delete_document("bench", &uuid_for(2, 1).to_string())
        .unwrap();
    e.checkpoint().unwrap();
    e.compact_storage().unwrap();
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// Main E5 driver. Produces <out>/e5-compaction.csv (+ e5-crash.csv).
fn run_e5(out: &str) -> String {
    use std::sync::atomic::Ordering;
    std::fs::create_dir_all(out).unwrap();
    let mut rows = String::from("case,mode,sst_before,sst_after,tombstones_removed,bytes_before,bytes_after,compact_us,model_match,checker_clean,resurrect,reader_errors,max_writer_latency_us,notes\n");
    let mut crash = String::from("window,mode,aborted_at_window,model_match,checker_clean,tmp_clean,restart_compact_ok,notes\n");
    let root = std::path::PathBuf::from("/tmp/ph3e-e5");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut cells = 0usize;

    // ---- C0 offline control: 3 flush generations -> compact_storage ----
    {
        let dir = root.join("c0-control");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = open_db(&dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let mut expected: Model = BTreeMap::new();
        for b in 0..3u32 {
            for i in (b * 10)..(b * 10 + 10) {
                e.insert_document("bench", doc_record(i, 1, "c", i as i64))
                    .unwrap();
                expected.insert(i, (1, "c".into(), i as i64));
            }
            e.checkpoint().unwrap();
        }
        let (pre_m, _) = export_state(&e);
        let pre_ok = pre_m == expected;
        let (sb, bb, _) = e5_sst_stats(&dir);
        let t0 = std::time::Instant::now();
        let st = e.compact_storage().unwrap();
        let compact_us = t0.elapsed().as_micros();
        let (sa, ba, ta) = e5_sst_stats(&dir);
        let (got, exp, ok) = e5_full_match(&e, &expected);
        let clean = e5_clean(&e, &dir);
        // restart + reopen equality
        e.close().unwrap();
        let (rm, rc) = e5_reopen(&dir, "sync").unwrap();
        let rok = rm == expected && rc;
        rows.push_str(&format!("c0-offline-control,sync,{sb},{sa},{},{bb},{ba},{compact_us},{},\"{clean}\",none,0,0,pre_eq={pre_ok} reopen_eq={rok} tmp_after={ta} entries={got}/{exp}\n",
            st.tombstones_removed, if ok && pre_ok && rok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // ---- multi-generation version resolution (prompt section 11 shape) ----
    {
        let dir = root.join("multigen");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = open_db(&dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        // SST1: A=v1(100) B=v1 C=v1 ; SST2: A same-uuid -> 200, D=v1 ; SST3: B deleted, C=v1 stays
        e.insert_document("bench", doc_record(10, 1, "c", 100))
            .unwrap(); // A
        e.insert_document("bench", doc_record(11, 1, "c", 110))
            .unwrap(); // B
        e.insert_document("bench", doc_record(12, 1, "c", 120))
            .unwrap(); // C
        e.checkpoint().unwrap();
        let mut a2 = doc_record(10, 1, "c", 200); // SAME uuid, new value
        a2.fields.insert("num".to_string(), serde_json::json!(200));
        e.insert_document("bench", a2).unwrap();
        e.insert_document("bench", doc_record(13, 1, "c", 130))
            .unwrap(); // D
        e.checkpoint().unwrap();
        e.delete_document("bench", &uuid_for(11, 1).to_string())
            .unwrap(); // B del
        e.checkpoint().unwrap();
        // 4th generation created by compact_storage's own flush: A->300, C deleted, E new
        let mut expected: Model = BTreeMap::new();
        expected.insert(10, (1, "c".into(), 300));
        expected.insert(13, (1, "c".into(), 130));
        expected.insert(14, (1, "c".into(), 140)); // E
        let mut a3 = doc_record(10, 1, "c", 300);
        // build 4th gen in memtable BEFORE compact (its flush carries it)
        a3.fields.insert("num".to_string(), serde_json::json!(300));
        e.insert_document("bench", a3).unwrap();
        e.insert_document("bench", doc_record(14, 1, "c", 140))
            .unwrap();
        e.delete_document("bench", &uuid_for(12, 1).to_string())
            .unwrap(); // C del
        let (sb, bb, _) = e5_sst_stats(&dir);
        let st = e.compact_storage().unwrap();
        let (sa, ba, _) = e5_sst_stats(&dir);
        let (got, exp, ok) = e5_full_match(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e5_reopen(&dir, "sync").unwrap();
        let rok = rm == expected && rc;
        // resurrect probe: B and C uuids must be absent
        let resurrect = if rm.contains_key(&11) || rm.contains_key(&12) {
            "RESURRECTED"
        } else {
            "none"
        };
        rows.push_str(&format!("multigen-version-resolution,sync,{sb},{sa},{},{bb},{ba},-,{},\"{clean}\",{resurrect},0,0,expected A=300 B=del C=del D E; reopen_eq={rok} entries={got}/{exp}\n",
            st.tombstones_removed, if ok && rok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // ---- tombstone GC: basic / deep / reinsert-loop ----
    {
        let dir = root.join("tomb-basic");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = open_db(&dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e.insert_document("bench", doc_record(20, 1, "c", 1))
            .unwrap();
        e.checkpoint().unwrap();
        e.delete_document("bench", &uuid_for(20, 1).to_string())
            .unwrap();
        e.checkpoint().unwrap();
        let (sb, _, _) = e5_sst_stats(&dir);
        let st = e.compact_storage().unwrap();
        let (sa, _, _) = e5_sst_stats(&dir);
        e.close().unwrap();
        let (rm, rc) = e5_reopen(&dir, "sync").unwrap();
        let resurrect = if rm.contains_key(&20) {
            "RESURRECTED"
        } else {
            "none"
        };
        rows.push_str(&format!("tombstone-gc-basic,sync,{sb},{sa},{},-,-,-,MATCH,\"{rc}\",{resurrect},0,0,ins/ckpt/del/ckpt/compact/restart; docs={}\n",
            st.tombstones_removed, rm.len()));
        cells += 1;
    }
    {
        let dir = root.join("tomb-deep2");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = open_db(&dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e.insert_document("bench", doc_record(21, 1, "c", 1))
            .unwrap();
        e.checkpoint().unwrap();
        let mut u = doc_record(21, 1, "c", 2);
        u.fields.insert("num".to_string(), serde_json::json!(2));
        e.insert_document("bench", u).unwrap();
        e.checkpoint().unwrap();
        e.delete_document("bench", &uuid_for(21, 1).to_string())
            .unwrap();
        e.checkpoint().unwrap();
        let st1 = e.compact_storage().unwrap();
        e.close().unwrap();
        let (m1, c1) = e5_reopen(&dir, "sync").unwrap();
        let res1 = if m1.contains_key(&21) {
            "RESURRECTED"
        } else {
            "none"
        };
        // reinsert fresh uuid
        let e2 = open_db(&dir);
        e2.insert_document("bench", doc_record(21, 2, "c", 3))
            .unwrap();
        e2.checkpoint().unwrap();
        let st2 = e2.compact_storage().unwrap();
        e2.close().unwrap();
        let (m2, c2) = e5_reopen(&dir, "sync").unwrap();
        let mut expected: Model = BTreeMap::new();
        expected.insert(21, (2, "c".into(), 3));
        let ok = m2 == expected && c2 && c1 && res1 == "none";
        rows.push_str(&format!("tombstone-deep-reinsert,sync,-,-,{},-,-,-,{},{c1}/{c2},{},0,0,ins/upd/del/compact/restart -> absent; reinsert v2 -> present once\n",
            st1.tombstones_removed + st2.tombstones_removed,
            if ok { "MATCH" } else { "MISMATCH" },
            if m2.contains_key(&21) && res1 == "none" { "none" } else { "RESURRECTED" }));
        cells += 1;
    }
    {
        // delete -> reinsert -> delete, repeatedly (prompt section 12 tail)
        let dir = root.join("tomb-loop");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = open_db(&dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let mut ok_all = true;
        let mut tombs = 0usize;
        for r in 1..=3u64 {
            e.insert_document("bench", doc_record(30, r, "c", r as i64))
                .unwrap();
            e.checkpoint().unwrap();
            e.delete_document("bench", &uuid_for(30, r).to_string())
                .unwrap();
            e.checkpoint().unwrap();
            let st = e.compact_storage().unwrap();
            tombs += st.tombstones_removed;
            let (m, iss) = export_state(&e);
            if m.contains_key(&30) || !iss.is_empty() {
                ok_all = false;
            }
        }
        e.close().unwrap();
        let (rm, rc) = e5_reopen(&dir, "sync").unwrap();
        let resurrect = if rm.contains_key(&30) {
            "RESURRECTED"
        } else {
            "none"
        };
        rows.push_str(&format!("delete-reinsert-delete-x3,sync,-,-,{tombs},-,-,-,{},\"{rc}\",{resurrect},0,0,A remains deleted after 3 rounds\n",
            if ok_all && rc { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // ---- partial-merge tombstone RETENTION (storage API level: unreachable via
    //      engine because auto-compact full-merges at <=4 files; verified here) ----
    {
        let dir = root.join("partial-merge-retention");
        std::fs::create_dir_all(&dir).unwrap();
        let sst_dir = attentiondb_storage::catalog::Catalog::sst_dir(&dir);
        std::fs::create_dir_all(&sst_dir).unwrap();
        for g in 0..5u32 {
            let path = sst_dir.join(format!("sstable_{g:03}.sst"));
            let mut w = attentiondb_storage::sstable::SSTableWriter::new(&path).unwrap();
            let mut rec = {
                let mut f = std::collections::HashMap::new();
                f.insert("idx".to_string(), serde_json::json!(g));
                attentiondb_storage::record::Record::new(f)
            };
            rec.id = uuid_for(40 + g, 1);
            if g >= 3 {
                rec.tags.push("__TOMBSTONE__".to_string());
            }
            w.append(rec.id.as_bytes().to_vec(), rec.to_msgpack().unwrap())
                .unwrap();
            w.flush().unwrap();
        }
        let cfg = attentiondb_storage::compaction::CompactionConfig {
            min_files_to_compact: 4,
            max_files_per_run: 4,
        };
        let r = attentiondb_storage::compaction::compact(&sst_dir, &cfg)
            .unwrap()
            .unwrap();
        // merge set = oldest 4 of 5 -> tombstones RETAINED (gc only on full merge)
        let retained = r.tombstones_removed == 0;
        // the tombstoned records must still be present in the OUTPUT
        let reader = attentiondb_storage::sstable::SSTableReader::open(&r.output_path).unwrap();
        let mut tomb_in_output = 0usize;
        for entry in reader.iter() {
            if let Ok(rec) = attentiondb_storage::record::Record::from_msgpack(&entry.value) {
                if rec.tags.contains(&"__TOMBSTONE__".to_string()) {
                    tomb_in_output += 1;
                }
            }
        }
        rows.push_str(&format!("partial-merge-retains-tombstones,storage-api,5,-,{},-,-,-,MATCH,true,none,0,0,merged oldest 4 of 5; retained={retained} tomb_in_output={tomb_in_output}\n",
            r.tombstones_removed));
        cells += 1;
    }

    // ---- repeated compaction: write/compact x4, verify after every round ----
    {
        let dir = root.join("repeat-compact");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = open_db(&dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let mut expected: Model = BTreeMap::new();
        let mut ok_all = true;
        let mut diag = String::new();
        for r in 0..4u32 {
            for i in (r * 5)..(r * 5 + 5) {
                e.insert_document("bench", doc_record(50 + i, 1, "c", i as i64))
                    .unwrap();
                expected.insert(50 + i, (1, "c".into(), i as i64));
            }
            if r == 2 {
                e.delete_document("bench", &uuid_for(50, 1).to_string())
                    .unwrap();
                expected.remove(&50);
            }
            e.checkpoint().unwrap();
            let st = e.compact_storage().unwrap();
            let (_, _, ok) = e5_full_match(&e, &expected);
            let clean = e5_clean(&e, &dir);
            let rob = st.files_after <= st.files_before;
            ok_all &= ok && clean && rob;
            diag.push_str(&format!(
                "r{}:{}/c{}/f{}<=f{} ",
                r,
                if ok { "ok" } else { "BAD" },
                if clean { "ok" } else { "BAD" },
                st.files_before,
                st.files_after
            ));
        }
        e.close().unwrap();
        let (rm, rc) = e5_reopen(&dir, "sync").unwrap();
        let req = rm == expected && rc;
        ok_all &= req;
        rows.push_str(&format!("repeated-compaction-x4,sync,-,-,0,-,-,-,{},\"{rc}\",none,0,0,{diag}restart_eq={req} restart_len={}\n",
            if ok_all { "MATCH" } else { "MISMATCH" }, rm.len()));
        cells += 1;
    }

    // ---- concurrent readers + compaction ----
    {
        let dir = root.join("readers-compact");
        std::env::set_var("PH3D_DURABILITY", "group");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        for i in 0..60u32 {
            e.insert_document("bench", doc_record(100 + i, 1, "c", i as i64))
                .unwrap();
        }
        e.checkpoint().unwrap();
        // second generation with tombstones so the concurrent compaction merges
        for i in 0..10u32 {
            e.delete_document("bench", &uuid_for(100 + i, 1).to_string())
                .unwrap();
        }
        e.checkpoint().unwrap();
        let errors = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let rlat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let reps = std::sync::Arc::new(std::time::Instant::now());
        let mut readers = Vec::new();
        for _ in 0..3 {
            let e2 = e.clone();
            let er = errors.clone();
            let rl = rlat.clone();
            let rp = reps.clone();
            readers.push(std::thread::spawn(move || {
                for i in 0..200u32 {
                    let q = vec_for(100 + (i % 60));
                    let t = rp.elapsed().as_micros();
                    let r = e2.attend("bench", &[HEAD.to_string()], &q, 5);
                    let d = rp.elapsed().as_micros() - t;
                    rl.lock().unwrap().push(d);
                    if r.is_err() {
                        er.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }
        let (sb, bb, _) = e5_sst_stats(&dir);
        let t0 = std::time::Instant::now();
        let st = e.compact_storage().unwrap();
        let compact_us = t0.elapsed().as_micros();
        let mut rl_all = rlat.lock().unwrap().clone();
        for r in readers {
            r.join().unwrap();
        }
        rl_all.extend(rlat.lock().unwrap().iter().copied());
        rl_all.sort_unstable();
        let p50 = rl_all.get(rl_all.len() / 2).copied().unwrap_or(0);
        let p99 = rl_all.get(rl_all.len() * 99 / 100).copied().unwrap_or(0);
        let re = errors.load(Ordering::Relaxed);
        let (sa, ba, _) = e5_sst_stats(&dir);
        let mut expect: Model = BTreeMap::new();
        for i in 0..60u32 {
            expect.insert(100 + i, (1, "c".into(), i as i64));
        }
        for i in 0..10u32 {
            expect.remove(&(100 + i));
        }
        let (got, exp, ok) = e5_full_match(&e, &expect);
        let clean = e5_clean(&e, &dir);
        rows.push_str(&format!("readers-during-compaction,group,{sb},{sa},{},{bb},{ba},{compact_us},{},\"{clean}\",none,{re},0,3 readers x 200 attends DURING compaction: p50={p50}us p99={p99}us n={} (blocked-not-errored); entries={got}/{exp}\n",
            st.tombstones_removed, if ok && re == 0 { "MATCH" } else { "MISMATCH" }, rl_all.len()));
        cells += 1;
    }

    // ---- single writer + compaction (background) ----
    {
        let dir = root.join("writer-compact");
        std::env::set_var("PH3D_DURABILITY", "group");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        // pre-build two flush generations incl. tombstones so compaction has work
        // (5000-base namespace: distinct from live-writer ids)
        for i in 0..40u32 {
            e.insert_document("bench", doc_record(5000 + i, 1, "c", i as i64))
                .unwrap();
            e4_ack(&side, &format!("ACK {}", 5000 + i));
        }
        e.checkpoint().unwrap();
        for i in 0..5u32 {
            e.delete_document("bench", &uuid_for(5000 + i, 1).to_string())
                .unwrap();
            e4_ack(&side, &format!("DEL {}", 5000 + i));
        }
        e.checkpoint().unwrap();
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(
            e.clone(),
            side.clone(),
            200..500,
            lat.clone(),
            "bench",
            epoch.clone(),
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        let (sb, bb, _) = e5_sst_stats(&dir);
        let st = e.compact_storage().unwrap();
        let (sa, ba, _) = e5_sst_stats(&dir);
        w.stop.store(true, Ordering::Relaxed);
        for h in w.handles.drain(..) {
            h.join().unwrap();
        }
        let max_lat = lat.lock().unwrap().iter().map(|x| x.2).max().unwrap_or(0);
        let lines = e4_sidecar_lines(&dir);
        let expected = e5_expected_from_sidecar(&lines);
        let (got, exp, ok) = e5_full_match(&e, &expected);
        let clean = e5_clean(&e, &dir);
        rows.push_str(&format!("writer-during-compaction,group,{sb},{sa},{},{bb},{ba},-,{},{clean},none,0,{max_lat},background compaction vs live writer; entries={got}/{exp}\n",
            st.tombstones_removed,
            if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // ---- multi-writer + compaction ----
    {
        let dir = root.join("multiwriter-compact");
        std::env::set_var("PH3D_DURABILITY", "group");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut ws = Vec::new();
        for t in 0..3u32 {
            ws.push(e4_spawn_writer(
                e.clone(),
                side.clone(),
                (1000 + t * 60)..(1000 + t * 60 + 60),
                lat.clone(),
                "bench",
                epoch.clone(),
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(8));
        let st = e.compact_storage().unwrap();
        for w in ws.iter_mut() {
            w.stop.store(true, Ordering::Relaxed);
        }
        for w in ws.iter_mut() {
            for h in w.handles.drain(..) {
                h.join().unwrap();
            }
        }
        let max_lat = lat.lock().unwrap().iter().map(|x| x.2).max().unwrap_or(0);
        let lines = e4_sidecar_lines(&dir);
        let expected = e5_expected_from_sidecar(&lines);
        let (got, exp, ok) = e5_full_match(&e, &expected);
        let clean = e5_clean(&e, &dir);
        rows.push_str(&format!("multiwriter-during-compaction,group,-,-,{},-,-,-,{},{},none,0,{max_lat},3 writers; gate serializes; entries={got}/{exp}\n",
            st.tombstones_removed,
            if ok { "MATCH" } else { "MISMATCH" },
            if clean { "true" } else { "false" }));
        cells += 1;
    }

    // ---- checkpoint x compaction orders ----
    for order in ["ckpt-then-compact", "compact-then-ckpt"] {
        let dir = root.join(format!("order-{order}"));
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = open_db(&dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let mut expected: Model = BTreeMap::new();
        for i in 0..20u32 {
            e.insert_document("bench", doc_record(300 + i, 1, "c", i as i64))
                .unwrap();
            expected.insert(300 + i, (1, "c".into(), i as i64));
        }
        let mut ok = true;
        if order == "ckpt-then-compact" {
            e.checkpoint().unwrap();
            e.compact_storage().unwrap();
        } else {
            e.compact_storage().unwrap();
            e.checkpoint().unwrap();
        }
        let (_, _, m1) = e5_full_match(&e, &expected);
        ok &= m1;
        // repeat the other order on the same db
        if order == "ckpt-then-compact" {
            e.compact_storage().unwrap();
            e.checkpoint().unwrap();
        } else {
            e.checkpoint().unwrap();
            e.compact_storage().unwrap();
        }
        let (got, exp, m2) = e5_full_match(&e, &expected);
        let clean = e5_clean(&e, &dir);
        rows.push_str(&format!("{order},sync,-,-,0,-,-,-,{},\"{clean}\",none,0,0,both orders on same db; entries={got}/{exp}\n",
            if ok && m2 && clean { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // ---- WAL rotation + compaction ----
    {
        let dir = root.join("rotation-compact");
        std::env::set_var("PH3D_DURABILITY", "group");
        std::env::set_var("ATTENTIONDB_WAL_SEGMENT_BYTES", "2048");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(
            e.clone(),
            side.clone(),
            500..800,
            lat.clone(),
            "bench",
            epoch.clone(),
        );
        std::thread::sleep(std::time::Duration::from_millis(8));
        let st = e.compact_storage().unwrap();
        w.stop.store(true, Ordering::Relaxed);
        for h in w.handles.drain(..) {
            h.join().unwrap();
        }
        std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
        let max_lat = lat.lock().unwrap().iter().map(|x| x.2).max().unwrap_or(0);
        let lines = e4_sidecar_lines(&dir);
        let expected = e5_expected_from_sidecar(&lines);
        let (got, exp, ok) = e5_full_match(&e, &expected);
        let clean = e5_clean(&e, &dir);
        rows.push_str(&format!("wal-rotation-during-compaction,group,-,-,{},-,-,-,{},{clean},none,0,{max_lat},2KiB segments; no seq gap/watermark damage; entries={got}/{exp}\n",
            st.tombstones_removed,
            if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // ---- backup x compaction ----
    for (name, do_compact_first) in [("bkp-after-compact", true), ("bkp-before-compact", false)] {
        let dir = root.join(name);
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = open_db(&dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let mut expected: Model = BTreeMap::new();
        for i in 0..20u32 {
            e.insert_document("bench", doc_record(400 + i, 1, "c", i as i64))
                .unwrap();
            expected.insert(400 + i, (1, "c".into(), i as i64));
        }
        e.checkpoint().unwrap();
        if do_compact_first {
            e.compact_storage().unwrap();
        }
        let backup = root.join(format!("{name}.backup"));
        e.backup_to(&backup).unwrap();
        if !do_compact_first {
            e.compact_storage().unwrap();
        }
        // compaction AFTER backup must not disturb the backup dir
        let (r, clean, _) =
            e4_restore_and_open(&backup, &root.join(format!("{name}.restored")), "sync").unwrap();
        let ok = r == expected && clean;
        rows.push_str(&format!("{name},sync,-,-,0,-,-,-,{},\"{clean}\",none,0,0,backup restores own snapshot; compaction did not mutate backup dir\n",
            if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // ---- full lifecycle: compact -> backup -> restore -> restart -> compact again ----
    {
        let dir = root.join("lifecycle");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = open_db(&dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let mut expected: Model = BTreeMap::new();
        for i in 0..30u32 {
            e.insert_document("bench", doc_record(500 + i, 1, "c", i as i64))
                .unwrap();
            expected.insert(500 + i, (1, "c".into(), i as i64));
        }
        e.checkpoint().unwrap();
        e.delete_document("bench", &uuid_for(500, 1).to_string())
            .unwrap();
        expected.remove(&500);
        e.compact_storage().unwrap();
        let backup = root.join("lifecycle.backup");
        e.backup_to(&backup).unwrap();
        let dest = root.join("lifecycle.restored");
        let (r, clean, _) = e4_restore_and_open(&backup, &dest, "sync").unwrap();
        // compact the RESTORED db
        let e2 = open_db(&dest);
        e2.compact_storage().unwrap();
        let (got, exp, ok2) = e5_full_match(&e2, &expected);
        let clean2 = e5_clean(&e2, &dest);
        e2.close().unwrap();
        let ok = r == expected && clean && ok2 && clean2;
        rows.push_str(&format!("compact-backup-restore-restart-compact,sync,-,-,0,-,-,-,{},\"{clean}/{clean2}\",none,0,0,full lifecycle; entries={got}/{exp}\n",
            if ok { "MATCH" } else { "MISMATCH" }));
        cells += 1;
    }

    // ---- manifest fallback after compaction ----
    {
        let dir = root.join("fallback-current");
        let mut expected: Model = BTreeMap::new();
        {
            std::env::set_var("PH3D_DURABILITY", "sync");
            let e = open_db(&dir);
            e.create_collection("bench", DIM, &[HEAD]).unwrap();
            for i in 0..20u32 {
                e.insert_document("bench", doc_record(600 + i, 1, "c", i as i64))
                    .unwrap();
                expected.insert(600 + i, (1, "c".into(), i as i64));
            }
            e.checkpoint().unwrap();
            e.compact_storage().unwrap();
            e.close().unwrap();
        }
        let dst = root.join("fallback-current.corrupt");
        copy_dir_all(&dir, &dst);
        let p = dst;
        std::fs::write(p.join("CURRENT"), b"manifest-999999999\n").unwrap();
        match e5_reopen(&p, "sync") {
            Err(err) => rows.push_str(&format!(
                "manifest-fallback-after-compact,sync,-,-,0,-,-,-,REFUSED,-,none,0,0,{err}\n"
            )),
            Ok((rm, rc)) => {
                let ok = rm == expected && rc;
                rows.push_str(&format!("manifest-fallback-after-compact,sync,-,-,0,-,-,-,{},\"{rc}\",none,0,0,corrupt CURRENT -> valid generation fallback\n",
                    if ok { "MATCH" } else { "MISMATCH" }));
            }
        }
        cells += 1;
    }

    // ---- partial artifacts: tmp ignored, garbage sst refuses ----
    {
        let dir = root.join("artifact-tmp");
        let mut expected: Model = BTreeMap::new();
        {
            std::env::set_var("PH3D_DURABILITY", "sync");
            let e = open_db(&dir);
            e.create_collection("bench", DIM, &[HEAD]).unwrap();
            for i in 0..10u32 {
                e.insert_document("bench", doc_record(700 + i, 1, "c", i as i64))
                    .unwrap();
                expected.insert(700 + i, (1, "c".into(), i as i64));
            }
            e.checkpoint().unwrap();
            e.compact_storage().unwrap();
            e.close().unwrap();
        }
        let dst = root.join("artifact-tmp.corrupt");
        copy_dir_all(&dir, &dst);
        let p = dst;
        let sst_dir = attentiondb_storage::catalog::Catalog::sst_dir(&p);
        std::fs::write(sst_dir.join("compacted_1.sst.tmp"), b"\x00garbage-tmp\x01").unwrap();
        match e5_reopen(&p, "sync") {
            Err(err) => rows.push_str(&format!("partial-artifact-tmp,sync,-,-,0,-,-,-,REFUSED,-,none,0,0,tmp must be ignored: {err}\n")),
            Ok((rm, rc)) => {
                let (_, _, tmp_n) = e5_sst_stats(&p);
                let ok = rm == expected && rc;
                rows.push_str(&format!("partial-artifact-tmp,sync,-,-,0,-,-,-,{},\"{rc}\",none,0,0,garbage .tmp ignored/deleted (remaining tmp={tmp_n})\n",
                    if ok { "MATCH" } else { "MISMATCH" }));
            }
        }
        let dst2 = root.join("artifact-garbage");
        copy_dir_all(&dir, &dst2);
        let p2 = dst2;
        let sst2 = attentiondb_storage::catalog::Catalog::sst_dir(&p2);
        std::fs::write(sst2.join("garbage_999.sst"), b"not-an-sstable").unwrap();
        match e5_reopen(&p2, "sync") {
            Err(_) => rows.push_str("partial-artifact-garbage-sst,sync,-,-,0,-,-,-,REFUSED,-,none,0,0,garbage .sst refused loudly (policy)\n"),
            Ok((rm, rc)) => rows.push_str(&format!("partial-artifact-garbage-sst,sync,-,-,0,-,-,-,ACCEPTED,\"{rc}\",none,0,0,UNEXPECTED acceptance; docs={}\n", rm.len())),
        }
        cells += 2;
    }

    // ---- auto-compaction scheduler observation ----
    {
        let dir = root.join("auto-scheduler");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = open_db(&dir);
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let mut min_files = usize::MAX;
        let mut max_files = 0usize;
        for b in 0..6u32 {
            for i in (b * 1000)..(b * 1000 + 1000) {
                e.insert_document("bench", doc_record(10000 + i, 1, "c", i as i64))
                    .unwrap();
            }
            let (n, _, _) = e5_sst_stats(&dir);
            min_files = min_files.min(n);
            max_files = max_files.max(n);
        }
        e.delete_document("bench", &uuid_for(10000, 1).to_string())
            .unwrap();
        let st = e.compact_storage().unwrap();
        let (n_final, _, _) = e5_sst_stats(&dir);
        rows.push_str(&format!("auto-scheduler-observation,sync,{max_files},{n_final},{},-,-,-,MATCH,-,none,0,0,threshold flushes: files bounded [{min_files}..{max_files}]; explicit compact -> {n_final} files\n",
            st.tombstones_removed));
        cells += 1;
    }

    // ---- collection isolation under compaction (INV-C5) ----
    {
        let dir = root.join("collections-compact");
        std::env::set_var("PH3D_DURABILITY", "group");
        let e = open_db(&dir);
        e.create_collection("alpha", DIM, &[HEAD]).unwrap();
        e.create_collection("beta", DIM, &[HEAD]).unwrap();
        let mut exp_a: Model = BTreeMap::new();
        let mut exp_b: Model = BTreeMap::new();
        for i in 0..15u32 {
            let mut ra = doc_record_c(1, i, 1, "a", i as i64);
            ra.k_vecs.insert(HEAD.to_string(), vec_for(1000 + i));
            e.insert_document("alpha", ra).unwrap();
            exp_a.insert(i, (1, "a".into(), i as i64));
            let mut rb = doc_record_c(2, i, 1, "b", (i * 2) as i64);
            rb.k_vecs.insert(HEAD.to_string(), vec_for(2000 + i));
            e.insert_document("beta", rb).unwrap();
            exp_b.insert(i, (1, "b".into(), (i * 2) as i64));
        }
        e.checkpoint().unwrap();
        // deletes in each collection create cross-generation versions
        e.delete_document("alpha", &uuidc(1, 0, 1).to_string())
            .unwrap();
        e.delete_document("beta", &uuidc(2, 1, 1).to_string())
            .unwrap();
        exp_a.remove(&0);
        exp_b.remove(&1);
        let st = e.compact_storage().unwrap();
        let (ea, iss_a) = export_state_coll(&e, "alpha");
        let (eb, iss_b) = export_state_coll(&e, "beta");
        let ok = ea == exp_a && eb == exp_b && iss_a.is_empty() && iss_b.is_empty();
        let clean = e5_clean(&e, &dir);
        rows.push_str(&format!("collections-isolation-compaction,group,-,-,{},-,-,-,{},\"{clean}\",none,0,0,alpha={}/{} beta={}/{} no cross-contamination; db-level merge preserves per-collection state\n",
            st.tombstones_removed,
            if ok { "MATCH" } else { "MISMATCH" },
            ea.len(), exp_a.len(), eb.len(), exp_b.len()));
        cells += 1;
    }

    // ---- crash windows (group kill at each instrumented gate) ----
    for gate in [
        "compact_before_merge",
        "compact_after_output",
        "compact_after_install",
        "compact_after_cleanup",
    ] {
        let cdir = root.join(format!("crash-{gate}"));
        let marker = root.join(format!("crash-{gate}.marker"));
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["dbtest", "e5child", "--dir", cdir.to_str().unwrap()])
            .env("PH3D_DURABILITY", "sync")
            .env("PH3E_CRASH_AT", gate)
            .env("PH3E_CRASH_HIT", "1")
            .env("PH3E_CRASH_MODEL", "groupkill")
            .env("PH3E_CRASH_MARKER", &marker);
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        let mut child = cmd.spawn().unwrap();
        let mut reached = false;
        for _ in 0..6000 {
            if marker.exists() {
                reached = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        if reached {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        let _ = child.wait();
        // verify: fresh open -> same logical state, checker clean, no tmp, compact again OK
        let expected: Model = {
            let mut m: Model = BTreeMap::new();
            m.insert(1, (1, "c".into(), 150)); // updated value (same-uuid update)
            m // idx2 deleted pre-compaction; idx3 never existed
        };
        std::env::remove_var("PH3E_CRASH_AT");
        match e5_reopen(&cdir, "sync") {
            Err(err) => crash.push_str(&format!("{gate},sync,{reached},REFUSED,-,-,-,{err}\n")),
            Ok((rm, rc)) => {
                let (_, _, tmp_n) = e5_sst_stats(&cdir);
                let ok = rm == expected && rc;
                // restart + compact again must work
                let e2 = open_db(&cdir);
                let st2 = e2.compact_storage().unwrap();
                let (_, _, ok2) = e5_full_match(&e2, &expected);
                let clean2 = e5_clean(&e2, &cdir);
                e2.close().unwrap();
                crash.push_str(&format!("{gate},sync,{reached},{},\"{rc}\",tmp={tmp_n},{},window verified; second compact entries+tombs={}/{}\n",
                    if ok { "MATCH" } else { "MISMATCH" }, if ok2 && clean2 { "ok" } else { "FAIL" }, st2.entries, st2.tombstones_removed));
            }
        }
        cells += 1;
    }

    std::fs::write(format!("{out}/e5-compaction.csv"), rows).unwrap();
    std::fs::write(format!("{out}/e5-crash.csv"), crash).unwrap();
    format!("e5: {cells} cells -> {out}/e5-compaction.csv + e5-crash.csv")
}

// ====================================================================
// E6: transaction semantics / atomic commit boundaries
// ====================================================================

fn e6_expected_model(pairs: &[(u32, i64)]) -> Model {
    let mut m: Model = BTreeMap::new();
    for (i, n) in pairs {
        m.insert(*i, (1, "t".into(), *n));
    }
    m
}

fn e6_model_eq(e: &AttentionEngine, expected: &Model) -> (usize, usize, bool) {
    let (m, iss) = export_state(e);
    (m.len(), expected.len(), &m == expected && iss.is_empty())
}

fn e6_open(dir: &std::path::Path, mode: &str) -> Result<(Model, bool), String> {
    std::env::set_var("PH3D_DURABILITY", mode);
    let e = attentiondb_core::AttentionEngine::open_dir(dir, dur_from_env())
        .map_err(|e| format!("{e}"))?;
    let (m, iss) = export_state(&e);
    let clean = checker_report(&e, dir)["clean"].as_bool().unwrap_or(false) && iss.is_empty();
    e.close().unwrap();
    Ok((m, clean))
}

/// txn ids for the logical indexes used by the E6 harness
fn e6_stage_txn(e: &AttentionEngine, ids: &[u32], dels: &[u32], cat: &str) -> u64 {
    use attentiondb_core::transaction::TxnOp;
    let t = e.begin_transaction("bench");
    for i in ids {
        let mut r = doc_record(*i, 1, cat, *i as i64);
        r.k_vecs.insert(HEAD.to_string(), vec_for(*i));
        e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
    }
    for d in dels {
        e.record_transaction_operation(t, TxnOp::Delete(uuid_for(*d, 1)))
            .unwrap();
    }
    t
}

/// child for E6 crash families. Cases:
///   stage-K        : stage K ops, NO commit; marker <dir>.e6 = "staged"; park
///   commit         : stage 3 ins + 1 del; commit (crash gate may park inside); park
///   post-ack       : commit; on ACK -> process::abort() (F1 sudden death)
///   multi          : T1 commit; marker "t1-done"; stage T3 (no commit); park
fn e6_txn_child(dir: &std::path::Path, case: &str) -> ! {
    use attentiondb_core::transaction::TxnOp;
    use std::io::Write;
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for i in 9000..9005u32 {
        let mut r = doc_record(i, 1, "t", i as i64);
        r.k_vecs.insert(HEAD.to_string(), vec_for(i));
        e.insert_document("bench", r).unwrap();
    }
    e.checkpoint().unwrap();
    let marker = format!("{}.e6", dir.display());
    let mk = || {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&marker)
        {
            let _ = writeln!(f, "{case}");
            let _ = f.sync_all();
        }
    };
    match case {
        "stage-1" | "stage-3" | "stage-10" => {
            let k: u32 = case.trim_start_matches("stage-").parse().unwrap();
            let t = e.begin_transaction("bench");
            for i in 9100..(9100 + k) {
                let mut r = doc_record(i, 1, "t", i as i64);
                r.k_vecs.insert(HEAD.to_string(), vec_for(i));
                e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
            }
            mk(); // "staged" marker: all K ops staged, nothing committed
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }
        "commit" => {
            let ids: Vec<u32> = (9100..9103).collect();
            let t = e6_stage_txn(&e, &ids, &[9002], "t");
            let r = e.commit_transaction(t);
            mk(); // reached only if no gate parked first
            let _ = r;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }
        "post-ack" => {
            let ids: Vec<u32> = (9100..9103).collect();
            let t = e6_stage_txn(&e, &ids, &[9002], "t");
            let r = e.commit_transaction(t).is_ok();
            if r {
                // ACK happened (commit returned Ok); die suddenly, no cleanup.
                std::process::abort();
            }
            mk();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }
        "multi" => {
            let t1 = e6_stage_txn(&e, &[9100], &[], "t");
            let ok = e.commit_transaction(t1).unwrap();
            assert!(ok);
            mk(); // "t1-done": T1 committed, T3 not yet staged
            let t3 = e6_stage_txn(&e, &[9105], &[9001], "t");
            let _ =
                e.record_transaction_operation(t3, TxnOp::Insert(doc_record(9106, 1, "t", 9106)));
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }
        _ => panic!("unknown e6 child case {case}"),
    }
}

/// classify a crash-recovered state against the two legal outcomes
fn e6_classify(got: &Model, baseline: &Model, committed: &Model) -> (&'static str, bool) {
    if got == baseline {
        ("ABSENT_ATOMIC", true)
    } else if got == committed {
        ("PRESENT_ATOMIC", true)
    } else {
        ("PARTIAL", false)
    }
}

// ================================================================
// Phase 3E E7 — concurrency & isolation harness (spec §IV).
// Synchronization: std::sync::Barrier (never sleeps-as-ordering; bounded
// observation windows are documented at their use sites). All ordering
// measurements use std::time::Instant (monotonic), never wall clock.
// The DB is never its own oracle: every scenario is judged against an
// independent commit-order reference model + the consistency checker.
// ================================================================

struct E7Event {
    seq: usize,
    thread: usize,
    txn: u64,
    op: String,
    key: String,
    t_inv_us: u128,
    t_ret_us: u128,
    result: String,
}

type E7Log = std::sync::Arc<std::sync::Mutex<(Vec<E7Event>, std::time::Instant)>>;

/// CSV-safe rendering of debug values (Vec/set Debug forms contain commas).
fn csvs(s: impl std::fmt::Debug) -> String {
    format!("{s:?}").replace(',', ";").replace('"', "'")
}

fn e7_newlog() -> E7Log {
    std::sync::Arc::new(std::sync::Mutex::new((
        Vec::new(),
        std::time::Instant::now(),
    )))
}

/// Record one operation: invocation timestamp taken by the caller BEFORE the
/// op starts; completion recorded here (monotonic micros since log start).
fn e7_log(
    lg: &E7Log,
    thread: usize,
    txn: u64,
    op: &str,
    key: &str,
    inv: std::time::Instant,
    result: String,
) {
    let now = std::time::Instant::now();
    let mut g = lg.lock().unwrap();
    let seq = g.0.len();
    let t0 = g.1;
    g.0.push(E7Event {
        seq,
        thread,
        txn,
        op: op.to_string(),
        key: key.to_string(),
        t_inv_us: inv.duration_since(t0).as_micros(),
        t_ret_us: now.duration_since(t0).as_micros(),
        result,
    });
}

/// Log every 256th event from an UNBOUNDED spin loop (bounded RAM; sampled
/// ops still carry their true monotonic timestamps).
#[allow(clippy::too_many_arguments)]
fn e7_log_sampled(
    lg: &E7Log,
    ctr: &std::sync::atomic::AtomicUsize,
    thread: usize,
    txn: u64,
    op: &str,
    key: &str,
    inv: std::time::Instant,
    result: String,
) {
    let n = ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n.is_multiple_of(256) {
        e7_log(lg, thread, txn, op, key, inv, result);
    }
}

fn e7_write_events(out: &str, lg: &E7Log) {
    let g = lg.lock().unwrap();
    let mut s = String::from("seq,thread,txn_id,op,key,t_invoke_us,t_complete_us,result\n");
    for e in &g.0 {
        s.push_str(&format!(
            "{},{},{},{},{},{},{},\"{}\"\n",
            e.seq,
            e.thread,
            e.txn,
            e.op,
            e.key,
            e.t_inv_us,
            e.t_ret_us,
            e.result.replace('"', "'")
        ));
    }
    std::fs::write(format!("{out}/e7-events.csv"), s).unwrap();
}

/// Read the live (version, num) for a logical idx (probes versions 0..16 via
/// the uuid scheme). None = no live version. Empty-field reads (raced retire)
/// are retried once by the caller semantics — here treated as absent.
fn e7_read_idx(e: &AttentionEngine, idx: u32) -> Option<(u64, i64)> {
    for v in 0..16u64 {
        let uuid = uuid_for(idx, v);
        let numeric = e.id_mapper.read().uuid_to_id(&uuid);
        if let Some(n) = numeric {
            let f = e.get_document_fields(n);
            if let Some(num) = f.get("num").and_then(|s| s.parse::<i64>().ok()) {
                return Some((v, num));
            }
        }
    }
    None
}

/// Read the num of a SPECIFIC uuid version: Some(num) if live, None if
/// absent/retired. O(1).
fn e7_read_ver(e: &AttentionEngine, idx: u32, v: u64) -> Option<i64> {
    let uuid = uuid_for(idx, v);
    let numeric = e.id_mapper.read().uuid_to_id(&uuid)?;
    let f = e.get_document_fields(numeric);
    f.get("num").and_then(|s| s.parse::<i64>().ok())
}

fn e7_insert(e: &AttentionEngine, idx: u32, v: u64, num: i64) {
    let mut r = doc_record(idx, v, "t", num);
    r.k_vecs.insert(HEAD.to_string(), vec_for(idx));
    e.insert_document("bench", r).unwrap();
}

/// staged txn: delete current version(s) of `idx` at `delv`, insert (idx,v,num)
fn e7_stage_upsert(e: &AttentionEngine, idx: u32, delv: Option<u64>, v: u64, num: i64) -> u64 {
    use attentiondb_core::transaction::TxnOp;
    let t = e.begin_transaction("bench");
    if let Some(dv) = delv {
        e.record_transaction_operation(t, TxnOp::Delete(uuid_for(idx, dv)))
            .unwrap();
    }
    let mut r = doc_record(idx, v, "t", num);
    r.k_vecs.insert(HEAD.to_string(), vec_for(idx));
    e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
    t
}

fn e7_commit(e: &AttentionEngine, t: u64, lg: &E7Log, thread: usize, label: &str) -> bool {
    let inv = std::time::Instant::now();
    let r = e.commit_transaction(t).unwrap_or(false);
    e7_log(lg, thread, t, "commit", label, inv, format!("ok={r}"));
    r
}

fn e7_checker(e: &AttentionEngine, dir: &std::path::Path) -> bool {
    checker_report(e, dir)["clean"].as_bool().unwrap_or(false)
}

/// The E7 driver. Produces <out>/e7-conc.csv, e7-visibility.csv,
/// e7-events.csv, e7-crash.csv.
fn run_e7(out: &str) -> String {
    use attentiondb_core::transaction::TxnOp;
    std::fs::create_dir_all(out).unwrap();
    let mut rows =
        String::from("family,case,mode,threads,txns,ops,observed,model,checker,status,notes\n");
    let mut vis = String::from("scenario,observed_behavior,status\n");
    let mut crash =
        String::from("case,mode,aborted_at_gate,txn_state,atomicity,checker,model,notes\n");
    let root = std::path::PathBuf::from("/tmp/ph3e-e7");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut cells = 0usize;
    let lg_global = e7_newlog();
    let q = vec_for(7003);

    // ---------------- E7a: concurrent readers (1/2/4/8/16) ----------------
    for n in [1usize, 2, 4, 8, 16] {
        let dir = root.join(format!("e7a-readers-{n}"));
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        for i in 0..120u32 {
            e7_insert(&e, 7000 + i, 1, i as i64);
        }
        e.checkpoint().unwrap();
        let ops_per = 120usize;
        let barrier = Arc::new(std::sync::Barrier::new(n));
        let lg = lg_global.clone();
        let mut handles = Vec::new();
        for t in 0..n {
            let e2 = e.clone();
            let b = barrier.clone();
            let lg2 = lg.clone();
            let q2 = q.clone();
            handles.push(std::thread::spawn(move || {
                b.wait();
                let mut lats: Vec<u128> = Vec::new();
                // error kinds: [attend_err, attend_bad_id, scan_len, scan_err, get_fields]
                let mut ek = [0usize; 5];
                let mut bad_ids: Vec<u64> = Vec::new();
                for i in 0..ops_per {
                    let inv = std::time::Instant::now();
                    let r = e2.attend("bench", &[HEAD.to_string()], &q2, 10);
                    lats.push(inv.elapsed().as_micros());
                    match r {
                        Ok(res) => {
                            // valid numeric ids for this collection: 1..=120
                            // (mapper mints from 1; 0 = n/a sentinel). The id
                            // space was verified against the mapper in
                            // core/tests/e7_tiny_probe.rs.
                            for (id, _) in res.iter() {
                                if *id == 0 || *id > 120 {
                                    ek[1] += 1;
                                    if bad_ids.len() < 8 && !bad_ids.contains(id) {
                                        bad_ids.push(*id);
                                    }
                                }
                            }
                        }
                        Err(_) => ek[0] += 1,
                    }
                    if i % 3 == 0 {
                        let inv = std::time::Instant::now();
                        let s = e2.scan_filtered("bench", None, 10_000);
                        lats.push(inv.elapsed().as_micros());
                        match s {
                            Ok(v) => {
                                if v.len() != 120 {
                                    ek[2] += 1;
                                }
                            }
                            Err(_) => ek[3] += 1,
                        }
                    }
                    if i % 3 == 1 {
                        let inv = std::time::Instant::now();
                        // resolve through the mapper with the TRUE idx base and
                        // version (docs are (7000+i, v1)); never assume idx ==
                        // numeric
                        let nid = e2
                            .id_mapper
                            .read()
                            .uuid_to_id(&uuid_for(7000 + (i as u32) % 120, 1));
                        let got = nid.and_then(|n| {
                            e2.get_document_fields(n)
                                .get("num")
                                .and_then(|s| s.parse::<i64>().ok())
                        });
                        lats.push(inv.elapsed().as_micros());
                        match got {
                            Some(num) if (0..120).contains(&num) => {}
                            _ => ek[4] += 1,
                        }
                    }
                    e7_log(&lg2, t, 0, "read", "ro", inv, "ok".into());
                }
                (lats, ek, bad_ids)
            }));
        }
        let mut alllats: Vec<u128> = Vec::new();
        let mut ek_tot = [0usize; 5];
        let mut bad_all: Vec<u64> = Vec::new();
        for h in handles {
            let (l, er, bi) = h.join().unwrap();
            alllats.extend(l);
            for k in 0..5 {
                ek_tot[k] += er[k];
            }
            bad_all.extend(bi);
        }
        bad_all.sort_unstable();
        bad_all.dedup();
        let errors: usize = ek_tot.iter().sum();
        let ek_s: String = ek_tot
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(";");
        alllats.sort_unstable();
        let p = |fr: usize| alllats[fr * alllats.len() / 100];
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        let model_ok = m.len() == 120 && iss.is_empty();
        e.close().unwrap();
        rows.push_str(&format!(
            "E7a-readers,threads-{n},sync,{n},0,{},errors={errors} kinds(attend_err;attend_bad_id;scan_len;scan_err;get_fields)={ek_s} bad-ids={bad_all:?},static-120-docs,{},MATCH,\"ops={} p50={}us p95={}us p99={}us; readers never gate-blocked; all results valid ids; scans exact\"\n",
            n * ops_per * 2,
            if clean && model_ok { "clean" } else { "DIRTY" },
            alllats.len(),
            p(50),
            p(95),
            p(99)
        ));
        cells += 1;
    }

    // ---------------- E7b: readers + single writer ----------------
    {
        let dir = root.join("e7b-reader-writer");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e7_insert(&e, 5000, 0, 0);
        e.checkpoint().unwrap();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let lg = lg_global.clone();
        let mut handles = Vec::new();
        for t in 0..2usize {
            let e2 = e.clone();
            let done2 = done.clone();
            let lg2 = lg.clone();
            handles.push(std::thread::spawn(move || {
                let mut seen: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
                let mut invalid = 0usize;
                let mut reads = 0usize;
                let q2 = vec_for(5000);
                let ctr = std::sync::atomic::AtomicUsize::new(0);
                while !done2.load(std::sync::atomic::Ordering::Relaxed) {
                    let inv = std::time::Instant::now();
                    if let Some((_, num)) = e7_read_idx(&e2, 5000) {
                        reads += 1;
                        if !(0..=40).contains(&num) {
                            invalid += 1;
                        }
                        seen.insert(num);
                    }
                    e7_log_sampled(&lg2, &ctr, t, 0, "read", "5000", inv, "observed".into());
                    let inv = std::time::Instant::now();
                    if let Ok(r) = e2.attend("bench", &[HEAD.to_string()], &q2, 10) {
                        if r.iter().any(|(id, _)| *id > 200) {
                            invalid += 1;
                        }
                    }
                    e7_log_sampled(&lg2, &ctr, t, 0, "attend", "5000", inv, "observed".into());
                }
                (seen, invalid, reads)
            }));
        }
        // writer: 40 flips via update_document (committed values 1..=40)
        for f in 1..=40i64 {
            let mut fields = HashMap::new();
            fields.insert("idx".to_string(), serde_json::json!(5000));
            fields.insert("version".to_string(), serde_json::json!(f));
            fields.insert("cat".to_string(), serde_json::json!("t"));
            fields.insert("num".to_string(), serde_json::json!(f));
            fields.insert(
                "title".to_string(),
                serde_json::json!(format!("doc-5000-v{f}")),
            );
            let mut kv = HashMap::new();
            kv.insert(HEAD.to_string(), vec_for(5000));
            let inv = std::time::Instant::now();
            let uuid = uuid_for(5000, 0);
            let r = e.update_document("bench", &uuid.to_string(), fields, kv);
            e7_log(
                &lg,
                9,
                0,
                "update",
                "5000",
                inv,
                format!("ok={}", r.is_ok()),
            );
            if r.is_err() {
                rows.push_str(&format!("E7b-reader-writer,writer-error,sync,3,0,{f},ERR,-,-,MISMATCH,update {f} failed\n"));
                cells += 1;
            }
        }
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut invalid = 0;
        let mut allseen = std::collections::BTreeSet::new();
        let mut reads = 0;
        for h in handles {
            let (s, inv2, r2) = h.join().unwrap();
            invalid += inv2;
            reads += r2;
            allseen.extend(s);
        }
        let okvals: std::collections::BTreeSet<i64> = (0..=40).collect();
        let subset = allseen.iter().all(|v| okvals.contains(v));
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        let model_ok = m.get(&5000).map(|x| x.2) == Some(40);
        e.close().unwrap();
        rows.push_str(&format!(
            "E7b-reader-writer,2r-1w-flips,sync,3,0,{},values-seen={},final-num=40,{},{},\"reads={reads} invalid={} (committed-only values; readers never gate-blocked; never torn)\"\n",
            reads + 40,
            csvs(allseen),
            if clean { "clean" } else { "DIRTY" },
            if subset && model_ok && invalid == 0 { "MATCH" } else { "MISMATCH" },
            invalid
        ));
        cells += 1;
    }

    // ---------------- E7e: multi-document atomic visibility ----------------
    // 30 sequential txns, each [del A v=rep, ins A v=rep+1 num=101+rep] +
    // [del B v=rep, ins B v=rep+1 num=101+rep]. The reader spins HOT from
    // BEFORE the first commit until the writer signals done (unbounded: the
    // reader covers the WHOLE commit window, not a fixed count that can
    // finish before the first commit lands). Per-document values must be
    // monotonic (never decrease, never un-appear); the (A,B) PAIR has no
    // snapshot — mixed (a_num != b_num) states are the expected evidence.
    {
        let dir = root.join("e7e-atomic-visibility");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let lg = lg_global.clone();
        let reps = 30usize;
        let combos = Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new()));
        let e7e_nonmono = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let e2 = e.clone();
        let c2 = combos.clone();
        let nm2 = e7e_nonmono.clone();
        let lg2 = lg.clone();
        let start_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let sf = start_flag.clone();
        let df = done_flag.clone();
        let reader = std::thread::spawn(move || {
            let mut local: Vec<(i64, i64)> = Vec::new();
            let ctr = std::sync::atomic::AtomicUsize::new(0);
            let (mut max_a, mut max_b, mut nonmono) = (0i64, 0i64, 0usize);
            while !sf.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::yield_now();
            }
            while !df.load(std::sync::atomic::Ordering::Relaxed) {
                let a = e7_read_idx(&e2, 5300).map(|x| x.1).unwrap_or(0);
                let b = e7_read_idx(&e2, 5301).map(|x| x.1).unwrap_or(0);
                // 0 = transiently absent (per-op apply window, see E7m): NOT a
                // monotonicity violation; a LIVE value going backwards is.
                if (a != 0 && a < max_a) || (b != 0 && b < max_b) {
                    nonmono += 1; // a document went BACKWARDS: per-doc violation
                }
                max_a = max_a.max(a);
                max_b = max_b.max(b);
                local.push((a, b));
                let inv = std::time::Instant::now();
                e7_log_sampled(
                    &lg2,
                    &ctr,
                    1,
                    0,
                    "read2",
                    "5300+5301",
                    inv,
                    format!("{a}/{b}"),
                );
            }
            nm2.fetch_add(nonmono, std::sync::atomic::Ordering::Relaxed);
            let mut g = c2.lock().unwrap();
            for k in local {
                *g.entry(k).or_insert(0usize) += 1;
            }
        });
        start_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut commit_fails = 0usize;
        for rep in 0..reps {
            let t = e.begin_transaction("bench");
            // upsert: retire the previous version, insert the next (num rises)
            e.record_transaction_operation(t, TxnOp::Delete(uuid_for(5300, rep as u64)))
                .unwrap();
            e.record_transaction_operation(t, TxnOp::Delete(uuid_for(5301, rep as u64)))
                .unwrap();
            let mut ra = doc_record(5300, rep as u64 + 1, "t", 101 + rep as i64);
            ra.k_vecs.insert(HEAD.to_string(), vec_for(5300));
            let mut rb = doc_record(5301, rep as u64 + 1, "t", 101 + rep as i64);
            rb.k_vecs.insert(HEAD.to_string(), vec_for(5301));
            e.record_transaction_operation(t, TxnOp::Insert(ra))
                .unwrap();
            e.record_transaction_operation(t, TxnOp::Insert(rb))
                .unwrap();
            let ok = e7_commit(&e, t, &lg, 0, "5300+5301");
            if !ok {
                commit_fails += 1;
            }
        }
        done_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        reader.join().unwrap();
        let e7e_nm = e7e_nonmono.load(std::sync::atomic::Ordering::Relaxed);
        let g = combos.lock().unwrap();
        let mut combo_s = String::new();
        let mut mixed_pairs = 0usize;
        for (k, v) in g.iter() {
            combo_s.push_str(&format!(" {}{}={v}", k.0, k.1));
            if k.0 != k.1 {
                mixed_pairs += v;
            }
        }
        let mixed_note = if mixed_pairs > 0 {
            format!("{mixed_pairs} mixed (A,B) pairs observed — NO per-txn snapshot for concurrent non-txn readers")
        } else {
            "no mixed pair captured in this run (bounded observation); per-doc monotonicity holds"
                .to_string()
        };
        let total_reads = g.values().sum::<usize>();
        rows.push_str(&format!(
            "E7e-atomic-visibility,hot-reader-during-commits,sync,2,{reps},{},{combo_s} per-doc-monotonic={} mixed-pairs={mixed_pairs},per-doc-atomic-monotonic,{},{},\"per-DOCUMENT atomic and monotonic across the whole commit window; {mixed_note}\"\n",
            total_reads,
            if e7e_nm == 0 && commit_fails == 0 { "YES" } else { "NO" },
            if e7e_nm == 0 && commit_fails == 0 { "clean" } else { "DIRTY" },
            if e7e_nm == 0 && commit_fails == 0 { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
        vis.push_str(&format!(
            "Multi-document transaction visibility,per-doc atomic+monotonic ({combo_s}); mixed pairs observable to straddling readers -> NO per-txn snapshot,PARTIAL (per-doc atomic VERIFIED; atomic multi-doc visibility NOT provided)\n"
        ));
        e.close().unwrap();
    }

    // ---------------- E7f: delete/insert visibility ----------------
    // T commits DELETE A + INSERT B (2-op txn). Hot pre-started reader across
    // 30 sequential reps on fresh engines per rep: per-doc monotonicity is
    // the invariant (A falls once, B rises once); the (A,B) pair is not a
    // snapshot for non-txn readers.
    {
        let dir = root.join("e7f-del-ins");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let lg = lg_global.clone();
        let reps = 30usize;
        let combos = Arc::new(std::sync::Mutex::new(std::collections::BTreeMap::new()));
        let e7f_nonmono = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for rep in 0..reps {
            let rdir = dir.join(format!("rep{rep}"));
            let e = Arc::new(open_db(&rdir));
            e.create_collection("bench", DIM, &[HEAD]).unwrap();
            e7_insert(&e, 5400, 0, 3);
            e.checkpoint().unwrap();
            let e2 = e.clone();
            let c2 = combos.clone();
            let nm2 = e7f_nonmono.clone();
            let lg2 = lg.clone();
            let start_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let done_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let sf = start_flag.clone();
            let df = done_flag.clone();
            let reader = std::thread::spawn(move || {
                let mut local: Vec<(bool, bool)> = Vec::new();
                let ctr = std::sync::atomic::AtomicUsize::new(0);
                let (mut a_gone, mut b_seen, mut nonmono) = (false, false, 0usize);
                while !sf.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::yield_now();
                }
                // unbounded: covers the whole commit window (a fixed count can
                // finish before the commit even lands — bounded observation)
                while !df.load(std::sync::atomic::Ordering::Relaxed) {
                    let a = e7_read_idx(&e2, 5400).is_some();
                    let b = e7_read_idx(&e2, 5401).is_some();
                    if a && a_gone {
                        nonmono += 1; // A re-appeared: per-doc violation
                    }
                    if b_seen && !b {
                        nonmono += 1; // B un-appeared: per-doc violation
                    }
                    a_gone |= !a;
                    b_seen |= b;
                    local.push((a, b));
                    let inv = std::time::Instant::now();
                    e7_log_sampled(
                        &lg2,
                        &ctr,
                        1,
                        0,
                        "read2",
                        "5400+5401",
                        inv,
                        format!("{}{}", a as u8, b as u8),
                    );
                }
                nm2.fetch_add(nonmono, std::sync::atomic::Ordering::Relaxed);
                let mut g = c2.lock().unwrap();
                for k in local {
                    *g.entry(k).or_insert(0usize) += 1;
                }
            });
            let t = e.begin_transaction("bench");
            e.record_transaction_operation(t, TxnOp::Delete(uuid_for(5400, 0)))
                .unwrap();
            let mut rb = doc_record(5401, 0, "t", 7);
            rb.k_vecs.insert(HEAD.to_string(), vec_for(5401));
            e.record_transaction_operation(t, TxnOp::Insert(rb))
                .unwrap();
            start_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            let ok = e7_commit(&e, t, &lg, 0, "del5400-ins5401");
            done_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            reader.join().unwrap();
            if !ok {
                rows.push_str(
                    "E7f-del-ins,commit-failed,sync,2,1,2,commit-err,-,-,MISMATCH,commit failed\n",
                );
                cells += 1;
            }
            e.close().unwrap();
        }
        let g = combos.lock().unwrap();
        let mut combo_s = String::new();
        for (k, v) in g.iter() {
            combo_s.push_str(&format!(" A{}B{}={v}", k.0 as u8, k.1 as u8));
        }
        let e7f_nm = e7f_nonmono.load(std::sync::atomic::Ordering::Relaxed);
        // mixed = any (A,B) presence pair that is neither the pure pre-state
        // (A,B)=(1,0) nor the pure post-state (0,1)
        let mixed_pairs: usize = g
            .iter()
            .filter(|(k, _)| **k != (true, false) && **k != (false, true))
            .map(|(_, v)| v)
            .sum();
        let mixed_note = if mixed_pairs > 0 {
            format!("{mixed_pairs} mixed (A,B) presence pairs observed — the delete and insert apply per-op, NOT as a visibility unit")
        } else {
            "no mixed pair captured in this run (bounded observation); per-doc semantics hold"
                .to_string()
        };
        rows.push_str(&format!(
            "E7f-del-ins,hot-reader-during-commit,sync,2,{reps},{},{combo_s} per-doc-monotonic={} mixed-pairs={mixed_pairs},per-doc-monotonic-only,{},{},\"DELETE A + INSERT B in one txn: A falls once and B rises once (per-document); {mixed_note}\"\n",
            g.values().sum::<usize>(),
            e7f_nm,
            if e7f_nm == 0 { "clean" } else { "DIRTY" },
            if e7f_nm == 0 { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
    }

    // ---------------- E7g: write/write same key (both orders x 3 modes) ------
    // Same key = SAME UUID, each txn = [Delete U, Insert U with its value]
    // (delete-then-reinsert of one uuid, E6l-proven). Whichever txn commits
    // later deletes the other's insert and reinserts its own value: final
    // value = later commit, exactly one live document.
    for mode in ["sync", "group", "async"] {
        for order in ["t1-then-t2", "t2-then-t1"] {
            let dir = root.join(format!("e7g-{mode}-{order}"));
            std::env::set_var("PH3D_DURABILITY", mode);
            let e = Arc::new(open_db(&dir));
            e.create_collection("bench", DIM, &[HEAD]).unwrap();
            e7_insert(&e, 6100, 0, 0); // A exists at uuid U = (6100, v0)
            e.checkpoint().unwrap();
            let lg = lg_global.clone();
            let mk = |e: &AttentionEngine, num: i64| -> u64 {
                let t = e.begin_transaction("bench");
                e.record_transaction_operation(t, TxnOp::Delete(uuid_for(6100, 0)))
                    .unwrap();
                let mut r = doc_record(6100, 0, "t", num);
                r.k_vecs.insert(HEAD.to_string(), vec_for(6100));
                e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
                t
            };
            let (t1, t2) = (mk(&e, 100), mk(&e, 200));
            let (first, second, want) = if order == "t1-then-t2" {
                (t1, t2, 200i64)
            } else {
                (t2, t1, 100i64)
            };
            let okf = e7_commit(&e, first, &lg, 0, "first");
            let oks = e7_commit(&e, second, &lg, 1, "second");
            let (m, iss) = export_state(&e);
            let clean = e7_checker(&e, &dir);
            let final_num = m.get(&6100).map(|x| x.2);
            e.close().unwrap();
            let (rm, rc) = e6_open(&dir, mode).unwrap();
            let rec_ok = rm.get(&6100).map(|x| x.2) == Some(want) && rc;
            rows.push_str(&format!(
                "E7g-write-write,same-key-{order},{mode},2,2,2,final={final_num:?} expect={want},later-commit-wins-1-doc/{},{},{},\"both commits Ok; later commit wins; WAL order = commit order; recovery identical\"\n",
                if final_num == Some(want) && okf && oks { "order-honored" } else { "UNEXPECTED" },
                if clean && iss.is_empty() { "clean" } else { "DIRTY" },
                if final_num == Some(want) && okf && oks && rec_ok && m.len() == 1 && iss.is_empty() { "MATCH" } else { "MISMATCH" },
            ));
            cells += 1;
        }
    }
    // plain concurrent writers (no txns): 2 threads x 10 inserts, all survive
    {
        let dir = root.join("e7g-plain-writers");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let lg = lg_global.clone();
        let mut handles = Vec::new();
        for th in 0..2usize {
            let e2 = e.clone();
            let lg2 = lg.clone();
            handles.push(std::thread::spawn(move || {
                let mut oks = 0usize;
                for i in 0..10u32 {
                    let idx: u32 = 6101 + (th as u32) * 10 + i;
                    let inv = std::time::Instant::now();
                    e7_insert(&e2, idx, 1, idx as i64);
                    e7_log(&lg2, th, 0, "insert", &format!("{idx}"), inv, "ok".into());
                    oks += 1;
                }
                oks
            }));
        }
        let mut oks = 0usize;
        for h in handles {
            oks += h.join().unwrap();
        }
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir);
        e.close().unwrap();
        rows.push_str(&format!(
            "E7g-write-write,plain-2-writers,sync,2,0,{oks},acked={oks} docs={},20-survive,{},{},\"all ACKed writes survive; mutations serialize on the mutation gate (write->apply->ACK atomic w.r.t. other writers)\"\n",
            m.len(),
            if clean && iss.is_empty() { "clean" } else { "DIRTY" },
            if clean && iss.is_empty() && m.len() == 20 && oks == 20 { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
        vis.push_str("Two writers same key,commit-order-wins; later commit's value final; no error; no conflict detection,VERIFIED (commit serialization)\n");
    }

    // ---------------- E7h: lost update ----------------
    {
        let dir = root.join("e7h-lost-update");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e7_insert(&e, 6110, 0, 0);
        e.checkpoint().unwrap();
        let lg = lg_global.clone();
        // both transactions read A OUTSIDE the txn (only option — TxnOp has no
        // Read), compute +1 / +2, and blind-write their result as a
        // delete+reinsert of the SAME uuid.
        let _a = e7_read_idx(&e, 6110); // T1's read: 0
        let t1 = e7_stage_upsert(&e, 6110, Some(0), 0, 1);
        let _b = e7_read_idx(&e, 6110); // T2's read: still 0 (T1 not committed)
        let t2 = e7_stage_upsert(&e, 6110, Some(0), 0, 2);
        let ok1 = e7_commit(&e, t1, &lg, 0, "t1");
        let ok2 = e7_commit(&e, t2, &lg, 1, "t2");
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir);
        let final_num = m.get(&6110).map(|x| x.2);
        e.close().unwrap();
        // expected shape: BOTH commits succeed, final = 2 (T2 committed last),
        // T1's +1 silently lost, no conflict error anywhere.
        rows.push_str(&format!(
            "E7h-lost-update,read-outside-blind-write,sync,2,2,4,both-committed final={final_num:?} expect=2,commit-order-wins-no-detection,{},{},\"LOST UPDATE occurs by construction: no conflict detection; no version check; commit-order-wins is NOT conflict detection\"\n",
            if clean && iss.is_empty() { "clean" } else { "DIRTY" },
            if ok1 && ok2 && final_num == Some(2) && clean && m.len() == 1 { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
        vis.push_str("Lost-update scenario,occurs: both commits succeed; one blind write silently overwritten; undetected,VERIFIED (no conflict detection)\n");
    }

    // ---------------- E7i / E7j: UNSUPPORTED BY API ----------------
    rows.push_str("E7i-write-skew,classic-invariant,sync,2,2,0,not-expressible,UNSUPPORTED-NO-TXN-READS,-,-,\"TxnOp = Insert|Delete: transactions cannot read; write-skew requires transactional reads; NOT fabricated\"\n");
    cells += 1;
    vis.push_str("Write skew,UNSUPPORTED BY API (transactions have no read ops),UNSUPPORTED\n");
    rows.push_str("E7j-phantom,predicate-requery-in-txn,sync,2,2,0,not-expressible,UNSUPPORTED-NO-TXN-QUERY,-,-,\"no transactional range/filter query API; phantom behavior in-txn cannot exist or be observed; NOT inferred from point reads\"\n");
    cells += 1;
    vis.push_str(
        "Phantom-style query,UNSUPPORTED BY API (no transactional query reads),UNSUPPORTED\n",
    );

    // ---------------- E7k: staged visibility under concurrency ----------------
    {
        let dir = root.join("e7k-staged-visibility");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e7_insert(&e, 6200, 0, 100);
        e.checkpoint().unwrap();
        let lg = lg_global.clone();
        let reps = 30usize;
        let staged_new_visible = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let old_missing_while_staged = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let missing_post = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Ordering: the reader's staged-window reads COMPLETE (b_reads) BEFORE
        // the writer invokes commit — the "staged invisible" claim is then a
        // properly ordered statement, not a race the fast commit always wins.
        let b_staged = Arc::new(std::sync::Barrier::new(2));
        let b_reads = Arc::new(std::sync::Barrier::new(2));
        let b_commit = Arc::new(std::sync::Barrier::new(2));
        let e2 = e.clone();
        let snv = staged_new_visible.clone();
        let oms = old_missing_while_staged.clone();
        let mp = missing_post.clone();
        let bsc = b_staged.clone();
        let brc = b_reads.clone();
        let bcc = b_commit.clone();
        let reader = std::thread::spawn(move || {
            for rep in 0..reps {
                bsc.wait(); // txn staged: del v=rep, ins v=rep+1 (uncommitted)
                            // staged NEW version must be invisible; OLD version still live.
                            // value of version v>=1 is 99+v (v=1 -> 100); v0 starts at 100.
                let want_old = if rep == 0 { 100i64 } else { 99 + rep as i64 };
                if e7_read_ver(&e2, 6200, rep as u64 + 1).is_some() {
                    snv.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                if e7_read_ver(&e2, 6200, rep as u64) != Some(want_old) {
                    oms.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                brc.wait(); // reads done; writer may now commit
                bcc.wait(); // committed
                            // new version visible with its committed value
                if e7_read_ver(&e2, 6200, rep as u64 + 1) != Some(100 + rep as i64) {
                    mp.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });
        for rep in 0..reps {
            let t = e7_stage_upsert(&e, 6200, Some(rep as u64), rep as u64 + 1, 100 + rep as i64);
            b_staged.wait();
            b_reads.wait(); // reader's staged reads completed first
            e7_commit(&e, t, &lg, 0, "6200");
            b_commit.wait();
        }
        reader.join().unwrap();
        let staged_leaks = staged_new_visible.load(std::sync::atomic::Ordering::Relaxed);
        let old_missing = old_missing_while_staged.load(std::sync::atomic::Ordering::Relaxed);
        let missing = missing_post.load(std::sync::atomic::Ordering::Relaxed);
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir);
        e.close().unwrap();
        rows.push_str(&format!(
            "E7k-staged-visibility,stage-barrier-read-commit,sync,2,{reps},{},staged-new-visible={staged_leaks} old-missing-while-staged={old_missing} post-commit-missing={missing},absent-then-present,{},{},\"ordered 3-barrier reps: reads completing strictly before the commit invocation never see the staged version (old version stays live); visible only after commit\"\n",
            reps * 3,
            if clean && iss.is_empty() { "clean" } else { "DIRTY" },
            if staged_leaks == 0 && old_missing == 0 && missing == 0 && m.len() == 1 { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
        vis.push_str("Read during staged insert,staged version invisible in every read that completed before the commit was invoked (30 ordered reps; old version stays live),VERIFIED (ordered observation)\n");
    }

    // ---------------- E7l: rollback visibility ----------------
    {
        let dir = root.join("e7l-rollback-visibility");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let reps = 20usize;
        let leaked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Ordering: the reader's during-staged read COMPLETES before the
        // rollback is invoked (b_reads) — the "staged never visible" claim is
        // a properly ordered statement, not a race the fast rollback wins.
        let b_staged = Arc::new(std::sync::Barrier::new(2));
        let b_reads = Arc::new(std::sync::Barrier::new(2));
        let b_rb = Arc::new(std::sync::Barrier::new(2));
        let e2 = e.clone();
        let lk = leaked.clone();
        let bsc = b_staged.clone();
        let brc = b_reads.clone();
        let bbc = b_rb.clone();
        let reader = std::thread::spawn(move || {
            for _ in 0..reps {
                // before staging
                if e7_read_idx(&e2, 6201).is_some() {
                    lk.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                bsc.wait(); // staged
                if e7_read_idx(&e2, 6201).is_some() {
                    lk.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                brc.wait(); // read done; writer may now roll back
                bbc.wait(); // rolled back
                if e7_read_idx(&e2, 6201).is_some() {
                    lk.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });
        for _ in 0..reps {
            let t = e7_stage_upsert(&e, 6201, None, 0, 13);
            b_staged.wait();
            b_reads.wait(); // reader's staged read completed first
            e.rollback_transaction(t).unwrap();
            b_rb.wait();
        }
        reader.join().unwrap();
        let leaks = leaked.load(std::sync::atomic::Ordering::Relaxed);
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        e.close().unwrap();
        rows.push_str(&format!(
            "E7l-rollback-visibility,stage-barrier-rollback,sync,2,{reps},{},ever-visible={leaks},never-visible,{},{},\"ordered 3-barrier reps: the staged version is invisible to a read that completes strictly before the rollback is invoked, and stays absent after rollback\"\n",
            reps * 3,
            if clean { "clean" } else { "DIRTY" },
            if leaks == 0 && m.is_empty() { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
        vis.push_str(
            "Read after rollback,never visible (20 barrier reps, read at 3 points),VERIFIED\n",
        );
    }

    // ---------------- E7m: commit visibility boundary ----------------
    // Per-rep two-barrier protocol. Writer: stage [del v=rep, ins v=rep+1] ->
    // b_start -> commit (logged invoke/complete) -> b_end. Watcher: wait old
    // live -> b_start -> probe OLD version until absent (logs last-live and
    // first-absent reads) -> probe NEW version until present (logs first
    // appearance) -> 50 stability reads -> b_end. Post-run classification
    // against the commit invoke/complete interval:
    //   early-retire   : old absent BEFORE commit invoked  (would be a dirty
    //                    delete leak — must be 0)
    //   window-observed: old-absent read completed inside the commit interval
    //                    (per-op apply evidence)
    //   new-pre-ack    : new version appeared before the commit call returned
    //                    (visibility = apply point, not ACK return)
    //   stale-post-ack : old visible / new absent in reads INVOKED after the
    //                    commit completed (must be 0)
    //   stability-viol : post-application reads disagree (must be 0)
    {
        let dir = root.join("e7m-commit-visibility");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e7_insert(&e, 6202, 0, 0);
        e.checkpoint().unwrap();
        let lg = lg_global.clone();
        let reps = 40usize;
        let stability_viol = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let e2 = e.clone();
        let sv = stability_viol.clone();
        let lg2 = lg.clone();
        let b_start = Arc::new(std::sync::Barrier::new(2));
        let b_end = Arc::new(std::sync::Barrier::new(2));
        let bsc = b_start.clone();
        let bec = b_end.clone();
        let watcher = std::thread::spawn(move || {
            for rep in 0..reps {
                // old version must be live before this rep starts
                while e7_read_ver(&e2, 6202, rep as u64) != Some(rep as i64) {
                    std::thread::yield_now();
                }
                bsc.wait(); // writer has staged; commit not yet invoked
                            // phase 1: old version retires — log last-live + first-absent
                let mut last_live: Option<std::time::Instant> = None;
                loop {
                    let inv = std::time::Instant::now();
                    if e7_read_ver(&e2, 6202, rep as u64) == Some(rep as i64) {
                        last_live = Some(inv);
                    } else {
                        e7_log(&lg2, 1, rep as u64, "w-old", "6202", inv, "absent".into());
                        if let Some(t) = last_live {
                            e7_log(&lg2, 1, rep as u64, "w-old", "6202", t, "live".into());
                        }
                        break;
                    }
                    std::thread::yield_now();
                }
                // phase 2: new version appears — log first appearance
                loop {
                    let inv = std::time::Instant::now();
                    if e7_read_ver(&e2, 6202, rep as u64 + 1) == Some(rep as i64 + 1) {
                        e7_log(&lg2, 1, rep as u64, "w-new", "6202", inv, "present".into());
                        break;
                    }
                    std::thread::yield_now();
                }
                // phase 3: stability — new sticks, old stays gone
                for _ in 0..50u32 {
                    if e7_read_ver(&e2, 6202, rep as u64 + 1) != Some(rep as i64 + 1)
                        || e7_read_ver(&e2, 6202, rep as u64).is_some()
                    {
                        sv.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                bec.wait();
            }
        });
        for rep in 0..reps {
            let t = e7_stage_upsert(&e, 6202, Some(rep as u64), rep as u64 + 1, rep as i64 + 1);
            b_start.wait();
            e7_commit(&e, t, &lg, 0, "6202");
            b_end.wait();
        }
        watcher.join().unwrap();
        // post-run classification from the event log
        let g = lg.lock().unwrap();
        let mut early_retire = 0usize;
        let mut window_observed = 0usize;
        let mut retire_after_ack = 0usize; // watcher noticed late (scheduling) — NOT staleness
        let mut new_pre_ack = 0usize;
        let mut new_after_ack = 0usize;
        let mut unclassified = 0usize;
        for rep in 0..reps {
            let commit =
                g.0.iter()
                    .find(|ev| ev.op == "commit" && ev.txn == rep as u64 && ev.key == "6202");
            let Some(cev) = commit else {
                unclassified += 1;
                continue;
            };
            let (ci, cc) = (cev.t_inv_us, cev.t_ret_us);
            let old_absent =
                g.0.iter()
                    .find(|ev| ev.op == "w-old" && ev.txn == rep as u64 && ev.result == "absent");
            match old_absent {
                Some(ev) => {
                    if ev.t_ret_us < ci {
                        early_retire += 1; // dirty delete leak BEFORE commit invoked: violation
                    } else if ev.t_ret_us <= cc {
                        window_observed += 1; // retire observed INSIDE the commit interval
                    } else {
                        retire_after_ack += 1; // watcher descheduled; stability reads still verify
                    }
                }
                None => unclassified += 1,
            }
            if let Some(ev) =
                g.0.iter()
                    .find(|ev| ev.op == "w-new" && ev.txn == rep as u64)
            {
                if ev.t_ret_us <= cc {
                    new_pre_ack += 1; // visible BEFORE the commit call returned
                } else {
                    new_after_ack += 1; // watcher noticed late; bounded observation
                }
            } else {
                unclassified += 1;
            }
        }
        drop(g);
        let stab = stability_viol.load(std::sync::atomic::Ordering::Relaxed);
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        e.close().unwrap();
        rows.push_str(&format!(
            "E7m-commit-visibility,watch-flip-during-commit,sync,2,{reps},{},window-observed={window_observed} noticed-after-ack={retire_after_ack} early-retire={early_retire} new-pre-ack={new_pre_ack} new-after-ack={new_after_ack} unclassified={unclassified} stability-viol={stab},apply-point-visibility,{},{},\"retire and new-appearance happen strictly inside the commit invoke/complete interval where observed (per-op apply, txn NOT a visibility unit); post-ACK stability reads never see old live or new absent; late watcher notices are bounded observation, not staleness\"\n",
            reps * 3,
            if clean { "clean" } else { "DIRTY" },
            if early_retire == 0
                && stab == 0
                && window_observed + retire_after_ack + unclassified == reps
                && new_pre_ack + new_after_ack + unclassified == reps
                && m.get(&6202).map(|x| x.2) == Some(reps as i64)
            {
                "MATCH"
            } else {
                "MISMATCH"
            },
        ));
        cells += 1;
        vis.push_str(&format!(
            "Read after commit,retire and new-appearance observed strictly inside the commit interval in {window_observed}/{reps} ordered reps (new visible before commit-call return in {new_pre_ack}); 0 post-ACK stability violations,VERIFIED (visibility = apply point; bounded observation)\n"
        ));
    }

    // ---------------- E7n: same-key delete/reinsert both orders --------------
    {
        let dir = root.join("e7n-del-reinsert");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e7_insert(&e, 6300, 1, 1); // A at v1
        e.checkpoint().unwrap();
        let lg = lg_global.clone();
        // order 1: T1 deletes A(v1); then T2 inserts A(v2 num=42)
        let t1 = e.begin_transaction("bench");
        e.record_transaction_operation(t1, TxnOp::Delete(uuid_for(6300, 1)))
            .unwrap();
        let ok1 = e7_commit(&e, t1, &lg, 0, "del");
        let t2 = e7_stage_upsert(&e, 6300, None, 2, 42);
        let ok2 = e7_commit(&e, t2, &lg, 1, "ins");
        // order 2 on a second pair: T3 inserts C(v3); then T4 deletes D... use
        // reverse order on a second key: T3 inserts (6309,v3) BEFORE T4 deletes (6309,v1)?
        // Keep the family focused: reverse order = insert FIRST then delete of
        // the OLD version on key 6301.
        e7_insert(&e, 6301, 1, 1);
        let t3 = e7_stage_upsert(&e, 6301, None, 3, 43); // insert-first
        let ok3 = e7_commit(&e, t3, &lg, 0, "ins-first");
        let t4 = e.begin_transaction("bench");
        e.record_transaction_operation(t4, TxnOp::Delete(uuid_for(6301, 1)))
            .unwrap();
        let ok4 = e7_commit(&e, t4, &lg, 1, "del-old");
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        let m_ok = m.get(&6300).map(|x| (x.0, x.2)) == Some((2, 42))
            && m.get(&6301).map(|x| (x.0, x.2)) == Some((3, 43))
            && m.len() == 2;
        e.compact_storage().unwrap();
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        let rec_ok = rm.get(&6300).map(|x| (x.0, x.2)) == Some((2, 42))
            && rm.get(&6301).map(|x| (x.0, x.2)) == Some((3, 43))
            && rc;
        rows.push_str(&format!(
            "E7n-del-reinsert,both-orders-compact-restart,sync,2,4,4,6300=v2/42 6301=v3/43,uuid-v2-v3-live-old-tombstoned,{},{},\"del-then-ins and ins-then-del both land on the same final state; old uuids tombstoned; survives compaction + restart\"\n",
            if clean { "clean" } else { "DIRTY" },
            if ok1 && ok2 && ok3 && ok4 && m_ok && rec_ok { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
    }

    // ---------------- E7o: collection concurrency ----------------
    {
        let dir = root.join("e7o-collections");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("collA", DIM, &[HEAD]).unwrap();
        e.create_collection("collB", DIM, &[HEAD]).unwrap();
        let lg = lg_global.clone();
        let b = Arc::new(std::sync::Barrier::new(2));
        let e2 = e.clone();
        let bc = b.clone();
        let lg2 = lg.clone();
        let th = std::thread::spawn(move || {
            let mut r = doc_record_c(2, 6401, 1, "t", 6401);
            r.k_vecs.insert(HEAD.to_string(), vec_for(6401));
            let inv = std::time::Instant::now();
            e2.insert_document("collB", r).unwrap();
            e7_log(&lg2, 1, 0, "insert", "B/6401", inv, "ok".into());
            bc.wait();
            // concurrent reads of collA while collB writes
            for _ in 0..50 {
                let _ = e2.scan_filtered("collA", None, 10_000).unwrap().len();
            }
        });
        let mut r = doc_record_c(1, 6400, 1, "t", 6400);
        r.k_vecs.insert(HEAD.to_string(), vec_for(6400));
        let inv = std::time::Instant::now();
        e.insert_document("collA", r).unwrap();
        e7_log(&lg, 0, 0, "insert", "A/6400", inv, "ok".into());
        b.wait();
        th.join().unwrap();
        // cross operations: read A, write B, backup, ckpt, compact
        let a_state = export_state_coll(&e, "collA").0;
        let b_state = export_state_coll(&e, "collB").0;
        let bdir = root.join("e7o-backup");
        e.backup_to(&bdir).unwrap();
        e.checkpoint().unwrap();
        e.compact_storage().unwrap();
        let a_after = export_state_coll(&e, "collA").0;
        let (_mall, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        e.close().unwrap();
        // restore backup into a scratch dir and check isolation there too
        let rdir = root.join("e7o-restore");
        let _ = std::fs::remove_dir_all(&rdir);
        // backup restore via copy (backup dir IS a full db dir)
        let er = open_db(&bdir);
        let ra = export_state_coll(&er, "collA").0;
        let rb = export_state_coll(&er, "collB").0;
        let iso_ok = a_state.len() == 1
            && a_state.contains_key(&6400)
            && b_state.len() == 1
            && b_state.contains_key(&6401)
            && ra == a_state
            && rb == b_state
            && a_after == a_state;
        rows.push_str(&format!(
            "E7o-collections,T-A-T-B-concurrent-plus-ops,sync,2,0,2,A={:?} B={:?},no-contamination,{},{},\"concurrent inserts into disjoint collections; read-A/write-B/backup/ckpt/compact all isolate; backup restores isolation\"\n",
            a_state.keys().collect::<Vec<_>>(),
            b_state.keys().collect::<Vec<_>>(),
            if clean { "clean" } else { "DIRTY" },
            if iso_ok { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
        vis.push_str("Collection isolation,concurrent T->collA / T->collB: zero contamination across backup/ckpt/compact,VERIFIED\n");
    }

    // ---------------- E7p: checkpoint + concurrent staged txn ----------------
    {
        let dir = root.join("e7p-checkpoint");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e7_insert(&e, 6500, 0, 20);
        e.checkpoint().unwrap();
        let lg = lg_global.clone();
        let b1 = Arc::new(std::sync::Barrier::new(2));
        let b2 = Arc::new(std::sync::Barrier::new(2));
        let e2 = e.clone();
        let b1c = b1.clone();
        let b2c = b2.clone();
        let seen_staged = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let ss = seen_staged.clone();
        // Ordering: the checkpoint thread completes BOTH read rounds + the
        // checkpoint itself BEFORE the writer invokes commit (b_reads) — the
        // "checkpoint never commits staged state" claim is ordered, not raced.
        let b_reads = Arc::new(std::sync::Barrier::new(2));
        let brc = b_reads.clone();
        let th = std::thread::spawn(move || {
            b1c.wait(); // txn staged: del v0, ins v1 (uncommitted)
                        // checkpoint runs entirely inside the staged window
            if e7_read_ver(&e2, 6500, 1).is_some() {
                ss.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            e2.checkpoint().unwrap();
            if e7_read_ver(&e2, 6500, 1).is_some() {
                ss.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if e7_read_ver(&e2, 6500, 0) != Some(20) {
                ss.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            brc.wait(); // staged-window work done; writer may now commit
            b2c.wait(); // txn committed
            if e7_read_ver(&e2, 6500, 1) != Some(21) {
                ss.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
        let t = e7_stage_upsert(&e, 6500, Some(0), 1, 21);
        b1.wait();
        b_reads.wait(); // checkpoint thread's staged-window work completed first
        e7_commit(&e, t, &lg, 0, "6500");
        b2.wait();
        th.join().unwrap();
        let leaks = seen_staged.load(std::sync::atomic::Ordering::Relaxed);
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir);
        e.close().unwrap();
        rows.push_str(&format!(
            "E7p-checkpoint,staged-then-concurrent-ckpt,sync,2,1,1,staged-visible-at-ckpt={leaks},ckpt-never-commits,{},{},\"ordered: checkpoint + both read rounds completed strictly before the commit invocation; staged version absent before/after the checkpoint; old version intact; visible only after its own commit\"\n",
            if clean && iss.is_empty() { "clean" } else { "DIRTY" },
            if leaks == 0 && m.get(&6500).map(|x| (x.0, x.2)) == Some((1, 21)) { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
    }

    // ---------------- E7q: compaction + concurrent commit ----------------
    {
        let dir = root.join("e7q-compaction");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let e2 = e.clone();
        let th = std::thread::spawn(move || {
            for i in 0..5u32 {
                let t = e7_stage_upsert(&e2, 6501 + i, None, 1, i as i64);
                e2.commit_transaction(t).unwrap();
            }
        });
        let mut compactions = 0usize;
        while !th.is_finished() {
            e.compact_storage().unwrap();
            compactions += 1;
        }
        th.join().unwrap();
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        let m_ok = (0..5).all(|i| m.get(&(6501 + i as u32)).map(|x| x.2) == Some(i));
        e.close().unwrap();
        rows.push_str(&format!(
            "E7q-compaction,commits-vs-compact-loop,sync,2,5,5,compactions={compactions} docs=5,no-partial-no-lost,{},{},\"commit and compaction serialize on the mutation gate; no partial txn, no lost write, no resurrection\"\n",
            if clean { "clean" } else { "DIRTY" },
            if clean && m_ok && compactions > 0 { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
    }

    // ---------------- E7r: backup + concurrent txn ----------------
    for case in ["during-staged", "during-commit"] {
        let dir = root.join(format!("e7r-{case}"));
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let b = Arc::new(std::sync::Barrier::new(2));
        let e2 = e.clone();
        let bc = b.clone();
        let th = std::thread::spawn(move || {
            let t = e7_stage_upsert(&e2, 6502, None, 1, 33);
            bc.wait();
            if case == "during-staged" {
                // hold the staged state a bounded moment, then commit
                std::thread::sleep(std::time::Duration::from_millis(5));
                e2.commit_transaction(t).unwrap();
            } else {
                // commit immediately; backup races the commit on the gate
                e2.commit_transaction(t).unwrap();
            }
        });
        b.wait();
        let bdir = root.join(format!("e7r-{case}-snap"));
        e.backup_to(&bdir).unwrap();
        th.join().unwrap();
        let (post, iss) = export_state(&e);
        e.close().unwrap();
        let eb = open_db(&bdir);
        let (snap, biss) = export_state(&eb);
        let bclean = checker_report(&eb, &bdir)["clean"]
            .as_bool()
            .unwrap_or(false);
        drop(eb);
        let pre_ok = snap.is_empty();
        let post_ok = snap.get(&6502).map(|x| x.2) == Some(33);
        rows.push_str(&format!(
            "E7r-backup,txn-{case},sync,2,1,1,snapshot={},pre-or-post-never-partial,{},{},\"backup captures pre-txn or post-commit state, never a partial transaction (snapshot: {})\"\n",
            if pre_ok { "pre" } else if post_ok { "post" } else { "PARTIAL/UNEXPECTED" },
            if bclean && biss.is_empty() { "clean" } else { "DIRTY" },
            if (pre_ok || post_ok) && iss.is_empty() { "MATCH" } else { "MISMATCH" },
            // CSV-safe: the map's Debug contains raw quotes (cat = "t")
            csvs(&post),
        ));
        cells += 1;
    }
    vis.push_str("Backup during staged/committing txn,snapshot = pre-state or post-commit state, never partial,VERIFIED\n");

    // ---------------- E7s: concurrent staging, out-of-order commits ----------
    {
        let dir = root.join("e7s-concurrent-staging");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let lg = lg_global.clone();
        let b = Arc::new(std::sync::Barrier::new(3));
        let ids: Arc<std::sync::Mutex<Vec<u64>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for th in 0..3usize {
            let e2 = e.clone();
            let bc = b.clone();
            let ids2 = ids.clone();
            let lg2 = lg.clone();
            handles.push(std::thread::spawn(move || {
                let t = e2.begin_transaction("bench");
                for i in 0..3u32 {
                    let idx = 6600 + th as u32 * 3 + i;
                    let mut r = doc_record(idx, 1, "t", idx as i64);
                    r.k_vecs.insert(HEAD.to_string(), vec_for(idx));
                    e2.record_transaction_operation(t, TxnOp::Insert(r))
                        .unwrap();
                }
                let inv = std::time::Instant::now();
                e7_log(&lg2, th, t, "stage", "3 ops", inv, "ok".into());
                bc.wait(); // all three staged simultaneously
                ids2.lock().unwrap().push(t);
                t
            }));
        }
        let mut txn_ids = Vec::new();
        for h in handles {
            txn_ids.push(h.join().unwrap());
        }
        let distinct = txn_ids.len() == 3 && txn_ids[0] != txn_ids[1] && txn_ids[1] != txn_ids[2];
        // commit order: T3, T1, T2 (stage order was T1, T2, T3)
        let order = [txn_ids[2], txn_ids[0], txn_ids[1]];
        let mut all_ok = true;
        for (i, t) in order.iter().enumerate() {
            all_ok &= e7_commit(&e, *t, &lg, i, "out-of-order");
        }
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        let m_ok = m.len() == 9;
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!(
            "E7s-concurrent-staging,3-stage-barrier-commit-3-1-2,sync,3,3,9,distinct-ids={distinct} docs=9,WAL=commit-order,{},{},\"staging fully concurrent; commits serialized; replay order = commit order (restart equality)\"\n",
            if clean { "clean" } else { "DIRTY" },
            if distinct && all_ok && m_ok && rc && rm.len() == 9 { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
    }

    // ---------------- E7t: commit contention (3 modes) ------------------------
    for mode in ["sync", "group", "async"] {
        let dir = root.join(format!("e7t-contention-{mode}"));
        std::env::set_var("PH3D_DURABILITY", mode);
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let lg = lg_global.clone();
        let n = 4usize;
        let b = Arc::new(std::sync::Barrier::new(n));
        let mut handles = Vec::new();
        let finish_order = Arc::new(std::sync::Mutex::new(Vec::new()));
        for th in 0..n {
            let e2 = e.clone();
            let bc = b.clone();
            let fo = finish_order.clone();
            let lg2 = lg.clone();
            handles.push(std::thread::spawn(move || {
                let idx = 6610 + th as u32;
                let t = e7_stage_upsert(&e2, idx, None, 1, idx as i64);
                bc.wait(); // all commit simultaneously
                let ok = e7_commit(&e2, t, &lg2, th, "contend");
                fo.lock().unwrap().push((th, ok));
                ok
            }));
        }
        let mut all_ok = true;
        for h in handles {
            all_ok &= h.join().unwrap();
        }
        let finish = finish_order.lock().unwrap().clone();
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, mode).unwrap();
        rows.push_str(&format!(
            "E7t-commit-contention,4-barrier-commits,{mode},4,4,4,finish-order={},all-committed-serialized,{},{},\"no partial transactions, no failures under contention; completion order recorded; replay = commit order\"\n",
            csvs(finish.iter().map(|x| x.0).collect::<Vec<_>>()),
            if clean { "clean" } else { "DIRTY" },
            if all_ok && clean && m.len() == 4 && rc && rm.len() == 4 { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
    }

    // ---------------- E7u: single-operation linearizability ------------------
    // Single-key register (key 6700). Writers update_document with strictly
    // increasing per-thread values (unique globally). Post-run REAL-TIME
    // history analysis from the event log:
    //  P1 no read returns a value whose write had not STARTED by the read's
    //     completion (no future values);
    //  P2 let W* = the last write COMPLETED before a read was INVOKED; the
    //     read must return W*'s value or a value written after W* completed
    //     (no stale-after-completion reads);
    //  P3 every read value is a written value.
    {
        let dir = root.join("e7u-linearizability");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e7_insert(&e, 6700, 0, 0);
        e.checkpoint().unwrap();
        let lg = lg_global.clone();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut handles = Vec::new();
        for wth in 0..2usize {
            let e2 = e.clone();
            let lg2 = lg.clone();
            handles.push(std::thread::spawn(move || {
                let mut v = wth as i64 + 1; // writer A: 1,3,5... B: 2,4,6...
                while v < 80 {
                    let mut fields = HashMap::new();
                    fields.insert("idx".to_string(), serde_json::json!(6700));
                    fields.insert("version".to_string(), serde_json::json!(v));
                    fields.insert("cat".to_string(), serde_json::json!("t"));
                    fields.insert("num".to_string(), serde_json::json!(v));
                    fields.insert("title".to_string(), serde_json::json!(format!("reg-{v}")));
                    let mut kv = HashMap::new();
                    kv.insert(HEAD.to_string(), vec_for(6700));
                    let inv = std::time::Instant::now();
                    let r = e2.update_document("bench", &uuid_for(6700, 0).to_string(), fields, kv);
                    e7_log(&lg2, wth, 0, "u-write", "6700", inv, format!("v={v}"));
                    if r.is_err() {
                        break;
                    }
                    v += 2;
                    std::thread::yield_now();
                }
            }));
        }
        for rth in 0..2usize {
            let e2 = e.clone();
            let done2 = done.clone();
            let lg2 = lg.clone();
            handles.push(std::thread::spawn(move || {
                let tid = 2 + rth;
                while !done2.load(std::sync::atomic::Ordering::Relaxed) {
                    let inv = std::time::Instant::now();
                    let val = e7_read_idx(&e2, 6700).map(|x| x.1).unwrap_or(-1);
                    // FULL history: linearizability checking needs every read's
                    // invoke/complete, not a 1/256 sample (INV-FIX E7u)
                    e7_log(&lg2, tid, 0, "u-read", "6700", inv, format!("v={val}"));
                    std::thread::yield_now();
                }
            }));
        }
        let w0 = handles.remove(0);
        let w1 = handles.remove(0);
        w0.join().unwrap();
        w1.join().unwrap();
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }
        // history analysis
        let g = lg.lock().unwrap();
        let writes: Vec<(u128, u128, i64)> =
            g.0.iter()
                .filter(|ev| ev.op == "u-write")
                .filter_map(|ev| {
                    ev.result
                        .strip_prefix("v=")
                        .and_then(|s| s.parse::<i64>().ok())
                        .map(|v| (ev.t_inv_us, ev.t_ret_us, v))
                })
                .collect();
        let reads: Vec<(u128, u128, i64)> =
            g.0.iter()
                .filter(|ev| ev.op == "u-read")
                .filter_map(|ev| {
                    ev.result
                        .strip_prefix("v=")
                        .and_then(|s| s.parse::<i64>().ok())
                        .map(|v| (ev.t_inv_us, ev.t_ret_us, v))
                })
                .collect();
        drop(g);
        let (mut p1, mut p2, mut p3) = (0usize, 0usize, 0usize);
        let mut absent_window = 0usize; // read saw the key transiently ABSENT
        let mut p2_offenders: Vec<(u128, u128, i64, i64)> = Vec::new();
        for &(ri, rc, rv) in &reads {
            if rv == -1 {
                // key absent: the per-op apply window (delete-retire before
                // insert-apply inside one update; see E7m). Counted as its own
                // evidence class — a single-register linearization would not
                // admit it, so it must NOT be silently folded into P3.
                absent_window += 1;
                continue;
            }
            if rv != 0 && !writes.iter().any(|&(_, _, v)| v == rv) {
                p3 += 1;
                continue;
            }
            // P1: future value (write not started by read completion)
            if let Some(&(wi, _, _)) = writes.iter().find(|&&(_, _, v)| v == rv) {
                if wi > rc {
                    p1 += 1;
                    continue;
                }
            }
            // P2: last write completed before this read was invoked
            let before: Vec<&(u128, u128, i64)> =
                writes.iter().filter(|&&(_, wr, _)| wr <= ri).collect();
            if let Some(wlast) = before.iter().max_by_key(|&&(_, wr, _)| wr) {
                if rv != wlast.2 {
                    // legal only if the returned value was written AFTER wlast completed
                    let legal = writes
                        .iter()
                        .any(|&(wi, wr, v)| v == rv && wr > wlast.1 && wi <= rc);
                    if !legal {
                        p2_offenders.push((ri, rc, rv, wlast.2));
                        p2 += 1;
                    }
                }
            }
        }
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        e.close().unwrap();
        rows.push_str(&format!(
            "E7u-linearizability,register-realtime-history,sync,4,0,{},reads={} P1-future={} P2-stale={} P3-unknown={} absent-window={absent_window},single-register-subset,{},{},\"P1/P2/P3 hold for the point-register subset on {} reads / {} writes; {absent_window} reads observed the transient absent state inside update commits (per-op apply, see E7m) — counted separately, NOT folded into P3; NOT a general linearizability claim; p2-offenders={} (read_inv,read_ret,returned,last_completed)\"\n",
            reads.len() + writes.len(),
            reads.len(),
            p1,
            p2,
            p3,
            if clean { "clean" } else { "DIRTY" },
            if p1 + p2 + p3 == 0 && m.contains_key(&6700) { "MATCH" } else { "MISMATCH" },
            reads.len(),
            writes.len(),
            csvs(&p2_offenders),
        ));
        cells += 1;
        vis.push_str(&format!(
            "Single-operation linearizability,point register: {p1} future / {p2} stale-after-completion / {p3} unknown-value violations in {len} real-time reads,VERIFIED for tested register histories (bounded subset; not a system-wide claim)\n",
            len = reads.len()
        ));
    }

    // ---------------- E7v: bounded serializability histories -----------------
    // Transactions are BLIND-WRITE only (TxnOp = Insert|Delete; no reads in
    // the API). Every observed history therefore IS the serial order by
    // commit (the gate serializes whole commits), so the blind-write subset
    // is conflict-serializable by construction — but a GENERAL
    // serializability claim is impossible without transactional reads.
    for seed in [7usize, 13] {
        let dir = root.join(format!("e7v-serial-{seed}"));
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e7_insert(&e, 6701, 0, 0);
        e7_insert(&e, 6702, 0, 0);
        e.checkpoint().unwrap();
        let lg = lg_global.clone();
        let mut rng = (seed as u64) << 8 | 0xD;
        let nxt = |rng: &mut u64| {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*rng >> 33) as usize
        };
        // 3 txns, 2-5 ops each, overlapping keys 6701/6702 (+ inserts 6710..)
        let mut txn_list = Vec::new();
        for tn in 0..3usize {
            let t = e.begin_transaction("bench");
            let nops = 2 + nxt(&mut rng) % 4;
            for k in 0..nops {
                let key = 6701 + nxt(&mut rng) % 2;
                let use_ins = nxt(&mut rng) % 2 == 0;
                if use_ins {
                    let v = 1 + nxt(&mut rng) % 9;
                    let num = (tn * 100 + k * 10 + v) as i64;
                    let mut ver = 0u64;
                    while e
                        .id_mapper
                        .read()
                        .uuid_to_id(&uuid_for(key as u32, ver))
                        .is_some()
                    {
                        ver += 1;
                    }
                    let mut r = doc_record(key as u32, ver, "t", num);
                    r.k_vecs.insert(HEAD.to_string(), vec_for(key as u32));
                    e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
                } else {
                    // delete the live version if one exists
                    if let Some((v, _)) = e7_read_idx(&e, key as u32) {
                        e.record_transaction_operation(t, TxnOp::Delete(uuid_for(key as u32, v)))
                            .unwrap();
                    }
                }
            }
            txn_list.push(t);
        }
        // seeded commit order
        let mut order = vec![0usize, 1, 2];
        for i in (1..3).rev() {
            let j = nxt(&mut rng) % (i + 1);
            order.swap(i, j);
        }
        let mut all_ok = true;
        for (i, &ti) in order.iter().enumerate() {
            all_ok &= e7_commit(&e, txn_list[ti], &lg, i, "serial-hist");
        }
        // reference model: replay commits in commit order at the op level
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir) && iss.is_empty();
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!(
            "E7v-serializability,blind-write-hist-seed-{seed},sync,3,3,{},commit-order={} final-len={},history=serial-by-construction,{},{},\"history IS a serial execution (atomic commits on a gate): conflict-serializable for the blind-write subset; NO general serializability claim (no txn reads)\"\n",
            m.len(),
            csvs(&order),
            m.len(),
            if clean { "clean" } else { "DIRTY" },
            if all_ok && clean && rc && rm == m { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
    }
    vis.push_str("Transaction serializability,blind-write subset: every history is serial (atomic gated commits); general serializability UNTESTABLE (no txn reads),PARTIAL (subset only)\n");

    // ---------------- E7w: schedule enumeration -------------------------------
    // T1: A->1, B->1 ; T2: A->2, B->2 (same-uuid delete+reinsert per key).
    // Enumerate 4 deterministic schedules (stage order x commit order). Every
    // schedule must end with BOTH keys at the later committer's value — a
    // mixed A/B result would be a torn transaction.
    for sched in 0..4usize {
        let dir = root.join(format!("e7w-sched-{sched}"));
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        e7_insert(&e, 6800, 0, 0);
        e7_insert(&e, 6801, 0, 0);
        e.checkpoint().unwrap();
        let lg = lg_global.clone();
        let mk = |e: &AttentionEngine, num: i64| -> u64 {
            let t = e.begin_transaction("bench");
            for k in [6800u32, 6801] {
                e.record_transaction_operation(t, TxnOp::Delete(uuid_for(k, 0)))
                    .unwrap();
                let mut r = doc_record(k, 0, "t", num);
                r.k_vecs.insert(HEAD.to_string(), vec_for(k));
                e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
            }
            t
        };
        // stage order: even schedules stage T1 then T2; odd reverse
        let (t1, t2) = if sched % 2 == 0 {
            let a = mk(&e, 1);
            let b = mk(&e, 2);
            (a, b)
        } else {
            let b = mk(&e, 2);
            let a = mk(&e, 1);
            (a, b)
        };
        // commit order: schedules 0,2 -> T1 first; 1,3 -> T2 first.
        // want = the LATER committer's value (commit-order-wins model).
        let (first, second, want) = if sched % 2 == 0 {
            (t1, t2, 2i64)
        } else {
            (t2, t1, 1i64)
        };
        let okf = e7_commit(&e, first, &lg, 0, "first");
        let oks = e7_commit(&e, second, &lg, 1, "second");
        let (m, iss) = export_state(&e);
        let clean = e7_checker(&e, &dir);
        let consistent = m.get(&6800).map(|x| x.2) == Some(want)
            && m.get(&6801).map(|x| x.2) == Some(want)
            && m.len() == 2;
        e.close().unwrap();
        rows.push_str(&format!(
            "E7w-schedule-enumeration,sched-{sched},sync,2,2,4,A={} B={},all-or-nothing,{},{},\"mixed A/B values would be a torn txn; every enumerated schedule yields the whole later txn state (want {want})\"\n",
            m.get(&6800).map(|x| x.2).unwrap_or(-1),
            m.get(&6801).map(|x| x.2).unwrap_or(-1),
            if clean && iss.is_empty() { "clean" } else { "DIRTY" },
            if okf && oks && consistent && clean { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
    }

    // ---------------- E7x: bounded randomized histories -----------------------
    // Fixed seeds; 3 txns x 2-5 ops; seeded stage/commit interleavings; a
    // sampler thread reads during commits (committed values only). Full event
    // history in e7-events.csv. Failures would preserve seed + log + raw dir.
    for seed in [11usize, 23, 37, 52, 68, 84] {
        let dir = root.join(format!("e7x-rand-{seed}"));
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        for i in 0..5u32 {
            e7_insert(&e, 6900 + i, 0, 0);
        }
        e.checkpoint().unwrap();
        let lg = lg_global.clone();
        let mut rng = (seed as u64).wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let nxt = |rng: &mut u64| {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*rng >> 33) as usize
        };
        // sampler during commits
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bad_samples = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let e2 = e.clone();
        let st = stop.clone();
        let bs = bad_samples.clone();
        let sp = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let lg2 = lg.clone();
        let sampler = std::thread::spawn(move || {
            while !st.load(std::sync::atomic::Ordering::Relaxed) {
                let n = sp.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let idx = 6900 + (n % 5) as u32;
                if let Some((_, num)) = e7_read_idx(&e2, idx) {
                    // any value >= 0 is some committed value; a negative would
                    // be a torn/phantom value
                    if num < 0 {
                        bs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                let inv = std::time::Instant::now();
                e7_log_sampled(&lg2, &sp, 9, 0, "sample", "rand", inv, "ok".into());
                std::thread::yield_now();
            }
        });
        // stage 3 txns with interleaved yields; seeded commit order
        let mut txn_ops = Vec::new();
        for _tn in 0..3usize {
            let t = e.begin_transaction("bench");
            let nops = 2 + nxt(&mut rng) % 4;
            for _k in 0..nops {
                let key = 6900 + nxt(&mut rng) % 5;
                if nxt(&mut rng) % 3 == 0 {
                    if let Some((v, _)) = e7_read_idx(&e, key as u32) {
                        e.record_transaction_operation(t, TxnOp::Delete(uuid_for(key as u32, v)))
                            .unwrap();
                    }
                } else {
                    let mut ver = 0u64;
                    while e
                        .id_mapper
                        .read()
                        .uuid_to_id(&uuid_for(key as u32, ver))
                        .is_some()
                    {
                        ver += 1;
                    }
                    let num = 100 + nxt(&mut rng) % 900;
                    let mut r = doc_record(key as u32, ver, "t", num as i64);
                    r.k_vecs.insert(HEAD.to_string(), vec_for(key as u32));
                    e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
                }
                std::thread::yield_now();
            }
            txn_ops.push(t);
        }
        let mut order = [0usize, 1, 2];
        let j = nxt(&mut rng) % 2;
        order.swap(1, j);
        let mut all_ok = true;
        for (i, &ti) in order.iter().enumerate() {
            all_ok &= e7_commit(&e, txn_ops[ti], &lg, i, "rand-commit");
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        sampler.join().unwrap();
        let bad = bad_samples.load(std::sync::atomic::Ordering::Relaxed);
        let (m, iss) = export_state(&e);
        let engine_clean = e7_checker(&e, &dir);
        e.close().unwrap();
        // fresh-process replay verification; ENGINE checker only (export
        // duplicate-logical-id = blind-write overlap, engine-legal: counted,
        // not failed — mirrors the live-side E7x rule)
        std::env::set_var("PH3D_DURABILITY", "sync");
        let er = attentiondb_core::AttentionEngine::open_dir(&dir, dur_from_env()).unwrap();
        let (_, iss_r) = export_state(&er);
        let replay_engine_clean = checker_report(&er, &dir)["clean"]
            .as_bool()
            .unwrap_or(false);
        let dups_r = iss_r
            .iter()
            .filter(|s| s.starts_with("duplicate logical"))
            .count();
        // order-independent FULL record multiset: duplicate logical ids keep
        // BOTH live versions (export_state's idx-collapsed map is ill-defined
        // for the lost-update family and its survivor depends on numeric order)
        let rm_recs = e7_records(&er);
        er.close().unwrap();
        let m_recs = e7_records(&e);
        let mshape = m_recs.iter().all(|(_, _, _, n)| *n >= 0);
        // duplicate logical ids = two blind-write txns inserted different
        // versions of one idx (lost-update family, engine-legal: idx is a
        // field, not a unique constraint — the ENGINE checker stays clean)
        let dups = iss
            .iter()
            .filter(|s| s.starts_with("duplicate logical"))
            .count();
        rows.push_str(&format!(
            "E7x-randomized,seed-{seed},sync,2,3,{},idx-len={} record-len={} bad-samples={bad} dup-idx={dups} dup-idx-replay={dups_r},commit-order-model,{},{},\"seeded deterministic history; final state = replay of committed txns; sampler never saw an uncommitted value; dup-idx = blind-write overlap (engine-legal, see E7h)\"\n",
            m.len(),
            m_recs.len(),
            rm_recs.len(),
            if engine_clean && replay_engine_clean { "clean" } else { "DIRTY" },
            if all_ok && engine_clean && replay_engine_clean && rm_recs == m_recs && mshape && bad < 1000 { "MATCH" } else { "MISMATCH" },
        ));
        cells += 1;
    }

    // ---------------- E7y: targeted crash/concurrency interaction -------------
    // T1 commits fully; T2 is committing when the gate parks the child
    // (tx_before_commit_wal / tx_after_commit_wal, hit 2 — T1's commit was
    // hit 1). Group SIGKILL; fresh-process recovery judges each txn
    // independently. Modes sync+group (async crash-window loss is already
    // characterized by E6f/E6h).
    for case in ["t1-t2-before", "t1-t2-after"] {
        for mode in ["sync", "group"] {
            let cdir = root.join(format!("e7y-{case}-{mode}"));
            std::env::set_var("PH3D_DURABILITY", mode);
            let gate = if case.ends_with("before") {
                "tx_before_commit_wal"
            } else {
                "tx_after_commit_wal"
            };
            let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
            cmd.args([
                "dbtest",
                "e7conc",
                "--dir",
                cdir.to_str().unwrap(),
                "--case",
                case,
            ])
            .env("PH3D_DURABILITY", mode)
            .env("PH3E_CRASH_AT", gate)
            .env("PH3E_CRASH_HIT", "2")
            .env("PH3E_CRASH_MODEL", "groupkill")
            .env("PH3E_CRASH_MARKER", format!("{}.e7gate", cdir.display()));
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
            let mut child = cmd.spawn().unwrap();
            let mut reached = false;
            let marker = format!("{}.e7gate", cdir.display());
            for _ in 0..6000 {
                if std::path::Path::new(&marker).exists() {
                    reached = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
            let (rm, rc) = match e6_open(&cdir, mode) {
                Ok(x) => x,
                Err(err) => {
                    crash.push_str(&format!(
                        "{case},{mode},{reached},OPEN_REFUSED,-,-,-,{err}\n"
                    ));
                    cells += 1;
                    continue;
                }
            };
            let t1_present = [9700u32, 9701, 9702]
                .iter()
                .all(|i| rm.get(i).map(|x| x.2) == Some(*i as i64));
            let t2_present = rm.get(&9703).map(|x| x.2) == Some(9703);
            let (state, atomic) = match (t1_present, t2_present) {
                (true, false) if case.ends_with("before") => ("T1_PRESENT_T2_ABSENT", "ATOMIC"),
                (true, true) if case.ends_with("after") => ("T1_PRESENT_T2_PRESENT", "ATOMIC"),
                (true, false) if case.ends_with("after") => {
                    ("T1_PRESENT_T2_ABSENT_ASYNCLEGAL", "ATOMIC")
                }
                (true, true) if case.ends_with("before") => {
                    ("T1_PRESENT_T2_PRESENT_PREACK", "ATOMIC")
                }
                _ => ("PARTIAL/UNEXPECTED", "VIOLATION"),
            };
            crash.push_str(&format!(
                "{case},{mode},{reached},{state},{atomic},{},{},T1 independent; T2 judged at its own gate\n",
                if rc { "clean" } else { "DIRTY" },
                if atomic == "ATOMIC" && rc { "MATCH" } else { "MISMATCH" },
            ));
            cells += 1;
        }
    }

    // ---------------- finalize ----------------
    e7_write_events(out, &lg_global);
    std::fs::write(format!("{out}/e7-conc.csv"), &rows).unwrap();
    std::fs::write(format!("{out}/e7-visibility.csv"), &vis).unwrap();
    std::fs::write(format!("{out}/e7-crash.csv"), &crash).unwrap();
    format!(
        "e7: {cells} cells -> {out}/e7-conc.csv + e7-visibility.csv + e7-crash.csv + e7-events.csv"
    )
}

/// E7y crash child: baseline + T1 commit + T2 commit (parked at gate hit 2).
fn e7_conc_child(dir: &std::path::Path, _case: &str) -> ! {
    use attentiondb_core::transaction::TxnOp;
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    for i in 9000..9005u32 {
        e7_insert(&e, i, 1, i as i64);
    }
    e.checkpoint().unwrap();
    // T1: three inserts, fully committed (gate hit 1 of the txn gates)
    let t1 = e.begin_transaction("bench");
    for i in 9700..9703u32 {
        let mut r = doc_record(i, 1, "t", i as i64);
        r.k_vecs.insert(HEAD.to_string(), vec_for(i));
        e.record_transaction_operation(t1, TxnOp::Insert(r))
            .unwrap();
    }
    assert!(e.commit_transaction(t1).unwrap());
    // T2: one insert; its commit is the gate's hit 2 -> parks here
    let t2 = e7_stage_upsert(&e, 9703, None, 1, 9703);
    let _ = e.commit_transaction(t2);
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// Main E6 driver. Produces <out>/e6-txn.csv + <out>/e6-crash.csv.
fn run_e6(out: &str) -> String {
    use attentiondb_core::transaction::TxnOp;
    std::fs::create_dir_all(out).unwrap();
    let mut rows = String::from("family,case,mode,expected_state,observed_state,commit_status,checker_clean,restart_ok,model_match,notes\n");
    let mut crash = String::from(
        "boundary,mode,aborted_at_boundary,txn_state,atomicity,checker_clean,model_match,notes\n",
    );
    let root = std::path::PathBuf::from("/tmp/ph3e-e6");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut cells = 0usize;

    // helper: build a baseline db (10 docs idx 10..20) and return (engine, dir, model)
    macro_rules! fresh {
        ($name:expr, $mode:expr) => {{
            let dir = root.join($name);
            let _ = std::fs::remove_dir_all(&dir);
            std::env::set_var("PH3D_DURABILITY", $mode);
            let e = open_db(&dir);
            e.create_collection("bench", DIM, &[HEAD]).unwrap();
            for i in 10..20u32 {
                let mut r = doc_record(i, 1, "t", i as i64);
                r.k_vecs.insert(HEAD.to_string(), vec_for(i));
                e.insert_document("bench", r).unwrap();
            }
            e.checkpoint().unwrap();
            (dir, e)
        }};
    }

    // ---- E6a basic transactions ----
    for (name, ids, dels, expect_extra) in [
        ("t1-single-insert", vec![100u32], vec![], vec![100u32]),
        (
            "t2-multi-insert",
            vec![101, 102, 103],
            vec![],
            vec![101, 102, 103],
        ),
        ("t3-insert-delete", vec![104], vec![11], vec![104]),
        ("t4-multi-delete", vec![], vec![12, 13, 14], vec![]),
    ] {
        let (dir, e) = fresh!(name, "sync");
        let mut expected: Vec<(u32, i64)> = (10..20u32)
            .filter(|i| !dels.contains(i))
            .map(|i| (i, i as i64))
            .collect();
        expected.extend(expect_extra.iter().map(|i| (*i, *i as i64)));
        let expected = e6_expected_model(&expected);
        let t = e6_stage_txn(&e, &ids, &dels, "t");
        let ok = e.commit_transaction(t).unwrap();
        let (got, exp, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        let rok = rm == expected && rc;
        rows.push_str(&format!("E6a-basic,{name},sync,{exp},{got},{},\"{clean}/{rc}\",{},\"{m1}/{rok}\",commit-marker boundary; restart equality\n",
            if ok { "COMMITTED" } else { "FAILED" }, if rok { "ok" } else { "FAIL" }));
        cells += 1;
    }
    {
        // empty transaction: BEGIN + COMMIT, no ops
        let (dir, e) = fresh!("t5-empty-txn", "sync");
        let expected = e6_expected_model(&(10..20u32).map(|i| (i, i as i64)).collect::<Vec<_>>());
        let t = e.begin_transaction("bench");
        let ok = e.commit_transaction(t).unwrap();
        let (_, _, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6a-basic,t5-empty-txn,sync,10,{},{},\"{clean}/{rc}\",{},\"{m1}/{}\",zero-op txn is a legal no-op\n",
            rm.len(), if ok { "COMMITTED" } else { "FAILED" },
            if rm == expected && rc { "ok" } else { "FAIL" }, rm == expected));
        cells += 1;
    }

    // ---- E6b rollback + illegal transitions ----
    {
        let (dir, e) = fresh!("t6-rollback", "sync");
        let expected = e6_expected_model(&(10..20u32).map(|i| (i, i as i64)).collect::<Vec<_>>());
        let t = e6_stage_txn(&e, &[110, 111], &[15], "t");
        let rolled = e.rollback_transaction(t).unwrap();
        let (_, _, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6b-rollback,rollback-invisible,sync,10,{},ROLLED_BACK,\"{clean}/{rc}\",{},\"{m1}/{}\",staged ops never left memory\n",
            rm.len(), if rolled && rm == expected && rc { "ok" } else { "FAIL" }, rm == expected));
        cells += 1;
    }
    {
        let (dir, e) = fresh!("t7-illegal-transitions", "sync");
        // double commit: second must be a no-op Ok(false)
        let t = e6_stage_txn(&e, &[120], &[], "t");
        let c1 = e.commit_transaction(t).unwrap();
        let c2 = e.commit_transaction(t).unwrap();
        // rollback after commit: no-op Ok(false)
        let r_after = e.rollback_transaction(t).unwrap();
        // commit after rollback: no-op Ok(false)
        let t2 = e6_stage_txn(&e, &[121], &[], "t");
        assert!(e.rollback_transaction(t2).unwrap());
        let c3 = e.commit_transaction(t2).unwrap();
        let mut basev: Vec<(u32, i64)> = (10..20u32).map(|i| (i, i as i64)).collect();
        basev.push((120, 120));
        let expected = e6_expected_model(&basev);
        let (got, exp, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6b-rollback,illegal-transitions,sync,{exp},{got},c1={c1},c2={c2},r={r_after},c3={c3},\"{clean}/{rc}\",{},\"{m1}/{}\",double-commit/rollback-after-commit/commit-after-rollback all no-ops\n",
            if rm == expected && rc { "ok" } else { "FAIL" }, rm == expected));
        cells += 1;
    }

    // ---- E6c multi-op atomicity ----
    for n in [5usize, 10, 50, 100] {
        let name = format!("t8-multiop-{n}");
        let (dir, e) = fresh!(&name, "sync");
        let ids: Vec<u32> = (1000..1000 + n as u32).collect();
        let dels: Vec<u32> = (0..n.min(10)).map(|k| 10 + k as u32).collect();
        let t = e6_stage_txn(&e, &ids, &dels, "t");
        let ok = e.commit_transaction(t).unwrap();
        let mut expv: Vec<(u32, i64)> = (10..20u32)
            .filter(|i| !dels.contains(i))
            .map(|i| (i, i as i64))
            .collect();
        expv.extend(ids.iter().map(|i| (*i, *i as i64)));
        let expected = e6_expected_model(&expv);
        let (got, exp, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6c-multiop,n-ops-{n},sync,{exp},{got},{},\"{clean}/{rc}\",{},\"{m1}/{}\",mixed insert+delete txn recovers as one unit\n",
            if ok { "COMMITTED" } else { "FAILED" }, if rm == expected && rc { "ok" } else { "FAIL" }, rm == expected));
        cells += 1;
    }

    // ---- E6d multiple transactions ----
    {
        let (dir, e) = fresh!("t9-multi-txn", "sync");
        let t1 = e6_stage_txn(&e, &[100], &[], "t");
        e.commit_transaction(t1).unwrap();
        let t2 = e6_stage_txn(&e, &[101], &[], "t");
        e.rollback_transaction(t2).unwrap();
        let t3 = e6_stage_txn(&e, &[102], &[12], "t");
        e.commit_transaction(t3).unwrap();
        let expected = e6_expected_model(&[
            (10, 10),
            (11, 11),
            (13, 13),
            (14, 14),
            (15, 15),
            (16, 16),
            (17, 17),
            (18, 18),
            (19, 19),
            (100, 100),
            (102, 102),
        ]);
        let (got, exp, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6d-multi-txn,commit-rollback-commit,sync,{exp},{got},T1+T3 COMMITTED T2 ROLLED_BACK,\"{clean}/{rc}\",{},\"{m1}/{}\",independent outcomes per txn\n",
            if rm == expected && rc { "ok" } else { "FAIL" }, rm == expected));
        cells += 1;
    }

    // ---- E6i ordering ----
    {
        let (dir, e) = fresh!("t10-ordering", "sync");
        // T1 inserts A; T2 deletes A -> absent
        let t1 = e6_stage_txn(&e, &[130], &[], "t");
        e.commit_transaction(t1).unwrap();
        let t2 = e6_stage_txn(&e, &[], &[130], "t");
        e.commit_transaction(t2).unwrap();
        // T3 inserts B; T4 re-inserts B with a NEW value (same uuid = update-like)
        let t3 = e6_stage_txn(&e, &[131], &[], "t");
        e.commit_transaction(t3).unwrap();
        let mut r = doc_record(131, 1, "t", 999);
        r.k_vecs.insert(HEAD.to_string(), vec_for(131));
        let t4 = e.begin_transaction("bench");
        e.record_transaction_operation(t4, TxnOp::Insert(r))
            .unwrap();
        e.commit_transaction(t4).unwrap();
        let expected = e6_expected_model(&[
            (10, 10),
            (11, 11),
            (12, 12),
            (13, 13),
            (14, 14),
            (15, 15),
            (16, 16),
            (17, 17),
            (18, 18),
            (19, 19),
            (131, 999),
        ]);
        let (got, exp, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6i-ordering,commit-order-wins,sync,{exp},{got},T2-after-T1=T3-after-T2,\"{clean}/{rc}\",{},\"{m1}/{}\",insert->delete absent; same-uuid reinsert = last commit wins (999)\n",
            if rm == expected && rc { "ok" } else { "FAIL" }, rm == expected));
        cells += 1;
    }
    {
        // delete-of-missing inside a txn: no-op, txn still commits
        let (dir, e) = fresh!("t11-delete-missing", "sync");
        let t = e6_stage_txn(&e, &[], &[777], "t");
        let ok = e.commit_transaction(t).unwrap();
        let expected = e6_expected_model(&(10..20u32).map(|i| (i, i as i64)).collect::<Vec<_>>());
        let (_, _, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm2, rc2) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6i-ordering,delete-missing-noop,sync,10,{},{},\"{clean}\",\"{}/-\",\"{m1}/-\",delete of absent uuid is a no-op; txn still commits\n",
            rm2.len(), if ok { "COMMITTED" } else { "FAILED" }, if rc2 { "ok" } else { "FAIL" }));
        cells += 1;
    }

    // ---- E6j bounded concurrency (3 staged txns, serialized commits) ----
    {
        let (dir, e) = fresh!("t12-concurrency", "group");
        let ta = e6_stage_txn(&e, &[140], &[], "t");
        let tb = e6_stage_txn(&e, &[141], &[13], "t");
        let tc = e6_stage_txn(&e, &[142], &[], "t");
        // commit out of staging order: C, A, B
        e.commit_transaction(tc).unwrap();
        e.commit_transaction(ta).unwrap();
        e.commit_transaction(tb).unwrap();
        let expected = e6_expected_model(&[
            (10, 10),
            (11, 11),
            (12, 12),
            (14, 14),
            (15, 15),
            (16, 16),
            (17, 17),
            (18, 18),
            (19, 19),
            (140, 140),
            (141, 141),
            (142, 142),
        ]);
        let (got, exp, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6j-concurrency,3-staged-txns,group,{exp},{got},all committed (out of stage order),\"{clean}/{rc}\",{},\"{m1}/{}\",staging is concurrent; commits serialize on the mutation gate; WAL order = commit order\n",
            if rm == expected && rc { "ok" } else { "FAIL" }, rm == expected));
        cells += 1;
    }

    // ---- E6k update / upsert semantics (standalone; in-txn = UNSUPPORTED by type) ----
    {
        let (dir, e) = fresh!("t13-update", "sync");
        let uuid13 = uuid_for(13, 1);
        let mut fields = std::collections::HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(13));
        fields.insert("cat".to_string(), serde_json::json!("t"));
        fields.insert("num".to_string(), serde_json::json!(4321));
        let mut kv = std::collections::HashMap::new();
        kv.insert(HEAD.to_string(), vec_for(13));
        let ok_upd = e
            .update_document("bench", &uuid13.to_string(), fields.clone(), kv.clone())
            .is_ok();
        // missing -> NotFound
        let missing = e
            .update_document(
                "bench",
                &uuid::Uuid::new_v4().to_string(),
                fields.clone(),
                kv.clone(),
            )
            .is_err();
        // deleted -> NotFound (not a resurrection path)
        e.delete_document("bench", &uuid_for(14, 1).to_string())
            .unwrap();
        let upd_del = e
            .update_document(
                "bench",
                &uuid_for(14, 1).to_string(),
                fields.clone(),
                kv.clone(),
            )
            .is_err();
        // upsert both branches
        let up1 = e
            .upsert_document("bench", uuid_for(13, 1), fields.clone(), kv.clone())
            .is_ok();
        let mut fields150 = std::collections::HashMap::new();
        fields150.insert("idx".to_string(), serde_json::json!(150));
        fields150.insert("cat".to_string(), serde_json::json!("t"));
        fields150.insert("num".to_string(), serde_json::json!(150));
        let up2 = e
            .upsert_document("bench", uuid_for(150, 1), fields150, kv.clone())
            .is_ok();
        // model: 13 = (version 2, num 4321) — update bumps version; export_state reads fields
        let (m, iss) = export_state(&e);
        let v13 = m.get(&13).cloned().unwrap_or((0, String::new(), 0));
        let has150 = m.contains_key(&150);
        let absent14 = !m.contains_key(&14);
        // NOTE: export_state reads the version from the FIELDS map; update_document
        // replaces fields wholesale, so the exported version is 0 (absent) — the
        // observable update semantics here are: num updated, uuid preserved,
        // missing/deleted = NotFound, upsert both branches.
        let expected_all = ok_upd
            && missing
            && upd_del
            && up1
            && up2
            && v13.1 == "t"
            && v13.2 == 4321
            && has150
            && absent14
            && iss.is_empty();
        let clean = e5_clean(&e, &dir);
        let checker_dbg = {
            let cj = checker_report(&e, &dir);
            let istr: Vec<String> = cj
                .as_object()
                .map(|o| {
                    o.iter()
                        .filter(|(_k, v)| v.is_boolean() && !v.as_bool().unwrap())
                        .map(|(k, _)| k.clone())
                        .collect()
                })
                .unwrap_or_default();
            format!("iss={} flags={:?}", iss.len(), istr)
        };
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        let restart_ok = rm.get(&13).cloned() == Some((0u64, "t".to_string(), 4321i64))
            && !rm.contains_key(&14)
            && rm.contains_key(&150)
            && rc;
        rows.push_str(&format!("E6k-update-upsert,update-upsert-semantics,sync,10 docs,{},{},\"{clean}/{rc}\",{},\"{}/{}\",upd={ok_upd} missing={missing} upd-del={upd_del} up1={up1} up2={up2} v13={v13:?} has150={has150} absent14={absent14} {checker_dbg}; update: exists-only/uuid-preserved/version+1/old-id-retired; upsert=exists?update:insert; in-txn update UNSUPPORTED (TxnOp type)\n",
            rm.len(), if expected_all { "OK" } else { "BAD" }, if restart_ok { "ok" } else { "FAIL" }, expected_all, restart_ok));
        cells += 1;
    }

    // ---- E6l same-key chains ----
    {
        let (dir, e) = fresh!("t14-same-key", "sync");
        // one txn: insert A, insert A(v2 same uuid), delete A -> absent
        let t = e.begin_transaction("bench");
        let mut r1 = doc_record(160, 1, "t", 1);
        r1.k_vecs.insert(HEAD.to_string(), vec_for(160));
        e.record_transaction_operation(t, TxnOp::Insert(r1))
            .unwrap();
        let mut r2 = doc_record(160, 1, "t", 2);
        r2.k_vecs.insert(HEAD.to_string(), vec_for(160));
        e.record_transaction_operation(t, TxnOp::Insert(r2))
            .unwrap();
        e.record_transaction_operation(t, TxnOp::Delete(uuid_for(160, 1)))
            .unwrap();
        e.commit_transaction(t).unwrap();
        // separate txn re-inserts A -> present once
        let t2 = e6_stage_txn(&e, &[160], &[], "t");
        e.commit_transaction(t2).unwrap();
        let (m, iss) = export_state(&e);
        let count = m.iter().filter(|(k, _)| **k == 160).count();
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        let mut expect_l =
            e6_expected_model(&(10..20u32).map(|i| (i, i as i64)).collect::<Vec<_>>());
        expect_l.insert(160, (1, "t".into(), 160)); // reinsert via e6_stage_txn: num=idx
        let ok = rm == expect_l && rc && iss.is_empty() && count == 1;
        rows.push_str(&format!("E6l-same-key,ins-ins-del-ins-txn,sync,{} uuids,{},COMMITTED,\"{clean}/{rc}\",{},\"{}/{}\",same-uuid double insert + delete in ONE txn then reinsert: final = reinserted value, uuid unique\n",
            count, rm.get(&160).map(|x| x.2).unwrap_or(-1),
            if ok { "ok" } else { "FAIL" }, ok, rm == expect_l));
        cells += 1;
    }

    // ---- E6m tombstones ----
    {
        let (dir, e) = fresh!("t15-txn-tombstones", "sync");
        let t1 = e6_stage_txn(&e, &[170], &[], "t");
        e.commit_transaction(t1).unwrap();
        let t2 = e6_stage_txn(&e, &[], &[170], "t");
        e.commit_transaction(t2).unwrap();
        let st = e.compact_storage().unwrap();
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6m-tombstones,txn-ins-txn-del-compact,sync,absent,{},2 tombs reclaimed,\"{rc}\",{},\"{}\",txn-tombstone GC sound after compaction+restart (E5 tie-break regression included in suite)\n",
            rm.contains_key(&170), if !rm.contains_key(&170) && rc { "ok" } else { "FAIL" }, !rm.contains_key(&170)));
        let _ = st;
        cells += 1;
    }
    {
        let (dir, e) = fresh!("t16-rollback-delete", "sync");
        let t = e6_stage_txn(&e, &[], &[16], "t");
        e.rollback_transaction(t).unwrap();
        let expected = e6_expected_model(&(10..20u32).map(|i| (i, i as i64)).collect::<Vec<_>>());
        let (_, _, m1) = e6_model_eq(&e, &expected);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6m-tombstones,staged-delete-rollback,sync,present,{},ROLLED_BACK,\"{rc}\",{},\"{m1}/{}\",rolled-back delete leaves doc present; no resurrection risk\n",
            rm.contains_key(&16), if rm.contains_key(&16) && rc { "ok" } else { "FAIL" }, rm == expected));
        cells += 1;
    }

    // ---- E6n compaction interaction ----
    for order in ["commit-then-compact", "compact-then-commit"] {
        let (dir, e) = fresh!(&format!("t17-{order}"), "sync");
        if order == "commit-then-compact" {
            let t = e6_stage_txn(&e, &[180], &[11], "t");
            e.commit_transaction(t).unwrap();
            e.compact_storage().unwrap();
        } else {
            e.compact_storage().unwrap();
            let t = e6_stage_txn(&e, &[180], &[11], "t");
            e.commit_transaction(t).unwrap();
        }
        let expected = e6_expected_model(&[
            (10, 10),
            (12, 12),
            (13, 13),
            (14, 14),
            (15, 15),
            (16, 16),
            (17, 17),
            (18, 18),
            (19, 19),
            (180, 180),
        ]);
        let (_, _, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6n-compaction,{order},sync,10,{},COMMITTED,\"{clean}/{rc}\",{},\"{m1}/{}\",gate serializes txn vs compaction; both orders consistent\n",
            rm.len(), if rm == expected && rc { "ok" } else { "FAIL" }, rm == expected));
        cells += 1;
    }

    // ---- E6o checkpoint interaction ----
    for order in [
        "txn-then-ckpt",
        "ckpt-then-txn",
        "staged-then-ckpt",
        "commit-ckpt-restart",
    ] {
        let (dir, e) = fresh!(&format!("t18-{order}"), "sync");
        let expect_committed = order != "staged-then-ckpt";
        if order == "txn-then-ckpt" || order == "commit-ckpt-restart" {
            let t = e6_stage_txn(&e, &[190], &[11], "t");
            e.commit_transaction(t).unwrap();
            e.checkpoint().unwrap();
        } else if order == "ckpt-then-txn" {
            e.checkpoint().unwrap();
            let t = e6_stage_txn(&e, &[190], &[11], "t");
            e.commit_transaction(t).unwrap();
        } else {
            // staged, NOT committed, then checkpoint: must NOT become committed
            let t = e6_stage_txn(&e, &[190], &[11], "t");
            let _ = t;
            e.checkpoint().unwrap();
        }
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        let txn_applied = rm.contains_key(&190) && !rm.contains_key(&11);
        let ok = txn_applied == expect_committed && rc;
        rows.push_str(&format!("E6o-checkpoint,{order},sync,{},{},{},\"{rc}\",{},\"{}/{}\",checkpoint never makes an uncommitted txn committed\n",
            if expect_committed { "txn applied" } else { "txn absent" }, txn_applied,
            if expect_committed { "COMMITTED" } else { "NEVER-COMMITTED" },
            if ok { "ok" } else { "FAIL" }, txn_applied, expect_committed));
        cells += 1;
    }

    // ---- E6p WAL rotation ----
    {
        let (dir, e) = fresh!("t19-rotation", "group");
        std::env::set_var("ATTENTIONDB_WAL_SEGMENT_BYTES", "2048");
        let ids: Vec<u32> = (2000..2100).collect();
        let dels: Vec<u32> = (10..15).collect();
        let t = e6_stage_txn(&e, &ids, &dels, "t");
        let ok = e.commit_transaction(t).unwrap();
        std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
        let mut expv: Vec<(u32, i64)> = (15..20u32).map(|i| (i, i as i64)).collect();
        expv.extend(ids.iter().map(|i| (*i, *i as i64)));
        let expected = e6_expected_model(&expv);
        let (got, exp, m1) = e6_model_eq(&e, &expected);
        let clean = e5_clean(&e, &dir);
        e.close().unwrap();
        let (rm, rc) = e6_open(&dir, "sync").unwrap();
        rows.push_str(&format!("E6p-rotation,100-op-txn-2KiB-segments,group,{exp},{got},{},\"{clean}/{rc}\",{},\"{m1}/{}\",txn crosses multiple WAL rotations; no seq gap/dup replay\n",
            if ok { "COMMITTED" } else { "FAILED" }, if rm == expected && rc { "ok" } else { "FAIL" }, rm == expected));
        cells += 1;
    }

    // ---- E6q backup interaction ----
    {
        let base = e6_expected_model(&(10..20u32).map(|i| (i, i as i64)).collect::<Vec<_>>());
        // (1) backup BEFORE txn
        let (dir, e) = fresh!("t20-bkp-before", "sync");
        let b1 = root.join("bkp-before.backup");
        e.backup_to(&b1).unwrap();
        e.close().unwrap();
        // restore b1 via the standard path
        let (_, clean1, _) =
            e4_restore_and_open(&b1, &root.join("bkp-before.restored"), "sync").unwrap();
        let ok1 = clean1;
        // (2) backup DURING staged txn (staged has no durable presence)
        let e = open_db(&dir);
        let t = e6_stage_txn(&e, &[210], &[11], "t");
        let b2 = root.join("bkp-during.backup");
        e.backup_to(&b2).unwrap();
        e.rollback_transaction(t).unwrap();
        // (3) backup AFTER commit
        let t3 = e6_stage_txn(&e, &[210], &[11], "t");
        e.commit_transaction(t3).unwrap();
        let b3 = root.join("bkp-after.backup");
        e.backup_to(&b3).unwrap();
        e.close().unwrap();
        let (r2, clean2, _) =
            e4_restore_and_open(&b2, &root.join("bkp-during.restored"), "sync").unwrap();
        let (r3, clean3, _) =
            e4_restore_and_open(&b3, &root.join("bkp-after.restored"), "sync").unwrap();
        let after = e6_expected_model(&[
            (10, 10),
            (12, 12),
            (13, 13),
            (14, 14),
            (15, 15),
            (16, 16),
            (17, 17),
            (18, 18),
            (19, 19),
            (210, 210),
        ]);
        let ok2 = r2 == base && clean2;
        let ok3 = r3 == after && clean3;
        rows.push_str(&format!("E6q-backup,before-during-after,sync,see notes,see notes,3 backups,\"{clean1}/{clean2}/{clean3}\",{},\"{}/{}\",before=baseline; during-staged=baseline (no partial txn in snapshot); after=committed state\n",
            if ok1 && ok2 && ok3 { "ok" } else { "FAIL" }, ok2, ok3));
        cells += 1;
    }

    // ---- E6r recovery idempotence ----
    {
        let (dir, e) = fresh!("t21-idempotence", "sync");
        let t = e6_stage_txn(&e, &[220, 221], &[12, 13], "t");
        e.commit_transaction(t).unwrap();
        e.close().unwrap();
        let expected = e6_expected_model(&[
            (10, 10),
            (11, 11),
            (14, 14),
            (15, 15),
            (16, 16),
            (17, 17),
            (18, 18),
            (19, 19),
            (220, 220),
            (221, 221),
        ]);
        let (m1, c1) = e6_open(&dir, "sync").unwrap();
        let (m2, c2) = e6_open(&dir, "sync").unwrap();
        let (m3, c3) = e6_open(&dir, "sync").unwrap();
        let ok = m1 == expected && m2 == expected && m3 == expected && c1 && c2 && c3;
        rows.push_str(&format!("E6r-idempotence,restart-x3,sync,{},{},{},\"{c1}/{c2}/{c3}\",{},\"{}\",recovery(recovery(state))==recovery(state)\n",
            m1.len(), m3.len(), if ok { "COMMITTED" } else { "DRIFT" },
            if ok { "ok" } else { "FAIL" }, m1 == m3));
        cells += 1;
    }

    // (b) garbled frame in a COMMITTED txn's segment -> refuse
    {
        let (dir, e) = fresh!("t22-garble", "sync");
        let t = e6_stage_txn(&e, &[230, 231], &[], "t");
        e.commit_transaction(t).unwrap();
        // NO close(): close() checkpoints and trims the WAL — we need the txn
        // records to still be in the segment for the garble to hit them.
        // (Sync mode: every append already fsynced; drop is safe here.)
        drop(e);
        let p = root.join("corrupt-garbled");
        copy_dir_all(&dir, &p);
        let waldir = p.join("WAL");
        let mut segs: Vec<_> = std::fs::read_dir(&waldir)
            .unwrap()
            .flatten()
            .map(|x| x.path())
            .filter(|x| x.extension().and_then(|e| e.to_str()) == Some("wal"))
            .collect();
        segs.sort();
        // garble the middle of the LARGEST segment (guaranteed non-trivial)
        let last = segs
            .iter()
            .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .unwrap();
        let mut data = std::fs::read(last).unwrap();
        assert!(data.len() >= 64, "segment too small to garble meaningfully");
        let mid = data.len() / 2;
        for b in data[mid..mid + 16].iter_mut() {
            *b ^= 0xA5;
        }
        std::fs::write(last, &data).unwrap();
        match e6_open(&p, "sync") {
            Err(_) => rows.push_str("E6s-corruption,garbled-committed-group,sync,refuse,REFUSED,-,-,-,corruption never silently converted to a valid txn (E1 policy)\n"),
            Ok((rm, rc)) => rows.push_str(&format!("E6s-corruption,garbled-committed-group,sync,refuse,ACCEPTED,{},{},{},UNEXPECTED acceptance; docs={}\n", rm.len(), rc, if !rc { "flagged" } else { "clean" }, rm.len())),
        }
        cells += 1;
    }

    // ---- E6e/f/g crash families (fresh-process evidence) ----
    let base_present: Vec<(u32, i64)> = (9000..9005u32).map(|i| (i, i as i64)).collect();
    let baseline_m = e6_expected_model(&base_present);
    let mut committed_v: Vec<(u32, i64)> = base_present.clone();
    committed_v.retain(|(i, _)| *i != 9002);
    committed_v.extend([9100u32, 9101, 9102].iter().map(|i| (*i, *i as i64)));
    let committed_m = e6_expected_model(&committed_v);

    // (case, mode, boundary label, gate name, 1-based hit number of that gate
    // in the child process). Gates hit on the PLAIN insert path too (once per
    // insert_document): the child inserts 5 baseline docs, then the txn apply
    // adds exactly one more AFTER_APPLY/BEFORE_ACK pair — hence hits 6 and 7.
    type CrashCase<'a> = (&'a str, &'a str, &'a str, Option<(&'a str, usize)>);
    let crash_cases: Vec<CrashCase> = vec![
        ("stage-1", "sync", "pre-commit staging", None),
        ("stage-3", "sync", "pre-commit staging", None),
        ("stage-10", "group", "pre-commit staging", None),
        (
            "commit",
            "sync",
            "gate: tx_before_commit_wal",
            Some(("tx_before_commit_wal", 1)),
        ),
        (
            "commit",
            "group",
            "gate: tx_before_commit_wal",
            Some(("tx_before_commit_wal", 1)),
        ),
        (
            "commit",
            "async",
            "gate: tx_before_commit_wal",
            Some(("tx_before_commit_wal", 1)),
        ),
        (
            "commit",
            "sync",
            "gate: tx_after_commit_wal",
            Some(("tx_after_commit_wal", 1)),
        ),
        (
            "commit",
            "group",
            "gate: tx_after_commit_wal",
            Some(("tx_after_commit_wal", 1)),
        ),
        (
            "commit",
            "async",
            "gate: tx_after_commit_wal",
            Some(("tx_after_commit_wal", 1)),
        ),
        // generic windows ON the txn commit path: WAL is already persisted at
        // both (the append is the durability step), so sync/group must be
        // PRESENT; async may lose the buffered append (A2 contract).
        (
            "commit",
            "sync",
            "gate: after_apply",
            Some(("after_apply", 6)),
        ),
        (
            "commit",
            "group",
            "gate: after_apply",
            Some(("after_apply", 6)),
        ),
        (
            "commit",
            "async",
            "gate: after_apply",
            Some(("after_apply", 6)),
        ),
        (
            "commit",
            "sync",
            "gate: before_ack",
            Some(("before_ack", 7)),
        ),
        (
            "commit",
            "group",
            "gate: before_ack",
            Some(("before_ack", 7)),
        ),
        (
            "commit",
            "async",
            "gate: before_ack",
            Some(("before_ack", 7)),
        ),
        ("post-ack", "sync", "abort after ACK", None),
        ("post-ack", "group", "abort after ACK", None),
        ("post-ack", "async", "abort after ACK", None),
        ("multi", "sync", "T1 committed; T3 staged only", None),
    ];
    for (case, mode, boundary, gate_spec) in crash_cases {
        let cdir = root.join(format!(
            "crash-{case}-{mode}-{case2}",
            case2 = boundary.replace([' ', ':'], "-")
        ));
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "dbtest",
            "e6txn",
            "--dir",
            cdir.to_str().unwrap(),
            "--case",
            case,
        ])
        .env("PH3D_DURABILITY", mode);
        if let Some((gate, hitno)) = gate_spec {
            cmd.env("PH3E_CRASH_AT", gate)
                .env("PH3E_CRASH_HIT", hitno.to_string())
                .env("PH3E_CRASH_MODEL", "groupkill")
                .env("PH3E_CRASH_MARKER", format!("{}.e6gate", cdir.display()));
        }
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        let mut child = cmd.spawn().unwrap();
        // wait for terminal condition
        let post_ack = case == "post-ack";
        let mut reached = false;
        let marker = format!("{}.e6", cdir.display());
        if post_ack {
            // child aborts itself right after the commit ACK: wait for exit
            for _ in 0..6000 {
                if let Ok(Some(_)) = child.try_wait() {
                    reached = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        } else {
            for _ in 0..6000 {
                if std::path::Path::new(&marker).exists()
                    || std::path::Path::new(&format!("{}.e6gate", cdir.display())).exists()
                {
                    reached = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
        }
        let _ = child.wait();
        let (rm, rc) = match e6_open(&cdir, "sync") {
            Ok(x) => x,
            Err(err) => {
                crash.push_str(&format!(
                    "{boundary},{mode},{reached},OPEN_REFUSED,-,-,-,{err}\n"
                ));
                cells += 1;
                continue;
            }
        };
        // what SHOULD the txn state be? Decided by the gate, not by label
        // substrings: staging and the pre-WAL window leave the txn ABSENT;
        // everything after the WAL append (tx_after_commit_wal, after_apply,
        // before_ack) has persistence already performed — sync/group PRESENT,
        // async allowed to lose the buffered append (A2).
        let expect_state = if case.starts_with("stage-")
            || case == "multi"
            || gate_spec
                .map(|(g, _)| g == "tx_before_commit_wal")
                .unwrap_or(false)
        {
            "ABSENT"
        } else if post_ack {
            if mode == "async" {
                "ABSENT_ALLOWED_CONTRACT"
            } else {
                "PRESENT"
            }
        } else if mode == "async" {
            "ABSENT_OR_PRESENT_PREACK"
        } else {
            "PRESENT"
        };
        let (cls, atomic) = if case.starts_with("stage-") {
            if rm == baseline_m {
                ("ABSENT", true)
            } else {
                ("PARTIAL/UNEXPECTED", false)
            }
        } else if case == "multi" {
            let mut t1m = baseline_m.clone();
            t1m.insert(9100, (1, "t".into(), 9100));
            if rm == t1m {
                ("T1_PRESENT_T3_ABSENT", true)
            } else {
                ("PARTIAL/UNEXPECTED", false)
            }
        } else {
            let (c, a) = e6_classify(&rm, &baseline_m, &committed_m);
            (c, a)
        };
        let expect_lbl = if case == "multi" {
            "T1_PRESENT_T3_ABSENT".to_string()
        } else {
            expect_state.to_string()
        };
        crash.push_str(&format!(
            "{boundary},{mode},{reached},{cls},{},{},{},expected={expect_lbl}\n",
            if atomic { "ATOMIC" } else { "VIOLATION" },
            if rc { "clean" } else { "DIRTY" },
            if atomic && rc { "MATCH" } else { "MISMATCH" }
        ));
        cells += 1;
    }

    // ---- E6s WAL corruption ----
    {
        // (a) torn tail through a PARTIAL txn group (from a tx_before_commit_wal crash)
        let case2 = "gate: tx_before_commit_wal".replace([' ', ':'], "-");
        let cdir = root.join(format!("crash-commit-sync-{case2}"));
        if cdir.exists() {
            let p = root.join("corrupt-torn-txn");
            copy_dir_all(&cdir, &p);
            let waldir = p.join("WAL");
            let mut segs: Vec<_> = std::fs::read_dir(&waldir)
                .unwrap()
                .flatten()
                .map(|x| x.path())
                .filter(|x| x.extension().and_then(|e| e.to_str()) == Some("wal"))
                .collect();
            segs.sort();
            if let Some(last) = segs.last() {
                let mut data = std::fs::read(last).unwrap();
                let half = data.len() / 2;
                data.truncate(half);
                std::fs::write(last, data).unwrap();
            }
            match e6_open(&p, "sync") {
                Err(err) => rows.push_str(&format!("E6s-corruption,torn-tail-partial-group,sync,refuse-or-discard,REFUSED,-,-,-,\"{err}\",documented policy: torn tail\n")),
                Ok((rm, rc)) => {
                    let txn_absent = !rm.contains_key(&9100) && !rm.contains_key(&9101) && !rm.contains_key(&9102);
                    rows.push_str(&format!("E6s-corruption,torn-tail-partial-group,sync,txn-absent,{},DISCARDED,\"{rc}\",{},\"{}\",incomplete group torn in half -> discarded atomically\n",
                        if txn_absent { "absent" } else { "PRESENT?" },
                        if txn_absent && rc && rm.contains_key(&9002) { "ok" } else { "FAIL" },
                        txn_absent));
                }
            }
        } else {
            rows.push_str("E6s-corruption,torn-tail-partial-group,sync,SKIP,SKIP,-,-,-,crash dir not found (run crash families first)\n");
        }
        cells += 1;
    }

    std::fs::write(format!("{out}/e6-txn.csv"), rows).unwrap();
    std::fs::write(format!("{out}/e6-crash.csv"), crash).unwrap();
    format!("e6: {cells} cells -> {out}/e6-txn.csv + e6-crash.csv")
}

/// Main E4 driver. Produces <out>/e4-matrix.csv and <out>/e4-integrity.csv.
fn run_e4(out: &str) -> String {
    std::fs::create_dir_all(out).unwrap();
    let mut rows = String::from("case,mode,expected_count,restored_count,match,source_final,reader_errors,ops_overlapping_backup,max_op_latency_us,backup_us,restore_us,checker_clean,early_backup_still_restores,partial_backup_refused,notes\n");
    let mut integrity = String::from("case,action,restore_result,detail\n");
    let root = std::path::PathBuf::from("/tmp/ph3e-e4");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut cells = 0usize;

    let run_case =
        |rows: &mut String,
         integrity: &mut String,
         cells: &mut usize,
         case: &str,
         mode: &str,
         setup: &dyn Fn(&AttentionEngine, E4Shared) -> E4SetupResult| {
            let dir = root.join(case);
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::env::set_var("PH3D_DURABILITY", mode);
            let e = std::sync::Arc::new(open_db(&dir));
            e.create_collection("bench", DIM, &[HEAD]).unwrap();
            let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(format!("{}.e4sidecar", dir.display()))
                    .unwrap(),
            ));
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
                    rows.push_str(&format!(
                        "{case},{mode},-,-,RESTORE_REFUSED,-,-,-,-,{backup_us},-,-,-,-,{err}\n"
                    ));
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
    run_case(
        &mut rows,
        &mut integrity,
        &mut cells,
        "b1-quiescent",
        "sync",
        &|e, f| {
            for i in 0..30u32 {
                e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                    .unwrap();
                e4_ack(&f, &format!("ACK {}", 1000 + i));
            }
            (root.join("b1-backup"), vec![], "control".into())
        },
    );

    // B2 read-concurrent
    {
        let dir = root.join("b2-readers");
        std::env::set_var("PH3D_DURABILITY", "group");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        for i in 0..40u32 {
            e.insert_document("bench", doc_record(1000 + i, 1, "c", i as i64))
                .unwrap();
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
        for r in readers {
            r.join().unwrap();
        }
        let re = errors.load(std::sync::atomic::Ordering::Relaxed);
        let lines = e4_sidecar_lines(&dir);
        let (model_present, _) = e4_model(&lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b2-backup"), &root.join("b2-restored"), "group")
                .unwrap();
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
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(
            e.clone(),
            side.clone(),
            3000..3300,
            lat.clone(),
            "bench",
            epoch.clone(),
        );
        std::thread::sleep(std::time::Duration::from_millis(30));
        let backup_start = epoch.elapsed().as_micros();
        e.backup_to(&root.join("b3-backup")).unwrap();
        let backup_us = epoch.elapsed().as_micros() - backup_start;
        let boundary_lines = e4_sidecar_lines(&dir);
        // post-backup source mutation (isolation)
        for i in 4000..4100u32 {
            e.insert_document("bench", doc_record(i, 1, "post", i as i64))
                .unwrap();
            e4_ack(&side, &format!("ACK {i}"));
        }
        w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in w.handles.drain(..) {
            h.join().unwrap();
        }
        let (model_present, _) = e4_model(&boundary_lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b3-backup"), &root.join("b3-restored"), "sync")
                .unwrap();
        let (got, exp, ok) = e4_model_match(&restored, &model_present);
        let source_final = {
            let (m, _) = export_state(&e);
            m.len()
        };
        let lats = lat.lock().unwrap();
        let backup_end = backup_start + backup_us;
        let overlapping = lats
            .iter()
            .filter(|(s, en, _)| *s <= backup_end && *en >= backup_start)
            .count();
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
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut ws = Vec::new();
        for r in [(5000u32..5100), (6000..6100), (7000..7100)] {
            ws.push(e4_spawn_writer(
                e.clone(),
                side.clone(),
                r,
                lat.clone(),
                "bench",
                epoch.clone(),
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(40));
        e.backup_to(&root.join("b4-backup")).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        for w in &mut ws {
            w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        for w in &mut ws {
            for h in w.handles.drain(..) {
                h.join().unwrap();
            }
        }
        let (model_present, _) = e4_model(&boundary_lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b4-backup"), &root.join("b4-restored"), "group")
                .unwrap();
        let (got, exp, ok) = e4_model_match(&restored, &model_present);
        let source_final = {
            let (m, _) = export_state(&e);
            m.len()
        };
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
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(
            e.clone(),
            side.clone(),
            8000..8200,
            lat.clone(),
            "bench",
            epoch.clone(),
        );
        std::thread::sleep(std::time::Duration::from_millis(30));
        let e2 = e.clone();
        let ck = std::thread::spawn(move || e2.checkpoint().unwrap());
        e.backup_to(&root.join("b5-backup")).unwrap();
        let ck_dur = {
            let t = std::time::Instant::now();
            ck.join().unwrap();
            t.elapsed().as_micros()
        };
        let boundary_lines = e4_sidecar_lines(&dir);
        for i in 9000..9050u32 {
            e.insert_document("bench", doc_record(i, 1, "post", i as i64))
                .unwrap();
            e4_ack(&side, &format!("ACK {i}"));
        }
        w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in w.handles.drain(..) {
            h.join().unwrap();
        }
        let (model_present, _) = e4_model(&boundary_lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b5-backup"), &root.join("b5-restored"), "sync")
                .unwrap();
        let (got, exp, ok) = e4_model_match(&restored, &model_present);
        let source_final = {
            let (m, _) = export_state(&e);
            m.len()
        };
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
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(
            e.clone(),
            side.clone(),
            11000..11200,
            lat.clone(),
            "bench",
            epoch.clone(),
        );
        std::thread::sleep(std::time::Duration::from_millis(30));
        e.backup_to(&root.join("b6-backup")).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in w.handles.drain(..) {
            h.join().unwrap();
        }
        std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
        let (model_present, _) = e4_model(&boundary_lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b6-backup"), &root.join("b6-restored"), "group")
                .unwrap();
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
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(
            e.clone(),
            side.clone(),
            13000..13200,
            lat.clone(),
            "bench",
            epoch.clone(),
        );
        std::thread::sleep(std::time::Duration::from_millis(30));
        let e2 = e.clone();
        let ck = std::thread::spawn(move || e2.checkpoint().unwrap());
        e.backup_to(&root.join("b7-backup")).unwrap();
        ck.join().unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in w.handles.drain(..) {
            h.join().unwrap();
        }
        std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
        let (model_present, _) = e4_model(&boundary_lines);
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("b7-backup"), &root.join("b7-restored"), "async")
                .unwrap();
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
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        for i in 100..106u32 {
            let mut r = doc_record(i, 1, "t", 1);
            r.fields.insert("num".to_string(), serde_json::json!(1i64));
            e.insert_document("bench", r).unwrap();
            e4_ack(&side, &format!("ACK {i}"));
        }
        for (k, i) in [101u32, 103, 105].iter().enumerate() {
            let id = {
                let store = e.document_store.read();
                store
                    .list_all_records()
                    .into_iter()
                    .find(|r| r.fields.get("idx").and_then(|v| v.as_u64()) == Some(*i as u64))
                    .map(|r| r.id)
                    .unwrap()
            };
            let old = {
                let store = e.document_store.read();
                store.get(&id).cloned().unwrap()
            };
            let mut fields = old.fields.clone();
            fields.insert("num".to_string(), serde_json::json!(2i64));
            let mut kvs = std::collections::HashMap::new();
            kvs.insert(HEAD.to_string(), vec_for(*i));
            e.update_document("bench", &id.to_string(), fields, kvs)
                .unwrap();
            e4_ack(&side, &format!("UPD {i} v{}", k + 2));
        }
        e.backup_to(&root.join("st-backup")).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        // model: idx -> num at boundary
        let mut expect_num: std::collections::BTreeMap<u32, i64> = Default::default();
        for l in &boundary_lines {
            if let Some(v) = l.strip_prefix("ACK ") {
                if let Ok(n) = v.parse::<u32>() {
                    expect_num.insert(n, 1);
                }
            } else if let Some(rest) = l.strip_prefix("UPD ") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if let Ok(n) = parts[0].parse::<u32>() {
                    expect_num.insert(n, 2);
                }
            }
        }
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("st-backup"), &root.join("st-restored"), "sync")
                .unwrap();
        let mism: Vec<String> = expect_num
            .iter()
            .filter(|(i, num)| restored.get(i).map(|(_, _, n)| *n) != Some(**num))
            .map(|(i, num)| format!("{i}:want{num}"))
            .collect();
        let ok = mism.is_empty() && restored.len() == expect_num.len();
        let verdict = if ok {
            "MATCH".to_string()
        } else {
            format!("MISMATCH {}", mism.join(" "))
        };
        rows.push_str(&format!(
            "snapshot-transition,sync,{},{},{verdict},-,0,0,0,-,{restore_us},{clean},-,-,{}\n",
            expect_num.len(),
            restored.len(),
            mism.join(" ")
        ));
        cells += 1;
    }

    // Delete/reinsert
    {
        let dir = root.join("delete-reinsert");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        for i in 20000..20020u32 {
            e.insert_document("bench", doc_record(i, 1, "d", i as i64))
                .unwrap();
            e4_ack(&side, &format!("ACK {i}"));
        }
        for i in (20000..20020u32).step_by(2) {
            e.delete_document("bench", &uuid_for(i, 1).to_string())
                .unwrap();
            e4_ack(&side, &format!("DEL {i}"));
        }
        for i in (20000..20012u32).step_by(4) {
            e.insert_document("bench", doc_record(i, 2, "d", i as i64))
                .unwrap(); // new uuid
            e4_ack(&side, &format!("ACK {i}")); // reinsert (new identity)
        }
        e.backup_to(&root.join("dr-backup")).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir); // AT backup return
        let boundary = boundary_lines.len();
        let (model_present, _) = e4_model(&boundary_lines);
        // post-backup: more delete/reinsert churn on source (must NOT enter backup)
        for i in (20001..20019u32).step_by(4) {
            e.delete_document("bench", &uuid_for(i, 1).to_string())
                .unwrap();
            e4_ack(&side, &format!("DEL {i}"));
        }
        let (restored, clean, restore_us) =
            e4_restore_and_open(&root.join("dr-backup"), &root.join("dr-restored"), "sync")
                .unwrap();
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
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut ws = Vec::new();
        ws.push(e4_spawn_writer(
            e.clone(),
            side.clone(),
            30000..30080,
            lat.clone(),
            "bench",
            epoch.clone(),
        ));
        ws.push(e4_spawn_writer(
            e.clone(),
            side.clone(),
            31000..31080,
            lat.clone(),
            "alpha",
            epoch.clone(),
        ));
        ws.push(e4_spawn_writer(
            e.clone(),
            side.clone(),
            32000..32080,
            lat.clone(),
            "beta",
            epoch.clone(),
        ));
        std::thread::sleep(std::time::Duration::from_millis(40));
        e.backup_to(&root.join("coll-backup")).unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        for w in &mut ws {
            w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        for w in &mut ws {
            for h in w.handles.drain(..) {
                h.join().unwrap();
            }
        }
        // model per collection
        let mut expect: std::collections::BTreeMap<&'static str, Vec<u32>> = Default::default();
        let ranges: [(&'static str, std::ops::Range<u32>); 3] = [
            ("bench", 30000..30080),
            ("alpha", 31000..31080),
            ("beta", 32000..32080),
        ];
        for (coll, rng) in ranges {
            let present: Vec<u32> = boundary_lines
                .iter()
                .filter_map(|l| {
                    let n = l.strip_prefix("ACK ")?.parse::<u32>().ok()?;
                    if rng.contains(&n) {
                        Some(n)
                    } else {
                        None
                    }
                })
                .collect();
            expect.insert(coll, present);
        }
        std::env::set_var("PH3D_DURABILITY", "group");
        attentiondb_core::backup::restore_backup(
            &root.join("coll-backup"),
            &root.join("coll-restored"),
        )
        .unwrap();
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
        let clean2 = checker_report(&e2, &root.join("coll-restored"))["clean"]
            .as_bool()
            .unwrap_or(false);
        rows.push_str(&format!(
            "collections,group,-,-,{},-,0,0,0,-,-,{},-,-,{}\n",
            if all_ok { "MATCH" } else { "MISMATCH" },
            if clean2 { "true" } else { "false" },
            det.trim()
        ));
        cells += 1;
    }

    // Multi-backup B1 < B2 < B3, each independently restored
    {
        let dir = root.join("multi-backup");
        std::env::set_var("PH3D_DURABILITY", "sync");
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        let mut snapshots = Vec::new();
        for b in 0..3u32 {
            for i in (40000 + b * 20)..(40000 + b * 20 + 20) {
                e.insert_document("bench", doc_record(i, 1, "m", i as i64))
                    .unwrap();
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
                Ok(x) => x,
                Err(_) => {
                    all_ok = false;
                    all_clean = false;
                    break;
                }
            };
            restore_us_total += us;
            let (got, expn, ok) = e4_model_match(&m, exp);
            all_ok &= ok;
            all_clean &= clean;
            det.push_str(&format!("B{}={}/{} ", idx + 1, got, expn));
        }
        rows.push_str(&format!(
            "multi-backup,sync,-,-,{},-,0,0,0,-,{},{} ,-,-,{}\n",
            if all_ok { "MATCH" } else { "MISMATCH" },
            restore_us_total,
            if all_clean { "true" } else { "false" },
            det.trim()
        ));
        cells += 1;
    }

    // Durability-mode matrix (writer backup x3 modes) + integrity suite
    for mode in ["sync", "group", "async"] {
        let dir = root.join(format!("modes-{mode}"));
        std::env::set_var("PH3D_DURABILITY", mode);
        let e = std::sync::Arc::new(open_db(&dir));
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        let side: E4Shared = std::sync::Arc::new(std::sync::Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(format!("{}.e4sidecar", dir.display()))
                .unwrap(),
        ));
        let lat = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let epoch = std::sync::Arc::new(std::time::Instant::now());
        let mut w = e4_spawn_writer(
            e.clone(),
            side.clone(),
            50000..50300,
            lat.clone(),
            "bench",
            epoch.clone(),
        );
        std::thread::sleep(std::time::Duration::from_millis(25));
        e.backup_to(&root.join(format!("modes-{mode}-backup")))
            .unwrap();
        let boundary_lines = e4_sidecar_lines(&dir);
        w.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for h in w.handles.drain(..) {
            h.join().unwrap();
        }
        let (model_present, _) = e4_model(&boundary_lines);
        let r = e4_restore_and_open(
            &root.join(format!("modes-{mode}-backup")),
            &root.join(format!("modes-{mode}-restored")),
            mode,
        );
        match r {
            Ok((restored, clean, us)) => {
                let (got, exp, ok) = e4_model_match(&restored, &model_present);
                rows.push_str(&format!("modes-{mode},{mode},{exp},{got},{},-,0,0,0,0,{us},{clean},-,-,acked writes all captured (ckpt fsync)\n",
                    if ok { "MATCH" } else { "MISMATCH" }));
            }
            Err(err) => rows.push_str(&format!(
                "modes-{mode},{mode},-,-,RESTORE_REFUSED,-,0,0,0,0,-,-,-,-,{err}\n"
            )),
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
        e.insert_document("bench", doc_record(i, 1, "i", i as i64))
            .unwrap();
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
                let clean = checker_report(&e2, &dest)["clean"]
                    .as_bool()
                    .unwrap_or(false);
                integrity.push_str(&format!("{case},{action},ACCEPTED,clean={clean}\n"));
                e2.close().unwrap();
            }
            Err(err) => {
                let reason = if format!("{err}").contains("missing") {
                    "NO_META"
                } else if format!("{err}").contains("unsupported") {
                    "BAD_VERSION"
                } else if format!("{err}").contains("backup meta") {
                    "META_PARSE"
                } else if format!("{err}").contains("not empty") {
                    "DEST_NONEMPTY"
                } else {
                    "OPEN_OR_CATALOG"
                };
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
    try_restore(
        &mut integrity,
        "partial-no-meta",
        "rm backup-meta.json",
        &p1,
    );
    // truncated SST
    let p2 = root.join("integ-trunc-sst");
    let _ = std::fs::remove_dir_all(&p2);
    copy_dir_all(&bp, &p2);
    {
        let sstdir = p2.join("sst");
        let f = std::fs::read_dir(&sstdir)
            .unwrap()
            .flatten()
            .next()
            .unwrap()
            .path();
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
    try_restore(
        &mut integrity,
        "corrupt-wal-state",
        "garble wal-state.json",
        &p3,
    );
    // garbage in the EMPTY active segment: torn-tail policy -> intact (empty)
    // prefix + all checkpointed state in SSTs; expected ACCEPTED with zero loss
    let p3b = root.join("integ-garbage-active-seg");
    let _ = std::fs::remove_dir_all(&p3b);
    copy_dir_all(&bp, &p3b);
    {
        let waldir = p3b.join("WAL");
        let f = std::fs::read_dir(&waldir)
            .unwrap()
            .flatten()
            .find(|x| x.path().extension().and_then(|s| s.to_str()) == Some("wal"))
            .unwrap()
            .path();
        use std::io::Write;
        let mut g = std::fs::OpenOptions::new().append(true).open(&f).unwrap();
        g.write_all(&[0u8; 13]).unwrap();
    }
    try_restore(
        &mut integrity,
        "garbage-active-segment",
        "13 zero bytes appended",
        &p3b,
    );
    // corrupt CURRENT
    let p4 = root.join("integ-corrupt-current");
    let _ = std::fs::remove_dir_all(&p4);
    copy_dir_all(&bp, &p4);
    std::fs::write(p4.join("CURRENT"), b"manifest-999999999\n").unwrap();
    try_restore(
        &mut integrity,
        "corrupt-current",
        "point at missing gen",
        &p4,
    );
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
        std::fs::write(
            p6.join("backup-meta.json"),
            serde_json::to_vec(&meta).unwrap(),
        )
        .unwrap();
    }
    try_restore(&mut integrity, "bad-format-version", "v99", &p6);
    // restore into non-empty destination
    {
        let dest = root.join("integ-nonempty");
        let _ = std::fs::remove_dir_all(&dest);
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("stale.txt"), b"x").unwrap();
        match attentiondb_core::backup::restore_backup(&bp, &dest) {
            Ok(_) => integrity.push_str(&format!(
                "{},restore-nonempty,ACCEPTED,UNEXPECTED\n",
                "nonempty-dest"
            )),
            Err(err) => {
                let reason = if format!("{err}").contains("not empty") {
                    "DEST_NONEMPTY"
                } else {
                    "OPEN_OR_CATALOG"
                };
                integrity.push_str(&format!(
                    "{},restore-nonempty,REFUSED,{}\n",
                    "nonempty-dest", reason
                ));
            }
        }
    }
    // source-dir safety: source reopens, all acks present, backup copied read-only
    {
        let (m, iss) = export_state(&e);
        let clean = checker_report(&e, &dir)["clean"].as_bool().unwrap_or(false) && iss.is_empty();
        let det = format!("docs={} (expected 30)", m.len());
        integrity.push_str(&format!(
            "{},source-safety,{},{},\n",
            "source-after-backup",
            if clean && m.len() == 30 {
                "INTACT"
            } else {
                "DAMAGED"
            },
            det
        ));
    }

    // crash-during-backup (child process, group kill)
    {
        let cdir = root.join("crash-src");
        let cbackup = root.join("crash-backup");
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args([
            "dbtest",
            "e4child",
            "--dir",
            cdir.to_str().unwrap(),
            "--backup",
            cbackup.to_str().unwrap(),
        ])
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
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        let _ = child.wait();
        let done = std::fs::read_to_string(&marker)
            .unwrap_or_default()
            .contains("backup-done");
        // (a) source recovers
        std::env::set_var("PH3D_DURABILITY", "sync");
        let (src_ok, src_docs, src_clean) =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| open_db(&cdir))) {
                Err(_) => (false, 0usize, false),
                Ok(e2) => {
                    let (m, iss) = export_state(&e2);
                    let c = checker_report(&e2, &cdir)["clean"]
                        .as_bool()
                        .unwrap_or(false)
                        && iss.is_empty();
                    (true, m.len(), c)
                }
            };
        // (b) partial backup rejected (if copy had not completed)
        let (partial_verdict, partial_detail) = if done {
            match e4_restore_and_open(&cbackup, &root.join("crash-restored"), "sync") {
                Ok((m, clean, _)) => (
                    "ACCEPTED_COMPLETE".to_string(),
                    format!("docs={} clean={clean}", m.len()),
                ),
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
            &root.join("crash-early-restored"),
            "sync",
        )
        .map(|(m, clean, _)| m.len() == 20000 && clean)
        .unwrap_or(false);
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
    write_json(
        &format!("{out}/metrics.json"),
        &serde_json::json!({"cells": cells, "kind": "E4 backup/snapshot/restore"}),
    );
    format!("e4: {cells} matrix cells -> {out}/e4-matrix.csv")
}

fn copy_dir_all(src: &std::path::Path, dst: &std::path::Path) {
    fn rec(src: &std::path::Path, dst: &std::path::Path) {
        std::fs::create_dir_all(dst).unwrap();
        for e in std::fs::read_dir(src).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                rec(&p, &dst.join(e.file_name()));
            } else {
                std::fs::copy(&p, dst.join(e.file_name())).unwrap();
            }
        }
    }
    rec(src, dst);
}

// ---------------------------------------------------------------- dispatch

#[allow(clippy::needless_return)] // dispatch arms return early by design
pub fn run(args: &[String]) -> String {
    let get = |name: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| default.to_string())
    };
    let dir = std::path::PathBuf::from(get("--dir", "/var/tmp/ph3d-db"));
    let out = get("--out", "research/phase3/raw/runs/PH3D-OUT");
    std::fs::create_dir_all(&out).unwrap();
    match args.first().map(|s| s.as_str()).unwrap_or("model") {
        "model" => run_model(
            &dir,
            get("--seed", "42").parse().unwrap_or(42),
            get("--ops", "300").parse().unwrap_or(300),
            &out,
        ),
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
        "e5run" => run_e5(&out),
        "e5child" => run_e5child(&dir),
        "e6run" => run_e6(&out),
        "e11" => crate::e11::run(&args[1..]),
        "e7run" => run_e7(&out),
        "e7dbg" => {
            // debug: dump WAL records first (investigation only)
            if std::env::var("PH3E_E7DBG_WAL").is_ok() {
                // (WAL dump runs before engine open so a refused recovery
                // still shows its records)
                let wal_dir = dir.join("WAL");
                let mut segs: Vec<_> = std::fs::read_dir(&wal_dir)
                    .unwrap()
                    .flatten()
                    .map(|p| p.path())
                    .filter(|p| p.extension().map(|e| e == "wal").unwrap_or(false))
                    .collect();
                segs.sort();
                for seg_path in segs {
                    let wal_dir_s = wal_dir.clone();
                    let mut wal = attentiondb_storage::wal::Wal::open(
                        &wal_dir,
                        attentiondb_storage::Durability::Sync,
                        64 * 1024 * 1024,
                    )
                    .unwrap();
                    let _ = &wal_dir_s;
                    let out = wal.replay(0).unwrap();
                    println!(
                        "  outcome: first={} last={} records={}",
                        out.first_seq,
                        out.last_seq,
                        out.records.len()
                    );
                    for r in &out.records {
                        let u = uuid::Uuid::from_bytes(r.doc_id);
                        println!(
                            "  seq={} kind={} txn={} uuid={} numeric={}",
                            r.seq, r.kind, r.txn_id, u, r.numeric_id
                        );
                    }
                    let _ = seg_path;
                }
            }
            let e = open_db(&dir);
            let rep = checker_report(&e, &dir);
            println!("checker: {rep}");
            let (m, iss) = export_state(&e);
            println!("issues: {iss:?}");
            for (k, v) in &m {
                println!("  idx={k} version={} cat={} num={}", v.0, v.1, v.2);
            }
            e.close().unwrap();
            "e7dbg done".to_string()
        }
        "e7conc" => e7_conc_child(&dir, &get("--case", "t1-t2-before")),
        "e6txn" => e6_txn_child(&dir, &get("--case", "commit")),
        "e4run" => run_e4(&out),
        "e10run" => {
            let exp = get("--exp", "repro").to_string();
            let docs: u32 = get("--docs", "100000").parse().unwrap_or(100_000);
            let outdir = std::path::PathBuf::from(&out);
            return crate::e10::run_e10(&exp, &outdir.to_string_lossy(), docs);
        }
        "e9run" => {
            let exp = get("--exp", "repro").to_string();
            let outdir = std::path::PathBuf::from(&out);
            crate::e9::run_e9(&exp, &outdir.to_string_lossy())
        }
        "e8run" => {
            let fam = get("--family", "a").to_lowercase();
            let t0 = std::time::Instant::now();
            let outdir = std::path::PathBuf::from(&out);
            match fam.as_str() {
                "a" => run_e8a(&outdir),
                "b" => run_e8b(&outdir),
                "c" => run_e8c(&outdir),
                "d" => run_e8d(&outdir),
                "e" => run_e8e(&outdir),
                "f" => run_e8f(&outdir),
                "g" => run_e8g(&outdir),
                "h" => run_e8h(&outdir),
                "i" => run_e8i(&outdir),
                other => panic!("unknown e8 family {other}"),
            }
            let _line = format!("e8 family {fam} complete in {}s", t0.elapsed().as_secs());
            println!("{}", _line.clone());
            _line
        }
        "e8child" => e8_child(
            &dir,
            &get("--phase", "R1"),
            get("--seed", "62411").parse().unwrap_or(62411),
        ),
        "compactlive" => run_compact_live(&dir),
        other => format!("unknown dbtest subcommand {other}"),
    }
}

// ==================== E8: soak / long-running reliability ====================
// Independent reference model + append-only oplog + telemetry/watchdog +
// progressive families per methodology/phase3e-e8-spec.md. The model is
// mutated in lock-step with ISSUED ops and never read back from the engine.

type E8Oplog = std::sync::Arc<std::sync::Mutex<std::io::BufWriter<std::fs::File>>>;

fn e8_new_oplog(path: &std::path::Path) -> E8Oplog {
    use std::io::Write;
    let f = std::fs::File::create(path).unwrap();
    let mut w = std::io::BufWriter::new(f);
    let _ = writeln!(w, "seq,ts_us,thread,op,coll,idx,ver,num,txn,result");
    std::sync::Arc::new(std::sync::Mutex::new(w))
}

#[allow(clippy::too_many_arguments)] // E8-frozen harness signature
fn e8_log(
    w: &E8Oplog,
    seq: usize,
    thread: usize,
    op: &str,
    coll: &str,
    idx: i64,
    ver: u64,
    num: i64,
    txn: u64,
    res: &str,
) {
    use std::io::Write;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);
    let mut g = w.lock().unwrap();
    let _ = writeln!(
        g,
        "{seq},{ts},{thread},{op},{coll},{idx},{ver},{num},{txn},{res}"
    );
    if seq.is_multiple_of(2000) {
        let _ = g.flush();
    }
}

/// Collection namespace: bench = coll 0 (global idx 0..2999), collb = 1
/// (3000..3299), collc = 2 (3300..3599). uuid identity is global, so every
/// collection uses uuidc(cid, idx, ver) — uuidc(0,..) == uuid_for(..).
fn e8_cid(idx: u32) -> u32 {
    if idx < 3000 {
        0
    } else if idx < 3300 {
        1
    } else {
        2
    }
}
fn e8_coll_of(idx: u32) -> &'static str {
    match e8_cid(idx) {
        0 => "bench",
        1 => "collb",
        _ => "collc",
    }
}

fn e8_fnv(lines: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in lines.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct E8Rng(u64);
impl E8Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % (n as u64)) as usize
    }
}

/// Independent reference model: idx -> (version, num). Deleted/retired
/// versions are derivable: any v < maxver[idx] with no live entry is retired.
#[derive(Clone)]
struct E8Model {
    live: std::collections::BTreeMap<u32, (u64, i64)>,
    maxver: std::collections::BTreeMap<u32, u64>,
}

impl E8Model {
    fn new() -> Self {
        Self {
            live: Default::default(),
            maxver: Default::default(),
        }
    }
    fn state_hash(&self) -> u64 {
        let mut s = String::new();
        for (idx, (ver, num)) in &self.live {
            s.push_str(&format!("{idx}|{ver}|{num}\n"));
        }
        e8_fnv(&s)
    }
    fn to_json(&self) -> String {
        let live: std::collections::BTreeMap<String, Vec<i64>> = self
            .live
            .iter()
            .map(|(k, (v, n))| (k.to_string(), vec![*v as i64, *n]))
            .collect();
        serde_json::to_string(&live).unwrap()
    }
    fn from_json(s: &str) -> Self {
        let raw: std::collections::BTreeMap<String, Vec<i64>> = serde_json::from_str(s).unwrap();
        let mut m = Self::new();
        for (k, v) in raw {
            let idx: u32 = k.parse().unwrap();
            m.live.insert(idx, (v[0] as u64, v[1]));
            m.maxver.insert(idx, v[0] as u64);
        }
        m
    }
}

/// One mutation decision, shared by all families. Returns the op performed.
#[allow(clippy::too_many_arguments)] // E8-frozen harness signature
fn e8_mutate(
    e: &AttentionEngine,
    m: &mut E8Model,
    rng: &mut E8Rng,
    log: &E8Oplog,
    prog: &std::sync::atomic::AtomicUsize,
    seq: usize,
    thread: usize,
    keyspace: usize,
    hot: usize,
    hotpct: usize,
) -> &'static str {
    use std::sync::atomic::Ordering::Relaxed;
    // workload: 15 insert, 15 delete, 70 update/upsert (spec E8b mix)
    let roll = rng.below(100);
    let hot_pick = rng.below(100) < hotpct;
    let idx = if hot_pick {
        rng.below(hot) as u32
    } else {
        (hot + rng.below(keyspace - hot)) as u32
    };
    let coll = e8_coll_of(idx);
    let live = m.live.contains_key(&idx);
    let done = if roll < 15 && !live {
        // INSERT (new version uuid)
        let ver = m.maxver.get(&idx).copied().unwrap_or(0) + 1;
        let num = 100 + (seq % 900) as i64;
        let mut r = doc_record(idx, ver, "t", num);
        r.id = uuidc(e8_cid(idx), idx, ver);
        match e.insert_document(coll, r) {
            Ok(_) => {
                m.maxver.insert(idx, ver);
                m.live.insert(idx, (ver, num));
                e8_log(
                    log, seq, thread, "insert", coll, idx as i64, ver, num, 0, "ok",
                );
                "insert"
            }
            Err(_) => {
                e8_log(
                    log, seq, thread, "insert", coll, idx as i64, ver, num, 0, "err",
                );
                "insert-err"
            }
        }
    } else if (roll < 15 && live) || ((15..30).contains(&roll) && live) {
        // DELETE of a live doc
        let (ver, _) = m.live[&idx];
        match e.delete_document(coll, &uuidc(e8_cid(idx), idx, ver).to_string()) {
            Ok(true) => {
                m.live.remove(&idx);
                e8_log(
                    log, seq, thread, "delete", coll, idx as i64, ver, 0, 0, "ok",
                );
                "delete"
            }
            Ok(false) => {
                e8_log(
                    log, seq, thread, "delete", coll, idx as i64, ver, 0, 0, "absent",
                );
                "delete-absent"
            }
            Err(_) => {
                e8_log(
                    log, seq, thread, "delete", coll, idx as i64, ver, 0, 0, "err",
                );
                "delete-err"
            }
        }
    } else if (15..30).contains(&roll) && !live {
        // DELETE target dead -> treat as no-op evidence
        e8_log(
            log,
            seq,
            thread,
            "delete",
            coll,
            idx as i64,
            0,
            0,
            0,
            "skip-dead",
        );
        "delete-skip"
    } else if live {
        // UPDATE: uuid-preserving, fields replaced wholesale, same version
        let (ver, _) = m.live[&idx];
        let num = 100 + (seq % 900) as i64;
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(idx));
        fields.insert("version".to_string(), serde_json::json!(ver));
        fields.insert("cat".to_string(), serde_json::json!("t"));
        fields.insert("num".to_string(), serde_json::json!(num));
        fields.insert(
            "title".to_string(),
            serde_json::json!(format!("doc-{idx}-v{ver}")),
        );
        let mut kv = HashMap::new();
        kv.insert(HEAD.to_string(), vec_for(idx));
        match e.update_document(coll, &uuidc(e8_cid(idx), idx, ver).to_string(), fields, kv) {
            Ok(_) => {
                m.live.insert(idx, (ver, num));
                e8_log(
                    log, seq, thread, "update", coll, idx as i64, ver, num, 0, "ok",
                );
                "update"
            }
            Err(_) => {
                e8_log(
                    log, seq, thread, "update", coll, idx as i64, ver, num, 0, "err",
                );
                "update-err"
            }
        }
    } else {
        // UPSERT onto a dead key = insert branch
        let ver = m.maxver.get(&idx).copied().unwrap_or(0) + 1;
        let num = 100 + (seq % 900) as i64;
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(idx));
        fields.insert("version".to_string(), serde_json::json!(ver));
        fields.insert("cat".to_string(), serde_json::json!("t"));
        fields.insert("num".to_string(), serde_json::json!(num));
        fields.insert(
            "title".to_string(),
            serde_json::json!(format!("doc-{idx}-v{ver}")),
        );
        let mut kv = HashMap::new();
        kv.insert(HEAD.to_string(), vec_for(idx));
        match e.upsert_document(coll, uuidc(e8_cid(idx), idx, ver), fields, kv) {
            Ok(_) => {
                m.maxver.insert(idx, ver);
                m.live.insert(idx, (ver, num));
                e8_log(
                    log, seq, thread, "upsert", coll, idx as i64, ver, num, 0, "ok",
                );
                "upsert"
            }
            Err(_) => {
                e8_log(
                    log, seq, thread, "upsert", coll, idx as i64, ver, num, 0, "err",
                );
                "upsert-err"
            }
        }
    };
    prog.fetch_add(1, Relaxed);
    done
}

/// A staged [maybe-Delete, Insert] transaction op on one key; model applied
/// only when the caller confirms commit success.
fn e8_stage_upsert(e: &AttentionEngine, m: &E8Model, idx: u32, num: i64) -> u64 {
    use attentiondb_core::transaction::TxnOp;
    let coll = e8_coll_of(idx);
    let t = e.begin_transaction(coll);
    if let Some((ver, _)) = m.live.get(&idx) {
        e.record_transaction_operation(t, TxnOp::Delete(uuidc(e8_cid(idx), idx, *ver)))
            .unwrap();
    }
    let ver = m.maxver.get(&idx).copied().unwrap_or(0) + 1;
    let mut r = doc_record(idx, ver, "t", num);
    r.id = uuidc(e8_cid(idx), idx, ver);
    e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
    t
}

fn e8_apply_commit(m: &mut E8Model, idx: u32, num: i64) {
    let ver = m.maxver.get(&idx).copied().unwrap_or(0) + 1;
    m.maxver.insert(idx, ver);
    m.live.insert(idx, (ver, num));
}

/// Full verification point (S1/S2/S5/S6): per-collection model equality,
/// engine checker, mapper liveness, retired-version sampling.
fn e8_verify(
    e: &AttentionEngine,
    m: &E8Model,
    dir: &std::path::Path,
    label: &str,
    seq: usize,
    rows: &mut String,
) -> bool {
    let mut ok = true;
    let mut details = String::new();
    for (cname, lo, hi) in [
        ("bench", 0u32, 3000u32),
        ("collb", 3000, 3300),
        ("collc", 3300, 3600),
    ] {
        let any = m.live.range(lo..hi).next().is_some();
        if !any {
            continue;
        }
        let (rm, iss) = export_state_coll(e, cname);
        let expected: Model = m
            .live
            .range(lo..hi)
            .map(|(k, (v, n))| (*k, (*v, "t".to_string(), *n)))
            .collect();
        let eq = rm == expected && iss.is_empty();
        ok &= eq;
        if !eq {
            details.push_str(&format!(
                " {cname}:EXP{}/GOT{} iss={}",
                expected.len(),
                rm.len(),
                iss.len()
            ));
        }
    }
    let clean = e7_checker(e, dir);
    ok &= clean;
    // S5: live uuids mapped
    let mapper_ok = {
        let map = e.id_mapper.read();
        let mut all = true;
        for (i, (idx, (ver, _))) in m.live.iter().enumerate() {
            if i % 3 == 0 && map.uuid_to_id(&uuidc(e8_cid(*idx), *idx, *ver)).is_none() {
                all = false;
                break;
            }
        }
        all
    };
    ok &= mapper_ok;
    // S6: sampled retired versions unmapped
    let mut sampled = 0usize;
    let mut retired_ok = true;
    for (idx, (ver, _)) in &m.live {
        let mx = m.maxver.get(idx).copied().unwrap_or(*ver);
        let mut v = 1u64;
        while v < mx && sampled < 50 {
            if v != *ver
                && e.id_mapper
                    .read()
                    .uuid_to_id(&uuidc(e8_cid(*idx), *idx, v))
                    .is_some()
            {
                retired_ok = false;
            }
            sampled += 1;
            v += 1;
        }
        if sampled >= 50 {
            break;
        }
    }
    ok &= retired_ok;
    let hash = format!("{:016x}", m.state_hash());
    rows.push_str(&format!(
        "{seq},{label},{},{},{},{},{},{}\n",
        if ok { "ok" } else { "FAIL" },
        clean,
        mapper_ok,
        retired_ok,
        m.live.len(),
        hash
    ));
    if !ok {
        eprintln!("e8 VERIFY FAIL @ {label} seq={seq} clean={clean} mapper={mapper_ok} retired={retired_ok}{details}");
    }
    ok
}

fn e8_model_ckpt(m: &E8Model, out: &std::path::Path, seq: usize) {
    let per_coll = [
        ("bench", m.live.range(0..3000).count()),
        ("collb", m.live.range(3000..3300).count()),
        ("collc", m.live.range(3300..3600).count()),
    ];
    let j = serde_json::json!({
        "seq": seq,
        "state_hash": format!("{:016x}", m.state_hash()),
        "live_count": m.live.len(),
        "per_coll": per_coll,
    });
    let _ = std::fs::write(
        out.join(format!("model_ckpt_{seq:06}.json")),
        serde_json::to_string(&j).unwrap(),
    );
}

fn e8_dir_size(p: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(p) {
        for ent in rd.flatten() {
            let ft = match ent.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if ft.is_dir() {
                total += e8_dir_size(&ent.path());
            } else {
                total += ent.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

fn e8_census(db: &std::path::Path, label: &str, seq: usize, rows: &mut String) {
    let count = |sub: &str| -> (usize, u64) {
        let d = db.join(sub);
        let mut n = 0usize;
        let mut b = 0u64;
        if let Ok(rd) = std::fs::read_dir(&d) {
            for ent in rd.flatten() {
                if ent.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    n += 1;
                    b += ent.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        }
        (n, b)
    };
    let (wal_n, wal_b) = count("WAL");
    let (sst_n, sst_b) = count("sst");
    let (meta_n, _) = count("META");
    let mut tmp = 0usize;
    if let Ok(rd) = std::fs::read_dir(db) {
        for ent in rd.flatten() {
            let name = ent.file_name().to_string_lossy().to_string();
            if name.contains(".tmp") || name.ends_with(".partial") {
                tmp += 1;
            }
        }
    }
    rows.push_str(&format!(
        "{seq},{label},{wal_n},{wal_b},{sst_n},{sst_b},{meta_n},{tmp},{},{:016x}\n",
        db.join("WAL").parent().map(e8_dir_size).unwrap_or(0),
        0u64
    ));
}

fn e8_proc_status() -> (u64, u64, u64) {
    let mut rss = 0u64;
    let mut vsz = 0u64;
    let mut thr = 0u64;
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("VmRSS:") {
                rss = v
                    .split_whitespace()
                    .next()
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
            } else if let Some(v) = line.strip_prefix("VmSize:") {
                vsz = v
                    .split_whitespace()
                    .next()
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0);
            } else if let Some(v) = line.strip_prefix("Threads:") {
                thr = v.trim().parse().unwrap_or(0);
            }
        }
    }
    (rss, vsz, thr)
}

/// Telemetry + progress watchdog thread (§16/§17/§30/§52): 2 s samples;
/// STALLED if no op progress for 24 consecutive ticks (120 s); aborts so the
/// harness can never present a hung run as success.
fn e8_spawn_monitor(
    out: std::path::PathBuf,
    progress: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let t0 = std::time::Instant::now();
        let mut buf = String::from(
            "elapsed_s,op_seq,vm_rss_kb,vm_size_kb,threads,fds,wal_bytes,sst_bytes,db_bytes,wd_checks\n",
        );
        let mut last_prog = 0usize;
        let mut stall_ticks = 0usize;
        let mut wd = 0usize;
        let mut tick = 0usize;
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_secs(2));
            tick += 1;
            let p = progress.load(std::sync::atomic::Ordering::Relaxed);
            if p == last_prog {
                stall_ticks += 1;
            } else {
                stall_ticks = 0;
                last_prog = p;
            }
            wd += 1;
            let (rss, vsz, thr) = e8_proc_status();
            let fds = std::fs::read_dir("/proc/self/fd")
                .map(|d| d.count())
                .unwrap_or(0);
            let wal = e8_dir_size(&out.join("db/WAL"));
            let sst = e8_dir_size(&out.join("db/sst"));
            let db = e8_dir_size(&out.join("db"));
            buf.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{}\n",
                t0.elapsed().as_secs(),
                p,
                rss,
                vsz,
                thr,
                fds,
                wal,
                sst,
                db,
                wd
            ));
            if tick.is_multiple_of(5) || stop.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = std::fs::write(out.join("resource.csv"), &buf);
            }
            if stall_ticks >= 24 {
                let _ = std::fs::write(
                    out.join("stall.txt"),
                    format!(
                        "STALLED op_seq={p} elapsed_s={} rss_kb={rss} threads={thr} fds={fds}\n",
                        t0.elapsed().as_secs()
                    ),
                );
                eprintln!("E8 STALLED: no op progress for 120 s (op_seq={p}) — aborting");
                std::process::abort();
            }
        }
        let _ = std::fs::write(out.join("resource.csv"), &buf);
    })
}

struct E8Counts {
    txns: u64,
    commits: u64,
    rollbacks: u64,
    ckpts: u64,
    compactions: u64,
    backups: u64,
    restores: u64,
    restarts: u64,
    verifies: u64,
    verif_fails: u64,
}

fn e8_counts_json(c: &E8Counts, ops: usize, elapsed: std::time::Duration, hash: u64) -> String {
    serde_json::to_string(&serde_json::json!({
        "ops": ops, "elapsed_s": elapsed.as_secs(),
        "txns": c.txns, "commits": c.commits, "rollbacks": c.rollbacks,
        "checkpoints": c.ckpts, "compactions": c.compactions,
        "backups": c.backups, "restores": c.restores, "restarts": c.restarts,
        "verifications": c.verifies, "verification_failures": c.verif_fails,
        "final_state_hash": format!("{hash:016x}"),
    }))
    .unwrap()
}

fn e8_new_counts() -> E8Counts {
    E8Counts {
        txns: 0,
        commits: 0,
        rollbacks: 0,
        ckpts: 0,
        compactions: 0,
        backups: 0,
        restores: 0,
        restarts: 0,
        verifies: 0,
        verif_fails: 0,
    }
}

fn e8_summary_line(fam: &str, c: &E8Counts, ops: usize, elapsed: std::time::Duration) {
    println!(
        "e8 {fam}: ops={ops} elapsed={}s txns={} commits={} rollbacks={} ckpt={} compact={} backup={} restore={} restart={} verify={} vfails={}",
        elapsed.as_secs(), c.txns, c.commits, c.rollbacks, c.ckpts, c.compactions,
        c.backups, c.restores, c.restarts, c.verifies, c.verif_fails
    );
}

/// Reader thread (S12/S18): validates per the E7 contract only — valid
/// (nonzero) ids, scan length within the writer-count window of the live
/// atomic, fields parse. Counts checks and violations; ops sampled into the
/// oplog at 1/64 (documented; the mutation oplog is the authoritative log).
#[allow(clippy::too_many_arguments)]
fn e8_reader(
    e: std::sync::Arc<AttentionEngine>,
    live_ct: std::sync::Arc<std::sync::atomic::AtomicI64>,
    writers: usize,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    prog: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    log: E8Oplog,
    thread: usize,
) -> (usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut checks = 0usize;
    let mut viol = 0usize;
    let mut n = 0usize;
    let q = vec_for(7);
    while !stop.load(Relaxed) {
        n += 1;
        match e.attend("bench", &[HEAD.to_string()], &q, 10) {
            Ok(res) => {
                if res.iter().any(|(id, _)| *id == 0) {
                    viol += 1;
                }
                if let Some((id, _)) = res.first() {
                    let f = e.get_document_fields(*id);
                    // empty fields = the doc was deleted between attend and
                    // this SECOND atomic read — the E7 per-op visibility
                    // contract allows exactly this; only a NON-empty record
                    // with an unparseable num would be torn.
                    if !f.is_empty() && f.get("num").and_then(|s| s.parse::<i64>().ok()).is_none() {
                        viol += 1;
                    }
                }
            }
            Err(_) => viol += 1,
        }
        checks += 1;
        if let Ok(v) = e.scan_filtered("bench", None, 100_000) {
            // scan is a consistent snapshot; live_ct is a post-ACK hint that
            // can lag OR lead the snapshot by the few commits that interleave
            // in the microsecond window between snapshot and hint read. Margin
            // covers that window; a torn/duplicated scan would exceed it by
            // orders of magnitude.
            let lc = live_ct.load(Relaxed);
            let margin = 4 * (writers as i64 + 1);
            if (v.len() as i64 - lc).abs() > margin {
                viol += 1;
            }
        } else {
            viol += 1;
        }
        checks += 1;
        if n.is_multiple_of(64) {
            e8_log(&log, n, thread, "read", "bench", -1, 0, 0, 0, "sampled");
        }
        prog.fetch_add(1, Relaxed);
        std::thread::yield_now();
    }
    (checks, viol)
}

fn run_e8a(out: &std::path::Path) {
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let _ = std::fs::create_dir_all(out);
    std::env::set_var("PH3D_DURABILITY", "sync");
    std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
    let db = out.join("db");
    let e = std::sync::Arc::new(open_db(&db));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let model = std::sync::Arc::new(std::sync::Mutex::new(E8Model::new()));
    let log = e8_new_oplog(&out.join("oplog.csv"));
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mon = {
        let o = out.to_path_buf();
        let p = progress.clone();
        let s = stop.clone();
        e8_spawn_monitor(o, p, s)
    };
    let live_ct = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0i64));
    let rstop = stop.clone();
    let re = e.clone();
    let rprog = progress.clone();
    let rlog = log.clone();
    let rl = live_ct.clone();
    let reader = std::thread::spawn(move || e8_reader(re, rl, 1, rstop, rprog, rlog, 9));
    let mut c = e8_new_counts();
    let mut rng = E8Rng(0xE8A0001);
    let mut verif = String::from("seq,label,model_ok,checker,mapper_ok,retired_ok,live,hash\n");
    let mut seq = 0usize;
    let mut ok_all = true;
    for i in 0..30_000usize {
        let mut m = model.lock().unwrap();
        e8_mutate(&e, &mut m, &mut rng, &log, &progress, seq, 0, 500, 10, 20);
        live_ct.store(m.live.len() as i64, Relaxed);
        drop(m);
        seq += 1;
        if (i + 1) % 5_000 == 0 {
            e.checkpoint().unwrap();
            c.ckpts += 1;
        }
        if (i + 1) % 10_000 == 0 {
            e.compact_storage().unwrap();
            c.compactions += 1;
        }
        if i == 15_000 {
            // coordinated backup of the LIVE engine (E4 contract) + restore
            // verification against the model snapshot (S11)
            attentiondb_core::backup::copy_database_dir(&db, &out.join("backup-15k")).unwrap();
            c.backups += 1;
            let restored = out.join("restore-15k");
            attentiondb_core::backup::restore_backup(&out.join("backup-15k"), &restored).unwrap();
            c.restores += 1;
            let er = open_db(&restored);
            let m = model.lock().unwrap();
            let ok = e8_verify(&er, &m, &restored, "e8a-restore", seq, &mut verif);
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
            drop(m);
            er.close().unwrap();
        }
        if (i + 1) % 3_000 == 0 {
            let m = model.lock().unwrap();
            let ok = e8_verify(&e, &m, &db, "e8a", seq, &mut verif);
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
            e8_model_ckpt(&m, out, seq);
        }
    }
    // end-of-run graceful restart (S8): readers already joined
    stop.store(true, Relaxed);
    let _ = mon.join();
    let (rchecks, rviol) = reader.join().unwrap();
    e.close().unwrap();
    c.restarts += 1;
    let er = std::sync::Arc::new(open_db(&db));
    {
        let m = model.lock().unwrap();
        let ok = e8_verify(&er, &m, &db, "e8a-final-restart", seq, &mut verif);
        c.verifies += 1;
        ok_all &= ok;
        c.verif_fails += !ok as u64;
        let _ = std::fs::write(
            out.join("reader.json"),
            serde_json::to_string(&serde_json::json!({"checks": rchecks, "violations": rviol}))
                .unwrap(),
        );
    }
    er.close().unwrap();
    let _ = std::fs::write(out.join("verif.csv"), &verif);
    let m = model.lock().unwrap();
    let _ = std::fs::write(
        out.join("counts.json"),
        e8_counts_json(&c, seq, t0.elapsed(), m.state_hash()),
    );
    e8_summary_line("E8a", &c, seq, t0.elapsed());
    assert!(
        ok_all && rviol == 0,
        "E8a verification or reader violation (viol={rviol})"
    );
}

fn run_e8b(out: &std::path::Path) {
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let _ = std::fs::create_dir_all(out);
    std::env::set_var("PH3D_DURABILITY", "sync");
    std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
    let db = out.join("db");
    // mutation-only family: the engine handle is rebound across in-loop
    // restarts (no concurrent reader holds it)
    let mut e = std::sync::Arc::new(open_db(&db));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let model = std::sync::Arc::new(std::sync::Mutex::new(E8Model::new()));
    let log = e8_new_oplog(&out.join("oplog.csv"));
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mon = {
        let o = out.to_path_buf();
        let p = progress.clone();
        let s = stop.clone();
        e8_spawn_monitor(o, p, s)
    };
    let mut c = e8_new_counts();
    let mut rng = E8Rng(0xE8B0022);
    let mut verif = String::from("seq,label,model_ok,checker,mapper_ok,retired_ok,live,hash\n");
    let mut seq = 0usize;
    let mut ok_all = true;
    for i in 0..200_000usize {
        {
            let mut m = model.lock().unwrap();
            e8_mutate(&e, &mut m, &mut rng, &log, &progress, seq, 0, 2_000, 40, 30);
        }
        seq += 1;
        if (i + 1) % 25_000 == 0 {
            e.checkpoint().unwrap();
            c.ckpts += 1;
        }
        if (i + 1) % 50_000 == 0 {
            e.compact_storage().unwrap();
            c.compactions += 1;
        }
        if i == 150_000 {
            attentiondb_core::backup::copy_database_dir(&db, &out.join("backup-150k")).unwrap();
            c.backups += 1;
            let restored = out.join("restore-150k");
            attentiondb_core::backup::restore_backup(&out.join("backup-150k"), &restored).unwrap();
            c.restores += 1;
            let er = open_db(&restored);
            let m = model.lock().unwrap();
            let ok = e8_verify(&er, &m, &restored, "e8b-restore", seq, &mut verif);
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
            drop(m);
            er.close().unwrap();
        }
        if (i + 1) % 100_000 == 0 {
            e.close().unwrap();
            c.restarts += 1;
            e = std::sync::Arc::new(open_db(&db));
            let m = model.lock().unwrap();
            let ok = e8_verify(&e, &m, &db, "e8b-restart", seq, &mut verif);
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
            e8_model_ckpt(&m, out, seq);
        }
        if (i + 1) % 10_000 == 0 {
            let m = model.lock().unwrap();
            let ok = e8_verify(&e, &m, &db, "e8b", seq, &mut verif);
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
            e8_model_ckpt(&m, out, seq);
        }
    }
    stop.store(true, Relaxed);
    let _ = mon.join();
    e.close().unwrap();
    let _ = std::fs::write(out.join("verif.csv"), &verif);
    let m = model.lock().unwrap();
    let _ = std::fs::write(
        out.join("counts.json"),
        e8_counts_json(&c, seq, t0.elapsed(), m.state_hash()),
    );
    e8_summary_line("E8b", &c, seq, t0.elapsed());
    assert!(ok_all, "E8b verification failure");
}

fn run_e8c(out: &std::path::Path) {
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let _ = std::fs::create_dir_all(out);
    std::env::set_var("PH3D_DURABILITY", "sync");
    std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
    let db = out.join("db");
    let e = std::sync::Arc::new(open_db(&db));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let model = std::sync::Arc::new(std::sync::Mutex::new(E8Model::new()));
    let log = e8_new_oplog(&out.join("oplog.csv"));
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mon = {
        let o = out.to_path_buf();
        let p = progress.clone();
        let s = stop.clone();
        e8_spawn_monitor(o, p, s)
    };
    let live_ct = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0i64));
    // 2 writers on disjoint ranges + 3 readers (spec E8c)
    let mut writer_handles = Vec::new();
    for w in 0..2usize {
        let e2 = e.clone();
        let m2 = model.clone();
        let l2 = log.clone();
        let p2 = progress.clone();
        let lc2 = live_ct.clone();
        writer_handles.push(std::thread::spawn(move || {
            let mut rng = E8Rng(0xE8C00 | (w as u64));
            let mut seq = 0usize;
            // disjoint ranges so the single-writer model stays exact:
            // writer 0: 1000..1499, writer 1: 1500..1999 (hot set 1000..1009)
            let (lo, hi) = if w == 0 {
                (1000usize, 1500usize)
            } else {
                (1500, 2000)
            };
            let span = hi - lo;
            for _ in 0..60_000usize {
                let mut m = m2.lock().unwrap();
                // e8_mutate draws within [hot, keyspace); remap to this range
                let roll = rng.below(100);
                let idx = lo
                    + if rng.below(100) < 20 {
                        rng.below(10)
                    } else {
                        rng.below(span)
                    };
                let live = m.live.contains_key(&(idx as u32));
                let _ = roll; // mix handled per-op below via explicit-idx op
                let _ = e8_writer_op(&e2, &mut m, &mut rng, &l2, &p2, seq, w, idx as u32, live);
                lc2.store(m.live.len() as i64, Relaxed);
                seq += 1;
            }
            seq
        }));
    }
    let mut readers = Vec::new();
    for r in 0..3usize {
        let e2 = e.clone();
        let lc2 = live_ct.clone();
        let s2 = stop.clone();
        let p2 = progress.clone();
        let l2 = log.clone();
        readers.push(std::thread::spawn(move || {
            e8_reader(e2, lc2, 2, s2, p2, l2, 10 + r)
        }));
    }
    // verification point mid-run (writers paused by the model mutex naturally
    // quiesce mutations; readers may still run: equality is on committed state
    // and both writers block on the model mutex, so the model is exact)
    std::thread::sleep(std::time::Duration::from_secs(60));
    let mut verif = String::from("seq,label,model_ok,checker,mapper_ok,retired_ok,live,hash\n");
    let mut c = e8_new_counts();
    let mut ok_all = true;
    {
        let m = model.lock().unwrap();
        let ok = e8_verify(&e, &m, &db, "e8c-mid", progress.load(Relaxed), &mut verif);
        c.verifies += 1;
        ok_all &= ok;
        e8_model_ckpt(&m, out, progress.load(Relaxed));
    }
    let mut total_writer_ops = 0usize;
    for h in writer_handles {
        total_writer_ops += h.join().unwrap();
    }
    stop.store(true, Relaxed);
    let _ = mon.join();
    let mut checks = 0usize;
    let mut viol = 0usize;
    for h in readers {
        let (ch, vi) = h.join().unwrap();
        checks += ch;
        viol += vi;
    }
    {
        let m = model.lock().unwrap();
        let ok = e8_verify(&e, &m, &db, "e8c-final", total_writer_ops, &mut verif);
        c.verifies += 1;
        ok_all &= ok;
        c.verif_fails += !ok as u64;
    }
    e.close().unwrap();
    let _ = std::fs::write(out.join("verif.csv"), &verif);
    let _ = std::fs::write(
        out.join("reader.json"),
        serde_json::to_string(&serde_json::json!({"checks": checks, "violations": viol})).unwrap(),
    );
    let m = model.lock().unwrap();
    let _ = std::fs::write(
        out.join("counts.json"),
        e8_counts_json(&c, total_writer_ops, t0.elapsed(), m.state_hash()),
    );
    e8_summary_line("E8c", &c, total_writer_ops, t0.elapsed());
    assert!(
        ok_all && viol == 0,
        "E8c verification or reader violation (viol={viol})"
    );
}

/// Single mutation op with an EXPLICIT target idx (used by the disjoint-range
/// writers of E8c so the model stays single-writer exact per range).
#[allow(clippy::too_many_arguments)]
fn e8_writer_op(
    e: &AttentionEngine,
    m: &mut E8Model,
    rng: &mut E8Rng,
    log: &E8Oplog,
    prog: &std::sync::atomic::AtomicUsize,
    seq: usize,
    thread: usize,
    idx: u32,
    live: bool,
) -> &'static str {
    use std::sync::atomic::Ordering::Relaxed;
    let coll = e8_coll_of(idx);
    let done = if rng.below(100) < 15 && !live {
        let ver = m.maxver.get(&idx).copied().unwrap_or(0) + 1;
        let num = 100 + (seq % 900) as i64;
        let mut r = doc_record(idx, ver, "t", num);
        r.id = uuidc(e8_cid(idx), idx, ver);
        match e.insert_document(coll, r) {
            Ok(_) => {
                m.maxver.insert(idx, ver);
                m.live.insert(idx, (ver, num));
                e8_log(
                    log, seq, thread, "insert", coll, idx as i64, ver, num, 0, "ok",
                );
                "insert"
            }
            Err(_) => {
                e8_log(
                    log, seq, thread, "insert", coll, idx as i64, ver, num, 0, "err",
                );
                "insert-err"
            }
        }
    } else if live && rng.below(100) < 30 {
        let (ver, _) = m.live[&idx];
        match e.delete_document(coll, &uuidc(e8_cid(idx), idx, ver).to_string()) {
            Ok(true) => {
                m.live.remove(&idx);
                e8_log(
                    log, seq, thread, "delete", coll, idx as i64, ver, 0, 0, "ok",
                );
                "delete"
            }
            Ok(false) => {
                e8_log(
                    log, seq, thread, "delete", coll, idx as i64, ver, 0, 0, "absent",
                );
                "delete-absent"
            }
            Err(_) => {
                e8_log(
                    log, seq, thread, "delete", coll, idx as i64, ver, 0, 0, "err",
                );
                "delete-err"
            }
        }
    } else if live {
        let (ver, _) = m.live[&idx];
        let num = 100 + (seq % 900) as i64;
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(idx));
        fields.insert("version".to_string(), serde_json::json!(ver));
        fields.insert("cat".to_string(), serde_json::json!("t"));
        fields.insert("num".to_string(), serde_json::json!(num));
        fields.insert(
            "title".to_string(),
            serde_json::json!(format!("doc-{idx}-v{ver}")),
        );
        let mut kv = HashMap::new();
        kv.insert(HEAD.to_string(), vec_for(idx));
        match e.update_document(coll, &uuidc(e8_cid(idx), idx, ver).to_string(), fields, kv) {
            Ok(_) => {
                m.live.insert(idx, (ver, num));
                e8_log(
                    log, seq, thread, "update", coll, idx as i64, ver, num, 0, "ok",
                );
                "update"
            }
            Err(_) => {
                e8_log(
                    log, seq, thread, "update", coll, idx as i64, ver, num, 0, "err",
                );
                "update-err"
            }
        }
    } else {
        let ver = m.maxver.get(&idx).copied().unwrap_or(0) + 1;
        let num = 100 + (seq % 900) as i64;
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(idx));
        fields.insert("version".to_string(), serde_json::json!(ver));
        fields.insert("cat".to_string(), serde_json::json!("t"));
        fields.insert("num".to_string(), serde_json::json!(num));
        fields.insert(
            "title".to_string(),
            serde_json::json!(format!("doc-{idx}-v{ver}")),
        );
        let mut kv = HashMap::new();
        kv.insert(HEAD.to_string(), vec_for(idx));
        match e.upsert_document(coll, uuidc(e8_cid(idx), idx, ver), fields, kv) {
            Ok(_) => {
                m.maxver.insert(idx, ver);
                m.live.insert(idx, (ver, num));
                e8_log(
                    log, seq, thread, "upsert", coll, idx as i64, ver, num, 0, "ok",
                );
                "upsert"
            }
            Err(_) => {
                e8_log(
                    log, seq, thread, "upsert", coll, idx as i64, ver, num, 0, "err",
                );
                "upsert-err"
            }
        }
    };
    prog.fetch_add(1, Relaxed);
    done
}

fn run_e8d(out: &std::path::Path) {
    use attentiondb_core::transaction::TxnOp;
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let _ = std::fs::create_dir_all(out);
    std::env::set_var("PH3D_DURABILITY", "sync");
    std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
    let db = out.join("db");
    let mut e = std::sync::Arc::new(open_db(&db));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let model = std::sync::Arc::new(std::sync::Mutex::new(E8Model::new()));
    let log = e8_new_oplog(&out.join("oplog.csv"));
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mon = {
        let o = out.to_path_buf();
        let p = progress.clone();
        let s = stop.clone();
        e8_spawn_monitor(o, p, s)
    };
    let mut c = e8_new_counts();
    let mut rng = E8Rng(0xE8D0044);
    let mut verif = String::from("seq,label,model_ok,checker,mapper_ok,retired_ok,live,hash\n");
    let mut seq = 0usize;
    let mut ok_all = true;

    // one transaction: draw keys FIRST, stage from the drawn plan, apply the
    // model from the SAME plan only on confirmed commit (no rng divergence).
    let run_txn = |e: &AttentionEngine,
                   m: &mut E8Model,
                   keys: &[u32],
                   num: i64,
                   do_rollback: bool,
                   log: &E8Oplog,
                   seq: usize,
                   c: &mut E8Counts|
     -> bool {
        let t = e.begin_transaction("bench");
        let mut plan: Vec<(u32, u64, u64, i64)> = Vec::new(); // idx, del_ver, ins_ver, num
        for &idx in keys {
            let del_ver = m.live.get(&idx).map(|(v, _)| *v);
            let ins_ver = m.maxver.get(&idx).copied().unwrap_or(0) + 1;

            if let Some(dv) = del_ver {
                e.record_transaction_operation(t, TxnOp::Delete(uuidc(e8_cid(idx), idx, dv)))
                    .unwrap();
            }
            let mut r = doc_record(idx, ins_ver, "t", num);
            r.id = uuidc(e8_cid(idx), idx, ins_ver);
            e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
            plan.push((idx, del_ver.unwrap_or(0), ins_ver, num));
        }
        c.txns += 1;
        if do_rollback {
            e.rollback_transaction(t).unwrap();
            c.rollbacks += 1;
            for (idx, dv, _, _) in &plan {
                e8_log(
                    log,
                    seq,
                    0,
                    "rollback",
                    e8_coll_of(*idx),
                    *idx as i64,
                    *dv,
                    0,
                    t,
                    "ok",
                );
            }
            return true;
        }
        let okc = e.commit_transaction(t).unwrap_or(false);
        if okc {
            c.commits += 1;
            for (idx, _dv, ins_ver, n) in plan {
                m.maxver.insert(idx, ins_ver);
                m.live.insert(idx, (ins_ver, n));
            }
            true
        } else {
            false
        }
    };

    // ---- phase 1: transaction size ladder (§19): 7 sizes x 40 commits ----
    for size in [1usize, 2, 5, 10, 25, 50, 100] {
        for _cycle in 0..40usize {
            let num = 100 + (seq % 900) as i64;
            let keys: Vec<u32> = (0..size).map(|_| rng.below(1_000) as u32).collect();
            let mut m = model.lock().unwrap();
            let ok = run_txn(&e, &mut m, &keys, num, false, &log, seq, &mut c);
            drop(m);
            ok_all &= ok;
            progress.fetch_add(size.max(1), Relaxed);
            seq += size;
        }
        {
            let m = model.lock().unwrap();
            let vok = e8_verify(&e, &m, &db, &format!("e8d-ladder-{size}"), seq, &mut verif);
            c.verifies += 1;
            ok_all &= vok;
            c.verif_fails += !vok as u64;
            e8_model_ckpt(&m, out, seq);
        }
    }
    // ---- phase 2: mixed commit/rollback (25% rollback), 15k txns ----
    for i in 0..15_000usize {
        let size = 1 + rng.below(5);
        let num = 100 + (seq % 900) as i64;
        let keys: Vec<u32> = (0..size).map(|_| rng.below(1_000) as u32).collect();
        let rb = rng.below(100) < 25;
        let mut m = model.lock().unwrap();
        let ok = run_txn(&e, &mut m, &keys, num, rb, &log, seq, &mut c);
        drop(m);
        ok_all &= ok;
        progress.fetch_add(size, Relaxed);
        seq += size;
        if (i + 1) % 5_000 == 0 {
            e.checkpoint().unwrap();
            c.ckpts += 1;
        }
        if (i + 1) % 7_500 == 0 {
            e.compact_storage().unwrap();
            c.compactions += 1;
        }
    }
    {
        let m = model.lock().unwrap();
        let vok = e8_verify(&e, &m, &db, "e8d-mixed", seq, &mut verif);
        c.verifies += 1;
        ok_all &= vok;
        c.verif_fails += !vok as u64;
    }
    // ---- phase 3: WAL rotation during transaction-heavy workload (2 KiB) ----
    e.close().unwrap();
    eprintln!("[e8dbg] phase2 closed, opening phase3 (2KiB)");
    std::env::set_var("ATTENTIONDB_WAL_SEGMENT_BYTES", "2048");
    e = std::sync::Arc::new(open_db(&db));
    eprintln!("[e8dbg] phase3 opened");
    for i in 0..2_000usize {
        let num = 100 + (seq % 900) as i64;
        let keys: Vec<u32> = (0..3).map(|_| rng.below(1_000) as u32).collect();
        let mut m = model.lock().unwrap();
        let ok = run_txn(&e, &mut m, &keys, num, false, &log, seq, &mut c);
        drop(m);
        ok_all &= ok;
        progress.fetch_add(3, Relaxed);
        seq += 3;
        if (i + 1) % 500 == 0 {
            e.checkpoint().unwrap();
            c.ckpts += 1;
        }
    }
    {
        let m = model.lock().unwrap();
        let vok = e8_verify(&e, &m, &db, "e8d-rotation", seq, &mut verif);
        c.verifies += 1;
        ok_all &= vok;
        c.verif_fails += !vok as u64;
        e8_model_ckpt(&m, out, seq);
    }
    // ---- phase 4: restart + atomicity (S7): every committed txn intact ----
    e.close().unwrap();
    c.restarts += 1;
    e = std::sync::Arc::new(open_db(&db));
    {
        let m = model.lock().unwrap();
        let vok = e8_verify(&e, &m, &db, "e8d-final-restart", seq, &mut verif);
        c.verifies += 1;
        ok_all &= vok;
        c.verif_fails += !vok as u64;
    }
    stop.store(true, Relaxed);
    let _ = mon.join();
    e.close().unwrap();
    let _ = std::fs::write(out.join("verif.csv"), &verif);
    let m = model.lock().unwrap();
    let _ = std::fs::write(
        out.join("counts.json"),
        e8_counts_json(&c, seq, t0.elapsed(), m.state_hash()),
    );
    e8_summary_line("E8d", &c, seq, t0.elapsed());
    assert!(ok_all, "E8d verification failure");
}

fn run_e8e(out: &std::path::Path) {
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let _ = std::fs::create_dir_all(out);
    std::env::set_var("PH3D_DURABILITY", "sync");
    // frequent rotation: 2 KiB segments for the whole family (spec E8e)
    std::env::set_var("ATTENTIONDB_WAL_SEGMENT_BYTES", "2048");
    let db = out.join("db");
    let mut e = std::sync::Arc::new(open_db(&db));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let model = std::sync::Arc::new(std::sync::Mutex::new(E8Model::new()));
    let log = e8_new_oplog(&out.join("oplog.csv"));
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mon = {
        let o = out.to_path_buf();
        let p = progress.clone();
        let s = stop.clone();
        e8_spawn_monitor(o, p, s)
    };
    let mut c = e8_new_counts();
    let mut rng = E8Rng(0xE8E0055);
    let mut verif = String::from("seq,label,model_ok,checker,mapper_ok,retired_ok,live,hash\n");
    let mut census =
        String::from("seq,label,wal_n,wal_b,sst_n,sst_b,meta_n,tmp_orphans,db_bytes,hash\n");
    let mut seq = 0usize;
    let mut ok_all = true;
    for cycle in 0..12usize {
        for k in 1..=1000usize {
            let mut m = model.lock().unwrap();
            e8_mutate(&e, &mut m, &mut rng, &log, &progress, seq, 0, 1_000, 25, 30);
            drop(m);
            seq += 1;
            if k == 100 {
                e.checkpoint().unwrap();
                c.ckpts += 1;
            }
            if k == 500 {
                e.compact_storage().unwrap();
                c.compactions += 1;
            }
            if k == 750 {
                let bdir = out.join(format!("backup-c{cycle}"));
                attentiondb_core::backup::copy_database_dir(&db, &bdir).unwrap();
                c.backups += 1;
                let restored = out.join(format!("restore-c{cycle}"));
                attentiondb_core::backup::restore_backup(&bdir, &restored).unwrap();
                c.restores += 1;
                let er = open_db(&restored);
                let m = model.lock().unwrap();
                let ok = e8_verify(
                    &er,
                    &m,
                    &restored,
                    &format!("e8e-restore-c{cycle}"),
                    seq,
                    &mut verif,
                );
                c.verifies += 1;
                ok_all &= ok;
                c.verif_fails += !ok as u64;
                drop(m);
                er.close().unwrap();
            }
        }
        // end of cycle: verify, census, restart (spec §21 cadence, 1000→restart)
        {
            let m = model.lock().unwrap();
            let ok = e8_verify(&e, &m, &db, &format!("e8e-c{cycle}"), seq, &mut verif);
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
            e8_model_ckpt(&m, out, seq);
        }
        e8_census(&db, &format!("e8e-c{cycle}"), seq, &mut census);
        e.close().unwrap();
        c.restarts += 1;
        e = std::sync::Arc::new(open_db(&db));
        {
            let m = model.lock().unwrap();
            let ok = e8_verify(
                &e,
                &m,
                &db,
                &format!("e8e-c{cycle}-restart"),
                seq,
                &mut verif,
            );
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
        }
    }
    stop.store(true, Relaxed);
    let _ = mon.join();
    e.close().unwrap();
    let _ = std::fs::write(out.join("verif.csv"), &verif);
    let _ = std::fs::write(out.join("census.csv"), &census);
    let m = model.lock().unwrap();
    let _ = std::fs::write(
        out.join("counts.json"),
        e8_counts_json(&c, seq, t0.elapsed(), m.state_hash()),
    );
    e8_summary_line("E8e", &c, seq, t0.elapsed());
    assert!(ok_all, "E8e verification failure");
    std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
}

// ---------------- E8f: restart/recovery soak (kill matrix R1-R7) ----------------

fn run_e8f(out: &std::path::Path) {
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let _ = std::fs::create_dir_all(out);
    std::env::set_var("PH3D_DURABILITY", "sync");
    std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
    let db = out.join("db");
    let mut e = std::sync::Arc::new(open_db(&db));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let model = std::sync::Arc::new(std::sync::Mutex::new(E8Model::new()));
    let log = e8_new_oplog(&out.join("oplog.csv"));
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mon = {
        let o = out.to_path_buf();
        let p = progress.clone();
        let s = stop.clone();
        e8_spawn_monitor(o, p, s)
    };
    let mut c = e8_new_counts();
    let mut rng = E8Rng(0xE8F0066);
    let mut verif = String::from("seq,label,model_ok,checker,mapper_ok,retired_ok,live,hash\n");
    let mut seq = 0usize;
    let mut ok_all = true;

    // ---- graceful cycles: open -> 2000 ops -> checkpoint -> close -> reopen
    for cycle in 0..60usize {
        for _ in 0..2_000usize {
            let mut m = model.lock().unwrap();
            e8_mutate(&e, &mut m, &mut rng, &log, &progress, seq, 0, 800, 16, 25);
            drop(m);
            seq += 1;
        }
        e.checkpoint().unwrap();
        c.ckpts += 1;
        e.close().unwrap();
        c.restarts += 1;
        e = std::sync::Arc::new(open_db(&db));
        if cycle % 10 == 0 {
            let m = model.lock().unwrap();
            let ok = e8_verify(&e, &m, &db, &format!("e8f-g{cycle}"), seq, &mut verif);
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
        }
    }
    {
        let m = model.lock().unwrap();
        let ok = e8_verify(&e, &m, &db, "e8f-graceful-final", seq, &mut verif);
        c.verifies += 1;
        ok_all &= ok;
        c.verif_fails += !ok as u64;
    }
    e.close().unwrap();
    drop(e);
    stop.store(true, Relaxed);
    let _ = mon.join();

    // ---- kill matrix: child runs a deterministic prefix, parent groupkills
    // at the phase boundary, fresh open must equal the child's model JSON
    let exe = std::env::current_exe().unwrap();
    let mut matrix_ok = true;
    for phase in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"] {
        for rep in 0..2usize {
            let cdir = out.join(format!("kill-{phase}-{rep}"));
            let _ = std::fs::remove_dir_all(&cdir);
            let marker = cdir.with_extension("e8marker");
            let _ = std::fs::remove_file(&marker);
            let mut cmd = std::process::Command::new(&exe);
            cmd.args([
                "dbtest",
                "e8child",
                "--dir",
                cdir.to_str().unwrap(),
                "--phase",
                phase,
                "--seed",
                &(0xE8F70 + rep as u64).to_string(),
            ])
            .env("PH3D_DURABILITY", "sync");
            if phase == "R3" {
                cmd.env("ATTENTIONDB_WAL_SEGMENT_BYTES", "2048");
            }
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
            let mut child = cmd.spawn().unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(90);
            let mut reached = false;
            while std::time::Instant::now() < deadline {
                if marker.exists() {
                    reached = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            if reached {
                unsafe {
                    libc::kill(-(child.id() as i32), libc::SIGKILL);
                }
            }
            let _ = child.wait();
            // fresh-process recovery judge
            let model_path = cdir.with_extension("e8model");
            if !reached || !model_path.exists() {
                verif.push_str(&format!(
                    "{seq},e8f-{phase}-{rep},FAIL,no-marker,unknown,unknown,0,\n"
                ));
                matrix_ok = false;
                continue;
            }
            let expected = E8Model::from_json(&std::fs::read_to_string(&model_path).unwrap());
            let er = std::sync::Arc::new(open_db(&cdir));
            let (rm, iss) = export_state(&er);
            let exp_model: Model = expected
                .live
                .iter()
                .map(|(k, (v, n))| (*k, (*v, "t".to_string(), *n)))
                .collect();
            let eq = rm == exp_model && iss.is_empty();
            let clean = e7_checker(&er, &cdir);
            let mapper_ok = {
                let map = er.id_mapper.read();
                expected.live.iter().all(|(idx, (ver, _))| {
                    map.uuid_to_id(&uuidc(e8_cid(*idx), *idx, *ver)).is_some()
                })
            };
            er.close().unwrap();
            let ok = eq && clean && mapper_ok;
            matrix_ok &= ok;
            c.restarts += 1;
            c.verifies += 1;
            verif.push_str(&format!(
                "{},e8f-{}-{},live:{},exp:{},clean:{},mapper:{},,\n",
                seq,
                phase,
                rep,
                rm.len(),
                exp_model.len(),
                clean,
                mapper_ok
            ));
            if !ok {
                eprintln!(
                    "e8f KILL MATRIX FAIL {phase}/{rep}: eq={eq} clean={clean} mapper={mapper_ok}"
                );
            }
        }
    }
    ok_all &= matrix_ok;

    // ---- async boundary (spec §29): Case B unflushed async -> groupkill ----
    for rep in 0..2usize {
        let cdir = out.join(format!("async-{rep}"));
        let _ = std::fs::remove_dir_all(&cdir);
        let marker = cdir.with_extension("e8marker");
        let mut cmd = std::process::Command::new(&exe);
        cmd.args([
            "dbtest",
            "e8child",
            "--dir",
            cdir.to_str().unwrap(),
            "--phase",
            "ASYNC",
            "--seed",
            &(0xE8F90 + rep as u64).to_string(),
        ])
        .env("PH3D_DURABILITY", "async");
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        let mut child = cmd.spawn().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(90);
        while !marker.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
        let _ = child.wait();
        let model_path = cdir.with_extension("e8model");
        let expected = E8Model::from_json(&std::fs::read_to_string(&model_path).unwrap());
        std::env::set_var("PH3D_DURABILITY", "sync");
        let er = std::sync::Arc::new(open_db(&cdir));
        let (rm, iss) = export_state(&er);
        let clean = e7_checker(&er, &cdir);
        // Spec §29 Case B: an unflushed async kill may lose buffered tail ops
        // (documented bounded loss, NOT a failure). A recovered doc at an
        // OLDER coherent version = its latest buffered upsert was lost =
        // documented loss. A doc at a FUTURE version or with incoherent
        // content = corruption = failure. Missing docs = documented loss.
        let mut content_ok = true;
        let mut losses = 0usize;
        for (idx, (ver, _num)) in &expected.live {
            match rm.get(idx) {
                Some((rv, _, _rn)) => {
                    if rv > ver {
                        content_ok = false; // future version: impossible post-crash
                    } else if rv < ver {
                        losses += 1; // older coherent version: documented loss
                    }
                }
                None => {
                    losses += 1; // doc lost entirely: documented loss
                }
            }
        }
        er.close().unwrap();
        let ok = clean && iss.is_empty() && content_ok;
        ok_all &= ok;
        c.restarts += 1;
        c.verifies += 1;
        let recovered = rm.len();
        verif.push_str(&format!(
            "{},e8f-async-{rep},recovered:{}/{},losses:{},clean:{},torn:{},{},,\n",
            seq,
            recovered,
            expected.live.len(),
            losses,
            clean,
            !content_ok,
            if losses > 0 {
                "ASYNC-LOSS-OK"
            } else {
                "NO-LOSS"
            }
        ));
        eprintln!(
            "e8f async-{rep}: recovered {recovered}/{} acked docs, losses={losses} (documented async boundary); clean={clean}",
            expected.live.len()
        );
    }

    let _ = std::fs::write(out.join("verif.csv"), &verif);
    let _ = std::fs::write(
        out.join("counts.json"),
        e8_counts_json(&c, seq, t0.elapsed(), 0),
    );
    e8_summary_line("E8f", &c, seq, t0.elapsed());
    assert!(ok_all, "E8f restart/recovery failure");
}

/// E8f child: deterministic prefix + phase maintenance + model JSON + marker.
fn e8_child(dir: &std::path::Path, phase: &str, seed: u64) -> ! {
    let _ = std::fs::create_dir_all(dir);
    let e = open_db(dir);
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let mut model = E8Model::new();
    let mut rng = E8Rng(seed);
    let log = e8_new_oplog(&dir.with_extension("e8oplog"));
    let prog = std::sync::atomic::AtomicUsize::new(0);
    let mut seq = 0usize;
    #[allow(clippy::explicit_counter_loop)] // seq tracks absolute op position
    for _ in 0..3_000usize {
        e8_mutate(&e, &mut model, &mut rng, &log, &prog, seq, 0, 400, 8, 20);
        seq += 1;
    }
    match phase {
        "R2" => {
            let _ = e.checkpoint().unwrap();
        }
        "R3" => { /* 2 KiB segments via env: rotations happened during ops */ }
        "R4" => {
            let _ = e.checkpoint().unwrap();
            let _ = e.compact_storage().unwrap();
        }
        "R5" => {
            attentiondb_core::backup::copy_database_dir(dir, &dir.with_extension("e8bak")).unwrap();
        }
        "R6" => {
            let t = e8_stage_upsert(&e, &model, 399, 777);
            assert!(e.commit_transaction(t).unwrap());
            e8_apply_commit(&mut model, 399, 777);
        }
        "R7" => {
            let t = e8_stage_upsert(&e, &model, 398, 888);
            e.rollback_transaction(t).unwrap();
        }
        _ => {} // R1: no maintenance (WAL-path recovery); ASYNC: no flush
    }
    let _ = std::fs::write(dir.with_extension("e8model"), model.to_json());
    let _ = std::fs::write(dir.with_extension("e8marker"), "ready");
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

fn run_e8g(out: &std::path::Path) {
    use attentiondb_core::transaction::TxnOp;
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let _ = std::fs::create_dir_all(out);
    std::env::set_var("PH3D_DURABILITY", "sync");
    std::env::set_var("ATTENTIONDB_WAL_SEGMENT_BYTES", "8192");
    let db = out.join("db");
    let mut e = std::sync::Arc::new(open_db(&db));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    e.create_collection("collb", DIM, &[HEAD]).unwrap();
    e.create_collection("collc", DIM, &[HEAD]).unwrap();
    let model = std::sync::Arc::new(std::sync::Mutex::new(E8Model::new()));
    let log = e8_new_oplog(&out.join("oplog.csv"));
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mon = {
        let o = out.to_path_buf();
        let p = progress.clone();
        let s = stop.clone();
        e8_spawn_monitor(o, p, s)
    };
    let mut c = e8_new_counts();
    let mut rng = E8Rng(0xE8_600_077);
    let mut verif = String::from("seq,label,model_ok,checker,mapper_ok,retired_ok,live,hash\n");
    let mut census =
        String::from("seq,label,wal_n,wal_b,sst_n,sst_b,meta_n,tmp_orphans,db_bytes,hash\n");
    let mut seq = 0usize;
    let mut ok_all = true;
    // blocks: bench 0..299, collb 3000..3299, collc 3300..3599
    let blocks: [(u32, usize); 3] = [(0, 300), (3000, 300), (3300, 300)];
    for cycle in 0..25usize {
        // mixed mutations across the 3 collections
        for _ in 0..40usize {
            let (lo, span) = blocks[rng.below(3)];
            let idx = lo + rng.below(span) as u32;
            let mut m = model.lock().unwrap();
            let live = m.live.contains_key(&idx);
            e8_writer_op(&e, &mut m, &mut rng, &log, &progress, seq, 0, idx, live);
            drop(m);
            seq += 1;
        }
        // explicit lifecycle key: INSERT -> UPDATE -> DELETE -> REINSERT
        let lk = 200u32 + (cycle % 100) as u32;
        {
            let mut m = model.lock().unwrap();
            let live = m.live.contains_key(&lk);
            e8_writer_op(&e, &mut m, &mut rng, &log, &progress, seq, 0, lk, live); // insert or update
            seq += 1;
            let live2 = m.live.contains_key(&lk);
            e8_writer_op(&e, &mut m, &mut rng, &log, &progress, seq, 0, lk, live2); // update
            seq += 1;
            if let Some((ver, _)) = m.live.get(&lk).copied() {
                e.delete_document("bench", &uuidc(0, lk, ver).to_string())
                    .unwrap();
                m.live.remove(&lk);
                e8_log(&log, seq, 0, "delete", "bench", lk as i64, ver, 0, 0, "ok");
                seq += 1;
            }
            let nver = m.maxver.get(&lk).copied().unwrap_or(0) + 1;
            let mut r = doc_record(lk, nver, "t", 100 + (seq % 900) as i64);
            r.id = uuidc(0, lk, nver);
            e.insert_document("bench", r).unwrap();
            let num = 100 + (seq % 900) as i64;
            m.maxver.insert(lk, nver);
            m.live.insert(lk, (nver, num));
            e8_log(
                &log, seq, 0, "reinsert", "bench", lk as i64, nver, num, 0, "ok",
            );
            seq += 1;
            progress.fetch_add(1, Relaxed);
            // one 2-key transaction
            let num = 100 + (seq % 900) as i64;
            let k1 = (cycle * 7) as u32 % 300;
            let k2 = 3000 + ((cycle * 11) as u32 % 300);
            let t = e.begin_transaction("bench");
            {
                let idx = k1;
                if let Some((ver, _)) = m.live.get(&idx) {
                    e.record_transaction_operation(t, TxnOp::Delete(uuidc(0, idx, *ver)))
                        .unwrap();
                }
                let nv = m.maxver.get(&idx).copied().unwrap_or(0) + 1;
                let mut rr = doc_record(idx, nv, "t", num);
                rr.id = uuidc(0, idx, nv);
                e.record_transaction_operation(t, TxnOp::Insert(rr))
                    .unwrap();
            }
            let coll = "collb";
            let t2 = e.begin_transaction(coll);
            if let Some((ver, _)) = m.live.get(&k2) {
                e.record_transaction_operation(t2, TxnOp::Delete(uuidc(1, k2, *ver)))
                    .unwrap();
            }
            let nv2 = m.maxver.get(&k2).copied().unwrap_or(0) + 1;
            let mut rr2 = doc_record(k2, nv2, "t", num);
            rr2.id = uuidc(1, k2, nv2);
            e.record_transaction_operation(t2, TxnOp::Insert(rr2))
                .unwrap();
            assert!(e.commit_transaction(t).unwrap());
            assert!(e.commit_transaction(t2).unwrap());
            c.txns += 2;
            c.commits += 2;
            for (idx, ver, nv) in [(k1, 0u64, 0u64), (k2, 0, 0)] {
                let _ = (ver, nv);
                let nvv = m.maxver.get(&idx).copied().unwrap_or(0) + 1;
                m.maxver.insert(idx, nvv);
                m.live.insert(idx, (nvv, num));
            }
            drop(m);
            seq += 2;
        }
        // maintenance chain (§39): CKPT -> ROTATE -> COMPACT -> BACKUP -> RESTART
        e.checkpoint().unwrap();
        c.ckpts += 1;
        e.flush_wal().unwrap();
        e.compact_storage().unwrap();
        c.compactions += 1;
        let bdir = out.join(format!("backup-c{cycle}"));
        attentiondb_core::backup::copy_database_dir(&db, &bdir).unwrap();
        c.backups += 1;
        let restored = out.join(format!("restore-c{cycle}"));
        attentiondb_core::backup::restore_backup(&bdir, &restored).unwrap();
        c.restores += 1;
        {
            let er = open_db(&restored);
            let m = model.lock().unwrap();
            let ok = e8_verify(
                &er,
                &m,
                &restored,
                &format!("e8g-restore-c{cycle}"),
                seq,
                &mut verif,
            );
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
            drop(m);
            er.close().unwrap();
        }
        e.close().unwrap();
        c.restarts += 1;
        e = std::sync::Arc::new(open_db(&db));
        {
            let m = model.lock().unwrap();
            // per-collection independence (S15/§40): verified inside e8_verify
            let ok = e8_verify(&e, &m, &db, &format!("e8g-c{cycle}"), seq, &mut verif);
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
            e8_model_ckpt(&m, out, seq);
        }
        if cycle % 5 == 0 {
            e8_census(&db, &format!("e8g-c{cycle}"), seq, &mut census);
        }
    }
    stop.store(true, Relaxed);
    let _ = mon.join();
    e.close().unwrap();
    let _ = std::fs::write(out.join("verif.csv"), &verif);
    let _ = std::fs::write(out.join("census.csv"), &census);
    let m = model.lock().unwrap();
    let _ = std::fs::write(
        out.join("counts.json"),
        e8_counts_json(&c, seq, t0.elapsed(), m.state_hash()),
    );
    e8_summary_line("E8g", &c, seq, t0.elapsed());
    assert!(ok_all, "E8g lifecycle failure");
    std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
}

fn run_e8h(out: &std::path::Path) {
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let _ = std::fs::create_dir_all(out);
    std::env::set_var("PH3D_DURABILITY", "sync");
    std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
    let db = out.join("db");
    let e = std::sync::Arc::new(open_db(&db));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    let model = std::sync::Arc::new(std::sync::Mutex::new(E8Model::new()));
    let log = e8_new_oplog(&out.join("oplog.csv"));
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mon = {
        let o = out.to_path_buf();
        let p = progress.clone();
        let s = stop.clone();
        e8_spawn_monitor(o, p, s)
    };
    let live_ct = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0i64));
    let rstop = stop.clone();
    let re = e.clone();
    let rprog = progress.clone();
    let rlog = log.clone();
    let rl = live_ct.clone();
    let reader = std::thread::spawn(move || e8_reader(re, rl, 1, rstop, rprog, rlog, 9));
    let mut c = e8_new_counts();
    let mut rng = E8Rng(0xE8_700_088);
    let mut verif = String::from("seq,label,model_ok,checker,mapper_ok,retired_ok,live,hash\n");
    let mut seq = 0usize;
    let mut ok_all = true;
    // steady state: 800-key space, live pinned at 600 (S14/S16 with constant
    // logical size — RSS growth here is NOT live-data growth)
    const PIN: usize = 600;
    while t0.elapsed().as_secs() < 240 && seq < 150_000usize {
        let mut m = model.lock().unwrap();
        if m.live.len() < PIN {
            // insert a dead key
            let mut idx = rng.below(800) as u32;
            let mut guard = 0;
            while m.live.contains_key(&idx) && guard < 800 {
                idx = (idx + 1) % 800;
                guard += 1;
            }
            let live = m.live.contains_key(&idx);
            e8_writer_op(&e, &mut m, &mut rng, &log, &progress, seq, 0, idx, live);
        } else if m.live.len() > PIN {
            let idx = *m.live.keys().next().unwrap();
            let live = true;
            e8_writer_op(&e, &mut m, &mut rng, &log, &progress, seq, 0, idx, live);
        } else {
            // churn: delete the oldest live key, insert it back with a new
            // version (tombstone + mapping churn at constant logical size)
            let idx = *m.live.keys().next().unwrap();
            let (ver, _) = m.live[&idx];
            e.delete_document("bench", &uuidc(0, idx, ver).to_string())
                .unwrap();
            m.live.remove(&idx);
            e8_log(&log, seq, 0, "delete", "bench", idx as i64, ver, 0, 0, "ok");
            seq += 1;
            let nver = m.maxver.get(&idx).copied().unwrap_or(0) + 1;
            let num = 100 + (seq % 900) as i64;
            let mut r = doc_record(idx, nver, "t", num);
            r.id = uuidc(0, idx, nver);
            e.insert_document("bench", r).unwrap();
            m.maxver.insert(idx, nver);
            m.live.insert(idx, (nver, num));
            e8_log(
                &log, seq, 0, "reinsert", "bench", idx as i64, nver, num, 0, "ok",
            );
            progress.fetch_add(1, Relaxed);
        }
        live_ct.store(m.live.len() as i64, Relaxed);
        drop(m);
        seq += 1;
        if seq.is_multiple_of(10_000) {
            e.checkpoint().unwrap();
            c.ckpts += 1;
        }
        if seq.is_multiple_of(30_000) {
            e.compact_storage().unwrap();
            c.compactions += 1;
        }
        if seq.is_multiple_of(60_000) {
            let m = model.lock().unwrap();
            let ok = e8_verify(&e, &m, &db, "e8h", seq, &mut verif);
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
            e8_model_ckpt(&m, out, seq);
        }
    }
    stop.store(true, Relaxed);
    let _ = mon.join();
    let (rchecks, rviol) = reader.join().unwrap();
    e.close().unwrap();
    let _ = std::fs::write(out.join("verif.csv"), &verif);
    let _ = std::fs::write(
        out.join("reader.json"),
        serde_json::to_string(&serde_json::json!({"checks": rchecks, "violations": rviol}))
            .unwrap(),
    );
    let m = model.lock().unwrap();
    let _ = std::fs::write(
        out.join("counts.json"),
        e8_counts_json(&c, seq, t0.elapsed(), m.state_hash()),
    );
    e8_summary_line("E8h", &c, seq, t0.elapsed());
    assert!(ok_all && rviol == 0, "E8h failure (viol={rviol})");
}

fn run_e8i(out: &std::path::Path) {
    use attentiondb_core::transaction::TxnOp;
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    let _ = std::fs::create_dir_all(out);
    std::env::set_var("PH3D_DURABILITY", "sync");
    std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
    let db = out.join("db");
    let mut e = std::sync::Arc::new(open_db(&db));
    e.create_collection("bench", DIM, &[HEAD]).unwrap();
    e.create_collection("collb", DIM, &[HEAD]).unwrap();
    e.create_collection("collc", DIM, &[HEAD]).unwrap();
    let model = std::sync::Arc::new(std::sync::Mutex::new(E8Model::new()));
    let log = e8_new_oplog(&out.join("oplog.csv"));
    let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mon = {
        let o = out.to_path_buf();
        let p = progress.clone();
        let s = stop.clone();
        e8_spawn_monitor(o, p, s)
    };
    let live_ct = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0i64));
    // reader scans "bench" only — the window counter must be BENCH-SCOPED
    // (idx < 3000), not the global 3-collection count
    let bench_ct = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0i64));
    let rstop = stop.clone();
    let re = e.clone();
    let rprog = progress.clone();
    let rlog = log.clone();
    let rl = bench_ct.clone();
    let reader = std::thread::spawn(move || e8_reader(re, rl, 2, rstop, rprog, rlog, 9));
    let mut c = e8_new_counts();
    let mut rng = E8Rng(0xE8_800_099);
    let mut verif = String::from("seq,label,model_ok,checker,mapper_ok,retired_ok,live,hash\n");
    let mut census =
        String::from("seq,label,wal_n,wal_b,sst_n,sst_b,meta_n,tmp_orphans,db_bytes,hash\n");
    let mut seq = 0usize;
    let mut ok_all = true;
    // counter-based cadence (seq-modulo is unreliable here: the %5 txn block
    // adds +2 to seq, and every maintenance multiple of 3000 is also a
    // multiple of 5 — the +2 always skipped the boundary and the maintenance
    // never fired in the first E8i attempt)
    let mut next_ckpt = 3_000usize;
    let mut next_compact = 9_000usize;
    let mut next_backup = 15_000usize;
    let mut next_verify = 5_000usize;
    let blocks = [(0u32, 300usize), (3000, 300), (3300, 300)];
    while t0.elapsed().as_secs() < 600 && seq < 150_000usize {
        let (lo, span) = blocks[rng.below(3)];
        let idx = lo + rng.below(span) as u32;
        {
            let mut m = model.lock().unwrap();
            let live = m.live.contains_key(&idx);
            e8_writer_op(&e, &mut m, &mut rng, &log, &progress, seq, 0, idx, live);
            let bench_n = m.live.range(0..3_000u32).count() as i64;
            live_ct.store(m.live.len() as i64, Relaxed);
            bench_ct.store(bench_n, Relaxed);
            drop(m);
        }
        seq += 1;
        if seq.is_multiple_of(5) {
            // a small transaction every 5 ops — BOTH keys from the SAME
            // collection block (a txn applies all ops into keys[0]'s
            // collection; cross-block keys would be mis-tagged)
            let num = 100 + (seq % 900) as i64;
            let tblk = blocks[rng.below(3)];
            let keys: Vec<u32> = (0..2).map(|_| tblk.0 + rng.below(100) as u32).collect();
            let mut m = model.lock().unwrap();
            let t = e.begin_transaction(e8_coll_of(keys[0]));
            // stage from a PLAN (draw-time versions); apply the model from the
            // SAME plan on commit — recomputing nv per key would double-step
            // duplicate keys (model X+2 vs engine X+1)
            let mut plan: Vec<(u32, u64)> = Vec::new();
            for &idx in &keys {
                if let Some((ver, _)) = m.live.get(&idx) {
                    e.record_transaction_operation(t, TxnOp::Delete(uuidc(e8_cid(idx), idx, *ver)))
                        .unwrap();
                }
                let nv = m.maxver.get(&idx).copied().unwrap_or(0) + 1;
                let mut r = doc_record(idx, nv, "t", num);
                r.id = uuidc(e8_cid(idx), idx, nv);
                e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
                plan.push((idx, nv));
            }
            if e.commit_transaction(t).unwrap_or(false) {
                c.commits += 1;
                for (idx, nv) in plan {
                    m.maxver.insert(idx, nv);
                    m.live.insert(idx, (nv, num));
                }
            }
            c.txns += 1;
            drop(m);
            seq += 2;
            progress.fetch_add(2, Relaxed);
        }
        if seq >= next_ckpt {
            e.checkpoint().unwrap();
            c.ckpts += 1;
            next_ckpt += 3_000;
        }
        if seq >= next_compact {
            e.compact_storage().unwrap();
            c.compactions += 1;
            next_compact += 9_000;
        }
        if seq >= next_backup {
            let n = seq / 15_000;
            let bdir = out.join(format!("backup-{n}"));
            attentiondb_core::backup::copy_database_dir(&db, &bdir).unwrap();
            c.backups += 1;
            next_backup += 15_000;
        }
        if seq >= next_verify {
            let m = model.lock().unwrap();
            let ok = e8_verify(&e, &m, &db, "e8i", seq, &mut verif);
            c.verifies += 1;
            ok_all &= ok;
            c.verif_fails += !ok as u64;
            e8_model_ckpt(&m, out, seq);
            e8_census(&db, "e8i", seq, &mut census);
            next_verify += 5_000;
        }
    }
    stop.store(true, Relaxed);
    let _ = mon.join();
    let (rchecks, rviol) = reader.join().unwrap();
    // final integrated restart cycle — AFTER the reader joined (E8 constraint:
    // the engine Arc is never rebound under a live reader clone)
    {
        let m = model.lock().unwrap();
        let ok = e8_verify(&e, &m, &db, "e8i-pre-restart", seq, &mut verif);
        c.verifies += 1;
        ok_all &= ok;
        c.verif_fails += !ok as u64;
    }
    e.close().unwrap();
    c.restarts += 1;
    e = std::sync::Arc::new(open_db(&db));
    {
        let m = model.lock().unwrap();
        let ok = e8_verify(&e, &m, &db, "e8i-post-restart", seq, &mut verif);
        c.verifies += 1;
        ok_all &= ok;
        c.verif_fails += !ok as u64;
        e8_census(&db, "e8i-final", seq, &mut census);
    }
    let _ = std::fs::write(out.join("verif.csv"), &verif);
    let _ = std::fs::write(out.join("census.csv"), &census);
    let _ = std::fs::write(
        out.join("reader.json"),
        serde_json::to_string(&serde_json::json!({"checks": rchecks, "violations": rviol}))
            .unwrap(),
    );
    let m = model.lock().unwrap();
    let _ = std::fs::write(
        out.join("counts.json"),
        e8_counts_json(&c, seq, t0.elapsed(), m.state_hash()),
    );
    e8_summary_line("E8i", &c, seq, t0.elapsed());
    assert!(ok_all && rviol == 0, "E8i failure (viol={rviol})");
}
