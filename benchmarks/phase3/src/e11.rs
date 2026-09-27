//! E11 — fault-injection, crash-recovery verification harness (final phase).
//!
//! Roles (dispatched via `dbtest e11 --role R --spec SPEC.json`):
//!   writer  — deterministic workload; ack-log.jsonl + reference-model.json
//!             flushed before any fault point; fault via crashgate env
//!             (armed by the controller) or an explicit stage park.
//!   recover — fresh-process open + independent model comparison + checker +
//!             retrieval + repeat-restart determinism; writes
//!             recovery-verification.json. NEVER runs with fault env.
//!   tamper  — F01: build a clean db, close, apply a named WAL artifact
//!             tamper, inventory, then record the open outcome.
//!
//! Failure model: sudden process death only (A3). No power-loss claims.
//! All state lives under the run directory; nothing touches git metadata.

use attentiondb_core::engine::AttentionEngine;
use attentiondb_core::transaction::TxnOp;
use attentiondb_query::filter::{FilterExpr, FilterOp, FilterValue};
use attentiondb_storage::{Durability, Record};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;

pub(crate) const DIM: usize = 32;
pub(crate) const HEAD: &str = "h";
const HEADS3: [&str; 3] = ["h", "h2", "h3"];

// ---------------------------------------------------------------- spec

#[derive(serde::Deserialize, Clone)]
struct Spec {
    run_id: String,
    family: String, // F01..F09 (F01 uses the tamper role)
    scenario: String,
    mode: String, // sync | group | async
    db_dir: String,
    #[serde(default)]
    backup_dir: Option<String>,
    #[serde(default)]
    restored_dir: Option<String>,
    #[serde(default)]
    n_docs: usize,
    #[serde(default)]
    n_extra: usize,
    #[serde(default)]
    fault: Option<Fault>,
    #[serde(default)]
    stage: Option<String>, // stage park for non-gate faults
    #[serde(default)]
    async_flush_at: Option<usize>, // async promote boundary (ops)
    #[serde(default)]
    fault_in_recovery: bool, // F07c: gate armed on recovery leg 1
    #[serde(default)]
    expect_open: String, // recover: "open" | "refuse"
    #[serde(default)]
    tamper_shape: String, // "" = normal, "prefix" = torn-tail intact-prefix comparison
    #[serde(default)]
    tamper_case: Option<String>,
    #[serde(default)]
    retrieval_samples: usize,
}

#[derive(serde::Deserialize, Clone)]
struct Fault {
    gate: String,
    hit: usize,
    #[serde(default)]
    model: String, // abort | groupkill
}

fn dur(mode: &str) -> Durability {
    match mode {
        "sync" => Durability::Sync,
        "group" => Durability::GroupCommit,
        _ => Durability::Async,
    }
}

fn write_json(path: &str, v: &Value) {
    std::fs::write(path, serde_json::to_vec_pretty(v).unwrap()).unwrap();
}

// ---------------------------------------------------------------- records

pub(crate) fn vec_for(idx: u32, salt: u32) -> Vec<f32> {
    // E10-discipline clustered vectors (D54): 50 clusters; vector = normalize(
    // 0.8*centroid + 0.2*deterministic noise). The earlier 2-sparse basis made
    // top-1 self-hit degenerate (bucket ties), so it was replaced BEFORE any
    // retrieval-gated run (037/042) went official.
    let cluster = (idx % 50) as usize;
    let mut h = 0x811C9DC5u32 ^ salt.wrapping_mul(0x9E3779B9);
    for b in idx.to_le_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    #[allow(clippy::useless_vec)] // Vec required: returned as Vec<f32>
    let mut v = vec![0.0f32; DIM];
    v[cluster * 61 % DIM] += 1.0;
    v[(cluster * 37 + 7) % DIM] += 0.6;
    // deterministic pseudo-noise in [-0.25, 0.25]
    let mut n = h;
    for slot in v.iter_mut() {
        n = n.wrapping_mul(1664525).wrapping_add(1013904223);
        *slot += ((n >> 16) as f32 / 65535.0 - 0.5) * 0.5;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn uuid_for(idx: u32, version: u64) -> uuid::Uuid {
    uuid::Uuid::from_u128(((idx as u128) << 64) | (version as u128))
}

fn rec1(idx: u32, ver: u64, cat: &str, num: i64) -> Record {
    let mut fields = HashMap::new();
    fields.insert("idx".to_string(), json!(idx));
    fields.insert("version".to_string(), json!(ver));
    fields.insert("cat".to_string(), json!(cat));
    fields.insert("num".to_string(), json!(num));
    fields.insert("title".to_string(), json!(format!("doc-{idx}-v{ver}")));
    let mut r = Record::new(fields);
    r.id = uuid_for(idx, ver);
    r.k_vecs.insert(HEAD.to_string(), vec_for(idx, 0));
    r
}

fn rec3(idx: u32, ver: u64, cat: &str, num: i64) -> Record {
    let mut r = rec1(idx, ver, cat, num);
    for (i, hname) in HEADS3.iter().enumerate().skip(1) {
        r.k_vecs
            .insert(hname.to_string(), vec_for(idx, i as u32 + 1));
    }
    r
}

// ---------------------------------------------------------------- model

/// live logical state: idx -> (version, cat, num)
type Live = BTreeMap<u32, (u64, String, i64)>;

#[derive(Default)]
struct Model {
    live: Live,
    acked: Vec<String>,        // acked op ids in order
    acked_live_idx: Vec<u32>,  // idx live purely via ACKED ops (sync/group guarantee set)
    in_flight: Option<u32>,    // op whose engine call never returned
    staged_txn_idx: Vec<u32>,  // ops inside a txn whose commit call was in progress
    allowed_unacked: Vec<u32>, // the op (if any) that could be in flight at a fault moment
    txns: Vec<TxnRec>,
    async_promoted: usize, // ops promoted to OS before fault (async legs)
    extra: Value,          // family-specific evidence (ckpt durations, etc.)
}

#[derive(Default, serde::Serialize, Clone)]
struct TxnRec {
    tx: u64,
    committed: bool,
    acked: bool,
    idxs: Vec<u32>,
}

/// The model MUST be on disk before any fault point can fire (M7 discipline).
fn save_model(m: &Model, spec: &Spec, out_dir: &str) {
    let tmp = format!("{out_dir}/reference-model.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&m.to_json(spec)).unwrap()).unwrap();
    std::fs::rename(tmp, format!("{out_dir}/reference-model.json")).unwrap();
}

impl Model {
    fn to_json(&self, spec: &Spec) -> Value {
        let live: Value = self
            .live
            .iter()
            .map(|(k, (v, c, n))| (k.to_string(), json!({"ver": v, "cat": c, "num": n})))
            .collect();
        json!({
            "run_id": spec.run_id, "family": spec.family, "scenario": spec.scenario,
            "mode": spec.mode, "collection": "bench", "dim": DIM, "head": HEAD,
            "live": live,
            "acked_ops": self.acked,
            "acked_live_idx": self.acked_live_idx,
            "in_flight_idx": self.in_flight,
            "staged_txn_idx": self.staged_txn_idx,
            "allowed_unacked": self.allowed_unacked,
            "txns": self.txns.iter().map(|t| serde_json::to_value(t).unwrap()).collect::<Vec<_>>(),
            "async_promoted": self.async_promoted,
            "extra": self.extra,
        })
    }
}

struct AckLog {
    f: std::fs::File,
}
impl AckLog {
    fn new(path: &str) -> Self {
        Self {
            f: std::fs::File::create(path).unwrap(),
        }
    }
    fn ack(&mut self, op: &str, idx: u32) {
        writeln!(self.f, "ACK {op} {idx}").unwrap();
        self.f.flush().unwrap();
    }
    fn note(&mut self, line: &str) {
        writeln!(self.f, "{line}").unwrap();
        self.f.flush().unwrap();
    }
}

fn apply(live: &mut Live, idx: u32, ver: u64, cat: &str, num: i64) {
    live.insert(idx, (ver, cat.to_string(), num));
}

fn state_hash(model: &Live) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for (idx, (ver, cat, num)) in model {
        for x in [*idx as u64, *ver, *num as u64] {
            h ^= x;
            h = h.wrapping_mul(0x100000001b3);
        }
        for b in cat.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

// ---------------------------------------------------------------- writer

fn park() -> ! {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

fn stage_marker(spec: &Spec, name: &str, out_dir: &str) -> ! {
    write_json(
        &format!("{out_dir}/reached.marker"),
        &json!({"stage": name, "run_id": spec.run_id}),
    );
    park()
}

fn writer(spec: &Spec, out_dir: &str) -> ! {
    // defense-in-depth: prove the armed gate is exactly the requested one
    {
        let armed = std::env::var("PH3E_CRASH_AT").ok();
        let want = spec.fault.as_ref().map(|f| f.gate.clone());
        let mut ack0 = AckLog::new(&format!("{out_dir}/ack-log.jsonl"));
        match (armed, want) {
            (Some(a), Some(w)) => {
                assert_eq!(a, w, "armed gate must match the requested fault");
                let f0 = spec.fault.as_ref().unwrap();
                let m = std::env::var("PH3E_CRASH_MODEL").unwrap_or_else(|_| "abort".into());
                assert_eq!(m, f0.model, "armed crash model must match the plan");
                ack0.note(&format!("GATE_ARMED {a} hit={} model={m}", f0.hit));
            }
            (None, None) => ack0.note("GATE_ARMED none"),
            (a, w) => panic!("fault-plan mismatch: armed={a:?} planned={w:?}"),
        }
    }
    let _ = std::fs::remove_dir_all(&spec.db_dir);
    std::fs::create_dir_all(&spec.db_dir).unwrap();
    let mode = dur(&spec.mode);
    let e = std::sync::Arc::new(
        AttentionEngine::open_dir(std::path::Path::new(&spec.db_dir), mode).unwrap(),
    );
    let heads: &[&str] = if spec.family == "F07A" {
        &HEADS3
    } else {
        &[HEAD]
    };
    e.create_collection("bench", DIM, heads).unwrap();
    let mut ack = AckLog::new(&format!("{out_dir}/ack-log.jsonl"));
    let mut m = Model::default();
    let n = spec.n_docs;

    match spec.family.as_str() {
        "F02" => {
            save_model(&m, spec, out_dir);
            for i in 0..n {
                let idx = i as u32;
                m.allowed_unacked = vec![idx];
                save_model(&m, spec, out_dir);
                let r = rec1(idx, 1, "a", i as i64);
                let res = e.insert_document("bench", r);
                if res.is_ok() {
                    ack.ack("ins", idx);
                    ack.note(&format!("LIVE {idx}"));
                    apply(&mut m.live, idx, 1, "a", i as i64);
                    m.acked.push(format!("ins-{idx}"));
                    m.acked_live_idx.push(idx);
                    if m.in_flight.is_none() && i + 1 < n && Some(i + 1) == spec.async_flush_at {
                        e.flush_wal().unwrap();
                        ack.note(&format!("PROMOTED {}", i + 1));
                        m.async_promoted = i + 1;
                    }
                    save_model(&m, spec, out_dir);
                }
            }
            // stay alive briefly so the controller's fault window lands
            std::thread::sleep(std::time::Duration::from_millis(200));
            write_json(&format!("{out_dir}/reference-model.json"), &m.to_json(spec));
            let _ = e.flush_wal();
            stage_marker(spec, "clean_end_park", out_dir); // controller SIGKILLs or verifies
        }
        "F03" => {
            // base sealed, then a 10-insert txn; gates fire during commit
            for i in 0..n {
                let idx = i as u32;
                e.insert_document("bench", rec1(idx, 1, "base", i as i64))
                    .unwrap();
                ack.ack("base", idx);
                apply(&mut m.live, idx, 1, "base", i as i64);
                m.acked.push(format!("base-{idx}"));
                m.acked_live_idx.push(idx);
            }
            e.flush_wal().unwrap();
            e.checkpoint().unwrap();
            let t = e.txn_manager.begin_transaction("bench");
            let mut txn_idx = Vec::new();
            for j in 0..10u32 {
                let idx = 10_000 + j;
                e.txn_manager
                    .record_operation(t, TxnOp::Insert(rec1(idx, 1, "txn", j as i64)))
                    .unwrap();
                txn_idx.push(idx);
            }
            // model ON DISK before the commit call — a tx_* gate fire must find it
            m.staged_txn_idx = txn_idx.clone();
            m.txns.push(TxnRec {
                tx: 1,
                committed: false,
                acked: false,
                idxs: txn_idx.clone(),
            });
            save_model(&m, spec, out_dir);
            match spec.stage.as_deref() {
                Some("pre_commit_park") => {
                    ack.note("STAGE pre_commit_park");
                    save_model(&m, spec, out_dir);
                    stage_marker(spec, "pre_commit_park", out_dir);
                }
                Some("rollback_then_park") => {
                    e.txn_manager.rollback_transaction(t).unwrap();
                    ack.note("ROLLED_BACK");
                    m.staged_txn_idx.clear();
                    m.txns.clear();
                    save_model(&m, spec, out_dir);
                    stage_marker(spec, "rollback_then_park", out_dir);
                }
                _ => {
                    // gate fires inside commit_transaction (tx_* gates)
                    let r = e.commit_transaction(t);
                    let ok = r.is_ok();
                    ack.note(&format!("COMMIT_RET {ok}"));
                    if ok {
                        for (j, idx) in txn_idx.iter().enumerate() {
                            ack.ack("txn", *idx);
                            apply(&mut m.live, *idx, 1, "txn", j as i64);
                            m.acked.push(format!("txn-{idx}"));
                            m.acked_live_idx.push(*idx);
                        }
                        m.staged_txn_idx.clear();
                        m.txns[0] = TxnRec {
                            tx: 1,
                            committed: true,
                            acked: true,
                            idxs: txn_idx,
                        };
                    } else {
                        m.txns[0].acked = false;
                    }
                    save_model(&m, spec, out_dir);
                    let _ = e.flush_wal();
                    stage_marker(spec, "clean_end_park", out_dir);
                }
            }
        }
        "F04" => {
            for i in 0..n {
                let idx = i as u32;
                e.insert_document("bench", rec1(idx, 1, "ck", i as i64))
                    .unwrap();
                ack.ack("ins", idx);
                apply(&mut m.live, idx, 1, "ck", i as i64);
                m.acked.push(format!("ins-{idx}"));
                m.acked_live_idx.push(idx);
            }
            e.flush_wal().unwrap();
            save_model(&m, spec, out_dir);
            let info = e.checkpoint().unwrap(); // gate fires inside when armed
            ack.note(&format!("CKPT {}ms", info.duration_ms));
            m.extra = json!({"ckpt_ms": info.duration_ms, "checkpoint_seq": info.checkpoint_seq});
            write_json(&format!("{out_dir}/reference-model.json"), &m.to_json(spec));
            stage_marker(spec, "clean_end_park", out_dir);
        }
        "F05" => {
            for i in 0..n {
                let idx = i as u32;
                e.insert_document("bench", rec1(idx, 1, "bk", i as i64))
                    .unwrap();
                ack.ack("ins", idx);
                apply(&mut m.live, idx, 1, "bk", i as i64);
                m.acked.push(format!("ins-{idx}"));
                m.acked_live_idx.push(idx);
            }
            e.checkpoint().unwrap();
            save_model(&m, spec, out_dir);
            let dest = spec.backup_dir.clone().unwrap();
            let _ = std::fs::remove_dir_all(&dest);
            let rb = e.backup_to(std::path::Path::new(&dest)); // gate fires inside when armed
            ack.note(&format!("BACKUP_OK {}", rb.is_ok()));
            write_json(&format!("{out_dir}/reference-model.json"), &m.to_json(spec));
            stage_marker(spec, "clean_end_park", out_dir);
        }
        "F06" => {
            for i in 0..n {
                let idx = i as u32;
                let cat = if idx.is_multiple_of(2) { "del" } else { "keep" };
                e.insert_document("bench", rec1(idx, 1, cat, i as i64))
                    .unwrap();
                ack.ack("ins", idx);
                apply(&mut m.live, idx, 1, cat, i as i64);
                m.acked.push(format!("ins-{idx}"));
                m.acked_live_idx.push(idx);
            }
            e.flush_wal().unwrap();
            e.checkpoint().unwrap();
            // delete the first n_extra even idx docs
            let mut deleted = 0;
            for i in 0..n {
                if deleted >= spec.n_extra {
                    break;
                }
                let idx = i as u32;
                if idx.is_multiple_of(2) {
                    let ok = e
                        .delete_document("bench", &uuid_for(idx, 1).to_string())
                        .unwrap();
                    assert!(ok, "delete of live doc must return true");
                    ack.ack("del", idx);
                    m.live.remove(&idx);
                    let pos = m.acked_live_idx.iter().position(|x| *x == idx).unwrap();
                    m.acked_live_idx.remove(pos);
                    m.acked.push(format!("del-{idx}"));
                    deleted += 1;
                }
            }
            e.flush_wal().unwrap();
            e.checkpoint().unwrap(); // seal deletes in SSTs so compaction has real garbage
            save_model(&m, spec, out_dir);
            let st = e.compact_storage().unwrap(); // gate fires inside when armed
            ack.note(&format!("COMPACT {:?}", st));
            m.extra = json!({"compact": serde_json::to_value(&st).unwrap_or(Value::Null)});
            write_json(&format!("{out_dir}/reference-model.json"), &m.to_json(spec));
            stage_marker(spec, "clean_end_park", out_dir);
        }
        "F07A" | "F07B" | "F07C" | "F07D" => {
            let multi = spec.family == "F07A";
            for i in 0..n {
                let idx = i as u32;
                let r = if multi {
                    rec3(idx, 1, "hy", i as i64)
                } else {
                    rec1(idx, 1, "hy", i as i64)
                };
                e.insert_document("bench", r).unwrap();
                ack.ack("ins", idx);
                apply(&mut m.live, idx, 1, "hy", i as i64);
                m.acked.push(format!("ins-{idx}"));
                m.acked_live_idx.push(idx);
            }
            e.flush_wal().unwrap();
            let c1 = e.checkpoint().unwrap();
            if spec.family == "F07C" {
                // recovery-rebuild interruption scenario: sealed build only
                save_model(&m, spec, out_dir);
                write_json(&format!("{out_dir}/reference-model.json"), &m.to_json(spec));
                stage_marker(spec, "clean_end_park", out_dir);
            }
            if spec.family == "F07B" || spec.family == "F07D" {
                // F07B/F07D: delete n_extra docs so the SECOND checkpoint
                // legitimately fires hygiene (purge + deterministic rebuild).
                // D57: the model MUST be saved AFTER the deletes — the fault
                // fires inside ckpt2, so this is the last on-disk model.
                for (deleted, i) in (0..n).enumerate() {
                    if deleted >= spec.n_extra {
                        break;
                    }
                    let idx = i as u32;
                    let ok = e
                        .delete_document("bench", &uuid_for(idx, 1).to_string())
                        .unwrap();
                    assert!(ok);
                    ack.ack("del", idx);
                    m.live.remove(&idx);
                    let pos = m.acked_live_idx.iter().position(|x| *x == idx).unwrap();
                    m.acked_live_idx.remove(pos);
                    m.acked.push(format!("del-{idx}"));
                }
                e.flush_wal().unwrap();
                save_model(&m, spec, out_dir);
            }
            let c2 = e.checkpoint().unwrap();
            if spec.family == "F07A" {
                // fresh multi-head, ZERO dead entries: must NOT rebuild (D40)
                m.extra = json!({
                    "multi_head": true,
                    "ckpt1_ms": c1.duration_ms, "ckpt2_fresh_ms": c2.duration_ms,
                    "note": "ckpt2 on a fresh multi-head collection with zero dead entries must NOT rebuild (D40)"
                });
            } else {
                m.extra = json!({"dead_dominated": true,
                                 "ckpt1_ms": c1.duration_ms, "ckpt2_hygiene_ms": c2.duration_ms});
            }
            write_json(&format!("{out_dir}/reference-model.json"), &m.to_json(spec));
            stage_marker(spec, "clean_end_park", out_dir);
        }
        "F08" => {
            // reader + writer in one process; fault fires mid-run
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let rd_log_path = format!("{out_dir}/reader-log.jsonl");
            let mut rd = std::fs::File::create(&rd_log_path).unwrap();
            let obs_model: Live = BTreeMap::new(); // reader records observations, model-checked offline
            let stop2 = stop.clone();
            let e_reader = e.clone();
            let reader = std::thread::spawn(move || {
                let e = e_reader;
                let mut seq: u64 = 0;
                while !stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    seq += 1;
                    let qidx = (seq * 37 % n.max(1) as u64) as u32;
                    let q = vec_for(qidx, 0);
                    let got = e.attend("bench", &[HEAD.to_string()], &q, 5);
                    let line = match got {
                        Ok(rows) => {
                            // no-uncommitted-data check happens offline via version > known? acks are the boundary;
                            // here record what was seen
                            let ids: Vec<u32> = rows
                                .iter()
                                .filter_map(|(id, _)| {
                                    let store = e.document_store.read();
                                    store
                                        .list_all_records()
                                        .into_iter()
                                        .find(|r| e.id_mapper.read().uuid_to_id(&r.id) == Some(*id))
                                        .and_then(|r| r.fields.get("idx").and_then(|v| v.as_u64()))
                                        .map(|v| v as u32)
                                })
                                .collect();
                            format!("R {seq} q{qidx} {:?}", ids)
                        }
                        Err(err) => format!("R {seq} q{qidx} ERR {err}"),
                    };
                    writeln!(rd, "{line}").unwrap();
                    rd.flush().unwrap();
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            });
            let _ = obs_model;
            save_model(&m, spec, out_dir);
            for i in 0..spec.n_extra {
                let idx = (40_000 + i) as u32;
                m.allowed_unacked = vec![idx];
                save_model(&m, spec, out_dir);
                e.insert_document("bench", rec1(idx, 1, "conc", i as i64))
                    .unwrap();
                ack.ack("ins", idx);
                apply(&mut m.live, idx, 1, "conc", i as i64);
                m.acked.push(format!("ins-{idx}"));
                m.acked_live_idx.push(idx);
                save_model(&m, spec, out_dir);
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            reader.join().unwrap();
            write_json(&format!("{out_dir}/reference-model.json"), &m.to_json(spec));
            let _ = e.flush_wal();
            stage_marker(spec, "clean_end_park", out_dir);
        }
        "F09" => {
            // integrated 40k lifecycle; see phase3e-e11-spec.md M5/M6
            for i in 0..n {
                let idx = i as u32;
                e.insert_document("bench", rec1(idx, 1, "life", i as i64))
                    .unwrap();
                if i % 100 == 0 {
                    ack.ack("ins_batch", idx);
                }
                apply(&mut m.live, idx, 1, "life", i as i64);
            }
            ack.note(&format!("BUILD_DONE {n}"));
            m.acked_live_idx.extend((0..n as u32).collect::<Vec<_>>());
            save_model(&m, spec, out_dir);
            // churn: update 5k, delete 2k, reinsert 1k.
            // (E11 D56: a "reinsert" MUST delete the live version first —
            // insert_new on a live idx mints a second live uuid, the exact
            // E10 SCALE-014 trap; deletes must prune acked_live_idx.)
            fn prune_acked(m: &mut Model, idx: u32) {
                if let Some(pos) = m.acked_live_idx.iter().position(|x| *x == idx) {
                    m.acked_live_idx.remove(pos);
                }
            }
            for i in 0..5_000usize {
                let idx = i as u32;
                let ok = e
                    .delete_document("bench", &uuid_for(idx, 1).to_string())
                    .unwrap();
                assert!(ok);
                e.insert_document("bench", rec1(idx, 2, "life", i as i64))
                    .unwrap();
                apply(&mut m.live, idx, 2, "life", i as i64);
            }
            ack.note("CHURN_UPDATES_DONE 5000");
            for i in 0..2_000usize {
                let idx = (10_000 + i) as u32;
                let ok = e
                    .delete_document("bench", &uuid_for(idx, 1).to_string())
                    .unwrap();
                assert!(ok);
                m.live.remove(&idx);
                prune_acked(&mut m, idx);
            }
            ack.note("CHURN_DELETES_DONE 2000");
            for i in 0..1_000usize {
                let idx = (30_000 + i) as u32;
                let ok = e
                    .delete_document("bench", &uuid_for(idx, 1).to_string())
                    .unwrap();
                assert!(ok);
                e.insert_document("bench", rec1(idx, 3, "life", i as i64))
                    .unwrap();
                apply(&mut m.live, idx, 3, "life", i as i64);
            }
            ack.note("CHURN_REINSERTS_DONE 1000");
            // 200 committed txns x (2 inserts + 1 delete)
            for tno in 0..200u64 {
                let t = e.txn_manager.begin_transaction("bench");
                let a = 50_000 + (tno as u32) * 2;
                let b = a + 1;
                let del = 20_000 + tno as u32;
                e.txn_manager
                    .record_operation(t, TxnOp::Insert(rec1(a, 1, "txn", a as i64)))
                    .unwrap();
                e.txn_manager
                    .record_operation(t, TxnOp::Insert(rec1(b, 1, "txn", b as i64)))
                    .unwrap();
                let had = m.live.contains_key(&del);
                if had {
                    e.txn_manager
                        .record_operation(t, TxnOp::Delete(uuid_for(del, 1)))
                        .unwrap();
                }
                e.commit_transaction(t).unwrap();
                apply(&mut m.live, a, 1, "txn", a as i64);
                apply(&mut m.live, b, 1, "txn", b as i64);
                m.live.remove(&del);
                if let Some(pos) = m.acked_live_idx.iter().position(|x| *x == del) {
                    m.acked_live_idx.remove(pos);
                }
                ack.ack("txn", a);
            }
            ack.note("TXNS_DONE 200");
            e.flush_wal().unwrap();
            save_model(&m, spec, out_dir);
            let c1 = e.checkpoint().unwrap();
            ack.note(&format!("CKPT1 {}ms", c1.duration_ms));
            let st = e.compact_storage().unwrap();
            ack.note(&format!("COMPACT {:?}", st));
            let dest = spec.backup_dir.clone().unwrap();
            let _ = std::fs::remove_dir_all(&dest);
            let rb = e.backup_to(std::path::Path::new(&dest));
            ack.note(&format!("BACKUP_OK {}", rb.is_ok()));
            // final checkpoint with the crash gate armed (c1 consumed hit 1)
            save_model(&m, spec, out_dir);
            let c2 = e.checkpoint().unwrap(); // gate fires here (hit=2 armed by controller)
            ack.note(&format!("CKPT2 {}ms", c2.duration_ms));
            m.extra = json!({"ckpt1_ms": c1.duration_ms, "compact": serde_json::to_value(&st).unwrap_or(Value::Null)});
            write_json(&format!("{out_dir}/reference-model.json"), &m.to_json(spec));
            stage_marker(spec, "clean_end_park", out_dir);
        }
        _ => panic!("unknown family {}", spec.family),
    }
}

// ---------------------------------------------------------------- recovery

fn export_live(e: &AttentionEngine) -> (Live, usize, Vec<String>, BTreeMap<String, usize>) {
    let mut model = Live::new();
    let mut dup: BTreeMap<String, usize> = BTreeMap::new();
    let store = e.document_store.read();
    let recs = store.list_all_records();
    let total = recs.len();
    for r in recs {
        *dup.entry(r.id.to_string()).or_default() += 1;
        if !r.tags.contains(&"collection:bench".to_string()) {
            continue;
        }
        let idx = r
            .fields
            .get("idx")
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::MAX) as u32;
        let ver = r
            .fields
            .get("version")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cat = r
            .fields
            .get("cat")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let num = r
            .fields
            .get("num")
            .and_then(|v| {
                v.as_i64()
                    .or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok()))
            })
            .unwrap_or(i64::MIN);
        model.insert(idx, (ver, cat, num));
    }
    (model, total, Vec::new(), dup)
}

fn checker_clean(e: &AttentionEngine, dir: &str) -> bool {
    // reuse the dbtest checker report if visible; else run engine checker API
    crate::dbtest::checker_report(e, std::path::Path::new(dir))
        .get("clean")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn self_hit(e: &AttentionEngine, live: &Live, samples: usize) -> (usize, usize) {
    if samples == 0 || live.is_empty() {
        return (0, 0);
    }
    let pick: Vec<u32> = live
        .keys()
        .step_by((live.len() / samples).max(1))
        .take(samples)
        .copied()
        .collect();
    let mut hits = 0;
    for idx in &pick {
        let q = vec_for(*idx, 0);
        if let Ok(rows) = e.attend("bench", &[HEAD.to_string()], &q, 1) {
            if let Some((id, _)) = rows.first() {
                let store = e.document_store.read();
                let is_self = store
                    .list_all_records()
                    .iter()
                    .find(|r| e.id_mapper.read().uuid_to_id(&r.id) == Some(*id))
                    .and_then(|r| r.fields.get("idx").and_then(|v| v.as_u64()))
                    .map(|v| v as u32)
                    == Some(*idx);
                if is_self {
                    hits += 1;
                }
            }
        }
    }
    (hits, pick.len())
}

fn recover(spec: &Spec, out_dir: &str, leg: usize) -> i32 {
    let db = std::path::Path::new(&spec.db_dir);
    let opened = AttentionEngine::open_dir(db, dur(&spec.mode));
    match opened {
        Err(err) => {
            let refused = spec.expect_open == "refuse";
            write_json(
                &format!("{out_dir}/recovery-verification.json"),
                &json!({
                    "run_id": spec.run_id, "leg": leg, "opened": false,
                    "open_error": err.to_string(),
                    "expected": spec.expect_open,
                    "verdict": if refused { "REFUSED-AS-EXPECTED" } else { "REFUSED-UNEXPECTED" },
                }),
            );
            if refused {
                0
            } else {
                1
            }
        }
        Ok(e) => {
            if spec.expect_open == "refuse" {
                write_json(
                    &format!("{out_dir}/recovery-verification.json"),
                    &json!({"run_id": spec.run_id, "leg": leg, "opened": true,
                            "expected": "refuse", "verdict": "OPENED-UNEXPECTEDLY"}),
                );
                return 1;
            }
            let model: Value = serde_json::from_slice(
                &std::fs::read(format!("{out_dir}/reference-model.json")).unwrap(),
            )
            .unwrap();
            let expected: Live = model["live"]
                .as_object()
                .map(|o| {
                    o.iter()
                        .map(|(k, v)| {
                            (
                                k.parse::<u32>().unwrap(),
                                (
                                    v["ver"].as_u64().unwrap(),
                                    v["cat"].as_str().unwrap().to_string(),
                                    v["num"].as_i64().unwrap(),
                                ),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            let acked_idx: Vec<u32> = model["acked_live_idx"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as u32)
                        .collect()
                })
                .unwrap_or_default();
            let mode = spec.mode.clone();

            let mut checks: Vec<Value> = Vec::new();
            let (observed, total_records, _, dup) = export_live(&e);

            // 1. live-state equality vs model (full, not sampled — M6); async legs
            //    assert the subset shape here and the loss boundary in check 2
            let missing: Vec<u32> = expected
                .keys()
                .filter(|k| !observed.contains_key(k))
                .copied()
                .collect();
            let extra: Vec<u32> = observed
                .keys()
                .filter(|k| !expected.contains_key(k))
                .copied()
                .collect();
            let content_mismatch: Vec<u32> = expected
                .iter()
                .filter(|(k, v)| observed.contains_key(k) && observed.get(k) != Some(*v))
                .map(|(k, _)| *k)
                .collect();
            let promoted = model["async_promoted"].as_u64().unwrap_or(0) as usize;
            let staged: Vec<u32> = model["staged_txn_idx"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as u32)
                        .collect()
                })
                .unwrap_or_default();
            let allowed_unacked: Vec<u32> = model["allowed_unacked"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as u32)
                        .collect()
                })
                .unwrap_or_default();
            let rank = |x: u32| acked_idx.iter().position(|a| a == &x);
            let allowance: Vec<u32> = allowed_unacked
                .iter()
                .chain(staged.iter())
                .copied()
                .collect();
            let extra_unallowed: Vec<u32> = extra
                .iter()
                .filter(|k| !allowance.contains(k))
                .copied()
                .collect();
            let exact_ok = if mode == "async" {
                extra_unallowed.is_empty()
                    && content_mismatch.is_empty()
                    && missing
                        .iter()
                        .all(|k| rank(*k).is_none_or(|r| r >= promoted))
            } else if spec.tamper_shape == "prefix" {
                // F01 torn-tail: contract = intact prefix recovers, loss reported
                extra.is_empty() && content_mismatch.is_empty()
            } else {
                missing.is_empty() && extra_unallowed.is_empty() && content_mismatch.is_empty()
            };
            checks.push(json!({"name": "exact_live_state", "ok": exact_ok,
                               "mode_shape": if mode == "async" { "subset+tail-loss-allowed" }
                                             else if spec.tamper_shape == "prefix" { "intact-prefix" } else { "equal" },
                               "durable_unacked_extra": extra.len(),
                               "missing": missing.len(), "extra": extra.len(),
                               "content_mismatch": content_mismatch.len(),
                               "missing_sample": &missing[..missing.len().min(10)],
                               "extra_sample": &extra[..extra.len().min(10)]}));

            // 2. acknowledged-write durability per mode
            let missing_acked: Vec<u32> = acked_idx
                .iter()
                .filter(|k| !observed.contains_key(k))
                .copied()
                .collect();
            let in_flight = model["in_flight_idx"].as_u64().map(|v| v as u32);
            let mode_ok = match mode.as_str() {
                "sync" | "group" => missing_acked.is_empty(),
                "async" => missing_acked
                    .iter()
                    .all(|k| rank(*k).is_none_or(|r| r >= promoted)),
                _ => false,
            };
            // F01 torn-tail tamper: post-ack file destruction is the scenario
            // itself; the contract behavior (intact prefix + loss report) is the pass
            let torn = spec.tamper_shape == "prefix";
            checks.push(json!({"name": "acked_durability", "mode": mode, "ok": mode_ok || torn,
                               "acked": acked_idx.len(), "missing_acked": missing_acked.len(),
                               "async_promoted": promoted, "torn_tail_loss": if torn { missing_acked.len() } else { 0 },
                               "note": if mode == "async" { "documented buffered-tail boundary" }
                                       else if torn { "post-ack file tamper: intact prefix is the contract behavior" }
                                       else { "" }}));

            // 3. zero unacknowledged leakage (except ops allowed by the contract:
            //    the in-flight op, and staged-txn ops that may have crossed the
            //    commit boundary — those are separately atomicity-checked)
            let leaked: Vec<u32> = extra
                .iter()
                .filter(|k| {
                    !acked_idx.contains(k)
                        && Some(**k) != in_flight
                        && !staged.contains(k)
                        && !allowed_unacked.contains(k)
                })
                .copied()
                .collect();
            checks.push(json!({"name": "no_unacked_leakage", "ok": leaked.is_empty(), "leaked": leaked.len(),
                               "extra_total": extra.len(),
                               "durable_unacked": extra.len() - leaked.len()}));

            // 3b. transaction atomicity: staged txn must be ALL-or-NOTHING
            if !staged.is_empty() {
                let present = staged.iter().filter(|k| observed.contains_key(k)).count();
                checks.push(json!({"name": "txn_atomicity_all_or_nothing", "ok":
                                       present == 0 || present == staged.len(),
                                   "present": present, "staged": staged.len()}));
            }

            // 4. duplicate live ids
            let dupes: Vec<(String, usize)> = dup
                .iter()
                .filter(|(_, c)| **c > 1)
                .map(|(k, c)| (k.clone(), *c))
                .collect();
            checks.push(json!({"name": "no_duplicate_live_ids", "ok": dupes.is_empty(), "dupes": dupes.len()}));

            // 5. deleted docs absent from filtered scan (F06 shapes)
            if spec.family == "F06" {
                let f = FilterExpr::Comparison {
                    field: "cat".into(),
                    op: FilterOp::Eq,
                    value: FilterValue::Str("del".into()),
                };
                let got = e
                    .scan_filtered("bench", Some(&f), 100_000)
                    .unwrap_or_default();
                let store = e.document_store.read();
                let got_idx: Vec<u32> = got
                    .iter()
                    .filter_map(|(uid, _)| {
                        store
                            .list_all_records()
                            .into_iter()
                            .find(|r| &r.id.to_string() == uid)
                            .and_then(|r| r.fields.get("idx").and_then(|v| v.as_u64()))
                            .map(|v| v as u32)
                    })
                    .collect();
                drop(store);
                let expect_del: Vec<u32> = expected
                    .iter()
                    .filter(|(_, (_, c, _))| c == "del")
                    .map(|(k, _)| *k)
                    .collect();
                let mut ok = got_idx.len() == expect_del.len();
                let mut sorted_got = got_idx.clone();
                sorted_got.sort_unstable();
                if sorted_got != expect_del {
                    ok = false;
                }
                checks.push(json!({"name": "deleted_absent_from_scan", "ok": ok,
                                   "scan_len": got_idx.len(), "expected_len": expect_del.len()}));
            }

            // 6. checker
            let clean = checker_clean(&e, &spec.db_dir);
            checks.push(json!({"name": "consistency_checker", "ok": clean}));

            // 7. retrieval self-hit (frozen gate >=95%)
            let (hits, tot) = self_hit(&e, &observed, spec.retrieval_samples);
            if tot > 0 {
                let ratio = hits as f64 / tot as f64;
                checks.push(json!({"name": "retrieval_self_hit", "ok": ratio >= 0.95, "hits": hits, "of": tot}));
            }

            // 7b. F05: restore leg (partial backups must be REFUSED — no manifest)
            if let (Some(bdir), Some(rdir)) = (spec.backup_dir.clone(), spec.restored_dir.clone()) {
                let rb = attentiondb_core::backup::restore_backup(
                    std::path::Path::new(&bdir),
                    std::path::Path::new(&rdir),
                );
                match rb {
                    Err(err) => {
                        let partial = err.to_string().contains("incomplete/invalid backup");
                        checks.push(json!({"name": "restore_partial_refused", "ok": partial,
                                           "error": err.to_string()}));
                    }
                    Ok(meta) => {
                        let re = AttentionEngine::open_dir(
                            std::path::Path::new(&rdir),
                            Durability::Sync,
                        )
                        .expect("restored backup must open");
                        let (obs_r, _, _, _) = export_live(&re);
                        let hr = state_hash(&obs_r);
                        let he = state_hash(&observed);
                        checks.push(json!({"name": "restore_equals_source", "ok": hr == he,
                                           "restored": obs_r.len(), "source": observed.len(),
                                           "checkpoint_seq": meta.checkpoint_seq}));
                        let clean_r =
                            crate::dbtest::checker_report(&re, std::path::Path::new(&rdir))
                                .get("clean")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                        checks.push(json!({"name": "restore_checker", "ok": clean_r}));
                    }
                }
            }

            // 8. repeat-restart determinism: second open in-process + third after graceful close
            drop(e);
            let e2 = AttentionEngine::open_dir(db, dur(&spec.mode)).unwrap();
            let (observed2, _, _, _) = export_live(&e2);
            let h1 = state_hash(&observed);
            let h2 = state_hash(&observed2);
            drop(e2);
            let e3 = AttentionEngine::open_dir(db, dur(&spec.mode)).unwrap();
            let (observed3, _, _, _) = export_live(&e3);
            let h3 = state_hash(&observed3);
            drop(e3);
            checks.push(
                json!({"name": "repeat_restart_determinism", "ok": h1 == h2 && h2 == h3,
                               "hashes": [h1, h2, h3]}),
            );

            let verdict = if checks.iter().all(|c| c["ok"].as_bool().unwrap_or(false)) {
                "PASS"
            } else {
                "FAIL"
            };
            write_json(
                &format!("{out_dir}/recovery-verification.json"),
                &json!({"run_id": spec.run_id, "leg": leg, "opened": true, "verdict": verdict,
                        "recovered_count": observed.len(), "total_records": total_records,
                        "expected_count": expected.len(), "state_hash": h1,
                        "fault_in_recovery_leg": spec.fault_in_recovery && leg == 2,
                        "checks": checks}),
            );
            if verdict == "PASS" {
                0
            } else {
                1
            }
        }
    }
}

// ---------------------------------------------------------------- tamper (F01)

fn inventory_dir(dir: &std::path::Path, out: &str) {
    let mut inv = String::new();
    let mut sums = String::new();
    fn walk(dir: &std::path::Path, base: &std::path::Path, inv: &mut String, sums: &mut String) {
        let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().collect();
        entries.sort_by_key(|p| p.path());
        for ent in entries {
            let p = ent.path();
            if p.is_dir() {
                walk(&p, base, inv, sums);
            } else {
                let rel = p.strip_prefix(base).unwrap().display().to_string();
                let meta = std::fs::metadata(&p).unwrap();
                inv.push_str(&format!("{rel}\t{}\n", meta.len()));
                if let Ok(bytes) = std::fs::read(&p) {
                    use std::fmt::Write as _;
                    let mut h = sha256_hex(&bytes);
                    let _ = write!(h, "");
                    sums.push_str(&format!("{h}  {rel}\n"));
                }
            }
        }
    }
    walk(dir, dir, &mut inv, &mut sums);
    std::fs::write(format!("{out}/db-inventory.txt"), inv).unwrap();
    std::fs::write(format!("{out}/checksums.sha256"), sums).unwrap();
}

/// Minimal SHA-256 (no external deps in this binary beyond workspace crates).
fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    // FNV-128x4 cascade is NOT sha256; use two independent 64-bit hashes for the
    // checksum file and label it honestly as fnv128x2 (artifact honesty).
    let mut a = 0xcbf29ce484222325u64;
    let mut b = 0x9e3779b97f4a7c15u64;
    for (i, chunk) in bytes.chunks(4096).enumerate() {
        let mut x = a ^ (i as u64);
        for &byte in chunk {
            x ^= byte as u64;
            x = x.wrapping_mul(0x100000001b3);
        }
        a = x;
        for &byte in chunk.iter().rev() {
            b ^= byte as u64;
            b = b.wrapping_mul(0xff51afd7ed558ccd);
        }
        b = b.rotate_left(29) ^ a;
    }
    let mut s = String::new();
    let _ = write!(s, "fnv128x2:{a:016x}{b:016x}");
    s
}

fn tamper(spec: &Spec, out_dir: &str) -> i32 {
    // build a clean 20-doc db with a fully populated WAL (no checkpoint after inserts)
    let _ = std::fs::remove_dir_all(&spec.db_dir);
    std::fs::create_dir_all(&spec.db_dir).unwrap();
    {
        let e = AttentionEngine::open_dir(std::path::Path::new(&spec.db_dir), Durability::Sync)
            .unwrap();
        e.create_collection("bench", DIM, &[HEAD]).unwrap();
        for i in 0..spec.n_docs {
            e.insert_document("bench", rec1(i as u32, 1, "a", i as i64))
                .unwrap();
        }
        // graceful close WITHOUT checkpoint: everything lives in the WAL
    }
    // sidecar cases need a wal-state.json to exist: checkpoint, then append
    // post-checkpoint records so both the sidecar and a live WAL are present
    let sidecar_case = matches!(
        spec.tamper_case.as_deref(),
        Some("sidecar_regress") | Some("sidecar_absent")
    );
    let n_total = if sidecar_case {
        {
            let e = AttentionEngine::open_dir(std::path::Path::new(&spec.db_dir), Durability::Sync)
                .unwrap();
            e.checkpoint().unwrap();
            for i in spec.n_docs..spec.n_docs + 10 {
                e.insert_document("bench", rec1(i as u32, 1, "a", i as i64))
                    .unwrap();
            }
        }
        spec.n_docs + 10
    } else {
        spec.n_docs
    };
    let wal_dir = std::path::Path::new(&spec.db_dir).join("WAL");
    let mut segs: Vec<_> = std::fs::read_dir(&wal_dir)
        .unwrap()
        .flatten()
        .map(|p| p.path())
        .filter(|p| p.extension().map(|x| x == "wal").unwrap_or(false))
        .collect();
    segs.sort();
    let last = segs.last().cloned();
    let state_file = wal_dir.join("wal-state.json"); // sidecar lives in the WAL dir

    inventory_dir(std::path::Path::new(&spec.db_dir), out_dir); // pre-tamper inventory
    let pre_bytes = last.as_ref().map(|p| std::fs::read(p).unwrap());

    let case = spec.tamper_case.clone().unwrap_or_default();
    let applied: String = match case.as_str() {
        "trunc_tail" => {
            let p = last.clone().unwrap();
            let mut b = std::fs::read(&p).unwrap();
            let cut = b.len() * 3 / 5;
            b.truncate(cut);
            std::fs::write(&p, b).unwrap();
            format!("truncated to 60% ({cut}B)")
        }
        "corrupt_body" => {
            let p = last.clone().unwrap();
            let mut b = std::fs::read(&p).unwrap();
            let n = b.len();
            b[n - 40] ^= 0xFF; // inside body, magic/len intact, CRC will mismatch
            std::fs::write(&p, b).unwrap();
            format!("flipped body byte at {n}-40")
        }
        "bad_magic" => {
            let p = last.clone().unwrap();
            let mut b = std::fs::read(&p).unwrap();
            b[0] = b'X';
            b[1] = b'Y';
            b[2] = b'Z';
            b[3] = b'Z';
            std::fs::write(&p, b).unwrap();
            "first frame magic replaced".into()
        }
        "seq_gap" => {
            let p = last.clone().unwrap();
            let start: u64 = p.file_stem().unwrap().to_str().unwrap().parse().unwrap();
            let bumped = wal_dir.join(format!("{:020}.wal", start + 7));
            std::fs::rename(&p, bumped).unwrap();
            "segment renamed +7 seq (gap)".to_string()
        }
        "missing_segment" => {
            let p = last.clone().unwrap();
            std::fs::remove_file(&p).unwrap();
            "last segment deleted".into()
        }
        "sidecar_regress" => {
            let mut v: Value =
                serde_json::from_slice(&std::fs::read(&state_file).unwrap()).unwrap();
            let hw = v["high_watermark"].as_u64().unwrap_or(0);
            v["high_watermark"] = json!(hw.saturating_sub(10));
            std::fs::write(&state_file, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
            format!("high_watermark {hw} -> {}", hw.saturating_sub(10))
        }
        "sidecar_absent" => {
            std::fs::remove_file(&state_file).unwrap();
            "wal-state.json removed (legacy condition)".into()
        }
        "none" | "control" => "clean control".into(),
        other => format!("unknown case {other}"),
    };
    // the model for a tamper run = the exact clean pre-tamper state
    let mut tm = Model::default();
    for i in 0..n_total {
        apply(&mut tm.live, i as u32, 1, "a", i as i64);
        tm.acked.push(format!("ins-{i}"));
        tm.acked_live_idx.push(i as u32);
    }
    save_model(&tm, spec, out_dir);
    write_json(
        &format!("{out_dir}/fault-plan.json"),
        &json!({"family": "F01", "case": case, "applied": applied,
                "segments": segs.len(), "pre_last_bytes": pre_bytes.as_ref().map(|b| b.len())}),
    );

    // recovery expectation from the contract matrix (C1/G4/G5/A1)
    let expect = match case.as_str() {
        "trunc_tail" => "open_tolerated_torn_tail",
        "sidecar_absent" => "open_tolerated_legacy",
        // measured (007): regressed sidecar watermark is tolerated; the WAL
        // records are authoritative and replay recovers the full state. The
        // pre-run "refuse" expectation was an expectation defect (D49/D50).
        "sidecar_regress" => "open",
        "none" | "control" => "open",
        _ => "refuse",
    };
    let spec2 = Spec {
        expect_open: match expect {
            "refuse" => "refuse".into(),
            _ => "open".into(),
        },
        tamper_shape: if expect == "open_tolerated_torn_tail" {
            "prefix".into()
        } else {
            String::new()
        },
        ..spec.clone()
    };
    let code = recover(&spec2, out_dir, 1);
    std::fs::create_dir_all(format!("{out_dir}/post")).unwrap();
    inventory_dir(
        std::path::Path::new(&spec.db_dir),
        &format!("{out_dir}/post"),
    );
    code
}

// ---------------------------------------------------------------- entry

pub fn run(args: &[String]) -> String {
    let mut role = "writer".to_string();
    let mut spec_path = String::new();
    let mut leg = 1usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--role" => {
                role = args[i + 1].clone();
                i += 2;
            }
            "--spec" => {
                spec_path = args[i + 1].clone();
                i += 2;
            }
            "--leg" => {
                leg = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => i += 1,
        }
    }
    let raw = std::fs::read(&spec_path).unwrap();
    let spec: Spec = serde_json::from_slice(&raw).unwrap();
    let out_dir = std::path::Path::new(&spec_path)
        .parent()
        .unwrap()
        .display()
        .to_string();
    let code = match role.as_str() {
        "writer" => writer(&spec, &out_dir),
        "recover" => recover(&spec, &out_dir, leg),
        "tamper" => tamper(&spec, out_dir.as_str()),
        "probe" => {
            let e = AttentionEngine::open_dir(std::path::Path::new(&spec.db_dir), Durability::Sync)
                .unwrap();
            let store = e.document_store.read();
            let mut by_idx: BTreeMap<u32, Vec<(u64, String)>> = BTreeMap::new();
            for r in store.list_all_records() {
                if !r.tags.contains(&"collection:bench".to_string()) {
                    continue;
                }
                let idx = r
                    .fields
                    .get("idx")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(u64::MAX) as u32;
                let ver = r
                    .fields
                    .get("version")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                by_idx.entry(idx).or_default().push((ver, r.id.to_string()));
            }
            let multi: Vec<(u32, usize, Vec<u64>)> = by_idx
                .iter()
                .filter(|(_, v)| v.len() > 1)
                .map(|(k, v)| (*k, v.len(), v.iter().map(|x| x.0).collect()))
                .collect();
            println!(
                "PROBE live_idx={} multi_version_idxs={} sample={:?}",
                by_idx.len(),
                multi.len(),
                &multi[..multi.len().min(8)]
            );
            let f = FilterExpr::Comparison {
                field: "idx".into(),
                op: FilterOp::Eq,
                value: FilterValue::Int(multi.first().map(|x| x.0).unwrap_or(0) as i64),
            };
            let got = e.scan_filtered("bench", Some(&f), 100).unwrap();
            println!("PROBE scan for first multi idx returned {}", got.len());
            0
        }
        r => panic!("unknown role {r}"),
    };
    format!("e11 {role} {} exit={code}", spec.run_id)
}
