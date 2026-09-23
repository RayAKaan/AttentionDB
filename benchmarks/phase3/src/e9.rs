//! E9 memory harness — control experiments with rich memory telemetry.
//!
//! Telemetry (every ~2 s, inline with the op loop): VmRSS/VmSize/RssAnon/
//! RssFile (/proc/self/status), Pss/Pss_Anon/Pss_File (/proc/self/smaps_rollup,
//! NA when unavailable), fd count, thread count, glibc mallinfo2 (uordblks =
//! heap in-use, fordblks = heap free retained by allocator), db-dir census
//! (WAL/SST bytes + file counts), and the read-only engine component census
//! (mapper / DocumentStore / txn manager / per-collection structures).
//!
//! Every experiment writes config.json, telemetry.csv, census.csv and
//! summary.json into its out dir. No oplog: correctness claims live in the
//! E1–E8 sealed suites; these runs exist to attribute memory, and the
//! optimized-soak correctness gate reuses the E8 families under new IDs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use attentiondb_core::engine::AttentionEngine;

use crate::dbtest::{doc_record, open_db, uuid_for, vec_for};

const DIM: usize = 32;

// ------------------------------------------------------------ glibc mallinfo2

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Mallinfo2 {
    arena: usize,
    ordblks: usize,
    smblks: usize,
    hblks: usize,
    hblkhd: usize,
    usmblks: usize,
    fsmblks: usize,
    uordblks: usize,
    fordblks: usize,
    keepcost: usize,
}

extern "C" {
    fn mallinfo2() -> Mallinfo2;
}

// ------------------------------------------------------------ /proc readers

pub(crate) fn read_status() -> (usize, usize, usize, usize, usize) {
    // (vm_rss_kb, vm_size_kb, rss_anon_kb, rss_file_kb, threads)
    let mut out = (0usize, 0usize, 0usize, 0usize, 0usize);
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        let get = |key: &str| -> usize {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix(key) {
                    let rest = rest.trim_start_matches(':').trim();
                    if let Some(num) = rest.split_whitespace().next() {
                        return num.parse().unwrap_or(0);
                    }
                }
            }
            0
        };
        // closure borrow trick: compute sequentially
        let rss = get("VmRSS");
        let vmz = get("VmSize");
        let anon = get("RssAnon");
        let file = get("RssFile");
        let thr = get("Threads");
        out = (rss, vmz, anon, file, thr);
    }
    out
}

pub(crate) fn read_pss() -> (Option<usize>, Option<usize>, Option<usize>) {
    // (Pss, Pss_Anon, Pss_File) in kB; None when smaps_rollup unavailable
    let mut pss = None;
    let mut pa = None;
    let mut pf = None;
    if let Ok(s) = std::fs::read_to_string("/proc/self/smaps_rollup") {
        for line in s.lines() {
            let mut it = line.split_whitespace();
            let key = it.next().unwrap_or("");
            let val = it.next().and_then(|v| v.parse::<usize>().ok());
            match key {
                "Pss:" => pss = val,
                "Pss_Anon:" => pa = val,
                "Pss_File:" => pf = val,
                _ => {}
            }
        }
    }
    (pss, pa, pf)
}

pub(crate) fn fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd").map(|d| d.count()).unwrap_or(0)
}

pub(crate) fn dir_census(db: &Path) -> (usize, usize, usize, usize, usize) {
    // (wal_bytes, wal_files, sst_bytes, sst_files, db_bytes)
    let mut wal_b = 0;
    let mut wal_n = 0;
    let mut sst_b = 0;
    let mut sst_n = 0;
    let mut all_b = 0;
    if let Ok(rd) = std::fs::read_dir(db) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if let Ok(sub) = std::fs::read_dir(&p) {
                    for f in sub.flatten() {
                        if let Ok(md) = f.metadata() {
                            if md.is_file() {
                                let l = md.len() as usize;
                                all_b += l;
                                if p.file_name().unwrap_or_default() == "wal" {
                                    wal_b += l;
                                    wal_n += 1;
                                }
                                if p.file_name().unwrap_or_default() == "sst"
                                    && f.file_name().to_string_lossy().ends_with(".sst")
                                {
                                    sst_b += l;
                                    sst_n += 1;
                                }
                            }
                        }
                    }
                }
            } else if let Ok(md) = e.metadata() {
                all_b += md.len() as usize;
            }
        }
    }
    (wal_b, wal_n, sst_b, sst_n, all_b)
}

// ------------------------------------------------------------ telemetry

struct Tel {
    out: PathBuf,
    t0: std::time::Instant,
    last_sample: f64,
    rows: Vec<String>,
    census_rows: Vec<String>,
    samples: Vec<[f64; 8]>, // t, op, rss, pss, anon, uord, ford, sst_files
}

impl Tel {
    fn new(out: &Path) -> Self {
        
        Tel {
            out: out.to_path_buf(),
            t0: std::time::Instant::now(),
            last_sample: -10.0,
            rows: vec![],
            census_rows: vec![],
            samples: vec![],
        }
    }
    fn maybe(&mut self, e: &AttentionEngine, db: &Path, op: usize, force: bool) {
        let el = self.t0.elapsed().as_secs_f64();
        if !force && el - self.last_sample < 2.0 {
            return;
        }
        self.last_sample = el;
        let (rss, vmz, anon, file, thr) = read_status();
        let (pss, pa, pf) = read_pss();
        let mi = unsafe { mallinfo2() };
        let fds = fd_count();
        let (walb, waln, sstb, sstn, dbb) = dir_census(db);
        let ps = pss.map(|v| v.to_string()).unwrap_or_else(|| "NA".into());
        let pas = pa.map(|v| v.to_string()).unwrap_or_else(|| "NA".into());
        let pfs = pf.map(|v| v.to_string()).unwrap_or_else(|| "NA".into());
        let arena = mi.arena;
        let uord = mi.uordblks;
        let ford = mi.fordblks;
        self.rows.push(format!(
            "{el:.1},{op},{rss},{vmz},{anon},{file},{ps},{pas},{pfs},{thr},{fds},{uord},{ford},{arena},{walb},{waln},{sstb},{sstn},{dbb}"
        ));
        self.samples.push([el, op as f64, rss as f64, pss.map(|v| v as f64).unwrap_or(0.0), anon as f64, mi.uordblks as f64, mi.fordblks as f64, sstn as f64]);
        let c = e.mem_census();
        let hdr = format!(
            "{el:.1},{op},mapper,{u2i},{i2u},{ret},{nid},store,{mt},{fl},{sr},{bc},txn,{ts},{to}",
            u2i = c.mapper_uuid_to_u64, i2u = c.mapper_u64_to_uuid, ret = c.mapper_retired, nid = c.mapper_next_id,
            mt = c.store_memtable, fl = c.store_flushed_records, sr = c.store_sst_readers,
            bc = c.store_block_cache_entries, ts = c.txn_staged, to = c.txn_staged_ops
        );
        self.census_rows.push(hdr);
        for cc in &c.collections {
            let nm = cc.name.clone();
            self.census_rows.push(format!(
                "{el:.1},{op},coll,{nm},{hd},{ri},vs,{vs},bm25t,{bt},{bp},bmdl,{dl},idfc,{ic}",
                hd = cc.heads, ri = cc.retired_ids, vs = cc.vector_store_len,
                bt = cc.bm25_terms, bp = cc.bm25_postings, dl = cc.bm25_doc_lengths, ic = cc.idf_cache
            ));
        }
    }
    fn flush(&self, summary: &serde_json::Value) {
        let _ = std::fs::write(
            self.out.join("telemetry.csv"),
            "t_s,op,rss_kb,vmz_kb,anon_kb,file_kb,pss_kb,pss_anon_kb,pss_file_kb,threads,fds,uord_kb,ford_kb,arena_kb,wal_bytes,wal_files,sst_bytes,sst_files,db_bytes\n"
                .to_string() + &self.rows.join("\n"),
        );
        let _ = std::fs::write(self.out.join("census.csv"), self.census_rows.join("\n"));
        let _ = std::fs::write(self.out.join("summary.json"), summary.to_string());
    }
}

struct W {
    e: AttentionEngine,
    db: PathBuf,
    tel: Tel,
    live: HashMap<u32, u64>, // idx -> current version
    n_ins: usize,
    n_del: usize,
    n_upd: usize,
    seq: usize,
}

impl W {
    fn new(out: &Path, exp: &str) -> Self {
        let db = out.join("db");
        std::fs::create_dir_all(&db).unwrap();
        let cfg = serde_json::json!({
            "experiment": exp, "dim": DIM, "durability": std::env::var("PH3D_DURABILITY").unwrap_or_else(|_| "sync".into()),
            "wal_segment_bytes": std::env::var("ATTENTIONDB_WAL_SEGMENT_BYTES").unwrap_or_else(|_| "default".into()),
            "commit": "3e65d50f8fd1a81cda339d4608260d6be98ba708 (E8 working tree)",
        });
        let _ = std::fs::write(out.join("config.json"), cfg.to_string());
        let e = open_db(&db);
        e.create_collection("bench", DIM, &["h"]).unwrap();
        W {
            e,
            db,
            tel: Tel::new(out),
            live: HashMap::new(),
            n_ins: 0,
            n_del: 0,
            n_upd: 0,
            seq: 0,
        }
    }
    fn insert(&mut self, idx: u32) {
        let ver = self.live.get(&idx).copied().unwrap_or(0) + 1;
        let r = doc_record(idx, ver, "e9", 100 + (self.seq % 900) as i64);
        self.e.insert_document("bench", r).unwrap();
        self.live.insert(idx, ver);
        self.n_ins += 1;
    }
    fn delete(&mut self, idx: u32) {
        let ver = self.live[&idx];
        self.e.delete_document("bench", &uuid_for(idx, ver).to_string()).unwrap();
        self.live.remove(&idx);
        self.n_del += 1;
    }
    fn update(&mut self, idx: u32) {
        let ver = self.live[&idx];
        let r = doc_record(idx, ver, "e9", 100 + (self.seq % 900) as i64);
        // update preserves the uuid (engine identity note)
        self.e
            .update_document("bench", &uuid_for(idx, ver).to_string(), r.fields.clone(), r.k_vecs.clone())
            .unwrap();
        self.n_upd += 1;
    }
}

fn summarize(w: &W, extra: serde_json::Value) -> serde_json::Value {
    let tel = &w.tel;
    let rssf = tel.samples.first().map(|s| s[2]).unwrap_or(0.0);
    let rssl = tel.samples.last().map(|s| s[2]).unwrap_or(0.0);
    let peak = tel.samples.iter().map(|s| s[2]).fold(0.0, f64::max);
    let pssl = tel.samples.last().map(|s| s[3]).unwrap_or(0.0);
    let uordl = tel.samples.last().map(|s| s[5]).unwrap_or(0.0);
    let fordl = tel.samples.last().map(|s| s[6]).unwrap_or(0.0);
    serde_json::json!({
        "ops": w.seq, "inserts": w.n_ins, "deletes": w.n_del, "updates": w.n_upd,
        "live_docs": w.live.len(), "elapsed_s": tel.t0.elapsed().as_secs_f64() as u64,
        "rss_first_kb": rssf, "rss_last_kb": rssl, "rss_peak_kb": peak, "pss_last_kb": pssl,
        "heap_used_kb": uordl, "heap_free_retained_kb": fordl,
        "tel_samples": tel.samples.len(), "ext": extra,
    })
}

fn churn_mix(w: &mut W, keys: u32, target_live: u32, budget: usize, ckpt: usize, compact: usize) {
    let mut h: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = move || {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        h
    };
    for _ in 0..budget {
        w.seq += 1;
        let r = rnd();
        let pick = (r % keys as u64) as u32;
        if w.live.len() < target_live as usize || (r >> 8) % 100 < 15 && !w.live.contains_key(&pick) {
            w.insert(pick);
        } else if (r >> 8) % 100 < 45 {
            if w.live.contains_key(&pick) {
                w.delete(pick);
            } else {
                w.insert(pick);
            }
        } else if w.live.contains_key(&pick) {
            w.update(pick);
        } else {
            w.insert(pick);
        }
        if ckpt > 0 && w.seq.is_multiple_of(ckpt) {
            w.e.checkpoint().unwrap();
        }
        if compact > 0 && w.seq.is_multiple_of(compact) {
            w.e.compact_storage().unwrap();
        }
        w.tel.maybe(&w.e, &w.db, w.seq, false);
    }
}

// ------------------------------------------------------------ experiments

fn ex_churn(out: &Path, keys: u32, live: u32, budget: usize, ckpt: usize, compact: usize, tag: &str) {
    let mut w = W::new(out, tag);
    churn_mix(&mut w, keys, live, budget, ckpt, compact);
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let s = summarize(&w, serde_json::json!({"keys": keys, "target_live": live, "ckpt_every": ckpt, "compact_every": compact}));
    w.tel.flush(&s);
    println!("e9 {tag}: ops={} ins={} del={} upd={} live={} rss_peak={}KB", w.seq, w.n_ins, w.n_del, w.n_upd, w.live.len(), s["rss_peak_kb"]);
}

fn ex_readonly(out: &Path) {
    let mut w = W::new(out, "e9c-readonly");
    for i in 0..2_000u32 {
        w.insert(i);
    }
    w.e.checkpoint().unwrap();
    let rss_after_load = read_status().0;
    let q = vec_for(9999);
    let mut gets = 0usize;
    let mut scans = 0usize;
    for i in 0..40_000usize {
        let idx = (i % 2_000) as u32;
        if i % 10 == 0 {
            let _ = w.e.scan_filtered("bench", None, 10_000).unwrap().len();
            scans += 1;
        } else {
            let _ = w.e.attend("bench", &["h".to_string()], &q, 10).unwrap();
        }
        if i % 3 == 0 {
            let ver = w.live[&idx];
            let _ = w.e.document_store.read().get(&uuid_for(idx, ver)).is_some();
            gets += 1;
        }
        w.seq += 1;
        w.tel.maybe(&w.e, &w.db, w.seq, false);
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let mut s = summarize(&w, serde_json::json!({"loaded": 2000, "gets": gets, "scans": scans, "rss_after_load_kb": rss_after_load}));
    s["rss_after_load_kb"] = serde_json::json!(rss_after_load);
    w.tel.flush(&s);
    println!("e9 e9c-readonly: ops={} rss_load={rss_after_load} rss_last={}KB", w.seq, s["rss_last_kb"]);
}

fn ex_updchurn(out: &Path) {
    let mut w = W::new(out, "e9d-update-churn");
    for i in 0..10_000u32 {
        w.insert(i);
    }
    w.e.checkpoint().unwrap();
    let rss_after_load = read_status().0;
    let n = w.live.len() as u32;
    for i in 0..100_000usize {
        w.seq += 1;
        let idx = (i % n as usize) as u32;
        w.update(idx);
        if w.seq.is_multiple_of(10_000) {
            w.e.checkpoint().unwrap();
        }
        if w.seq.is_multiple_of(30_000) {
            w.e.compact_storage().unwrap();
        }
        w.tel.maybe(&w.e, &w.db, w.seq, false);
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let s = summarize(&w, serde_json::json!({"loaded": 10_000, "rss_after_load_kb": rss_after_load, "fixed_cardinality": true}));
    w.tel.flush(&s);
    println!("e9 e9d-update-churn: ops={} rss_load={rss_after_load} rss_last={}KB", w.seq, s["rss_last_kb"]);
}

fn ex_growing(out: &Path) {
    let mut w = W::new(out, "e9e-growing");
    for i in 0..80_000u32 {
        w.seq += 1;
        w.insert(i);
        if w.seq.is_multiple_of(10_000) {
            w.e.checkpoint().unwrap();
        }
        if w.seq.is_multiple_of(30_000) {
            w.e.compact_storage().unwrap();
        }
        w.tel.maybe(&w.e, &w.db, w.seq, false);
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let s = summarize(&w, serde_json::json!({"growing": true}));
    w.tel.flush(&s);
    println!("e9 e9e-growing: docs={} rss_last={}KB", w.seq, s["rss_last_kb"]);
}

fn ex_idzchurn(out: &Path) {
    let mut w = W::new(out, "e9f-insert-delete-churn");
    for i in 0..1_000u32 {
        w.insert(i);
    }
    let mut h: u64 = 0xDEADBEEFCAFEF00D;
    let mut rnd = move || {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        h
    };
    for _ in 0..60_000usize {
        w.seq += 1;
        let idx = (rnd() % 1_000) as u32;
        if w.live.contains_key(&idx) {
            w.delete(idx);
        } else {
            w.insert(idx);
        }
        w.tel.maybe(&w.e, &w.db, w.seq, false);
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let s = summarize(&w, serde_json::json!({"keys": 1000, "no_maintenance": true}));
    w.tel.flush(&s);
    println!("e9 e9f-idz-churn: ops={} rss_last={}KB sst_files={}", w.seq, s["rss_last_kb"], {
        let (_, _, _, sstn, _) = dir_census(&w.db);
        sstn
    });
}

fn ex_compactheavy(out: &Path) {
    let mut w = W::new(out, "e9h-compaction-heavy");
    for i in 0..40_000u32 {
        w.seq += 1;
        w.insert(i);
        if w.seq.is_multiple_of(5_000) {
            w.tel.maybe(&w.e, &w.db, w.seq, false);
        }
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let before = read_status().0;
    let mut compaction_trace = Vec::new();
    for c in 0..10 {
        let pre = read_status().0;
        w.e.compact_storage().unwrap();
        let post = read_status().0;
        compaction_trace.push(serde_json::json!({"c": c, "rss_pre_kb": pre, "rss_post_kb": post}));
        w.tel.maybe(&w.e, &w.db, w.seq, false);
    }
    let after = read_status().0;
    let mut s = summarize(&w, serde_json::json!({
        "rss_before_compaction_kb": before, "rss_after_compactions_kb": after,
        "compaction_trace": compaction_trace,
    }));
    // restart: assignment drops the old engine exactly once
    w.e = open_db(&w.db);
    let after_restart = read_status().0;
    let c2 = w.e.mem_census();
    s["rss_after_restart_kb"] = serde_json::json!(after_restart);
    s["census_after_restart"] = serde_json::json!({
        "flushed": c2.store_flushed_records,
        "vstore": c2.collections.iter().map(|c| c.vector_store_len).sum::<usize>()
    });
    w.tel.flush(&s);
    println!("e9 e9h-compact-heavy: before={before} after={after} after_restart={after_restart} flushed_after_restart={}", c2.store_flushed_records);
}

fn ex_walrot(out: &Path) {
    std::env::set_var("ATTENTIONDB_WAL_SEGMENT_BYTES", "2048");
    let mut w = W::new(out, "e9i-wal-rotation");
    for i in 0..2_000u32 {
        w.insert(i);
    }
    let n = 2_000u32;
    for i in 0..30_000usize {
        w.seq += 1;
        w.update((i % n as usize) as u32);
        if w.seq.is_multiple_of(6_000) {
            w.e.checkpoint().unwrap();
        }
        w.tel.maybe(&w.e, &w.db, w.seq, false);
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    std::env::remove_var("ATTENTIONDB_WAL_SEGMENT_BYTES");
    let s = summarize(&w, serde_json::json!({"segment_bytes": 2048}));
    w.tel.flush(&s);
    println!("e9 e9i-wal-rotation: ops={} rss_last={}KB", w.seq, s["rss_last_kb"]);
}

fn ex_ckptheavy(out: &Path) {
    let mut w = W::new(out, "e9j-checkpoint-heavy");
    for i in 0..5_000u32 {
        w.insert(i);
    }
    w.tel.maybe(&w.e, &w.db, 0, true);
    let before = read_status().0;
    for c in 0..150 {
        w.e.checkpoint().unwrap();
        w.seq += 1;
        if c % 10 == 0 {
            w.tel.maybe(&w.e, &w.db, w.seq, false);
        }
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let after = read_status().0;
    std::thread::sleep(std::time::Duration::from_secs(10));
    let after_idle = read_status().0;
    let s = summarize(&w, serde_json::json!({"checkpoints": 150, "rss_before_kb": before, "rss_after_kb": after, "rss_after_idle_kb": after_idle}));
    w.tel.flush(&s);
    println!("e9 e9j-ckpt-heavy: before={before} after={after} idle={after_idle}");
}

fn ex_backupheavy(out: &Path) {
    let mut w = W::new(out, "e9k-backup-heavy");
    for i in 0..5_000u32 {
        w.insert(i);
    }
    w.tel.maybe(&w.e, &w.db, 0, true);
    let before = read_status().0;
    for c in 0..25 {
        let bdir = out.join(format!("bk-{c}"));
        attentiondb_core::backup::copy_database_dir(&w.db, &bdir).unwrap();
        w.seq += 1;
        if c >= 3 {
            let _ = std::fs::remove_dir_all(&bdir);
        }
        if c % 5 == 0 {
            w.tel.maybe(&w.e, &w.db, w.seq, false);
        }
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let after = read_status().0;
    let s = summarize(&w, serde_json::json!({"backups": 25, "rss_before_kb": before, "rss_after_kb": after}));
    w.tel.flush(&s);
    println!("e9 e9k-backup-heavy: before={before} after={after}");
}

fn ex_txnheavy(out: &Path) {
    use attentiondb_core::transaction::TxnOp;
    let mut w = W::new(out, "e9l-transaction-heavy");
    for i in 0..2_000u32 {
        w.insert(i);
    }
    w.e.checkpoint().unwrap();
    for i in 0..15_000usize {
        w.seq += 1;
        let idx = (i % 2_000) as u32;
        let ver = w.live[&idx];
        let t = w.e.begin_transaction("bench");
        let mut r = doc_record(idx, ver, "e9", 100 + (i % 900) as i64);
        r.id = uuid_for(idx, ver);
        w.e.record_transaction_operation(t, TxnOp::Insert(r)).unwrap();
        let r2 = doc_record(idx + 5_000, 1, "e9", i as i64);
        w.e.record_transaction_operation(t, TxnOp::Insert(r2)).unwrap();
        let committed = w.e.commit_transaction(t).unwrap_or(false);
        if committed {
            // r2 added a doc outside live-tracking; delete it again immediately
            let _ = w.e.delete_document("bench", &uuid_for(idx + 5_000, 1).to_string());
        }
        if i % 5 == 0 {
            // rollback-heavy component
            let t = w.e.begin_transaction("bench");
            let r3 = doc_record(idx + 6_000, 1, "e9", i as i64);
            w.e.record_transaction_operation(t, TxnOp::Insert(r3)).unwrap();
            let _ = w.e.rollback_transaction(t);
        }
        if w.seq.is_multiple_of(2_000) {
            w.tel.maybe(&w.e, &w.db, w.seq, false);
        }
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let c = w.e.mem_census();
    let s = summarize(&w, serde_json::json!({"txn_staged_end": c.txn_staged, "staged_ops_end": c.txn_staged_ops}));
    w.tel.flush(&s);
    println!("e9 e9l-txn-heavy: ops={} rss_last={}KB staged_end={}", w.seq, s["rss_last_kb"], c.txn_staged);
}

fn ex_queryiso(out: &Path) {
    let mut w = W::new(out, "e9m-query-path-isolation");
    for i in 0..2_000u32 {
        w.insert(i);
    }
    w.e.checkpoint().unwrap();
    let q = vec_for(7777);
    let phases = ["attend", "get", "scan"];
    let mut phase_rss = Vec::new();
    for ph in phases {
        let _pre = read_status().0;
        for i in 0..15_000usize {
            w.seq += 1;
            match ph {
                "attend" => {
                    let _ = w.e.attend("bench", &["h".to_string()], &q, 10).unwrap();
                }
                "get" => {
                    let idx = (i % 2_000) as u32;
                    let ver = w.live[&idx];
                    let _ = w.e.document_store.read().get(&uuid_for(idx, ver)).is_some();
                }
                _ => {
                    let _ = w.e.scan_filtered("bench", None, 10_000).unwrap().len();
                }
            }
            w.tel.maybe(&w.e, &w.db, w.seq, false);
        }
        phase_rss.push(serde_json::json!({"phase": ph, "rss_after_kb": read_status().0}));
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let s = summarize(&w, serde_json::json!({"phases": phase_rss}));
    w.tel.flush(&s);
    println!("e9 e9m-query-iso: ops={} rss_last={}KB", w.seq, s["rss_last_kb"]);
}

fn ex_mapper(out: &Path) {
    let mut w = W::new(out, "e9n-id-mapper-churn");
    for i in 0..500u32 {
        w.insert(i);
    }
    let mut h: u64 = 0x0DDB1A5E5BAD5EED;
    let mut rnd = move || {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        h
    };
    for _ in 0..40_000usize {
        w.seq += 1;
        let idx = (rnd() % 500) as u32;
        if w.live.contains_key(&idx) {
            w.delete(idx);
        } else {
            w.insert(idx);
        }
        if w.seq.is_multiple_of(2_000) {
            w.tel.maybe(&w.e, &w.db, w.seq, false);
        }
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let before = w.e.mem_census();
    let rss_before = read_status().0;
    let mut s = summarize(&w, serde_json::json!({}));
    // restart via assignment (single live engine at all times)
    w.e = open_db(&w.db);
    let after = w.e.mem_census();
    let rss_after = read_status().0;
    s["mapper_retired_before_restart"] = serde_json::json!(before.mapper_retired);
    s["mapper_retired_after_restart"] = serde_json::json!(after.mapper_retired);
    s["mapper_u2i_after_restart"] = serde_json::json!(after.mapper_uuid_to_u64);
    s["rss_before_restart_kb"] = serde_json::json!(rss_before);
    s["rss_after_restart_kb"] = serde_json::json!(rss_after);
    w.tel.flush(&s);
    println!("e9 e9n-mapper: retired before={} after_restart={} rss {}->{}KB", before.mapper_retired, after.mapper_retired, rss_before, rss_after);
}

fn ex_accounting(out: &Path) {
    let mut w = W::new(out, "e9op-hnsw-store-accounting");
    let mut rungs = Vec::new();
    let mut idx: u32 = 0;
    for docs in [1_000usize, 2_000, 5_000, 10_000] {
        while (idx as usize) < docs {
            w.seq += 1;
            w.insert(idx);
            idx += 1;
        }
        w.e.checkpoint().unwrap();
        let c = w.e.mem_census();
        let rss = read_status().0;
        rungs.push(serde_json::json!({
            "docs": docs, "rss_kb": rss,
            "flushed_records": c.store_flushed_records,
            "vector_store_len": c.collections.iter().map(|x| x.vector_store_len).sum::<usize>(),
            "mapper_u2i": c.mapper_uuid_to_u64, "mapper_retired": c.mapper_retired,
            "bm25_postings": c.collections.iter().map(|x| x.bm25_postings).sum::<usize>(),
            "block_cache_entries": c.store_block_cache_entries,
            "sst_readers": c.store_sst_readers,
        }));
        w.tel.maybe(&w.e, &w.db, docs, true);
    }
    let mut s = summarize(&w, serde_json::json!({"rungs": rungs}));
    w.e = open_db(&w.db);
    let c2 = w.e.mem_census();
    let rss_after_restart = read_status().0;
    s["rss_after_restart_kb"] = serde_json::json!(rss_after_restart);
    s["after_restart"] = serde_json::json!({
        "flushed_records": c2.store_flushed_records,
        "vector_store_len": c2.collections.iter().map(|x| x.vector_store_len).sum::<usize>(),
        "mapper_u2i": c2.mapper_uuid_to_u64, "mapper_retired": c2.mapper_retired,
        "bm25_postings": c2.collections.iter().map(|x| x.bm25_postings).sum::<usize>(),
    });
    w.tel.flush(&s);
    println!("e9 e9op-accounting: rss_after_restart={rss_after_restart} flushed_after_restart={}", c2.store_flushed_records);
}

fn ex_restartreset(out: &Path) {
    let mut w = W::new(out, "e9q-restart-reset");
    for i in 0..2_000u32 {
        w.insert(i);
    }
    let rss_load = read_status().0;
    churn_mix(&mut w, 2_000, 1_500, 70_000, 10_000, 30_000);
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let n = w.tel.samples.len();
    let (rss_a, slope_a) = {
        let (a, b) = (w.tel.samples[n / 2], w.tel.samples[n - 1]);
        let slope = if b[1] > a[1] { (b[2] - a[2]) / ((b[1] - a[1]) / 1000.0) } else { 0.0 };
        (b[2], slope)
    };
    let c_before = w.e.mem_census();
    // restart: assignment drops the old engine exactly once (never two live
    // engines on one dir — E8d lesson)
    let seq_a = w.seq;
    w.e = open_db(&w.db);
    let rss_restart = read_status().0;
    let c_after = w.e.mem_census();
    // phase B: update-only churn using the REAL uuids from a live scan
    let all = w.e.scan_filtered("bench", None, 100_000).unwrap();
    let uuids: Vec<String> = all.into_iter().map(|(id, _v)| id).collect();
    let phase_b_start_rss_kb = read_status().0;
    let mut h: u64 = 0xA5A5A5A5A5A5A5A5;
    let mut rnd = move || {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        h
    };
    for i in 0..15_000usize {
        w.seq += 1;
        let uid = &uuids[(rnd() as usize) % uuids.len()];
        let r = doc_record(0, 99, "e9", 100 + (i % 900) as i64);
        let _ = w.e.update_document("bench", uid, r.fields.clone(), r.k_vecs.clone());
        if (w.seq - seq_a).is_multiple_of(5_000) {
            w.e.checkpoint().unwrap();
        }
        w.tel.maybe(&w.e, &w.db, w.seq, false);
    }
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let _m = w.tel.samples.len();
    let slope_b = {
        // samples after the restart marker (first sample with op > seq_a)
        let post: Vec<[f64; 8]> = w.tel.samples.iter().copied().filter(|s| s[1] > seq_a as f64).collect();
        if post.len() >= 3 {
            let (a, b) = (post[0], post[post.len() - 1]);
            if b[1] > a[1] {
                (b[2] - a[2]) / ((b[1] - a[1]) / 1000.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    };
    let rss_b_end = read_status().0;
    let s = summarize(&w, serde_json::json!({
        "rss_after_load_kb": rss_load, "rss_phase_a_end_kb": rss_a, "slope_a_kb_per_1kops": slope_a,
        "rss_after_restart_kb": rss_restart,
        "flushed_before_restart": c_before.store_flushed_records,
        "flushed_after_restart": c_after.store_flushed_records,
        "vstore_before_restart": c_before.collections.iter().map(|x| x.vector_store_len).sum::<usize>(),
        "vstore_after_restart": c_after.collections.iter().map(|x| x.vector_store_len).sum::<usize>(),
        "mapper_retired_before_restart": c_before.mapper_retired,
        "mapper_retired_after_restart": c_after.mapper_retired,
        "phase_b_start_rss_kb": phase_b_start_rss_kb, "phase_b_end_rss_kb": rss_b_end,
        "slope_b_kb_per_1kops": slope_b,
    }));
    w.tel.flush(&s);
    println!("e9 e9q-restart-reset: load={rss_load} phaseA_end={rss_a} (slope {slope_a:.0}KB/1kops) restart={rss_restart} phaseB slope={slope_b:.0}KB/1kops flushed {}->{} vstore {}->{} retired {}->{}",
        c_before.store_flushed_records, c_after.store_flushed_records,
        c_before.collections.iter().map(|x| x.vector_store_len).sum::<usize>(),
        c_after.collections.iter().map(|x| x.vector_store_len).sum::<usize>(),
        c_before.mapper_retired, c_after.mapper_retired);
}

/// Dispatch: `dbtest e9run --exp NAME --out DIR`.
pub fn run_e9(exp: &str, out: &str) -> String {
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    match exp {
        "repro" => ex_churn(&out, 800, 600, 150_000, 10_000, 30_000, "e9a-repro-h"),
        "readonly" => ex_readonly(&out),
        "updchurn" => ex_updchurn(&out),
        "growing" => ex_growing(&out),
        "idzchurn" => ex_idzchurn(&out),
        "nocompact" => ex_churn(&out, 2_000, 1_500, 60_000, 10_000, 0, "e9g-no-compaction"),
        "compactheavy" => ex_compactheavy(&out),
        "walrot" => ex_walrot(&out),
        "ckptheavy" => ex_ckptheavy(&out),
        "backupheavy" => ex_backupheavy(&out),
        "txnheavy" => ex_txnheavy(&out),
        "queryiso" => ex_queryiso(&out),
        "mapper" => ex_mapper(&out),
        "accounting" => ex_accounting(&out),
        "restartreset" => ex_restartreset(&out),
        "idle" => ex_idle(&out),
        other => panic!("unknown e9 experiment {other}"),
    }
    format!("e9 {exp} done")
}

fn ex_idle(out: &Path) {
    let mut w = W::new(out, "e9r-idle-decay");
    churn_mix(&mut w, 1_000, 800, 40_000, 10_000, 20_000);
    w.tel.maybe(&w.e, &w.db, w.seq, true);
    let churn_end = read_status().0;
    let t_end = std::time::Instant::now();
    let mut idle_rows = Vec::new();
    while t_end.elapsed().as_secs() < 40 {
        std::thread::sleep(std::time::Duration::from_secs(2));
        let (rss, _, anon, file, _) = read_status();
        let mi = unsafe { mallinfo2() };
        let (pss, _, _) = read_pss();
        let ps = pss.map(|v| v.to_string()).unwrap_or_else(|| "NA".into());
        idle_rows.push(format!("{},{},{},{},{},{},{}", t_end.elapsed().as_secs(), rss, anon, file, ps, mi.uordblks, mi.fordblks));
        w.tel.maybe(&w.e, &w.db, w.seq, false);
    }
    let _ = std::fs::write(out.join("idle.csv"), format!("t_s,rss_kb,anon_kb,file_kb,pss_kb,uord_kb,ford_kb\n{}", idle_rows.join("\n")));
    let idle_end = read_status().0;
    let s = summarize(&w, serde_json::json!({"churn_end_rss_kb": churn_end, "idle_end_rss_kb": idle_end}));
    w.tel.flush(&s);
    println!("e9 e9r-idle: churn_end={churn_end} idle_end={idle_end} heap_used={} heap_free_retained={}", {
        let mi = unsafe { mallinfo2() };
        mi.uordblks
    }, {
        let mi = unsafe { mallinfo2() };
        mi.fordblks
    });
}

