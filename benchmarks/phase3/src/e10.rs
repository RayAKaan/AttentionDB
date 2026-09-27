//! E10 scale-envelope harness — document ladder, retrieval/head/dimension
//! scaling, churn at scale, hygiene timings, concurrency, integrated run.
//!
//! Reuses the E9 telemetry instruments (no second system): /proc status +
//! smaps_rollup + dir census + the engine's read-only mem_census, plus
//! per-phase timers and per-query latencies. The reference model lives
//! harness-side (idx → version), is mutated in lock-step, and is NEVER
//! derived from engine state (anti-circular). Budget gates per the frozen
//! E10 spec: M3 pre-flight projection, M4 live guard at 85% of launch
//! MemAvailable.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use attentiondb_core::checker::check_engine;
use attentiondb_core::engine::AttentionEngine;
use attentiondb_storage::Record;

use crate::dbtest::{open_db, uuid_for};
use crate::e9::{dir_census, fd_count, read_pss, read_status};

const HEADS: [&str; 4] = ["h", "h2", "h3", "h4"];

// ------------------------------------------------------------ dataset (M8)

fn fnv1a(seed: u64, x: u64) -> u64 {
    // mix a u64 into the FNV-1a stream
    let mut h = seed;
    for b in x.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn normalized(dim: usize, h: u64) -> Vec<f32> {
    // deterministic pseudo-vector from a hash: unit-normalized
    let mut v = vec![0.0f32; dim];
    let mut s = h;
    for x in v.iter_mut().take(dim) {
        s = fnv1a(0x9E3779B97F4A7C15, s);
        *x = ((s >> 11) % 2001) as f32 - 1000.0;
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    v
}

/// Clustered vector scheme: cluster = idx % 50, centroid from hash, member
/// = normalize(centroid*0.8 + noise*0.2); every 16th idx is a near-duplicate
/// of idx-1 plus epsilon. Independent of the engine (pure hash math).
fn gen_vec(dim: usize, idx: u32) -> Vec<f32> {
    let cluster = (idx % 50) as u64;
    let centroid = normalized(dim, fnv1a(0xC10D5, cluster));
    let noise = normalized(dim, fnv1a(0xD07A, idx as u64));
    if idx.is_multiple_of(16) && idx > 0 {
        let base = gen_vec(dim, idx - 1);
        let mut v = base;
        v[(idx as usize) % dim] += 0.01;
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= n;
        }
        return v;
    }
    let mut v = vec![0.0f32; dim];
    for i in 0..dim {
        v[i] = 0.8 * centroid[i] + 0.2 * noise[i];
    }
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in v.iter_mut() {
        *x /= n;
    }
    v
}

fn title_for(idx: u32) -> String {
    // mixed document lengths: idx mod 7 + 1 words; E10_SLIM=1 selects the
    // documented slim-payload configuration (1-word titles) for the
    // payload-vs-count boundary experiment
    if std::env::var("E10_SLIM").as_deref() == Ok("1") {
        return format!("s{}", idx % 97);
    }
    let words = idx % 7 + 1;
    (0..words)
        .map(|w| {
            format!(
                "w{}-{}",
                w,
                fnv1a(0x717, (idx as u64) * 131 + w as u64) % 9973
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn gen_record(dim: usize, idx: u32, ver: u64, heads: &[String]) -> Record {
    let mut fields = HashMap::new();
    fields.insert("idx".to_string(), serde_json::json!(idx));
    fields.insert("version".to_string(), serde_json::json!(ver));
    fields.insert(
        "cat".to_string(),
        serde_json::json!(format!("c{}", idx % 50)),
    );
    fields.insert("num".to_string(), serde_json::json!((idx % 9973) as i64));
    fields.insert("title".to_string(), serde_json::json!(title_for(idx)));
    let mut r = Record::new(fields);
    r.id = uuid_for(idx, ver);
    let v = gen_vec(dim, idx);
    for h in heads {
        r.k_vecs.insert(h.clone(), v.clone());
    }
    r
}

fn dataset_hash(dim: usize, docs: u32) -> u64 {
    // pass-1 canonical stream hash, computed WITHOUT the engine
    let mut h = 0u64;
    for idx in 0..docs {
        let r = gen_record(dim, idx, 1, &[]);
        let line = format!("{}|{}|{}|{}", idx, r.id, title_for(idx).len(), idx % 9973);
        for b in line.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

// ------------------------------------------------------------ telemetry

fn avail_kb() -> usize {
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                return rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            }
        }
    }
    0
}

struct Tel {
    out: PathBuf,
    t0: std::time::Instant,
    last: f64,
    rows: Vec<String>,
    guard_rss_kb: usize,
    halt: Option<String>,
}

impl Tel {
    fn new(out: &Path) -> Self {
        let guard = (avail_kb() as f64 * 0.85) as usize;
        Tel {
            out: out.to_path_buf(),
            t0: std::time::Instant::now(),
            last: -10.0,
            rows: vec![],
            guard_rss_kb: guard,
            halt: None,
        }
    }
    fn maybe(&mut self, _e: &AttentionEngine, db: &Path, op: usize, force: bool) {
        let el = self.t0.elapsed().as_secs_f64();
        if !force && el - self.last < 2.0 {
            return;
        }
        self.last = el;
        let (rss, vmz, anon, file, thr) = read_status();
        let (pss, pa, pf) = read_pss();
        let ps = pss.map(|v| v.to_string()).unwrap_or_else(|| "NA".into());
        let pas = pa.map(|v| v.to_string()).unwrap_or_else(|| "NA".into());
        let pfs = pf.map(|v| v.to_string()).unwrap_or_else(|| "NA".into());
        let fds = fd_count();
        let (walb, waln, sstb, sstn, dbb) = dir_census(db);
        let av = avail_kb();
        self.rows.push(format!(
            "{el:.1},{op},{rss},{vmz},{anon},{file},{ps},{pas},{pfs},{thr},{fds},{av},{walb},{waln},{sstb},{sstn},{dbb}"
        ));
        if rss > self.guard_rss_kb && self.halt.is_none() {
            self.halt = Some(format!(
                "live RSS {rss} KB exceeded 85% of launch MemAvailable ({} KB) at op {op}",
                self.guard_rss_kb
            ));
        }
        // incremental flush so OOM-killed runs still preserve telemetry
        if self.rows.len().is_multiple_of(5) {
            let _ = std::fs::write(
                self.out.join("telemetry.csv"),
                format!("t_s,op,rss_kb,vmz_kb,anon_kb,file_kb,pss_kb,pss_anon_kb,pss_file_kb,threads,fds,avail_kb,wal_bytes,wal_files,sst_bytes,sst_files,db_bytes\n{}", self.rows.join("\n")),
            );
        }
    }
    fn flush(&self, extra_header: &str) {
        let _ = std::fs::write(
            self.out.join("telemetry.csv"),
            format!("{extra_header}\n{}", self.rows.join("\n")),
        );
    }
}

// ------------------------------------------------------------ model

#[derive(Default)]
struct Model {
    live: HashMap<u32, u64>, // idx -> version
}

impl Model {
    #[allow(dead_code)] // retained: canonical state hash available for deeper audits
    fn state_hash(&self) -> u64 {
        let mut keys: Vec<u32> = self.live.keys().copied().collect();
        keys.sort_unstable();
        let mut h = 0u64;
        for k in keys {
            for b in k.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for b in self.live[&k].to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }
}

struct Ctx {
    e: std::sync::Arc<AttentionEngine>,
    db: PathBuf,
    #[allow(dead_code)] // kept for artifact symmetry with e9 harness
    out: PathBuf,
    tel: Tel,
    model: Model,
    dim: usize,
    heads: Vec<String>,
    seq: usize,
}

impl Ctx {
    fn new(out: &Path, exp: &str, dim: usize, heads: usize) -> Self {
        let db = out.join("db");
        std::fs::create_dir_all(&db).unwrap();
        let cfg = serde_json::json!({
            "experiment": exp, "dim": dim, "heads": heads,
            "durability": "sync", "hnsw": "HNSWConfig::default() (max_nb_connection=16, ef_construction=400, ef_search=64, store_vectors=true, max_elements=100000)",
            "hygiene": "E9 policy unchanged", "commit": "3e65d50f (E9 tree)",
            "dataset_scheme": "clustered-50 + near-dup-16 + mixed-length payloads (spec M8)",
        });
        let _ = std::fs::write(out.join("config.json"), cfg.to_string());
        let e = open_db(&db);
        let hnames: Vec<String> = HEADS[..heads].iter().map(|s| s.to_string()).collect();
        let hrefs: Vec<&str> = hnames.iter().map(|s| s.as_str()).collect();
        e.create_collection("bench", dim, &hrefs).unwrap();
        Ctx {
            e: std::sync::Arc::new(e),
            db,
            out: out.to_path_buf(),
            tel: Tel::new(out),
            model: Model::default(),
            dim,
            heads: hnames,
            seq: 0,
        }
    }
    fn insert_new(&mut self, idx: u32) {
        let ver = self.model.live.get(&idx).copied().unwrap_or(0) + 1;
        let r = gen_record(self.dim, idx, ver, &self.heads);
        self.e.insert_document("bench", r).unwrap();
        self.model.live.insert(idx, ver);
    }
}

#[derive(Default)]
struct Verif {
    count_engine: usize,
    count_model: usize,
    sample_ok: usize,
    sample_bad: usize,
    samples: usize,
    checker_clean: bool,
    self_hits: usize,
    self_queries: usize,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    qps: f64,
}

fn verify(ctx: &mut Ctx, n_samples: usize) -> Verif {
    let all = ctx.e.scan_filtered("bench", None, 2_000_000).unwrap();
    let count_engine = all.len();
    let count_model = ctx.model.live.len();
    // sampled exact equality: uuid present AND fields match the model
    let mut idxs: Vec<u32> = ctx.model.live.keys().copied().collect();
    idxs.sort_unstable();
    let stride = (idxs.len() / n_samples).max(1);
    let mut sample_ok = 0;
    let mut sample_bad = 0;
    for &idx in idxs.iter().step_by(stride) {
        let ver = ctx.model.live[&idx];
        let uid = uuid_for(idx, ver);
        match ctx.e.document_store.read().get(&uid) {
            Some(r) => {
                if r.fields.get("idx").and_then(|v| v.as_i64()) == Some(idx as i64) {
                    sample_ok += 1;
                } else {
                    sample_bad += 1;
                }
            }
            None => sample_bad += 1,
        }
    }
    let samples = sample_ok + sample_bad;
    let checker_clean = check_engine(&ctx.e).is_empty();
    // frozen self-query set: 100 live docs at stride; exact numeric-id hit in
    // top-10 (E9-strengthened S18 instrument)
    let mut self_idx: Vec<u32> = (0..100usize)
        .map(|i| idxs[(i * 7919) % idxs.len()])
        .collect();
    self_idx.sort_unstable();
    self_idx.dedup();
    let mut self_hits = 0;
    let mut lat_us: Vec<f64> = Vec::new();
    let mapper = ctx.e.id_mapper.clone();
    for &idx in &self_idx {
        let ver = ctx.model.live[&idx];
        let uid = uuid_for(idx, ver);
        let numeric = mapper.read().uuid_to_id(&uid);
        let q = gen_vec(ctx.dim, idx);
        let t0 = std::time::Instant::now();
        let hits = ctx.e.attend("bench", &ctx.heads, &q, 10).unwrap();
        lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
        if let Some(n) = numeric {
            if hits.iter().any(|h| h.0 == n) {
                self_hits += 1;
            }
        }
    }
    // frozen noise-query latency set: 100 hash-derived vectors
    for i in 0..100u64 {
        let q = normalized(ctx.dim, fnv1a(0x9EA7, i));
        let t0 = std::time::Instant::now();
        let _ = ctx.e.attend("bench", &ctx.heads, &q, 10).unwrap();
        lat_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |f: f64| -> f64 { lat_us[((lat_us.len() as f64 - 1.0) * f).round() as usize] };
    let total: f64 = lat_us.iter().sum();
    let nq = self_idx.len() + 100;
    Verif {
        count_engine,
        count_model,
        sample_ok,
        sample_bad,
        samples,
        checker_clean,
        self_hits,
        self_queries: self_idx.len(),
        p50_us: p(0.50),
        p95_us: p(0.95),
        p99_us: p(0.99),
        qps: if total > 0.0 {
            nq as f64 / (total / 1e6)
        } else {
            0.0
        },
    }
}

fn census_json(e: &AttentionEngine) -> serde_json::Value {
    let c = e.mem_census();
    serde_json::json!({
        "mapper_u2i": c.mapper_uuid_to_u64, "mapper_retired": c.mapper_retired,
        "flushed": c.store_flushed_records, "memtable": c.store_memtable,
        "sst_readers": c.store_sst_readers,
        "block_cache_entries": c.store_block_cache_entries,
        "vstore": c.collections.iter().map(|x| x.vector_store_len).sum::<usize>(),
        "bm25_postings": c.collections.iter().map(|x| x.bm25_postings).sum::<usize>(),
        "txn_staged": c.txn_staged,
    })
}

fn classify_and_write(ctx: &Tel, out: &Path, summary: &mut serde_json::Value) -> String {
    let mut classification = "VERIFIED".to_string();
    if let Some(reason) = &ctx.halt {
        classification = "OBSERVED_LIMIT".to_string();
        summary["halt_reason"] = serde_json::json!(reason);
    }
    summary["classification"] = serde_json::json!(classification);
    let _ = std::fs::write(out.join("summary.json"), summary.to_string());
    classification
}

// ------------------------------------------------------------ experiments

fn ex_ladder(out: &str, docs: u32, exp: &str) {
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    // M3 pre-flight: projected peak = base 30 MB + per-doc marginal (measured
    // from the previous tier via E10_PER_DOC_KB, default 7.0 KB/doc)
    let per_doc_kb: f64 = std::env::var("E10_PER_DOC_KB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7.0);
    let avail = avail_kb();
    let projected = 30_000.0 + per_doc_kb * docs as f64;
    if projected > 0.80 * avail as f64 {
        let s = serde_json::json!({
            "experiment": exp, "docs": docs, "classification": "OBSERVED_LIMIT",
            "reason": format!("M3 budget gate: projected peak {projected:.0} KB > 80% of available {avail} KB (per-doc marginal {per_doc_kb} KB) — launch forbidden by frozen spec"),
            "dataset_hash": format!("{:016x}", dataset_hash(32, docs)),
            "per_doc_kb_estimate": per_doc_kb, "avail_kb_at_gate": avail,
        });
        let _ = std::fs::write(out.join("summary.json"), s.to_string());
        println!("e10 {exp}: OBSERVED_LIMIT (M3 budget gate) — projected {projected:.0} KB vs 80% of {avail} KB");
        return;
    }
    let ds_hash = dataset_hash(32, docs);
    let mut ctx = Ctx::new(&out, exp, 32, 1);
    // build
    let t_build = std::time::Instant::now();
    for idx in 0..docs {
        ctx.seq += 1;
        ctx.insert_new(idx);
        ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
        if ctx.tel.halt.is_some() {
            break;
        }
    }
    let build_s = t_build.elapsed().as_secs_f64();
    let rss0 = ctx
        .tel
        .rows
        .first()
        .map(|r| {
            r.split(',')
                .nth(2)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0)
        })
        .unwrap_or(0);
    let t_ckpt = std::time::Instant::now();
    if ctx.tel.halt.is_none() {
        ctx.e.checkpoint().unwrap();
    }
    let ckpt_ms = t_ckpt.elapsed().as_millis() as u64;
    let census_post_build = census_json(&ctx.e);
    ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, true);
    let v1 = if ctx.tel.halt.is_none() {
        verify(&mut ctx, 5_000)
    } else {
        Verif::default()
    };
    // reopen
    let t_reopen = std::time::Instant::now();
    if ctx.tel.halt.is_none() {
        ctx.e = std::sync::Arc::new(open_db(&ctx.db));
    }
    let reopen_s = t_reopen.elapsed().as_secs_f64();
    let census_post_reopen = census_json(&ctx.e);
    let v2 = if ctx.tel.halt.is_none() {
        verify(&mut ctx, 5_000)
    } else {
        Verif::default()
    };
    ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, true);
    let peak_kb = ctx
        .tel
        .rows
        .iter()
        .filter_map(|r| r.split(',').nth(2).and_then(|v| v.parse::<usize>().ok()))
        .max()
        .unwrap_or(0);
    let last = ctx.tel.rows.last().cloned().unwrap_or_default();
    let p = |i: usize| -> String { last.split(',').nth(i).unwrap_or("NA").to_string() };
    let (_, _, sstn, _, dbb) = dir_census(&ctx.db);
    let mut s = serde_json::json!({
        "experiment": exp, "docs_requested": docs, "ops": ctx.seq,
        "live_docs": ctx.model.live.len(), "dataset_hash": format!("{ds_hash:016x}"),
        "build_s": build_s, "insert_throughput_dps": if build_s > 0.0 { docs as f64 / build_s } else { 0.0 },
        "checkpoint_ms": ckpt_ms, "reopen_s": reopen_s,
        "rss_first_kb": rss0, "rss_peak_kb": peak_kb,
        "pss_last_kb": p(6), "anon_last_kb": p(3), "file_last_kb": p(4),
        "fds_last": p(10), "threads_last": p(9), "avail_last_kb": p(11),
        "sst_files": sstn, "db_bytes": dbb,
        "census_post_build": census_post_build, "census_post_reopen": census_post_reopen,
        "verify_post_build": v1j(&v1), "verify_post_reopen": v2j(&v2),
        "per_doc_marginal_kb": (peak_kb as f64 - rss0 as f64) / docs as f64,
    });
    let cls = classify_and_write(&ctx.tel, &out, &mut s);
    ctx.tel.flush("t_s,op,rss_kb,vmz_kb,anon_kb,file_kb,pss_kb,pss_anon_kb,pss_file_kb,threads,fds,avail_kb,wal_bytes,wal_files,sst_bytes,sst_files,db_bytes");
    println!(
        "e10 {exp}: {cls} ops={} build={build_s:.1}s ckpt={ckpt_ms}ms reopen={reopen_s:.1}s peak={peak_kb}KB selfhit={}/{} p50={:.0}us",
        ctx.seq, v2.self_hits, v2.self_queries, v2.p50_us
    );
}

fn v1j(v: &Verif) -> serde_json::Value {
    serde_json::json!({
        "count_engine": v.count_engine, "count_model": v.count_model,
        "samples": v.samples, "sample_ok": v.sample_ok, "sample_bad": v.sample_bad,
        "checker_clean": v.checker_clean, "self_hits": v.self_hits,
        "self_queries": v.self_queries, "p50_us": v.p50_us, "p95_us": v.p95_us,
        "p99_us": v.p99_us, "qps": v.qps,
    })
}

fn v2j(v: &Verif) -> serde_json::Value {
    v1j(v)
}

fn ex_retrieval(out: &str, docs: u32) {
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let mut ctx = Ctx::new(&out, "e10-retrieval", 32, 1);
    let t_build = std::time::Instant::now();
    for idx in 0..docs {
        ctx.seq += 1;
        ctx.insert_new(idx);
        ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
    }
    ctx.e.checkpoint().unwrap();
    let build_s = t_build.elapsed().as_secs_f64();
    // 1,000-query batch (100 self + 900 noise), per-query latency
    let idxs: Vec<u32> = (0..docs)
        .step_by((docs / 100).max(1) as usize)
        .take(100)
        .collect();
    let mut self_hits = 0;
    let mut lat: Vec<f64> = Vec::new();
    for &idx in &idxs {
        let ver = ctx.model.live[&idx];
        let numeric = ctx.e.id_mapper.read().uuid_to_id(&uuid_for(idx, ver));
        let q = gen_vec(32, idx);
        let t0 = std::time::Instant::now();
        let hits = ctx.e.attend("bench", &ctx.heads, &q, 10).unwrap();
        lat.push(t0.elapsed().as_secs_f64() * 1e6);
        if let Some(n) = numeric {
            if hits.iter().any(|h| h.0 == n) {
                self_hits += 1;
            }
        }
    }
    for i in 0..900u64 {
        let q = normalized(32, fnv1a(0xBEEF, i));
        let t0 = std::time::Instant::now();
        let _ = ctx.e.attend("bench", &ctx.heads, &q, 10).unwrap();
        lat.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |f: f64| lat[((lat.len() as f64 - 1.0) * f).round() as usize];
    let total: f64 = lat.iter().sum();
    let qps = 1000.0 / (total / 1e6);
    ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, true);
    let mut s = serde_json::json!({
        "experiment": "e10-retrieval", "docs": docs, "ops": ctx.seq,
        "build_s": build_s, "queries": 1000, "self_hits": self_hits, "self_queries": idxs.len(),
        "p50_us": p(0.50), "p95_us": p(0.95), "p99_us": p(0.99), "qps": qps,
    });
    // honest classification: the M4 live guard fires when the build crossed
    // 85% of launch MemAvailable (80k-class builds do on this host)
    if ctx.tel.halt.is_some() {
        s["classification"] = serde_json::json!("OBSERVED_LIMIT");
        s["halt_reason"] = serde_json::json!(ctx.tel.halt.clone());
    } else if (self_hits as f64) < 0.95 * idxs.len() as f64 {
        // M6(e) retrieval gate: grown-graph recall below 95% at k=10 is a
        // reliability boundary (SCALE-001 showed reopen-rebuild restores it)
        s["classification"] = serde_json::json!("OBSERVED_LIMIT");
        s["halt_reason"] = serde_json::json!(format!(
            "retrieval self-hit {self_hits}/{} below the 95% M6(e) gate on the grown graph",
            idxs.len()
        ));
    } else {
        s["classification"] = serde_json::json!("VERIFIED");
    }
    let _ = std::fs::write(out.join("summary.json"), s.to_string());
    ctx.tel.flush("t_s,op,rss_kb,vmz_kb,anon_kb,file_kb,pss_kb,pss_anon_kb,pss_file_kb,threads,fds,avail_kb,wal_bytes,wal_files,sst_bytes,sst_files,db_bytes");
    println!(
        "e10 retrieval@{docs}: selfhit={}/{} p50={:.0}us p95={:.0}us p99={:.0}us qps={qps:.0}",
        self_hits,
        idxs.len(),
        p(0.50),
        p(0.95),
        p(0.99)
    );
}

fn ex_heads(out: &str, heads: usize, docs: u32) {
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let mut ctx = Ctx::new(&out, "e10-heads", 32, heads);
    let t_build = std::time::Instant::now();
    for idx in 0..docs {
        ctx.seq += 1;
        ctx.insert_new(idx);
        ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
    }
    let build_s = t_build.elapsed().as_secs_f64();
    let t_ckpt = std::time::Instant::now();
    ctx.e.checkpoint().unwrap();
    let ckpt_ms = t_ckpt.elapsed().as_millis() as u64;
    let census = census_json(&ctx.e);
    let v = verify(&mut ctx, 2_000);
    ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, true);
    let peak_kb = ctx
        .tel
        .rows
        .iter()
        .filter_map(|r| r.split(',').nth(2).and_then(|x| x.parse::<usize>().ok()))
        .max()
        .unwrap_or(0);
    let s = serde_json::json!({
        "experiment": "e10-heads", "heads": heads, "docs": docs, "ops": ctx.seq,
        "build_s": build_s, "checkpoint_ms": ckpt_ms, "rss_peak_kb": peak_kb,
        "census": census,
        "verify": v1j(&v),
    });
    let _ = std::fs::write(out.join("summary.json"), s.to_string());
    ctx.tel.flush("t_s,op,rss_kb,vmz_kb,anon_kb,file_kb,pss_kb,pss_anon_kb,pss_file_kb,threads,fds,avail_kb,wal_bytes,wal_files,sst_bytes,sst_files,db_bytes");
    println!(
        "e10 heads={heads}@{docs}: build={build_s:.1}s peak={peak_kb}KB selfhit={}/{} p50={:.0}us",
        v.self_hits, v.self_queries, v.p50_us
    );
}

fn ex_dims(out: &str, dim: usize, docs: u32) {
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let mut ctx = Ctx::new(&out, "e10-dims", dim, 1);
    let t_build = std::time::Instant::now();
    for idx in 0..docs {
        ctx.seq += 1;
        ctx.insert_new(idx);
        ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
    }
    let build_s = t_build.elapsed().as_secs_f64();
    let t_ckpt = std::time::Instant::now();
    ctx.e.checkpoint().unwrap();
    let ckpt_ms = t_ckpt.elapsed().as_millis() as u64;
    let census = census_json(&ctx.e);
    let v = verify(&mut ctx, 2_000);
    ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, true);
    let peak_kb = ctx
        .tel
        .rows
        .iter()
        .filter_map(|r| r.split(',').nth(2).and_then(|x| x.parse::<usize>().ok()))
        .max()
        .unwrap_or(0);
    let s = serde_json::json!({
        "experiment": "e10-dims", "dim": dim, "docs": docs, "ops": ctx.seq,
        "build_s": build_s, "checkpoint_ms": ckpt_ms, "rss_peak_kb": peak_kb,
        "census": census, "verify": v1j(&v),
    });
    let _ = std::fs::write(out.join("summary.json"), s.to_string());
    ctx.tel.flush("t_s,op,rss_kb,vmz_kb,anon_kb,file_kb,pss_kb,pss_anon_kb,pss_file_kb,threads,fds,avail_kb,wal_bytes,wal_files,sst_bytes,sst_files,db_bytes");
    println!(
        "e10 dim={dim}@{docs}: build={build_s:.1}s peak={peak_kb}KB selfhit={}/{} p50={:.0}us",
        v.self_hits, v.self_queries, v.p50_us
    );
}

fn ex_churn(out: &str) {
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let mut ctx = Ctx::new(&out, "e10-churn", 32, 1);
    let t_build = std::time::Instant::now();
    for idx in 0..60_000u32 {
        ctx.seq += 1;
        ctx.insert_new(idx);
        if ctx.seq.is_multiple_of(2_000) {
            ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
        }
    }
    let build_s = t_build.elapsed().as_secs_f64();
    ctx.e.checkpoint().unwrap();
    let census_base = census_json(&ctx.e);
    // 200k mixed ops: 45% update, 25% delete, 25% reinsert, 5% upsert-new;
    // hot keys = idx % 1000 < 50 get 10x weight
    let mut h: u64 = 0x5CA1AB1E5CA1AB1E;
    let mut rnd = move || {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        h
    };
    let t_churn = std::time::Instant::now();
    let mut n_upd = 0usize;
    let mut n_del = 0usize;
    let mut n_re = 0usize;
    let mut rebuilds_observed: Vec<(usize, usize)> = Vec::new(); // (op, vstore)
    for _i in 0..200_000usize {
        ctx.seq += 1;
        let r = rnd();
        let hot = r % 10 == 0;
        let pick = if hot {
            (r >> 8) % 50
        } else {
            (r >> 8) % 60_000
        };
        let idx = pick as u32;
        let mode = (r >> 20) % 100;
        if mode < 45 && ctx.model.live.contains_key(&idx) {
            // update (uuid preserved, numeric remapped by engine)
            let ver = ctx.model.live[&idx];
            let rec = gen_record(32, idx, ver, &ctx.heads);
            ctx.e
                .update_document(
                    "bench",
                    &uuid_for(idx, ver).to_string(),
                    rec.fields.clone(),
                    rec.k_vecs.clone(),
                )
                .unwrap();
            n_upd += 1;
        } else if mode < 70 && ctx.model.live.contains_key(&idx) {
            let deleted = ctx
                .e
                .delete_document("bench", &uuid_for(idx, ctx.model.live[&idx]).to_string())
                .unwrap();
            assert!(
                deleted,
                "model/engine divergence: delete of a modeled-live doc returned false"
            );
            ctx.model.live.remove(&idx);
            n_del += 1;
        } else if !ctx.model.live.contains_key(&idx) {
            // insert only when absent (correct upsert semantics: the E10-014
            // defect inserted a SECOND live uuid for a live idx)
            ctx.insert_new(idx);
            n_re += 1;
        } else {
            // upsert on a live doc = update
            let ver = ctx.model.live[&idx];
            let rec = gen_record(32, idx, ver, &ctx.heads);
            ctx.e
                .update_document(
                    "bench",
                    &uuid_for(idx, ver).to_string(),
                    rec.fields.clone(),
                    rec.k_vecs.clone(),
                )
                .unwrap();
            n_upd += 1;
        }
        if ctx.seq.is_multiple_of(20_000) {
            ctx.e.checkpoint().unwrap();
            let c = census_json(&ctx.e);
            rebuilds_observed.push((ctx.seq, c["vstore"].as_u64().unwrap_or(0) as usize));
        }
        ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
        if ctx.tel.halt.is_some() {
            break;
        }
    }
    let churn_s = t_churn.elapsed().as_secs_f64();
    ctx.e.checkpoint().unwrap();
    let census_end = census_json(&ctx.e);
    let v = verify(&mut ctx, 5_000);
    ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, true);
    let peak_kb = ctx
        .tel
        .rows
        .iter()
        .filter_map(|r| r.split(',').nth(2).and_then(|x| x.parse::<usize>().ok()))
        .max()
        .unwrap_or(0);
    let s = serde_json::json!({
        "experiment": "e10-churn", "base_docs": 60_000, "ops": ctx.seq,
        "updates": n_upd, "deletes": n_del, "reinserts": n_re,
        "build_s": build_s, "churn_s": churn_s, "rss_peak_kb": peak_kb,
        "census_base": census_base, "census_end": census_end,
        "vstore_trace": rebuilds_observed, "verify": v1j(&v),
    });
    let _ = std::fs::write(out.join("summary.json"), s.to_string());
    ctx.tel.flush("t_s,op,rss_kb,vmz_kb,anon_kb,file_kb,pss_kb,pss_anon_kb,pss_file_kb,threads,fds,avail_kb,wal_bytes,wal_files,sst_bytes,sst_files,db_bytes");
    println!("e10 churn: ops={} upd={n_upd} del={n_del} re={n_re} peak={peak_kb}KB selfhit={}/{} vstore_end={}", ctx.seq, v.self_hits, v.self_queries, census_end["vstore"]);
}

fn ex_concurrency(out: &str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let mut ctx = Ctx::new(&out, "e10-concurrency", 32, 1);
    for idx in 0..40_000u32 {
        ctx.seq += 1;
        ctx.insert_new(idx);
        if ctx.seq.is_multiple_of(5_000) {
            ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
        }
    }
    ctx.e.checkpoint().unwrap();
    // bounded concurrency: 1 writer (new docs 100k..130k + hot del/reinsert),
    // 1 reader (attend/scan + stability spot-checks), engine Arc shared
    let stop = Arc::new(AtomicBool::new(false));
    let e2 = ctx.e.clone();
    let we = ctx.e.clone();
    let rstop = stop.clone();
    let reader = std::thread::spawn(move || {
        let mut checks = 0usize;
        let mut lat: Vec<f64> = Vec::new();
        let q = normalized(32, 0xA11CE);
        while !rstop.load(Ordering::Relaxed) && checks < 20_000 {
            std::thread::sleep(std::time::Duration::from_millis(2));
            let t0 = std::time::Instant::now();
            let _ = e2.attend("bench", &["h".to_string()], &q, 10).unwrap();
            lat.push(t0.elapsed().as_secs_f64() * 1e6);
            let _ = e2.scan_filtered("bench", None, 5_000).unwrap().len();
            // stability spot: docs 20_000..20_020 (immutable base) stay addressable throughout
            for idx in (20_000..20_020u32).step_by(4) {
                let uid = uuid_for(idx, 1);
                if e2.document_store.read().get(&uid).is_none() {
                    panic!("stability spot-check failed: {uid} missing during concurrent phase");
                }
            }
            checks += 1;
        }
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |f: f64| lat[((lat.len() as f64 - 1.0) * f).round() as usize];
        (checks, p(0.5), p(0.95), p(0.99))
    });
    let t_conc = std::time::Instant::now();
    for i in 0..20_000usize {
        let idx = 40_000u32 + i as u32;
        let ver = 1u64;
        let mut r = gen_record(32, idx, ver, &ctx.heads);
        r.id = uuid_for(idx, ver);
        we.insert_document("bench", r).unwrap();
        ctx.seq += 1;
        if i.is_multiple_of(3_000) {
            we.checkpoint().unwrap();
        }
        if i == 15_000 {
            we.compact_storage().unwrap();
        }
        ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
    }
    let conc_s = t_conc.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let (checks, rp50, rp95, rp99) = reader.join().unwrap();
    // model: add the writer's docs
    for i in 0..20_000usize {
        ctx.model.live.insert(40_000u32 + i as u32, 1);
    }
    // final checkpoint BEFORE reopen (the sealed E8 pattern): skipping it in
    // async durability loses the buffered WAL tail (E8f documented boundary —
    // measured 7 acked writer docs in the pre-fix attempt)
    ctx.e.checkpoint().unwrap();
    ctx.e = std::sync::Arc::new(open_db(&ctx.db));
    let v = verify(&mut ctx, 5_000);
    ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, true);
    let peak_kb = ctx
        .tel
        .rows
        .iter()
        .filter_map(|r| r.split(',').nth(2).and_then(|x| x.parse::<usize>().ok()))
        .max()
        .unwrap_or(0);
    let s = serde_json::json!({
        "experiment": "e10-concurrency", "base_docs": 40_000, "ops": ctx.seq,
        "writer_ops": 20_000, "reader_checks": checks, "conc_s": conc_s,
        "reader_p50_us": rp50, "reader_p95_us": rp95, "reader_p99_us": rp99,
        "rss_peak_kb": peak_kb, "verify_post_restart": v1j(&v),
    });
    let _ = std::fs::write(out.join("summary.json"), s.to_string());
    ctx.tel.flush("t_s,op,rss_kb,vmz_kb,anon_kb,file_kb,pss_kb,pss_anon_kb,pss_file_kb,threads,fds,avail_kb,wal_bytes,wal_files,sst_bytes,sst_files,db_bytes");
    println!("e10 concurrency: ops={} reader_checks={checks} rp50={rp50:.0}us rp99={rp99:.0}us selfhit={}/{}", ctx.seq, v.self_hits, v.self_queries);
}

fn ex_integrated(out: &str, docs: u32) {
    use attentiondb_core::transaction::TxnOp;
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let mut ctx = Ctx::new(&out, "e10-integrated", 32, 1);
    let t_build = std::time::Instant::now();
    for idx in 0..docs {
        ctx.seq += 1;
        ctx.insert_new(idx);
        if ctx.seq.is_multiple_of(5_000) {
            ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
        }
    }
    let build_s = t_build.elapsed().as_secs_f64();
    // churn phase: 20k mixed ops on the top 10k idx range
    let mut h: u64 = 0x1A7E6A7E;
    let mut rnd = move || {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        h
    };
    let mut churn_ops = 0usize;
    for _ in 0..20_000usize {
        ctx.seq += 1;
        churn_ops += 1;
        let idx = docs.saturating_sub(1) - (rnd() % 10_000) as u32;
        let mode = rnd() % 100;
        if mode < 50 && ctx.model.live.contains_key(&idx) {
            let ver = ctx.model.live[&idx];
            let rec = gen_record(32, idx, ver, &ctx.heads);
            ctx.e
                .update_document(
                    "bench",
                    &uuid_for(idx, ver).to_string(),
                    rec.fields.clone(),
                    rec.k_vecs.clone(),
                )
                .unwrap();
        } else if mode < 75 && ctx.model.live.contains_key(&idx) {
            let deleted = ctx
                .e
                .delete_document("bench", &uuid_for(idx, ctx.model.live[&idx]).to_string())
                .unwrap();
            assert!(
                deleted,
                "model/engine divergence: delete of a modeled-live doc returned false"
            );
            ctx.model.live.remove(&idx);
        } else if !ctx.model.live.contains_key(&idx) {
            ctx.insert_new(idx);
        }
        if ctx.seq.is_multiple_of(5_000) {
            ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
        }
    }
    // txn phase: 2,000 committed 2-op txns on fresh uuids
    let mut txn_ok = 0usize;
    for i in 0..2_000usize {
        ctx.seq += 1;
        let idx = docs + 1_000 + i as u32;
        let t = ctx.e.begin_transaction("bench");
        let r1 = gen_record(32, idx, 1, &ctx.heads);
        ctx.e
            .record_transaction_operation(t, TxnOp::Insert(r1))
            .unwrap();
        let r2 = gen_record(32, idx, 2, &ctx.heads);
        ctx.e
            .record_transaction_operation(t, TxnOp::Insert(r2))
            .unwrap();
        if ctx.e.commit_transaction(t).unwrap_or(false) {
            txn_ok += 1;
            // r2 (ver 2) supersedes r1; model keeps idx at the highest ver
            ctx.model.live.insert(idx, 2);
            let _ = ctx
                .e
                .delete_document("bench", &uuid_for(idx, 1).to_string());
        }
        if ctx.seq.is_multiple_of(5_000) {
            ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
        }
    }
    let t_ckpt = std::time::Instant::now();
    ctx.e.checkpoint().unwrap();
    let ckpt_ms = t_ckpt.elapsed().as_millis() as u64;
    let census_pre_hygiene = census_json(&ctx.e);
    let t_compact = std::time::Instant::now();
    ctx.e.compact_storage().unwrap();
    let compact_ms = t_compact.elapsed().as_millis() as u64;
    // backup
    let t_backup = std::time::Instant::now();
    let bdir = out.join("backup-int");
    attentiondb_core::backup::copy_database_dir(&ctx.db, &bdir).unwrap();
    let backup_ms = t_backup.elapsed().as_millis() as u64;
    let _ = std::fs::remove_dir_all(&bdir);
    // restart + verify
    ctx.e = std::sync::Arc::new(open_db(&ctx.db));
    let v = verify(&mut ctx, 5_000);
    ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, true);
    let peak_kb = ctx
        .tel
        .rows
        .iter()
        .filter_map(|r| r.split(',').nth(2).and_then(|x| x.parse::<usize>().ok()))
        .max()
        .unwrap_or(0);
    let s = serde_json::json!({
        "experiment": "e10-integrated", "docs_base": docs, "ops": ctx.seq,
        "build_s": build_s, "churn_ops": churn_ops, "txn_ok": txn_ok,
        "checkpoint_ms": ckpt_ms, "compact_ms": compact_ms, "backup_ms": backup_ms,
        "rss_peak_kb": peak_kb,
        "census_pre_hygiene": census_pre_hygiene, "verify_post_restart": v1j(&v),
    });
    let _ = std::fs::write(out.join("summary.json"), s.to_string());
    ctx.tel.flush("t_s,op,rss_kb,vmz_kb,anon_kb,file_kb,pss_kb,pss_anon_kb,pss_file_kb,threads,fds,avail_kb,wal_bytes,wal_files,sst_bytes,sst_files,db_bytes");
    println!("e10 integrated@{docs}: ops={} txn_ok={txn_ok} ckpt={ckpt_ms}ms compact={compact_ms}ms backup={backup_ms}ms peak={peak_kb}KB selfhit={}/{} p50={:.0}us", ctx.seq, v.self_hits, v.self_queries, v.p50_us);
}

fn ex_maintenance(out: &str, docs: u32) {
    let out = PathBuf::from(out);
    std::fs::create_dir_all(&out).unwrap();
    let mut ctx = Ctx::new(&out, "e10-maintenance", 32, 1);
    let t_build = std::time::Instant::now();
    for idx in 0..docs {
        ctx.seq += 1;
        ctx.insert_new(idx);
        if ctx.seq.is_multiple_of(5_000) {
            ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, false);
        }
    }
    let build_s = t_build.elapsed().as_secs_f64();
    // churn some deletions so compaction has tombstones to reclaim
    for idx in 0..docs / 10 {
        ctx.seq += 1;
        let ver = ctx.model.live[&idx];
        let deleted = ctx
            .e
            .delete_document("bench", &uuid_for(idx, ver).to_string())
            .unwrap();
        assert!(deleted);
        ctx.model.live.remove(&idx);
    }
    let t_ckpt = std::time::Instant::now();
    ctx.e.checkpoint().unwrap();
    let ckpt_ms = t_ckpt.elapsed().as_millis() as u64;
    let (wb, wn, sb, sn, _) = dir_census(&ctx.db);
    let t_compact = std::time::Instant::now();
    ctx.e.compact_storage().unwrap();
    let compact_ms = t_compact.elapsed().as_millis() as u64;
    let (_, _, sa, sna, _) = dir_census(&ctx.db);
    let t_backup = std::time::Instant::now();
    let bdir = out.join("backup-maint");
    attentiondb_core::backup::copy_database_dir(&ctx.db, &bdir).unwrap();
    let backup_ms = t_backup.elapsed().as_millis() as u64;
    let _ = std::fs::remove_dir_all(&bdir);
    let t_reopen = std::time::Instant::now();
    ctx.e = std::sync::Arc::new(open_db(&ctx.db));
    let reopen_s = t_reopen.elapsed().as_secs_f64();
    let v = verify(&mut ctx, 5_000);
    ctx.tel.maybe(&ctx.e, &ctx.db, ctx.seq, true);
    let peak_kb = ctx
        .tel
        .rows
        .iter()
        .filter_map(|r| r.split(',').nth(2).and_then(|x| x.parse::<usize>().ok()))
        .max()
        .unwrap_or(0);
    let s = serde_json::json!({
        "experiment": "e10-maintenance", "docs": docs, "ops": ctx.seq,
        "build_s": build_s, "checkpoint_ms": ckpt_ms,
        "sst_files_pre_compact": sn, "sst_bytes_pre_compact": sb,
        "sst_files_post_compact": sna, "sst_bytes_post_compact": sa,
        "wal_bytes_pre_compact": wb, "wal_files_pre_compact": wn,
        "compact_ms": compact_ms, "backup_ms": backup_ms, "reopen_s": reopen_s,
        "rss_peak_kb": peak_kb, "verify_post_reopen": v1j(&v),
    });
    let _ = std::fs::write(out.join("summary.json"), s.to_string());
    ctx.tel.flush("t_s,op,rss_kb,vmz_kb,anon_kb,file_kb,pss_kb,pss_anon_kb,pss_file_kb,threads,fds,avail_kb,wal_bytes,wal_files,sst_bytes,sst_files,db_bytes");
    println!("e10 maintenance@{docs}: ckpt={ckpt_ms}ms compact={compact_ms}ms (sst {sn}-> {sna}, bytes {sb}-> {sa}) backup={backup_ms}ms reopen={reopen_s:.1}s selfhit={}/{}", v.self_hits, v.self_queries);
}

/// Dispatch: `dbtest e10run --exp NAME --out DIR [--docs N]`.
pub fn run_e10(exp: &str, out: &str, docs: u32) -> String {
    match exp {
        "repro" => ex_ladder(out, 80_000, "e10-repro-80k"),
        "ladder" => ex_ladder(out, docs, &format!("e10-ladder-{docs}")),
        "retrieval" => ex_retrieval(out, docs),
        "heads" => ex_heads(out, 4, 40_000),
        "heads2" => ex_heads(out, 2, 40_000),
        "dims128" => ex_dims(out, 128, 20_000),
        "dims256" => ex_dims(out, 256, 20_000),
        "churn" => ex_churn(out),
        "concurrency" => ex_concurrency(out),
        "integrated" => ex_integrated(out, docs),
        "maintenance" => ex_maintenance(out, docs),
        other => panic!("unknown e10 experiment {other}"),
    }
    format!("e10 {exp} done")
}
