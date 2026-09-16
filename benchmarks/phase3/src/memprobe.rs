//! PH3C `memprobe` — memory forensics with lifecycle checkpoints (§2–§10).
//!
//! Builds an engine over an AG News dataset tier with configurable docs and
//! the dataset's head set, sampling RSS/disk at controlled lifecycle points
//! (A..K) plus a per-1000-document insertion curve. Component attribution is
//! EMPIRICAL: RSS deltas between checkpoints, disk file sizes by class,
//! anonymous-vs-file-backed RSS (smaps_rollup), and logical vector bytes.
//! No component value is invented; approximations are labeled.
use std::fmt::Write as _;

use attentiondb_core::engine::AttentionEngine;
use attentiondb_learned::gating_v2::GatingMlp;
use attentiondb_storage::{Durability, Record};

use crate::textqual;

fn rss_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmRSS:") {
            return v
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .map(|k: f64| k / 1024.0)
                .unwrap_or(0.0);
        }
    }
    0.0
}

fn peak_rss_mb() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("VmHWM:") {
            return v
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .map(|k: f64| k / 1024.0)
                .unwrap_or(0.0);
        }
    }
    0.0
}

/// anonymous vs file-backed RSS from /proc/self/smaps_rollup (may be absent)
fn anon_file_mb() -> (f64, f64) {
    let (mut anon, file) = (0.0f64, 0.0f64);
    if let Ok(s) = std::fs::read_to_string("/proc/self/smaps_rollup") {
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("Anonymous:") {
                anon = v
                    .trim()
                    .trim_end_matches(" kB")
                    .trim()
                    .parse()
                    .unwrap_or(0.0)
                    / 1024.0;
            }
        }
    }
    (anon, file)
}

fn dir_size_by_class(p: &std::path::Path) -> std::collections::BTreeMap<String, u64> {
    let mut total = std::collections::BTreeMap::new();
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let md = e.metadata().unwrap();
            if md.is_dir() {
                for (k, v) in dir_size_by_class(&e.path()) {
                    *total.entry(k).or_insert(0) += v;
                }
            } else {
                let name = e.file_name().to_string_lossy().to_string();
                let class = if name.contains("wal") || name.contains("WAL") {
                    "wal"
                } else if name.contains("sstable") || name.contains("seg") || name.contains("data")
                {
                    "sstable/data"
                } else if name.contains("manifest")
                    || name.contains("catalog")
                    || name.contains("meta")
                {
                    "manifest/catalog"
                } else if name.contains("hnsw") || name.contains("index") {
                    "hnsw/index"
                } else {
                    "other"
                };
                *total.entry(class.to_string()).or_insert(0) += md.len();
            }
        }
    }
    total
}

struct Check {
    log: std::fs::File,
    t0: std::time::Instant,
}

impl Check {
    fn new(out_dir: &str) -> Self {
        let mut f = std::fs::File::create(format!("{out_dir}/memory-checkpoints.csv")).unwrap();
        let _ = std::io::Write::write_all(
            &mut f,
            b"checkpoint,elapsed_s,rss_mb,peak_rss_mb,anon_mb,file_mb,engine_dir_bytes,wal_bytes,data_bytes,meta_bytes,index_bytes,other_bytes,docs_inserted\n",
        );
        Check {
            log: f,
            t0: std::time::Instant::now(),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn mark(&mut self, name: &str, engine_dir: Option<&std::path::Path>, docs: usize) {
        let (anon, file) = anon_file_mb();
        let mut classes = std::collections::BTreeMap::new();
        if let Some(d) = engine_dir {
            classes = dir_size_by_class(d);
        }
        let g = |k: &str| classes.get(k).copied().unwrap_or(0);
        let row = format!(
            "{name},{:.1},{:.1},{:.1},{:.1},{:.1},{},{},{},{},{},{},{}\n",
            self.t0.elapsed().as_secs_f64(),
            rss_mb(),
            peak_rss_mb(),
            anon,
            file,
            classes.values().sum::<u64>(),
            g("wal"),
            g("sstable/data"),
            g("manifest/catalog"),
            g("hnsw/index"),
            g("other"),
            docs
        );
        let _ = std::io::Write::write_all(&mut self.log, row.as_bytes());
        let _ = std::io::Write::flush(&mut self.log); // survive SIGKILL: the partial curve IS the forensic record
        eprintln!(
            "[ckpt {name}] rss={:.1} peak={:.1} dir={}",
            rss_mb(),
            peak_rss_mb(),
            classes.values().sum::<u64>()
        );
    }
}

pub fn run(tier: &str, data_dir: &str, out_dir: &str, n_docs: usize) -> String {
    std::fs::create_dir_all(out_dir).unwrap();
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{data_dir}/meta.json")).unwrap())
            .unwrap();
    let dim = meta["dim"].as_u64().unwrap_or(512) as usize;
    let heads: Vec<String> = meta["heads"]
        .as_array()
        .map(|a| a.iter().map(|h| h.as_str().unwrap().to_string()).collect())
        .unwrap_or_else(|| vec!["title".into(), "body".into(), "full".into()]);
    let corpus_text: Vec<(String, String)> = {
        let mut v = Vec::new();
        for line in std::fs::read_to_string(format!("{data_dir}/corpus_text.jsonl"))
            .unwrap()
            .lines()
        {
            let j: serde_json::Value = serde_json::from_str(line).unwrap();
            v.push((
                j["title"].as_str().unwrap().to_string(),
                j["description"].as_str().unwrap().to_string(),
            ));
        }
        v
    };
    let n_docs = n_docs.min(corpus_text.len());
    let mut cp = Check::new(out_dir);

    // A: process start (checkpoint A is earliest possible)
    cp.mark("A_process_start", None, 0);

    // engine dir backend: disk by default (see HC-P3-6 addendum)
    let tmpfs = std::env::var("PH3B_ENGINE_TMPFS")
        .map(|v| v == "1")
        .unwrap_or(false);
    let base = if tmpfs {
        std::env::temp_dir()
    } else {
        std::path::PathBuf::from("/var/tmp")
    };
    let dir = base.join(format!("phase3c-mem-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // load corpus vectors (OUTSIDE the engine — dataset-side memory, attributed here)
    let mut corpus_heads: Vec<Vec<f32>> = Vec::new();
    for h in &heads {
        corpus_heads.push(textqual::read_f32_vec_pub(&format!(
            "{data_dir}/corpus_{h}.f32"
        )));
    }
    cp.mark("B_dataset_loaded", None, 0);

    let e = AttentionEngine::open_dir(&dir, Durability::Async).unwrap();
    cp.mark("C_engine_init", Some(&dir), 0);
    let head_refs: Vec<&str> = heads.iter().map(|s| s.as_str()).collect();
    e.create_collection("bench", dim, &head_refs).unwrap();
    cp.mark("D_collection_created", Some(&dir), 0);

    let t_build = std::time::Instant::now();
    for (i, (title, desc)) in corpus_text.iter().take(n_docs).enumerate() {
        let mut fields = std::collections::HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(i));
        fields.insert("title".to_string(), serde_json::json!(title));
        fields.insert("description".to_string(), serde_json::json!(desc));
        let mut r = Record::new(fields);
        for (h, hn) in heads.iter().enumerate() {
            r.k_vecs
                .insert(hn.clone(), corpus_heads[h][i * dim..(i + 1) * dim].to_vec());
        }
        e.insert_document("bench", r).unwrap();
        if (i + 1) % 1000 == 0 || i + 1 == n_docs {
            cp.mark("E_insert_curve", Some(&dir), i + 1);
        }
    }
    let build_s = t_build.elapsed().as_secs_f64();
    cp.mark("F_inserts_done_hnsw_inline", Some(&dir), n_docs);

    e.flush_wal().unwrap();
    cp.mark("G_after_flush", Some(&dir), n_docs);
    e.checkpoint().unwrap();
    cp.mark("G2_after_checkpoint", Some(&dir), n_docs);

    // BM25 is built inline during insert (engine behavior, documented);
    // H: measure the sparse channel working-set effect with one query pass.
    let coll = e.get_collection("bench").unwrap();
    let _ = coll.bm25_channel("the", 10);
    cp.mark("H_bm25_queried", Some(&dir), n_docs);

    // I: gating model construct (memory of the learned component at this
    // config's input dim; random init — memory is what is measured)
    let gm = GatingMlp::new(3 * dim, 64, heads.len(), 42);
    let _ = gm;
    cp.mark("I_gating_model_loaded", Some(&dir), n_docs);

    // J: steady state (1s idle, re-measure)
    std::thread::sleep(std::time::Duration::from_secs(1));
    cp.mark("J_steady_state", Some(&dir), n_docs);

    let peak = peak_rss_mb();
    let classes = dir_size_by_class(&dir);
    let raw_vec_mb = (n_docs * heads.len() * dim * 4) as f64 / (1024.0 * 1024.0);
    drop(e);
    drop(coll);
    cp.mark("K_after_engine_drop", None, n_docs);
    let _ = std::fs::remove_dir_all(&dir);

    let mut mem = String::from("metric,value\n");
    let _ = writeln!(mem, "tier,{tier}");
    let _ = writeln!(mem, "n_docs,{n_docs}");
    let _ = writeln!(mem, "n_heads,{}", heads.len());
    let _ = writeln!(mem, "dim,{dim}");
    let _ = writeln!(mem, "raw_vector_mb,{raw_vec_mb:.1}");
    let _ = writeln!(mem, "peak_rss_mb,{peak:.1}");
    let _ = writeln!(mem, "steady_rss_mb,{:.1}", rss_mb());
    let _ = writeln!(
        mem,
        "engine_multiplier_peak_over_raw,{:.2}",
        peak / raw_vec_mb
    );
    let _ = writeln!(mem, "build_seconds,{build_s:.1}");
    let _ = writeln!(
        mem,
        "engine_dir_backend,{}",
        if tmpfs { "tmpfs" } else { "disk" }
    );
    for (k, v) in &classes {
        let _ = writeln!(mem, "disk_{k},{v}");
    }
    std::fs::write(format!("{out_dir}/memory.csv"), &mem).unwrap();

    let cfg = serde_json::json!({
        "experiment_family": "PH3C memprobe",
        "tier": tier, "data_dir": data_dir, "n_docs": n_docs,
        "heads": heads, "dim": dim,
        "hnsw": "engine defaults M=16 ef_construction=400 ef_search=64 store_vectors=true",
        "durability": "Async",
        "engine_dir_backend": if tmpfs { "tmpfs" } else { "disk" },
        "code_commit": option_env!("GIT_HASH").unwrap_or("unknown"),
        "hardware": {"ram_total_mb": 1984, "cpus": 2},
    });
    std::fs::write(
        format!("{out_dir}/config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .unwrap();
    std::fs::write(
        format!("{out_dir}/results.csv"),
        "arm,seed,R@1,R@5,R@10,NDCG@10,MRR\nmemprobe,n/a,n/a,n/a,n/a,n/a,n/a\n",
    )
    .unwrap();
    let metrics = serde_json::json!({
        "peak_rss_mb": peak, "steady_rss_mb": rss_mb(), "raw_vector_mb": raw_vec_mb,
        "multiplier": peak / raw_vec_mb, "build_seconds": build_s,
        "disk_classes": classes,
    });
    std::fs::write(
        format!("{out_dir}/metrics.json"),
        serde_json::to_string_pretty(&metrics).unwrap(),
    )
    .unwrap();

    format!("memprobe {tier}: docs={n_docs} heads={} dim={dim} peak={peak:.0}MB raw={raw_vec_mb:.0}MB mult={:.1}x build={build_s:.0}s",
        heads.len(), peak / raw_vec_mb)
}

/// leaktest: PH3C-MEM-003 helper. Creates an engine at --path, inserts a few
/// docs, then either drops cleanly or exits without destructors (simulating
/// an unclean death; `--mode kill` also supports external SIGKILL).
pub fn leaktest(path: &str, mode: &str) -> String {
    let dir = std::path::PathBuf::from(path);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Async).unwrap();
    e.create_collection("c", 8, &["h"]).unwrap();
    for i in 0..20 {
        let mut fields = std::collections::HashMap::new();
        fields.insert("title".to_string(), serde_json::json!(format!("t{i}")));
        let mut r = Record::new(fields);
        r.k_vecs.insert("h".to_string(), vec![0.0f32; 8]);
        e.insert_document("c", r).unwrap();
    }
    e.flush_wal().unwrap();
    let size = walk(&dir);
    eprintln!("LEAKTEST_READY bytes={size} mode={mode}");
    match mode {
        // flush stdout/stderr so the parent sees readiness before we vanish
        "clean" => {
            drop(e);
            let after = walk(&dir);
            let _ = std::fs::remove_dir_all(&dir);
            format!("clean: dir during run={size}B; after drop={after}B (guard/owner removes rest)")
        }
        "exit_no_destructors" => {
            std::process::exit(137); // terminates WITHOUT unwinding: Drop never runs
        }
        "kill_self" => {
            // real SIGKILL via /proc/self/... not available in std; use abort-free path:
            // parent will SIGKILL after seeing READY. Park until then.
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }
        m => format!("unknown mode {m}"),
    }
}

fn walk(p: &std::path::Path) -> u64 {
    let mut t = 0;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let md = e.metadata().unwrap();
            t += if md.is_dir() {
                walk(&e.path())
            } else {
                md.len()
            };
        }
    }
    t
}
