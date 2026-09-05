//! Phase 2 ablation harness (§23–28, §40–41).
//!
//! Modes (identical query set, identical corpus, fixed seed):
//!   A SingleHead     — one HNSW head, raw generator scores
//!   B FixedFusion    — multi-head union + fixed weights
//!   C LearnedGating  — + query-conditioned head gating
//!   D QKAttention    — + candidate QK attention (identity-init ⇒ ≈ C)
//!   E Full           — + exact rerank of the fused pool
//!
//! Honesty rules (§55): every number below is measured on this machine in
//! this run; nothing is invented. `ablation.md` interprets; this binary only
//! measures. Unfavorable results are reported as-is.

use attentiondb_core::collection::{PipelineStats, RetrievalMode};
use attentiondb_core::AttentionEngine;
use attentiondb_storage::{Durability, Record};
use std::collections::HashMap;
use std::time::Instant;

const DIM: usize = 32;
const N_DOCS: usize = 10_000;
const N_CLUSTERS: usize = 100;
const N_QUERIES: usize = 100;
const TOP_K: usize = 10;
const SEED: u64 = 0xC0FFEE;

/// xorshift64* — deterministic, dependency-free.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn gauss(&mut self) -> f32 {
        // sum of 4 uniforms — cheap, deterministic, ~N(0,~0.577); scaled below
        (self.next_f32() + self.next_f32() + self.next_f32() + self.next_f32()) * 0.866
    }
    fn gauss_vec(&mut self, dim: usize, sigma: f32) -> Vec<f32> {
        (0..dim).map(|_| self.gauss() * sigma).collect()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn l2normalize(v: &mut [f32]) {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

struct Dataset {
    /// cluster centroids (kept for reference; scoring uses true_vecs)
    #[allow(dead_code)]
    centroids: Vec<Vec<f32>>,
    /// doc → cluster
    doc_cluster: Vec<usize>,
    /// TRUE data-generating doc vectors: v_i = normalize(centroid + u_i),
    /// u_i a distinct per-doc direction. Heads observe v_i + N(0, sigma_h).
    /// Ground truth = exact cosine(v_i, query) — well-defined intra-cluster
    /// ordering, unlike the old all-same-centroid tie lottery.
    true_vecs: Vec<Vec<f32>>,
    /// per-head observation noise
    head_sigmas: Vec<f32>,
    queries: Vec<Vec<f32>>,
    /// ground-truth top-10 doc indices per query (exact cosine over true_vecs)
    gt: Vec<Vec<usize>>,
}

fn build_dataset() -> Dataset {
    let mut rng = Rng::new(SEED);
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| {
            let mut c = rng.gauss_vec(DIM, 1.0);
            l2normalize(&mut c);
            c
        })
        .collect();
    let mut doc_cluster = Vec::with_capacity(N_DOCS);
    let mut true_vecs = Vec::with_capacity(N_DOCS);
    // head noise levels: head 0 cleanest → single-head A is a STRONG baseline
    let head_sigmas = vec![0.02f32, 0.04, 0.07, 0.12, 0.05, 0.08, 0.11, 0.14];
    let mut queries = Vec::with_capacity(N_QUERIES);
    for i in 0..N_DOCS {
        let c = (rng.next_u64() as usize) % N_CLUSTERS;
        doc_cluster.push(c);
        // distinct per-doc direction: intra-cluster ordering becomes real
        let mut u = rng.gauss_vec(DIM, 1.0);
        l2normalize(&mut u);
        let mut v: Vec<f32> = centroids[c]
            .iter()
            .zip(u.iter())
            .map(|(a, b)| a + 0.35 * b)
            .collect();
        l2normalize(&mut v);
        true_vecs.push(v);
        // the query set doubles as the first N_QUERIES docs' centroids
        if i < N_QUERIES {
            let mut q = centroids[c].clone();
            for x in q.iter_mut() {
                *x += rng.gauss() * 0.1;
            }
            l2normalize(&mut q);
            queries.push(q);
        }
    }
    // ground truth: cosine of query against cluster centroids is degenerate
    // (centroid-level); true GT = rank DOCS by cosine(centroid[cluster(doc)],
    // query) with intra-cluster tie-break by doc index.
    let mut gt = Vec::with_capacity(N_QUERIES);
    for q in &queries {
        let mut scored: Vec<(usize, f32)> =
            (0..N_DOCS).map(|d| (d, cosine(&true_vecs[d], q))).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        gt.push(gt_top(&scored));
    }
    Dataset {
        centroids,
        doc_cluster,
        true_vecs,
        head_sigmas,
        queries,
        gt,
    }
}

fn gt_top(scored: &[(usize, f32)]) -> Vec<usize> {
    scored.iter().take(TOP_K).map(|x| x.0).collect()
}

fn insert_corpus(e: &AttentionEngine, ds: &Dataset) {
    let mut rng = Rng::new(SEED ^ 0xB10B);
    for i in 0..N_DOCS {
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(i));
        fields.insert(
            "cluster".to_string(),
            serde_json::json!(ds.doc_cluster[i] as i64),
        );
        fields.insert(
            "body".to_string(),
            serde_json::json!(format!("cluster {}", ds.doc_cluster[i])),
        );
        let mut r = Record::new(fields);
        for (h, sigma) in ds.head_sigmas.iter().enumerate() {
            let head_name = head_name(h);
            let mut v: Vec<f32> = ds.true_vecs[i]
                .iter()
                .map(|x| x + rng.gauss() * sigma)
                .collect();
            l2normalize(&mut v);
            r.k_vecs.insert(head_name, v);
        }
        e.insert_document("bench", r).unwrap();
    }
}

fn head_name(h: usize) -> String {
    [
        "default", "semantic", "lexical", "graph", "h4", "h5", "h6", "h7",
    ][h]
        .to_string()
}

fn all_heads() -> Vec<String> {
    (0..8).map(head_name).collect()
}

struct ModeResult {
    mode: &'static str,
    recall10: f64,
    ndcg10: f64,
    mrr: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    qps: f64,
    union_avg: f64,
    rerank_avg: f64,
}

fn run_mode(
    e: &AttentionEngine,
    ds: &Dataset,
    mode: RetrievalMode,
    mode_name: &'static str,
    heads: &[String],
) -> ModeResult {
    let mut lat_us: Vec<f64> = Vec::with_capacity(N_QUERIES);
    let (mut recall_sum, mut ndcg_sum, mut mrr_sum) = (0f64, 0f64, 0f64);
    let (mut union_sum, mut rerank_sum) = (0usize, 0usize);
    for (qi, q) in ds.queries.iter().enumerate() {
        let t = Instant::now();
        let (ranked, stats): (Vec<_>, PipelineStats) = e
            .get_collection("bench")
            .unwrap()
            .attend_detailed_with_stats(heads, q, TOP_K, None, Some(mode), None, None, None)
            .unwrap();
        lat_us.push(t.elapsed().as_secs_f64() * 1e6);
        union_sum += stats.union_size;
        rerank_sum += stats.rerank_size;
        // recall@10 / MRR / NDCG@10 against centroid ground truth
        let got: Vec<usize> = ranked
            .iter()
            .filter_map(|r| {
                // get_document_fields is the stringified projection — parse idx
                e.get_document_fields(r.id)
                    .get("idx")
                    .and_then(|v| v.parse::<usize>().ok())
            })
            .collect();
        let gt10 = &ds.gt[qi];
        let hits = got.iter().filter(|g| gt10.contains(g)).count();
        recall_sum += hits as f64 / TOP_K as f64;
        let mrr = got
            .iter()
            .position(|g| gt10.contains(g))
            .map(|p| 1.0 / (p + 1) as f64)
            .unwrap_or(0.0);
        mrr_sum += mrr;
        let dcg: f64 = got
            .iter()
            .enumerate()
            .filter(|(_, g)| gt10.contains(g))
            .map(|(i, _)| 1.0 / ((i + 2) as f64).log2())
            .sum::<f64>();
        let idcg: f64 = (0..TOP_K)
            .map(|i| 1.0 / ((i + 2) as f64).log2())
            .sum::<f64>();
        let ndcg = dcg / idcg;
        ndcg_sum += ndcg;
    }
    lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| lat_us[((q * (lat_us.len() - 1) as f64) as usize).min(lat_us.len() - 1)];
    let n = N_QUERIES as f64;
    ModeResult {
        mode: mode_name,
        recall10: recall_sum / n,
        ndcg10: ndcg_sum / n,
        mrr: mrr_sum / n,
        p50_us: p(0.50),
        p95_us: p(0.95),
        p99_us: p(0.99),
        qps: 1e6 / lat_us.iter().sum::<f64>() * n,
        union_avg: union_sum as f64 / n,
        rerank_avg: rerank_sum as f64 / n,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "benchmarks/phase2".to_string());
    std::fs::create_dir_all(&out_dir).unwrap();
    println!("building corpus: {N_DOCS} docs, dim {DIM}, {N_CLUSTERS} clusters, 4 heads");
    let ds = build_dataset();
    let dir = tempfile_dir();
    let e = AttentionEngine::open_dir(&dir, Durability::Async).unwrap();
    e.create_collection(
        "bench",
        DIM,
        &[
            "default", "semantic", "lexical", "graph", "h4", "h5", "h6", "h7",
        ],
    )
    .unwrap();
    let t_build = Instant::now();
    insert_corpus(&e, &ds);
    let build_secs = t_build.elapsed().as_secs_f64();
    let rss_kb = read_rss_kb();
    println!(
        "corpus built in {build_secs:.1}s, RSS ≈ {} MB",
        rss_kb / 1024
    );

    let heads = all_heads();
    let mut results = Vec::new();
    // warmup passes (not recorded)
    let _ = run_mode(&e, &ds, RetrievalMode::FixedFusion, "warmup", &heads);
    results.push(run_mode(
        &e,
        &ds,
        RetrievalMode::SingleHead,
        "A_single_head",
        &["default".to_string()],
    ));
    // per-head single-head baselines: the honest "more heads?" comparison is
    // against the BEST single head, not the worst (§41 head-diversity rule)
    for (h, name) in heads.iter().enumerate() {
        results.push(run_mode(
            &e,
            &ds,
            RetrievalMode::SingleHead,
            match h {
                0 => "A0_head_default",
                1 => "A1_head_semantic",
                2 => "A2_head_lexical",
                3 => "A3_head_graph",
                4 => "A4_head_h4",
                5 => "A5_head_h5",
                6 => "A6_head_h6",
                _ => "A7_head_h7",
            },
            std::slice::from_ref(name),
        ));
    }
    results.push(run_mode(
        &e,
        &ds,
        RetrievalMode::FixedFusion,
        "B_multi_head_fixed",
        &heads,
    ));
    results.push(run_mode(
        &e,
        &ds,
        RetrievalMode::LearnedGating,
        "C_learned_gating",
        &heads,
    ));
    results.push(run_mode(
        &e,
        &ds,
        RetrievalMode::QKAttention,
        "D_qk_attention",
        &heads,
    ));
    results.push(run_mode(
        &e,
        &ds,
        RetrievalMode::Full,
        "E_full_exact_rerank",
        &heads,
    ));

    // CSV (§28): identical queries per mode, measured in-run
    let mut csv = String::from(
        "mode,recall_at_10,ndcg_at_10,mrr,p50_us,p95_us,p99_us,qps,union_avg,rerank_avg\n",
    );
    for r in &results {
        csv.push_str(&format!(
            "{},{:.4},{:.4},{:.4},{:.0},{:.0},{:.0},{:.0},{:.1},{:.1}\n",
            r.mode,
            r.recall10,
            r.ndcg10,
            r.mrr,
            r.p50_us,
            r.p95_us,
            r.p99_us,
            r.qps,
            r.union_avg,
            r.rerank_avg
        ));
    }
    std::fs::write(format!("{out_dir}/ablation.csv"), csv).unwrap();
    println!("wrote {out_dir}/ablation.csv");
    for r in &results {
        println!(
            "{:>22}  R@10={:.3}  NDCG={:.3}  MRR={:.3}  p50={:6.0}µs  p99={:6.0}µs  QPS={:5.0}  union={:4.0}  rerank={:4.0}",
            r.mode, r.recall10, r.ndcg10, r.mrr, r.p50_us, r.p99_us, r.qps, r.union_avg, r.rerank_avg
        );
    }

    // Head-scaling (§25): 1/2/4 heads on the same corpus + queries.
    let mut scaling_csv =
        String::from("heads,parallel,recall_at_10,p50_us,p95_us,p99_us,qps,union_avg\n");
    for h in [1usize, 2, 4, 8] {
        for parallel in [true, false] {
            let hs = heads[..h].to_vec();
            {
                let coll = e.get_collection("bench").unwrap();
                let mut cfg = coll.retrieval_config.read().clone();
                cfg.parallel = parallel;
                *coll.retrieval_config.write() = cfg;
            }
            let r = run_mode(&e, &ds, RetrievalMode::FixedFusion, "scaling", &hs);
            scaling_csv.push_str(&format!(
                "{},{},{:.4},{:.0},{:.0},{:.0},{:.0},{:.1}\n",
                h, parallel, r.recall10, r.p50_us, r.p95_us, r.p99_us, r.qps, r.union_avg
            ));
            println!(
                "heads={h} parallel={parallel}  R@10={:.3}  p50={:6.0}µs  p99={:6.0}µs",
                r.recall10, r.p50_us, r.p99_us
            );
        }
    }
    std::fs::write(format!("{out_dir}/heads_scaling.csv"), scaling_csv).unwrap();
    println!("wrote {out_dir}/heads_scaling.csv");
    let info = format!(
        "run_timestamp_utc={}\ncorpus={{docs:{N_DOCS}, dim:{DIM}, clusters:{N_CLUSTERS}, heads:8, head_sigmas:[0.02,0.04,0.07,0.12,0.05,0.08,0.11,0.14]}}\nqueries={N_QUERIES}\ntop_k={TOP_K}\nseed={SEED}\ncorpus_build_seconds={build_secs:.1}\ninsert_throughput_docs_per_s={:.0}\npeak_rss_mb={}\nnote=single process, release build, in-run measurement; no numbers invented\n",
        chrono_like_now(),
        N_DOCS as f64 / build_secs,
        rss_kb / 1024,
    );
    std::fs::write(format!("{out_dir}/run_info.txt"), info).unwrap();
    println!("wrote {out_dir}/run_info.txt");
}

fn chrono_like_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map(|s| format!("unix_epoch_seconds:{s}"))
        .unwrap_or_else(|_| "unknown".to_string())
}

fn read_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|x| x.parse().ok()))
        })
        .unwrap_or(0)
}

fn tempfile_dir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("phase2-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}
