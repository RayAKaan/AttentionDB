//! Phase 3 benchmark driver — real-world validation of the FROZEN
//! Phase 2 architecture (Phase 3 spec §2–§11).
//!
//! Subcommands:
//!   quality --tier S|M --data DIR --out DIR [--seeds 42,7,1]
//!
//! Frozen retrieval path under test:
//!   per-head ANN → candidate union → trained gating → weighted fusion → top-K
//! (baseline arms isolate single-head ANN, uniform fusion, exact reference.)
//!
//! Dataset: PH3-DS-FM (Fashion-MNIST multi-view; real images; exact
//! full-view-cosine ground truth — see research/phase3/datasets/).

use attentiondb_core::engine::AttentionEngine;
use attentiondb_learned::eval::{fuse_weighted, head_ranking, rank_metrics, RankMetrics};
use attentiondb_learned::gating_v2::{
    train_gating, DetRng, GatingDataset, GatingMlp, HeadExample, ModelCard, Objective,
    QualityTarget, QueryExample, Split, TrainOutcome, TrainingConfig, TrainingMeta,
};
use attentiondb_storage::{Durability, Record};

mod headsqual;
mod memprobe;
mod textqual;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Read as _;

const POOL: usize = 100; // per-head candidate budget (§11 default)
const GT_K: usize = 10;
const HEADS: [&str; 5] = ["full", "q0", "q1", "q2", "q3"];
const ENGINE_DIM: usize = 196;

fn read_f32(path: &str) -> Vec<f32> {
    let mut f = std::fs::File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    buf.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(path: &str) -> Vec<u32> {
    let mut f = std::fs::File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    buf.chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

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

fn percentile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((p / 100.0) * (s.len() as f64 - 1.0)).round() as usize % s.len()]
}

fn avg(rs: &[RankMetrics]) -> RankMetrics {
    let n = rs.len().max(1) as f64;
    RankMetrics {
        recall_at_1: rs.iter().map(|r| r.recall_at_1).sum::<f64>() / n,
        recall_at_5: rs.iter().map(|r| r.recall_at_5).sum::<f64>() / n,
        recall_at_10: rs.iter().map(|r| r.recall_at_10).sum::<f64>() / n,
        recall_at_50: rs.iter().map(|r| r.recall_at_50).sum::<f64>() / n,
        ndcg_at_10: rs.iter().map(|r| r.ndcg_at_10).sum::<f64>() / n,
        mrr: rs.iter().map(|r| r.mrr).sum::<f64>() / n,
    }
}

/// rank_metrics consumes slice ORDER as the ranking (HC-5): sort here.
fn rank_sorted(ranked: &[(u64, f32)], gt: &[u64], k: usize) -> RankMetrics {
    let mut v = ranked.to_vec();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    rank_metrics(&v, gt, k)
}

struct Loaded {
    corpus: Vec<[[f32; ENGINE_DIM]; 5]>, // doc → 5 heads × 196-d
    queries: Vec<[[f32; ENGINE_DIM]; 5]>,
    gt: Vec<Vec<u64>>, // per query, top-10 corpus doc indices
}

fn load(data_dir: &str) -> Loaded {
    let meta = std::fs::read_to_string(format!("{data_dir}/meta.json")).unwrap();
    // light parse: pull n_docs / n_queries
    let get = |key: &str| -> usize {
        let pat = format!("\"{key}\": ");
        let i = meta
            .find(&pat)
            .unwrap_or_else(|| panic!("meta missing {key}"));
        meta[i + pat.len()..]
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap()
    };
    let (n_docs, n_q) = (get("n_docs"), get("n_queries"));
    let corpus_raw = read_f32(&format!("{data_dir}/corpus_heads.f32"));
    let query_raw = read_f32(&format!("{data_dir}/query_heads.f32"));
    let gt_raw = read_u32(&format!("{data_dir}/gt.u32"));
    assert_eq!(corpus_raw.len(), n_docs * 5 * ENGINE_DIM);
    assert_eq!(query_raw.len(), n_q * 5 * ENGINE_DIM);
    assert_eq!(gt_raw.len(), n_q * 10);
    let mut corpus = Vec::with_capacity(n_docs);
    for d in corpus_raw.chunks(5 * ENGINE_DIM) {
        let mut heads = [[0f32; ENGINE_DIM]; 5];
        for (h, chunk) in d.chunks(ENGINE_DIM).enumerate() {
            heads[h].copy_from_slice(chunk);
        }
        corpus.push(heads);
    }
    let mut queries = Vec::with_capacity(n_q);
    for d in query_raw.chunks(5 * ENGINE_DIM) {
        let mut heads = [[0f32; ENGINE_DIM]; 5];
        for (h, chunk) in d.chunks(ENGINE_DIM).enumerate() {
            heads[h].copy_from_slice(chunk);
        }
        queries.push(heads);
    }
    let gt = gt_raw
        .chunks(10)
        .map(|c| c.iter().map(|&x| x as u64).collect())
        .collect();
    Loaded {
        corpus,
        queries,
        gt,
    }
}

mod guard {
    pub struct TempDir(pub std::path::PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn open_engine() -> (guard::TempDir, AttentionEngine) {
    let dir = std::env::temp_dir().join(format!("phase3-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Async).unwrap();
    (guard::TempDir(dir), e)
}

type HeadPools = Vec<(Vec<u64>, Vec<f32>, Vec<f32>)>;

struct Pools {
    /// per query: per head: (engine_ids, raw_scores, norm_scores)
    heads: Vec<HeadPools>,
    #[allow(dead_code)] // id resolution is used by persistence scenarios (§14)
    engine_to_idx: HashMap<u64, usize>,
    idx_to_engine: Vec<u64>,
    recall_frac: f64,
    queries_all_covered: usize,
}

/// Candidate generation ONCE from the engine (frozen per-head ANN path),
/// then everything downstream trains/evaluates on the cache (Phase 2B §4
/// discipline: training never touches HNSW).
fn generate_pools(e: &AttentionEngine, loaded: &Loaded) -> (Pools, f64, f64) {
    let coll = e.get_collection("bench").unwrap();
    let (mut engine_to_idx, mut idx_to_engine) = (HashMap::new(), vec![0u64; loaded.corpus.len()]);
    {
        let store = e.document_store.read();
        let mapper = e.id_mapper.read();
        for rec in store.list_all_records() {
            if let Some(idx) = rec.fields.get("idx").and_then(|v| v.as_u64()) {
                if let Some(num) = mapper.uuid_to_id(&rec.id) {
                    idx_to_engine[idx as usize] = num;
                    engine_to_idx.insert(num, idx as usize);
                }
            }
        }
    }
    let t0 = std::time::Instant::now();
    let mut heads_pools = Vec::with_capacity(loaded.queries.len());
    for q in &loaded.queries {
        let mut per_head = Vec::with_capacity(HEADS.len());
        for (h, hname) in HEADS.iter().enumerate() {
            let hname_string = hname.to_string();
            let ranked: Vec<(u64, f32)> = coll
                .attend_detailed(
                    std::slice::from_ref(&hname_string),
                    &q[h],
                    POOL,
                    None,
                    Some(attentiondb_core::collection::RetrievalMode::SingleHead),
                    None,
                    None,
                    None,
                )
                .unwrap()
                .into_iter()
                .map(|r| (r.id, r.final_score))
                .collect();
            let raw: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();
            let mut norm = raw.clone();
            minmax(&mut norm);
            per_head.push((
                ranked.iter().map(|(id, _)| *id).collect::<Vec<u64>>(),
                raw,
                norm,
            ));
        }
        heads_pools.push(per_head);
    }
    let gen_s = t0.elapsed().as_secs_f64();
    // Candidate recall is a MEASURED quantity (Phase 3 spec §8/§11): HNSW
    // (ef_search=64, pool=100) does not guarantee true-NN recovery on real
    // data. Record per-query GT coverage over the union; hard-assert only
    // pool structure (non-empty, engine-id linkage).
    let mut covered = 0usize;
    let mut total_gt = 0usize;
    let mut queries_all_covered = 0usize;
    for (qi, pools) in heads_pools.iter().enumerate() {
        assert!(
            !pools.is_empty() && !pools[0].0.is_empty(),
            "empty pool at query {qi} — candidate generation broken"
        );
        let mut hit = 0usize;
        let n_gt = loaded.gt[qi].len();
        for &gt_idx in &loaded.gt[qi] {
            let eid = idx_to_engine[gt_idx as usize];
            if pools.iter().any(|(ids, _, _)| ids.contains(&eid)) {
                hit += 1;
            }
        }
        covered += hit;
        total_gt += n_gt;
        if hit == n_gt {
            queries_all_covered += 1;
        }
    }
    let recall_frac = covered as f64 / total_gt.max(1) as f64;
    let verify_s = t0.elapsed().as_secs_f64() - gen_s;
    eprintln!(
        "[pools] candidate recall: gt-frac={recall_frac:.4} queries-full={} ({})",
        queries_all_covered,
        loaded.queries.len()
    );
    (
        Pools {
            heads: heads_pools,
            engine_to_idx,
            idx_to_engine,
            recall_frac,
            queries_all_covered,
        },
        gen_s,
        verify_s,
    )
}

fn minmax(v: &mut [f32]) {
    let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1e-12);
    for x in v.iter_mut() {
        *x = (*x - lo) / span;
    }
}

// multiple parallel per-query arrays are indexed by qi below — the
// explicit index is the honest form here
#[allow(clippy::needless_range_loop)]
fn build_gating_dataset(
    loaded: &Loaded,
    pools: &Pools,
    idx_to_engine: &[u64],
    split_seed: u64,
) -> GatingDataset {
    let n = loaded.queries.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = DetRng::new(split_seed ^ 0x5EED);
    rng.shuffle(&mut order);
    let (n_train, n_val) = (n * 7 / 10, n * 15 / 100);
    let mut split_of = vec![Split::Test; n];
    for (rank, &qi) in order.iter().enumerate() {
        split_of[qi] = match rank {
            r if r < n_train => Split::Train,
            r if r < n_train + n_val => Split::Val,
            _ => Split::Test,
        };
    }
    let mut queries = Vec::with_capacity(n);
    for qi in 0..n {
        let gt_engine: Vec<u64> = loaded.gt[qi]
            .iter()
            .map(|&i| idx_to_engine[i as usize])
            .collect();
        let mut heads = Vec::with_capacity(HEADS.len());
        for (ids, raw, norm) in &pools.heads[qi] {
            let ranked: Vec<(u64, f32)> = ids.iter().copied().zip(raw.iter().copied()).collect();
            let m = rank_sorted(&ranked, &gt_engine, GT_K);
            heads.push(HeadExample {
                candidates: ids.clone(),
                raw_scores: raw.clone(),
                norm_scores: norm.clone(),
                exact_scores: raw.clone(), // engine score ≡ cosine here (§ datasets doc)
                recall_at_k: m.recall_at_10 as f32,
                ndcg_at_k: m.ndcg_at_10 as f32,
                mrr: m.mrr as f32,
            });
        }
        let mut qvec = Vec::with_capacity(5 * ENGINE_DIM);
        for h in 0..5 {
            qvec.extend_from_slice(&loaded.queries[qi][h]);
        }
        queries.push(QueryExample {
            query_id: qi as u64,
            query: qvec,
            query_group: None,
            split: split_of[qi],
            ground_truth: gt_engine,
            heads,
        });
    }
    GatingDataset {
        format: "attentiondb-gating-dataset".into(),
        version: 1,
        num_heads: 5,
        input_dim: 5 * ENGINE_DIM,
        top_k: GT_K,
        corpus_desc: "PH3-DS-FM fashion multiview (real images)".into(),
        seed: split_seed,
        queries,
    }
}

fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}

fn train_gating_grid(
    ds: &GatingDataset,
    seed: u64,
) -> Vec<(Objective, GatingMlp, f32, f32, usize, TrainOutcome)> {
    let mut out = Vec::new();
    for &obj in [
        Objective::QualityRegression,
        Objective::SoftTarget,
        Objective::Pairwise,
    ]
    .iter()
    {
        for &lr in [0.01f32, 0.003].iter() {
            for &hidden in [32usize, 64].iter() {
                let cfg = TrainingConfig {
                    objective: obj,
                    max_epochs: 300,
                    patience: 15,
                    lr,
                    hidden,
                    seed,
                    ..TrainingConfig::default()
                };
                let outcome = train_gating(ds, &cfg, QualityTarget::Recall);
                out.push((obj, outcome.model.clone(), 0.0, lr, hidden, outcome));
            }
        }
    }
    out
}

fn eval_gating(m: &GatingMlp, ds: &GatingDataset, t: f32, split: Split) -> RankMetrics {
    let rs: Vec<RankMetrics> = ds
        .queries
        .iter()
        .filter(|q| q.split == split)
        .map(|q| {
            let lg = m.logits(&q.query);
            let w = softmax(&lg.iter().map(|x| x / t).collect::<Vec<_>>());
            let ranked = fuse_weighted(q, &w);
            rank_sorted(&ranked, &q.ground_truth, ds.top_k)
        })
        .collect();
    avg(&rs)
}

fn fit_temperature(m: &GatingMlp, ds: &GatingDataset) -> f32 {
    let mut best = (0.25f32, -1.0f64);
    for &t in [0.25f32, 0.5, 1.0, 2.0].iter() {
        let r = eval_gating(m, ds, t, Split::Val).recall_at_10;
        if r > best.1 {
            best = (t, r);
        }
    }
    best.0
}

fn cmd_quality(tier: &str, data_dir: &str, out_dir: &str, seeds: &[u64]) -> String {
    std::fs::create_dir_all(format!("{out_dir}/models")).unwrap();
    let loaded = load(data_dir);
    let n_docs = loaded.corpus.len();
    let n_q = loaded.queries.len();
    let (_guard, e) = open_engine();
    let heads: [&str; 5] = HEADS;
    e.create_collection("bench", ENGINE_DIM, &heads).unwrap();
    let t_build = std::time::Instant::now();
    for (i, doc) in loaded.corpus.iter().enumerate() {
        let mut fields = std::collections::HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(i));
        let mut r = Record::new(fields);
        for (h, hname) in HEADS.iter().enumerate() {
            r.k_vecs.insert(hname.to_string(), doc[h].to_vec());
        }
        e.insert_document("bench", r).unwrap();
    }
    let build_s = t_build.elapsed().as_secs_f64();
    let rss_after_build = rss_mb();
    let peak_build = peak_rss_mb();

    let (pools, gen_s, verify_s) = generate_pools(&e, &loaded);
    let ds = build_gating_dataset(&loaded, &pools, &pools.idx_to_engine, 42);
    // pools are engine-generated and HNSW construction is OS-seeded: the
    // cached dataset IS the reproducibility unit (Phase 2B doctrine)
    std::fs::write(
        format!("{out_dir}/ds.json"),
        serde_json::to_string(&ds).unwrap(),
    )
    .unwrap();
    let (tr, va, te) = ds.split_counts();
    let ds_hash = ds.content_hash();

    // ---- gating training + val selection (seed 42 grid; protocol of 2B) ----
    let grid = train_gating_grid(&ds, 42);
    let mut val_grid = String::from("objective,lr,hidden,T,val_r10\n");
    let mut best: Option<(Objective, GatingMlp, f32, f32, usize, f64)> = None;
    for (obj, m, _t, lr, hidden, _out) in &grid {
        let t = fit_temperature(m, &ds);
        let r = eval_gating(m, &ds, t, Split::Val).recall_at_10;
        let _ = writeln!(val_grid, "{obj:?},{lr},{hidden},{t},{r:.4}");
        if best.as_ref().map(|b| r > b.5).unwrap_or(true) {
            best = Some((*obj, m.clone(), t, *lr, *hidden, r));
        }
    }
    std::fs::write(format!("{out_dir}/val_grid_gating.csv"), &val_grid).unwrap();
    let (gobj, gm, gt_fit, glr, ghidden, gval) = best.unwrap();

    // ---- arms on TEST ----
    let mut rows: Vec<(String, String, RankMetrics)> = Vec::new();

    // A: single-head ANN (full view only)
    let single: Vec<RankMetrics> = ds
        .queries
        .iter()
        .filter(|q| q.split == Split::Test)
        .map(|q| rank_sorted(&head_ranking(q, 0), &q.ground_truth, GT_K))
        .collect();
    rows.push(("single_head_ann".into(), "agg".into(), avg(&single)));

    // B: uniform multi-head
    let uniform: Vec<RankMetrics> = ds
        .queries
        .iter()
        .filter(|q| q.split == Split::Test)
        .map(|q| {
            rank_sorted(
                &fuse_weighted(q, &uniform_weights(5)),
                &q.ground_truth,
                GT_K,
            )
        })
        .collect();
    rows.push(("uniform_multihead".into(), "agg".into(), avg(&uniform)));

    // C: trained gating (seed grid over `seeds`)
    let mut per_seed = Vec::new();
    for (si, &seed) in seeds.iter().enumerate() {
        let (obj, m, t, lr, hid, _out): (
            Objective,
            GatingMlp,
            f32,
            f32,
            usize,
            Option<TrainOutcome>,
        ) = if si == 0 {
            (gobj, gm.clone(), gt_fit, glr, ghidden, None)
        } else {
            let g = train_gating_grid(&ds, seed);
            let mut best2: Option<(Objective, GatingMlp, f32, f32, usize, Option<TrainOutcome>)> =
                None;
            let mut best_r = -1.0f64;
            for (obj, m, _t, lr, hidden, o) in &g {
                let t = fit_temperature(m, &ds);
                let r = eval_gating(m, &ds, t, Split::Val).recall_at_10;
                if r > best_r {
                    best_r = r;
                    best2 = Some((*obj, m.clone(), t, *lr, *hidden, Some(o.clone())));
                }
            }
            best2.unwrap()
        };
        let r = eval_gating(&m, &ds, t, Split::Test);
        per_seed.push((format!("gating({obj:?},lr={lr},h={hid},T={t})"), seed, r));
        if si == 0 {
            let meta = TrainingMeta {
                seed,
                dataset_hash: ds_hash,
                objective: format!("{obj:?}"),
                learning_rate: lr,
                batch_size: 32,
                epochs_run: _out.as_ref().map(|o| o.curves.len()).unwrap_or(0),
                best_val_loss: _out.as_ref().map(|o| o.best_val_loss).unwrap_or(0.0),
                l2: 1e-4,
                timestamp_unix: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                code_commit: option_env!("GIT_HASH").unwrap_or("unknown").to_string(),
                hardware: format!(
                    "cpus={}",
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1)
                ),
            };
            let card =
                ModelCard::from_mlp(&m, meta, &format!("ph3-gating-{tier}-s{seed}"), "gating");
            card.save(std::path::Path::new(&format!(
                "{out_dir}/models/gating_s{seed}.json"
            )))
            .unwrap();
        }
    }
    let gate_ms: Vec<RankMetrics> = per_seed.iter().map(|(_, _, m)| *m).collect();
    rows.push(("trained_gating".into(), "agg".into(), avg(&gate_ms)));
    for (tag, s, m) in &per_seed {
        let _ = tag;
        rows.push(("trained_gating".into(), format!("{s}"), *m));
    }

    // F: exact reference (brute force full-view COSINE = the GT definition —
    // unit-normalized vectors; a plain dot on raw f32 is NOT cosine here).
    // Sanity anchor: this arm MUST score R@10 = 1.0.
    let t_exact = std::time::Instant::now();
    let mut exact = Vec::new();
    {
        // normalize corpus full-view once
        let mut cn: Vec<f32> = Vec::with_capacity(n_docs * ENGINE_DIM);
        for d in &loaded.corpus {
            let v = &d[0];
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            cn.extend(v.iter().map(|x| x / n));
        }
        for q in ds.queries.iter() {
            if q.split != Split::Test {
                continue;
            }
            let qid = q.query_id as usize;
            let qv = &loaded.queries[qid][0];
            let qn: f32 = qv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let mut ranked: Vec<(u64, f32)> = loaded
                .corpus
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let cnv = &cn[i * ENGINE_DIM..(i + 1) * ENGINE_DIM];
                    let dot: f32 = qv.iter().zip(cnv).map(|(a, b)| a * b).sum::<f32>() / qn;
                    (i as u64, dot)
                })
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            // exact arm ranks in CORPUS-INDEX space → compare against the
            // doc-index ground truth (q.ground_truth is ENGINE-id space)
            exact.push(rank_sorted(&ranked, &loaded.gt[qid], GT_K));
        }
    }
    let exact_s = t_exact.elapsed().as_secs_f64();
    rows.push(("exact_reference".into(), "agg".into(), avg(&exact)));

    // candidate recall (union of the 5 pools)
    let cr: f64 = ds
        .queries
        .iter()
        .map(|q| {
            let union: std::collections::HashSet<u64> = q
                .heads
                .iter()
                .flat_map(|h| h.candidates.iter().copied())
                .collect();
            q.ground_truth.iter().filter(|g| union.contains(g)).count() as f64
                / q.ground_truth.len().max(1) as f64
        })
        .sum::<f64>()
        / ds.queries.len() as f64;

    // ---- latency (measurement over ALL queries, warm; frozen seed-42 path) ----
    let coll = e.get_collection("bench").unwrap();
    let reps = 5;
    let mut lat_single = Vec::new();
    let mut lat_fuse5 = Vec::new();
    let _lat_gating: Vec<f64> = Vec::new();
    let mut lat_exact = Vec::new();
    for _ in 0..reps {
        for (qi, q) in loaded.queries.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let full = "full".to_string();
            let _ = coll
                .attend_detailed(
                    std::slice::from_ref(&full),
                    &q[0],
                    POOL,
                    None,
                    Some(attentiondb_core::collection::RetrievalMode::SingleHead),
                    None,
                    None,
                    None,
                )
                .unwrap();
            lat_single.push(t0.elapsed().as_secs_f64() * 1e6);
            let t1 = std::time::Instant::now();
            let mut acc: HashMap<u64, f32> = HashMap::new();
            for (hi, hname) in HEADS.iter().enumerate() {
                let hname_string = hname.to_string();
                for r in coll
                    .attend_detailed(
                        std::slice::from_ref(&hname_string),
                        &q[hi],
                        POOL,
                        None,
                        Some(attentiondb_core::collection::RetrievalMode::SingleHead),
                        None,
                        None,
                        None,
                    )
                    .unwrap()
                {
                    *acc.entry(r.id).or_insert(0.0) += r.final_score / 5.0;
                }
            }
            let _ = acc;
            lat_fuse5.push(t1.elapsed().as_secs_f64() * 1e6);
            let ex = &loaded.queries[qi][0];
            let exn: f32 = ex.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let t2 = std::time::Instant::now();
            let mut top = [f32::NEG_INFINITY; 10];
            for d in &loaded.corpus {
                let mut dot: f32 = ex.iter().zip(d[0].iter()).map(|(a, b)| a * b).sum();
                dot /= exn; // cosine reference path (norm of unit corpus vec ≡ 1 only if normalized; see datasets doc)
                            // running insertion into a fixed-size top-10 (branchy but
                            // representative of the reference exact path)
                let mut k = 0usize;
                while k < 10 && top[k] >= dot {
                    k += 1;
                }
                if k < 10 {
                    let mut j = 9usize;
                    while j > k {
                        top[j] = top[j - 1];
                        j -= 1;
                    }
                    top[k] = dot;
                }
            }
            let _ = top;
            lat_exact.push(t2.elapsed().as_secs_f64() * 1e6);
        }
    }
    // gating model latency (frozen seed-42 model, forward only)
    let mut lat_gm = Vec::new();
    for q in ds
        .queries
        .iter()
        .filter(|q| q.split == Split::Test)
        .take(150)
    {
        for _ in 0..reps {
            let t = std::time::Instant::now();
            let _ = gm.logits(&q.query);
            lat_gm.push(t.elapsed().as_secs_f64() * 1e6);
        }
    }
    let lat_row = |name: &str, v: &[f64]| {
        format!(
            "{name},{:.2},{:.2},{:.2},{:.0}\n",
            percentile(v, 50.0),
            percentile(v, 95.0),
            percentile(v, 99.0),
            1e6 / percentile(v, 50.0)
        )
    };
    let mut lat = String::from("component,p50_us,p95_us,p99_us,qps\n");
    lat.push_str(&lat_row("ann_single_head", &lat_single));
    lat.push_str(&lat_row("ann_5head_plus_fusion", &lat_fuse5));
    lat.push_str(&lat_row("gating_mlp_forward", &lat_gm));
    lat.push_str(&lat_row("exact_bruteforce_196d", &lat_exact));

    // ---- gating weight diagnostics (mean weight per head over test) ----
    {
        let mut wsum = [0.0f64; 5];
        let mut n = 0.0f64;
        for q in ds.queries.iter().filter(|q| q.split == Split::Test) {
            let lg = gm.logits(&q.query);
            let w = softmax(&lg.iter().map(|x| x / gt_fit).collect::<Vec<_>>());
            for (i, x) in w.iter().enumerate() {
                wsum[i] += *x as f64;
            }
            n += 1.0;
        }
        let mut wd = String::from("head,mean_weight_test\n");
        for (i, h) in HEADS.iter().enumerate() {
            let _ = writeln!(wd, "{h},{:.4}", wsum[i] / n);
        }
        std::fs::write(format!("{out_dir}/gating_weights.csv"), &wd).unwrap();
    }

    // ---- emit artifacts ----
    let mut csv = String::from("arm,seed,R@1,R@5,R@10,NDCG@10,MRR\n");
    for (arm, seed, m) in &rows {
        let _ = writeln!(
            csv,
            "{arm},{seed},{:.4},{:.4},{:.4},{:.4},{:.4}",
            m.recall_at_1, m.recall_at_5, m.recall_at_10, m.ndcg_at_10, m.mrr
        );
    }
    std::fs::write(format!("{out_dir}/results.csv"), &csv).unwrap();
    std::fs::write(format!("{out_dir}/latency.csv"), &lat).unwrap();
    let cfg = serde_json::json!({
        "experiment": format!("PH3-QUAL-FM-{tier}"),
        "dataset": {"dir": data_dir, "tier": tier, "n_docs": n_docs, "n_queries": n_q,
                     "heads": HEADS, "engine_dim": ENGINE_DIM, "gt": "exact full-view cosine top-10"},
        "engine": {"hnsw": {"M": 16, "ef_construction": 400, "ef_search": 64}, "pool": POOL, "top_k": GT_K,
                    "durability": "Async (in-session tempdir; persistence scenarios are separate PH3 runs)"},
        "gating": {"grid": "3 objectives x lr{0.01,0.003} x hidden{32,64}, val-select + val temperature",
                    "frozen": {"objective": format!("{gobj:?}"), "lr": glr, "hidden": ghidden, "T": gt_fit, "val_r10": gval}},
        "splits": {"train": tr, "val": va, "test": te, "split_seed": 42},
        "seeds": seeds,
        "build_seconds": build_s, "candidate_gen_seconds": gen_s, "pool_verify_seconds": verify_s,
        "rss_after_build_mb": rss_after_build, "peak_rss_mb": peak_build,
        "exact_eval_seconds": exact_s,
        "candidate_recall_union": cr,
        "pool_candidate_recall_gt_frac": pools.recall_frac,
        "pool_queries_full_coverage": pools.queries_all_covered,
        "dataset_hash": format!("{ds_hash:#x}"),
    });
    std::fs::write(
        format!("{out_dir}/config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .unwrap();

    let m_of = |a: &str| {
        rows.iter()
            .find(|(x, s, _)| x == a && s == "agg")
            .unwrap()
            .2
    };
    format!(
        "PH3-QUAL-FM-{tier}: docs={n_docs} queries={n_q} (tr{tr}/va{va}/te{te}) build={build_s:.1}s\n\
         TEST R@10: single={:.4} uniform={:.4} gating={:.4} (val {gval:.4}) exact={:.4} | pool-union-cand-recall={cr:.4}\n\
         p50 µs: single={:.1} 5head+fusion={:.1} gating_fwd={:.2} exact={:.0} | peak RSS {peak_build:.0} MB",
        m_of("single_head_ann").recall_at_10, m_of("uniform_multihead").recall_at_10,
        m_of("trained_gating").recall_at_10, m_of("exact_reference").recall_at_10,
        percentile(&lat_single, 50.0), percentile(&lat_fuse5, 50.0),
        percentile(&lat_gm, 50.0), percentile(&lat_exact, 50.0),
    )
}

fn uniform_weights(n: usize) -> Vec<f32> {
    vec![1.0 / n as f32; n]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("quality");
    let mut tier = "S".to_string();
    let mut data = "/tmp/phase3/fashion-S".to_string();
    let mut out = "research/phase3/raw/runs/PH3-QUAL-FM-S".to_string();
    let mut seeds = vec![42u64, 7, 1];
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--tier" => {
                tier = args[i + 1].clone();
                i += 2;
            }
            "--data" => {
                data = args[i + 1].clone();
                i += 2;
            }
            "--out" => {
                out = args[i + 1].clone();
                i += 2;
            }
            "--seeds" => {
                seeds = args[i + 1]
                    .split(',')
                    .filter_map(|s| s.parse().ok())
                    .collect();
                i += 2;
            }
            "--docs" | "--path" | "--mode" => {
                i += 2; // consumed by subcommand handlers
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    match cmd {
        "quality" => {
            std::fs::create_dir_all(&out).unwrap();
            println!("{}", cmd_quality(&tier, &data, &out, &seeds));
        }
        "textquality" => {
            std::fs::create_dir_all(&out).unwrap();
            println!("{}", crate::textqual::run(&tier, &data, &out, &seeds));
        }
        "headsquality" => {
            std::fs::create_dir_all(&out).unwrap();
            println!("{}", crate::headsqual::run(&tier, &data, &out, &seeds));
        }
        "memprobe" => {
            let docs: usize = args
                .iter()
                .position(|a| a == "--docs")
                .and_then(|i| args.get(i + 1).and_then(|v| v.parse().ok()))
                .unwrap_or(10_000);
            std::fs::create_dir_all(&out).unwrap();
            println!("{}", crate::memprobe::run(&tier, &data, &out, docs));
        }
        "leaktest" => {
            let path = args
                .iter()
                .position(|a| a == "--path")
                .and_then(|i| args.get(i + 1).cloned())
                .unwrap_or_else(|| "/var/tmp/leaktest-dir".to_string());
            let mode = args
                .iter()
                .position(|a| a == "--mode")
                .and_then(|i| args.get(i + 1).cloned())
                .unwrap_or_else(|| "clean".to_string());
            println!("{}", crate::memprobe::leaktest(&path, &mode));
        }
        other => {
            eprintln!("unknown command {other}");
            std::process::exit(2);
        }
    }
}
