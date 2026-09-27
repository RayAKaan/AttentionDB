//! PH3C `headsqual` — head-count quality/latency scaling + candidate-budget
//! sweep + serial-vs-parallel decomposition on the AG News multi-field
//! workload (§13–§19). Head set, defining-head map, and dim come from the
//! dataset meta.json (H1/H2/H4/H8 ladder + S reference); GT is IDENTICAL
//! across the ladder (title/body/full spaces, verified at build time).
//!
//! Budget protocol (§16): pools generated at POOL_MAX=200; budgets
//! {10,25,50,100,200} are the first-K prefixes of the same head-sorted
//! pools (attend returns sorted top-K, so prefixes are exact sub-pools).
//! The gate is TRAINED once per seed on K=100 pools (the §11 default) and
//! EVALUATED at every K — training per-K would confound budget effects
//! with training noise. K is never tuned on test.
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::io::Read as _;

use attentiondb_core::engine::AttentionEngine;
use attentiondb_learned::eval::{fuse_weighted, rank_metrics, RankMetrics};
use attentiondb_learned::gating_v2::{
    train_gating, DetRng, GatingDataset, GatingMlp, HeadExample, ModelCard, Objective,
    QualityTarget, QueryExample, Split, TrainOutcome, TrainingConfig, TrainingMeta,
};
use attentiondb_storage::{Durability, Record};

use crate::textqual::{avg, percentile, rank_sorted, read_f32_vec_pub, softmax};

pub const GT_K: usize = 10;
pub const POOL_MAX: usize = 200;
pub const BUDGETS: [usize; 5] = [10, 25, 50, 100, 200];
/// gate trained at this budget (the §11 default), evaluated at every budget
pub const TRAIN_K: usize = 100;

struct LoadedH {
    n_docs: usize,
    dim: usize,
    heads: Vec<String>,
    defining: HashMap<String, String>, // query type -> defining head
    corpus_heads: HashMap<String, Vec<f32>>,
    queries: Vec<Vec<Vec<f32>>>, // qi -> head_i -> vec
    /// per query: DEFINING head's query vector (anchor space; may differ
    /// from engine heads on sub-view ladders)
    anchor_q: Vec<Vec<f32>>,
    qtype: Vec<u8>,
    gt: Vec<Vec<u64>>,
}

mod guard {
    pub struct TempDir(pub std::path::PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn read_i64_vec(path: &str) -> Vec<i64> {
    let mut f = std::fs::File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    buf.chunks(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn load(data_dir: &str) -> LoadedH {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{data_dir}/meta.json")).unwrap())
            .unwrap();
    let n_docs = meta["corpus"]["n_docs"].as_u64().unwrap() as usize;
    let dim = meta["dim"].as_u64().unwrap_or(512) as usize;
    let heads: Vec<String> = meta["heads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h.as_str().unwrap().to_string())
        .collect();
    let defining: HashMap<String, String> = meta["defining"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
        .collect();
    let mut corpus_heads = HashMap::new();
    let mut load_heads: Vec<String> = heads.clone();
    for dh in defining.values() {
        if !load_heads.contains(dh) {
            load_heads.push(dh.clone());
        }
    }
    for h in &load_heads {
        let v = read_f32_vec_pub(&format!("{data_dir}/corpus_{h}.f32"));
        assert_eq!(v.len(), n_docs * dim, "corpus_{h}");
        corpus_heads.insert(h.clone(), v);
    }
    let types = ["title", "body", "mixed"];
    let mut n_per_type = [0usize; 3]; // used for loader shapes
    let mut queries = Vec::new();
    let mut anchor_q = Vec::new();
    let mut qtype = Vec::new();
    #[allow(unused_assignments, unused_mut)]
    let mut qtext_unused: Vec<Vec<String>> = vec![Vec::new(); 3];
    let mut gt = Vec::new();
    for (ti, t) in types.iter().enumerate() {
        n_per_type[ti] = meta["queries"]["per_type"][t].as_u64().unwrap() as usize;
        let mut texts = vec![String::new(); n_per_type[ti]];
        for line in std::fs::read_to_string(format!("{data_dir}/query_text.jsonl"))
            .unwrap()
            .lines()
        {
            let j: serde_json::Value = serde_json::from_str(line).unwrap();
            if j["type"].as_str().unwrap() == *t {
                texts[j["i"].as_u64().unwrap() as usize] = j["text"].as_str().unwrap().to_string();
            }
        }
        qtext_unused[ti] = texts;
        let _ = texts;
        let mut head_vecs: HashMap<String, Vec<f32>> = HashMap::new();
        for h in &load_heads {
            let v = read_f32_vec_pub(&format!("{data_dir}/queries_{t}_{h}.f32"));
            assert_eq!(v.len(), n_per_type[ti] * dim, "queries_{t}_{h}");
            head_vecs.insert(h.clone(), v);
        }
        let g = read_i64_vec(&format!("{data_dir}/gt_{t}.i64"));
        let dh: String = defining
            .get(*t)
            .cloned()
            .unwrap_or_else(|| "full".to_string());
        for q in 0..n_per_type[ti] {
            let row: Vec<Vec<f32>> = heads
                .iter()
                .map(|h| head_vecs[h][q * dim..(q + 1) * dim].to_vec())
                .collect();
            queries.push(row);
            qtype.push(ti as u8);
            anchor_q.push(head_vecs[&dh][q * dim..(q + 1) * dim].to_vec());
            gt.push(
                g[q * GT_K..(q + 1) * GT_K]
                    .iter()
                    .map(|&x| x as u64)
                    .collect(),
            );
        }
    }
    LoadedH {
        n_docs,
        dim,
        heads,
        defining,
        corpus_heads,
        queries,
        anchor_q,
        qtype,
        gt,
    }
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

fn cpu_time_s() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap();
    let parts: Vec<&str> = s
        .split(')')
        .next_back()
        .unwrap_or("")
        .split_whitespace()
        .collect();
    // fields after comm: state(0) ... utime(11) stime(12) in clock ticks
    let ut: f64 = parts.get(11).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let st: f64 = parts.get(12).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    (ut + st) / 100.0
}

type HeadPool = (Vec<u64>, Vec<f32>, Vec<f32>);

struct PoolsH {
    /// qi -> head_i -> (engine_ids sorted by head score, raw, minmax-norm)
    heads: Vec<Vec<HeadPool>>,
    idx_to_engine: Vec<u64>,
    #[allow(dead_code)] // engine_to_idx: used by persistence scenarios (§14)
    engine_to_idx: HashMap<u64, usize>,
}

fn generate_pools(e: &AttentionEngine, loaded: &LoadedH, pool: usize) -> (PoolsH, f64) {
    let t0 = std::time::Instant::now();
    let coll = e.get_collection("bench").unwrap();
    let mut idx_to_engine = vec![0u64; loaded.n_docs];
    let mut engine_to_idx = HashMap::new();
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
    let mut heads_pools: Vec<Vec<HeadPool>> = Vec::with_capacity(loaded.queries.len());
    for q in &loaded.queries {
        let mut per_head = Vec::with_capacity(loaded.heads.len());
        for (h, hn) in loaded.heads.iter().enumerate() {
            let ranked: Vec<(u64, f32)> = coll
                .attend_detailed(
                    std::slice::from_ref(hn),
                    &q[h],
                    pool,
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
            per_head.push((ranked.iter().map(|(id, _)| *id).collect(), raw, norm));
        }
        heads_pools.push(per_head);
    }
    (
        PoolsH {
            heads: heads_pools,
            idx_to_engine,
            engine_to_idx,
        },
        t0.elapsed().as_secs_f64(),
    )
}

pub fn minmax(v: &mut [f32]) {
    let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1e-12);
    for x in v.iter_mut() {
        *x = (*x - lo) / span;
    }
}

#[allow(clippy::needless_range_loop)] // parallel per-query arrays indexed by qi
fn build_ds(loaded: &LoadedH, pools: &PoolsH, k: usize, split_seed: u64) -> GatingDataset {
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
    let nh = loaded.heads.len();
    let mut queries = Vec::with_capacity(n);
    for qi in 0..n {
        let gt_engine: Vec<u64> = loaded.gt[qi]
            .iter()
            .map(|&i| pools.idx_to_engine[i as usize])
            .collect();
        let mut hs = Vec::with_capacity(nh);
        for (ids, raw, norm) in &pools.heads[qi] {
            let ids = &ids[..k.min(ids.len())];
            let raw = &raw[..k.min(raw.len())];
            let norm = &norm[..k.min(norm.len())];
            let ranked: Vec<(u64, f32)> = ids.iter().copied().zip(raw.iter().copied()).collect();
            let m = rank_sorted(&ranked, &gt_engine, GT_K);
            hs.push(HeadExample {
                candidates: ids.to_vec(),
                raw_scores: raw.to_vec(),
                norm_scores: norm.to_vec(),
                exact_scores: raw.to_vec(),
                recall_at_k: m.recall_at_10 as f32,
                ndcg_at_k: m.ndcg_at_10 as f32,
                mrr: m.mrr as f32,
            });
        }
        let mut qvec = Vec::with_capacity(nh * loaded.dim);
        for h in 0..nh {
            qvec.extend_from_slice(&loaded.queries[qi][h]);
        }
        queries.push(QueryExample {
            query_id: qi as u64,
            query: qvec,
            query_group: Some(loaded.qtype[qi] as u32),
            split: split_of[qi],
            ground_truth: gt_engine,
            heads: hs,
        });
    }
    GatingDataset {
        format: "attentiondb-gating-dataset".into(),
        version: 1,
        num_heads: nh,
        input_dim: nh * loaded.dim,
        top_k: GT_K,
        corpus_desc: "PH3C head ladder (AG News multi-field)".into(),
        seed: split_seed,
        queries,
    }
}

fn train_grid(
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

fn eval_gate(m: &GatingMlp, ds: &GatingDataset, t: f32, split: Split) -> RankMetrics {
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

fn fit_t(m: &GatingMlp, ds: &GatingDataset) -> f32 {
    let mut best = (0.25f32, -1.0f64);
    for &t in [0.25f32, 0.5, 1.0, 2.0].iter() {
        let r = eval_gate(m, ds, t, Split::Val).recall_at_10;
        if r > best.1 {
            best = (t, r);
        }
    }
    best.0
}

fn gate_weights(m: &GatingMlp, q: &QueryExample, t: f32) -> Vec<f32> {
    let lg = m.logits(&q.query);
    softmax(&lg.iter().map(|x| x / t).collect::<Vec<_>>())
}

fn exact_ranking(v: &[f32], q: &[f32], dim: usize) -> Vec<(u64, f32)> {
    let qn: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    let mut ranked: Vec<(u64, f32)> = Vec::with_capacity(v.len() / dim);
    for i in 0..v.len() / dim {
        let d = &v[i * dim..(i + 1) * dim];
        let dot: f32 = q.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f32>() / qn;
        ranked.push((i as u64, (dot * 10_000.0).round_ties_even() / 10_000.0));
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    ranked
}

pub fn run(tier: &str, data_dir: &str, out_dir: &str, seeds: &[u64]) -> String {
    std::fs::create_dir_all(format!("{out_dir}/models")).unwrap();
    let loaded = load(data_dir);
    let nh = loaded.heads.len();
    let n_q = loaded.queries.len();
    let types = ["title", "body", "mixed"];

    // engine (disk-backed dir; see HC-P3-6 addendum)
    let dir =
        std::path::PathBuf::from("/var/tmp").join(format!("phase3c-hq-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Async).unwrap();
    let _guard = guard::TempDir(dir.clone());
    let head_refs: Vec<&str> = loaded.heads.iter().map(|s| s.as_str()).collect();
    e.create_collection("bench", loaded.dim, &head_refs)
        .unwrap();
    let t_build = std::time::Instant::now();
    {
        let text_all = std::fs::read_to_string(format!("{data_dir}/corpus_text.jsonl")).unwrap();
        let mut lines = text_all.lines();
        for i in 0..loaded.n_docs {
            let j: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
            let mut fields = HashMap::new();
            fields.insert("idx".to_string(), serde_json::json!(i));
            fields.insert(
                "title".to_string(),
                serde_json::json!(j["title"].as_str().unwrap()),
            );
            fields.insert(
                "description".to_string(),
                serde_json::json!(j["description"].as_str().unwrap()),
            );
            let mut r = Record::new(fields);
            for hn in loaded.heads.iter() {
                let v = &loaded.corpus_heads[hn][i * loaded.dim..(i + 1) * loaded.dim];
                r.k_vecs.insert(hn.clone(), v.to_vec());
            }
            e.insert_document("bench", r).unwrap();
        }
    }
    let build_s = t_build.elapsed().as_secs_f64();
    let peak_build = peak_rss_mb();

    let (pools, pool_s) = generate_pools(&e, &loaded, POOL_MAX);
    // datasets: training at TRAIN_K; evaluation at every budget re-slices pools
    let ds_train = build_ds(&loaded, &pools, TRAIN_K, 42);
    let ds_hash = ds_train.content_hash();
    std::fs::write(
        format!("{out_dir}/ds.json"),
        serde_json::to_string(&ds_train).unwrap(),
    )
    .unwrap();
    let (tr, va, te) = ds_train.split_counts();

    // ---- gating: one grid per seed, val-select at TRAIN_K ----
    let mut gate_models: Vec<(u64, GatingMlp, f32, String)> = Vec::new();
    let mut val_grid = String::from("seed,objective,lr,hidden,T,val_r10\n");
    for (si, &seed) in seeds.iter().enumerate() {
        let grid = train_grid(&ds_train, seed);
        let mut best: Option<(Objective, GatingMlp, f32, f32, usize, f64)> = None;
        for (obj, m, _t, lr, hidden, _o) in &grid {
            let t = fit_t(m, &ds_train);
            let r = eval_gate(m, &ds_train, t, Split::Val).recall_at_10;
            let _ = writeln!(val_grid, "{seed},{obj:?},{lr},{hidden},{t},{r:.4}");
            if best.as_ref().map(|b| r > b.5).unwrap_or(true) {
                best = Some((*obj, m.clone(), t, *lr, *hidden, r));
            }
        }
        let (obj, m, t, lr, hid, _) = best.unwrap();
        if si == 0 {
            let meta = TrainingMeta {
                seed,
                dataset_hash: ds_hash,
                objective: format!("{obj:?}"),
                learning_rate: lr,
                batch_size: 32,
                epochs_run: grid
                    .iter()
                    .find(|(o, ..)| *o == obj)
                    .map(|x| x.5.curves.len())
                    .unwrap_or(0),
                best_val_loss: 0.0,
                l2: 1e-4,
                timestamp_unix: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                code_commit: option_env!("GIT_HASH").unwrap_or("unknown").to_string(),
                hardware: format!("cpus=2,heads={nh},dim={}", loaded.dim),
            };
            let card = ModelCard::from_mlp(&m, meta, &format!("ph3c-hq-{tier}-s{seed}"), "gating");
            card.save(std::path::Path::new(&format!(
                "{out_dir}/models/gating_s{seed}.json"
            )))
            .unwrap();
        }
        gate_models.push((seed, m, t, format!("gating({obj:?},lr={lr},h={hid},T={t})")));
    }
    std::fs::write(format!("{out_dir}/val_grid_gating.csv"), &val_grid).unwrap();

    let test_qi: Vec<usize> = (0..n_q)
        .filter(|&i| ds_train.queries[i].split == Split::Test)
        .collect();
    let val_qi: Vec<usize> = (0..n_q)
        .filter(|&i| ds_train.queries[i].split == Split::Val)
        .collect();

    // ---- arms per budget ----
    let mut results =
        String::from("budget,arm,seed,R@1,R@5,R@10,NDCG@10,MRR,R@10_std,candidate_recall\n");
    type BudgetRow = (String, String, RankMetrics, f64, f64);
    let mut budget_rows: BTreeMap<usize, Vec<BudgetRow>> = BTreeMap::new();
    let mut cr_by_k: BTreeMap<usize, f64> = BTreeMap::new();

    for &k in BUDGETS.iter() {
        // candidate recall at K
        let (mut cov, mut tot) = (0usize, 0usize);
        for &qi in &test_qi {
            let union: std::collections::HashSet<u64> = pools.heads[qi]
                .iter()
                .flat_map(|(ids, _, _)| ids.iter().take(k).copied())
                .collect();
            cov += loaded.gt[qi]
                .iter()
                .filter(|g| union.contains(&pools.idx_to_engine[**g as usize]))
                .count();
            tot += loaded.gt[qi].len();
        }
        let cr = cov as f64 / tot.max(1) as f64;
        cr_by_k.insert(k, cr);

        // gbest: head selected on VAL at this K
        let mut head_val = vec![0.0f64; nh];
        for (h, hv) in head_val.iter_mut().enumerate() {
            let rs: Vec<RankMetrics> = val_qi
                .iter()
                .map(|&qi| {
                    let (ids, _, norm) = &pools.heads[qi][h];
                    let n = k.min(ids.len());
                    let ranked: Vec<(u64, f32)> = ids[..n]
                        .iter()
                        .copied()
                        .zip(norm[..n].iter().copied())
                        .collect();
                    rank_sorted(&ranked, &ds_train.queries[qi].ground_truth, GT_K)
                })
                .collect();
            *hv = avg(&rs).recall_at_10;
        }
        let bh = head_val
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then(b.0.cmp(&a.0)))
            .unwrap()
            .0;
        let gbest: Vec<RankMetrics> = test_qi
            .iter()
            .map(|&qi| {
                let (ids, _, norm) = &pools.heads[qi][bh];
                let n = k.min(ids.len());
                let ranked: Vec<(u64, f32)> = ids[..n]
                    .iter()
                    .copied()
                    .zip(norm[..n].iter().copied())
                    .collect();
                rank_sorted(&ranked, &ds_train.queries[qi].ground_truth, GT_K)
            })
            .collect();
        budget_rows.entry(k).or_default().push((
            "global_best_single".into(),
            "agg".into(),
            avg(&gbest),
            0.0,
            cr,
        ));

        // uniform
        let uniform: Vec<RankMetrics> = test_qi
            .iter()
            .map(|&qi| {
                let mut acc: HashMap<u64, f32> = HashMap::new();
                for (ids, _, norm) in &pools.heads[qi] {
                    let n = k.min(ids.len());
                    for (c, s) in ids[..n].iter().zip(norm[..n].iter()) {
                        *acc.entry(*c).or_insert(0.0) += s / nh as f32;
                    }
                }
                let ranked: Vec<(u64, f32)> = acc.into_iter().collect();
                rank_sorted(&ranked, &ds_train.queries[qi].ground_truth, GT_K)
            })
            .collect();
        budget_rows.get_mut(&k).unwrap().push((
            "uniform_multihead".into(),
            "agg".into(),
            avg(&uniform),
            0.0,
            cr,
        ));

        // gating (fixed trained models, evaluated at K)
        let mut per_seed_m: Vec<RankMetrics> = Vec::new();
        for (seed, m, t, _) in &gate_models {
            let rs: Vec<RankMetrics> = test_qi
                .iter()
                .map(|&qi| {
                    let q = &ds_train.queries[qi];
                    let wfull = gate_weights(m, q, *t);
                    let mut acc: HashMap<u64, f32> = HashMap::new();
                    for (h, (ids, _, norm)) in pools.heads[qi].iter().enumerate() {
                        let n = k.min(ids.len());
                        let w = wfull.get(h).copied().unwrap_or(0.0);
                        for (c, s) in ids[..n].iter().zip(norm[..n].iter()) {
                            *acc.entry(*c).or_insert(0.0) += w * s;
                        }
                    }
                    let ranked: Vec<(u64, f32)> = acc.into_iter().collect();
                    rank_sorted(&ranked, &ds_train.queries[qi].ground_truth, GT_K)
                })
                .collect();
            per_seed_m.push(avg(&rs));
            budget_rows.get_mut(&k).unwrap().push((
                "trained_gating".into(),
                format!("{seed}"),
                avg(&rs),
                0.0,
                cr,
            ));
        }
        let gate_agg = avg(&per_seed_m);
        let sd = {
            let n = per_seed_m.len() as f64;
            if n > 1.0 {
                let mu = per_seed_m.iter().map(|m| m.recall_at_10).sum::<f64>() / n;
                (per_seed_m
                    .iter()
                    .map(|m| {
                        let d = m.recall_at_10 - mu;
                        d * d
                    })
                    .sum::<f64>()
                    / (n - 1.0))
                    .sqrt()
            } else {
                0.0
            }
        };
        budget_rows.get_mut(&k).unwrap().push((
            "trained_gating".into(),
            "agg".into(),
            gate_agg,
            sd,
            cr,
        ));

        // oracle head at K
        let oracle: Vec<RankMetrics> = test_qi
            .iter()
            .map(|&qi| {
                let mut besth = (0usize, -1.0f64, -1.0f64);
                for h in 0..nh {
                    let (ids, _, _) = &pools.heads[qi][h];
                    let n = k.min(ids.len());
                    let ranked: Vec<(u64, f32)> = ids[..n]
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(r, id)| (id, -(r as f32)))
                        .collect();
                    let m = rank_metrics(&ranked, &ds_train.queries[qi].ground_truth, GT_K);
                    if besth.1 < 0.0 || (m.recall_at_10, m.ndcg_at_10) > (besth.1, besth.2) {
                        besth = (h, m.recall_at_10, m.ndcg_at_10);
                    }
                }
                let (ids, _, norm) = &pools.heads[qi][besth.0];
                let n = k.min(ids.len());
                let ranked: Vec<(u64, f32)> = ids[..n]
                    .iter()
                    .copied()
                    .zip(norm[..n].iter().copied())
                    .collect();
                rank_sorted(&ranked, &ds_train.queries[qi].ground_truth, GT_K)
            })
            .collect();
        budget_rows.get_mut(&k).unwrap().push((
            "oracle_head_empirical".into(),
            "agg".into(),
            avg(&oracle),
            0.0,
            cr,
        ));

        // exact anchor (defining-head space; doc-index GT)
        let exact: Vec<RankMetrics> = test_qi
            .iter()
            .map(|&qi| {
                let t = types[loaded.qtype[qi] as usize];
                let dh = &loaded.defining[t];
                let ranked =
                    exact_ranking(&loaded.corpus_heads[dh], &loaded.anchor_q[qi], loaded.dim);
                rank_sorted(&ranked, &loaded.gt[qi], GT_K)
            })
            .collect();
        let ex = avg(&exact);
        assert!(
            (ex.recall_at_10 - 1.0).abs() < 1e-9,
            "exact anchor FAILED at k={k}: R@10={:.4}",
            ex.recall_at_10
        );
    }

    // ---- emit budget results ----
    for (k, rows) in &budget_rows {
        for (arm, seed, m, sd, cr) in rows {
            let _ = writeln!(
                results,
                "{k},{arm},{seed},{:.4},{:.4},{:.4},{:.4},{:.4},{sd:.4},{cr:.4}",
                m.recall_at_1, m.recall_at_5, m.recall_at_10, m.ndcg_at_10, m.mrr
            );
        }
    }
    std::fs::write(format!("{out_dir}/results.csv"), &results).unwrap();

    // by-type at K=100 for every arm (gating agg over seeds)
    let mut per_type = String::from("arm,seed,qtype,n,R@1,R@5,R@10,NDCG@10,MRR\n");
    let mut emit = |name: &str, seed: &str, f: &dyn Fn(usize) -> RankMetrics| {
        for (ti, t) in types.iter().enumerate() {
            let rs: Vec<RankMetrics> = test_qi
                .iter()
                .filter(|&&qi| loaded.qtype[qi] as usize == ti)
                .map(|&qi| f(qi))
                .collect();
            let m = avg(&rs);
            let _ = writeln!(
                per_type,
                "{name},{seed},{t},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
                rs.len(),
                m.recall_at_1,
                m.recall_at_5,
                m.recall_at_10,
                m.ndcg_at_10,
                m.mrr
            );
        }
    };
    // gbest @100
    {
        let mut head_val = vec![0.0f64; nh];
        for (h, hv) in head_val.iter_mut().enumerate() {
            let rs: Vec<RankMetrics> = val_qi
                .iter()
                .map(|&qi| {
                    let (ids, _, norm) = &pools.heads[qi][h];
                    let ranked: Vec<(u64, f32)> =
                        ids.iter().copied().zip(norm.iter().copied()).collect();
                    rank_sorted(&ranked, &ds_train.queries[qi].ground_truth, GT_K)
                })
                .collect();
            *hv = avg(&rs).recall_at_10;
        }
        let bh = head_val
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then(b.0.cmp(&a.0)))
            .unwrap()
            .0;
        emit("global_best_single", "agg", &|qi| {
            let (ids, _, norm) = &pools.heads[qi][bh];
            let ranked: Vec<(u64, f32)> = ids.iter().copied().zip(norm.iter().copied()).collect();
            rank_sorted(&ranked, &ds_train.queries[qi].ground_truth, GT_K)
        });
    }
    emit("uniform_multihead", "agg", &|qi| {
        let mut acc: HashMap<u64, f32> = HashMap::new();
        for (ids, _, norm) in &pools.heads[qi] {
            for (c, s) in ids.iter().zip(norm.iter()) {
                *acc.entry(*c).or_insert(0.0) += s / nh as f32;
            }
        }
        let ranked: Vec<(u64, f32)> = acc.into_iter().collect();
        rank_sorted(&ranked, &ds_train.queries[qi].ground_truth, GT_K)
    });
    // gating agg across seeds, per type + weights diagnostics (seed-42 model)
    {
        let (s0, m0, t0, _) = &gate_models[0];
        let _ = s0;
        let mut wcsv = String::from("qid,qtype,"); // dynamic head columns
        for h in &loaded.heads {
            let _ = write!(wcsv, "w_{h},");
        }
        let _ = writeln!(wcsv, "argmax_head,oracle_head,agree,entropy");
        let mut agree_ct = 0usize;
        let mut n = 0usize;
        for &qi in &test_qi {
            let q = &ds_train.queries[qi];
            let w = gate_weights(m0, q, *t0);
            let arg = w
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then(b.0.cmp(&a.0)))
                .unwrap()
                .0;
            let mut besth = (0usize, -1.0f64, -1.0f64);
            for h in 0..nh {
                let (ids, _, _) = &pools.heads[qi][h];
                let ranked: Vec<(u64, f32)> = ids
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(r, id)| (id, -(r as f32)))
                    .collect();
                let m = rank_metrics(&ranked, &q.ground_truth, GT_K);
                if besth.1 < 0.0 || (m.recall_at_10, m.ndcg_at_10) > (besth.1, besth.2) {
                    besth = (h, m.recall_at_10, m.ndcg_at_10);
                }
            }
            let agree = (arg == besth.0) as usize;
            agree_ct += agree;
            n += 1;
            let ent: f64 = -w
                .iter()
                .map(|p| {
                    let p = *p as f64;
                    if p > 0.0 {
                        p * p.ln()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();
            let _ = write!(wcsv, "{},{},", qi, types[loaded.qtype[qi] as usize]);
            for x in &w {
                let _ = write!(wcsv, "{x:.4},");
            }
            let _ = writeln!(
                wcsv,
                "{},{},{},{:.4}",
                loaded.heads[arg], loaded.heads[besth.0], agree, ent
            );
        }
        let _ = writeln!(
            wcsv,
            "# oracle_agreement_rate,{:.4}",
            agree_ct as f64 / n.max(1) as f64
        );
        std::fs::write(format!("{out_dir}/gating_weights_test.csv"), &wcsv).unwrap();
    }
    for (seed, m, t, _) in &gate_models {
        emit("trained_gating", &format!("{seed}"), &|qi| {
            let q = &ds_train.queries[qi];
            let w = gate_weights(m, q, *t);
            let ranked = fuse_weighted(q, &w);
            rank_sorted(&ranked, &q.ground_truth, GT_K)
        });
    }
    std::fs::write(format!("{out_dir}/query_groups.csv"), &per_type).unwrap();

    // ---- candidate-recall decomposition at TRAIN_K (§17) ----
    {
        let (_s0, m0, t0, _) = &gate_models[0];
        let mut dcsv = String::from("disposition,count,frac_of_gt_misses\n");
        let (mut absent, mut ranked_out, mut sel_err) = (0usize, 0usize, 0usize);
        let mut gt_misses = 0usize;
        for &qi in &test_qi {
            let q = &ds_train.queries[qi];
            let w = gate_weights(m0, q, *t0);
            let mut acc: HashMap<u64, f32> = HashMap::new();
            let mut in_pool: HashMap<u64, bool> = HashMap::new();
            for (h, (ids, _, norm)) in pools.heads[qi].iter().enumerate() {
                let ww = w.get(h).copied().unwrap_or(0.0);
                for (c, s) in ids.iter().zip(norm.iter()) {
                    *acc.entry(*c).or_insert(0.0) += ww * s;
                    in_pool.insert(*c, true);
                }
            }
            let mut ranked: Vec<(u64, f32)> = acc.into_iter().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            let top: std::collections::HashSet<u64> =
                ranked.iter().take(GT_K).map(|(id, _)| *id).collect();
            let t = types[loaded.qtype[qi] as usize];
            let dh = &loaded.defining[t];
            let dh_idx = loaded
                .heads
                .iter()
                .position(|h| h == dh)
                .unwrap_or(usize::MAX);
            for g in &loaded.gt[qi] {
                let ge = pools.idx_to_engine[*g as usize];
                if top.contains(&ge) {
                    continue;
                }
                gt_misses += 1;
                if !in_pool.contains_key(&ge) {
                    absent += 1;
                } else {
                    ranked_out += 1;
                    // view-selection error: defining head's pool had it in top-10
                    if dh_idx != usize::MAX {
                        let (ids, _, _) = &pools.heads[qi][dh_idx];
                        if ids.iter().take(GT_K).any(|c| *c == ge) {
                            sel_err += 1;
                        }
                    }
                }
            }
        }
        let f = |c: usize| {
            if gt_misses > 0 {
                format!("{:.4}", c as f64 / gt_misses as f64)
            } else {
                "n/a".into()
            }
        };
        let _ = writeln!(dcsv, "absent_from_all_pools,{absent},{}", f(absent));
        let _ = writeln!(
            dcsv,
            "in_pool_ranked_outside_top10,{ranked_out},{}",
            f(ranked_out)
        );
        let _ = writeln!(
            dcsv,
            "ranked_out_defining_head_had_it_top10 (view-selection error),{sel_err},{}",
            f(sel_err)
        );
        let _ = writeln!(dcsv, "gt_misses_total,{gt_misses},1.0");
        std::fs::write(format!("{out_dir}/candidate_decomposition.csv"), &dcsv).unwrap();
    }

    // ---- latency: serial vs parallel head path (§14/PH3C-HEAD-002) ----
    {
        let coll = e.get_collection("bench").unwrap();
        let reps = 3;
        let mut lat = String::from("mode,head_count,n,p50_us,p95_us,p99_us,qps,cpu_s_total\n");
        let cpu0 = cpu_time_s();
        let mut serial = Vec::new();
        for _ in 0..reps {
            for &qi in &test_qi {
                let t0 = std::time::Instant::now();
                for (h, hn) in loaded.heads.iter().enumerate() {
                    let _ = coll
                        .attend_detailed(
                            std::slice::from_ref(hn),
                            &loaded.queries[qi][h],
                            TRAIN_K,
                            None,
                            Some(attentiondb_core::collection::RetrievalMode::SingleHead),
                            None,
                            None,
                            None,
                        )
                        .unwrap();
                }
                serial.push(t0.elapsed().as_secs_f64() * 1e6);
            }
        }
        let cpu_serial = cpu_time_s() - cpu0;
        for mode_i in 0..1 {
            let _ = mode_i;
            let p = |v: &Vec<f64>, q: f64| percentile(v, q);
            let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
            let _ = writeln!(
                lat,
                "serial,{nh},{},{:.1},{:.1},{:.1},{:.0},{:.2}",
                serial.len(),
                p(&serial, 50.0),
                p(&serial, 95.0),
                p(&serial, 99.0),
                1e6 / mean(&serial),
                cpu_serial
            );
        }
        // parallel: 2 workers, round-robin heads via scoped threads
        let cpu1 = cpu_time_s();
        let mut par: Vec<f64> = Vec::new();
        let heads_arc = &loaded.heads;
        for _ in 0..reps {
            for &qi in &test_qi {
                let t0 = std::time::Instant::now();
                std::thread::scope(|s| {
                    let (mut lo, mut hi) = (0usize, 0usize);
                    let mut handles = Vec::new();
                    let nthreads = 2usize;
                    for tid in 0..nthreads {
                        let coll = &coll;
                        let qs = &loaded.queries[qi];
                        let hs = heads_arc;
                        handles.push(s.spawn(move || {
                            let mut h = tid;
                            loop {
                                if h >= hs.len() {
                                    break;
                                }
                                let _ = coll
                                    .attend_detailed(
                                        std::slice::from_ref(&hs[h]),
                                        &qs[h],
                                        TRAIN_K,
                                        None,
                                        Some(
                                            attentiondb_core::collection::RetrievalMode::SingleHead,
                                        ),
                                        None,
                                        None,
                                        None,
                                    )
                                    .unwrap();
                                h += nthreads;
                            }
                        }));
                        let _ = (&mut lo, &mut hi);
                    }
                    for handle in handles {
                        let _ = handle.join();
                    }
                });
                par.push(t0.elapsed().as_secs_f64() * 1e6);
            }
        }
        let cpu_par = cpu_time_s() - cpu1;
        let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        let _ = writeln!(
            lat,
            "parallel2,{nh},{},{:.1},{:.1},{:.1},{:.0},{:.2}",
            par.len(),
            percentile(&par, 50.0),
            percentile(&par, 95.0),
            percentile(&par, 99.0),
            1e6 / mean(&par),
            cpu_par
        );
        std::fs::write(format!("{out_dir}/latency.csv"), &lat).unwrap();
    }

    // ---- config/metrics/memory ----
    let cfg = serde_json::json!({
        "experiment_family": "PH3C headsqual",
        "tier": tier, "data_dir": data_dir,
        "heads": loaded.heads, "dim": loaded.dim,
        "defining": loaded.defining,
        "pool_max": POOL_MAX, "budgets": BUDGETS, "train_k": TRAIN_K,
        "gating_protocol": "2B grid per seed; val-select at TRAIN_K; evaluated at all budgets",
        "splits": {"seed": 42, "counts": [tr, va, te]},
        "seeds": seeds,
        "parallel_workers": 2,
        "code_commit": option_env!("GIT_HASH").unwrap_or("unknown"),
        "hardware": {"ram_total_mb": 1984, "cpus": 2},
    });
    std::fs::write(
        format!("{out_dir}/config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .unwrap();
    let k100 = budget_rows.get(&TRAIN_K).unwrap();
    let g100 = k100
        .iter()
        .find(|(a, s, ..)| a == "trained_gating" && s == "agg")
        .unwrap();
    let metrics = serde_json::json!({
        "heads": loaded.heads, "dim": loaded.dim,
        "gating_r10_k100": g100.2.recall_at_10, "gating_r10_std": g100.3,
        "candidate_recall_by_k": cr_by_k,
        "peak_rss_mb": peak_build, "build_seconds": build_s, "pool_seconds": pool_s,
    });
    std::fs::write(
        format!("{out_dir}/metrics.json"),
        serde_json::to_string_pretty(&metrics).unwrap(),
    )
    .unwrap();
    let mut mem = String::from("metric,value\n");
    let _ = writeln!(mem, "peak_rss_mb,{peak_build:.1}");
    let _ = writeln!(mem, "rss_now_mb,{:.1}", rss_mb());
    let _ = writeln!(mem, "build_seconds,{build_s:.1}");
    let _ = writeln!(mem, "pool_generation_seconds,{pool_s:.1}");
    let _ = writeln!(
        mem,
        "raw_vector_mb,{:.1}",
        (loaded.n_docs * nh * loaded.dim * 4) as f64 / 1048576.0
    );
    std::fs::write(format!("{out_dir}/memory.csv"), &mem).unwrap();

    format!("headsqual {tier}: heads={nh} dim={} docs={} q={} (tr{tr}/va{va}/te{te}) build={build_s:.0}s\n\
             K=100 R@10: gating={:.4}±{:.4} | budgets+arms in results.csv",
        loaded.dim, loaded.n_docs, n_q, g100.2.recall_at_10, g100.3)
}
