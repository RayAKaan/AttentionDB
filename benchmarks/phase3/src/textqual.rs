//! PH3B `textquality` — complementary multi-field TEXT retrieval + BM25/hybrid
//! baselines on real data (AG News), frozen architecture, Phase 3B spec.
//!
//! Arms: global-best single head (val-selected) / uniform multi-head /
//! trained gating / BM25 (engine sparse channel) / hybrid RRF(BM25, full-view)
//! / engine attend_hybrid / empirical-oracle head / group-oracle head / exact
//! brute-force reference (anchor: must be R@10 = 1.0 or the run is invalid).
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Read as _;

use attentiondb_core::engine::AttentionEngine;
use attentiondb_learned::eval::{fuse_weighted, rank_metrics, RankMetrics};
use attentiondb_learned::gating_v2::{
    train_gating, DetRng, GatingDataset, GatingMlp, HeadExample, ModelCard, Objective,
    QualityTarget, QueryExample, Split, TrainOutcome, TrainingConfig, TrainingMeta,
};
use attentiondb_storage::{Durability, Record};

pub const DIM: usize = 512;
pub const HEADS: [&str; 3] = ["title", "body", "full"];
/// query types; index i == defining head index (title->title, body->body,
/// mixed->full — see datasets/build_agnews.py DEFINING map)
pub const TYPES: [&str; 3] = ["title", "body", "mixed"];
pub const POOL: usize = 100;
pub const GT_K: usize = 10;
pub const BM25_LIMIT: usize = 100;
/// pre-registered primary hybrid constant (Phase 2 convention rrf_k60)
pub const RRF_K: f64 = 60.0;

type HeadPoolsT = Vec<(Vec<u64>, Vec<f32>, Vec<f32>)>;

struct LoadedT {
    n_docs: usize,
    /// embedding dimension (from meta.json — 512 standard; 256 for the
    /// memory-constrained M20D256 probe)
    dim: usize,
    /// per head, row-major n_docs × dim
    corpus_heads: [Vec<f32>; 3],
    /// per query: 3 head vectors (order: title-block, body-block, mixed-block)
    queries: Vec<[Vec<f32>; 3]>,
    qtype: Vec<u8>,
    /// per type: query texts
    qtext: [Vec<String>; 3],
    /// per query: GT top-10 DOC INDICES, best first
    gt: Vec<Vec<u64>>,
    n_per_type: [usize; 3],
    file_hashes: HashMap<String, String>,
}

mod guard {
    pub struct TempDir(pub std::path::PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
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

fn std_of(rs: &[RankMetrics], pick: fn(&RankMetrics) -> f64) -> f64 {
    let n = rs.len() as f64;
    if rs.len() < 2 {
        return 0.0;
    }
    let mu = rs.iter().map(pick).sum::<f64>() / n;
    (rs.iter()
        .map(|r| {
            let d = pick(r) - mu;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0))
        .sqrt()
}

/// rank_metrics consumes slice ORDER as the ranking (HC-5): sort first.
fn rank_sorted(ranked: &[(u64, f32)], gt: &[u64], k: usize) -> RankMetrics {
    let mut v = ranked.to_vec();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    rank_metrics(&v, gt, k)
}

fn softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = v.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}

fn read_f32_vec(path: &str) -> Vec<f32> {
    let mut f = std::fs::File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    buf.chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i64_vec(path: &str) -> Vec<i64> {
    let mut f = std::fs::File::open(path).unwrap();
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    buf.chunks(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn load(data_dir: &str) -> LoadedT {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{data_dir}/meta.json")).unwrap())
            .unwrap();
    let n_docs = meta["corpus"]["n_docs"].as_u64().unwrap() as usize;
    let dim = meta["dim"].as_u64().unwrap_or(DIM as u64) as usize;
    let mut n_per_type = [0usize; 3];
    for (i, t) in TYPES.iter().enumerate() {
        n_per_type[i] = meta["queries"]["per_type"][t].as_u64().unwrap() as usize;
    }
    let mut corpus_heads: [Vec<f32>; 3] = Default::default();
    for (h, hname) in HEADS.iter().enumerate() {
        let v = read_f32_vec(&format!("{data_dir}/corpus_{hname}.f32"));
        assert_eq!(v.len(), n_docs * dim, "corpus_{hname} size mismatch");
        corpus_heads[h] = v;
    }
    let mut queries = Vec::new();
    let mut qtype = Vec::new();
    let mut qtext: [Vec<String>; 3] = Default::default();
    let mut gt = Vec::new();
    for (ti, t) in TYPES.iter().enumerate() {
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
        qtext[ti] = texts;
        let mut head_vecs: [Vec<f32>; 3] = Default::default();
        for (h, hname) in HEADS.iter().enumerate() {
            head_vecs[h] = read_f32_vec(&format!("{data_dir}/queries_{t}_{hname}.f32"));
        }
        let g = read_i64_vec(&format!("{data_dir}/gt_{t}.i64"));
        for q in 0..n_per_type[ti] {
            let row = [
                head_vecs[0][q * dim..(q + 1) * dim].to_vec(),
                head_vecs[1][q * dim..(q + 1) * dim].to_vec(),
                head_vecs[2][q * dim..(q + 1) * dim].to_vec(),
            ];
            queries.push(row);
            qtype.push(ti as u8);
            gt.push(
                g[q * GT_K..(q + 1) * GT_K]
                    .iter()
                    .map(|&x| x as u64)
                    .collect(),
            );
        }
    }
    let mut file_hashes = HashMap::new();
    if let Some(files) = meta["files"].as_object() {
        for (k, v) in files {
            file_hashes.insert(k.clone(), v.as_str().unwrap().to_string());
        }
    }
    LoadedT {
        n_docs,
        dim,
        corpus_heads,
        queries,
        qtype,
        qtext,
        gt,
        n_per_type,
        file_hashes,
    }
}

fn open_engine() -> (guard::TempDir, AttentionEngine) {
    // Engine dir backend matters in constrained sandboxes: /tmp may be
    // tmpfs (RAM-backed), so engine files silently consume memory budget.
    // Default: disk-backed /var/tmp. Set PH3B_ENGINE_TMPFS=1 to reproduce
    // the tmpfs environment of the original COMP-001 run (recorded in
    // config.json either way).
    let tmpfs = std::env::var("PH3B_ENGINE_TMPFS").map(|v| v == "1").unwrap_or(false);
    let base = if tmpfs {
        std::env::temp_dir()
    } else {
        std::path::PathBuf::from("/var/tmp")
    };
    let dir = base.join(format!("phase3b-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Async).unwrap();
    (guard::TempDir(dir), e)
}

struct PoolsT {
    /// per query: per head: (engine_ids, raw, minmax-norm)
    heads: Vec<HeadPoolsT>,
    idx_to_engine: Vec<u64>,
    engine_to_idx: HashMap<u64, usize>,
}

fn generate_pools(e: &AttentionEngine, loaded: &LoadedT) -> (PoolsT, f64) {
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
    let mut heads_pools: Vec<HeadPoolsT> = Vec::with_capacity(loaded.queries.len());
    for q in &loaded.queries {
        let mut per_head = Vec::with_capacity(HEADS.len());
        for (h, hname) in HEADS.iter().enumerate() {
            let hn = hname.to_string();
            let ranked: Vec<(u64, f32)> = coll
                .attend_detailed(
                    std::slice::from_ref(&hn),
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
            per_head.push((ranked.iter().map(|(id, _)| *id).collect(), raw, norm));
        }
        heads_pools.push(per_head);
    }
    let gen_s = t0.elapsed().as_secs_f64();
    (
        PoolsT {
            heads: heads_pools,
            idx_to_engine,
            engine_to_idx,
        },
        gen_s,
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

#[allow(clippy::needless_range_loop)]
fn build_gating_dataset(loaded: &LoadedT, pools: &PoolsT, split_seed: u64) -> GatingDataset {
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
            .map(|&i| pools.idx_to_engine[i as usize])
            .collect();
        let mut heads = Vec::with_capacity(HEADS.len());
        for (ids, raw, norm) in &pools.heads[qi] {
            let ranked: Vec<(u64, f32)> = ids.iter().copied().zip(raw.iter().copied()).collect();
            let m = rank_sorted(&ranked, &gt_engine, GT_K);
            heads.push(HeadExample {
                candidates: ids.clone(),
                raw_scores: raw.clone(),
                norm_scores: norm.clone(),
                exact_scores: raw.clone(),
                recall_at_k: m.recall_at_10 as f32,
                ndcg_at_k: m.ndcg_at_10 as f32,
                mrr: m.mrr as f32,
            });
        }
        let mut qvec = Vec::with_capacity(3 * loaded.dim);
        for h in 0..3 {
            qvec.extend_from_slice(&loaded.queries[qi][h]);
        }
        queries.push(QueryExample {
            query_id: qi as u64,
            query: qvec,
            query_group: Some(loaded.qtype[qi] as u32),
            split: split_of[qi],
            ground_truth: gt_engine,
            heads,
        });
    }
    GatingDataset {
        format: "attentiondb-gating-dataset".into(),
        version: 1,
        num_heads: 3,
        input_dim: 3 * loaded.dim,
        top_k: GT_K,
        corpus_desc: "PH3B-DS-AG agnews multi-field (real text)".into(),
        seed: split_seed,
        queries,
    }
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

/// exact cosine top-10 in `head` space; cosines rounded to 1e-4 with
/// round-ties-even (SAME rule as the numpy builder — HC-P3-4 parity), ties
/// by lower doc index. Ranks in DOC-INDEX space.
fn exact_ranking(loaded: &LoadedT, head: usize, q: &[f32]) -> Vec<(u64, f32)> {
    let qn: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    let dim = loaded.dim;
    let mut ranked: Vec<(u64, f32)> = Vec::with_capacity(loaded.n_docs);
    for i in 0..loaded.n_docs {
        let d = &loaded.corpus_heads[head][i * dim..(i + 1) * dim];
        let dot: f32 = q.iter().zip(d.iter()).map(|(a, b)| a * b).sum::<f32>() / qn;
        let r = (dot * 10_000.0).round_ties_even() / 10_000.0;
        ranked.push((i as u64, r));
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    ranked
}

fn rrf_fuse(lists: &[&[u64]], k: f64) -> Vec<(u64, f32)> {
    let mut acc: HashMap<u64, f64> = HashMap::new();
    for l in lists {
        for (rank, id) in l.iter().enumerate() {
            *acc.entry(*id).or_insert(0.0) += 1.0 / (k + rank as f64 + 1.0);
        }
    }
    acc.into_iter().map(|(id, s)| (id, s as f32)).collect()
}

fn empirical_oracle_head(ds: &GatingDataset, pools: &PoolsT, qi: usize) -> usize {
    // per-head pool rankings are already in head order; score = -(rank)
    let mut best = (0usize, -1.0f64, -1.0f64);
    for h in 0..3 {
        let (ids, _, _) = &pools.heads[qi][h];
        let ranked: Vec<(u64, f32)> = ids
            .iter()
            .copied()
            .enumerate()
            .map(|(r, id)| (id, -(r as f32)))
            .collect();
        let m = rank_metrics(&ranked, &ds.queries[qi].ground_truth, GT_K);
        if best.1 < 0.0 || (m.recall_at_10, m.ndcg_at_10) > (best.1, best.2) {
            best = (h, m.recall_at_10, m.ndcg_at_10);
        }
    }
    best.0
}

fn gap_recovered(gating: f64, uniform: f64, oracle: f64) -> serde_json::Value {
    let denom = oracle - uniform;
    if denom.abs() < 1e-12 {
        serde_json::Value::Null
    } else {
        serde_json::json!((gating - uniform) / denom)
    }
}

fn write_gating_diagnostics(
    ds: &GatingDataset,
    loaded: &LoadedT,
    pools: &PoolsT,
    m: &GatingMlp,
    t_fit: f32,
    out_dir: &str,
) {
    let mut w =
        String::from("qid,qtype,w_title,w_body,w_full,argmax_head,oracle_head,agree,entropy\n");
    let mut agree_ct = 0usize;
    let mut n = 0usize;
    for (qi, q) in ds.queries.iter().enumerate() {
        if q.split != Split::Test {
            continue;
        }
        let lg = m.logits(&q.query);
        let ws = softmax(&lg.iter().map(|x| x / t_fit).collect::<Vec<_>>());
        let arg = ws
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then(b.0.cmp(&a.0)))
            .unwrap()
            .0;
        let oracle = empirical_oracle_head(ds, pools, qi);
        let agree = (arg == oracle) as usize;
        agree_ct += agree;
        n += 1;
        let ent: f64 = -ws
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
        let _ = writeln!(
            w,
            "{},{},{:.4},{:.4},{:.4},{},{},{},{:.4}",
            qi,
            TYPES[loaded.qtype[qi] as usize],
            ws[0],
            ws[1],
            ws[2],
            HEADS[arg],
            HEADS[oracle],
            agree,
            ent
        );
    }
    let _ = writeln!(
        w,
        "# oracle_agreement_rate,{:.4}",
        agree_ct as f64 / n.max(1) as f64
    );
    std::fs::write(format!("{out_dir}/gating_weights_test.csv"), &w).unwrap();
}

fn emit_by_type(
    out: &mut String,
    name: &str,
    seed_label: &str,
    per_q: &[RankMetrics],
    test_qi: &[usize],
    loaded: &LoadedT,
) {
    for (ti, tname) in TYPES.iter().enumerate() {
        let rs: Vec<RankMetrics> = test_qi
            .iter()
            .filter(|&&qi| loaded.qtype[qi] as usize == ti)
            .map(|&qi| per_q[qi])
            .collect();
        let m = avg(&rs);
        let _ = writeln!(
            out,
            "{name},{seed_label},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}",
            tname,
            rs.len(),
            m.recall_at_1,
            m.recall_at_5,
            m.recall_at_10,
            m.ndcg_at_10,
            m.mrr
        );
    }
}

fn walk_size(p: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let md = e.metadata().unwrap();
            if md.is_dir() {
                total += walk_size(&e.path());
            } else {
                total += md.len();
            }
        }
    }
    total
}

/// per-query metrics of ONE gating model over test queries, indexed by qi
fn gating_per_query(ds: &GatingDataset, m: &GatingMlp, t: f32) -> Vec<RankMetrics> {
    ds.queries
        .iter()
        .map(|q| {
            if q.split != Split::Test {
                return RankMetrics {
                    recall_at_1: 0.0,
                    recall_at_5: 0.0,
                    recall_at_10: 0.0,
                    recall_at_50: 0.0,
                    ndcg_at_10: 0.0,
                    mrr: 0.0,
                };
            }
            let lg = m.logits(&q.query);
            let w = softmax(&lg.iter().map(|x| x / t).collect::<Vec<_>>());
            let ranked = fuse_weighted(q, &w);
            rank_sorted(&ranked, &q.ground_truth, ds.top_k)
        })
        .collect()
}

struct LatLog {
    stages: std::collections::BTreeMap<String, Vec<f64>>,
}

impl LatLog {
    fn new() -> Self {
        LatLog {
            stages: std::collections::BTreeMap::new(),
        }
    }
    fn add(&mut self, stage: &str, us: f64) {
        self.stages.entry(stage.to_string()).or_default().push(us);
    }
    fn get(&self, stage: &str) -> Vec<f64> {
        self.stages.get(stage).cloned().unwrap_or_default()
    }
    fn render(&self) -> String {
        let mut s = String::from("stage,n,p50_us,p95_us,p99_us,qps\n");
        for (name, v) in &self.stages {
            let p50 = percentile(v, 50.0);
            let qps = if name.ends_with("_path_total") && p50 > 0.0 {
                format!("{:.0}", 1e6 / (v.iter().sum::<f64>() / v.len() as f64))
            } else {
                String::new()
            };
            let _ = writeln!(
                s,
                "{name},{},{:.1},{:.1},{:.1},{qps}",
                v.len(),
                p50,
                percentile(v, 95.0),
                percentile(v, 99.0)
            );
        }
        s
    }
}

pub fn run(tier: &str, data_dir: &str, out_dir: &str, seeds: &[u64]) -> String {
    std::fs::create_dir_all(format!("{out_dir}/models")).unwrap();
    let loaded = load(data_dir);
    let n_docs = loaded.n_docs;
    let n_q = loaded.queries.len();
    let (_guard, e) = open_engine();
    e.create_collection("bench", loaded.dim, &HEADS).unwrap();

    // ---- build: vectors + TEXT fields (BM25 channel auto-populated from fields) ----
    let t_build = std::time::Instant::now();
    let corpus_text: Vec<(String, String)> = {
        let mut v = Vec::with_capacity(n_docs);
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
    assert_eq!(corpus_text.len(), n_docs);
    for (i, (title, desc)) in corpus_text.iter().enumerate() {
        let mut fields = std::collections::HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(i));
        fields.insert("title".to_string(), serde_json::json!(title));
        fields.insert("description".to_string(), serde_json::json!(desc));
        let mut r = Record::new(fields);
        for (h, hname) in HEADS.iter().enumerate() {
            let v = &loaded.corpus_heads[h][i * loaded.dim..(i + 1) * loaded.dim];
            r.k_vecs.insert(hname.to_string(), v.to_vec());
        }
        e.insert_document("bench", r).unwrap();
    }
    let build_s = t_build.elapsed().as_secs_f64();
    let rss_after_build = rss_mb();
    let peak_build = peak_rss_mb();
    let dir_size: u64 = walk_size(std::path::Path::new(&_guard.0));

    let (pools, pool_s) = generate_pools(&e, &loaded);
    let ds = build_gating_dataset(&loaded, &pools, 42);
    std::fs::write(
        format!("{out_dir}/ds.json"),
        serde_json::to_string(&ds).unwrap(),
    )
    .unwrap();
    let (tr, va, te) = ds.split_counts();
    let ds_hash = ds.content_hash();

    // ---- gating grid (seed seeds[0]; val-only selection + val temperature) ----
    let grid = train_gating_grid(&ds, seeds[0]);
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

    let test_qi: Vec<usize> = (0..n_q)
        .filter(|&i| ds.queries[i].split == Split::Test)
        .collect();
    let val_qi: Vec<usize> = (0..n_q)
        .filter(|&i| ds.queries[i].split == Split::Val)
        .collect();
    let coll = e.get_collection("bench").unwrap();

    // ---- A: global-best single head (head selected on VAL only, §5) ----
    let mut head_val = [0.0f64; 3];
    for (h, hv) in head_val.iter_mut().enumerate() {
        let rs: Vec<RankMetrics> = val_qi
            .iter()
            .map(|&qi| {
                let (ids, _, norm) = &pools.heads[qi][h];
                let ranked: Vec<(u64, f32)> =
                    ids.iter().copied().zip(norm.iter().copied()).collect();
                rank_sorted(&ranked, &ds.queries[qi].ground_truth, GT_K)
            })
            .collect();
        *hv = avg(&rs).recall_at_10;
    }
    let best_head = head_val
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then(b.0.cmp(&a.0)))
        .unwrap()
        .0;
    let per_q_global: Vec<RankMetrics> = (0..n_q)
        .map(|qi| {
            if ds.queries[qi].split != Split::Test {
                return RankMetrics {
                    recall_at_1: 0.0,
                    recall_at_5: 0.0,
                    recall_at_10: 0.0,
                    recall_at_50: 0.0,
                    ndcg_at_10: 0.0,
                    mrr: 0.0,
                };
            }
            let (ids, _, norm) = &pools.heads[qi][best_head];
            let ranked: Vec<(u64, f32)> = ids.iter().copied().zip(norm.iter().copied()).collect();
            rank_sorted(&ranked, &ds.queries[qi].ground_truth, GT_K)
        })
        .collect();

    // ALL-arm aggregation MUST be test-only (per_query zero-fills non-test)
    let test_avg = |v: &Vec<RankMetrics>| -> RankMetrics {
        let rs: Vec<RankMetrics> = test_qi.iter().map(|&qi| v[qi]).collect();
        avg(&rs)
    };
    // helper: run a per-query closure over ALL queries, zeroing non-test
    let per_query = |f: &dyn Fn(usize) -> RankMetrics| -> Vec<RankMetrics> {
        (0..n_q)
            .map(|qi| {
                if ds.queries[qi].split != Split::Test {
                    RankMetrics {
                        recall_at_1: 0.0,
                        recall_at_5: 0.0,
                        recall_at_10: 0.0,
                        recall_at_50: 0.0,
                        ndcg_at_10: 0.0,
                        mrr: 0.0,
                    }
                } else {
                    f(qi)
                }
            })
            .collect()
    };

    // ---- B: uniform multi-head ----
    let per_q_uniform = per_query(&|qi| {
        let ranked = fuse_weighted(&ds.queries[qi], &[1.0 / 3.0; 3]);
        rank_sorted(&ranked, &ds.queries[qi].ground_truth, GT_K)
    });

    // ---- D: BM25 (engine sparse channel) + top-10 dump for independent verify ----
    let mut bm25_top10_csv = String::from("qid,qtype,rank,doc_idx\n");
    let per_q_bm25 = per_query(&|qi| {
        let t = loaded.qtype[qi] as usize;
        let text = &loaded.qtext[t][row_of_type(&loaded, qi)];
        let got = coll.bm25_channel(text, BM25_LIMIT);
        rank_sorted(&got, &ds.queries[qi].ground_truth, GT_K)
    });
    {
        for (row, &qi) in test_qi.iter().enumerate() {
            let t = TYPES[loaded.qtype[qi] as usize];
            let text = loaded.qtext[loaded.qtype[qi] as usize][row_of_type(&loaded, qi)].clone();
            let got = coll.bm25_channel(&text, BM25_LIMIT);
            for (rank, (id, _)) in got.iter().take(10).enumerate() {
                let _ = writeln!(
                    bm25_top10_csv,
                    "{row},{t},{},{}",
                    rank + 1,
                    pools.engine_to_idx[id]
                );
            }
        }
    }
    std::fs::write(format!("{out_dir}/bm25_top10.csv"), &bm25_top10_csv).unwrap();

    // ---- E: hybrid RRF(BM25 top-100, FULL-head pool), k=60 pre-registered ----
    let per_q_hybrid = per_query(&|qi| {
        let t = loaded.qtype[qi] as usize;
        let text = &loaded.qtext[t][row_of_type(&loaded, qi)];
        let bm_ids: Vec<u64> = coll
            .bm25_channel(text, BM25_LIMIT)
            .iter()
            .map(|(id, _)| *id)
            .collect();
        let full_ids: Vec<u64> = pools.heads[qi][2].0.clone();
        let ranked = rrf_fuse(&[&bm_ids, &full_ids], RRF_K);
        rank_sorted(&ranked, &ds.queries[qi].ground_truth, GT_K)
    });

    // ---- E2: engine attend_hybrid (built-in RRF, dense side = full view) ----
    let per_q_hybrid_engine = per_query(&|qi| {
        let t = loaded.qtype[qi] as usize;
        let text = loaded.qtext[t][row_of_type(&loaded, qi)].clone();
        let got = e
            .attend_hybrid(
                "bench",
                &["full".to_string()],
                &loaded.queries[qi][2],
                &text,
                GT_K,
            )
            .unwrap();
        rank_sorted(&got, &ds.queries[qi].ground_truth, GT_K)
    });

    // ---- oracle arms (empirical per-query best head; group = defining head) ----
    let per_q_oracle_emp = per_query(&|qi| {
        let h = empirical_oracle_head(&ds, &pools, qi);
        let (ids, _, norm) = &pools.heads[qi][h];
        let ranked: Vec<(u64, f32)> = ids.iter().copied().zip(norm.iter().copied()).collect();
        rank_sorted(&ranked, &ds.queries[qi].ground_truth, GT_K)
    });
    let per_q_oracle_group = per_query(&|qi| {
        let h = loaded.qtype[qi] as usize;
        let (ids, _, norm) = &pools.heads[qi][h];
        let ranked: Vec<(u64, f32)> = ids.iter().copied().zip(norm.iter().copied()).collect();
        rank_sorted(&ranked, &ds.queries[qi].ground_truth, GT_K)
    });

    // ---- F: exact reference in the DEFINING head space (anchor) ----
    let t_exact0 = std::time::Instant::now();
    let per_q_exact = per_query(&|qi| {
        let h = loaded.qtype[qi] as usize;
        let ranked = exact_ranking(&loaded, h, &loaded.queries[qi][h]);
        rank_sorted(&ranked, &loaded.gt[qi], GT_K)
    });
    let exact_s = t_exact0.elapsed().as_secs_f64();
    let exact_test: Vec<RankMetrics> = test_qi.iter().map(|&qi| per_q_exact[qi]).collect();
    let exact_anchor = avg(&exact_test).recall_at_10;
    assert!(
        (exact_anchor - 1.0).abs() < 1e-9,
        "exact-reference anchor FAILED: R@10={exact_anchor} — GT/rounding parity broken (HC-class); run NOT valid for recording"
    );

    // ---- trained gating across seeds (test evaluated once per seed) ----
    let mut per_seed: Vec<(u64, String, RankMetrics)> = Vec::new();
    let mut per_seed_pq: Vec<(u64, Vec<RankMetrics>)> = Vec::new();
    for (si, &seed) in seeds.iter().enumerate() {
        let (obj, m, t, lr, hid): (Objective, GatingMlp, f32, f32, usize) = if si == 0 {
            (gobj, gm.clone(), gt_fit, glr, ghidden)
        } else {
            let g = train_gating_grid(&ds, seed);
            let mut best2: Option<(Objective, GatingMlp, f32, f32, usize)> = None;
            let mut best_r = -1.0f64;
            for (obj, m, _t, lr, hidden, _o) in &g {
                let t = fit_temperature(m, &ds);
                let r = eval_gating(m, &ds, t, Split::Val).recall_at_10;
                if r > best_r {
                    best_r = r;
                    best2 = Some((*obj, m.clone(), t, *lr, *hidden));
                }
            }
            best2.unwrap()
        };
        let r = eval_gating(&m, &ds, t, Split::Test);
        per_seed.push((seed, format!("gating({obj:?},lr={lr},h={hid},T={t})"), r));
        per_seed_pq.push((seed, gating_per_query(&ds, &m, t)));
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
                best_val_loss: gval as f32,
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
                ModelCard::from_mlp(&m, meta, &format!("ph3b-gating-{tier}-s{seed}"), "gating");
            card.save(std::path::Path::new(&format!(
                "{out_dir}/models/gating_s{seed}.json"
            )))
            .unwrap();
            write_gating_diagnostics(&ds, &loaded, &pools, &m, t, out_dir);
        }
    }
    let gate_metrics: Vec<RankMetrics> = per_seed.iter().map(|(_, _, m)| *m).collect();
    let gate_agg = avg(&gate_metrics);
    // per-query gating aggregate = mean over seeds (for by-type rows)
    let per_q_gate_agg: Vec<RankMetrics> = (0..n_q)
        .map(|qi| {
            let rs: Vec<RankMetrics> = per_seed_pq.iter().map(|(_, v)| v[qi]).collect();
            avg(&rs)
        })
        .collect();

    // ---- hybrid k sensitivity on VAL ONLY (never tuned on test) ----
    let mut hybrid_k_val = String::from("k,val_r10,val_ndcg,val_mrr\n");
    for &k in [20.0f64, 60.0, 120.0].iter() {
        let rs: Vec<RankMetrics> = val_qi
            .iter()
            .map(|&qi| {
                let t = loaded.qtype[qi] as usize;
                let text = &loaded.qtext[t][row_of_type(&loaded, qi)];
                let bm_ids: Vec<u64> = coll
                    .bm25_channel(text, BM25_LIMIT)
                    .iter()
                    .map(|(id, _)| *id)
                    .collect();
                let full_ids: Vec<u64> = pools.heads[qi][2].0.clone();
                let ranked = rrf_fuse(&[&bm_ids, &full_ids], k);
                rank_sorted(&ranked, &ds.queries[qi].ground_truth, GT_K)
            })
            .collect();
        let m = avg(&rs);
        let _ = writeln!(
            hybrid_k_val,
            "{k},{:.4},{:.4},{:.4}",
            m.recall_at_10, m.ndcg_at_10, m.mrr
        );
    }
    std::fs::write(format!("{out_dir}/hybrid_k_val.csv"), &hybrid_k_val).unwrap();

    // ---- candidate recall per type BEFORE fusion (§9 mandatory) ----
    let mut cr_csv =
        String::from("subset,gt_covered_frac,queries_full_coverage,avg_pool_candidates\n");
    let mut cr_by_type = [0.0f64; 3];
    for (ti, t) in TYPES.iter().enumerate() {
        let (mut covered, mut total, mut full_ct, mut cand_sum, mut n) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        let _ = t;
        for &qi in &test_qi {
            if loaded.qtype[qi] as usize != ti {
                continue;
            }
            let union: std::collections::HashSet<u64> = pools.heads[qi]
                .iter()
                .flat_map(|(ids, _, _)| ids.iter().copied())
                .collect();
            cand_sum += union.len();
            n += 1;
            let mut hit = 0;
            for g in &loaded.gt[qi] {
                if union.contains(&pools.idx_to_engine[*g as usize]) {
                    hit += 1;
                }
            }
            covered += hit;
            total += loaded.gt[qi].len();
            if hit == loaded.gt[qi].len() {
                full_ct += 1;
            }
        }
        let frac = covered as f64 / total.max(1) as f64;
        cr_by_type[ti] = frac;
        let _ = writeln!(
            cr_csv,
            "{t},{frac:.4},{full_ct},{:.1}",
            cand_sum as f64 / n.max(1) as f64
        );
    }
    {
        let (mut covered, mut total) = (0usize, 0usize);
        for &qi in &test_qi {
            let union: std::collections::HashSet<u64> = pools.heads[qi]
                .iter()
                .flat_map(|(ids, _, _)| ids.iter().copied())
                .collect();
            covered += loaded.gt[qi]
                .iter()
                .filter(|g| union.contains(&pools.idx_to_engine[**g as usize]))
                .count();
            total += loaded.gt[qi].len();
        }
        let _ = writeln!(cr_csv, "ALL,{:.4},,", covered as f64 / total.max(1) as f64);
    }
    std::fs::write(format!("{out_dir}/candidate_recall.csv"), &cr_csv).unwrap();

    // ---- latency (warm, 3 reps over test queries; frozen seed-42 path) ----
    let mut lat = LatLog::new();
    for _ in 0..3 {
        for &qi in &test_qi {
            let t = loaded.qtype[qi] as usize;
            let text = loaded.qtext[t][row_of_type(&loaded, qi)].clone();
            let mut ann_us = [0.0f64; 3];
            for h in 0..3 {
                let hn = HEADS[h].to_string();
                let t0 = std::time::Instant::now();
                let _ = coll
                    .attend_detailed(
                        std::slice::from_ref(&hn),
                        &loaded.queries[qi][h],
                        POOL,
                        None,
                        Some(attentiondb_core::collection::RetrievalMode::SingleHead),
                        None,
                        None,
                        None,
                    )
                    .unwrap();
                ann_us[h] = t0.elapsed().as_secs_f64() * 1e6;
            }
            lat.add("ann_head_title", ann_us[0]);
            lat.add("ann_head_body", ann_us[1]);
            lat.add("ann_head_full", ann_us[2]);
            let t0 = std::time::Instant::now();
            let lg = gm.logits(&ds.queries[qi].query);
            let w = softmax(&lg.iter().map(|x| x / gt_fit).collect::<Vec<_>>());
            let gate_us = t0.elapsed().as_secs_f64() * 1e6;
            lat.add("gating_forward", gate_us);
            let t0 = std::time::Instant::now();
            let ranked = fuse_weighted(&ds.queries[qi], &w);
            let _ = rank_sorted(&ranked, &ds.queries[qi].ground_truth, GT_K);
            let fuse_us = t0.elapsed().as_secs_f64() * 1e6;
            lat.add("fusion_rank", fuse_us);
            lat.add(
                "gating_path_total",
                ann_us.iter().sum::<f64>() + gate_us + fuse_us,
            );
            let t0 = std::time::Instant::now();
            let bm = coll.bm25_channel(&text, BM25_LIMIT);
            let bm_us = t0.elapsed().as_secs_f64() * 1e6;
            lat.add("bm25_query", bm_us);
            let t0 = std::time::Instant::now();
            let bm_ids: Vec<u64> = bm.iter().map(|(id, _)| *id).collect();
            let full_ids: Vec<u64> = pools.heads[qi][2].0.clone();
            let fused = rrf_fuse(&[&bm_ids, &full_ids], RRF_K);
            let _ = rank_sorted(&fused, &ds.queries[qi].ground_truth, GT_K);
            let rrf_us = t0.elapsed().as_secs_f64() * 1e6;
            lat.add("rrf_combine", rrf_us);
            lat.add("hybrid_path_total", ann_us[2] + bm_us + rrf_us);
            let t0 = std::time::Instant::now();
            let _ = exact_ranking(&loaded, t, &loaded.queries[qi][t]);
            lat.add("exact_bruteforce", t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    std::fs::write(format!("{out_dir}/latency.csv"), lat.render()).unwrap();

    // ---- memory / build diagnostics ----
    let raw_vec_mb = (n_docs * 3 * loaded.dim * 4) as f64 / (1024.0 * 1024.0);
    let mut mem_csv = String::from("metric,value\n");
    let _ = writeln!(mem_csv, "n_docs,{n_docs}");
    let _ = writeln!(mem_csv, "peak_rss_mb,{peak_build:.1}");
    let _ = writeln!(mem_csv, "rss_after_build_mb,{rss_after_build:.1}");
    let _ = writeln!(mem_csv, "raw_vector_mb,{raw_vec_mb:.1}");
    let _ = writeln!(
        mem_csv,
        "engine_multiplier_peak_over_raw,{:.2}",
        peak_build / raw_vec_mb
    );
    let _ = writeln!(
        mem_csv,
        "engine_dir_size_mb,{:.1}",
        dir_size as f64 / (1024.0 * 1024.0)
    );
    let _ = writeln!(mem_csv, "build_seconds,{build_s:.1}");
    let _ = writeln!(mem_csv, "pool_generation_seconds,{pool_s:.1}");
    let _ = writeln!(mem_csv, "exact_reference_seconds,{exact_s:.1}");
    std::fs::write(format!("{out_dir}/memory.csv"), &mem_csv).unwrap();

    // ---- results.csv (ALL) ----
    let mut rows: Vec<(String, String, RankMetrics, f64)> = Vec::new();
    rows.push((
        "global_best_single".into(),
        "agg".into(),
        test_avg(&per_q_global),
        0.0,
    ));
    rows.push((
        "uniform_multihead".into(),
        "agg".into(),
        test_avg(&per_q_uniform),
        0.0,
    ));
    for (seed, _, m) in &per_seed {
        rows.push(("trained_gating".into(), format!("{seed}"), *m, 0.0));
    }
    rows.push((
        "trained_gating".into(),
        "agg".into(),
        gate_agg,
        std_of(&gate_metrics, |m| m.recall_at_10),
    ));
    rows.push(("bm25".into(), "agg".into(), test_avg(&per_q_bm25), 0.0));
    rows.push((
        "hybrid_rrf_k60".into(),
        "agg".into(),
        test_avg(&per_q_hybrid),
        0.0,
    ));
    rows.push((
        "hybrid_engine".into(),
        "agg".into(),
        test_avg(&per_q_hybrid_engine),
        0.0,
    ));
    rows.push((
        "oracle_head_empirical".into(),
        "agg".into(),
        test_avg(&per_q_oracle_emp),
        0.0,
    ));
    rows.push((
        "oracle_head_group".into(),
        "agg".into(),
        test_avg(&per_q_oracle_group),
        0.0,
    ));
    rows.push((
        "exact_reference".into(),
        "agg".into(),
        test_avg(&per_q_exact),
        0.0,
    ));

    let mut csv = String::from("arm,seed,R@1,R@5,R@10,NDCG@10,MRR,R@10_std\n");
    for (arm, seed, m, sd) in &rows {
        let _ = writeln!(
            csv,
            "{arm},{seed},{:.4},{:.4},{:.4},{:.4},{:.4},{sd:.4}",
            m.recall_at_1, m.recall_at_5, m.recall_at_10, m.ndcg_at_10, m.mrr
        );
    }
    std::fs::write(format!("{out_dir}/results.csv"), &csv).unwrap();

    // ---- query_groups.csv (by query type; gating = per seed + seed-agg) ----
    let mut qg = String::from("arm,seed,qtype,n,R@1,R@5,R@10,NDCG@10,MRR\n");
    emit_by_type(
        &mut qg,
        "global_best_single",
        "agg",
        &per_q_global,
        &test_qi,
        &loaded,
    );
    emit_by_type(
        &mut qg,
        "uniform_multihead",
        "agg",
        &per_q_uniform,
        &test_qi,
        &loaded,
    );
    for (seed, pq) in &per_seed_pq {
        emit_by_type(
            &mut qg,
            "trained_gating",
            &format!("{seed}"),
            pq,
            &test_qi,
            &loaded,
        );
    }
    emit_by_type(
        &mut qg,
        "trained_gating",
        "agg",
        &per_q_gate_agg,
        &test_qi,
        &loaded,
    );
    emit_by_type(&mut qg, "bm25", "agg", &per_q_bm25, &test_qi, &loaded);
    emit_by_type(
        &mut qg,
        "hybrid_rrf_k60",
        "agg",
        &per_q_hybrid,
        &test_qi,
        &loaded,
    );
    emit_by_type(
        &mut qg,
        "hybrid_engine",
        "agg",
        &per_q_hybrid_engine,
        &test_qi,
        &loaded,
    );
    emit_by_type(
        &mut qg,
        "oracle_head_empirical",
        "agg",
        &per_q_oracle_emp,
        &test_qi,
        &loaded,
    );
    emit_by_type(
        &mut qg,
        "oracle_head_group",
        "agg",
        &per_q_oracle_group,
        &test_qi,
        &loaded,
    );
    emit_by_type(
        &mut qg,
        "exact_reference",
        "agg",
        &per_q_exact,
        &test_qi,
        &loaded,
    );
    std::fs::write(format!("{out_dir}/query_groups.csv"), &qg).unwrap();

    // ---- config.json ----
    let cfg = serde_json::json!({
        "experiment_family": "PH3B textquality (COMP/BM25/HYBRID)",
        "tier": tier,
        "data_dir": data_dir,
        "dataset_files_sha256": loaded.file_hashes,
        "corpus": {
            "n_docs": n_docs, "heads": HEADS, "dim": loaded.dim,
            "head_fields": {"title": "title", "body": "description", "full": "title + description"},
        },
        "queries": {
            "n_total": n_q, "per_type": loaded.n_per_type, "types": TYPES,
            "protocol": "one field at a time (title/body/mixed); each embedded in all 3 head spaces",
        },
        "harness": {
            "pool": POOL, "gt_k": GT_K, "bm25_limit": BM25_LIMIT, "rrf_k": RRF_K,
            "hybrid_dense_side": "full view",
            "normalization": "per-head minmax over pool scores (pipeline stage 3)",
            "hnsw": "engine defaults M=16 ef_construction=400 ef_search=64",
            "splits": {"seed": 42, "ratios": "0.70/0.15/0.15", "counts": [tr, va, te]},
            "gating_protocol": "2B grid: {QualityRegression,SoftTarget,Pairwise} x lr{0.01,0.003} x hidden{32,64}, val-select + val temperature {0.25,0.5,1,2}",
            "seeds": seeds,
            "exact_rounding": "1e-4 round-ties-even (numpy parity, HC-P3-4)",
            "bm25_channel": "engine Bm25Index over joined record text fields; engine tokenizer: lowercase, trim punctuation, stopword filter, porter stem (engine-internal, documented)",
        },
        "hardware": {
            "cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            "ram_total_mb": 1984,
            "engine_dir_backend": if std::env::var("PH3B_ENGINE_TMPFS").map(|v| v == "1").unwrap_or(false) {
                "tmpfs (/tmp, RAM-backed)"
            } else {
                "disk (/var/tmp)"
            },
        },
        "code_commit": option_env!("GIT_HASH").unwrap_or("unknown"),
    });
    std::fs::write(
        format!("{out_dir}/config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .unwrap();

    // ---- metrics.json ----
    let uniform_all = test_avg(&per_q_uniform);
    let oracle_all = test_avg(&per_q_oracle_emp);
    let metrics = serde_json::json!({
        "arms_ALL": rows.iter().map(|(a, s, m, sd)| serde_json::json!({
            "arm": a, "seed": s,
            "R@1": m.recall_at_1, "R@5": m.recall_at_5, "R@10": m.recall_at_10,
            "NDCG@10": m.ndcg_at_10, "MRR": m.mrr, "R@10_std": sd,
        })).collect::<Vec<_>>(),
        "candidate_recall_union_by_type": TYPES.iter().enumerate()
            .map(|(i, t)| serde_json::json!({"type": t, "frac": cr_by_type[i]})).collect::<Vec<_>>(),
        "gating_gap_recovered_overall": gap_recovered(gate_agg.recall_at_10, uniform_all.recall_at_10, oracle_all.recall_at_10),
        "global_best_head": HEADS[best_head],
        "head_val_r10": head_val.to_vec(),
        "split_counts": {"train": tr, "val": va, "test": te},
    });
    std::fs::write(
        format!("{out_dir}/metrics.json"),
        serde_json::to_string_pretty(&metrics).unwrap(),
    )
    .unwrap();

    let s = format!(
        "PH3B textquality {tier}: docs={n_docs} queries={n_q} (tr{tr}/va{va}/te{te}) build={build_s:.1}s\n\
         TEST R@10: gbest[{gb}]={gbv:.4} uniform={uv:.4} gating={gv:.4} bm25={bv:.4} rrf={rv:.4} hengine={hev:.4} exact={ev:.4} | cand-recall={cr:.4}\n\
         p50 µs: ann_full={af:.0} gating_fwd={gf:.0} bm25={bf:.0} | peak RSS {rss:.0} MB",
        gb = HEADS[best_head], gbv = test_avg(&per_q_global).recall_at_10,
        uv = uniform_all.recall_at_10, gv = gate_agg.recall_at_10,
        bv = test_avg(&per_q_bm25).recall_at_10, rv = test_avg(&per_q_hybrid).recall_at_10,
        hev = test_avg(&per_q_hybrid_engine).recall_at_10, ev = exact_anchor,
        cr = cr_by_type.iter().sum::<f64>() / 3.0,
        af = percentile(&lat.get("ann_head_full"), 50.0),
        gf = percentile(&lat.get("gating_forward"), 50.0),
        bf = percentile(&lat.get("bm25_query"), 50.0),
        rss = peak_build,
    );
    s
}

fn row_of_type(loaded: &LoadedT, qi: usize) -> usize {
    // position of query qi within its own contiguous type block
    let t = loaded.qtype[qi] as usize;
    let start: usize = loaded.n_per_type[..t].iter().sum();
    qi - start
}
