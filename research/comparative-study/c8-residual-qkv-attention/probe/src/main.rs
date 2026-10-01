//! c8pilot — C8 residual candidate-level attention run driver.
//!
//! C8 protocol arms:
//!   "A"  = C8-A: canonical single-head baseline (CANONICAL, C8 OFF)
//!   "B"  = C8-B: multi-head union baseline (TITLE/BODY/CITE, C8 OFF)
//!   "C"  = C8-C: truncated-identity attention machinery, lambda = 0 (control)
//!   "D"  = C8-D: unrestricted learned projection (W = dW, no residual base)
//!   "E"  = C8-E: learned residual projection (W = W0 + alpha*dW), no evidence
//!   "F"  = C8-F: C8-E + retrieval evidence
//!   "G"  = C8-G: C8-F + cross-head disagreement variance
//!   "H"  = C8-H: C8-E weights trained with KL distillation (inference = E)
//!   "I"  = C8-I: C8-E + in-memory document-side K/V cache
//!
//! Contract (C8 §1):
//!   S_final = S_base + residual_scale * dS_attention
//! `S_base` comes from the untouched retrieval pipeline (the same union used by
//! B), and `residual_scale` lives in the C8 config, never in the fusion weights.
//! Therefore lambda = 0 (arm C) reproduces arm B bit-for-bit for every returned
//! candidate, and any ranking difference in D/E/F/G/H/I is attributable to the
//! residual correction alone.
//!
//! Budget/shared settings across all arms (§C8-3):
//!   candidate_budget 500, min_candidates_per_head 20,
//!   max_candidates_per_head 300, ef_search 64 (every arm), k = 10,
//!   dim 384, H = 3 heads (A uses H = 1, CANONICAL only).
//!
//! Subcommands:
//!   run    — load the collection, run warmup + timed queries, emit per-query
//!            rows with C8 observability (per-candidate baseline, delta, applied
//!            correction, A_d / logits / output / entropy, mean entropy,
//!            mean|correction|, Spearman delta<->final) plus the ranked-union
//!            final-score ledger for cross-run rank-change vs B.
//!   train  — mine hard negatives with the deterministic stratified miner from
//!            the identical union (mode B, C8 OFF, ef 64, budget 500) on the
//!            VALIDATION split and train the C8 residual projection with
//!            `ResidualQkvTrainer` (contrastive + alpha_r*residual + beta*distill,
//!            real mini-batches). Emits a `C8ModelCard` JSON consumed by `run`.
//!
//! Input contract (single JSON, --config path):
//! {
//!   "subcommand": "run" | "train",
//!   "mode": "A|B|C|D|E|F|G|H|I",
//!   "k": 10, "seed": 20260925, "warmup": 20,
//!   "n_queries": 323, "n_docs": 3633, "dim": 384,
//!   "collection_heads": [...], "attend_heads": [...],
//!   "doc_vectors": {"<head>": "<raw LE f32 path>"},
//!   "query_vectors": "<raw LE f32 path>",
//!   "qrels": {"<row>": {"<docrow>": grade}},
//!   "subsample": [rows],
//!   "candidate_budget": 500, "min_candidates_per_head": 20,
//!   "max_candidates_per_head": 300, "ef_search": 64,
//!   "c8": {                            // C/D/E/F/G/H/I only
//!       "enabled": true,
//!       "arch": "truncated-identity" | "residual" | "unrestricted",
//!       "attention_dim": 384, "key_dim": 384, "value_dim": 384,
//!       "residual_scale": 0.1,
//!       "use_evidence": false,
//!       "scorer": {"w_attn": 1.0, "w_evidence": 0.0, "w_disagree": 0.0, "bias": 0.0},
//!       "cache": false,
//!       "model": ""                    // residual/unrestricted: C8 model card path
//!   },
//!   "fusion": {"attention": 0.3, "multi_head_similarity": 0.5, "bm25": 0.2},
//!   "emit_full_trace": false,
//!   "configuration_id": "...",
//!   "deadline_us": 0
//! }
//!
//! Train input adds a "train" block:
//!   "train": {
//!     "arm": "D"|"E"|"F"|"G"|"H",
//!     "epochs": 5, "learning_rate": 1e-2, "temperature": 0.07, "l2": 1e-4,
//!     "seed": 20260925, "batch_size": 8, "negatives_per_query": 8,
//!     "residual_scale": 0.1, "residual_alpha": 0.1,
//!     "residual_regularization": 0.0, "distillation_weight": 0.0,
//!     "distillation_temperature": 0.5, "use_distillation": false,
//!     "use_evidence": false,
//!     "rows": [...],
//!     "model_id": "scifact-c8-e", "arch": "residual-qkv"
//!   }
//!
//! Emits one JSON doc on stdout (and --out).

use attentiondb_attention::{
    AlignmentProjection, C8AttentionConfig, C8ModelCard, C8QkvDatasetBuilder, C8QkvDataset,
    C8QkvExample, C8TrainingConfig, HardNegativeConfig, HardNegativeMiner, PoolEntry,
    ResidualQkvProjection, ResidualQkvTrainer, ResidualScorer, RetrievalEvidence,
};
use attentiondb_core::{collection::RetrievalMode, retrieval::FusionWeights, AttentionEngine};
use attentiondb_storage::{Durability, Record};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Instant;

fn read_f32(path: &str, n: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(bytes.len(), n * 4, "file {path} len {} != {n}*4", bytes.len());
    let mut out = Vec::with_capacity(n);
    for ch in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = na.sqrt() * nb.sqrt();
    if d == 0.0 {
        0.0
    } else {
        dot / d
    }
}

fn brute_topk(vecs: &[(u64, Vec<f32>)], q: &[f32], k: usize) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = vecs.iter().map(|(id, v)| (*id, cosine(q, v))).collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(k.min(scored.len()));
    scored
}

fn ndcg10(retrieved: &[u64], relevant: &HashMap<u64, f32>, k: usize) -> f32 {
    let mut dcg = 0.0f32;
    for (rank, id) in retrieved.iter().enumerate() {
        if rank >= k {
            break;
        }
        let rel = relevant.get(id).copied().unwrap_or(0.0);
        dcg += rel / (rank as f32 + 2.0).log2();
    }
    let mut ideal: Vec<f32> = relevant.values().copied().collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let mut idcg = 0.0f32;
    for (rank, rel) in ideal.iter().enumerate() {
        if rank >= k {
            break;
        }
        if *rel <= 0.0 {
            continue;
        }
        idcg += rel / (rank as f32 + 2.0).log2();
    }
    if idcg > 0.0 {
        dcg / idcg
    } else {
        0.0
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Deterministic LCG for query order shuffling.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0xA0761D6478BD642F);
        let mut x = self.0;
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51AFD7ED558CCD);
        x ^= x >> 33;
        x
    }
    fn usize_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut rng = Lcg::new(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.usize_below(i + 1);
        idx.swap(i, j);
    }
    idx
}

/// Spearman rank correlation over the common rows of two (row -> score) maps.
fn spearman_on_shared(a: &HashMap<u64, f32>, b: &HashMap<u64, f32>) -> f32 {
    let rows: Vec<u64> = a.keys().filter(|r| b.contains_key(r)).copied().collect();
    if rows.len() < 3 {
        return 0.0;
    }
    let mut av: Vec<(u64, f32)> = rows.iter().map(|&r| (r, *a.get(&r).unwrap())).collect();
    let mut bv: Vec<(u64, f32)> = rows.iter().map(|&r| (r, *b.get(&r).unwrap())).collect();
    av.sort_by(|x, y| x.1.total_cmp(&y.1));
    bv.sort_by(|x, y| x.1.total_cmp(&y.1));
    let mut ra: HashMap<u64, f64> = HashMap::new();
    let mut rb: HashMap<u64, f64> = HashMap::new();
    for (i, (r, _)) in av.iter().enumerate() {
        ra.insert(*r, i as f64 + 1.0);
    }
    for (i, (r, _)) in bv.iter().enumerate() {
        rb.insert(*r, i as f64 + 1.0);
    }
    let arr: Vec<(f64, f64)> = rows.iter().map(|&r| (ra[&r], rb[&r])).collect();
    let n = arr.len() as f64;
    let ma = arr.iter().map(|p| p.0).sum::<f64>() / n;
    let mb = arr.iter().map(|p| p.1).sum::<f64>() / n;
    let mut cov = 0.0f64;
    let mut va = 0.0f64;
    let mut vb = 0.0f64;
    for &(x, y) in &arr {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    let d = (va * vb).sqrt();
    if d == 0.0 {
        0.0
    } else {
        (cov / d) as f32
    }
}

fn parse_fusion(v: &Value) -> (FusionWeights, String) {
    if let Some(f) = v.get("fusion").and_then(|x| x.as_object()) {
        let atn = f.get("attention").and_then(|x| x.as_f64()).unwrap_or(0.3) as f32;
        let mhs = f
            .get("multi_head_similarity")
            .and_then(|x| x.as_f64())
            .unwrap_or(0.5) as f32;
        let bm = f.get("bm25").and_then(|x| x.as_f64()).unwrap_or(0.2) as f32;
        let w = FusionWeights {
            attention: atn,
            multi_head_similarity: mhs,
            bm25: bm,
        };
        w.validate().expect("invalid fusion weights");
        let label = format!("a:{atn:.4},mhs:{mhs:.4},bm25:{bm:.4}");
        (w, label)
    } else {
        let w = FusionWeights {
            attention: 0.3,
            multi_head_similarity: 0.5,
            bm25: 0.2,
        };
        (w, "default(0.3,0.5,0.2)".into())
    }
}

fn parse_scorer(v: &Value) -> ResidualScorer {
    if let Some(s) = v.get("scorer").and_then(|x| x.as_object()) {
        let wa = s.get("w_attn").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32;
        let we = s.get("w_evidence").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
        let wd = s.get("w_disagree").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
        let bias = s.get("bias").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
        ResidualScorer::with_disagreement(wa, we, wd, bias)
    } else {
        ResidualScorer::default()
    }
}

fn load_projection(path: &str) -> ResidualQkvProjection {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read c8 model {path}: {e}"));
    let card: Value = serde_json::from_str(&txt).expect("c8 model JSON");
    serde_json::from_value(card.get("residual_projection").cloned().expect("residual_projection"))
        .expect("residual projection")
}

/// Build a `C8AttentionConfig` for an arm from its JSON block.
fn build_c8_cfg(v: &Value, heads: &[String]) -> Option<C8AttentionConfig> {
    let c8 = v.get("c8")?;
    if !c8.get("enabled").and_then(|e| e.as_bool()).unwrap_or(false) {
        return None;
    }
    let d_a = c8.get("attention_dim").and_then(|v| v.as_u64()).unwrap_or(384) as usize;
    let d_k = c8.get("key_dim").and_then(|v| v.as_u64()).unwrap_or(384) as usize;
    let d_v = c8.get("value_dim").and_then(|v| v.as_u64()).unwrap_or(384) as usize;
    let use_evidence = c8
        .get("use_evidence")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let scale = c8
        .get("residual_scale")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.1) as f32;
    let cache = c8.get("cache").and_then(|v| v.as_bool()).unwrap_or(false);
    let scorer = parse_scorer(c8);
    let aligns: Vec<AlignmentProjection> =
        heads.iter().map(|_| AlignmentProjection::identity(d_a)).collect();
    match c8.get("arch").and_then(|a| a.as_str()) {
        Some("truncated-identity") | None => {
            let mut c = C8AttentionConfig::truncated_identity_control(heads.len(), d_a, d_k, d_v);
            c.use_evidence = use_evidence;
            c.scorer = scorer;
            c.cache_enabled = cache;
            c.residual_scale = scale;
            Some(c)
        }
        Some("residual") | Some("unrestricted") => {
            let path = c8.get("model").and_then(|p| p.as_str()).unwrap_or("");
            assert!(!path.is_empty(), "c8.model required for learned arms");
            let proj = load_projection(path);
            let mut c = C8AttentionConfig::residual(
                d_a,
                d_k,
                d_v,
                aligns,
                AlignmentProjection::identity(d_a),
                proj,
                scale,
                use_evidence,
                scorer,
            );
            c.cache_enabled = cache;
            Some(c)
        }
        other => panic!("unknown c8.arch: {other:?}"),
    }
}

fn build_engine(
    n_docs: usize,
    dim: usize,
    doc_vecs: &HashMap<String, Vec<f32>>,
    mode: &str,
    heads: &[&str],
) -> (AttentionEngine, HashMap<u64, usize>) {
    let dir = std::env::temp_dir().join(format!("c8pilot-{}-{}", std::process::id(), mode));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Async)
        .unwrap_or_else(|err| panic!("open engine: {err}"));
    e.create_collection("c8p", dim, heads).unwrap();
    let mut doc_map: HashMap<u64, usize> = HashMap::new();
    for row in 0..n_docs {
        let mut fields: HashMap<String, Value> = HashMap::new();
        fields.insert("idx".to_string(), json!(row));
        let mut rec = Record::new(fields);
        for h in heads {
            let vec = doc_vecs
                .get(*h)
                .unwrap_or_else(|| panic!("missing doc head {h}"));
            rec.k_vecs
                .insert((*h).to_string(), vec[row * dim..(row + 1) * dim].to_vec());
        }
        let uuid = e.insert_document("c8p", rec).unwrap();
        let nid = e
            .id_mapper
            .read()
            .uuid_to_id(&uuid.parse().unwrap())
            .unwrap();
        doc_map.insert(nid, row);
    }
    (e, doc_map)
}

/// Per-arm execution contract for a shared-engine multi-arm `run`.
#[derive(Clone)]
struct ArmSpec {
    name: String,
    mode: String,
    ef_search: Option<usize>,
    search_k: Option<usize>,
    attend_heads: Vec<String>,
    c8: Option<C8AttentionConfig>,
    fingerprint: Option<u64>,
    fusion: FusionWeights,
    fusion_label: String,
    config_id: String,
    emit_full_trace: bool,
}

fn parse_arm_spec(v: &Value, default_heads: &[String]) -> ArmSpec {
    let mode = v["mode"].as_str().unwrap_or("B").to_string();
    let name = v["name"].as_str().unwrap_or(&mode).to_string();
    let ef_search = v["ef_search"].as_u64().map(|e| e as usize);
    let search_k = v["search_k"].as_u64().map(|e| e as usize);
    let attend_heads: Vec<String> = if let Some(arr) = v["attend_heads"].as_array() {
        arr.iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect()
    } else {
        default_heads.to_vec()
    };
    let c8 = build_c8_cfg(v, &attend_heads);
    let fingerprint = c8.as_ref().map(C8AttentionConfig::fingerprint);
    let (fusion, fusion_label) = parse_fusion(v);
    let config_id = v["configuration_id"].as_str().unwrap_or("").to_string();
    let emit_full_trace = v["emit_full_trace"].as_bool().unwrap_or(false);
    ArmSpec {
        name,
        mode,
        ef_search,
        search_k,
        attend_heads,
        c8,
        fingerprint,
        fusion,
        fusion_label,
        config_id,
        emit_full_trace,
    }
}

fn rsub_run(cfg: &Value, out_path: Option<&str>) -> (Value, bool) {
    let k = cfg["k"].as_u64().unwrap_or(10) as usize;
    let seed = cfg["seed"].as_u64().unwrap_or(20260925);
    let warmup = cfg["warmup"].as_u64().unwrap_or(20) as usize;
    let n_docs = cfg["n_docs"].as_u64().unwrap() as usize;
    let n_queries = cfg["n_queries"].as_u64().unwrap() as usize;
    let dim = cfg["dim"].as_u64().unwrap() as usize;

    let cand_budget = cfg["candidate_budget"].as_u64().unwrap_or(500) as usize;
    let min_cand_head = cfg["min_candidates_per_head"].as_u64().unwrap_or(20) as usize;
    let ef_search = cfg["ef_search"].as_u64().unwrap_or(64) as usize;
    let base_config_id = cfg["configuration_id"].as_str().unwrap_or("").to_string();
    let deadline_us = cfg["deadline_us"].as_u64().unwrap_or(0) as u64;
    let base_emit_full_trace = cfg["emit_full_trace"].as_bool().unwrap_or(false);

    let default_heads: Vec<String> = cfg["attend_heads"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|x| x.as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default();
    let arms: Vec<ArmSpec> = if let Some(arr) = cfg["arms"].as_array() {
        arr.iter().map(|a| parse_arm_spec(a, &default_heads)).collect()
    } else {
        let mode = cfg["mode"].as_str().unwrap_or("B").to_string();
        if !matches!(
            mode.as_str(),
            "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I"
        ) {
            panic!("mode must be A|B|C|D|E|F|G|H|I, got {mode}");
        }
        let c8 = if matches!(mode.as_str(), "C" | "D" | "E" | "F" | "G" | "H" | "I") {
            build_c8_cfg(cfg, &default_heads)
        } else {
            None
        };
        let (fusion, fusion_label) = parse_fusion(cfg);
        vec![ArmSpec {
            name: mode.clone(),
            mode: mode.clone(),
            ef_search: Some(ef_search),
            search_k: cfg["search_k"].as_u64().map(|e| e as usize),
            attend_heads: default_heads.clone(),
            fingerprint: c8.as_ref().map(C8AttentionConfig::fingerprint),
            c8,
            fusion,
            fusion_label,
            config_id: base_config_id.clone(),
            emit_full_trace: base_emit_full_trace,
        }]
    };
    let multi_arm = cfg["arms"].as_array().is_some();

    // ---- load doc vectors per head -------------------------------------
    let mut doc_vecs: HashMap<String, Vec<f32>> = HashMap::new();
    let mut canonical: Vec<f32> = Vec::new();
    if let Some(map) = cfg["doc_vectors"].as_object() {
        for (head, path) in map {
            let p = path.as_str().unwrap();
            let v = read_f32(p, n_docs * dim);
            if head == "CANONICAL" {
                canonical = v.clone();
            }
            doc_vecs.insert(head.clone(), v);
        }
    }
    let qflat = read_f32(cfg["query_vectors"].as_str().unwrap(), n_queries * dim);
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(n_queries);
    for r in 0..n_queries {
        queries.push(qflat[r * dim..(r + 1) * dim].to_vec());
    }
    for q in &queries {
        for x in q {
            assert!(x.is_finite(), "non-finite query value");
        }
    }

    // ---- qrels ----------------------------------------------------------
    let mut qrels: HashMap<usize, HashMap<u64, f32>> = HashMap::new();
    if let Some(qr) = cfg["qrels"].as_object() {
        for (qrow, rels) in qr {
            let qr_i = qrow.parse::<usize>().unwrap();
            let mut m = HashMap::new();
            if let Some(arr) = rels.as_array() {
                for v in arr {
                    if let Some(ridx) = v.as_u64() {
                        m.insert(ridx, 1.0);
                    }
                }
            } else if let Some(obj) = rels.as_object() {
                for (drow, grade) in obj {
                    m.insert(drow.parse::<u64>().unwrap(), grade.as_f64().unwrap() as f32);
                }
            }
            qrels.insert(qr_i, m);
        }
    }

    // ---- run set + seeded order ----------------------------------------
    let run_rows: Vec<usize> = if let Some(sub) = cfg["subsample"].as_array() {
        sub.iter().map(|v| v.as_u64().unwrap() as usize).collect()
    } else {
        (0..n_queries).collect()
    };
    let order = shuffled_indices(run_rows.len(), seed);
    let query_order: Vec<usize> = order.iter().map(|&o| run_rows[o]).collect();

    // ---- exact oracle (CANONICAL top-k) --------------------------------
    let vlist: Vec<(u64, Vec<f32>)> = if canonical.is_empty() {
        Vec::new()
    } else {
        (0..n_docs)
            .map(|d| (d as u64, canonical[d * dim..(d + 1) * dim].to_vec()))
            .collect()
    };

    // ---- shared engine: ONE collection/graph, all arms reuse it --------
    let engine_heads: Vec<String> = {
        let mut h: Vec<String> = Vec::new();
        for arm in &arms {
            for hh in &arm.attend_heads {
                if !h.contains(hh) {
                    h.push(hh.clone());
                }
            }
        }
        h
    };
    let heads_refs: Vec<&str> = engine_heads.iter().map(|s| s.as_str()).collect();
    let (e, doc_map) = build_engine(n_docs, dim, &doc_vecs, "M", &heads_refs);
    let coll = e.get_collection("c8p").unwrap();

    let max_cand_head = cfg["max_candidates_per_head"].as_u64().unwrap_or(300) as usize;

    // ---- run query protocol: warmup + timed, one pass per arm ----------
    let mut all_exact: Vec<Vec<u64>> = Vec::new();
    struct ArmRun {
        arm: ArmSpec,
        per_query: Vec<Value>,
        latencies: Vec<f64>,
        trace_count: usize,
        ent_sum: f32,
        spearman_sum: f32,
        corr_abs_sum: f32,
        per_head_accum: Vec<f64>,
        deadline_misses: usize,
    }
    let mut runs: Vec<ArmRun> = arms
        .iter()
        .map(|a| ArmRun {
            arm: a.clone(),
            per_query: Vec::new(),
            latencies: Vec::new(),
            trace_count: 0,
            ent_sum: 0.0,
            spearman_sum: 0.0,
            corr_abs_sum: 0.0,
            per_head_accum: vec![0.0; a.attend_heads.len()],
            deadline_misses: 0,
        })
        .collect();

    for (pos, &qr) in query_order.iter().enumerate() {
        let q = &queries[qr];
        let exact = if vlist.is_empty() {
            Vec::new()
        } else {
            brute_topk(&vlist, q, k)
        };
        let ids: Vec<u64> = exact.iter().map(|(id, _)| *id).collect();
        all_exact.push(ids.clone());
        let relevant = qrels.get(&qr).cloned().unwrap_or_default();
        let rel_set: HashSet<u64> = relevant.keys().copied().collect();
        let t = Instant::now();
        let deadline = if deadline_us > 0 && pos >= warmup {
            Some(Instant::now() + std::time::Duration::from_micros(deadline_us))
        } else {
            None
        };

        for run in runs.iter_mut() {
            let arm_cfg = {
                let mut rc = coll.retrieval_config.read().clone();
                rc.mode = if run.arm.mode == "A" {
                    RetrievalMode::SingleHead
                } else {
                    RetrievalMode::FixedFusion
                };
                rc.candidate_budget = cand_budget;
                rc.min_candidates_per_head = min_cand_head;
                rc.ef_search = Some(run.arm.ef_search.unwrap_or(ef_search));
                rc.search_k = run.arm.search_k;
                rc.cross_refine = None;
                rc.adaptive = None;
                rc.fusion = run.arm.fusion.clone();
                rc.attention = None;
                rc.c8_attention = run.arm.c8.clone();
                rc
            };
            let mode_override = if run.arm.mode == "A" {
                Some(RetrievalMode::SingleHead)
            } else {
                Some(RetrievalMode::FixedFusion)
            };

            let arm_t = Instant::now();
            match coll.attend_detailed_c8(
                &run.arm.attend_heads,
                q,
                cand_budget,
                None,
                mode_override,
                None,
                Some(&arm_cfg),
                deadline,
            ) {
                Ok((res, stats, _cross, _adaptive, c8_trace)) => {
                    let lat_us = arm_t.elapsed().as_secs_f64() * 1e6;
                    if pos >= warmup {
                        run.latencies.push(lat_us);
                    }
                    let hits: Vec<u64> = res.iter().take(k).map(|r| r.id).collect();
                    let hits_rows: Vec<u64> = hits
                        .iter()
                        .map(|h| doc_map.get(h).copied().unwrap_or(*h as usize) as u64)
                        .collect();
                    let hits_cnt = hits_rows.iter().filter(|id| rel_set.contains(id)).count();
                    let recall = if rel_set.is_empty() {
                        0.0
                    } else {
                        hits_cnt as f32 / rel_set.len() as f32
                    };
                    let recall_exact = if ids.is_empty() {
                        0.0
                    } else {
                        let exact_set: HashSet<u64> = ids.iter().copied().collect();
                        hits_rows
                            .iter()
                            .filter(|r| exact_set.contains(r))
                            .count() as f32
                            / ids.len() as f32
                    };
                    let ndcg = if rel_set.is_empty() {
                        0.0
                    } else {
                        ndcg10(&hits_rows, &relevant, k)
                    };

                    // ---- C8 observability ------------------------------
                    let mut rank_ledger: Vec<Value> = Vec::with_capacity(res.len());
                    let mut final_by_row: HashMap<u64, f32> = HashMap::new();
                    for (rank, r) in res.iter().enumerate() {
                        let row = doc_map.get(&r.id).copied().unwrap_or(r.id as usize) as u64;
                        let hs: Vec<Value> = r
                            .features
                            .head_scores
                            .iter()
                            .map(|s| match s {
                                Some(v) => json!(v),
                                None => Value::Null,
                            })
                            .collect();
                        final_by_row.insert(row, r.final_score);
                        rank_ledger.push(json!({
                            "row": row, "rank": rank, "final": r.final_score,
                            "mhs": r.features.multi_head_similarity,
                            "head_sims": hs,
                        }));
                    }

                    let trace_row = if let Some(trace) = &c8_trace {
                        let rows: Vec<Value> = trace
                            .candidates
                            .iter()
                            .map(|c| {
                                let row = doc_map.get(&c.id).copied().unwrap_or(c.id as usize) as u64;
                                let mut obj = serde_json::Map::new();
                                obj.insert("row".into(), json!(row));
                                obj.insert("baseline_score".into(), json!(c.baseline_score));
                                obj.insert("attention_delta".into(), json!(c.attention_delta));
                                obj.insert("residual_scale".into(), json!(c.residual_scale));
                                obj.insert("applied_correction".into(), json!(c.applied_correction));
                                obj.insert("final_score".into(), json!(c.final_score));
                                obj.insert("entropy".into(), json!(c.entropy));
                                if run.arm.emit_full_trace {
                                    obj.insert("weights".into(), json!(c.weights));
                                    obj.insert("logits".into(), json!(c.logits));
                                    obj.insert("output".into(), json!(c.output));
                                    obj.insert(
                                        "weights_sum".into(),
                                        json!(c.weights.iter().sum::<f32>()),
                                    );
                                }
                                json!(obj)
                            })
                            .collect();
                        // Spearman(delta, final) and Spearman(delta, baseline) over
                        // shared rows. delta==0 (arm C) is a constant series -> 0.0.
                        let mut delta_by_row: HashMap<u64, f32> = HashMap::new();
                        for c in &trace.candidates {
                            let row = doc_map.get(&c.id).copied().unwrap_or(c.id as usize) as u64;
                            delta_by_row.insert(row, c.attention_delta);
                        }
                        let base_by_row: HashMap<u64, f32> = trace
                            .candidates
                            .iter()
                            .map(|c| {
                                (
                                    doc_map.get(&c.id).copied().unwrap_or(c.id as usize) as u64,
                                    c.baseline_score,
                                )
                            })
                            .collect();
                        let sp = spearman_on_shared(&delta_by_row, &final_by_row);
                        let sp_base = spearman_on_shared(&delta_by_row, &base_by_row);
                        run.spearman_sum += sp;
                        run.corr_abs_sum += sp_base.abs();
                        run.trace_count += 1;
                        for (i, ph) in trace.per_head_mean.iter().enumerate() {
                            if i < run.per_head_accum.len() {
                                run.per_head_accum[i] += *ph as f64;
                            }
                        }
                        run.ent_sum += trace.mean_entropy;
                        json!({
                            "head_names": trace.head_names,
                            "n_candidates": trace.candidates.len(),
                            "per_head_mean": trace.per_head_mean,
                            "mean_entropy": trace.mean_entropy,
                            "mean_abs_correction": trace.mean_abs_correction,
                            "timings": {
                                "query_alignment_us": trace.timings.query_alignment_us,
                                "candidate_alignment_us": trace.timings.candidate_alignment_us,
                                "q_projection_us": trace.timings.q_projection_us,
                                "kv_projection_us": trace.timings.kv_projection_us,
                                "cache_lookup_us": trace.timings.cache_lookup_us,
                                "attention_us": trace.timings.attention_us,
                                "residual_fusion_us": trace.timings.residual_fusion_us,
                                "total_us": trace.timings.total_us,
                            },
                            "cache": {
                                "hits": trace.cache_stats.hits,
                                "misses": trace.cache_stats.misses,
                                "entries": trace.cache_stats.entries,
                            },
                            "config_fingerprint": trace.config_fingerprint,
                            "candidates": rows,
                            "spearman_delta_final": sp,
                            "spearman_delta_baseline": sp_base,
                        })
                    } else {
                        Value::Null
                    };

                    run.per_query.push(json!({
                        "pos": pos, "query_row": qr,
                        "latency_us": lat_us,
                        "configuration_id": run.arm.config_id,
                        "candidate_count": stats.union_size,
                        "heads_present": stats.heads_present,
                        "recall10_qrels": recall, "recall10_exact": recall_exact, "ndcg10_qrels": ndcg,
                        "hits": hits_rows, "engine_hits": hits, "n_rel": rel_set.len(),
                        "relevant_ids": rel_set.iter().cloned().collect::<Vec<_>>(),
                        "c8_enabled": run.arm.c8.is_some(),
                        "c8_fingerprint": run.arm.fingerprint,
                        "fusion_label": run.arm.fusion_label,
                        "union_ledger": rank_ledger,
                        "c8_trace": trace_row,
                    }));
                }
                Err(err) => {
                    let err_str = err.to_string();
                    if err_str.contains("deadline") {
                        run.deadline_misses += 1;
                        run.per_query.push(json!({
                            "pos": pos, "query_row": qr,
                            "latency_us": t.elapsed().as_secs_f64() * 1e6,
                            "configuration_id": run.arm.config_id,
                            "deadline_exceeded": true, "error": err_str,
                            "recall10_qrels": 0.0, "recall10_exact": 0.0, "ndcg10_qrels": 0.0,
                            "hits": [], "engine_hits": [], "n_rel": rel_set.len(),
                            "relevant_ids": [], "c8_enabled": run.arm.c8.is_some(),
                            "c8_fingerprint": run.arm.fingerprint, "fusion_label": run.arm.fusion_label,
                            "union_ledger": [], "c8_trace": null,
                        }));
                    } else {
                        panic!("attend error: {err}");
                    }
                }
            }
        }
    }

    // ---- per-arm aggregation + output ---------------------------------
    let build_arm_output = |run: &ArmRun, per_query: Vec<Value>| -> Value {
        let mut slat = run.latencies.clone();
        slat.sort_by(|a, b| a.total_cmp(b));
        let mean = if slat.is_empty() {
            0.0
        } else {
            slat.iter().sum::<f64>() / slat.len() as f64
        };
        let rec_qrels: Vec<f32> = per_query
            .iter()
            .map(|r| r["recall10_qrels"].as_f64().unwrap() as f32)
            .collect();
        let rec_exact: Vec<f32> = per_query
            .iter()
            .map(|r| r["recall10_exact"].as_f64().unwrap() as f32)
            .collect();
        let mean_qrels = rec_qrels.iter().sum::<f32>() / rec_qrels.len().max(1) as f32;
        let mean_exact = rec_exact.iter().sum::<f32>() / rec_exact.len().max(1) as f32;
        let c8_aggregate = if run.trace_count > 0 {
            json!({
                "queries_with_trace": run.trace_count,
                "mean_entropy_over_queries": run.ent_sum / run.trace_count as f32,
                "per_head_mean_over_queries":
                    run.per_head_accum.iter().map(|v| v / run.trace_count as f64).collect::<Vec<_>>(),
                "mean_spearman_delta_final": run.spearman_sum / run.trace_count as f32,
                "mean_abs_spearman_delta_baseline": run.corr_abs_sum / run.trace_count as f32,
            })
        } else {
            Value::Null
        };
        let engine_arms = json!({
            "collection": "c8p",
            "heads": engine_heads,
            "attend_heads": run.arm.attend_heads,
            "retrieval_mode": if run.arm.mode == "A" { "SingleHead" } else { "FixedFusion" },
            "candidate_budget": cand_budget,
            "min_candidates_per_head": min_cand_head,
            "max_candidates_per_head": max_cand_head,
            "ef_search": run.arm.ef_search.unwrap_or(ef_search),
            "fusion": run.arm.fusion_label,
            "c8_enabled": run.arm.c8.is_some(),
            "c8_fingerprint": run.arm.fingerprint,
            "configuration_id": run.arm.config_id,
            "deadline_us": deadline_us,
            "lane": if multi_arm { "multi" } else { "single" },
        });
        json!({
            "subcommand": "pilot", "mode": run.arm.mode, "arm": run.arm.name, "ok": true,
            "engine": engine_arms,
            "n_docs": n_docs, "n_queries": run_rows.len(),
            "warmup": warmup, "seed": seed, "k": k,
            "latency_us_post_warmup": {"count": slat.len(), "mean": mean,
                "p50": percentile(&slat, 0.50), "p90": percentile(&slat, 0.90), "p95": percentile(&slat, 0.95)},
            "recall10_qrels_mean": mean_qrels, "recall10_exact_mean": mean_exact,
            "deadline_exceeded_count": run.deadline_misses,
            "c8_aggregate": c8_aggregate,
            "per_query": per_query,
        })
    };

    if multi_arm {
        let header = json!({
            "subcommand": "pilot", "multi_arm": true, "ok": true,
            "engine": {
                "collection": "c8p", "heads": engine_heads,
                "candidate_budget": cand_budget,
                "min_candidates_per_head": min_cand_head,
                "max_candidates_per_head": max_cand_head,
                "ef_search": ef_search,
                "deadline_us": deadline_us,
            },
            "n_docs": n_docs, "n_queries": run_rows.len(),
            "warmup": warmup, "seed": seed, "k": k,
        });
        if let Some(op) = out_path {
            use std::io::Write as _;
            let mut w = std::io::BufWriter::new(
                std::fs::File::create(op).unwrap_or_else(|e| panic!("create out {op}: {e}")),
            );
            w.write_all(b"{\n").unwrap();
            let mut first = true;
            for (key, val) in header.as_object().unwrap() {
                if !first {
                    w.write_all(b",\n").unwrap();
                }
                first = false;
                w.write_all(format!("  \"{key}\": ").as_bytes()).unwrap();
                serde_json::to_writer_pretty(&mut w, val).unwrap();
            }
            w.write_all(b",\n  \"exact_top10\": ").unwrap();
            serde_json::to_writer_pretty(&mut w, &all_exact).unwrap();
            w.write_all(b",\n  \"arms\": {\n").unwrap();
            for (j, run) in runs.iter_mut().enumerate() {
                if j > 0 {
                    w.write_all(b",\n").unwrap();
                }
                let per_query = std::mem::take(&mut run.per_query);
                let val = build_arm_output(run, per_query);
                w.write_all(format!("    \"{}\": ", run.arm.name).as_bytes())
                    .unwrap();
                serde_json::to_writer_pretty(&mut w, &val).unwrap();
                drop(val);
            }
            w.write_all(b"\n  }\n}").unwrap();
            w.flush().unwrap();
            return (
                json!({
                    "subcommand": "pilot", "multi_arm": true, "ok": true, "streamed": true,
                    "n_arms": runs.len(), "out": op,
                }),
                true,
            );
        }
        let mut arms_out = serde_json::Map::new();
        for run in &runs {
            arms_out.insert(
                run.arm.name.clone(),
                build_arm_output(run, run.per_query.clone()),
            );
        }
        (
            json!({
                "subcommand": "pilot", "multi_arm": true, "ok": true,
                "engine": {
                    "collection": "c8p", "heads": engine_heads,
                    "candidate_budget": cand_budget,
                    "min_candidates_per_head": min_cand_head,
                    "max_candidates_per_head": max_cand_head,
                    "ef_search": ef_search,
                    "deadline_us": deadline_us,
                },
                "n_docs": n_docs, "n_queries": run_rows.len(),
                "warmup": warmup, "seed": seed, "k": k,
                "exact_top10": all_exact,
                "arms": Value::Object(arms_out),
            }),
            false,
        )
    } else {
        let run = &runs[0];
        let mut out = build_arm_output(run, run.per_query.clone());
        if let Some(obj) = out.as_object_mut() {
            obj.insert("exact_top10".into(), json!(all_exact));
        }
        (out, false)
    }
}

/// Build full retrieval evidence from a candidate's per-head similarities.
/// Ranks are not carried by `CandidateFeatures`, so they are left absent; the
/// aggregate and variance terms used by C8 depend only on the sims.
fn evidence_from_scores(scores: &[Option<f32>]) -> RetrievalEvidence {
    RetrievalEvidence {
        head_sims: scores.to_vec(),
        head_ranks: vec![None; scores.len()],
        head_present: scores.iter().map(|s| s.is_some()).collect(),
    }
}

fn rsub_train(cfg: &Value) -> Value {
    let seed = cfg["seed"].as_u64().unwrap_or(20260925);
    let n_docs = cfg["n_docs"].as_u64().unwrap() as usize;
    let n_queries = cfg["n_queries"].as_u64().unwrap() as usize;
    let dim = cfg["dim"].as_u64().unwrap() as usize;
    let cand_budget = cfg["candidate_budget"].as_u64().unwrap_or(500) as usize;
    let min_cand_head = cfg["min_candidates_per_head"].as_u64().unwrap_or(20) as usize;
    let ef_search = cfg["ef_search"].as_u64().unwrap_or(64) as usize;

    let tr = cfg.get("train").expect("train block");
    let arm = tr["arm"].as_str().unwrap();
    assert!(
        matches!(arm, "D" | "E" | "F" | "G" | "H"),
        "train arm must be D|E|F|G|H, got {arm}"
    );
    let epochs = tr["epochs"].as_u64().unwrap_or(5) as usize;
    let lr = tr["learning_rate"].as_f64().unwrap_or(1e-2) as f32;
    let tau = tr["temperature"].as_f64().unwrap_or(0.07) as f32;
    let l2 = tr["l2"].as_f64().unwrap_or(1e-4) as f32;
    let batch_size = tr["batch_size"].as_u64().unwrap_or(8) as usize;
    let negatives_per_query = tr["negatives_per_query"].as_u64().unwrap_or(8) as usize;
    let residual_scale = tr["residual_scale"].as_f64().unwrap_or(0.1) as f32;
    let residual_alpha = tr["residual_alpha"].as_f64().unwrap_or(0.1) as f32;
    // Reduced key/value dims (C8 decision: d_k = d_v = 64 primary). The
    // attention/query dim stays `dim`; only the projected K/V width changes.
    let key_dim = tr["key_dim"].as_u64().unwrap_or(dim as u64) as usize;
    let value_dim = tr["value_dim"].as_u64().unwrap_or(dim as u64) as usize;
    assert!(
        key_dim > 0 && value_dim > 0 && key_dim <= dim && value_dim <= dim,
        "train key_dim/value_dim must be in (0, dim]"
    );
    let residual_reg = tr["residual_regularization"].as_f64().unwrap_or(0.0) as f32;
    let distill_weight = tr["distillation_weight"].as_f64().unwrap_or(0.0) as f32;
    let distill_temp = tr["distillation_temperature"].as_f64().unwrap_or(0.5) as f32;
    let use_distillation = tr["use_distillation"].as_bool().unwrap_or(false);
    let use_evidence = tr["use_evidence"].as_bool().unwrap_or(false);
    let w_disagree = tr["w_disagree"].as_f64().unwrap_or(0.0) as f32;
    let model_id = tr["model_id"].as_str().unwrap_or("c8-model");
    let arch = tr["arch"].as_str().unwrap_or("residual-qkv");
    let train_rows: Vec<usize> = tr["rows"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_u64().unwrap() as usize).collect())
        .unwrap_or_default();

    // ---- vectors + qrels ----------------------------------------------
    let mut doc_vecs: HashMap<String, Vec<f32>> = HashMap::new();
    if let Some(map) = cfg["doc_vectors"].as_object() {
        for (head, path) in map {
            doc_vecs.insert(head.clone(), read_f32(path.as_str().unwrap(), n_docs * dim));
        }
    }
    let qflat = read_f32(cfg["query_vectors"].as_str().unwrap(), n_queries * dim);
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(n_queries);
    for r in 0..n_queries {
        queries.push(qflat[r * dim..(r + 1) * dim].to_vec());
    }
    let mut qrels: HashMap<usize, HashMap<u64, f32>> = HashMap::new();
    if let Some(qr) = cfg["qrels"].as_object() {
        for (qrow, rels) in qr {
            let qr_i = qrow.parse::<usize>().unwrap();
            let mut m = HashMap::new();
            if let Some(arr) = rels.as_array() {
                for v in arr {
                    if let Some(ridx) = v.as_u64() {
                        m.insert(ridx, 1.0);
                    }
                }
            } else if let Some(obj) = rels.as_object() {
                for (drow, grade) in obj {
                    m.insert(drow.parse::<u64>().unwrap(), grade.as_f64().unwrap() as f32);
                }
            }
            qrels.insert(qr_i, m);
        }
    }

    let attend_heads: Vec<String> = cfg["attend_heads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let heads2: Vec<&str> = attend_heads.iter().map(|s| s.as_str()).collect();

    let (e, doc_map) = build_engine(n_docs, dim, &doc_vecs, "B", &heads2);
    let coll = e.get_collection("c8p").unwrap();
    {
        let mut rc = coll.retrieval_config.write();
        rc.mode = RetrievalMode::FixedFusion;
        rc.candidate_budget = cand_budget;
        rc.min_candidates_per_head = min_cand_head;
        rc.ef_search = Some(ef_search);
        rc.cross_refine = None;
        rc.adaptive = None;
        rc.attention = None; // C8 OFF for mining (§C8-4)
        rc.c8_attention = None;
    }

    let order = shuffled_indices(train_rows.len(), seed);
    let query_order: Vec<usize> = order.iter().map(|&o| train_rows[o]).collect();

    let miner = HardNegativeMiner::new(HardNegativeConfig {
        max_negatives: negatives_per_query,
        ..HardNegativeConfig::default()
    });

    let mut builder = C8QkvDatasetBuilder::new(dim, attend_heads.clone());
    let mut n_examples = 0usize;
    let mut n_skipped = 0usize;
    let mut mined_sources: HashMap<String, usize> = HashMap::new();

    for &qr in &query_order {
        let q = &queries[qr];
        let res = match coll.attend_detailed_c8(
            &attend_heads,
            q,
            cand_budget,
            None,
            None,
            None,
            None,
            None,
        ) {
            Ok((r, _, _, _, _)) => r,
            Err(e) => panic!("attend error: {e}"),
        };
        let relevant = qrels.get(&qr);
        let relevant_set: BTreeSet<u64> = relevant
            .map(|m| m.keys().copied().collect())
            .unwrap_or_default();

        // Positive: highest-ranked relevant row in the union (deterministic).
        let mut positive: Option<(u64, usize, f32)> = None;
        let mut pool: Vec<PoolEntry> = Vec::with_capacity(res.len());
        let mut id_to_row: HashMap<u64, usize> = HashMap::new();
        for r in &res {
            let row = doc_map.get(&r.id).copied().unwrap_or(r.id as usize);
            id_to_row.insert(r.id, row);
            let is_rel = relevant_set.contains(&(row as u64));
            if is_rel && positive.is_none() {
                positive = Some((r.id, row, r.final_score));
            }
            pool.push(PoolEntry {
                id: r.id,
                baseline_score: r.final_score,
                evidence: evidence_from_scores(&r.features.head_scores),
                is_positive: is_rel,
            });
        }
        let Some((pos_id, _pos_row, pos_baseline)) = positive else {
            n_skipped += 1;
            continue;
        };

        let negatives = match miner.mine(&pool, &relevant_set, &BTreeSet::new()) {
            Ok(n) => n,
            Err(e) => panic!("mine error: {e}"),
        };
        if negatives.is_empty() {
            n_skipped += 1;
            continue;
        }

        // Positive per-head vectors + full evidence from its union features.
        let pos_feat = res.iter().find(|r| r.id == pos_id);
        let pos_vec: Vec<Vec<f32>> = attend_heads
            .iter()
            .map(|h| {
                let v = doc_vecs.get(h).unwrap();
                v[_pos_row * dim..(_pos_row + 1) * dim].to_vec()
            })
            .collect();
        let pos_evidence = pos_feat.map(|r| evidence_from_scores(&r.features.head_scores));

        let mut negs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(negatives.len());
        let mut neg_baseline: Vec<f32> = Vec::with_capacity(negatives.len());
        let mut neg_ev: Vec<Option<RetrievalEvidence>> = Vec::with_capacity(negatives.len());
        let mut neg_sources = Vec::with_capacity(negatives.len());
        for hn in &negatives {
            let row = id_to_row[&hn.candidate_id];
            let z: Vec<Vec<f32>> = attend_heads
                .iter()
                .map(|h| {
                    let v = doc_vecs.get(h).unwrap();
                    v[row * dim..(row + 1) * dim].to_vec()
                })
                .collect();
            negs.push(z);
            neg_baseline.push(hn.baseline_score);
            let ev = if use_evidence {
                res.iter()
                    .find(|r| r.id == hn.candidate_id)
                    .map(|r| evidence_from_scores(&r.features.head_scores))
            } else {
                None
            };
            neg_ev.push(ev);
            neg_sources.push(hn.source);
            *mined_sources
                .entry(hn.source.as_str().to_string())
                .or_insert(0) += 1;
        }

        let ex = C8QkvExample {
            q_a: q.clone(),
            positive: pos_vec,
            positive_baseline: pos_baseline,
            positive_evidence: if use_evidence { pos_evidence } else { None },
            negatives: negs,
            negatives_baseline: neg_baseline,
            negatives_evidence: neg_ev,
            negative_sources: neg_sources,
        };
        builder.push(ex).expect("push example");
        n_examples += 1;
    }

    let dataset: C8QkvDataset = builder.build();
    if dataset.examples.is_empty() {
        panic!("empty training dataset (no queries had a relevant doc in the union)");
    }
    let dataset_hash = dataset.fingerprint();
    let negative_provenance_hash = dataset.negative_provenance_hash;

    let scorer = if w_disagree != 0.0 {
        ResidualScorer::with_disagreement(1.0, if use_evidence { 1.0 } else { 0.0 }, w_disagree, 0.0)
    } else {
        ResidualScorer::new(1.0, if use_evidence { 1.0 } else { 0.0 }, 0.0)
    };

    let init = if arm == "D" {
        ResidualQkvProjection::unrestricted(dim, key_dim, value_dim, seed).expect("unrestricted init")
    } else {
        ResidualQkvProjection::truncated_identity_residual(dim, key_dim, value_dim, residual_alpha)
            .expect("residual init")
    };

    let tcfg = C8TrainingConfig {
        seed,
        learning_rate: lr,
        epochs,
        batch_size,
        temperature: tau,
        l2,
        residual_scale,
        residual_regularization: residual_reg,
        distillation_weight: distill_weight,
        distillation_temperature: distill_temp,
        use_distillation,
    };
    let mut trainer = ResidualQkvTrainer::new(tcfg.clone());
    let trained = trainer
        .train(init, &dataset, &scorer)
        .expect("train");

    let aligns: Vec<AlignmentProjection> =
        attend_heads.iter().map(|_| AlignmentProjection::identity(dim)).collect();
    let card: C8ModelCard = trained
        .to_model_card(
            model_id,
            arch,
            attend_heads.clone(),
            aligns,
            AlignmentProjection::identity(dim),
            scorer.clone(),
            use_evidence,
            residual_scale,
            negative_provenance_hash,
        )
        .expect("model card");

    json!({
        "subcommand": "train", "arm": arm, "ok": true,
        "model_id": model_id, "arch": arch,
        "use_evidence": use_evidence,
        "scorer": {
            "w_attn": scorer.w_attn, "w_evidence": scorer.w_evidence,
            "w_disagree": scorer.w_disagree, "bias": scorer.bias,
        },
        "n_examples": n_examples,
        "n_skipped_queries_no_positive_in_union": n_skipped,
        "mined_negative_sources": mined_sources,
        "negative_provenance_hash": negative_provenance_hash,
        "dataset_hash": dataset_hash,
        "attention_dim": dim, "key_dim": key_dim, "value_dim": value_dim,
        "head_names": attend_heads,
        "residual_scale": residual_scale,
        "residual_alpha": residual_alpha,
        "loss_history": trained.loss_history,
        "contrastive_history": trained.contrastive_history,
        "residual_history": trained.residual_history,
        "distill_history": trained.distill_history,
        "optimizer_steps": trained.optimizer_steps,
        "epochs_run": trained.epochs_run,
        "seed": seed,
        "batch_size": batch_size,
        "negatives_per_query": negatives_per_query,
        "temperature": tau, "learning_rate": lr, "l2": l2,
        "use_distillation": use_distillation,
        "distillation_weight": distill_weight,
        "distillation_temperature": distill_temp,
        "model_card": card,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut config = String::new();
    let mut out_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                config = args[i + 1].clone();
                i += 2;
            }
            "--out" => {
                out_path = Some(args[i + 1].clone());
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(64);
            }
        }
    }
    let txt = if !config.is_empty() {
        std::fs::read_to_string(&config).unwrap_or_else(|e| panic!("read config: {e}"))
    } else {
        eprintln!("no --config given");
        std::process::exit(64);
    };
    let cfg: Value = serde_json::from_str(&txt).expect("config JSON");

    let (doc, streamed) = match cfg["subcommand"].as_str().unwrap_or("run") {
        "run" => rsub_run(&cfg, out_path.as_deref()),
        "train" => (rsub_train(&cfg), false),
        other => panic!("subcommand must be run|train, got {other}"),
    };
    let s = serde_json::to_string_pretty(&doc).unwrap();
    if let Some(op) = out_path {
        if !streamed {
            std::fs::write(op, &s).unwrap();
        }
    }
    println!("{s}");
}
