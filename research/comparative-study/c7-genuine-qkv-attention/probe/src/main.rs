//! c7pilot — C7 genuine candidate-level Q/K/V attention run driver.
//!
//! C7 protocol arms:
//!   "A"  = C7-A: canonical single-head baseline (CANONICAL, attention OFF)
//!   "B"  = C7-B: multi-head union baseline (TITLE/BODY/CITE, attention OFF)
//!   "C"  = C7-C: identity-QKV attention + learned gate g*  (fusion override)
//!   "D"  = C7-D: identity-QKV attention + fixed fusion
//!   "E"  = C7-E: learned QKV attention + fixed fusion (no evidence)
//!   "F"  = C7-F: learned QKV attention + evidence            (use_evidence)
//!
//! Budget/shared settings across all arms (§C7-3):
//!   candidate_budget 500, min_candidates_per_head 20,
//!   max_candidates_per_head 300, ef_search 64 (every arm), k = 10,
//!   dim 384, H = 3 heads (A uses H = 1, CANONICAL only).
//!
//! B/C/D/E/F share the IDENTICAL candidate union per query (same heads,
//! same FixedFusion membership pipeline, same ef/budget); only the `attention`
//! channel drives any ranking difference, so C7 is attributable (§C7).
//!
//! Subcommands:
//!   run    — load the collection, run warmup + timed queries, emit per-query
//!            rows with C7 observability (per-candidate A_d / logits / entropy
//!            / attention score, per-head means, mean entropy, Spearman
//!            attention<->final) plus the full ranked-union final-score ledger
//!            for cross-run rank-change vs B.
//!   train  — mine hard negatives from the identical union (mode B, attention
//!            OFF, ef 64, budget 500) on the VALIDATION split and train the
//!            C7-E/F Q/K/V projections with the deterministic contrastive
//!            trainer (InfoNCE over 8 hard negatives, §C7-4). Emits a model
//!            JSON with the projection (consumed by `run` for E/F via
//!            `attention.qkv_model`).
//!
//! Input contract (single JSON, --config path or stdin):
//! {
//!   "subcommand": "run" | "train",
//!   "mode": "A|B|C|D|E|F",
//!   "k": 10, "seed": 20260925, "warmup": 20,
//!   "n_queries": 323, "n_docs": 3633, "dim": 384,
//!   "collection_heads": [...], "attend_heads": [...],
//!   "doc_vectors": {"<head>": "<raw LE f32 path>"},
//!   "query_vectors": "<raw LE f32 path>",
//!   "qrels": {"<row>": {"<docrow>": grade}},
//!   "subsample": [rows],
//!   "candidate_budget": 500, "min_candidates_per_head": 20,
//!   "max_candidates_per_head": 300, "ef_search": 64,
//!   "attention": {                    // C/D/E/F only
//!       "enabled": true,
//!       "arch": "identity-qkv" | "learned-qkv",
//!       "attention_dim": 384, "key_dim": 384, "value_dim": 384,
//!       "use_evidence": false,
//!       "scorer": {"w_attn": 1.0, "w_evidence": 0.0, "bias": 0.0},
//!       "qkv_model": ""               // learned-qkv: path to trained model JSON
//!   },
//!   "fusion": {"attention": g, "multi_head_similarity": 1-g, "bm25": 0},
//!   "emit_full_trace": false,         // include A_d/logits/output per candidate
//!   "configuration_id": "...",
//!   "deadline_us": 0
//! }
//!
//! Train input adds a "train" block:
//!   "train": {
//!     "arm": "E"|"F",
//!     "epochs": 5, "learning_rate": 1e-2, "temperature": 0.07, "l2": 1e-4,
//!     "seed": 20260925, "batch_size": 8, "negatives_per_query": 8,
//!     "w_attn": 1.0, "w_evidence": 0.0,   // arm F: w_evidence = 1.0
//!     "use_evidence": false,              // arm F: true
//!     "rows": [...],                      // VALIDATION query rows
//!     "model_id": "scifact-c7-e", "arch": "contrastive-qkv"
//!   }
//!
//! Emits one JSON doc on stdout (and --out).

use attentiondb_attention::{
    config_fingerprint, AlignmentProjection, AttentionConfig, AttentionScorer, QkvProjection,
};
use attentiondb_attention::training::{
    ContrastiveQkvTrainer, QkvDatasetBuilder, QkvExample, TrainingConfig,
};
use attentiondb_core::{
    collection::RetrievalMode,
    retrieval::FusionWeights,
    AttentionEngine,
};
use attentiondb_storage::{Durability, Record};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
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
    if d == 0.0 { 0.0 } else { dot / d }
}

fn brute_topk(vecs: &[(u64, Vec<f32>)], q: &[f32], k: usize) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = vecs
        .iter()
        .map(|(id, v)| (*id, cosine(q, v)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(k.min(scored.len()));
    scored
}

fn ndcg10(retrieved: &[u64], relevant: &HashMap<u64, f32>, k: usize) -> f32 {
    let mut dcg = 0.0f32;
    for (rank, id) in retrieved.iter().enumerate() {
        if rank >= k { break; }
        let rel = relevant.get(id).copied().unwrap_or(0.0);
        dcg += rel / (rank as f32 + 2.0).log2();
    }
    let mut ideal: Vec<f32> = relevant.values().copied().collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let mut idcg = 0.0f32;
    for (rank, rel) in ideal.iter().enumerate() {
        if rank >= k { break; }
        if *rel <= 0.0 { continue; }
        idcg += rel / (rank as f32 + 2.0).log2();
    }
    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi { sorted[lo] } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Deterministic LCG for query order shuffling.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xA0761D6478BD642F);
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
/// Rows present in both maps only. Constant/degenerate series -> 0.0.
fn spearman_on_shared(
    a: &HashMap<u64, f32>,
    b: &HashMap<u64, f32>,
) -> f32 {
    let rows: Vec<u64> = a.keys().filter(|r| b.contains_key(r)).copied().collect();
    if rows.len() < 3 { return 0.0; }
    let mut av: Vec<(u64, f32)> = rows.iter().map(|&r| (r, *a.get(&r).unwrap())).collect();
    let mut bv: Vec<(u64, f32)> = rows.iter().map(|&r| (r, *b.get(&r).unwrap())).collect();
    av.sort_by(|x, y| x.1.total_cmp(&y.1));
    bv.sort_by(|x, y| x.1.total_cmp(&y.1));
    let mut ra: HashMap<u64, f64> = HashMap::new();
    let mut rb: HashMap<u64, f64> = HashMap::new();
    for (i, (r, _)) in av.iter().enumerate() { ra.insert(*r, i as f64 + 1.0); }
    for (i, (r, _)) in bv.iter().enumerate() { rb.insert(*r, i as f64 + 1.0); }
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
    if d == 0.0 { 0.0 } else { (cov / d) as f32 }
}

fn parse_fusion(v: &Value) -> (FusionWeights, String) {
    if let Some(f) = v.get("fusion").and_then(|x| x.as_object()) {
        let atn = f.get("attention").and_then(|x| x.as_f64()).unwrap_or(0.3) as f32;
        let mhs = f.get("multi_head_similarity").and_then(|x| x.as_f64()).unwrap_or(0.5) as f32;
        let bm = f.get("bm25").and_then(|x| x.as_f64()).unwrap_or(0.2) as f32;
        let w = FusionWeights { attention: atn, multi_head_similarity: mhs, bm25: bm };
        w.validate().expect("invalid fusion weights");
        let label = format!("a:{atn:.4},mhs:{mhs:.4},bm25:{bm:.4}");
        (w, label)
    } else {
        let w = FusionWeights { attention: 0.3, multi_head_similarity: 0.5, bm25: 0.2 };
        (w, "default(0.3,0.5,0.2)".into())
    }
}

fn build_attention_cfg(cfg: &Value, heads: &[String]) -> Option<AttentionConfig> {
    let att = cfg.get("attention")?;
    if !att.get("enabled").and_then(|e| e.as_bool()).unwrap_or(false) {
        return None;
    }
    let d_a = att.get("attention_dim").and_then(|v| v.as_u64()).unwrap_or(384) as usize;
    let d_k = att.get("key_dim").and_then(|v| v.as_u64()).unwrap_or(384) as usize;
    let d_v = att.get("value_dim").and_then(|v| v.as_u64()).unwrap_or(384) as usize;
    let use_evidence = att.get("use_evidence").and_then(|v| v.as_bool()).unwrap_or(false);
    let (w_attn, w_ev, bias) = if let Some(s) = att.get("scorer").and_then(|x| x.as_object()) {
        (
            s.get("w_attn").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32,
            s.get("w_evidence").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
            s.get("bias").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
        )
    } else {
        (1.0, 0.0, 0.0)
    };
    let scorer = AttentionScorer::new(w_attn, w_ev, bias);
    match att.get("arch").and_then(|a| a.as_str()) {
        Some("identity-qkv") | None => {
            let mut c = AttentionConfig::fixed_identity(heads.len(), d_a, d_k, d_v);
            c.use_evidence = use_evidence;
            c.scorer = scorer;
            Some(c)
        }
        Some("learned-qkv") => {
            let path = att.get("qkv_model").and_then(|p| p.as_str()).unwrap_or("");
            assert!(!path.is_empty(), "qkv_model required for learned-qkv");
            let txt = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("read qkv_model {path}: {e}"));
            let card: Value = serde_json::from_str(&txt).expect("qkv model JSON");
            let qkv: QkvProjection =
                serde_json::from_value(card.get("projection").cloned().unwrap())
                    .expect("projection in qkv model");
            let head_alignments: Vec<AlignmentProjection> =
                heads.iter().map(|_| AlignmentProjection::identity(d_a)).collect();
            let query_alignment = AlignmentProjection::identity(d_a);
            Some(AttentionConfig::learned(
                heads.len(),
                d_a,
                d_k,
                d_v,
                head_alignments,
                query_alignment,
                qkv,
                use_evidence,
                scorer,
            ))
        }
        other => panic!("unknown attention.arch: {other:?}"),
    }
}

fn build_engine(
    n_docs: usize,
    dim: usize,
    doc_vecs: &HashMap<String, Vec<f32>>,
    mode: &str,
    heads: &[&str],
) -> (AttentionEngine, HashMap<u64, usize>) {
    let dir = std::env::temp_dir().join(format!("c7pilot-{}-{}", std::process::id(), mode));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Async)
        .unwrap_or_else(|err| panic!("open engine: {err}"));
    e.create_collection("c7p", dim, heads).unwrap();
    let mut doc_map: HashMap<u64, usize> = HashMap::new();
    for row in 0..n_docs {
        let mut fields: HashMap<String, Value> = HashMap::new();
        fields.insert("idx".to_string(), json!(row));
        let mut rec = Record::new(fields);
        for h in heads {
            let vec = doc_vecs.get(*h).unwrap_or_else(|| panic!("missing doc head {h}"));
            rec.k_vecs.insert((*h).to_string(), vec[row * dim..(row + 1) * dim].to_vec());
        }
        let uuid = e.insert_document("c7p", rec).unwrap();
        let nid = e.id_mapper.read().uuid_to_id(&uuid.parse().unwrap()).unwrap();
        doc_map.insert(nid, row);
    }
    (e, doc_map)
}

/// Per-arm execution contract for a shared-engine multi-arm `run`.
/// Each arm reuses the same collection/graph; union membership is bit-identical
/// across arms because it depends only on heads/ef/budgets (§C7-2), while the
/// attention channel and fusion weights may differ per arm.
#[derive(Clone)]
struct ArmSpec {
    name: String,
    mode: String,
    ef_search: Option<usize>,
    search_k: Option<usize>,
    attend_heads: Vec<String>,
    attention: Option<AttentionConfig>,
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
        arr.iter().map(|x| x.as_str().unwrap().to_string()).collect()
    } else {
        default_heads.to_vec()
    };
    let attention = build_attention_cfg(v, &attend_heads);
    let fingerprint = attention.as_ref().map(config_fingerprint);
    let (fusion, fusion_label) = parse_fusion(v);
    let config_id = v["configuration_id"].as_str().unwrap_or("").to_string();
    let emit_full_trace = v["emit_full_trace"].as_bool().unwrap_or(false);
    ArmSpec {
        name,
        mode,
        ef_search,
        search_k,
        attend_heads,
        attention,
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

    // Arms: explicit `arms` array (multi-arm, shared engine) or a single arm
    // synthesized from the top-level mode/attention/fusion (backward compat).
    let default_heads: Vec<String> = cfg["attend_heads"]
        .as_array()
        .map(|a| a.iter().map(|x| x.as_str().unwrap().to_string()).collect())
        .unwrap_or_default();
    let arms: Vec<ArmSpec> = if let Some(arr) = cfg["arms"].as_array() {
        arr.iter().map(|a| parse_arm_spec(a, &default_heads)).collect()
    } else {
        let mode = cfg["mode"].as_str().unwrap_or("B").to_string();
        if !matches!(mode.as_str(), "A" | "B" | "C" | "D" | "E" | "F") {
            panic!("mode must be A|B|C|D|E|F, got {mode}");
        }
        let attention = if matches!(mode.as_str(), "C" | "D" | "E" | "F") {
            build_attention_cfg(cfg, &default_heads)
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
            attention,
            fingerprint: None,
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
            if head == "CANONICAL" { canonical = v.clone(); }
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
        (0..n_docs).map(|d| (d as u64, canonical[d * dim..(d + 1) * dim].to_vec())).collect()
    };

    // ---- shared engine: ONE collection/graph, all arms reuse it --------
    // Union membership (candidate_union) depends only on heads + ef + budgets;
    // attention/fusion are scoring-only overrides, so per-arm calls share a
    // bit-identical union (§C7-2, stop condition #4).
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
    let coll = e.get_collection("c7p").unwrap();

    // Per-arm RetrievalConfig built from the collection's defaults so every
    // arm shares candidate_multiplier/budgets (union identity), then overrides
    // only mode/fusion/attention (scoring channels).
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
            per_head_accum: vec![0.0; a.attend_heads.len()],
            deadline_misses: 0,
        })
        .collect();

    for (pos, &qr) in query_order.iter().enumerate() {
        let q = &queries[qr];
        let exact = if vlist.is_empty() { Vec::new() } else { brute_topk(&vlist, q, k) };
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
            // per-arm retrieval config (scoring override; union params fixed)
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
                rc.attention = run.arm.attention.clone();
                rc
            };
            let mode_override = if run.arm.mode == "A" {
                Some(RetrievalMode::SingleHead)
            } else {
                Some(RetrievalMode::FixedFusion)
            };

            // Fetch the WHOLE union (top_k = candidate_budget) so the emitted
            // ledger carries every candidate's final score (rank-change vs B).
            let arm_t = Instant::now();
            match coll.attend_detailed_c7(
                &run.arm.attend_heads, q, cand_budget, None,
                mode_override, None, Some(&arm_cfg), deadline,
            ) {
                Ok((res, stats, _cross, _adaptive, c7_trace)) => {
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
                    let recall = if rel_set.is_empty() { 0.0 }
                        else { hits_cnt as f32 / rel_set.len() as f32 };
                    let recall_exact = if ids.is_empty() { 0.0 }
                        else {
                            let exact_set: HashSet<u64> = ids.iter().copied().collect();
                            hits_rows.iter().filter(|r| exact_set.contains(r)).count() as f32 / ids.len() as f32
                        };
                    let ndcg = if rel_set.is_empty() { 0.0 } else { ndcg10(&hits_rows, &relevant, k) };

                    // ---- C7 observability ------------------------------
                    // Ranked-union final-score ledger (rank-change vs B).
                    let mut rank_ledger: Vec<Value> = Vec::with_capacity(res.len());
                    let mut final_by_row: HashMap<u64, f32> = HashMap::new();
                    for (rank, r) in res.iter().enumerate() {
                        let row = doc_map.get(&r.id).copied().unwrap_or(r.id as usize) as u64;
                        let mhs = r.features.multi_head_similarity;
                        let att = r.features.attention;
                        let hs: Vec<Value> = r.features.head_scores.iter().map(|s| match s {
                            Some(v) => json!(v),
                            None => Value::Null,
                        }).collect();
                        final_by_row.insert(row, r.final_score);
                        rank_ledger.push(json!({
                            "row": row, "rank": rank, "final": r.final_score,
                            "mhs": mhs, "attention": att, "head_sims": hs,
                        }));
                    }

                    // C7Trace per-candidate attention detail (union order).
                    let trace_row = if let Some(trace) = &c7_trace {
                        let rows: Vec<Value> = trace
                            .candidates
                            .iter()
                            .map(|c| {
                                let row = doc_map.get(&c.id).copied().unwrap_or(c.id as usize) as u64;
                                let mut obj = serde_json::Map::new();
                                obj.insert("row".into(), json!(row));
                                obj.insert("attention_score".into(), json!(c.attention_score));
                                obj.insert("entropy".into(), json!(c.entropy));
                                if run.arm.emit_full_trace {
                                    obj.insert("weights".into(), json!(c.weights));
                                    obj.insert("logits".into(), json!(c.logits));
                                    obj.insert("output".into(), json!(c.output));
                                    obj.insert("weights_sum".into(),
                                        json!(c.weights.iter().sum::<f32>()));
                                    obj.insert("entropy_recompute".into(),
                                        json!(c.weights.iter().map(|&a|
                                            if a > 0.0 { -a * a.ln() } else { 0.0 }).sum::<f32>()));
                                }
                                json!(obj)
                            })
                            .collect();
                        // Spearman(attention, final) over shared rows.
                        let mut att_by_row: HashMap<u64, f32> = HashMap::new();
                        for c in &trace.candidates {
                            let row = doc_map.get(&c.id).copied().unwrap_or(c.id as usize) as u64;
                            att_by_row.insert(row, c.attention_score);
                        }
                        let sp = spearman_on_shared(&att_by_row, &final_by_row);
                        run.spearman_sum += sp;
                        run.trace_count += 1;
                        for (i, ph) in trace.per_head_mean.iter().enumerate() {
                            run.per_head_accum[i] += *ph as f64;
                        }
                        run.ent_sum += trace.mean_entropy;
                        json!({
                            "head_names": trace.head_names,
                            "n_candidates": trace.candidates.len(),
                            "per_head_mean": trace.per_head_mean,
                            "mean_entropy": trace.mean_entropy,
                            "compute_time_us": trace.compute_time_us,
                            "candidates": rows,
                            "spearman_attention_final": sp,
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
                        "attention_enabled": run.arm.attention.is_some(),
                        "attention_fingerprint": run.arm.fingerprint,
                        "fusion_label": run.arm.fusion_label,
                        "union_ledger": rank_ledger,
                        "c7_trace": trace_row,
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
                            "relevant_ids": [], "attention_enabled": run.arm.attention.is_some(),
                            "attention_fingerprint": run.arm.fingerprint, "fusion_label": run.arm.fusion_label,
                            "union_ledger": [], "c7_trace": null,
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
        let mean = if slat.is_empty() { 0.0 } else { slat.iter().sum::<f64>() / slat.len() as f64 };
        let rec_qrels: Vec<f32> = per_query.iter().map(|r| r["recall10_qrels"].as_f64().unwrap() as f32).collect();
        let rec_exact: Vec<f32> = per_query.iter().map(|r| r["recall10_exact"].as_f64().unwrap() as f32).collect();
        let mean_qrels = rec_qrels.iter().sum::<f32>() / rec_qrels.len().max(1) as f32;
        let mean_exact = rec_exact.iter().sum::<f32>() / rec_exact.len().max(1) as f32;
        let c7_aggregate = if run.trace_count > 0 {
            json!({
                "queries_with_trace": run.trace_count,
                "mean_entropy_over_queries": run.ent_sum / run.trace_count as f32,
                "per_head_mean_over_queries":
                    run.per_head_accum.iter().map(|v| v / run.trace_count as f64).collect::<Vec<_>>(),
                "mean_spearman_attention_final": run.spearman_sum / run.trace_count as f32,
            })
        } else {
            Value::Null
        };
        let engine_arms = json!({
            "collection": "c7p",
            "heads": engine_heads,
            "attend_heads": run.arm.attend_heads,
            "retrieval_mode": if run.arm.mode == "A" { "SingleHead" } else { "FixedFusion" },
            "candidate_budget": cand_budget,
            "min_candidates_per_head": min_cand_head,
            "max_candidates_per_head": max_cand_head,
            "ef_search": run.arm.ef_search.unwrap_or(ef_search),
            "fusion": run.arm.fusion_label,
            "attention_enabled": run.arm.attention.is_some(),
            "attention_fingerprint": run.arm.fingerprint,
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
            "c7_aggregate": c7_aggregate,
            "per_query": per_query,
        })
    };

    if multi_arm {
        let header = json!({
            "subcommand": "pilot", "multi_arm": true, "ok": true,
            "engine": {
                "collection": "c7p", "heads": engine_heads,
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
            // Stream the multi-arm output arm-by-arm so peak memory is one
            // arm's result, not the whole quote x arms Value tree (the gate
            // scan at 200 queries x 11 arms would otherwise need > 2 GB).
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
                w.write_all(format!("    \"{}\": ", run.arm.name).as_bytes()).unwrap();
                serde_json::to_writer_pretty(&mut w, &val).unwrap();
                drop(val);
            }
            w.write_all(b"\n  }\n}").unwrap();
            w.flush().unwrap();
            return (json!({
                "subcommand": "pilot", "multi_arm": true, "ok": true, "streamed": true,
                "n_arms": runs.len(), "out": op,
            }), true);
        }
        let mut arms_out = serde_json::Map::new();
        for run in &runs {
            arms_out.insert(run.arm.name.clone(),
                            build_arm_output(run, run.per_query.clone()));
        }
        (json!({
            "subcommand": "pilot", "multi_arm": true, "ok": true,
            "engine": {
                "collection": "c7p", "heads": engine_heads,
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
        }), false)
    } else {
        let run = &runs[0];
        let mut out = build_arm_output(run, run.per_query.clone());
        if let Some(obj) = out.as_object_mut() {
            obj.insert("exact_top10".into(), json!(all_exact));
        }
        (out, false)
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
    assert!(matches!(arm, "E" | "F"), "train arm must be E or F");
    let epochs = tr["epochs"].as_u64().unwrap_or(5) as usize;
    let lr = tr["learning_rate"].as_f64().unwrap_or(1e-2) as f32;
    let tau = tr["temperature"].as_f64().unwrap_or(0.07) as f32;
    let l2 = tr["l2"].as_f64().unwrap_or(1e-4) as f32;
    let batch_size = tr["batch_size"].as_u64().unwrap_or(8) as usize;
    let negatives_per_query = tr["negatives_per_query"].as_u64().unwrap_or(8) as usize;
    let w_attn = tr["w_attn"].as_f64().unwrap_or(1.0) as f32;
    let w_evidence = tr["w_evidence"].as_f64().unwrap_or(0.0) as f32;
    let use_evidence = tr["use_evidence"].as_bool().unwrap_or(false);
    let model_id = tr["model_id"].as_str().unwrap_or("c7-model");
    let arch = tr["arch"].as_str().unwrap_or("contrastive-qkv");
    let train_rows: Vec<usize> = tr["rows"].as_array().map(|a|
        a.iter().map(|v| v.as_u64().unwrap() as usize).collect()
    ).unwrap_or_default();

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

    let collection_heads: Vec<String> = cfg["collection_heads"]
        .as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let attend_heads: Vec<String> = cfg["attend_heads"]
        .as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let heads2: Vec<&str> = collection_heads.iter().map(|s| s.as_str()).collect();

    let (e, doc_map) = build_engine(n_docs, dim, &doc_vecs, "B", &heads2);
    let coll = e.get_collection("c7p").unwrap();
    {
        let mut rc = coll.retrieval_config.write();
        rc.mode = RetrievalMode::FixedFusion;
        rc.candidate_budget = cand_budget;
        rc.min_candidates_per_head = min_cand_head;
        rc.ef_search = Some(ef_search);
        rc.cross_refine = None;
        rc.adaptive = None;
        rc.attention = None; // identity-QKV OFF for mining (§C7-4)
    }

    // ---- mine hard negatives from the identical union ------------------
    let order = shuffled_indices(train_rows.len(), seed);
    let query_order: Vec<usize> = order.iter().map(|&o| train_rows[o]).collect();

    let mut builder = QkvDatasetBuilder::new(dim, attend_heads.clone());
    let mut n_examples = 0usize;
    let mut n_skipped = 0usize;

    for &qr in &query_order {
        let q = &queries[qr];
        // Full union ranked (fixed fusion, mhs only, attention OFF).
        let res = match coll.attend_detailed_c7(
            &attend_heads, q, cand_budget, None, None, None, None, None,
        ) {
            Ok((r, _, _, _, _)) => r,
            Err(e) => panic!("attend error: {e}"),
        };
        let relevant = qrels.get(&qr);
        // Positive: highest-ranked relevant row in the union (deterministic).
        let mut positive: Option<(u64, usize)> = None;
        let mut neg_cands: Vec<(usize, f32, Vec<Option<f32>>)> = Vec::new();
        for r in &res {
            let row = doc_map.get(&r.id).copied().unwrap_or(r.id as usize);
            let is_rel = relevant.map(|m| m.contains_key(&(row as u64))).unwrap_or(false);
            if is_rel {
                if positive.is_none() {
                    positive = Some((r.id, row));
                }
            } else if neg_cands.len() < negatives_per_query {
                neg_cands.push((
                    row,
                    r.final_score,
                    r.features.head_scores.iter().map(|s| *s).collect(),
                ));
            }
        }
        let Some((pos_id, pos_row)) = positive else {
            n_skipped += 1;
            continue;
        };
        // Evidence helper: mean of present normalized head sims.
        let evidence = |scores: &[Option<f32>]| -> Option<f32> {
            let mut sum = 0.0f32;
            let mut n = 0.0f32;
            for s in scores {
                if let Some(v) = s {
                    sum += v;
                    n += 1.0;
                }
            }
            if n > 0.0 { Some(sum / n) } else { None }
        };
        // Positive per-head vectors + evidence from its union features.
        let pos_vec: Vec<Vec<f32>> = attend_heads
            .iter()
            .map(|h| {
                let v = doc_vecs.get(h).unwrap();
                let row = doc_map.get(&pos_id).copied().unwrap_or(pos_row);
                v[row * dim..(row + 1) * dim].to_vec()
            })
            .collect();
        let pos_evidence_opt = res.iter().find(|r| r.id == pos_id)
            .and_then(|r| evidence(&r.features.head_scores));
        let pos_evidence = if use_evidence { pos_evidence_opt } else { None };

        let mut negs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(neg_cands.len());
        let mut neg_ev: Vec<Option<f32>> = Vec::with_capacity(neg_cands.len());
        for (row, _score, scores) in &neg_cands {
            let z: Vec<Vec<f32>> = attend_heads
                .iter()
                .map(|h| {
                    let v = doc_vecs.get(h).unwrap();
                    v[*row * dim..(*row + 1) * dim].to_vec()
                })
                .collect();
            negs.push(z);
            neg_ev.push(if use_evidence { evidence(scores) } else { None });
        }

        let ex = QkvExample {
            q_a: q.clone(),
            positive: pos_vec,
            positive_evidence: pos_evidence,
            negatives: negs,
            negatives_evidence: neg_ev,
        };
        builder.push(ex).expect("push example");
        n_examples += 1;
    }

    let dataset = builder.build();
    if dataset.examples.is_empty() {
        panic!("empty training dataset (no queries had a relevant doc in the union)");
    }
    let dataset_hash = dataset.fingerprint();

    let tcfg = TrainingConfig {
        seed,
        learning_rate: lr,
        epochs,
        batch_size,
        temperature: tau,
        l2,
    };
    let mut trainer = ContrastiveQkvTrainer::new(tcfg.clone());
    let trained = trainer
        .train(QkvProjection::identity(dim), &dataset, w_attn, w_evidence)
        .expect("train");

    json!({
        "subcommand": "train", "arm": arm, "ok": true,
        "model_id": model_id, "arch": arch,
        "r#use_evidence": use_evidence,
        "scorer": {"w_attn": w_attn, "w_evidence": w_evidence, "bias": 0.0},
        "n_examples": n_examples,
        "n_skipped_queries_no_positive_in_union": n_skipped,
        "dataset_hash": dataset_hash,
        "attention_dim": dim, "key_dim": dim, "value_dim": dim,
        "head_names": attend_heads,
        "loss_history": trained.loss_history,
        "epochs_run": trained.epochs_run,
        "seed": seed,
        "batch_size": batch_size,
        "negatives_per_query": negatives_per_query,
        "temperature": tau, "learning_rate": lr, "l2": l2,
        "projection": trained.projection,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut config = String::new();
    let mut out_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => { config = args[i + 1].clone(); i += 2; }
            "--out" => { out_path = Some(args[i + 1].clone()); i += 2; }
            other => { eprintln!("unknown arg {other}"); std::process::exit(64); }
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
        if !streamed { std::fs::write(op, &s).unwrap(); }
    }
    println!("{s}");
}