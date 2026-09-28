//! c6pilot — C6 adaptive retrieval run driver.
//!
//! C6 protocol arms:
//!   "A"  = C6-A: Canonical single-head baseline (CANONICAL)
//!   "B"  = C6-B: Independent multi-head union (TITLE/BODY/CITE)
//!   "C"  = C6-C: Static equal budget split across 3 heads
//!   "D"  = C6-D: Query-adaptive allocation via head centroids
//!   "E"  = C6-E: Interaction-guided (stage 1 at 30% budget → inspect → stage 2)
//!   "F"  = C6-F: C6-E + C5 cross-refine interaction (λ=0.10)
//!   "RAND" = Negative control: randomized allocation
//!
//! Budget is FIXED across all arms:
//!   total_candidates = 500, total_ef_work = 192 (64 * 3 heads)
//!   min_per_head = 20, max_per_head = 300
//!
//! Input contract (single JSON, --config path or stdin):
//! {
//!   "mode": "A|B|C|D|E|F|RAND",
//!   "k": 10,
//!   "seed": 20260925,
//!   "warmup": 20,
//!   "n_queries": 323,
//!   "n_docs": 3633,
//!   "dim": 384,
//!   "collection_heads": [...],
//!   "attend_heads": [...],
//!   "doc_vectors": {"<head>": "<raw LE f32 path>"},
//!   "query_vectors": "<raw LE f32 path>",
//!   "qrels": {"<row>": {"<docrow>": grade}},
//!   "subsample": [rows],
//!   "candidate_budget": 500,
//!   "min_candidates_per_head": 20,
//!   "max_candidates_per_head": 300,
//!   "total_ef_work": 192,
//!   "ef_search": 64,              // used by modes A/B as baseline
//!   "lambda": 0.10,               // used by mode F (C5 cross-refine)
//!   "stage1_fraction": 0.3,       // used by mode E/F
//!   "overlap_threshold": 0.3,     // used by mode E/F
//!   "entropy_threshold": 1.0,     // used by mode E/F
//!   "randomized_seed": 20260925,  // used by RAND mode
//!   "configuration_id": "...",
//!   "deadline_us": 0
//! }
//!
//! Emits one JSON doc on stdout (and --out): per-query rows, exact oracle
//! top-10, aggregates, and causal ledgers (C5 cross-refine + C6 adaptive).

use attentiondb_core::{
    collection::RetrievalMode,
    retrieval::CrossRefineConfig,
    {AdaptivePolicyType, AdaptiveRetrievalConfig},
};
use attentiondb_core::AttentionEngine;
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
    let mode = cfg["mode"].as_str().unwrap();
    let k = cfg["k"].as_u64().unwrap_or(10) as usize;
    let seed = cfg["seed"].as_u64().unwrap_or(20260925);
    let warmup = cfg["warmup"].as_u64().unwrap_or(20) as usize;
    let n_docs = cfg["n_docs"].as_u64().unwrap() as usize;
    let n_queries = cfg["n_queries"].as_u64().unwrap() as usize;
    let dim = cfg["dim"].as_u64().unwrap() as usize;

    let cand_budget = cfg["candidate_budget"].as_u64().unwrap_or(500) as usize;
    let min_cand_head = cfg["min_candidates_per_head"].as_u64().unwrap_or(20) as usize;
    let max_cand_head = cfg["max_candidates_per_head"].as_u64().unwrap_or(300) as usize;
    let total_ef_work = cfg["total_ef_work"].as_u64().unwrap_or(192) as usize;
    let ef_search = cfg["ef_search"].as_u64().unwrap_or(64) as usize;
    let lambda = cfg["lambda"].as_f64().unwrap_or(0.10) as f32;
    let stage1_fraction = cfg["stage1_fraction"].as_f64().unwrap_or(0.3) as f32;
    let overlap_threshold = cfg["overlap_threshold"].as_f64().unwrap_or(0.3) as f32;
    let entropy_threshold = cfg["entropy_threshold"].as_f64().unwrap_or(1.0) as f32;
    let randomized_seed = cfg["randomized_seed"].as_u64().unwrap_or(20260925);
    let config_id = cfg["configuration_id"].as_str().unwrap_or("").to_string();
    let deadline_us = cfg["deadline_us"].as_u64().unwrap_or(0) as u64;

    if !matches!(mode, "A" | "B" | "C" | "D" | "E" | "F" | "RAND") {
        panic!("mode must be A|B|C|D|E|F|RAND, got {mode}");
    }

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

    // ---- exact oracle for every run query ------------------------------
    let vlist: Vec<(u64, Vec<f32>)> = if canonical.is_empty() {
        Vec::new()
    } else {
        (0..n_docs).map(|d| (d as u64, canonical[d * dim..(d + 1) * dim].to_vec())).collect()
    };

    // ---- engine: build collection + insert docs ------------------------
    let dir = std::env::temp_dir().join(format!("c6pilot-{}-{}", std::process::id(), mode));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Async)
        .unwrap_or_else(|err| panic!("open engine: {err}"));

    let heads: Vec<&str> = cfg["collection_heads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    e.create_collection("c6p", dim, &heads).unwrap();

    let mut doc_map: HashMap<u64, usize> = HashMap::new();
    for row in 0..n_docs {
        let mut fields: HashMap<String, Value> = HashMap::new();
        fields.insert("idx".to_string(), json!(row));
        let mut rec = Record::new(fields);
        for h in &heads {
            let vec = doc_vecs.get(*h).unwrap_or_else(|| panic!("missing doc head {h}"));
            rec.k_vecs.insert((*h).to_string(), vec[row * dim..(row + 1) * dim].to_vec());
        }
        let uuid = e.insert_document("c6p", rec).unwrap();
        let nid = e.id_mapper.read().uuid_to_id(&uuid.parse().unwrap()).unwrap();
        doc_map.insert(nid, row);
    }

    let coll = e.get_collection("c6p").unwrap();

    // ---- C6: configure adaptive retrieval + optional cross-refine ------
    {
        let mut rc = coll.retrieval_config.write();
        rc.mode = if mode == "A" {
            RetrievalMode::SingleHead
        } else {
            RetrievalMode::FixedFusion
        };
        rc.candidate_budget = cand_budget;
        rc.min_candidates_per_head = min_cand_head;
        rc.ef_search = if mode == "A" || mode == "B" { Some(ef_search) } else { None };

        // Cross-refine (C5 interaction) — only for mode F
        rc.cross_refine = if mode == "F" {
            Some(CrossRefineConfig { lambda })
        } else {
            None
        };

        // Adaptive retrieval (C6) — for modes C/D/E/F/RAND
        rc.adaptive = if matches!(mode, "C" | "D" | "E" | "F" | "RAND") {
            let policy_type = match mode {
                "C" => AdaptivePolicyType::StaticEqual,
                "D" => AdaptivePolicyType::QueryAdaptive,
                "E" => AdaptivePolicyType::InteractionGuided,
                "F" => AdaptivePolicyType::InteractionGuided,
                "RAND" => AdaptivePolicyType::RandomizedControl,
                _ => AdaptivePolicyType::StaticEqual,
            };
            Some(AdaptiveRetrievalConfig {
                policy_type,
                stage1_fraction,
                overlap_threshold,
                entropy_threshold,
                randomized_seed,
            })
        } else {
            None
        };
    }

    let attend_heads: Vec<String> = cfg["attend_heads"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    // ---- run query protocol: warmup + timed ----------------------------
    let mut per_query = Vec::new();
    let mut latencies: Vec<f64> = Vec::new();
    let mut all_exact: Vec<Vec<u64>> = Vec::new();
    for (pos, &qr) in query_order.iter().enumerate() {
        let q = &queries[qr];
        let exact = if vlist.is_empty() { Vec::new() } else { brute_topk(&vlist, q, k) };
        let ids: Vec<u64> = exact.iter().map(|(id, _)| *id).collect();
        all_exact.push(ids.clone());
        let t = Instant::now();
        let deadline = if deadline_us > 0
            && pos >= warmup {
            Some(Instant::now() + std::time::Duration::from_micros(deadline_us))
        } else {
            None
        };
        match coll.attend_detailed_traced(
            &attend_heads, q, k, None, None, None, None, deadline,
        ) {
            Ok((res, stats, cross_trace, adaptive_trace)) => {
                let lat_us = t.elapsed().as_secs_f64() * 1e6;
                if pos >= warmup {
                    latencies.push(lat_us);
                }
                let hits: Vec<u64> = res.iter().map(|r| r.id).collect();
                let hits_rows: Vec<u64> = hits
                    .iter()
                    .map(|h| doc_map.get(h).copied().unwrap_or(*h as usize) as u64)
                    .collect();
                let relevant = qrels.get(&qr).cloned().unwrap_or_default();
                let rel_set: HashSet<u64> = relevant.keys().copied().collect();
                let hits_cnt = hits_rows.iter().filter(|id| rel_set.contains(id)).count();
                let recall = if rel_set.is_empty() { 0.0 }
                    else { hits_cnt as f32 / rel_set.len() as f32 };
                let r10 = hits_rows.iter().map(|&r| r).collect::<Vec<_>>();
                let recall_exact = if ids.is_empty() { 0.0 }
                    else {
                        let exact_set: HashSet<u64> = ids.iter().copied().collect();
                        hits_rows.iter().filter(|r| exact_set.contains(r)).count() as f32 / ids.len() as f32
                    };
                let ndcg = if rel_set.is_empty() { 0.0 } else { ndcg10(&r10, &relevant, k) };

                // C5 cross-refine ledger (mode F only)
                let cross_ledger = if mode == "F" {
                    let surprise_total: usize = cross_trace.surprise.iter().map(|v| v.len()).sum();
                    let round2_added_total: usize = cross_trace.round2_added.iter().map(|v| v.len()).sum();
                    json!({
                        "heads": cross_trace.heads,
                        "round1_counts": cross_trace.round1.iter().map(|v| v.len()).collect::<Vec<_>>(),
                        "surprise_counts": cross_trace.surprise.iter().map(|v| v.len()).collect::<Vec<_>>(),
                        "surprise_ids": cross_trace.surprise,
                        "refined": cross_trace.refined,
                        "round2_added_counts": cross_trace.round2_added.iter().map(|v| v.len()).collect::<Vec<_>>(),
                        "round2_added_ids": cross_trace.round2_added,
                        "surprise_ids_total": surprise_total,
                        "round2_added_ids_total": round2_added_total,
                        "interaction_union_additions": cross_trace.interaction_union_additions,
                        "per_head_capped": cross_trace.per_head_capped,
                        "union_pre": cross_trace.union_pre,
                        "union_post": cross_trace.union_post,
                        "ef_r1": cross_trace.ef_r1,
                        "ef_r2": cross_trace.ef_r2,
                    })
                } else {
                    json!(null)
                };

                // C6 adaptive ledger (modes C/D/E/F/RAND)
                let adaptive_ledger = if matches!(mode, "C" | "D" | "E" | "F" | "RAND") {
                    json!({
                        "initial_allocation": adaptive_trace.initial_allocation,
                        "redistributions": adaptive_trace.redistributions,
                        "final_allocation": adaptive_trace.final_allocation,
                        "budget_conserved": adaptive_trace.budget_conserved,
                        "total_candidates_used": adaptive_trace.total_candidates_used,
                        "total_ef_work_used": adaptive_trace.total_ef_work_used,
                    })
                } else {
                    json!(null)
                };

                per_query.push(json!({
                    "pos": pos, "query_row": qr,
                    "latency_us": lat_us,
                    "configuration_id": config_id,
                    "candidate_count": stats.union_size,
                    "heads_present": stats.heads_present,
                    "recall10_qrels": recall, "recall10_exact": recall_exact, "ndcg10_qrels": ndcg,
                    "hits": r10, "engine_hits": hits, "n_rel": rel_set.len(),
                    "relevant_ids": rel_set.iter().cloned().collect::<Vec<_>>(),
                    "cross_refine_ledger": cross_ledger,
                    "adaptive_ledger": adaptive_ledger,
                }));
            }
            Err(err) => {
                let lat_us = t.elapsed().as_secs_f64() * 1e6;
                let err_str = err.to_string();
                if err_str.contains("deadline") {
                    per_query.push(json!({
                        "pos": pos, "query_row": qr,
                        "latency_us": lat_us,
                        "configuration_id": config_id,
                        "deadline_exceeded": true, "error": err_str,
                        "recall10_qrels": 0.0, "recall10_exact": 0.0, "ndcg10_qrels": 0.0,
                        "hits": [], "engine_hits": [], "n_rel": if qrels.contains_key(&qr) { qrels[&qr].keys().len() } else { 0 },
                        "relevant_ids": [], "cross_refine_ledger": null, "adaptive_ledger": null,
                    }));
                } else {
                    panic!("attend error: {err}");
                }
            }
        }
    }

    let mut slat = latencies.clone();
    slat.sort_by(|a, b| a.total_cmp(b));
    let mean = if slat.is_empty() { 0.0 } else { slat.iter().sum::<f64>() / slat.len() as f64 };
    let rec_qrels: Vec<f32> = per_query.iter().map(|r| r["recall10_qrels"].as_f64().unwrap() as f32).collect();
    let rec_exact: Vec<f32> = per_query.iter().map(|r| r["recall10_exact"].as_f64().unwrap() as f32).collect();
    let mean_qrels = rec_qrels.iter().sum::<f32>() / rec_qrels.len().max(1) as f32;
    let mean_exact = rec_exact.iter().sum::<f32>() / rec_exact.len().max(1) as f32;

    let deadline_misses = per_query.iter().filter(|r| r.get("deadline_exceeded").map(|v| v.as_bool() == Some(true)).unwrap_or(false)).count();

    // Aggregate C5 cross-refine stats (mode F)
    let mut cand_changed = 0usize;
    let mut surprise_empty_queries = 0usize;
    if mode == "F" {
        for qr in &per_query {
            if let Some(led) = qr.get("cross_refine_ledger") {
                if led["surprise_ids_total"].as_u64().unwrap_or(0) == 0 {
                    surprise_empty_queries += 1;
                }
            }
        }
        for qr in &per_query {
            if let Some(led) = qr.get("cross_refine_ledger") {
                let added = led["interaction_union_additions"]
                    .as_array().map(|a| a.len()).unwrap_or(0);
                if added > 0 {
                    cand_changed += 1;
                }
            }
        }
    }

    // Aggregate C6 adaptive stats (modes C/D/E/F/RAND)
    let mut adaptive_changed = 0usize;
    let mut adaptive_redistributions = 0usize;
    if matches!(mode, "C" | "D" | "E" | "F" | "RAND") {
        let empty_vec = Vec::new();
        for qr in &per_query {
            if let Some(led) = qr.get("adaptive_ledger") {
                if led["redistributions"].as_array().map(|a| a.len()).unwrap_or(0) > 0 {
                    adaptive_redistributions += 1;
                }
                // Check if final allocation differs from initial
                let init = led["initial_allocation"].as_array().unwrap_or(&empty_vec);
                let final_a = led["final_allocation"].as_array().unwrap_or(&empty_vec);
                if init.len() != final_a.len() {
                    adaptive_changed += 1;
                } else {
                    for (i, a) in init.iter().enumerate() {
                        if i < final_a.len() && a["candidates"] != final_a[i]["candidates"] {
                            adaptive_changed += 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    let doc = json!({
        "subcommand": "pilot", "mode": mode, "ok": true,
        "engine": {
            "collection": "c6p",
            "heads": heads,
            "attend_heads": attend_heads,
            "retrieval_mode": mode,
            "candidate_budget": cand_budget,
            "min_candidates_per_head": min_cand_head,
            "max_candidates_per_head": max_cand_head,
            "total_ef_work": total_ef_work,
            "ef_search": ef_search,
            "lambda": lambda,
            "cross_refine": mode == "F",
            "adaptive_policy": mode,
            "stage1_fraction": stage1_fraction,
            "overlap_threshold": overlap_threshold,
            "entropy_threshold": entropy_threshold,
            "randomized_seed": randomized_seed,
            "configuration_id": config_id,
            "deadline_us": deadline_us,
        },
        "n_docs": n_docs, "n_queries": run_rows.len(),
        "warmup": warmup, "seed": seed, "k": k,
        "latency_us_post_warmup": {"count": slat.len(), "mean": mean,
            "p50": percentile(&slat, 0.50), "p90": percentile(&slat, 0.90), "p95": percentile(&slat, 0.95)},
        "recall10_qrels_mean": mean_qrels, "recall10_exact_mean": mean_exact,
        "deadline_exceeded_count": deadline_misses,
        "candidate_change_count": if mode == "F" { cand_changed } else { 0 },
        "surprise_empty_queries": if mode == "F" { surprise_empty_queries } else { 0 },
        "adaptive_changed_queries": if matches!(mode, "C" | "D" | "E" | "F" | "RAND") { adaptive_changed } else { 0 },
        "adaptive_redistributed_queries": if matches!(mode, "C" | "D" | "E" | "F" | "RAND") { adaptive_redistributions } else { 0 },
        "per_query": per_query, "exact_top10": all_exact,
    });
    let s = serde_json::to_string_pretty(&doc).unwrap();
    if let Some(op) = out_path { std::fs::write(op, &s).unwrap(); }
    println!("{s}");
}