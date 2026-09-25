//! c2pilot — C3 pilot run driver (additive; does NOT modify audited code).
//!
//! Drives the REAL AttentionDB engine over external embeddings (NFCorpus /
//! SciFact) with the C1 fixed-qrels latency protocol:
//!   - 5 fresh-process reps (one rep per process invocation)
//!   - seeded query order, first `warmup` queries executed but excluded
//!   - per-query latency + Recall@10 vs exact brute-force oracle + vs qrels
//!   - modes B1 (SingleHead/CANONICAL), B2 (FixedFusion 3 heads),
//!     B3 (LearnedGating + modelcard), B7 (Full default)
//!
//! Input contract (single JSON on stdin or --config path):
//! {
//!   "mode": "B1|B2|B3|B7",
//!   "k": 10,
//!   "seed": 20260925,
//!   "warmup": 20,
//!   "n_queries": 323,
//!   "collection_heads": ["HEAD-TITLE", ...],   // heads the collection is created with
//!   "attend_heads":     ["HEAD-TITLE", ...],   // heads passed to engine.attend()
//!   "doc_vectors": { "<head>": "<raw LE f32 path (n_docs*dim)>", ... },
//!   "query_vectors": "<raw LE f32 path (n_queries*dim)>",
//!   "n_docs": 3633,
//!   "dim": 384,
//!   "modelcard": "<path to ModelCard JSON>",   // B3 only
//!   "qrels": { "<query_row>": [<relevant doc rows>], ... },
//!   "subsample": [<query rows to run>]          // seeded subset; optional (default all)
//! }
//!
//! Emits a single JSON document on stdout (and to --out if given) containing
//! per-query rows, the in-process brute-force oracle top-10 for every query,
//! and parameter aggregates. Registration/environment are the Python
//! orchestrator's responsibility.

use attentiondb_core::collection::RetrievalMode;
use attentiondb_core::AttentionEngine;
use attentiondb_learned::gating_v2::ModelCard;
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

/// Exact top-k: score DESC, id ASC (engine tie order).
fn brute_topk(vecs: &[(u64, Vec<f32>)], q: &[f32], k: usize) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = vecs
        .iter()
        .map(|(id, v)| (*id, cosine(q, v)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(k.min(scored.len()));
    scored
}

struct Lcg(u64);
impl Lcg {
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
    let mut rng = Lcg(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.usize_below(i + 1);
        idx.swap(i, j);
    }
    idx
}

fn ndcg10(retrieved: &[u64], relevant: &HashMap<u64, f32>, k: usize) -> f32 {
    let mut dcg = 0.0f32;
    for (rank, id) in retrieved.iter().enumerate() {
        if rank >= k { break; }
        let rel = relevant.get(id).copied().unwrap_or(0.0);
        dcg += rel / (rank as f32 + 2.0).log2();
    }
    // ideal = top k relevant by grade (grades are small ints; sort desc)
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

    // ---- C4 overrides (budget/ef knobs, optional; defaults == C1/C3) ----
    let cand_budget = cfg["candidate_budget"].as_u64().unwrap_or(500) as usize;
    let min_cand_head = cfg["min_candidates_per_head"].as_u64().unwrap_or(20) as usize;
    let cand_mult = cfg["candidate_multiplier"].as_u64().unwrap_or(0) as usize;
    let ef_search = cfg["ef_search"].as_u64().unwrap_or(64) as usize;
    let ef_construction = cfg["ef_construction"].as_u64().unwrap_or(400) as usize;
    let hnsw_m = cfg["m"].as_u64().unwrap_or(16) as usize;
    let config_id = cfg["configuration_id"].as_str().unwrap_or("").to_string();

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
    // query vectors: row-major n_queries*dim
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

    // ---- qrels: query row -> (doc row -> grade) ------------------------
    let mut qrels: HashMap<usize, HashMap<u64, f32>> = HashMap::new();
    if let Some(qr) = cfg["qrels"].as_object() {
        for (qrow, rels) in qr {
            let qr_i = qrow.parse::<usize>().unwrap();
            let mut m = HashMap::new();
            if let Some(arr) = rels.as_array() {
                for v in arr {
                    // entries are either doc row ints (grade 1) or [row, grade]
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

    // ---- deduce query run set (seeded shuffle is orchestrator's job) ---
    let run_rows: Vec<usize> = if let Some(sub) = cfg["subsample"].as_array() {
        sub.iter().map(|v| v.as_u64().unwrap() as usize).collect()
    } else {
        (0..n_queries).collect()
    };
    // deterministic per-process order inside the rep (C1: seeded order)
    let order = shuffled_indices(run_rows.len(), seed);
    let query_order: Vec<usize> = order.iter().map(|&o| run_rows[o]).collect();

    // ---- B0-style exact oracle (canonical) for every run query ---------
    let vlist: Vec<(u64, Vec<f32>)> = if canonical.is_empty() {
        Vec::new()
    } else {
        (0..n_docs).map(|d| (d as u64, canonical[d * dim..(d + 1) * dim].to_vec())).collect()
    };
    if mode == "B0" {
        // pure exact-oracle cell: no engine. Report oracle rows only.
        let mut per_query = Vec::new();
        let mut all_exact: Vec<Vec<u64>> = Vec::new();
        for (pos, &qr) in query_order.iter().enumerate() {
            let exact = brute_topk(&vlist, &queries[qr], k);
            let ids: Vec<u64> = exact.iter().map(|(id, _)| *id).collect();
            all_exact.push(ids.clone());
            let relevant = qrels.get(&qr).cloned().unwrap_or_default();
            let rel_set: HashSet<u64> = relevant.keys().copied().collect();
            let hits_cnt = ids.iter().filter(|id| rel_set.contains(id)).count();
            let recall = if rel_set.is_empty() { 0.0 }
                else { hits_cnt as f32 / rel_set.len() as f32 };
            let ndcg = if rel_set.is_empty() { 0.0 } else { ndcg10(&ids, &relevant, k) };
            per_query.push(json!({
                "pos": pos, "query_row": qr,
                "recall10_qrels": recall, "ndcg10_qrels": ndcg,
                "hits": ids,
                "n_rel": rel_set.len(),
            }));
        }
        let doc = json!({"subcommand": "pilot", "mode": "B0", "ok": true,
            "n_queries": run_rows.len(), "per_query": per_query, "exact_top10": all_exact});
        let s = serde_json::to_string_pretty(&doc).unwrap();
        if let Some(op) = out_path { std::fs::write(op, &s).unwrap(); }
        println!("{s}");
        return;
    }

    // ---- engine: build collection + insert docs ------------------------
    let dir = std::env::temp_dir().join(format!("c2pilot-{}-{}", std::process::id(), mode));
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
    e.create_collection("c3p", dim, &heads).unwrap();

    // insert docs, tracking numeric id (register() mints sequentially from 0)
    let mut doc_map: HashMap<u64, usize> = HashMap::new(); // numeric id -> row
    for row in 0..n_docs {
        let mut fields: HashMap<String, Value> = HashMap::new();
        fields.insert("idx".to_string(), json!(row));
        let mut rec = Record::new(fields);
        for h in &heads {
            let vec = doc_vecs.get(*h).unwrap_or_else(|| panic!("missing doc head {h}"));
            rec.k_vecs.insert((*h).to_string(), vec[row * dim..(row + 1) * dim].to_vec());
        }
        let uuid = e.insert_document("c3p", rec).unwrap();
        let nid = e.id_mapper.read().uuid_to_id(&uuid.parse().unwrap()).unwrap();
        doc_map.insert(nid, row);
    }

    let coll = e.get_collection("c3p").unwrap();
    let rc0_cand_mult = coll.retrieval_config.read().candidate_multiplier;
    {
        let mut rc = coll.retrieval_config.write();
        rc.mode = match mode {
            "B1" => RetrievalMode::SingleHead,
            "B2" => RetrievalMode::FixedFusion,
            "B3" => RetrievalMode::LearnedGating,
            "B7" => RetrievalMode::Full,
            other => panic!("unsupported mode {other}"),
        };
        rc.candidate_budget = cand_budget;
        rc.min_candidates_per_head = min_cand_head;
        if cand_mult > 0 {
            rc.candidate_multiplier = cand_mult;
        }
    }
    if mode == "B3" {
        let card_path = cfg["modelcard"].as_str().unwrap();
        let card = ModelCard::load(std::path::Path::new(card_path)).unwrap();
        e.install_gating_card("c3p", card).unwrap();
    }

    // ---- C4: apply HNSW settings (ef_search / ef_construction / m) -----
    let mut hsettings = coll.settings.write();
    hsettings.ef_search = ef_search;
    hsettings.ef_construction = ef_construction;
    hsettings.max_nb_connection = hnsw_m;
    hsettings.enable_exact_reranking = true;
    drop(hsettings);

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
        let (res, stats) = coll
            .attend_detailed_with_stats(
                &attend_heads, q, k, None, None, None, None, None,
            )
            .unwrap();
        let lat_us = t.elapsed().as_secs_f64() * 1e6;
        if pos >= warmup {
            latencies.push(lat_us);
        }
        let hits: Vec<u64> = res.iter().map(|r| r.id).collect();
        // map engine numeric ids back to doc rows
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
        per_query.push(json!({
            "pos": pos, "query_row": qr,
            "latency_us": lat_us,
            "configuration_id": config_id,
            "candidate_count": stats.union_size,
            "heads_present": stats.heads_present,
            "recall10_qrels": recall, "recall10_exact": recall_exact, "ndcg10_qrels": ndcg,
            "hits": r10, "engine_hits": hits, "n_rel": rel_set.len(),
            "relevant_ids": rel_set.iter().cloned().collect::<Vec<_>>(),
        }));
    }

    let mut slat = latencies.clone();
    slat.sort_by(|a, b| a.total_cmp(b));
    let mean = if slat.is_empty() { 0.0 } else { slat.iter().sum::<f64>() / slat.len() as f64 };
    let rec_qrels: Vec<f32> = per_query.iter().map(|r| r["recall10_qrels"].as_f64().unwrap() as f32).collect();
    let rec_exact: Vec<f32> = per_query.iter().map(|r| r["recall10_exact"].as_f64().unwrap() as f32).collect();
    let mean_qrels = rec_qrels.iter().sum::<f32>() / rec_qrels.len().max(1) as f32;
    let mean_exact = rec_exact.iter().sum::<f32>() / rec_exact.len().max(1) as f32;

    let doc = json!({
        "subcommand": "pilot", "mode": mode, "ok": true,
        "engine": {
            "collection": "c3p",
            "heads": heads,
            "attend_heads": attend_heads,
            "retrieval_mode": mode,
            "candidate_budget": cand_budget,
            "min_candidates_per_head": min_cand_head,
            "candidate_multiplier": if cand_mult > 0 { cand_mult } else { rc0_cand_mult },
            "ef_search": ef_search, "ef_construction": ef_construction, "m": hnsw_m,
            "modelcard": if mode == "B3" { cfg["modelcard"].as_str().unwrap().to_string() } else { String::new() },
            "configuration_id": config_id,
        },
        "n_docs": n_docs, "n_queries": run_rows.len(),
        "warmup": warmup, "seed": seed, "k": k,
        "latency_us_post_warmup": {"count": slat.len(), "mean": mean,
            "p50": percentile(&slat, 0.50), "p90": percentile(&slat, 0.90), "p95": percentile(&slat, 0.95)},
        "recall10_qrels_mean": mean_qrels, "recall10_exact_mean": mean_exact,
        "per_query": per_query, "exact_top10": all_exact,
    });
    let s = serde_json::to_string_pretty(&doc).unwrap();
    if let Some(op) = out_path { std::fs::write(op, &s).unwrap(); }
    println!("{s}");
}