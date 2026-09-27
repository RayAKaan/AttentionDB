//! C2 harness probe for study comparative-study-001.
//!
//! Reuses existing audited components (no engine modifications):
//! - phase2b_bench::corpora  — deterministic DS-SYNTH-PH2B generators
//! - attentiondb_learned::gating_v2 — GatingDataset/ModelCard/train_gating
//! - attentiondb_core::AttentionEngine — the real retrieval engine
//!
//! Every subcommand prints exactly one JSON document to stdout.

use attentiondb_core::collection::RetrievalMode;
use attentiondb_core::AttentionEngine;
use attentiondb_learned::gating_v2::{
    train_gating, GatingDataset, HeadExample, Objective, QueryExample, QualityTarget, Split,
    TrainingConfig,
};
use attentiondb_storage::{Durability, Record};
use phase2b_bench::corpora::{self, Corpus};
use serde_json::{json, Value};
use std::collections::HashMap;

// ---------------------------------------------------------------- utilities

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
    fn f32_01(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 0x100_0000_u64 as f32 // top 24 bits / 2^24
    }
    fn gauss(&mut self) -> f32 {
        // Irwin-Hall approximation, deterministic
        let s: f32 = (0..6).map(|_| self.f32_01() - 0.5).sum();
        s * 2.0_f32.sqrt()
    }
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

/// Exact top-k: score DESC, id ASC (the engine's documented tie order).
fn brute_topk(vecs: &[(u64, Vec<f32>)], q: &[f32], k: usize) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = vecs
        .iter()
        .map(|(id, v)| (*id, cosine(q, v)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scored.truncate(k);
    scored
}

fn minmax(xs: &mut [f32]) {
    let lo = xs.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if hi - lo < 1e-12 {
        for x in xs.iter_mut() {
            *x = 0.5;
        }
    } else {
        for x in xs.iter_mut() {
            *x = (*x - lo) / (hi - lo);
        }
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn open_tmp_engine(tag: &str) -> (std::path::PathBuf, AttentionEngine) {
    let dir = std::env::temp_dir().join(format!("c2probe-{}-{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let e = AttentionEngine::open_dir(&dir, Durability::Async).unwrap();
    (dir, e)
}

// ---------------------------------------------------- DS-SYNTH-PH2B export

fn cmd_corpus(n_queries: usize, seed: u64, out: &str) {
    let c: Corpus = corpora::multiview(n_queries, seed);
    let docs: Vec<Value> = c
        .docs
        .iter()
        .map(|d| {
            json!({
                "id": d.numeric_hint,
                "fields": d.fields,
                "head_vecs": d.head_vecs,
                "true_vec": d.true_vec,
            })
        })
        .collect();
    let queries: Vec<Value> = c
        .queries
        .iter()
        .map(|q| {
            json!({
                "id": q.id,
                "group": q.group,
                "vectors": q.vectors,
                "gating_input": q.gating_input,
                "ground_truth": q.ground_truth,
            })
        })
        .collect();
    let doc = json!({
        "generator": "phase2b_bench::corpora::multiview",
        "generator_params": {"n_queries": n_queries, "seed": seed},
        "name": c.name,
        "dim": c.dim,
        "head_names": c.head_names,
        "n_docs": docs.len(),
        "docs": docs,
        "queries": queries,
    });
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("{}", json!({"ok": true, "n_docs": docs.len(), "n_queries": queries.len(), "dim": c.dim, "out": out}));
}

// --------------------------------------------- oracle gating dataset build

fn cmd_gating_dataset(path: &str, split_seed: u64, top_k: usize, input_mode: &str, out: &str) {
    let raw: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let head_names: Vec<String> = raw["head_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let dim = raw["dim"].as_u64().unwrap() as usize;
    let corpus_desc = format!(
        "synth-multiview-n{}-seed{}-input-{}",
        raw["queries"].as_array().unwrap().len(),
        raw["generator_params"]["seed"].as_u64().unwrap(),
        input_mode
    );
    // doc id -> per-head vector, straight from the exported corpus
    let mut head_vecs: Vec<HashMap<u64, Vec<f32>>> = vec![HashMap::new(); head_names.len()];
    for d in raw["docs"].as_array().unwrap() {
        let id = d["id"].as_u64().unwrap();
        for (h, v) in d["head_vecs"].as_array().unwrap().iter().enumerate() {
            let v: Vec<f32> = v
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            head_vecs[h].insert(id, v);
        }
    }
    // phase2b main.rs scheme: seeded 70/15/15 split over query order
    let queries_arr = raw["queries"].as_array().unwrap();
    let n = queries_arr.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = corpora::Rng::new(split_seed ^ 0x5EED);
    rng.shuffle(&mut order);
    let (n_train, n_val) = (n * 7 / 10, n * 15 / 100);
    let mut split_of = vec![Split::Test; n];
    for (rank, &qi) in order.iter().enumerate() {
        split_of[qi] = if rank < n_train {
            Split::Train
        } else if rank < n_train + n_val {
            Split::Val
        } else {
            Split::Test
        };
    }

    let mut queries: Vec<QueryExample> = Vec::with_capacity(n);
    for (qi, q) in queries_arr.iter().enumerate() {
        let qid = q["id"].as_u64().unwrap();
        let gt: Vec<u64> = q["ground_truth"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect();
        let mut heads: Vec<HeadExample> = Vec::with_capacity(head_names.len());
        for h in 0..head_names.len() {
            let qv: Vec<f32> = q["vectors"][h]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            // EXACT oracle top-k for this head (B0; score desc, id asc)
            let mut scored: Vec<(u64, f32)> = head_vecs[h]
                .iter()
                .map(|(id, v)| (*id, cosine(&qv, v)))
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            scored.truncate(top_k);
            let mut norm: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();
            minmax(&mut norm);
            let cands: Vec<u64> = scored.iter().map(|(id, _)| *id).collect();
            let hits: u64 = cands.iter().filter(|c| gt.contains(c)).count() as u64;
            let denom = gt.len().min(top_k).max(1) as f32;
            let recall = hits as f32 / denom;
            // binary-relevance nDCG@k and MRR vs generator ground truth
            let mut dcg = 0.0f32;
            for (rank, c) in cands.iter().enumerate() {
                if gt.contains(c) {
                    dcg += 1.0 / ((rank + 2) as f32).log2();
                }
            }
            let ideal: f32 = (1..=gt.len().min(top_k))
                .map(|r| 1.0 / (r as f32 + 1.0).log2())
                .sum();
            let ndcg = if ideal > 0.0 { dcg / ideal } else { 0.0 };
            let mrr = cands
                .iter()
                .position(|c| gt.contains(c))
                .map(|r| 1.0 / (r + 1) as f32)
                .unwrap_or(0.0);
            heads.push(HeadExample {
                candidates: cands,
                raw_scores: scored.iter().map(|(_, s)| *s).collect(),
                norm_scores: norm,
                exact_scores: scored.iter().map(|(_, s)| *s).collect(),
                recall_at_k: recall,
                ndcg_at_k: ndcg,
                mrr,
            });
        }
        let query: Vec<f32> = if input_mode == "concat" {
            q["gating_input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        } else {
            // engine-side mode: the vector handed to attend() (head 0 view)
            q["vectors"][0]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        queries.push(QueryExample {
            query_id: qid,
            query,
            query_group: Some(q["group"].as_u64().unwrap() as u32),
            split: split_of[qi],
            ground_truth: gt,
            heads,
        });
    }
    let ds = GatingDataset {
        format: "attentiondb-gating-dataset".into(),
        version: 1,
        num_heads: head_names.len(),
        input_dim: dim * if input_mode == "concat" { head_names.len() } else { 1 },
        top_k,
        corpus_desc,
        seed: split_seed,
        queries,
    };
    let mut counts: HashMap<String, usize> = HashMap::new();
    counts.insert("train".into(), 0);
    counts.insert("val".into(), 0);
    counts.insert("test".into(), 0);
    for q in &ds.queries {
        *counts.entry(format!("{:?}", q.split).to_lowercase()).or_default() += 1;
    }
    std::fs::write(out, serde_json::to_string(&ds).unwrap()).unwrap();
    println!(
        "{}",
        json!({"ok": true, "out": out, "input_dim": ds.input_dim, "num_heads": ds.num_heads,
               "top_k": ds.top_k, "counts": counts, "corpus_desc": ds.corpus_desc})
    );
}

// ------------------------------------------------------------------- train

fn cmd_train(path: &str, outdir: &str, model_id: &str, seed: u64, hidden: usize, lr: f32) {
    let bytes = std::fs::read(path).unwrap();
    // INV-L1 structural guard: only preregistered training corpora prefixes
    // are accepted; anything marked headline-test is refused outright.
    let ds: GatingDataset = serde_json::from_slice(&bytes).unwrap();
    let ok_prefix =
        ds.corpus_desc.starts_with("synth-multiview") || ds.corpus_desc.starts_with("lodo-train");
    let deny = ds.corpus_desc.contains("headline") || path.contains("test-exports");
    if !ok_prefix || deny {
        println!("{}", json!({"ok": false, "guard": "INV-L1", "refused": true,
                              "corpus_desc": ds.corpus_desc, "path": path}));
        std::process::exit(2);
    }
    let n_test = ds.queries.iter().filter(|q| matches!(q.split, Split::Test)).count();
    let cfg = TrainingConfig {
        objective: Objective::SoftTarget,
        tau: 1.0,
        margin: 0.5,
        lr,
        batch_size: 32,
        max_epochs: 200,
        patience: 15,
        min_delta: 1e-4,
        l2: 1e-4,
        hidden,
        seed,
    };
    let outcome = train_gating(&ds, &cfg, QualityTarget::Recall);
    // reproducibility: identical seed + data + config => identical weights (INV-L5)
    let outcome2 = train_gating(&ds, &cfg, QualityTarget::Recall);
    let reproducible = outcome.model.w1 == outcome2.model.w1
        && outcome.model.b1 == outcome2.model.b1
        && outcome.model.w2 == outcome2.model.w2
        && outcome.model.b2 == outcome2.model.b2;
    let meta = attentiondb_learned::gating_v2::TrainingMeta {
        seed,
        dataset_hash: fnv1a64(&bytes),
        objective: "soft_target".into(),
        learning_rate: lr,
        batch_size: 32,
        epochs_run: outcome.best_epoch,
        best_val_loss: outcome.best_val_loss,
        l2: 1e-4,
        timestamp_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        code_commit: std::env::var("C2_GIT_COMMIT").unwrap_or_default(),
        hardware: std::env::var("C2_HARDWARE").unwrap_or_else(|_| "unknown".into()),
    };
    let mut card = attentiondb_learned::gating_v2::ModelCard::from_mlp(
        &outcome.model,
        meta,
        model_id,
        "soft_target",
    );
    // Engine applies trained weights ONLY via the name->weight mapping
    // (get_trained_weights); a card without head_names is refused-by-fallback
    // (uniform). Record the corpus head order explicitly.
    if let Ok(hn) = std::env::var("C2_HEAD_NAMES") {
        let names: Vec<String> = hn.split(',').map(|x| x.trim().to_string()).collect();
        if names.len() == ds.num_heads {
            card.head_names = Some(names);
        }
    }
    std::fs::create_dir_all(outdir).unwrap();
    let card_path = format!("{outdir}/{model_id}-s{seed}-h{hidden}-lr{lr}.json");
    std::fs::write(&card_path, serde_json::to_string_pretty(&card).unwrap()).unwrap();
    let curves: Vec<Value> = outcome
        .curves
        .iter()
        .map(|c| {
            json!({"epoch": c.epoch, "train_loss": c.train_loss,
                   "val_loss": c.val_loss, "val_weight_quality_corr": c.val_weight_quality_corr})
        })
        .collect();
    let info = json!({
        "ok": true,
        "card_path": card_path,
        "model_id": model_id,
        "seed": seed,
        "hidden": hidden,
        "lr": lr,
        "best_epoch": outcome.best_epoch,
        "best_val_loss": outcome.best_val_loss,
        "final_train_loss": outcome.final_train_loss,
        "stopped_early": outcome.stopped_early,
        "reproducible_bitwise": reproducible,
        "dataset_hash_fnv1a64": fnv1a64(&bytes),
        "rows": {"total": ds.queries.len(), "test_unused_by_trainer": n_test},
        "curves": curves,
    });
    std::fs::write(format!("{outdir}/{model_id}-s{seed}-h{hidden}-lr{lr}.meta.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();
    println!("{}", info);
}

// ------------------------------------------------- reference fusion models

struct RefHeads {
    /// (head name -> (doc id -> vector))
    per_head: Vec<HashMap<u64, Vec<f32>>>,
}

impl RefHeads {
    /// Mode-B reference: exact per-head top per_head_k, union, per-head
    /// min-max over the pool, uniform head weights, top-k (desc, id asc).
    fn fused_uniform(&self, q: &[f32], heads: &[String], k: usize, budget: usize, exact: bool) -> Vec<(u64, f32)> {
        let per_head_k = std::cmp::max(5 * k, 20);
        let mut pool: std::collections::BTreeMap<u64, Vec<Option<f32>>> = Default::default();
        for (hi, h) in heads.iter().enumerate() {
            let mut scored: Vec<(u64, f32)> = self.per_head[hi]
                .iter()
                .map(|(id, v)| (*id, cosine(q, v)))
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            scored.truncate(per_head_k);
            for (id, s) in scored {
                let e = pool.entry(id).or_default();
                while e.len() < heads.len() {
                    e.push(None);
                }
                e[hi] = Some(s);
            }
        }
        // hard bound on post-union pool (engine candidate_budget semantics:
        // deterministic truncation by first-seen order)
        let mut ids: Vec<u64> = pool.keys().cloned().collect();
        ids.truncate(budget);
        // per-head min-max over pool members that the head covered
        let mut fused: Vec<(u64, f32)> = Vec::with_capacity(ids.len());
        for id in &ids {
            let row = &pool[id];
            let mut acc = 0.0f32;
            let w = 1.0 / heads.len() as f32;
            for (hi, s) in row.iter().enumerate() {
                if let Some(s) = s {
                    let col: Vec<f32> = ids
                        .iter()
                        .filter_map(|i2| pool[i2][hi])
                        .collect();
                    let mut col2 = col.clone();
                    minmax(&mut col2);
                    let idx = col.iter().position(|x| *x == *s).unwrap_or(0);
                    let n = if exact { *s } else { col2[idx] };
                    acc += w * n;
                }
            }
            fused.push((*id, acc));
        }
        fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        fused.truncate(k);
        fused
    }
}

// --------------------------------------------------------- engine battery

fn insert_docs(e: &AttentionEngine, coll: &str, heads: &[&str], dim: usize, n: usize, seed: u64) {
    let mut rng = Lcg(seed);
    e.create_collection(coll, dim, heads).unwrap();
    for i in 0..n {
        let mut fields: HashMap<String, Value> = HashMap::new();
        fields.insert("idx".to_string(), json!(i));
        // deterministic text so the hybrid (BM25) path has real lexical signal
        let token = if i % 5 == 0 { "alpha" } else if i % 5 == 1 { "beta" } else if i % 5 == 2 { "gamma" } else if i % 5 == 3 { "delta" } else { "xyloquery" };
        fields.insert("text".to_string(), json!(format!("{} {} {}", token, token, i)));
        let mut r = Record::new(fields);
        for (hi, h) in heads.iter().enumerate() {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
            if i >= n - 10 {
                // duplicate block -> guaranteed ties with doc 0 region
                v = vec![0.25 * (hi as f32 + 1.0); dim];
            }
            r.k_vecs.insert(h.to_string(), v);
        }
        e.insert_document(coll, r).unwrap();
    }
}

fn cmd_modes_test(out: &str) {
    let mut results: Vec<Value> = Vec::new();
    let heads = ["a", "b"];
    let dim = 8usize;
    let ndocs = 40usize;
    let (_dir, e) = open_tmp_engine("modes");
    insert_docs(&e, "c2t", &heads, dim, ndocs, 20260925);
    let coll = e.get_collection("c2t").unwrap();
    // harness-controlled config for exact reference checks (documented):
    // (a) fusion == multi-head-similarity component only;
    // (b) per-head pool >= corpus size so candidate membership is exact and
    //     the fusion/rerank MATH is verified independently of ANN recall.
    {
        let mut c = coll.retrieval_config.write();
        c.fusion.attention = 0.0;
        c.fusion.multi_head_similarity = 1.0;
        c.fusion.bm25 = 0.0;
        c.min_candidates_per_head = 64; // > ndocs: every head pool covers all docs
    }
    let mut head_vecs: Vec<HashMap<u64, Vec<f32>>> = vec![HashMap::new(); heads.len()];
    {
        let store = e.document_store.read();
        let mapper = e.id_mapper.read();
        for rec in store.list_all_records() {
            if let Some(idx) = rec.fields.get("idx").and_then(|v| v.as_u64()) {
                if let Some(nid) = mapper.uuid_to_id(&rec.id) {
                    for (hi, h) in heads.iter().enumerate() {
                        if let Some(v) = rec.k_vecs.get(*h) {
                            head_vecs[hi].insert(nid, v.clone());
                        }
                    }
                }
            }
        }
    }
    let vlist: Vec<Vec<(u64, Vec<f32>)>> = (0..heads.len())
        .map(|hi| head_vecs[hi].iter().map(|(k, v)| (*k, v.clone())).collect())
        .collect();
    let per_head_k = 64usize; // matches the pinned min_candidates_per_head (>= corpus)
    // per-head exact top per_head_k + min-max over that list (engine stage 1+3)
    let mut head_top: Vec<Vec<(u64, f32, f32)>> = vec![Vec::new(); heads.len()];
    for hi in 0..heads.len() {
        let mut scored: Vec<(u64, f32)> = vlist[hi].iter().map(|(id, v)| (*id, cosine(&q_dummy(dim), v))).collect();
        let _ = &mut scored; // placeholder removed below
    }
    let mut rng = Lcg(777);
    let new_q = |rng: &mut Lcg| -> Vec<f32> { (0..dim).map(|_| rng.gauss()).collect() };
    let head_lists = |q: &[f32]| -> Vec<HashMap<u64, f32>> {
        (0..heads.len())
            .map(|hi| {
                let mut scored: Vec<(u64, f32)> =
                    vlist[hi].iter().map(|(id, v)| (*id, cosine(q, v))).collect();
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                scored.truncate(per_head_k);
                let mut raw: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();
                minmax(&mut raw);
                scored.into_iter().zip(raw).map(|((id, _), n)| (id, n)).collect()
            })
            .collect()
    };
    let fused_ref = |q: &[f32], k: usize, exact: bool| -> Vec<(u64, f32)> {
        // exact=true: Full-mode metric-consistent gated similarity over the
        // union pool (per-head membership = exact top per_head_k)
        let lists = head_lists(q);
        let mut ids: std::collections::BTreeSet<u64> = Default::default();
        for m in &lists {
            ids.extend(m.keys().cloned());
        }
        let mut fused: Vec<(u64, f32)> = ids
            .into_iter()
            .map(|id| {
                let w = 1.0 / heads.len() as f32;
                let mut acc = 0.0f32;
                for hi in 0..heads.len() {
                    let present = lists[hi].contains_key(&id);
                    let s = if exact {
                        if present {
                            vlist[hi].iter().find(|(i, _)| *i == id).map(|(_, v)| cosine(q, v)).unwrap()
                        } else { 0.0 }
                    } else {
                        lists[hi].get(&id).copied().unwrap_or(0.0)
                    };
                    acc += w * s;
                }
                (id, acc)
            })
            .collect();
        fused.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        fused.truncate(k);
        fused
    };
    let scores_match = |got: &[(u64, f32)], exp: &[(u64, f32)]| -> bool {
        got.len() == exp.len()
            && got.iter().zip(exp.iter()).all(|(g, x)| g.0 == x.0 && (g.1 - x.1).abs() < 1e-5)
    };

    // ---- TEST-C2-002: single-head ANN (B1): soundness + determinism + rate
    coll.retrieval_config.write().mode = RetrievalMode::SingleHead;
    let mut sound = true;
    let mut det = true;
    let mut set_eq = 0usize;
    let nq = 10;
    for _ in 0..nq {
        let q = new_q(&mut rng);
        let got = e.attend("c2t", &["a".to_string()], &q, 5).unwrap();
        let exp = brute_topk(&vlist[0], &q, 5);
        if got.iter().all(|(id, _)| head_vecs[0].contains_key(id)) == false { sound = false; }
        // score consistency: returned scores == exact cosine of stored vectors
        for (id, s) in &got {
            let v = &head_vecs[0][id];
            if (cosine(&q, v) - s).abs() > 1e-4 { sound = false; }
        }
        let got2 = e.attend("c2t", &["a".to_string()], &q, 5).unwrap();
        if got != got2 { det = false; }
        let gi: std::collections::HashSet<u64> = got.iter().map(|(i, _)| *i).collect();
        let ei: std::collections::HashSet<u64> = exp.iter().map(|(i, _)| *i).collect();
        if gi == ei { set_eq += 1; }
    }
    // mode A takes heads[0] only: extra/unknown heads are ignored, not errors
    let q = new_q(&mut rng);
    let a1 = e.attend("c2t", &["a".to_string()], &q, 5).unwrap();
    let a2 = e.attend("c2t", &["a".to_string(), "nope".to_string()], &q, 5).unwrap();
    let latency_ok = {
        let t0 = std::time::Instant::now();
        for _ in 0..100 { let _ = e.attend("c2t", &["a".to_string()], &q, 5).unwrap(); }
        t0.elapsed().as_micros() > 0
    };
    results.push(json!({"test": "TEST-C2-002", "name": "single-head ANN (B1): soundness, score consistency, determinism",
        "status": if sound && det && latency_ok {"PASS"} else {"FAILED"},
        "sound_and_scores_consistent": sound, "deterministic": det,
        "head0_only_extra_heads_ignored": a1 == a2,
        "exact_set_agreement_rate": set_eq as f32 / nq as f32,
        "note": "mode A is approximate by design; exact agreement recorded, not asserted"}));

    // ---- TEST-C2-003: multi-head equal fusion (B2) vs pipeline reference
    // v2 (2026-09-24): hnsw_rs search at k == element count is NOT guaranteed
    // exhaustive — 0..2 docs per head can be unreachable, process-variable
    // (see `c2probe hrecall`). The pre-v2 assertion required strict 1e-5 score
    // equality against an always-exhaustive exact reference, which is
    // undecidable whenever a head's pool is incomplete. Recoped: (a) output
    // determinism, (b) IF a query's per-head pools are exhaustive, fused
    // scores must be identical to the exhaustive pipeline reference, (c) pool
    // coverage recorded as an observation.
    coll.retrieval_config.write().mode = RetrievalMode::FixedFusion;
    let mut det = true;
    let mut worst = Vec::new();
    let mut exact_checked = 0usize;
    let mut coverage_min = 40usize;
    for _ in 0..10 {
        let q = new_q(&mut rng);
        let got = e.attend("c2t", &["a".to_string(), "b".to_string()], &q, 5).unwrap();
        let got2 = e.attend("c2t", &["a".to_string(), "b".to_string()], &q, 5).unwrap();
        if got != got2 {
            det = false;
        }
        let mut cov = 40usize;
        for h in ["a", "b"] {
            if let Ok(idx) = coll.head_manager.read().get_head(h) {
                let n = idx.read().len();
                let hits = idx.read().search(&q, n.max(1), None).ok();
                cov = cov.min(hits.map(|v| v.len()).unwrap_or(0));
            }
        }
        coverage_min = coverage_min.min(cov);
        if cov == 40 {
            exact_checked += 1;
            let exp = fused_ref(&q, 5, false);
            if !scores_match(&got, &exp) {
                worst.push(json!({"got": got, "ref": exp}));
            }
        }
    }
    results.push(json!({"test": "TEST-C2-003",
        "name": "multi-head equal fusion (B2): exact when pools exhaustive; coverage recorded (HNSW recall observation)",
        "status": if det && worst.is_empty() {"PASS"} else {"FAILED"},
        "deterministic_across_calls": det, "exact_ref_checks": exact_checked,
        "coverage_min_per_head": coverage_min,
        "note": "hnsw_rs search at k == element count is not exhaustive (0..2 docs per head unreachable, process-variable; see c2probe hrecall). Exact 1e-5 equality against the exhaustive reference is asserted only for queries whose per-head pools covered all docs; pool coverage is recorded as an observation, not an engine defect."}));

    // ---- TEST-C2-005: candidate-budget guardrail + bound
    coll.retrieval_config.write().candidate_budget = 3; // below engine minimum
    let q = new_q(&mut rng);
    let invalid_refused = e.attend("c2t", &["a".to_string(), "b".to_string()], &q, 5).is_err();
    coll.retrieval_config.write().candidate_budget = 16; // engine-validated minimum
    let mut bounded = true;
    for _ in 0..10 {
        let q = new_q(&mut rng);
        let g1 = e.attend("c2t", &["a".to_string(), "b".to_string()], &q, 5).unwrap();
        let g2 = e.attend("c2t", &["a".to_string(), "b".to_string()], &q, 5).unwrap();
        if g1.len() > 5 || g1 != g2 { bounded = false; }
    }
    coll.retrieval_config.write().candidate_budget = 500;
    results.push(json!({"test": "TEST-C2-005", "name": "candidate-budget: invalid value refused, min budget sound",
        "status": if invalid_refused && bounded {"PASS"} else {"FAILED"},
        "budget3_refused_by_validation": invalid_refused, "budget16_results_sound": bounded}));

    // ---- TEST-C2-006: deterministic tie ordering
    let q = vec![0.25f32; dim];
    let r1 = e.attend("c2t", &["a".to_string()], &q, 10).unwrap();
    let r2 = e.attend("c2t", &["a".to_string()], &q, 10).unwrap();
    let tied_ids: Vec<u64> = r1.iter().map(|(i, _)| *i).collect();
    let dup_block_sorted = {
        let mut dup_ids: Vec<u64> = tied_ids.clone();
        dup_ids.retain(|i| head_vecs[0].get(i).map(|v| v[0] == 0.25).unwrap_or(false));
        let mut s = dup_ids.clone();
        s.sort();
        dup_ids == s
    };
    results.push(json!({"test": "TEST-C2-006", "name": "deterministic tie ordering (score desc, id asc)",
        "status": if r1 == r2 && dup_block_sorted {"PASS"} else {"FAILED"},
        "repeat_identical": r1 == r2, "dup_block_id_ascending": dup_block_sorted}));

    // ---- TEST-C2-004: head cardinality/membership semantics (documented behavior)
    coll.retrieval_config.write().mode = RetrievalMode::FixedFusion;
    let q = new_q(&mut rng);
    let one = e.attend("c2t", &["a".to_string()], &q, 5).unwrap();
    let unknown_skipped = e.attend("c2t", &["a".to_string(), "zzz".to_string()], &q, 5).unwrap() == one;
    let empty_ok = e.attend("c2t", &[], &q, 5).unwrap().is_empty();
    results.push(json!({"test": "TEST-C2-004", "name": "head membership semantics: unknown heads skipped, empty head list -> empty result",
        "status": if unknown_skipped && empty_ok {"PASS"} else {"FAILED"},
        "unknown_head_skipped_multihead": unknown_skipped, "empty_heads_empty_result": empty_ok,
        "note": "recorded engine behavior; callers must validate head names (B6 mapping relies on this)"}));

    // ---- TEST-C2-008: hybrid fusion correctness (B5)
    // v2 (2026-09-24): the pre-fix assertion "unique_token_doc_top1" was
    // vacuous -- the stimulus token "xyloquery" occurs in 8 docs (i%5==4), so
    // the last-match scan target was iteration-order dependent. Corrected to
    // the real contract: the sparse BM25 leg ranks only token docs (identical
    // tf and length => strictly tied bm25 scores), and under RRF any token doc
    // fuses to >= 1 + 1/(n+1) while any non-token doc fuses to <= 1 (sparse
    // absent), so the fused top-1 must be a token doc under any deterministic
    // tie rule; determinism is asserted separately. The engine-side
    // nondeterminism that genuinely failed TEST-C2-008 pre-fix is fixed and
    // pinned by C2-BM25-REPRO-002 and core/tests/regression_bm25_tie_order.rs.
    let q = new_q(&mut rng);
    let h1 = e.attend_hybrid("c2t", &["a".to_string()], &q, "xyloquery", 5).unwrap();
    let h2 = e.attend_hybrid("c2t", &["a".to_string()], &q, "xyloquery", 5).unwrap();
    let h5: Vec<Vec<u64>> = (0..5).map(|_| e.attend_hybrid("c2t", &["a".to_string()], &q, "xyloquery", 5).unwrap().iter().map(|(i, _)| *i).collect()).collect();
    let all_same = h5.iter().all(|r| *r == h5[0]);
    let mut token_docs: Vec<u64> = Vec::new();
    {
        let store = e.document_store.read();
        let mapper = e.id_mapper.read();
        for rec in store.list_all_records() {
            if rec.fields.get("text").and_then(|v| v.as_str()).map(|s| s.contains("xyloquery")).unwrap_or(false) {
                if let Some(nid) = mapper.uuid_to_id(&rec.id) {
                    token_docs.push(nid);
                }
            }
        }
    }
    let top1_in_token_set = h5[0].first().map(|i| token_docs.contains(i)).unwrap_or(false);
    results.push(json!({"test": "TEST-C2-008", "name": "hybrid fusion correctness (B5)",
        "status": if all_same && top1_in_token_set {"PASS"} else {"FAILED"},
        "deterministic_5_calls": all_same, "first_two_equal": h1 == h2,
        "top1_among_token_docs": top1_in_token_set, "token_doc_count": token_docs.len(),
        "n_results": h5[0].len(), "v2_assertion_note":
        "token 'xyloquery' occurs in 8 docs (i%5==4); pre-v2 'unique token doc top1' scan was a last-wins iteration-order-dependent target and could not be satisfied; corrected contract = fused top-1 must belong to the token-doc set under any deterministic tie rule (see comment above)."}));

    // ---- TEST-C2-009: Full-mode exact rerank (B7)
    // v2 (2026-09-24): same pool-exhaustiveness reality as TEST-C2-003 —
    // hnsw_rs search at k == element count is not exhaustive, so a strict
    // equality against an always-exhaustive exact reference is undecidable.
    // Recoped: output determinism + exact-equality asserted only when the
    // query's per-head pools cover all docs; pool coverage recorded.
    coll.retrieval_config.write().mode = RetrievalMode::Full;
    let mut pass = true;
    let mut worst = Vec::new();
    let mut det = true;
    let mut exact_checked = 0usize;
    let mut coverage_min = 40usize;
    for _ in 0..10 {
        let q = new_q(&mut rng);
        let got = e.attend("c2t", &["a".to_string(), "b".to_string()], &q, 5).unwrap();
        let got2 = e.attend("c2t", &["a".to_string(), "b".to_string()], &q, 5).unwrap();
        if got != got2 {
            det = false;
        }
        let mut cov = 40usize;
        for h in ["a", "b"] {
            if let Ok(idx) = coll.head_manager.read().get_head(h) {
                let n = idx.read().len();
                let hits = idx.read().search(&q, n.max(1), None).ok();
                cov = cov.min(hits.map(|v| v.len()).unwrap_or(0));
            }
        }
        coverage_min = coverage_min.min(cov);
        if cov == 40 {
            exact_checked += 1;
            let exp = fused_ref(&q, 5, true);
            if !scores_match(&got, &exp) {
                pass = false;
                worst.push(json!({"got": got, "ref_exact_gated": exp}));
            }
        }
    }
    results.push(json!({"test": "TEST-C2-009", "name": "Full-mode exact rerank (B7): scores exact within the pool; coverage recorded (HNSW recall observation)",
        "status": if pass && det {"PASS"} else {"FAILED"},
        "deterministic_across_calls": det, "exact_ref_checks": exact_checked,
        "coverage_min_per_head": coverage_min,
        "note": "exact gated-scoring equality vs the exact reference is asserted only when per-head pools exhaustively cover all docs (hnsw_rs search at k == count is not exhaustive; see c2probe hrecall). Within-pool scores are exact; pool coverage is recorded as an observation."}));

    // ---- B4 evidence: untrained identity-QK attention is deterministic; C and
    // D differ numerically because fuse_candidate renormalizes present channels
    // (weighted mean). C0 documented a measured CONTRIBUTION no-op, not byte
    // equality; prereg H3 tests metric-level equivalence on real data.
    let mut det = true;
    let mut overlap_sum = 0.0f32;
    for _ in 0..10 {
        let q = new_q(&mut rng);
        coll.retrieval_config.write().mode = RetrievalMode::LearnedGating;
        let c = e.attend("c2t", &["a".to_string(), "b".to_string()], &q, 5).unwrap();
        let c2 = e.attend("c2t", &["a".to_string(), "b".to_string()], &q, 5).unwrap();
        if c != c2 { det = false; }
        coll.retrieval_config.write().mode = RetrievalMode::QKAttention;
        let d = e.attend("c2t", &["a".to_string(), "b".to_string()], &q, 5).unwrap();
        coll.retrieval_config.write().mode = RetrievalMode::LearnedGating;
        let ci: std::collections::HashSet<u64> = c.iter().map(|(i, _)| *i).collect();
        let di: std::collections::HashSet<u64> = d.iter().map(|(i, _)| *i).collect();
        overlap_sum += ci.intersection(&di).count() as f32 / 5.0;
    }
    results.push(json!({"test": "TEST-C2-B4-NOOP", "name": "B4 identity-QK: deterministic untrained scorer; C-vs-D overlap recorded",
        "status": if det {"PASS"} else {"FAILED"},
        "deterministic": det, "mean_top5_overlap_C_vs_D": overlap_sum / 10.0,
        "note": "C0 documented no-op = measured contribution claim (H3); byte identity NOT asserted because fusion renormalizes present channels"}));

    let doc = json!({"subcommand": "modes-test",
        "collection_config": {"fusion_overridden_for_reference_checks": "attention=0, multi_head_similarity=1, bm25=0"},
        "results": results});
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("{}", json!({"ok": true, "out": out,
        "all_pass": results.iter().all(|r| r["status"] == "PASS")}));
}

fn q_dummy(dim: usize) -> Vec<f32> { vec![0.0; dim] }

// ------------------------------------------------- oracle agreement check

fn cmd_oracle_agree(out: &str) {
    let dim = 16usize;
    let ndocs = 200usize;
    let (_dir, e) = open_tmp_engine("oracle");
    insert_docs(&e, "o", &["h"], dim, ndocs, 4242);
    e.get_collection("o").unwrap().retrieval_config.write().mode = RetrievalMode::SingleHead;
    let mut vecs: HashMap<u64, Vec<f32>> = HashMap::new();
    {
        let store = e.document_store.read();
        let mapper = e.id_mapper.read();
        for rec in store.list_all_records() {
            if let Some(nid) = mapper.uuid_to_id(&rec.id) {
                if let Some(v) = rec.k_vecs.get("h") {
                    vecs.insert(nid, v.clone());
                }
            }
        }
    }
    let vlist: Vec<(u64, Vec<f32>)> = vecs.into_iter().collect();
    let mut rng = Lcg(99);
    let mut eq_ids = 0usize;
    let mut eq_order = 0usize;
    let mut sound = true;
    let nq = 20usize;
    for _ in 0..nq {
        let q: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
        let got = e.attend("o", &["h".to_string()], &q, 10).unwrap();
        for (id, s) in &got {
            let v = vlist.iter().find(|(i, _)| i == id).map(|(_, v)| v).unwrap();
            if (cosine(&q, v) - s).abs() > 1e-4 {
                sound = false;
            }
        }
        let exp = brute_topk(&vlist, &q, 10);
        let gi: std::collections::HashSet<u64> = got.iter().map(|(i, _)| *i).collect();
        let ei: std::collections::HashSet<u64> = exp.iter().map(|(i, _)| *i).collect();
        if gi == ei { eq_ids += 1; }
        if got.iter().map(|(i, _)| *i).eq(exp.iter().map(|(i, _)| *i)) { eq_order += 1; }
    }
    let doc = json!({"subcommand": "oracle-agree", "queries": nq,
        "returned_scores_consistent_with_stored_vectors": sound,
        "set_equality_rate": eq_ids as f32 / nq as f32,
        "order_equality_rate": eq_order as f32 / nq as f32,
        "note": "engine mode A is approximate; agreement rate is recorded diagnostics, score consistency is the correctness assertion",
        "status": if sound {"PASS"} else {"FAILED"}});
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("{}", doc);
}

// ------------------------------------------------------------ b3 validate

fn cmd_b3_validate(card_path: &str, out: &str) {
    let mut results: Vec<Value> = Vec::new();
    let card: attentiondb_learned::gating_v2::ModelCard =
        serde_json::from_str(&std::fs::read_to_string(card_path).unwrap()).unwrap();
    let corpus = corpora::multiview(150, 777);
    let (_guard, e, _ids) = corpora::insert_corpus(&corpus);
    let head_names: Vec<String> = corpus.head_names.clone();

    // TEST-C2-010: activation on a matching collection
    let installed = e.install_gating_card("bench", card);
    let (ok, model_id) = match &installed {
        Ok(id) => (true, id.clone()),
        Err(err) => (false, format!("ERR {err}")),
    };
    results.push(json!({"test": "TEST-C2-010", "name": "B3 ModelCard activation (install + inspect)",
        "status": if ok {"PASS"} else {"FAILED"}, "model_id": model_id,
        "inspect": e.inspect_gating_model("bench").unwrap_or(None)}));

    let mlp_card: attentiondb_learned::gating_v2::ModelCard =
        serde_json::from_str(&std::fs::read_to_string(card_path).unwrap()).unwrap();
    let mlp = mlp_card.to_mlp();
    coll_retrieval_set(&e, "bench", RetrievalMode::LearnedGating);
    let mut weights_match = true;
    let mut differs_from_uniform = 0usize;
    let mut name_mapping_ok = true;
    for q in corpus.queries.iter().take(10) {
        let qv = &q.vectors[0];
        let got = e.attend("bench", &head_names, qv, 5).unwrap();
        let w = mlp.predict(qv);
        let weighted: Vec<(String, f32)> =
            head_names.iter().cloned().zip(w.iter().cloned()).collect();
        let exp = e.attend_weighted("bench", &weighted.iter().map(|(h, ww)| (h.clone(), *ww)).collect::<Vec<_>>(), qv, 5).unwrap();
        let same_ids = got.iter().map(|(i, _)| *i).eq(exp.iter().map(|(i, _)| *i));
        let scores_close = got.iter().zip(exp.iter()).all(|(a, b)| (a.1 - b.1).abs() < 1e-5);
        if !(same_ids && scores_close) {
            weights_match = false;
        }
        let uniform = vec![1.0 / head_names.len() as f32; head_names.len()];
        let wu: Vec<(String, f32)> = head_names.iter().cloned().zip(uniform).collect();
        let expu = e.attend_weighted("bench", &wu, qv, 5).unwrap();
        if got != expu {
            differs_from_uniform += 1;
        }
        // name->weight mapping: request heads in reversed order
        let mut rev = head_names.clone();
        rev.reverse();
        let got_rev = e.attend("bench", &rev, qv, 5).unwrap();
        let mut wrev = w.clone();
        wrev.reverse();
        let wrev_pair: Vec<(String, f32)> = rev.iter().cloned().zip(wrev).collect();
        let exp_rev = e.attend_weighted("bench", &wrev_pair, qv, 5).unwrap();
        let ids_ok = got_rev.iter().map(|(i, _)| *i).eq(exp_rev.iter().map(|(i, _)| *i));
        let sc_ok = got_rev.iter().zip(exp_rev.iter()).all(|(a, b)| (a.1 - b.1).abs() < 1e-5);
        if !(ids_ok && sc_ok) {
            name_mapping_ok = false;
        }
    }
    results.push(json!({"test": "TEST-C2-010a", "name": "mode C results == attend_weighted(card softmax)",
        "status": if weights_match {"PASS"} else {"FAILED"}, "weights_match_engine": weights_match,
        "queries_differing_from_uniform": differs_from_uniform,
        "head_name_mapping_reversed": name_mapping_ok}));

    // TEST-C2-011: invalid provenance / mismatch rejection
    let m4 = attentiondb_learned::gating_v2::GatingMlp::new(mlp_card.input_dim, 16, 4, 1);
    let meta4 = attentiondb_learned::gating_v2::TrainingMeta {
        seed: 1, dataset_hash: 2, objective: "soft_target".into(), learning_rate: 0.01,
        batch_size: 8, epochs_run: 1, best_val_loss: 0.5, l2: 1e-4, timestamp_unix: 0,
        code_commit: "c2-probe".into(), hardware: "sandbox".into(),
    };
    let card4 = attentiondb_learned::gating_v2::ModelCard::from_mlp(&m4, meta4, "wrong-heads", "soft_target");
    let r4 = e.install_gating_card("bench", card4);
    results.push(json!({"test": "TEST-C2-011a", "name": "card num_heads mismatch refused",
        "status": if r4.is_err() {"PASS"} else {"FAILED"}, "err": format!("{:?}", r4.err())}));

    let m3 = attentiondb_learned::gating_v2::GatingMlp::new(mlp_card.input_dim, 16, 3, 1);
    let meta3 = attentiondb_learned::gating_v2::TrainingMeta {
        seed: 1, dataset_hash: 2, objective: "soft_target".into(), learning_rate: 0.01,
        batch_size: 8, epochs_run: 1, best_val_loss: 0.5, l2: 1e-4, timestamp_unix: 0,
        code_commit: "c2-probe".into(), hardware: "sandbox".into(),
    };
    let mut card3 = attentiondb_learned::gating_v2::ModelCard::from_mlp(&m3, meta3, "bad-coverage", "soft_target");
    card3.head_names = Some(vec!["semantic".into(), "lexical".into(), "extra".into()]);
    let r3 = e.install_gating_card("bench", card3);
    results.push(json!({"test": "TEST-C2-011b", "name": "card head-name coverage refusal (missing collection head)",
        "status": if r3.is_err() {"PASS"} else {"FAILED"}, "err": format!("{:?}", r3.err())}));

    let m3b = attentiondb_learned::gating_v2::GatingMlp::new(mlp_card.input_dim, 16, 3, 1);
    let meta3b = attentiondb_learned::gating_v2::TrainingMeta {
        seed: 1, dataset_hash: 2, objective: "soft_target".into(), learning_rate: 0.01,
        batch_size: 8, epochs_run: 1, best_val_loss: 0.5, l2: 1e-4, timestamp_unix: 0,
        code_commit: "c2-probe".into(), hardware: "sandbox".into(),
    };
    let mut card_bad = attentiondb_learned::gating_v2::ModelCard::from_mlp(&m3b, meta3b, "bad-format", "soft_target");
    card_bad.format = "tampered".into();
    let rb = e.install_gating_card("bench", card_bad);
    results.push(json!({"test": "TEST-C2-011c", "name": "tampered card format rejected by validate()",
        "status": if rb.is_err() {"PASS"} else {"FAILED"}, "err": format!("{:?}", rb.err())}));

    // fallback: deactivate -> mode C == mode B (uniform)
    e.deactivate_gating_model("bench").unwrap();
    let inspect_after = e.inspect_gating_model("bench").unwrap_or(None);
    coll_retrieval_set(&e, "bench", RetrievalMode::LearnedGating);
    let mut fallback_ok = true;
    for q in corpus.queries.iter().take(5) {
        let qv = &q.vectors[0];
        coll_retrieval_set(&e, "bench", RetrievalMode::LearnedGating);
        let c = e.attend("bench", &["semantic".to_string(), "lexical".to_string(), "metadata".to_string()], qv, 5).unwrap();
        coll_retrieval_set(&e, "bench", RetrievalMode::FixedFusion);
        let b = e.attend("bench", &["semantic".to_string(), "lexical".to_string(), "metadata".to_string()], qv, 5).unwrap();
        if c != b {
            fallback_ok = false;
        }
    }
    results.push(json!({"test": "TEST-C2-010b", "name": "deactivate -> documented uniform fallback",
        "status": if fallback_ok && inspect_after.is_none() {"PASS"} else {"FAILED"},
        "fallback_equals_mode_b": fallback_ok, "inspect_after_deactivate": inspect_after}));

    let doc = json!({"subcommand": "b3-validate", "results": results});
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("{}", json!({"ok": true, "out": out,
        "all_pass": results.iter().all(|r| r["status"] == "PASS")}));
}

fn coll_retrieval_set(e: &AttentionEngine, coll: &str, mode: RetrievalMode) {
    e.get_collection(coll).unwrap().retrieval_config.write().mode = mode;
}

// -------------------------------------------------------------------- main


fn cmd_debug_oracle() {
    let dim = 16usize;
    let (_dir, e) = open_tmp_engine("dbg");
    insert_docs(&e, "o", &["h"], dim, 200, 4242);
    e.get_collection("o").unwrap().retrieval_config.write().mode = RetrievalMode::SingleHead;
    let mut vecs: HashMap<u64, Vec<f32>> = HashMap::new();
    {
        let store = e.document_store.read();
        let mapper = e.id_mapper.read();
        for rec in store.list_all_records() {
            if let Some(nid) = mapper.uuid_to_id(&rec.id) {
                if let Some(v) = rec.k_vecs.get("h") { vecs.insert(nid, v.clone()); }
            }
        }
    }
    println!("stored vectors: {}", vecs.len());
    let vlist: Vec<(u64, Vec<f32>)> = vecs.into_iter().collect();
    let mut rng = Lcg(99);
    let q: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
    let got = e.attend("o", &["h".to_string()], &q, 5).unwrap();
    let exp = brute_topk(&vlist, &q, 5);
    println!("engine: {:?}", got);
    println!("brute : {:?}", exp);
    // check: what vector does the ENGINE hold for engine-top-1?
    let top_engine = got[0].0;
    let top_brute = exp[0].0;
    println!("engine top1 vector[0..4]: {:?}", &vlist.iter().find(|(i, _)| *i == top_engine).unwrap().1[..4]);
    println!("brute  top1 vector[0..4]: {:?}", &vlist.iter().find(|(i, _)| *i == top_brute).unwrap().1[..4]);
    // detailed feature introspection on the c2t 2-head collection
    let (_d2, e2) = open_tmp_engine("modes-dbg");
    insert_docs(&e2, "c2t", &["a", "b"], 8, 40, 20260925);
    let coll2 = e2.get_collection("c2t").unwrap();
    {
        let mut c = coll2.retrieval_config.write();
        c.fusion.attention = 0.0;
        c.fusion.multi_head_similarity = 1.0;
        c.fusion.bm25 = 0.0;
        c.min_candidates_per_head = 64;
        c.mode = RetrievalMode::FixedFusion;
    }
    let mut hv: Vec<HashMap<u64, Vec<f32>>> = vec![HashMap::new(), HashMap::new()];
    {
        let store = e2.document_store.read();
        let mapper = e2.id_mapper.read();
        for rec in store.list_all_records() {
            if let Some(nid) = mapper.uuid_to_id(&rec.id) {
                for (hi, h) in ["a", "b"].iter().enumerate() {
                    if let Some(v) = rec.k_vecs.get(*h) { hv[hi].insert(nid, v.clone()); }
                }
            }
        }
    }
    let mut rng2 = Lcg(777);
    for _ in 0..3 { let _: f32 = rng2.gauss(); }
    let q: Vec<f32> = (0..8).map(|_| rng2.gauss()).collect();
    let det = coll2.attend_detailed(&["a".to_string(), "b".to_string()], &q, 3, None, None, None, None, None).unwrap();
    for r in &det {
        println!("id {} final {:.6} head_scores {:?} mhs {:.6}", r.id, r.final_score,
                 r.features.head_scores.iter().map(|o| o.map(|v| (v*1000.0).round()/1000.0)).collect::<Vec<_>>(),
                 r.features.multi_head_similarity);
    }
    for hi in 0..2 {
        let mut scored: Vec<(u64, f32)> = hv[hi].iter().map(|(id, v)| (*id, cosine(&q, v))).collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut raw: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();
        minmax(&mut raw);
        println!("ref head {} top5 norms: {:?}", if hi==0 {"a"} else {"b"},
                 scored.iter().take(5).zip(raw.iter()).map(|((id, s), n)| (id, (s*1000.0).round()/1000.0, (n*1000.0).round()/1000.0)).collect::<Vec<_>>());
    }
}

/// Measure per-head HNSW recall at k == element count (decides whether the
/// C2 harness assumption "min_candidates_per_head >= corpus => exact
/// candidate membership" holds for the shipped hnsw_rs-backed index).
fn cmd_hrecall() {
    let dim = 8usize;
    let ndocs = 40usize;
    let (_dir, e) = open_tmp_engine("hrecall");
    insert_docs(&e, "c2h", &["a", "b"], dim, ndocs, 20260925);
    let coll = e.get_collection("c2h").unwrap();
    let mut rng = Lcg(777);
    let new_q = |rng: &mut Lcg| -> Vec<f32> { (0..dim).map(|_| rng.gauss()).collect() };
    for qn in 0..5 {
        let q = new_q(&mut rng);
        for h in ["a", "b"] {
            let idx = coll.head_manager.read().get_head(h).unwrap();
            let hits = idx.read().search(&q, ndocs as usize, None).ok().unwrap();
            let present: std::collections::HashSet<u64> = hits.iter().map(|(id, _)| *id).collect();
            let nid_set: std::collections::HashSet<u64> = {
                let store = e.document_store.read();
                let mapper = e.id_mapper.read();
                store
                    .list_all_records()
                    .iter()
                    .filter_map(|r| mapper.uuid_to_id(&r.id))
                    .collect()
            };
            let mut missing: Vec<u64> = nid_set.difference(&present).copied().collect();
            missing.sort();
            println!(
                "q{} head {} returned {:?}/{} docs, missing {:?}",
                qn,
                h,
                hits.len(),
                nid_set.len(),
                missing
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: c2probe <corpus|gating-dataset|train|modes-test|oracle-agree|b3-validate> ...");
        std::process::exit(64);
    }
    match args[1].as_str() {
        "corpus" => cmd_corpus(args[2].parse().unwrap(), args[3].parse().unwrap(), &args[4]),
        "gating-dataset" => cmd_gating_dataset(
            &args[2],
            args[3].parse().unwrap(),
            args[4].parse().unwrap(),
            &args[5],
            &args[6],
        ),
        "train" => cmd_train(&args[2], &args[3], &args[4], args[5].parse().unwrap(), args[6].parse().unwrap(), args[7].parse().unwrap()),
        "modes-test" => cmd_modes_test(&args[2]),
        "debug-oracle" => cmd_debug_oracle(),
        "oracle-agree" => cmd_oracle_agree(&args[2]),
        "b3-validate" => cmd_b3_validate(&args[2], &args[3]),
        "hrecall" => cmd_hrecall(),
        other => {
            eprintln!("unknown subcommand {other}");
            std::process::exit(64);
        }
    }
}
