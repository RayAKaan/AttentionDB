//! Phase 2B benchmark driver (§23–29).
//!
//! Subcommands:
//!   generate --corpus controlled|noise|multiview --out DIR
//!   run      --corpus controlled|noise|multiview --out DIR [--seeds 42,7,1]
//!
//! `run` = generate (cached dataset.json) → train (all three objectives,
//! validation-selected) → evaluate (train/val/TEST + §29 central table +
//! §7 diagnostics + §35 per-group + §21 diversity). All numbers are measured
//! in-run from the cached dataset; training never touches HNSW (§4).

use attentiondb_core::collection::RetrievalMode;
use attentiondb_learned::eval::{
    evaluate_rrf, evaluate_single_head, evaluate_weighting, global_best_head, head_diversity,
    rank_metrics, uniform_weights,
};
use attentiondb_learned::gating_v2::{
    diagnostics, train_gating, GatingDataset, HeadExample, ModelCard, Objective, QualityTarget,
    QueryExample, Split, TrainingConfig, TrainingMeta,
};
use phase2b_bench::corpora::{self, Corpus};
mod qk_sanity;
use std::collections::HashMap;

const POOL: usize = 100; // per-head candidate pool (§4)
const GT_K: usize = 10;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("run");
    let mut corpus = "controlled";
    let mut out = "benchmarks/phase2b".to_string();
    let mut dataset_path = String::new();
    let mut seeds = vec![42u64];
    let mut n_queries: Option<usize> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                corpus = args[i + 1].as_str();
                i += 2;
            }
            "--out" => {
                out = args[i + 1].clone();
                i += 2;
            }
            "--dataset" => {
                dataset_path = args[i + 1].clone();
                i += 2;
            }
            "--seeds" => {
                seeds = args[i + 1]
                    .split(',')
                    .filter_map(|s| s.parse().ok())
                    .collect();
                i += 2;
            }
            "--queries" => {
                n_queries = args[i + 1].parse().ok();
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    std::fs::create_dir_all(&out).unwrap();
    match cmd {
        "generate" => {
            let c = build_corpus(corpus);
            let ds = generate_dataset(&c, 42);
            let p = if dataset_path.is_empty() {
                format!("{out}/dataset.json")
            } else {
                dataset_path
            };
            std::fs::write(&p, serde_json::to_string_pretty(&ds).unwrap()).unwrap();
            println!(
                "wrote {p} ({} queries, hash {:#x})",
                ds.queries.len(),
                ds.content_hash()
            );
        }
        "train" => {
            let json = std::fs::read_to_string(&dataset_path).unwrap();
            let ds = GatingDataset::parse(&json).unwrap();
            train_and_evaluate(&ds, &out, corpus);
        }
        "run" => {
            let c = build_corpus_q(corpus, n_queries);
            let ds = generate_dataset(&c, 42);
            let ds_path = format!("{out}/dataset.json");
            std::fs::write(&ds_path, serde_json::to_string_pretty(&ds).unwrap()).unwrap();
            println!(
                "dataset: {} queries (hash {:#x})",
                ds.queries.len(),
                ds.content_hash()
            );
            train_and_evaluate(&ds, &out, corpus);
            // multi-seed study (§12): same dataset/split, training seeds vary.
            if seeds.len() > 1 {
                run_multiseed(&ds, &out, corpus, &seeds);
            }
        }
        "rerank-study" => {
            // §32: exact-rerank weighting study on CACHED datasets.
            let json = std::fs::read_to_string(&dataset_path).unwrap();
            let ds = GatingDataset::parse(&json).unwrap();
            run_rerank_study(&ds, &out, corpus);
        }
        "latency" => {
            run_latency_study(&out);
        }
        "qk-sanity" => {
            // PH2C-QK-001: candidate-level QK sanity dataset (Phase 2C §6).
            let msg = qk_sanity::run(&out, &seeds);
            println!("{msg}");
        }
        other => {
            eprintln!("unknown command {other}");
            std::process::exit(2);
        }
    }
    let _ = seeds; // multi-seed runs are driven externally; single-seed default
}

fn build_corpus(name: &str) -> Corpus {
    build_corpus_q(name, None)
}

fn build_corpus_q(name: &str, n_queries: Option<usize>) -> Corpus {
    match name {
        "controlled" => corpora::controlled(4, 12, n_queries.unwrap_or(300), 0xC0B),
        "noise" | "noise_ladder" => corpora::noise_ladder(n_queries.unwrap_or(300), 0xC0FFEE),
        "multiview" => corpora::multiview(n_queries.unwrap_or(1200), 0x1D3A),
        other => {
            eprintln!("unknown corpus {other}");
            std::process::exit(2);
        }
    }
}

/// §4: candidate generation ONCE; every later training iteration reads the
/// cached dataset.
fn generate_dataset(corpus: &Corpus, split_seed: u64) -> GatingDataset {
    let (_guard, e, _ids) = corpora::insert_corpus(corpus);
    let coll = e.get_collection("bench").unwrap();
    // engine numeric id → my doc index, resolved through the document store
    // (fields.idx == corpus doc numeric_hint). Never assume IdMapper's base.
    let mut id_to_idx: HashMap<u64, usize> = HashMap::new();
    {
        let store = e.document_store.read();
        let mapper = e.id_mapper.read();
        for rec in store.list_all_records() {
            if let Some(idx_field) = rec.fields.get("idx").and_then(|v| v.as_u64()) {
                if let Some(numeric) = mapper.uuid_to_id(&rec.id) {
                    id_to_idx.insert(numeric, idx_field as usize);
                }
            }
        }
    }

    // seeded 70/15/15 split over query order (§3)
    let mut order: Vec<usize> = (0..corpus.queries.len()).collect();
    let mut rng = corpora::Rng::new(split_seed ^ 0x5EED);
    rng.shuffle(&mut order);
    let n = order.len();
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

    // invert: doc index (== corpus numeric_hint) → ENGINE numeric id.
    // GT must be expressed in ENGINE ids or every recall is silently wrong.
    let mut idx_to_engine = vec![0u64; corpus.docs.len()];
    for (&engine_id, &di) in id_to_idx.iter() {
        idx_to_engine[di] = engine_id;
    }

    let mut queries = Vec::with_capacity(corpus.queries.len());
    for (qi, bq) in corpus.queries.iter().enumerate() {
        let mut heads = Vec::with_capacity(corpus.head_names.len());
        for (h, _head_name) in corpus.head_names.iter().enumerate() {
            // independent per-head search (§4: per-head candidates)
            let ranked: Vec<(u64, f32)> = coll
                .attend_detailed(
                    std::slice::from_ref(&corpus.head_names[h]),
                    &bq.vectors[h],
                    POOL,
                    None,
                    Some(RetrievalMode::SingleHead),
                    None,
                    None,
                    None,
                )
                .unwrap()
                .into_iter()
                .map(|r| (r.id, r.final_score))
                .collect();
            // exact scores from the generator's own copy of the doc vectors
            let raw: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();
            let mut norm = raw.clone();
            attentiondb_learned::eval::minmax(&mut norm);
            let exact: Vec<f32> = ranked
                .iter()
                .map(|(id, _)| {
                    let di = id_to_idx[id];
                    corpora::cosine(&bq.vectors[h], &corpus.docs[di].head_vecs[h])
                })
                .collect();
            let metrics = rank_metrics(&ranked, &bq.ground_truth, GT_K);
            heads.push(HeadExample {
                candidates: ranked.iter().map(|(id, _)| *id).collect(),
                raw_scores: raw,
                norm_scores: norm,
                exact_scores: exact,
                recall_at_k: metrics.recall_at_10 as f32,
                ndcg_at_k: metrics.ndcg_at_10 as f32,
                mrr: metrics.mrr as f32,
            });
        }
        let gt_engine: Vec<u64> = bq
            .ground_truth
            .iter()
            .map(|&hint| idx_to_engine[hint as usize])
            .collect();
        // recompute per-head quality against ENGINE-id ground truth
        for head in heads.iter_mut() {
            let ranked: Vec<(u64, f32)> = head
                .candidates
                .iter()
                .copied()
                .zip(head.raw_scores.iter().copied())
                .collect();
            let m = rank_metrics(&ranked, &gt_engine, GT_K);
            head.recall_at_k = m.recall_at_10 as f32;
            head.ndcg_at_k = m.ndcg_at_10 as f32;
            head.mrr = m.mrr as f32;
        }
        queries.push(QueryExample {
            query_id: bq.id,
            query: bq.gating_input.clone(),
            query_group: Some(bq.group),
            split: split_of[qi],
            ground_truth: gt_engine,
            heads,
        });
    }

    GatingDataset {
        format: "attentiondb-gating-dataset".into(),
        version: 1,
        num_heads: corpus.head_names.len(),
        input_dim: corpus.queries[0].gating_input.len(),
        top_k: GT_K,
        corpus_desc: format!(
            "{} docs={} heads={}",
            corpus.name,
            corpus.docs.len(),
            corpus.head_names.len()
        ),
        seed: split_seed,
        queries,
    }
}

/// §19 calibration: inference temperature fitted on the VALIDATION split
/// only. T<1 sharpens the predicted weights. Never touches test.
fn calibrated_predict(
    m: &attentiondb_learned::gating_v2::GatingMlp,
    q: &QueryExample,
    t: f32,
) -> Vec<f32> {
    let logits = m.logits(&q.query);
    let scaled: Vec<f32> = logits.iter().map(|l| l / t).collect();
    attentiondb_learned::gating_v2::GatingMlp::softmax(&scaled)
}

fn fit_temperature(
    m: &attentiondb_learned::gating_v2::GatingMlp,
    ds: &GatingDataset,
) -> (f32, f64) {
    let candidates = [4.0f32, 2.0, 1.0, 0.75, 0.5, 0.35, 0.25];
    let mut best = (1.0f32, f64::NEG_INFINITY);
    for t in candidates {
        let r10 = evaluate_weighting(ds, Split::Val, &|q| calibrated_predict(m, q, t))
            .metrics
            .recall_at_10;
        if r10 > best.1 {
            best = (t, r10);
        }
    }
    best
}

fn hardware_desc() -> String {
    format!(
        "cpus={}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    )
}

fn commit_desc() -> String {
    option_env!("GIT_HASH").unwrap_or("unknown").to_string()
}

/// Train all objectives, select on VALIDATION, evaluate on train/val/TEST,
/// write every §23–29 artifact.
fn train_and_evaluate(ds: &GatingDataset, out: &str, corpus_name: &str) {
    let (tr, va, te) = ds.split_counts();
    println!("splits: train={tr} val={va} test={te} (test untouched until final report)");
    let objectives = [
        Objective::QualityRegression,
        Objective::SoftTarget,
        Objective::Pairwise,
    ];
    let training_dir = format!("{out}/training");
    std::fs::create_dir_all(&training_dir).unwrap();

    // §6/§31: hyperparameter grid selected ENTIRELY on validation R@10 of the
    // temperature-calibrated model. The test split is not consulted.
    let grid: Vec<TrainingConfig> = objectives
        .iter()
        .flat_map(|obj| {
            [
                TrainingConfig {
                    objective: *obj,
                    ..TrainingConfig::default()
                },
                TrainingConfig {
                    objective: *obj,
                    hidden: 64,
                    lr: 0.003,
                    batch_size: 16,
                    patience: 25,
                    ..TrainingConfig::default()
                },
            ]
        })
        .collect();
    let mut best: Option<(
        Objective,
        attentiondb_learned::gating_v2::TrainOutcome,
        TrainingConfig,
        f32,
        f64,
    )> = None;
    for cfg in &grid {
        let outcome = train_gating(ds, cfg, QualityTarget::Recall);
        let (temp, val_r10) = fit_temperature(&outcome.model, ds);
        println!(
            "  {:>18} hidden={:>2} lr={:.3} batch={:>2}: epochs={} val_loss={:.4} val_R10@T={:.4}",
            cfg.objective.name(),
            cfg.hidden,
            cfg.lr,
            cfg.batch_size,
            outcome.curves.len(),
            outcome.best_val_loss,
            val_r10
        );
        // §25: machine-readable training curves
        let mut csv = String::from("epoch,train_loss,val_loss,val_weight_quality_corr\n");
        for p in &outcome.curves {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.4}\n",
                p.epoch, p.train_loss, p.val_loss, p.val_weight_quality_corr
            ));
        }
        std::fs::write(
            format!(
                "{training_dir}/curves_{}_h{}_lr{}.csv",
                cfg.objective.name(),
                cfg.hidden,
                cfg.lr
            ),
            csv,
        )
        .unwrap();
        if best.as_ref().map(|b| val_r10 > b.4).unwrap_or(true) {
            best = Some((cfg.objective, outcome, cfg.clone(), temp, val_r10));
        }
    }
    let (obj, outcome, cfg, temp, _val_r10) = best.unwrap();
    let _ = temp;
    // re-derive curves CSV for the selected config is already written above

    // §15/§24: full model card with reproducibility metadata
    let meta = TrainingMeta {
        seed: cfg.seed,
        dataset_hash: ds.content_hash(),
        objective: obj.name().into(),
        learning_rate: cfg.lr,
        batch_size: cfg.batch_size,
        epochs_run: outcome.curves.len(),
        best_val_loss: outcome.best_val_loss,
        l2: cfg.l2,
        timestamp_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        code_commit: commit_desc(),
        hardware: hardware_desc(),
    };
    let meta_cloned = meta.clone();
    let card = ModelCard::from_mlp(
        &outcome.model,
        meta,
        &format!("{corpus_name}-gating-v1"),
        obj.name(),
    );
    card.validate().unwrap();
    let model_path = format!("{training_dir}/model-v1.json");
    card.save(std::path::Path::new(&model_path)).unwrap();

    // registry (§17): activate through the real registry API
    {
        use attentiondb_learned::registry::ModelRegistry;
        let reg = ModelRegistry::new(std::path::Path::new(&format!("{out}/models")));
        let v = reg.save_new(&card).unwrap();
        reg.activate(v).unwrap();
    }

    // ---------------- §19 calibration (VAL-only temperature fit) --------
    let (temp, val_r10_at_t) = fit_temperature(&outcome.model, ds);
    println!(
        "calibration: temperature={temp} (val R@10 at T: {val_r10_at_t:.4}; raw: {:.4})",
        evaluate_weighting(ds, Split::Val, &|q| outcome.model.predict(&q.query))
            .metrics
            .recall_at_10
    );

    // ---------------- §29 central table (TEST split) ----------------
    let n_heads = ds.num_heads;
    let (gb_head, gb_train_recall) = global_best_head(ds, Split::Train);
    let mut table = String::from("approach,R@1,R@5,R@10,R@50,NDCG@10,MRR\n");
    let rows: Vec<(&str, attentiondb_learned::eval::EvalReport)> = vec![
        (
            "uniform_multihead",
            evaluate_weighting(ds, Split::Test, &|_| uniform_weights(n_heads)),
        ),
        (
            "global_best_single_head",
            evaluate_single_head(ds, Split::Test, gb_head),
        ),
        (
            "trained_gating",
            evaluate_weighting(ds, Split::Test, &|q| {
                calibrated_predict(&outcome.model, q, temp)
            }),
        ),
        ("rrf_k60", evaluate_rrf(ds, Split::Test, 60.0)),
        (
            "oracle_per_query_head",
            evaluate_weighting(ds, Split::Test, &|q| {
                attentiondb_learned::eval::oracle_weights(q)
            }),
        ),
    ];
    for (name, r) in &rows {
        table.push_str(&format!(
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
            name,
            r.metrics.recall_at_1,
            r.metrics.recall_at_5,
            r.metrics.recall_at_10,
            r.metrics.recall_at_50,
            r.metrics.ndcg_at_10,
            r.metrics.mrr
        ));
    }
    std::fs::write(format!("{out}/eval_test.csv"), table.clone()).unwrap();
    println!("== TEST split (§29 central table) ==");
    print!("{table}");

    // ---------------- §26 overfitting chain ----------------
    let chain = format!(
        "split,trained_R10,uniform_R10\ntrain,{:.4},{:.4}\nval,{:.4},{:.4}\ntest,{:.4},{:.4}\n",
        evaluate_weighting(ds, Split::Train, &|q| calibrated_predict(
            &outcome.model,
            q,
            temp
        ))
        .metrics
        .recall_at_10,
        evaluate_weighting(ds, Split::Train, &|_| uniform_weights(n_heads))
            .metrics
            .recall_at_10,
        evaluate_weighting(ds, Split::Val, &|q| calibrated_predict(
            &outcome.model,
            q,
            temp
        ))
        .metrics
        .recall_at_10,
        evaluate_weighting(ds, Split::Val, &|_| uniform_weights(n_heads))
            .metrics
            .recall_at_10,
        evaluate_weighting(ds, Split::Test, &|q| calibrated_predict(
            &outcome.model,
            q,
            temp
        ))
        .metrics
        .recall_at_10,
        evaluate_weighting(ds, Split::Test, &|_| uniform_weights(n_heads))
            .metrics
            .recall_at_10,
    );
    std::fs::write(format!("{out}/overfit_chain.csv"), chain).unwrap();

    // ---------------- §7 diagnostics ----------------
    let mut ws = Vec::new();
    let mut qs = Vec::new();
    for q in ds.queries.iter().filter(|q| q.split == Split::Test) {
        ws.push(calibrated_predict(&outcome.model, q, temp));
        qs.push(q.heads.iter().map(|h| h.recall_at_k).collect::<Vec<_>>());
    }
    let diag = diagnostics(&ws, &qs);
    std::fs::write(
        format!("{out}/diagnostics.json"),
        serde_json::to_string_pretty(&diag).unwrap(),
    )
    .unwrap();
    println!(
        "diagnostics: entropy={:.3} corr={:.3} avg_weights={:?} sel_freq={:?}",
        diag.mean_normalized_entropy,
        diag.weight_quality_pearson,
        diag.avg_weights
            .iter()
            .map(|w| format!("{w:.3}"))
            .collect::<Vec<_>>(),
        diag.selection_frequency
            .iter()
            .map(|w| format!("{w:.3}"))
            .collect::<Vec<_>>(),
    );

    // ---------------- §35/§37 per-group breakdown ----------------
    let mut by_group =
        String::from("group,uniform_R10,trained_R10,oracle_R10,best_head_for_group\n");
    let groups: Vec<u32> = {
        let mut g: Vec<u32> = ds.queries.iter().filter_map(|q| q.query_group).collect();
        g.sort_unstable();
        g.dedup();
        g
    };
    for g in groups {
        let (mut u, mut t, mut o, mut cnt) = (Vec::new(), Vec::new(), Vec::new(), [0usize; 16]);
        for q in ds
            .queries
            .iter()
            .filter(|q| q.split == Split::Test && q.query_group == Some(g))
        {
            let ranked_u = attentiondb_learned::eval::fuse_weighted(q, &uniform_weights(n_heads));
            let ranked_t = attentiondb_learned::eval::fuse_weighted(
                q,
                &calibrated_predict(&outcome.model, q, temp),
            );
            let ranked_o = attentiondb_learned::eval::fuse_weighted(
                q,
                &attentiondb_learned::eval::oracle_weights(q),
            );
            u.push(rank_metrics(&ranked_u, &q.ground_truth, GT_K).recall_at_10);
            t.push(rank_metrics(&ranked_t, &q.ground_truth, GT_K).recall_at_10);
            o.push(rank_metrics(&ranked_o, &q.ground_truth, GT_K).recall_at_10);
            // empirical best head for this group on TRAIN
            if let Some(qe) = ds
                .queries
                .iter()
                .find(|x| x.split == Split::Train && x.query_group == Some(g))
            {
                let mut bh = 0;
                for (h, head) in qe.heads.iter().enumerate() {
                    if head.recall_at_k > qe.heads[bh].recall_at_k {
                        bh = h;
                    }
                }
                if bh < 16 {
                    cnt[bh] += 1;
                }
            }
        }
        let best_head_for_group = (0..16).max_by_key(|&i| cnt[i]).unwrap_or(0);
        let mean = |v: &[f64]| {
            if v.is_empty() {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        let head_label = format!("head{best_head_for_group}");
        by_group.push_str(&format!(
            "{g},{:.4},{:.4},{:.4},{head_label}\n",
            mean(&u),
            mean(&t),
            mean(&o)
        ));
    }
    std::fs::write(format!("{out}/by_group.csv"), by_group).unwrap();

    // ---------------- §21/§22 diversity AND utility ----------------
    let pairs = head_diversity(ds, Split::Test, 10);
    let mut div = String::from("head_a,head_b,jaccard_at_10\n");
    for p in &pairs {
        div.push_str(&format!(
            "{},{},{:.4}\n",
            p.head_a, p.head_b, p.jaccard_at_k
        ));
    }
    // utility: per-head mean recall on test
    let mut utility = String::from("head,mean_recall_at_10_test\n");
    for h in 0..n_heads {
        let rs: Vec<f64> = ds
            .queries
            .iter()
            .filter(|q| q.split == Split::Test)
            .map(|q| q.heads[h].recall_at_k as f64)
            .collect();
        utility.push_str(&format!(
            "{h},{:.4}\n",
            if rs.is_empty() {
                f64::NAN
            } else {
                rs.iter().sum::<f64>() / rs.len() as f64
            }
        ));
    }
    std::fs::write(format!("{out}/diversity.csv"), div).unwrap();
    std::fs::write(format!("{out}/utility.csv"), utility).unwrap();

    // ---------------- run_info (§24) ----------------
    let info = format!(
        "corpus={corpus_name}\ndataset_hash={:#x}\nseed={}\ncalibration_temperature={temp}\nsplits=train={tr},val={va},test={te}\nobjective_selected={}\nbest_val_loss={:.6}\nglobal_best_head={gb_head} (train_recall={gb_train_recall:.4})\nmodel_params={}\nmodel_bytes={}\ntimestamp_unix={}\ncommit={}\nhardware={}\n",
        ds.content_hash(),
        cfg.seed,
        obj.name(),
        outcome.best_val_loss,
        outcome.model.w1.len() + outcome.model.b1.len() + outcome.model.w2.len() + outcome.model.b2.len(),
        model_size_bytes(&card),
        meta_cloned.timestamp_unix,
        meta_cloned.code_commit,
        meta_cloned.hardware,
    );
    std::fs::write(format!("{out}/run_info.txt"), info).unwrap();
    println!("wrote artifacts to {out}/");
}

fn model_size_bytes(card: &ModelCard) -> usize {
    serde_json::to_vec(card).map(|v| v.len()).unwrap_or(0)
}

/// §12: mean ± std over training seeds on the SAME dataset and split.
fn run_multiseed(ds: &GatingDataset, out: &str, corpus_name: &str, seeds: &[u64]) {
    use attentiondb_learned::eval::evaluate_weighting;
    let mut rows = String::from("seed,train_loss,val_loss,R@1,R@5,R@10,R@50,NDCG@10,MRR\n");
    let mut per_seed = Vec::new();
    for seed in seeds {
        let cfg = TrainingConfig {
            seed: *seed,
            ..TrainingConfig::default()
        };
        let outcome = train_gating(ds, &cfg, QualityTarget::Recall);
        let (temp, _) = fit_temperature(&outcome.model, ds);
        let r = evaluate_weighting(ds, Split::Test, &|q| {
            calibrated_predict(&outcome.model, q, temp)
        });
        let m = r.metrics;
        rows.push_str(&format!(
            "{seed},{:.6},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
            outcome.final_train_loss,
            outcome.best_val_loss,
            m.recall_at_1,
            m.recall_at_5,
            m.recall_at_10,
            m.recall_at_50,
            m.ndcg_at_10,
            m.mrr
        ));
        per_seed.push(m);
        println!(
            "seed {seed}: R@10={:.4} NDCG={:.4}",
            m.recall_at_10, m.ndcg_at_10
        );
    }
    let stats = |get: fn(&attentiondb_learned::eval::RankMetrics) -> f64| -> (f64, f64) {
        let xs: Vec<f64> = per_seed.iter().map(get).collect();
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        (mean, var.sqrt())
    };
    let mut summary = String::from("metric,mean,std\n");
    type Getter = fn(&attentiondb_learned::eval::RankMetrics) -> f64;
    let getters: [(&str, Getter); 5] = [
        ("R@1", |m| m.recall_at_1),
        ("R@5", |m| m.recall_at_5),
        ("R@10", |m| m.recall_at_10),
        ("NDCG@10", |m| m.ndcg_at_10),
        ("MRR", |m| m.mrr),
    ];
    for (name, get) in getters {
        let (m, s) = stats(get);
        summary.push_str(&format!("{name},{m:.4},{s:.4}\n"));
    }
    std::fs::write(format!("{out}/multiseed.csv"), rows).unwrap();
    std::fs::write(
        format!("{out}/multiseed_summary_{corpus_name}.csv"),
        summary,
    )
    .unwrap();
    println!("wrote {out}/multiseed.csv (+summary)");
}

/// §32: exact-vs-normalized fusion weighting study on cached TEST candidates.
fn run_rerank_study(ds: &GatingDataset, out: &str, corpus_name: &str) {
    use attentiondb_learned::eval::{
        fuse_weighted, fuse_weighted_scores, global_best_head, oracle_weights, rank_metrics,
        uniform_weights,
    };
    let (gb, _) = global_best_head(ds, Split::Train);
    let n_heads = ds.num_heads;
    struct Row {
        name: &'static str,
        r10: f64,
        ndcg: f64,
        mrr: f64,
    }
    let mut acc: Vec<Row> = Vec::new();
    let mut eval_case =
        |name: &'static str, weights: &dyn Fn(&QueryExample) -> Vec<f32>, exact: bool| {
            let (mut r10s, mut nds, mut mrrs) = (Vec::new(), Vec::new(), Vec::new());
            for q in ds.queries.iter().filter(|q| q.split == Split::Test) {
                let w = weights(q);
                let ranked = if exact {
                    fuse_weighted_scores(q, &w, |h, i| h.exact_scores[i])
                } else {
                    fuse_weighted(q, &w)
                };
                let m = rank_metrics(&ranked, &q.ground_truth, ds.top_k);
                r10s.push(m.recall_at_10);
                nds.push(m.ndcg_at_10);
                mrrs.push(m.mrr);
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
            acc.push(Row {
                name,
                r10: mean(&r10s),
                ndcg: mean(&nds),
                mrr: mean(&mrrs),
            });
        };
    eval_case("norm_fusion_uniform", &|_| uniform_weights(n_heads), false);
    eval_case("exact_fusion_uniform", &|_| uniform_weights(n_heads), true);
    eval_case(
        "exact_fusion_best_head",
        &move |_q| {
            let mut w = vec![0.0f32; n_heads];
            w[gb] = 1.0;
            w
        },
        true,
    );
    eval_case("exact_fusion_oracle_head", &oracle_weights, true);
    let mut csv = String::from("method,R@10,NDCG@10,MRR\n");
    for r in &acc {
        csv.push_str(&format!(
            "{},{:.4},{:.4},{:.4}\n",
            r.name, r.r10, r.ndcg, r.mrr
        ));
    }
    std::fs::create_dir_all(out).unwrap();
    std::fs::write(format!("{out}/rerank.csv"), &csv).unwrap();
    println!("== rerank study ({corpus_name}, TEST) ==");
    print!("{csv}");
}

/// §38/§39: trained-model inference cost micro-bench.
fn run_latency_study(out: &str) {
    let card_path = "benchmarks/phase2b/controlled/training/model-v1.json";
    let Ok(card) = ModelCard::load(std::path::Path::new(card_path)) else {
        eprintln!("latency: train the controlled corpus first ({card_path} missing)");
        std::process::exit(1);
    };
    let mlp = card.to_mlp();
    let dim = card.input_dim;
    let mut rng = corpora::Rng::new(7);
    let probe: Vec<Vec<f32>> = (0..200)
        .map(|_| (0..dim).map(|_| rng.gauss()).collect())
        .collect();
    for q in &probe {
        let _ = mlp.predict(q);
    }
    let t = std::time::Instant::now();
    let reps = 2000;
    for _ in 0..reps {
        for q in &probe {
            let _ = mlp.predict(q);
        }
    }
    let per_call_us = t.elapsed().as_secs_f64() * 1e6 / (reps * probe.len()) as f64;
    let params = card.w1.len() + card.b1.len() + card.w2.len() + card.b2.len();
    let bytes = std::fs::metadata(card_path).map(|m| m.len()).unwrap_or(0);
    let csv = format!(
        "component,heads,input_dim,value,unit\nmodel_inference,{},{},{:.2},us_per_query\nmodel_params,{},{},count\nmodel_serialized_bytes,{},{},bytes\n",
        card.num_heads,
        card.input_dim,
        per_call_us,
        card.num_heads,
        params,
        card.num_heads,
        bytes
    );
    std::fs::create_dir_all(out).unwrap();
    std::fs::write(format!("{out}/latency.csv"), csv).unwrap();
    println!("model inference: {per_call_us:.2}µs/query, {params} params, {bytes} bytes");
}
