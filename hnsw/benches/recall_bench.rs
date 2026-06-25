use attentiondb_hnsw::{HNSWConfig, HNSWIndex};
use clap::Parser;
use criterion::{BenchmarkId, Criterion};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "recall_bench")]
#[command(about = "Research-grade HNSW recall benchmark")]
struct Args {
    #[arg(long)]
    dataset: Option<String>,

    #[arg(long, default_value_t = 300)]
    dim: usize,

    #[arg(long)]
    size: Option<usize>,

    #[arg(long, default_value_t = 300)]
    queries: usize,

    #[arg(long, default_value = "All")]
    config: String,

    #[arg(long)]
    quick: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub dim: usize,
    pub dataset_size: usize,
    pub num_queries: usize,
    pub k: usize,
    pub ef_values: Vec<usize>,
    pub graph_configs: Vec<GraphConfig>,
    pub use_normalization: bool,
    pub dataset_path: Option<String>,
    pub output_json: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    pub name: String,
    pub max_nb_connection: usize,
    pub ef_construction: usize,
}

impl BenchmarkConfig {
    fn from_args(args: &Args) -> Self {
        let mut config = Self::default();

        if let Some(path) = &args.dataset {
            config.dataset_path = Some(path.clone());
        }
        config.dim = args.dim;
        config.num_queries = args.queries;

        if args.quick {
            config.dataset_size = 30_000;
            config.num_queries = 100;
        }

        if let Some(size) = args.size {
            config.dataset_size = size;
        }

        if args.config != "All" {
            config.graph_configs.retain(|g| g.name == args.config);
        }

        config
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dim: 300,
            dataset_size: 100_000,
            num_queries: 300,
            k: 10,
            ef_values: vec![64, 128, 256],
            graph_configs: vec![
                GraphConfig {
                    name: "Balanced".to_string(),
                    max_nb_connection: 16,
                    ef_construction: 400,
                },
                GraphConfig {
                    name: "HighQuality".to_string(),
                    max_nb_connection: 32,
                    ef_construction: 600,
                },
                GraphConfig {
                    name: "MaxQuality".to_string(),
                    max_nb_connection: 48,
                    ef_construction: 800,
                },
            ],
            use_normalization: true,
            dataset_path: Some("data/glove-100k.fvecs".to_string()),
            output_json: Some("recall_results.json".to_string()),
        }
    }
}

fn normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect()
}

fn load_fvecs(
    path: &str,
    max_vectors: usize,
) -> Result<Vec<(u64, Vec<f32>)>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    let mut id = 0u64;

    while data.len() < max_vectors {
        let mut dim_buf = [0u8; 4];
        if file.read_exact(&mut dim_buf).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;

        let mut vec = vec![0f32; dim];
        let mut bytes = vec![0u8; dim * 4];
        file.read_exact(&mut bytes)?;

        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            vec[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }

        data.push((id, vec));
        id += 1;
    }
    Ok(data)
}

fn generate_semantic_dataset(dim: usize, size: usize) -> Vec<(u64, Vec<f32>)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(size);
    let clusters = 25;
    let per_cluster = size / clusters;

    for c in 0..clusters {
        let mut center: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        center = normalize(&center);

        for i in 0..per_cluster {
            let mut vec = center.clone();
            for v in &mut vec {
                *v += rng.gen_range(-0.035..0.035);
            }
            vec = normalize(&vec);
            let id = (c * per_cluster + i) as u64;
            data.push((id, vec));
        }
    }
    data
}

fn brute_force_top_k(query: &[f32], data: &[(u64, Vec<f32>)], k: usize) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = data
        .iter()
        .map(|(id, vec)| {
            let score: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            (*id, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

fn recall_at_k(results: &[(u64, f32)], gt: &[(u64, f32)]) -> f32 {
    let r: HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
    let g: HashSet<u64> = gt.iter().map(|(id, _)| *id).collect();
    r.intersection(&g).count() as f32 / g.len() as f32
}

fn mrr(results: &[(u64, f32)], gt: &[(u64, f32)]) -> f32 {
    let g: HashSet<u64> = gt.iter().map(|(id, _)| *id).collect();
    for (rank, (id, _)) in results.iter().enumerate() {
        if g.contains(id) {
            return 1.0 / (rank + 1) as f32;
        }
    }
    0.0
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx]
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResult {
    graph_config: String,
    ef: usize,
    recall_at_10: f32,
    mrr: f32,
    mean_latency_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

fn recall_benchmark(c: &mut Criterion, args: &Args) {
    let config = BenchmarkConfig::from_args(&args);

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║     AttentionDB HNSW Recall Benchmark (Research Grade)     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let raw_dataset = if let Some(path) = &config.dataset_path {
        println!("Loading dataset from {}...", path);
        match load_fvecs(path, config.dataset_size) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Failed to load file: {}. Using synthetic data.", e);
                generate_semantic_dataset(config.dim, config.dataset_size)
            }
        }
    } else {
        println!("No dataset provided. Generating synthetic data...");
        generate_semantic_dataset(config.dim, config.dataset_size)
    };

    let dataset: Vec<(u64, Vec<f32>)> = if config.use_normalization {
        raw_dataset
            .into_iter()
            .map(|(id, vec)| (id, normalize(&vec)))
            .collect()
    } else {
        raw_dataset
    };

    println!(
        "Dataset ready: {} vectors (dim={}, normalized={})\n",
        dataset.len(),
        config.dim,
        config.use_normalization
    );

    let mut all_results: Vec<BenchmarkResult> = Vec::new();
    let mut group = c.benchmark_group("hnsw_recall");

    for graph_cfg in &config.graph_configs {
        println!(
            "▶ Building graph: {} (conn={}, ef_constr={})",
            graph_cfg.name, graph_cfg.max_nb_connection, graph_cfg.ef_construction
        );

        let hnsw_config = HNSWConfig {
            max_nb_connection: graph_cfg.max_nb_connection,
            ef_construction: graph_cfg.ef_construction,
            ef_search: 64,
            store_vectors: true,
            max_elements: config.dataset_size + 50_000,
        };

        let mut index = HNSWIndex::new("recall_bench", config.dim, hnsw_config);

        for (id, vec) in &dataset {
            let _ = index.insert(*id, vec);
        }
        println!("   Index built with {} vectors.\n", index.len());

        let queries: Vec<Vec<f32>> = (0..config.num_queries)
            .map(|_| {
                let v = random_vector(config.dim);
                if config.use_normalization {
                    normalize(&v)
                } else {
                    v
                }
            })
            .collect();

        let ground_truths: Vec<Vec<(u64, f32)>> = queries
            .iter()
            .map(|q| brute_force_top_k(q, &dataset, config.k))
            .collect();

        for &ef in &config.ef_values {
            let mut recalls = Vec::new();
            let mut mrrs = Vec::new();
            let mut latencies = Vec::new();

            for (query, gt) in queries.iter().zip(ground_truths.iter()) {
                let start = Instant::now();
                let res = index.search(query, config.k, Some(ef)).unwrap();
                let latency = start.elapsed().as_secs_f64() * 1000.0;

                recalls.push(recall_at_k(&res, gt));
                mrrs.push(mrr(&res, gt));
                latencies.push(latency);
            }

            let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
            let avg_mrr = mrrs.iter().sum::<f32>() / mrrs.len() as f32;
            let mean_lat = latencies.iter().sum::<f64>() / latencies.len() as f64;
            let p50 = percentile(&latencies, 50.0);
            let p95 = percentile(&latencies, 95.0);
            let p99 = percentile(&latencies, 99.0);

            println!(
                "   ef={:>3} | Recall@10 = {:.3} | MRR = {:.3} | mean={:.2}ms | p50={:.2} p95={:.2} p99={:.2}",
                ef, avg_recall, avg_mrr, mean_lat, p50, p95, p99
            );

            all_results.push(BenchmarkResult {
                graph_config: graph_cfg.name.clone(),
                ef,
                recall_at_10: avg_recall,
                mrr: avg_mrr,
                mean_latency_ms: mean_lat,
                p50_ms: p50,
                p95_ms: p95,
                p99_ms: p99,
            });

            group.bench_with_input(
                BenchmarkId::new(format!("{}_ef", graph_cfg.name), ef),
                &ef,
                |b, &ef| {
                    b.iter(|| {
                        for query in &queries {
                            let _ = index.search(query, config.k, Some(ef));
                        }
                    });
                },
            );
        }
    }

    group.finish();

    println!("\n╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║                           FINAL RESULTS SUMMARY                            ║");
    println!("╠════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:<15} {:>6} {:>10} {:>8} {:>10} {:>8} {:>8} {:>8} ║",
        "Config", "ef", "Recall@10", "MRR", "Mean(ms)", "p50", "p95", "p99"
    );
    println!("╠════════════════════════════════════════════════════════════════════════════╣");

    for r in &all_results {
        println!(
            "║ {:<15} {:>6} {:>10.3} {:>8.3} {:>10.2} {:>8.2} {:>8.2} {:>8.2} ║",
            r.graph_config,
            r.ef,
            r.recall_at_10,
            r.mrr,
            r.mean_latency_ms,
            r.p50_ms,
            r.p95_ms,
            r.p99_ms
        );
    }
    println!("╚════════════════════════════════════════════════════════════════════════════╝\n");

    if let Some(path) = &config.output_json {
        let json = serde_json::to_string_pretty(&all_results).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(json.as_bytes()).unwrap();
        println!("Results saved to {}\n", path);
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().collect();
    let filtered: Vec<String> = raw
        .iter()
        .filter(|a| !a.starts_with("--bench"))
        .cloned()
        .collect();
    let args = Args::parse_from(&filtered);

    let mut criterion = Criterion::default().configure_from_args();
    recall_benchmark(&mut criterion, &args);
}
