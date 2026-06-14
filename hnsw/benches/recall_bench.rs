//! Research-Grade HNSW Recall Benchmark for AttentionDB
//!
//! ## Purpose
//! This benchmark is designed to evaluate HNSW retrieval quality using **real embedding data**.
//!
//! ## How to Use Real Embedding Data (Recommended)
//!
//! 1. Download a dataset in `.fvecs` format.
//!    Recommended datasets:
//!    - **GloVe** (word embeddings): https://nlp.stanford.edu/data/glove.6B.zip
//!    - **SIFT** (image descriptors): http://corpus-texmex.irisa.fr/
//!    - **Deep1B** (deep learning embeddings)
//!
//! 2. Convert the dataset to `.fvecs` format if needed (many are already in this format).
//!
//! 3. Place the file in a `data/` folder and update the config below:
//!
//! ```rust
//! dataset_path: Some("data/glove-100k.fvecs".to_string()),
//! ```
//!
//! 4. Run the benchmark:
//!    ```bash
//!    cargo bench --bench recall_bench
//!    ```
//!
//! ## Key Configuration
//! You can tune the following in `BenchmarkConfig`:
//! - `ef_construction` and `max_nb_connection` -> Graph quality
//! - `ef_values` -> Search quality vs speed trade-off
//! - `use_normalization` -> Should almost always be `true` for real embeddings
//!
//! Results are automatically saved to `recall_results.json`.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use attentiondb_hnsw::{HNSWIndex, HNSWConfig};
use rand::Rng;
use std::time::Instant;
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use serde::{Serialize, Deserialize};

/// ==================== BENCHMARK CONFIGURATION ====================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub dim: usize,
    pub dataset_size: usize,
    pub num_queries: usize,
    pub k: usize,
    pub ef_values: Vec<usize>,
    pub max_nb_connection: usize,
    pub ef_construction: usize,
    pub use_normalization: bool,
    /// Path to a .fvecs file. Set this to use real embedding data.
    pub dataset_path: Option<String>,
    pub output_json: Option<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dim: 256,
            dataset_size: 100_000,
            num_queries: 300,
            k: 10,
            ef_values: vec![32, 64, 128, 256],
            max_nb_connection: 16,
            ef_construction: 400,
            use_normalization: true,
            dataset_path: None,
            output_json: Some("recall_results.json".to_string()),
        }
    }
}

/// ==================== HELPER FUNCTIONS ====================

fn normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 { vec.iter().map(|x| x / norm).collect() } else { vec.to_vec() }
}

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect()
}

/// Generate semantic-like synthetic data (fallback)
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
            for v in &mut vec { *v += rng.gen_range(-0.035..0.035); }
            vec = normalize(&vec);
            let id = (c * per_cluster + i) as u64;
            data.push((id, vec));
        }
    }
    data
}

/// Load .fvecs file
fn load_fvecs(path: &str, max_vectors: usize) -> Result<Vec<(u64, Vec<f32>)>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    let mut id = 0u64;

    while data.len() < max_vectors {
        let mut dim_buf = [0u8; 4];
        if file.read_exact(&mut dim_buf).is_err() { break; }
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

/// Brute-force exact search
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
    if values.is_empty() { return 0.0; }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx]
}

/// ==================== MAIN BENCHMARK ====================

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResult {
    ef: usize,
    recall_at_10: f32,
    mrr: f32,
    mean_latency_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

fn recall_benchmark(c: &mut Criterion) {
    let config = BenchmarkConfig::default();

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║     AttentionDB HNSW Recall Benchmark (Research Grade)     ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    println!("{:#?}\n", config);

    // Load dataset
    let dataset = if let Some(path) = &config.dataset_path {
        println!("Loading real embedding dataset from {}...", path);
        match load_fvecs(path, config.dataset_size) {
            Ok(data) => {
                println!("Successfully loaded {} vectors.\n", data.len());
                data
            }
            Err(e) => {
                println!("Failed to load file: {}. Using synthetic data.\n", e);
                generate_semantic_dataset(config.dim, config.dataset_size)
            }
        }
    } else {
        println!("Generating high-quality synthetic semantic dataset...\n");
        generate_semantic_dataset(config.dim, config.dataset_size)
    };

    // Build index once
    println!("Building HNSW index (conn={}, ef_constr={})...",
             config.max_nb_connection, config.ef_construction);

    let hnsw_config = HNSWConfig {
        max_nb_connection: config.max_nb_connection,
        ef_construction: config.ef_construction,
        ef_search: 64,
        store_vectors: true,
        max_elements: config.dataset_size + 50_000,
    };

    let mut index = HNSWIndex::new("recall_bench", config.dim, hnsw_config);

    for (id, vec) in &dataset {
        let v = if config.use_normalization { normalize(vec) } else { vec.clone() };
        index.insert(*id, &v).unwrap();
    }
    println!("Index ready with {} vectors.\n", index.len());

    // Generate queries
    let queries: Vec<Vec<f32>> = (0..config.num_queries)
        .map(|_| {
            let v = random_vector(config.dim);
            if config.use_normalization { normalize(&v) } else { v }
        })
        .collect();

    // Precompute ground truth
    println!("Precomputing ground truth...");
    let ground_truths: Vec<Vec<(u64, f32)>> = queries
        .iter()
        .map(|q| brute_force_top_k(q, &dataset, config.k))
        .collect();

    let mut group = c.benchmark_group("hnsw_recall");
    let mut results: Vec<BenchmarkResult> = Vec::new();

    println!("\nEvaluating different ef values...\n");

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
            "ef={:>3} | Recall@10 = {:.3} | MRR = {:.3} | Latency: mean={:.2}ms | p50={:.2} p95={:.2} p99={:.2}",
            ef, avg_recall, avg_mrr, mean_lat, p50, p95, p99
        );

        results.push(BenchmarkResult {
            ef,
            recall_at_10: avg_recall,
            mrr: avg_mrr,
            mean_latency_ms: mean_lat,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
        });

        group.bench_with_input(BenchmarkId::new("ef", ef), &ef, |b, &ef| {
            b.iter(|| {
                for query in &queries {
                    let _ = index.search(query, config.k, Some(ef));
                }
            });
        });
    }

    group.finish();

    // Export results
    if let Some(path) = &config.output_json {
        let json = serde_json::to_string_pretty(&results).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(json.as_bytes()).unwrap();
        println!("\nResults saved to {}", path);
    }

    println!("\nBenchmark completed.\n");
}

criterion_group!(benches, recall_benchmark);
criterion_main!(benches);
