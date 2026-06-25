use attentiondb_hnsw::{HNSWConfig, HNSWIndex};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;

fn generate_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect()
}

fn hnsw_benchmark(c: &mut Criterion) {
    let dim = 256;
    let config = HNSWConfig::new()
        .with_max_elements(200_000)
        .with_vector_storage(false);

    let mut index = HNSWIndex::new("bench", dim, config);

    println!("Building index with 100,000 vectors...");
    let mut rng = rand::thread_rng();
    for i in 0..100_000 {
        let vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        index.insert(i as u64, &vec).unwrap();
    }

    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();

    let mut group = c.benchmark_group("hnsw_search");

    for ef in [16, 32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::new("ef", ef), ef, |b, &ef| {
            b.iter(|| {
                let results = index.search(black_box(&query), 10, Some(ef));
                black_box(results)
            });
        });
    }

    group.finish();

    println!(
        "\nDone. {} vectors indexed, {} queries across 5 ef values",
        index.len(),
        5
    );
}

fn hnsw_insert_benchmark(c: &mut Criterion) {
    let dim = 256;
    let config = HNSWConfig::new().with_vector_storage(true);

    c.bench_function("hnsw_insert_10k", |b| {
        b.iter_with_setup(
            || {
                let mut index = HNSWIndex::new("bench_insert", dim, config.clone());
                let mut rng = rand::thread_rng();
                let vectors: Vec<(u64, Vec<f32>)> = (0..10_000)
                    .map(|i| (i, (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect()))
                    .collect();
                (index, vectors)
            },
            |(mut index, vectors)| {
                for (id, vec) in vectors {
                    index.insert(id, &vec).unwrap();
                }
            },
        )
    });
}

criterion_group!(benches, hnsw_benchmark, hnsw_insert_benchmark);
criterion_main!(benches);
