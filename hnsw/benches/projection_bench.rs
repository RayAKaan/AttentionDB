use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use attentiondb_hnsw::{HNSWIndex, HNSWConfig, CollectionSettings};

fn generate_vectors(dim: usize, count: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect())
        .collect()
}

fn projection_benchmark(c: &mut Criterion) {
    let dim = 256;
    let matrix: Vec<f32> = (0..dim * dim).map(|_| rand::random::<f32>() - 0.5).collect();

    let mut group = c.benchmark_group("projection");

    for batch_size in [10, 50, 100, 500].iter() {
        let vectors = generate_vectors(dim, *batch_size);

        let index = HNSWIndex::with_settings("bench", dim, HNSWConfig::default(), CollectionSettings::default()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("cpu", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let _ = index.project_batch(&matrix, &vectors);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, projection_benchmark);
criterion_main!(benches);
