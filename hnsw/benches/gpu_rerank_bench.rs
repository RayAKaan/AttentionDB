use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use attentiondb_hnsw::gpu::{CpuBackend, GpuBackend};
use rand::Rng;

fn bench_rerank_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_rerank");
    group.measurement_time(std::time::Duration::from_secs(10));

    let dim = 256;
    let candidate_sizes = [100, 1_000, 10_000];
    let k = 10;
    let mut rng = rand::thread_rng();
    let backend = CpuBackend;

    for &num_candidates in &candidate_sizes {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let candidates: Vec<(u64, Vec<f32>)> = (0..num_candidates)
            .map(|i| (i as u64, (0..dim).map(|_| rng.gen::<f32>()).collect()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("cpu", num_candidates),
            &candidate_sizes,
            |b, _| {
                b.iter(|| {
                    backend.rerank_exact(&query, &candidates, k).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_rerank_cpu);
criterion_main!(benches);
