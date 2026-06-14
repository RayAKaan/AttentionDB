use criterion::{criterion_group, criterion_main, Criterion, black_box};
use attentiondb_learned::ProjectionTrainer;

fn training_benchmark(c: &mut Criterion) {
    let mut trainer = ProjectionTrainer::new(256, 0.001);

    let dim = 256;
    let positive_pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..100)
        .map(|i| {
            let base: Vec<f32> = (0..dim).map(|x| ((i + x) as f32).sin() * 0.1).collect();
            let mut pos = base.clone();
            pos[0] += 0.01;
            (base, pos)
        })
        .collect();

    let negatives: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..dim).map(|x| ((i * 7 + x) as f32).cos() * 0.1).collect())
        .collect();

    c.bench_function("train_step_100_pairs", |b| {
        b.iter(|| {
            let _ = trainer.train_step(black_box(&positive_pairs), black_box(&negatives));
        });
    });

    c.bench_function("project_key_256d", |b| {
        let sample = vec![0.1; dim];
        let proj = trainer.get_projection();
        b.iter(|| {
            let _ = proj.project_key(black_box(&sample));
        });
    });
}

criterion_group!(benches, training_benchmark);
criterion_main!(benches);
