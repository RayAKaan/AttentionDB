use attentiondb_multihead::{HeadConfig, HeadType, MultiHeadManager};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn fusion_benchmark(c: &mut Criterion) {
    let mut manager = MultiHeadManager::new(256, 3);
    manager.add_head(HeadConfig::new("semantic", HeadType::Semantic, 256));
    manager.add_head(HeadConfig::new("temporal", HeadType::Temporal, 256));
    manager.add_head(HeadConfig::new("structural", HeadType::Structural, 256));

    let query_emb: Vec<f32> = (0..256).map(|i| (i as f32).sin() * 0.1).collect();
    let head_results = vec![
        (
            "semantic".to_string(),
            vec![(1, 0.92), (2, 0.85), (3, 0.71)],
        ),
        (
            "temporal".to_string(),
            vec![(2, 0.88), (4, 0.79), (5, 0.67)],
        ),
        (
            "structural".to_string(),
            vec![(1, 0.81), (3, 0.77), (6, 0.63)],
        ),
    ];

    c.bench_function("gating_forward", |b| {
        b.iter(|| manager.gating.forward(black_box(&query_emb)));
    });

    c.bench_function("fuse_scores", |b| {
        let gates = vec![0.5, 0.3, 0.2];
        b.iter(|| attentiondb_multihead::fuse_scores(black_box(&head_results), black_box(&gates)));
    });

    c.bench_function("full_fuse", |b| {
        b.iter(|| {
            let _ = manager.fuse(black_box(&query_emb), black_box(&head_results));
        });
    });
}

criterion_group!(benches, fusion_benchmark);
criterion_main!(benches);
