use attentiondb_query::{parse_aql, plan_query, QueryExecutor};
use criterion::{criterion_group, criterion_main, Criterion};

const AQL: &str = r#"ATTEND TO papers WHERE QUERY "attention mechanisms in transformers" HEADS [semantic, temporal] TOP_K 10 MIN_WEIGHT 0.05 TEMPORAL_DECAY 0.3"#;

fn query_benchmark(c: &mut Criterion) {
    let parsed = parse_aql(AQL).unwrap();
    let plan = plan_query(parsed).unwrap();
    let query_vector = vec![0.1; 256];

    c.bench_function("aql_full_pipeline", |b| {
        b.iter(|| {
            let parsed = parse_aql(AQL).unwrap();
            let plan = plan_query(parsed).unwrap();
            let _ = QueryExecutor::execute(&plan, &query_vector);
        });
    });

    c.bench_function("aql_execute_only", |b| {
        b.iter(|| {
            let _ = QueryExecutor::execute(&plan, &query_vector);
        });
    });
}

fn parse_benchmark(c: &mut Criterion) {
    c.bench_function("aql_parse", |b| {
        b.iter(|| {
            let _ = parse_aql(AQL);
        });
    });
}

criterion_group!(benches, query_benchmark, parse_benchmark);
criterion_main!(benches);
