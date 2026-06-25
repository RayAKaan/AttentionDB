use attentiondb_hnsw::{HNSWConfig, HNSWIndex};
use attentiondb_query::{parse_aql, plan_query, AQLStatement, QueryExecutor};
use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;

const AQL: &str = r#"ATTEND TO papers WHERE QUERY "attention mechanisms in transformers" HEADS [semantic, temporal] TOP_K 10 MIN_WEIGHT 0.05 TEMPORAL_DECAY 0.3"#;

fn query_benchmark(c: &mut Criterion) {
    let parsed = parse_aql(AQL).unwrap();
    let AQLStatement::Query(query) = parsed else {
        panic!("expected Query statement");
    };
    let plan = plan_query(query).unwrap();

    let dim = 256;
    let config = HNSWConfig::new().with_max_elements(200_000);
    let mut index = HNSWIndex::new("bench_query", dim, config);

    let mut rng = rand::thread_rng();
    for i in 0..10_000 {
        let vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        index.insert(i as u64, &vec).unwrap();
    }
    let query_vector = vec![0.1; 256];

    c.bench_function("aql_full_pipeline", |b| {
        b.iter(|| {
            let parsed = parse_aql(AQL).unwrap();
            let AQLStatement::Query(query) = parsed else {
                panic!("expected Query statement");
            };
            let plan = plan_query(query).unwrap();
            let _ = QueryExecutor::execute(&plan, &index, &query_vector);
        });
    });

    c.bench_function("aql_execute_only", |b| {
        b.iter(|| {
            let _ = QueryExecutor::execute(&plan, &index, &query_vector);
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
