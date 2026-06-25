use super::types::BenchmarkConfig;
use anyhow::{ensure, Result};

pub fn validate_config(config: &BenchmarkConfig) -> Result<()> {
    validate_general(&config.general)?;
    validate_scales(&config.scales)?;
    validate_databases(&config.databases)?;
    validate_experiments(&config.experiments)?;
    validate_throughput(&config.throughput)?;
    validate_pareto(&config.pareto_sweep)?;
    Ok(())
}

fn validate_general(g: &super::types::GeneralConfig) -> Result<()> {
    ensure!(g.vector_dimension >= 64, "vector_dimension must be >= 64");
    ensure!(g.vector_dimension <= 4096, "vector_dimension must be <= 4096");
    ensure!(g.warmup_runs > 0, "warmup_runs must be > 0");
    ensure!(g.bootstrap_resamples >= 100, "bootstrap_resamples must be >= 100");
    ensure!(g.confidence_level > 0.0 && g.confidence_level < 1.0,
        "confidence_level must be between 0 and 1");
    ensure!(!g.output_dir.is_empty(), "output_dir must not be empty");
    Ok(())
}

fn validate_scales(s: &super::types::ScaleConfig) -> Result<()> {
    ensure!(!s.values.is_empty(), "at least one scale value required");
    for &v in &s.values {
        ensure!(v >= 100, "scale values must be >= 100, got {}", v);
    }
    Ok(())
}

fn validate_databases(d: &super::types::DatabaseConfig) -> Result<()> {
    let any_enabled = d.attentiondb_cpu.enabled
        || d.qdrant.enabled
        || d.pgvector.enabled
        || d.milvus.enabled
        || d.weaviate.enabled
        || d.elasticsearch.enabled;
    ensure!(any_enabled, "at least one database must be enabled");
    if d.pinecone.enabled {
        ensure!(!d.pinecone.host.is_empty() || std::env::var("PINECONE_API_KEY").is_ok(),
            "Pinecone requires PINECONE_API_KEY env var or host config");
    }
    Ok(())
}

fn validate_experiments(e: &super::types::ExperimentConfig) -> Result<()> {
    ensure!(e.run_standard || e.run_head_ablations || e.run_failure_modes,
        "at least one experiment type must be enabled");
    Ok(())
}

fn validate_throughput(t: &super::types::ThroughputConfig) -> Result<()> {
    ensure!(!t.concurrency_levels.is_empty(), "at least one concurrency level required");
    ensure!(t.measurement_duration_secs >= 10, "measurement duration must be >= 10s");
    Ok(())
}

fn validate_pareto(p: &super::types::ParetoSweepConfig) -> Result<()> {
    ensure!(!p.ef_search_values.is_empty(), "at least one ef_search value required");
    ensure!(p.queries_per_point >= 100, "queries_per_point must be >= 100");
    ensure!(p.reps_per_point >= 1, "reps_per_point must be >= 1");
    Ok(())
}
