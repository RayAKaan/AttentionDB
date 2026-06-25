#![allow(dead_code)]

use clap::Parser;
use tracing_subscriber::EnvFilter;
use std::path::Path;

mod config;
mod workload;
mod adapters;
mod metrics;
mod stats;
mod executor;
mod reporting;

#[derive(Parser, Debug)]
#[command(name = "attentiondb-bench", version, about = "Publication-grade benchmark for AttentionDB")]
struct Cli {
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,

    #[arg(short, long)]
    output_dir: Option<String>,

    #[arg(long)]
    resume: bool,

    #[arg(long)]
    quick: bool,

    #[arg(long)]
    list_databases: bool,

    #[arg(long)]
    run_ann_benchmarks: bool,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long, help = "Kill existing server, wipe WAL, and start fresh")]
    clean_start: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("attentiondb_bench=info".parse().unwrap()))
        .init();

    let cli = Cli::parse();

    if cli.list_databases {
        println!("Available database adapters:");
        println!("  - AttentionDB (CPU)     [enabled by default]");
        println!("  - AttentionDB (GPU)     [feature: gpu]");
        println!("  - Qdrant                [feature: qdrant, enabled by default]");
        println!("  - Milvus                [feature: milvus]");
        println!("  - Weaviate              [feature: weaviate]");
        println!("  - pgvector              [feature: pgvector, enabled by default]");
        println!("  - Elasticsearch         [feature: elasticsearch, enabled by default]");
        println!("  - Pinecone              [feature: pinecone]");
        return Ok(());
    }

    let config_path = if cli.quick {
        Path::new("config/quick.toml")
    } else {
        Path::new(&cli.config)
    };

    tracing::info!("Loading config from {}", config_path.display());
    let mut base_config = config::parser::load_config(config_path)?;

    if let Some(seed) = cli.seed {
        base_config.general.random_seed = seed;
    }

    if let Some(ref output_dir) = cli.output_dir {
        base_config.general.output_dir = output_dir.clone();
    }
    if cli.quick {
        base_config.general.warmup_runs = 2;
        base_config.general.bootstrap_resamples = 1000;
    }

    config::validation::validate_config(&base_config)?;
    tracing::info!("Config validated");

    if cli.clean_start {
        let port = base_config.databases.attentiondb_cpu.port;
        executor::server_manager::ServerManager::clean_start(port).await?;
    }

    std::fs::create_dir_all(&base_config.general.output_dir)?;

    // Save the config used for reproducibility
    let config_output = toml::to_string_pretty(&base_config)?;
    let config_path_out = format!("{}/config.toml", base_config.general.output_dir);
    std::fs::write(&config_path_out, &config_output)?;
    tracing::info!("Config saved to {}", config_path_out);

    // Run hardware profiling
    let hardware = reporting::json_output::HardwareProfile::capture();
    tracing::info!("Hardware: {} ({} cores, {} GB RAM)",
        hardware.cpu_model, hardware.cpu_threads, hardware.ram_total_gb);

    // Run power analysis
    let mut orchestrator = executor::orchestrator::Orchestrator::new(
        base_config.clone(),
        &base_config.general.output_dir,
    );
    let power_results = orchestrator.run_power_analysis()?;
    for pr in &power_results {
        tracing::info!("  Power: d={:.1} N_req={} N30_power={:.2}{}",
            pr.target_effect_size_d, pr.required_n, pr.power_at_n30,
            if pr.n30_sufficient { " ✓" } else { " ✗" });
    }

    // Run the benchmark
    tracing::info!("Starting benchmark...");
    let results = orchestrator.run_full_benchmark().await?;

    // Write all output formats
    let output_dir = &base_config.general.output_dir;
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");

    let raw_path = format!("{}/{}_raw.json", output_dir, timestamp);
    reporting::json_output::JsonOutput::write_results(&results, &raw_path)?;

    let agg_path = format!("{}/{}_aggregated.json", output_dir, timestamp);
    reporting::json_output::JsonOutput::write_aggregated(&results, &agg_path)?;

    let csv_path = format!("{}/{}_results.csv", output_dir, timestamp);
    reporting::csv_output::CsvOutput::write_results(&results, &csv_path)?;

    // Render terminal table
    println!("\nBenchmark Results Summary:");
    println!("{}", reporting::table_renderer::TableRenderer::render_comparison_table(&results));

    // LaTeX tables
    if !results.is_empty() {
        let scales: Vec<usize> = {
            let mut s: Vec<usize> = results.iter().map(|r| r.scale).collect();
            s.sort();
            s.dedup();
            s
        };
        for &scale in &scales {
            let latex = reporting::latex_output::LatexOutput::generate_main_comparison_table(
                &results, scale,
            );
            let latex_path = format!("{}/{}_scale_{}_table.tex", output_dir, timestamp, scale);
            std::fs::write(&latex_path, &latex)?;
        }
    }

    // Pareto plots
    if !results.is_empty() {
        for result in &results {
            if !result.pareto_result.points.is_empty() {
                let plot_path = format!("{}/{}_{}_pareto_{}.svg",
                    output_dir, timestamp, result.database.replace(' ', "_"), result.scale);
                reporting::pareto_plot::ParetoPlot::render_pareto_curve(
                    &[result.pareto_result.clone()],
                    &plot_path,
                    &format!("{} - scale={}", result.database, result.scale),
                )?;
            }
        }
    }

    // ANN benchmarks output
    if base_config.experiments.run_ann_benchmark_compat {
        let ann_path = format!("{}/{}_ann_benchmark.json", output_dir, timestamp);
        reporting::ann_benchmark_report::AnnBenchmarkReport::write_output(
            &results, &ann_path,
        )?;
    }

    // Write hardware profile
    let hw_path = format!("{}/{}_hardware.json", output_dir, timestamp);
    let hw_json = serde_json::to_string_pretty(&hardware)?;
    std::fs::write(&hw_path, &hw_json)?;

    tracing::info!("All results written to {}", output_dir);
    println!("\nResults saved to: {}", output_dir);
    println!("  Raw data: {}", raw_path);
    println!("  CSV: {}", csv_path);

    Ok(())
}
