use serde::{Deserialize, Serialize};
use crate::workload::{HeadCombination, ExperimentType};
use crate::workload::difficulty::{DifficultyLevel, MeasuredDifficultyProperties};
use crate::adapters::translation::TranslationStrategy;
use crate::config::types::RunProtocol;
use crate::metrics::quality::QualityMetrics;
use crate::metrics::throughput::ThroughputProfile;
use crate::metrics::pareto::ParetoResult;
use crate::metrics::energy::EnergyMeasurement;
use crate::executor::network_overhead::NetworkOverheadProfile;
use crate::stats::aggregator::AggregatedMetric;
use crate::stats::hypothesis::WelchTestResult;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationResult {
    pub benchmark_version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub git_hash: Option<String>,
    pub config_hash: String,
    pub hardware: HardwareProfile,

    pub database: String,
    pub database_version: String,
    pub dataset: String,
    pub scale: usize,
    pub difficulty: DifficultyLevel,
    pub experiment_type: ExperimentType,
    pub head_combination: HeadCombination,
    pub failure_mode: Option<String>,

    pub translation_strategy: TranslationStrategy,
    pub run_protocol: RunProtocol,
    pub measured_difficulty: MeasuredDifficultyProperties,

    pub recall_at_1: AggregatedMetric,
    pub recall_at_10: AggregatedMetric,
    pub recall_at_100: AggregatedMetric,
    pub mrr: AggregatedMetric,
    pub ndcg_at_10: AggregatedMetric,
    pub ndcg_at_100: AggregatedMetric,

    pub latency_mean_ms: AggregatedMetric,
    pub latency_p50_ms: AggregatedMetric,
    pub latency_p90_ms: AggregatedMetric,
    pub latency_p95_ms: AggregatedMetric,
    pub latency_p99_ms: AggregatedMetric,
    pub latency_p999_ms: AggregatedMetric,

    pub network_overhead_profile: Option<NetworkOverheadProfile>,

    pub throughput_profile: ThroughputProfile,
    pub pareto_result: ParetoResult,

    pub peak_memory_rss_mb: AggregatedMetric,
    pub disk_usage_gb: f64,
    pub index_build_time_s: f64,

    pub energy: EnergyMeasurement,
    pub statistical_comparisons: Vec<WelchTestResult>,

    pub consensus_accuracy: Option<f64>,
    pub gating_efficiency_vs_uniform: Option<f64>,
    pub per_head_weights: Option<HashMap<String, AggregatedMetric>>,

    pub raw_runs: Vec<SingleRunResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleRunResult {
    pub run_id: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub quality: QualityMetrics,
    pub latency_mean_ms: f64,
    pub duration_ms: f64,
    pub energy: EnergyMeasurement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub ram_total_gb: f64,
    pub gpu_model: Option<String>,
    pub gpu_vram_gb: Option<f64>,
    pub os: String,
    pub kernel: String,
    pub rust_version: String,
    pub benchmark_version: String,
    pub git_hash: Option<String>,
}

impl HardwareProfile {
    pub fn capture() -> Self {
        let cpu_model = Self::read_cpu_model();
        let (cores, threads) = Self::read_cpu_count();
        let ram_gb = Self::read_ram_gb();
        let (gpu_model, gpu_vram) = Self::read_gpu_info();

        Self {
            cpu_model,
            cpu_cores: cores,
            cpu_threads: threads,
            ram_total_gb: ram_gb,
            gpu_model,
            gpu_vram_gb: gpu_vram,
            os: std::env::consts::OS.to_string(),
            kernel: std::env::consts::FAMILY.to_string(),
            rust_version: env!("CARGO_PKG_VERSION").to_string(),
            benchmark_version: env!("CARGO_PKG_VERSION").to_string(),
            git_hash: std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8(o.stdout).ok()
                        .map(|s| s.trim().to_string())
                }),
        }
    }

    fn read_cpu_model() -> String {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("model name") {
                        return line.split(':').nth(1).unwrap_or("unknown").trim().to_string();
                    }
                }
            }
        }
        "unknown".to_string()
    }

    fn read_cpu_count() -> (usize, usize) {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        (cores / 2, cores)
    }

    fn read_ram_gb() -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        let kb: f64 = line.split_whitespace()
                            .nth(1).unwrap_or("0")
                            .parse().unwrap_or(0.0);
                        return kb / (1024.0 * 1024.0);
                    }
                }
            }
        }
        16.0
    }

    fn read_gpu_info() -> (Option<String>, Option<f64>) {
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=name,memory.total", "--format=csv,noheader"])
                .output()
            {
                let stdout = String::from_utf8(output.stdout).unwrap_or_default();
                let parts: Vec<&str> = stdout.trim().split(", ").collect();
                if parts.len() >= 2 {
                    let vram_str = parts[1].trim().trim_end_matches(" MiB");
                    let vram: f64 = vram_str.parse().unwrap_or(0.0);
                    return (Some(parts[0].to_string()), Some(vram / 1024.0));
                }
            }
        }
        (None, None)
    }
}

pub struct JsonOutput;

impl JsonOutput {
    pub fn write_results(
        results: &[PublicationResult],
        path: &str,
    ) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(results)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn write_raw_runs(
        results: &[PublicationResult],
        path: &str,
    ) -> anyhow::Result<()> {
        let runs: Vec<&SingleRunResult> = results.iter()
            .flat_map(|r| r.raw_runs.iter())
            .collect();
        let json = serde_json::to_string_pretty(&runs)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn write_aggregated(
        results: &[PublicationResult],
        path: &str,
    ) -> anyhow::Result<()> {
        let mut aggregated = Vec::new();
        for r in results {
            aggregated.push(serde_json::json!({
                "database": r.database,
                "dataset": r.dataset,
                "scale": r.scale,
                "difficulty": r.difficulty,
                "recall_at_10": {
                    "mean": r.recall_at_10.mean,
                    "ci_lower": r.recall_at_10.ci_lower_95,
                    "ci_upper": r.recall_at_10.ci_upper_95,
                },
                "ndcg_at_10": {
                    "mean": r.ndcg_at_10.mean,
                    "ci_lower": r.ndcg_at_10.ci_lower_95,
                    "ci_upper": r.ndcg_at_10.ci_upper_95,
                },
                "latency_mean_ms": {
                    "mean": r.latency_mean_ms.mean,
                    "ci_lower": r.latency_mean_ms.ci_lower_95,
                    "ci_upper": r.latency_mean_ms.ci_upper_95,
                },
                "index_build_time_s": r.index_build_time_s,
            }));
        }
        let json = serde_json::to_string_pretty(&aggregated)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
