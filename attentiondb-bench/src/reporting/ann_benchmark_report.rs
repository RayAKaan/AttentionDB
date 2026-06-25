use crate::reporting::json_output::PublicationResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnBenchmarkResult {
    pub dataset: String,
    pub algorithm: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub pareto_curve: Vec<(f64, f64)>,
    pub at_90_recall: Option<AnnBenchmarkPoint>,
    pub at_95_recall: Option<AnnBenchmarkPoint>,
    pub at_99_recall: Option<AnnBenchmarkPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnBenchmarkPoint {
    pub recall_at_10: f64,
    pub qps: f64,
    pub p99_latency_ms: f64,
    pub build_time_s: f64,
    pub index_size_mb: f64,
}

impl AnnBenchmarkResult {
    pub fn from_publication_result(
        result: &PublicationResult,
    ) -> Self {
        let pareto_curve: Vec<(f64, f64)> = result.pareto_result.points.iter()
            .map(|p| (p.recall_at_10_mean, p.qps_single_thread))
            .collect();

        let get_point = |recall: f64| -> Option<AnnBenchmarkPoint> {
            result.pareto_result.qps_at_recall(recall).map(|qps| {
                AnnBenchmarkPoint {
                    recall_at_10: recall,
                    qps,
                    p99_latency_ms: result.latency_p99_ms.mean,
                    build_time_s: result.index_build_time_s,
                    index_size_mb: result.disk_usage_gb * 1024.0,
                }
            })
        };

        Self {
            dataset: result.dataset.clone(),
            algorithm: result.database.clone(),
            parameters: HashMap::new(),
            pareto_curve,
            at_90_recall: get_point(0.90),
            at_95_recall: get_point(0.95),
            at_99_recall: get_point(0.99),
        }
    }

    pub fn to_ann_benchmarks_json(
        results: &[PublicationResult],
    ) -> Vec<AnnBenchmarkResult> {
        results.iter().map(Self::from_publication_result).collect()
    }
}

pub struct AnnBenchmarkReport;

impl AnnBenchmarkReport {
    pub fn write_output(
        results: &[PublicationResult],
        path: &str,
    ) -> anyhow::Result<()> {
        let ann_results = AnnBenchmarkResult::to_ann_benchmarks_json(results);
        let json = serde_json::to_string_pretty(&ann_results)?;
        std::fs::write(path, json)?;
        tracing::info!("ANN benchmark output written to {}", path);
        Ok(())
    }
}
