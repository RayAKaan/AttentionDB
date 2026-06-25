use crate::adapters::DatabaseAdapter;
use crate::workload::{HeadEmbedding, HeadType, Query};
use crate::workload::difficulty::DifficultyLevel;

pub struct WarmupManager;

impl WarmupManager {
    pub async fn warmup(
        adapter: &dyn DatabaseAdapter,
        dimension: usize,
        num_runs: usize,
    ) -> anyhow::Result<WarmupReport> {
        let mut latencies = Vec::with_capacity(num_runs);

        for i in 0..num_runs {
            let query = Query {
                id: format!("warmup_{}", i),
                embeddings: vec![HeadEmbedding {
                    head_name: HeadType::Semantic,
                    vector: vec![0.1; dimension],
                }],
                enabled_heads: vec![HeadType::Semantic],
                ground_truth: vec![],
                difficulty: DifficultyLevel::Easy,
                failure_mode: None,
            };

            let result = adapter.query(&query, 10).await?;
            latencies.push(result.latency.as_secs_f64() * 1000.0);
        }

        let total: f64 = latencies.iter().sum();
        let mean = total / latencies.len() as f64;

        Ok(WarmupReport {
            num_runs,
            mean_latency_ms: mean,
            total_duration_ms: total,
        })
    }

    pub async fn warmup_and_stabilize(
        adapter: &dyn DatabaseAdapter,
        dimension: usize,
        max_warmup: usize,
        stability_threshold: f64,
    ) -> anyhow::Result<WarmupReport> {
        let mut prev_mean = 0.0;
        let mut latencies = Vec::new();

        for i in 0..max_warmup {
            let query = Query {
                id: format!("warmup_stab_{}", i),
                embeddings: vec![HeadEmbedding {
                    head_name: HeadType::Semantic,
                    vector: vec![0.1; dimension],
                }],
                enabled_heads: vec![HeadType::Semantic],
                ground_truth: vec![],
                difficulty: DifficultyLevel::Easy,
                failure_mode: None,
            };

            let result = adapter.query(&query, 10).await?;
            latencies.push(result.latency.as_secs_f64() * 1000.0);

            if i >= 10 && i % 5 == 0 {
                let recent: &[f64] = &latencies[latencies.len().saturating_sub(10)..];
                let current_mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;

                if prev_mean > 0.0 {
                    let change = (current_mean - prev_mean).abs() / prev_mean;
                    if change < stability_threshold {
                        return Ok(WarmupReport {
                            num_runs: i + 1,
                            mean_latency_ms: current_mean,
                            total_duration_ms: latencies.iter().sum(),
                        });
                    }
                }
                prev_mean = current_mean;
            }
        }

        let total: f64 = latencies.iter().sum();
        let mean = total / latencies.len() as f64;
        Ok(WarmupReport {
            num_runs: max_warmup,
            mean_latency_ms: mean,
            total_duration_ms: total,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WarmupReport {
    pub num_runs: usize,
    pub mean_latency_ms: f64,
    pub total_duration_ms: f64,
}
