use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoSweepConfig {
    pub param_values: Vec<usize>,
    pub queries_per_point: usize,
    pub reps_per_point: usize,
    pub target_recalls: Vec<f64>,
    pub concurrency: usize,
}

impl Default for ParetoSweepConfig {
    fn default() -> Self {
        Self {
            param_values: vec![10, 20, 40, 80, 100, 150, 200, 400, 800],
            queries_per_point: 1000,
            reps_per_point: 5,
            target_recalls: vec![0.90, 0.95, 0.99],
            concurrency: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub param_value: usize,
    pub recall_at_10_mean: f64,
    pub recall_at_10_std: f64,
    pub latency_mean_ms: f64,
    pub latency_p99_ms: f64,
    pub qps_single_thread: f64,
    pub qps_concurrent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoResult {
    pub database: String,
    pub dataset: String,
    pub scale: usize,
    pub points: Vec<ParetoPoint>,
    pub qps_at_target_recall: HashMap<String, f64>,
    pub dominates: HashMap<String, bool>,
}

impl ParetoResult {
    pub fn qps_at_recall(&self, target: f64) -> Option<f64> {
        let mut sorted = self.points.clone();
        sorted.sort_by(|a, b| {
            a.recall_at_10_mean.partial_cmp(&b.recall_at_10_mean).unwrap()
        });

        for window in sorted.windows(2) {
            let lo = &window[0];
            let hi = &window[1];
            if lo.recall_at_10_mean <= target && target <= hi.recall_at_10_mean {
                let t = (target - lo.recall_at_10_mean)
                    / (hi.recall_at_10_mean - lo.recall_at_10_mean);
                return Some(lo.qps_single_thread
                    + t * (hi.qps_single_thread - lo.qps_single_thread));
            }
        }
        None
    }

    pub fn interpolate_qps_at_recall(&self, target: f64) -> f64 {
        self.qps_at_recall(target).unwrap_or(0.0)
    }

    pub fn dominates_system(&self, other: &ParetoResult) -> bool {
        let targets = [0.90, 0.95, 0.99];
        targets.iter().all(|&t| {
            match (self.qps_at_recall(t), other.qps_at_recall(t)) {
                (Some(self_qps), Some(other_qps)) => self_qps >= other_qps,
                (Some(_), None) => true,
                _ => false,
            }
        })
    }
}

#[async_trait::async_trait]
pub trait SweepableAdapter: Send + Sync {
    fn sweep_param_name(&self) -> &'static str;
    async fn set_search_param(&self, value: usize) -> anyhow::Result<()>;
    async fn reset_search_param(&self) -> anyhow::Result<()>;
}
