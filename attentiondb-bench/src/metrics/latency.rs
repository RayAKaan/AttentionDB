use crate::stats::confidence::percentile;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct LatencyTracker {
    latencies: Vec<f64>,
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self { latencies: Vec::new() }
    }

    pub fn record(&mut self, duration: Duration) {
        self.latencies.push(duration.as_secs_f64() * 1000.0);
    }

    pub fn record_ms(&mut self, ms: f64) {
        self.latencies.push(ms);
    }

    pub fn results(&self) -> LatencyResult {
        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted.len() as f64;
        let mean = if !sorted.is_empty() {
            sorted.iter().sum::<f64>() / len
        } else { 0.0 };

        LatencyResult {
            count: sorted.len(),
            mean_ms: mean,
            p50_ms: if sorted.is_empty() { 0.0 } else { percentile(&sorted, 0.50) },
            p90_ms: if sorted.is_empty() { 0.0 } else { percentile(&sorted, 0.90) },
            p95_ms: if sorted.is_empty() { 0.0 } else { percentile(&sorted, 0.95) },
            p99_ms: if sorted.is_empty() { 0.0 } else { percentile(&sorted, 0.99) },
            p999_ms: if sorted.is_empty() { 0.0 } else { percentile(&sorted, 0.999) },
            min_ms: sorted.first().copied().unwrap_or(0.0),
            max_ms: sorted.last().copied().unwrap_or(0.0),
        }
    }

    pub fn all_latencies(&self) -> &[f64] {
        &self.latencies
    }

    pub fn len(&self) -> usize {
        self.latencies.len()
    }

    pub fn clear(&mut self) {
        self.latencies.clear();
    }
}

impl Default for LatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyResult {
    pub count: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub p999_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

pub struct Stopwatch {
    start: Instant,
}

impl Stopwatch {
    pub fn start() -> Self {
        Self { start: Instant::now() }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}
