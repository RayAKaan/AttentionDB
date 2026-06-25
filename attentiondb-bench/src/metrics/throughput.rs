use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputConfig {
    pub concurrency_levels: Vec<usize>,
    pub warmup_duration_secs: u64,
    pub measurement_duration_secs: u64,
}

impl Default for ThroughputConfig {
    fn default() -> Self {
        Self {
            concurrency_levels: vec![1, 2, 4, 8, 16, 32, 64],
            warmup_duration_secs: 10,
            measurement_duration_secs: 60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputAtConcurrency {
    pub concurrency: usize,
    pub sustained_qps: f64,
    pub latency_mean_ms: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub error_rate: f64,
    pub measurement_duration_secs: u64,
    pub total_requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputProfile {
    pub database: String,
    pub points: Vec<ThroughputAtConcurrency>,
    pub saturation_concurrency: usize,
    pub peak_qps: f64,
}

impl ThroughputProfile {
    pub fn compute_saturation(&mut self) {
        if let Some(peak) = self.points.iter()
            .max_by(|a, b| a.sustained_qps.partial_cmp(&b.sustained_qps).unwrap())
        {
            self.saturation_concurrency = peak.concurrency;
            self.peak_qps = peak.sustained_qps;
        }
    }
}

pub async fn run_sustained_load(
    query_fn: impl Fn(usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<f64, String>> + Send>> + Send + Sync + 'static,
    query_count: usize,
    concurrency: usize,
    duration_secs: u64,
    record: bool,
) -> ThroughputAtConcurrency {
    let deadline = Instant::now() + Duration::from_secs(duration_secs);
    let completed = Arc::new(AtomicU64::new(0));
    let errors = Arc::new(AtomicU64::new(0));
    let latencies = Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));
    let query_fn = Arc::new(query_fn);

    let mut handles = Vec::new();

    for worker_id in 0..concurrency {
        let completed = completed.clone();
        let errors = errors.clone();
        let latencies = latencies.clone();
        let query_fn = query_fn.clone();

        handles.push(tokio::spawn(async move {
            let mut idx = worker_id;
            while Instant::now() < deadline {
                match query_fn(idx % query_count).await {
                    Ok(lat_ms) => {
                        completed.fetch_add(1, Ordering::Relaxed);
                        if record {
                            if let Ok(mut lats) = latencies.lock() {
                                lats.push(lat_ms);
                            }
                        }
                    }
                    Err(_) => {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
                idx += concurrency;
            }
        }));
    }

    for h in handles {
        let _ = h.await;
    }

    let total = completed.load(Ordering::Relaxed);
    let total_errors = errors.load(Ordering::Relaxed);
    let lats = latencies.lock().unwrap();

    let qps = if record && duration_secs > 0 {
        total as f64 / duration_secs as f64
    } else { 0.0 };

    let mut sorted = lats.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = if !lats.is_empty() { lats.iter().sum::<f64>() / lats.len() as f64 } else { 0.0 };

    ThroughputAtConcurrency {
        concurrency,
        sustained_qps: qps,
        latency_mean_ms: mean,
        latency_p50_ms: if sorted.is_empty() { 0.0 } else { crate::stats::confidence::percentile(&sorted, 0.50) },
        latency_p95_ms: if sorted.is_empty() { 0.0 } else { crate::stats::confidence::percentile(&sorted, 0.95) },
        latency_p99_ms: if sorted.is_empty() { 0.0 } else { crate::stats::confidence::percentile(&sorted, 0.99) },
        error_rate: if total > 0 { total_errors as f64 / total as f64 } else { 0.0 },
        measurement_duration_secs: duration_secs,
        total_requests: total,
    }
}
