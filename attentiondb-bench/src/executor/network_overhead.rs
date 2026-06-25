use crate::adapters::DatabaseAdapter;
use crate::workload::{HeadEmbedding, HeadType, Query};
use crate::stats::confidence::{percentile, std_dev};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TransportType {
    InProcess,
    LocalGrpc,
    LocalRest,
    LocalTcp,
    Cloud,
}

impl TransportType {
    pub fn expected_overhead_ms(&self) -> (f64, f64) {
        match self {
            Self::InProcess  => (0.000, 0.010),
            Self::LocalGrpc  => (0.050, 0.300),
            Self::LocalRest  => (0.200, 1.000),
            Self::LocalTcp   => (0.100, 0.500),
            Self::Cloud      => (10.0,  200.0),
        }
    }

    pub fn is_directly_comparable_to_in_process(&self) -> bool {
        matches!(self, Self::InProcess | Self::LocalGrpc)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOverheadProfile {
    pub database: String,
    pub transport_type: TransportType,
    pub null_query_p50_ms: f64,
    pub null_query_p99_ms: f64,
    pub null_query_std_ms: f64,
    pub num_samples: usize,
}

pub struct NetworkOverheadCalibrator;

impl NetworkOverheadCalibrator {
    pub async fn calibrate(
        adapter: &dyn DatabaseAdapter,
        num_samples: usize,
        transport_type: TransportType,
    ) -> anyhow::Result<NetworkOverheadProfile> {
        for _ in 0..50 {
            let _ = adapter.health_check().await;
        }

        let mut latencies_ms = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let start = Instant::now();
            let null_query = Query {
                id: "null_calibration".into(),
                embeddings: vec![HeadEmbedding {
                    head_name: HeadType::Semantic,
                    vector: vec![0.0; 128],
                }],
                enabled_heads: vec![HeadType::Semantic],
                ground_truth: vec![],
                difficulty: crate::workload::difficulty::DifficultyLevel::Easy,
                failure_mode: None,
            };
            let _ = adapter.query(&null_query, 1).await?;
            latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let mut sorted = latencies_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Ok(NetworkOverheadProfile {
            database: adapter.name().to_string(),
            transport_type,
            null_query_p50_ms: percentile(&sorted, 0.50),
            null_query_p99_ms: percentile(&sorted, 0.99),
            null_query_std_ms: std_dev(&latencies_ms),
            num_samples,
        })
    }

    pub fn subtract_transport_overhead(
        raw_latencies_ms: &[f64],
        overhead_profile: &NetworkOverheadProfile,
    ) -> Vec<f64> {
        let overhead = overhead_profile.null_query_p50_ms;
        raw_latencies_ms
            .iter()
            .map(|&lat| (lat - overhead).max(0.001))
            .collect()
    }
}
