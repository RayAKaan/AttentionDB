//! C9 deterministic microbenchmark helpers for scalar-vs-batch attention.
//! These helpers measure the attention kernel only; they do not include HNSW,
//! network, storage, or dataset loading time and must not be reported as end-to-end latency.

use crate::errors::{AttentionError, Result};
use crate::qkv::{AttentionEngine, CandidateAttention};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C9BenchmarkReport {
    pub candidate_count: usize,
    pub repetitions: usize,
    pub scalar_median_micros: u128,
    pub scalar_p95_micros: u128,
    pub batch_median_micros: u128,
    pub batch_p95_micros: u128,
    pub scalar_candidates_per_second: f64,
    pub batch_candidates_per_second: f64,
    pub speedup: f64,
    pub exact_output_parity: bool,
}

fn percentile_nearest_rank(samples: &[u128], percentile: usize) -> u128 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let index = ((percentile * sorted.len() + 99) / 100).saturating_sub(1);
    sorted[index.min(sorted.len() - 1)]
}

fn same_output(a: &[CandidateAttention], b: &[CandidateAttention]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.weights == y.weights
                && x.logits == y.logits
                && x.output == y.output
                && x.entropy == y.entropy
        })
}

/// Compare the scalar reference with the C9 batch path on the exact same inputs.
/// Two warm-up rounds are discarded. At least three measured repetitions are required.
pub fn benchmark_scalar_vs_batch(
    engine: &AttentionEngine,
    query: &[f32],
    candidates: &[Vec<Vec<f32>>],
    repetitions: usize,
) -> Result<C9BenchmarkReport> {
    if repetitions < 3 {
        return Err(AttentionError::EmptyInput(
            "C9 benchmark requires at least 3 measured repetitions".into(),
        ));
    }

    for _ in 0..2 {
        let scalar: Result<Vec<_>> = candidates.iter().map(|c| engine.attend(query, c)).collect();
        scalar?;
        engine.attend_batch(query, candidates)?;
    }

    let mut scalar_times = Vec::with_capacity(repetitions);
    let mut batch_times = Vec::with_capacity(repetitions);
    let mut exact_output_parity = true;
    for _ in 0..repetitions {
        let start = Instant::now();
        let scalar: Vec<_> = candidates
            .iter()
            .map(|c| engine.attend(query, c))
            .collect::<Result<_>>()?;
        scalar_times.push(start.elapsed().as_micros());

        let start = Instant::now();
        let batch = engine.attend_batch(query, candidates)?;
        batch_times.push(start.elapsed().as_micros());
        exact_output_parity &= same_output(&scalar, &batch);
    }

    let scalar_median_micros = percentile_nearest_rank(&scalar_times, 50);
    let batch_median_micros = percentile_nearest_rank(&batch_times, 50);
    let scalar_total: u128 = scalar_times.iter().sum();
    let batch_total: u128 = batch_times.iter().sum();
    let scalar_candidates_per_second = if scalar_total == 0 {
        0.0
    } else {
        (candidates.len() * repetitions) as f64 * 1_000_000.0 / scalar_total as f64
    };
    let batch_candidates_per_second = if batch_total == 0 {
        0.0
    } else {
        (candidates.len() * repetitions) as f64 * 1_000_000.0 / batch_total as f64
    };
    let speedup = if batch_median_micros == 0 {
        0.0
    } else {
        scalar_median_micros as f64 / batch_median_micros as f64
    };

    Ok(C9BenchmarkReport {
        candidate_count: candidates.len(),
        repetitions,
        scalar_median_micros,
        scalar_p95_micros: percentile_nearest_rank(&scalar_times, 95),
        batch_median_micros,
        batch_p95_micros: percentile_nearest_rank(&batch_times, 95),
        scalar_candidates_per_second,
        batch_candidates_per_second,
        speedup,
        exact_output_parity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::QkvProjection;

    #[test]
    fn benchmark_reports_parity_and_valid_percentiles() {
        let engine = AttentionEngine::new(QkvProjection::identity(4));
        let query = [0.1, 0.2, 0.3, 0.4];
        let candidates = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]],
        ];
        let report = benchmark_scalar_vs_batch(&engine, &query, &candidates, 3).unwrap();
        assert!(report.exact_output_parity);
        assert_eq!(report.candidate_count, 2);
        assert!(report.scalar_p95_micros >= report.scalar_median_micros);
        assert!(report.batch_p95_micros >= report.batch_median_micros);
    }

    #[test]
    fn benchmark_rejects_too_few_repetitions() {
        let engine = AttentionEngine::new(QkvProjection::identity(2));
        assert!(benchmark_scalar_vs_batch(&engine, &[0.0, 0.0], &[], 2).is_err());
    }
}
