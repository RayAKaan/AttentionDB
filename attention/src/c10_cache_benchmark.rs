//! C10 cache-aware serving benchmark and lifecycle correctness gates.
//!
//! Timings cover document-side K/V projection and cache operations only. They
//! are not end-to-end retrieval latency and must not be reported as such.

use crate::cache::{AttentionKVCache, CacheFingerprint, CacheStats, CachedCandidateKV};
use crate::errors::{AttentionError, Result};
use crate::projection::QkvProjection;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::mem::size_of;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C10CacheBenchmarkReport {
    pub candidate_count: usize,
    pub head_count: usize,
    pub attention_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub repetitions: usize,
    pub uncached_projection_median_micros: u128,
    pub uncached_projection_p95_micros: u128,
    pub cold_fill_median_micros: u128,
    pub cold_fill_p95_micros: u128,
    pub warm_lookup_median_micros: u128,
    pub warm_lookup_p95_micros: u128,
    pub warm_hits: u64,
    pub warm_misses: u64,
    pub warm_hit_rate: f32,
    pub resident_entries: usize,
    /// Payload estimate only; excludes HashMap, allocator, and vector overhead.
    pub estimated_kv_payload_bytes: usize,
    pub exact_kv_parity: bool,
}

fn percentile_nearest_rank(samples: &[u128], percentile: usize) -> u128 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let index = (percentile * sorted.len()).div_ceil(100).saturating_sub(1);
    sorted[index.min(sorted.len() - 1)]
}

fn project_candidate(z_d: &[Vec<f32>], qkv: &QkvProjection) -> Result<CachedCandidateKV> {
    if z_d.is_empty() {
        return Err(AttentionError::EmptyInput(
            "C10 candidate representation cannot be empty".into(),
        ));
    }
    Ok(CachedCandidateKV {
        keys: qkv.project_k(z_d)?,
        values: qkv.project_v(z_d)?,
    })
}

/// Benchmark uncached projection, cold cache fill, and warm cache lookup on
/// identical candidates. IDs must be unique; at least three measured rounds
/// are required. Two warm-up rounds are discarded for each measured path.
pub fn benchmark_cache_lifecycle(
    qkv: &QkvProjection,
    candidates: &[(u64, Vec<Vec<f32>>)],
    model_fingerprint: u64,
    repetitions: usize,
) -> Result<C10CacheBenchmarkReport> {
    if candidates.is_empty() {
        return Err(AttentionError::EmptyInput(
            "C10 benchmark requires at least one candidate".into(),
        ));
    }
    if repetitions < 3 {
        return Err(AttentionError::EmptyInput(
            "C10 benchmark requires at least 3 measured repetitions".into(),
        ));
    }

    let expected_heads = candidates[0].1.len();
    if expected_heads == 0 {
        return Err(AttentionError::EmptyInput(
            "C10 candidate representation cannot be empty".into(),
        ));
    }
    let mut ids = HashSet::with_capacity(candidates.len());
    for (id, representation) in candidates {
        if representation.len() != expected_heads {
            return Err(AttentionError::HeadCountMismatch {
                expected: expected_heads,
                found: representation.len(),
            });
        }
        if !ids.insert(*id) {
            return Err(AttentionError::Config(format!(
                "C10 candidate id {id} is duplicated"
            )));
        }
        // Validate every candidate before starting the timed runs.
        project_candidate(representation, qkv)?;
    }

    let fingerprint =
        CacheFingerprint::from_projection(model_fingerprint, qkv, expected_heads);
    let reference: Vec<_> = candidates
        .iter()
        .map(|(_, z_d)| project_candidate(z_d, qkv))
        .collect::<Result<_>>()?;

    for _ in 0..2 {
        for (_, z_d) in candidates {
            let _ = project_candidate(z_d, qkv)?;
        }
        let mut cache = AttentionKVCache::new(fingerprint);
        for (id, z_d) in candidates {
            cache.insert(*id, project_candidate(z_d, qkv)?);
        }
        let mut stats = CacheStats::default();
        for (id, _) in candidates {
            let _ = cache.lookup(*id, &mut stats);
        }
    }

    let mut uncached_times = Vec::with_capacity(repetitions);
    let mut cold_fill_times = Vec::with_capacity(repetitions);
    let mut warm_lookup_times = Vec::with_capacity(repetitions);
    let mut exact_kv_parity = true;
    let mut final_cache = AttentionKVCache::new(fingerprint);
    let mut aggregate_warm_stats = CacheStats::default();

    for _ in 0..repetitions {
        let start = Instant::now();
        let uncached: Vec<_> = candidates
            .iter()
            .map(|(_, z_d)| project_candidate(z_d, qkv))
            .collect::<Result<_>>()?;
        uncached_times.push(start.elapsed().as_micros());
        exact_kv_parity &= uncached == reference;

        let mut cold_cache = AttentionKVCache::new(fingerprint);
        let start = Instant::now();
        for ((id, z_d), expected) in candidates.iter().zip(&reference) {
            let kv = project_candidate(z_d, qkv)?;
            exact_kv_parity &= expected == &kv;
            cold_cache.insert(*id, kv);
        }
        cold_fill_times.push(start.elapsed().as_micros());
        exact_kv_parity &= cold_cache.validate_geometry().is_ok();

        let mut stats = CacheStats::default();
        let start = Instant::now();
        for (id, _) in candidates {
            if cold_cache.lookup(*id, &mut stats).is_none() {
                exact_kv_parity = false;
            }
        }
        warm_lookup_times.push(start.elapsed().as_micros());
        exact_kv_parity &= stats.hits == candidates.len() as u64 && stats.misses == 0;
        aggregate_warm_stats.merge(&stats);
        final_cache = cold_cache;
    }

    final_cache.seal_stats(&mut aggregate_warm_stats);
    let estimated_kv_payload_bytes = final_cache.len()
        * fingerprint.head_count
        * (fingerprint.key_dim + fingerprint.value_dim)
        * size_of::<f32>();

    Ok(C10CacheBenchmarkReport {
        candidate_count: candidates.len(),
        head_count: fingerprint.head_count,
        attention_dim: fingerprint.attention_dim,
        key_dim: fingerprint.key_dim,
        value_dim: fingerprint.value_dim,
        repetitions,
        uncached_projection_median_micros: percentile_nearest_rank(&uncached_times, 50),
        uncached_projection_p95_micros: percentile_nearest_rank(&uncached_times, 95),
        cold_fill_median_micros: percentile_nearest_rank(&cold_fill_times, 50),
        cold_fill_p95_micros: percentile_nearest_rank(&cold_fill_times, 95),
        warm_lookup_median_micros: percentile_nearest_rank(&warm_lookup_times, 50),
        warm_lookup_p95_micros: percentile_nearest_rank(&warm_lookup_times, 95),
        warm_hits: aggregate_warm_stats.hits,
        warm_misses: aggregate_warm_stats.misses,
        warm_hit_rate: aggregate_warm_stats.hit_rate(),
        resident_entries: aggregate_warm_stats.entries,
        estimated_kv_payload_bytes,
        exact_kv_parity,
    })
}

/// Configurable dimensions for a reproducible cache-lifecycle sweep.
/// Use `C10SweepConfig::default()` for the preregistered matrix; callers can
/// pass smaller vectors for smoke runs. The default matrix is intentionally
/// not executed by unit tests because it is a long-running experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C10SweepConfig {
    pub candidate_counts: Vec<usize>,
    pub head_counts: Vec<usize>,
    pub attention_dims: Vec<usize>,
    pub key_dims: Vec<usize>,
    pub value_dims: Vec<usize>,
    pub repetitions: usize,
    pub seed: u64,
}

impl Default for C10SweepConfig {
    fn default() -> Self {
        Self {
            candidate_counts: vec![32, 64, 128, 256, 500, 1_000, 2_000],
            head_counts: vec![1, 3, 8],
            attention_dims: vec![64, 128, 384],
            key_dims: vec![32, 64, 128],
            value_dims: vec![32, 64, 128],
            repetitions: 5,
            seed: 20261009,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct C10SweepCell {
    pub cell_index: usize,
    pub seed: u64,
    pub report: C10CacheBenchmarkReport,
}

fn deterministic_candidate(
    doc_id: usize,
    head_count: usize,
    attention_dim: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    (0..head_count)
        .map(|head| {
            (0..attention_dim)
                .map(|dimension| {
                    let mixed = seed
                        .wrapping_add((doc_id as u64).wrapping_mul(1_000_003))
                        .wrapping_add((head as u64).wrapping_mul(9_176))
                        .wrapping_add((dimension as u64).wrapping_mul(131))
                        .wrapping_mul(2_654_435_761);
                    ((mixed % 2_001) as i32 - 1_000) as f32 / 1_000.0
                })
                .collect()
        })
        .collect()
}

/// Run the full requested Cartesian sweep and return one provenance-keyed
/// report per cell. The function does not write files: callers should serialize
/// each returned cell as JSONL and preserve it as an immutable raw run.
pub fn run_cache_lifecycle_sweep(config: &C10SweepConfig) -> Result<Vec<C10SweepCell>> {
    if config.candidate_counts.is_empty()
        || config.head_counts.is_empty()
        || config.attention_dims.is_empty()
        || config.key_dims.is_empty()
        || config.value_dims.is_empty()
    {
        return Err(AttentionError::Config(
            "C10 sweep dimensions must all be non-empty".into(),
        ));
    }
    if config.repetitions < 3 {
        return Err(AttentionError::Config(
            "C10 sweep requires at least 3 measured repetitions".into(),
        ));
    }
    if config.candidate_counts.contains(&0)
        || config.head_counts.contains(&0)
        || config.attention_dims.contains(&0)
        || config.key_dims.contains(&0)
        || config.value_dims.contains(&0)
    {
        return Err(AttentionError::Config(
            "C10 sweep dimensions must be positive".into(),
        ));
    }

    let cell_count = config.candidate_counts.len()
        * config.head_counts.len()
        * config.attention_dims.len()
        * config.key_dims.len()
        * config.value_dims.len();
    let mut cells = Vec::with_capacity(cell_count);
    let mut cell_index = 0usize;

    for &candidate_count in &config.candidate_counts {
        for &head_count in &config.head_counts {
            for &attention_dim in &config.attention_dims {
                for &key_dim in &config.key_dims {
                    for &value_dim in &config.value_dims {
                        let cell_seed = config.seed.wrapping_add(cell_index as u64);
                        let qkv = QkvProjection::random(
                            attention_dim,
                            key_dim,
                            value_dim,
                            cell_seed,
                        );
                        let candidates: Vec<_> = (0..candidate_count)
                            .map(|doc_id| {
                                (
                                    doc_id as u64,
                                    deterministic_candidate(
                                        doc_id,
                                        head_count,
                                        attention_dim,
                                        cell_seed,
                                    ),
                                )
                            })
                            .collect();
                        let fingerprint = qkv.fingerprint();
                        let report = benchmark_cache_lifecycle(
                            &qkv,
                            &candidates,
                            fingerprint,
                            config.repetitions,
                        )?;
                        cells.push(C10SweepCell {
                            cell_index,
                            seed: cell_seed,
                            report,
                        });
                        cell_index += 1;
                    }
                }
            }
        }
    }

    Ok(cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (QkvProjection, Vec<(u64, Vec<Vec<f32>>)>) {
        let qkv = QkvProjection::random(4, 3, 2, 10);
        let candidates = vec![
            (11, vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 0.0, 1.0]]),
            (12, vec![vec![0.0, 0.0, 1.0, 1.0], vec![1.0, 1.0, 0.0, 0.0]]),
        ];
        (qkv, candidates)
    }

    #[test]
    fn benchmark_reports_exact_parity_and_warm_hits() {
        let (qkv, candidates) = fixture();
        let report = benchmark_cache_lifecycle(&qkv, &candidates, 123, 3).unwrap();
        assert!(report.exact_kv_parity);
        assert_eq!(report.candidate_count, 2);
        assert_eq!(report.warm_hits, 6);
        assert_eq!(report.warm_misses, 0);
        assert_eq!(report.warm_hit_rate, 1.0);
        assert_eq!(report.resident_entries, 2);
        assert!(report.estimated_kv_payload_bytes > 0);
        assert!(report.uncached_projection_p95_micros >= report.uncached_projection_median_micros);
        assert!(report.cold_fill_p95_micros >= report.cold_fill_median_micros);
        assert!(report.warm_lookup_p95_micros >= report.warm_lookup_median_micros);
    }

    #[test]
    fn benchmark_rejects_empty_candidates_and_too_few_repetitions() {
        let (qkv, candidates) = fixture();
        assert!(benchmark_cache_lifecycle(&qkv, &[], 123, 3).is_err());
        assert!(benchmark_cache_lifecycle(&qkv, &candidates, 123, 2).is_err());
    }

    #[test]
    fn benchmark_rejects_duplicate_ids_and_malformed_dimensions() {
        let (qkv, candidates) = fixture();
        let duplicate = vec![candidates[0].clone(), candidates[0].clone()];
        assert!(benchmark_cache_lifecycle(&qkv, &duplicate, 123, 3).is_err());

        let malformed = vec![(1, vec![vec![1.0, 2.0]])];
        assert!(benchmark_cache_lifecycle(&qkv, &malformed, 123, 3).is_err());

        let uneven_heads = vec![
            (1, vec![vec![0.0; 4], vec![1.0; 4]]),
            (2, vec![vec![0.5; 4]]),
        ];
        assert!(benchmark_cache_lifecycle(&qkv, &uneven_heads, 123, 3).is_err());
    }

    #[test]
    fn deterministic_sweep_runs_a_small_cell_and_rejects_invalid_config() {
        let config = C10SweepConfig {
            candidate_counts: vec![2],
            head_counts: vec![2],
            attention_dims: vec![4],
            key_dims: vec![3],
            value_dims: vec![2],
            repetitions: 3,
            seed: 77,
        };
        let cells = run_cache_lifecycle_sweep(&config).unwrap();
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].cell_index, 0);
        assert_eq!(cells[0].seed, 77);
        assert!(cells[0].report.exact_kv_parity);
        assert_eq!(cells[0].report.warm_hits, 6);

        let invalid = C10SweepConfig {
            candidate_counts: Vec::new(),
            ..config
        };
        assert!(run_cache_lifecycle_sweep(&invalid).is_err());
    }

    #[test]
    fn cache_fingerprint_changes_when_model_changes() {
        let (qkv, _) = fixture();
        let original = CacheFingerprint::from_projection(123, &qkv, 2);
        let changed_model = CacheFingerprint::from_projection(124, &qkv, 2);
        let changed_geometry =
            CacheFingerprint::from_projection(123, &QkvProjection::random(4, 2, 2, 10), 2);
        assert!(!original.accepts(&changed_model));
        assert!(!original.accepts(&changed_geometry));
    }
}
