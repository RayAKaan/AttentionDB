use crate::workload::ground_truth::euclidean_distance;
use crate::stats::confidence::{mean, std_dev};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubnessResult {
    pub k: usize,
    pub mean_occurrence: f64,
    pub std_occurrence: f64,
    pub skewness_s_k: f64,
    pub hub_fraction: f64,
    pub antihub_fraction: f64,
}

pub fn compute_hubness(
    queries: &[Vec<f32>],
    documents: &[Vec<f32>],
    k: usize,
) -> HubnessResult {
    let doc_sample: Vec<&Vec<f32>> = documents.iter().collect();
    let query_sample: Vec<&Vec<f32>> = queries.iter().take(1000).collect();

    let mut occurrence_counts: Vec<usize> = vec![0; doc_sample.len()];

    for query in &query_sample {
        let mut dists: Vec<(usize, f32)> = doc_sample.iter()
            .enumerate()
            .map(|(i, d)| (i, euclidean_distance(query, d)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for (doc_idx, _) in dists.iter().take(k.min(dists.len())) {
            occurrence_counts[*doc_idx] += 1;
        }
    }

    let counts_f64: Vec<f64> = occurrence_counts.iter().map(|&c| c as f64).collect();
    let mean_count = mean(&counts_f64);
    let std_count = std_dev(&counts_f64);

    let n = counts_f64.len() as f64;
    let skewness = if std_count > f64::EPSILON {
        counts_f64.iter()
            .map(|x| ((x - mean_count) / std_count).powi(3))
            .sum::<f64>() / n
    } else { 0.0 };

    let hub_threshold = 2.0 * mean_count;
    let hub_fraction = counts_f64.iter()
        .filter(|&&c| c >= hub_threshold).count() as f64 / n;

    let antihub_fraction = counts_f64.iter()
        .filter(|&&c| c < 0.5).count() as f64 / n;

    HubnessResult {
        k,
        mean_occurrence: mean_count,
        std_occurrence: std_count,
        skewness_s_k: skewness,
        hub_fraction,
        antihub_fraction,
    }
}
