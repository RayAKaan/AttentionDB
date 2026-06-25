use crate::workload::ground_truth::euclidean_distance;

pub fn estimate_lid(
    _query_vec: &[f32],
    neighbor_distances: &[f32],
) -> f64 {
    let k = neighbor_distances.len();
    if k < 2 { return 1.0; }

    let r_k = neighbor_distances[k - 1] as f64;
    if r_k < f64::EPSILON { return 1.0; }

    let sum_log: f64 = neighbor_distances[..k - 1]
        .iter()
        .map(|&r| {
            let ratio = r as f64 / r_k;
            if ratio < f64::EPSILON { 0.0 } else { ratio.ln() }
        })
        .sum();

    if sum_log.abs() < f64::EPSILON { return k as f64; }

    -((k - 1) as f64) / sum_log
}

pub fn mean_lid_over_dataset(
    queries: &[Vec<f32>],
    documents: &[Vec<f32>],
    k: usize,
) -> (f64, f64) {
    let sample: Vec<&Vec<f32>> = queries.iter().take(1000).collect();
    let doc_sample: Vec<&Vec<f32>> = documents.iter().take(10_000).collect();

    if sample.is_empty() || doc_sample.is_empty() {
        return (3.0, 1.0);
    }

    let lids: Vec<f64> = sample.iter().map(|q| {
        let mut dists: Vec<f32> = doc_sample.iter()
            .map(|d| euclidean_distance(q, d))
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dists.truncate(k.min(dists.len()));
        if dists.len() < 2 { return 1.0; }
        estimate_lid(q, &dists)
    }).collect();

    let mean: f64 = lids.iter().sum::<f64>() / lids.len() as f64;
    let variance: f64 = lids.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / lids.len() as f64;
    (mean, variance.sqrt())
}
