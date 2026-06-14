pub fn fuse_scores(
    head_results: &[(String, Vec<(u64, f32)>)],
    gate_weights: &[f32],
) -> Vec<(u64, f32)> {
    let mut aggregated: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();

    for ((_head_name, results), &gate) in head_results.iter().zip(gate_weights.iter()) {
        for (id, score) in results {
            let weighted = score * gate;
            *aggregated.entry(*id).or_insert(0.0) += weighted;
        }
    }

    let mut final_scores: Vec<_> = aggregated.into_iter().collect();
    final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    final_scores
}

pub fn normalize_scores(scores: &mut [(u64, f32)]) {
    let max_score = scores.iter().map(|(_, s)| *s).fold(0.0, f32::max);
    if max_score > 0.0 {
        for (_, score) in scores.iter_mut() {
            *score /= max_score;
        }
    }
}

pub fn weighted_fuse(
    head_results: &[(String, Vec<(u64, f32)>)],
    head_weights: &[(&str, f32)],
) -> Vec<(u64, f32)> {
    let mut aggregated: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();

    let weight_map: std::collections::HashMap<&str, f32> = head_weights.iter()
        .map(|(n, w)| (*n, *w))
        .collect();

    for (head_name, results) in head_results {
        let head_weight = weight_map.get(head_name.as_str()).copied().unwrap_or(1.0);
        for (id, score) in results {
            *aggregated.entry(*id).or_insert(0.0) += score * head_weight;
        }
    }

    let mut final_scores: Vec<_> = aggregated.into_iter().collect();
    final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    final_scores
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_results() -> Vec<(String, Vec<(u64, f32)>)> {
        vec![
            ("semantic".to_string(), vec![(1, 0.9), (2, 0.8)]),
            ("temporal".to_string(), vec![(2, 0.85), (3, 0.7)]),
        ]
    }

    #[test]
    fn test_fuse_aggregates_by_id() {
        let results = make_results();
        let gates = vec![0.6, 0.4];
        let fused = fuse_scores(&results, &gates);
        // ID 2 appears in both heads: 0.8*0.6 + 0.85*0.4 = 0.48 + 0.34 = 0.82
        let id2_score = fused.iter().find(|(id, _)| *id == 2).map(|(_, s)| *s);
        assert!((id2_score.unwrap() - 0.82).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_sorted_descending() {
        let results = make_results();
        let gates = vec![1.0, 1.0];
        let fused = fuse_scores(&results, &gates);
        for w in fused.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn test_normalize_scores() {
        let mut scores = vec![(1, 0.5), (2, 1.0), (3, 0.25)];
        normalize_scores(&mut scores);
        assert!((scores[1].1 - 1.0).abs() < 1e-5);
        assert!((scores[0].1 - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_fuse() {
        let results = make_results();
        let weights = vec![("semantic".to_string(), 1.0), ("temporal".to_string(), 0.5)];
        // Need different type signature
        let weight_refs: Vec<(&str, f32)> = weights.iter().map(|(n, w)| (n.as_str(), *w)).collect();
        let _fused = weighted_fuse(&results, &weight_refs);
    }
}
