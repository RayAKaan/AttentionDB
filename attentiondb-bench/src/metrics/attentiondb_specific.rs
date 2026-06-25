use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDBSpecificMetrics {
    pub multi_head_consensus_accuracy: f64,
    pub gating_efficiency_vs_uniform: f64,
    pub per_head_weights: Vec<HeadWeight>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadWeight {
    pub head_name: String,
    pub mean_weight: f64,
    pub std_weight: f64,
    pub contribution_pct: f64,
}

pub fn compute_consensus_accuracy(
    attentiondb_ranked: &[Vec<String>],
    competitor_ranked: &[Vec<String>],
    k: usize,
) -> f64 {
    let mut agreements = 0;
    let total = attentiondb_ranked.len().min(competitor_ranked.len());

    for (a, b) in attentiondb_ranked.iter().zip(competitor_ranked.iter()).take(total) {
        let a_set: std::collections::HashSet<&String> = a.iter().take(k).collect();
        let b_set: std::collections::HashSet<&String> = b.iter().take(k).collect();
        let intersection = a_set.intersection(&b_set).count();
        let union = a_set.union(&b_set).count();
        if union > 0 {
            agreements += intersection;
        }
    }

    agreements as f64 / total as f64
}

pub fn compute_gating_efficiency(
    attentiondb_ndcg: f64,
    uniform_ndcg: f64,
) -> f64 {
    if uniform_ndcg < f64::EPSILON { return 0.0; }
    (attentiondb_ndcg - uniform_ndcg) / uniform_ndcg
}
