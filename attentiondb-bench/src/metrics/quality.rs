use crate::workload::DocumentId;
use std::collections::HashSet;
use std::collections::HashMap;

pub struct QualityScorer;

impl QualityScorer {
    pub fn recall_at_k_binary(
        retrieved: &[DocumentId],
        ground_truth: &[DocumentId],
        k: usize,
    ) -> f64 {
        if ground_truth.is_empty() { return 0.0; }

        let retrieved_top_k: HashSet<&DocumentId> =
            retrieved.iter().take(k).collect();
        let all_relevant: HashSet<&DocumentId> =
            ground_truth.iter().collect();

        let intersection = retrieved_top_k
            .intersection(&all_relevant)
            .count();

        intersection as f64 / all_relevant.len() as f64
    }

    pub fn recall_at_k_exact(
        retrieved: &[DocumentId],
        ground_truth_set: &HashSet<DocumentId>,
        k: usize,
    ) -> f64 {
        if ground_truth_set.is_empty() { return 0.0; }
        let retrieved_top_k: HashSet<DocumentId> =
            retrieved.iter().take(k).cloned().collect();
        let intersection = retrieved_top_k
            .intersection(ground_truth_set)
            .count();
        intersection as f64 / ground_truth_set.len() as f64
    }

    pub fn mrr_binary(
        retrieved: &[DocumentId],
        ground_truth: &[DocumentId],
    ) -> f64 {
        let relevant: HashSet<&DocumentId> = ground_truth.iter().collect();
        for (rank, doc_id) in retrieved.iter().enumerate() {
            if relevant.contains(doc_id) {
                return 1.0 / (rank as f64 + 1.0);
            }
        }
        0.0
    }

    pub fn ndcg_at_k_binary(
        retrieved: &[DocumentId],
        ground_truth: &[DocumentId],
        k: usize,
    ) -> f64 {
        let relevant: HashSet<&DocumentId> = ground_truth.iter().collect();
        let dcg = Self::dcg_binary(retrieved, &relevant, k);
        let ideal_dcg = Self::ideal_dcg_binary(ground_truth.len(), k);
        if ideal_dcg < f64::EPSILON { return 0.0; }
        (dcg / ideal_dcg).min(1.0)
    }

    fn dcg_binary(
        ranked: &[DocumentId],
        relevant: &HashSet<&DocumentId>,
        k: usize,
    ) -> f64 {
        ranked
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, doc_id)| {
                let rel = if relevant.contains(doc_id) { 1.0 } else { 0.0 };
                rel / (i as f64 + 2.0).log2()
            })
            .sum()
    }

    fn ideal_dcg_binary(num_relevant: usize, k: usize) -> f64 {
        (0..num_relevant.min(k))
            .map(|i| 1.0 / (i as f64 + 2.0).log2())
            .sum()
    }

    pub fn ndcg_at_k_graded(
        retrieved: &[DocumentId],
        rel_grades: &HashMap<DocumentId, u8>,
        k: usize,
    ) -> f64 {
        let dcg: f64 = retrieved
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, doc_id)| {
                let rel = *rel_grades.get(doc_id).unwrap_or(&0) as f64;
                (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2()
            })
            .sum();

        let mut ideal_grades: Vec<f64> = rel_grades.values()
            .map(|&g| g as f64)
            .collect();
        ideal_grades.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let ideal_dcg: f64 = ideal_grades
            .iter()
            .take(k)
            .enumerate()
            .map(|(i, &rel)| (2.0_f64.powf(rel) - 1.0) / (i as f64 + 2.0).log2())
            .sum();

        if ideal_dcg < f64::EPSILON { return 0.0; }
        (dcg / ideal_dcg).min(1.0)
    }

    pub fn precision_at_k(
        retrieved: &[DocumentId],
        ground_truth: &[DocumentId],
        k: usize,
    ) -> f64 {
        if k == 0 || retrieved.is_empty() { return 0.0; }
        let relevant: HashSet<&DocumentId> = ground_truth.iter().collect();
        let count = retrieved.iter().take(k).filter(|id| relevant.contains(*id)).count();
        count as f64 / k as f64
    }

    pub fn compute_all(
        retrieved: &[DocumentId],
        ground_truth: &[DocumentId],
        k: usize,
    ) -> QualityMetrics {
        QualityMetrics {
            recall_at_1: Self::recall_at_k_binary(retrieved, ground_truth, 1),
            recall_at_10: Self::recall_at_k_binary(retrieved, ground_truth, 10.min(k)),
            recall_at_100: Self::recall_at_k_binary(retrieved, ground_truth, 100.min(k)),
            mrr: Self::mrr_binary(retrieved, ground_truth),
            ndcg_at_10: Self::ndcg_at_k_binary(retrieved, ground_truth, 10.min(k)),
            ndcg_at_100: Self::ndcg_at_k_binary(retrieved, ground_truth, 100.min(k)),
            precision_at_10: Self::precision_at_k(retrieved, ground_truth, 10.min(k)),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityMetrics {
    pub recall_at_1: f64,
    pub recall_at_10: f64,
    pub recall_at_100: f64,
    pub mrr: f64,
    pub ndcg_at_10: f64,
    pub ndcg_at_100: f64,
    pub precision_at_10: f64,
}
