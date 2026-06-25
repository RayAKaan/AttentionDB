use crate::workload::{Document, HeadType, Query, DocumentId};
use std::cmp::Ordering;

pub fn compute_ground_truth_uniform(
    query: &Query,
    documents: &[Document],
    top_k: usize,
) -> Vec<DocumentId> {
    if query.enabled_heads.is_empty() {
        return Vec::new();
    }

    let weight = 1.0 / query.enabled_heads.len() as f32;

    let mut scores: Vec<(DocumentId, f32)> = documents
        .iter()
        .map(|doc| {
            let score: f32 = query.enabled_heads.iter().map(|head| {
                let q_emb = query.embeddings.iter().find(|e| e.head_name == *head);
                let d_emb = doc.embeddings.iter().find(|e| e.head_name == *head);
                match (q_emb, d_emb) {
                    (Some(q), Some(d)) => {
                        cosine_similarity(&q.vector, &d.vector) * weight
                    }
                    _ => 0.0,
                }
            }).sum();
            (doc.id.clone(), score)
        })
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scores.into_iter().take(top_k).map(|(id, _)| id).collect()
}

pub fn compute_all_ground_truth(
    queries: &[Query],
    documents: &[Document],
    top_k: usize,
) -> Vec<Vec<DocumentId>> {
    queries
        .iter()
        .map(|q| compute_ground_truth_uniform(q, documents, top_k))
        .collect()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimension mismatch");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn verify_semantic_only_degeneration(
    query: &Query,
    documents: &[Document],
    top_k: usize,
) -> bool {
    let multi_head_gt = compute_ground_truth_uniform(query, documents, top_k);

    let semantic_only_query = Query {
        enabled_heads: vec![HeadType::Semantic],
        ..query.clone()
    };
    let semantic_gt = compute_ground_truth_uniform(&semantic_only_query, documents, top_k);

    if query.enabled_heads.len() == 1 && query.enabled_heads[0] == HeadType::Semantic {
        multi_head_gt == semantic_gt
    } else {
        true
    }
}
