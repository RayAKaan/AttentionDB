use crate::workload::{Document, HeadType, Query, DocumentId};
use crate::workload::ground_truth::{cosine_similarity, euclidean_distance};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    VeryHard,
    SuperHard,
    ExtremelyHard,
}

impl DifficultyLevel {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Easy" => Some(DifficultyLevel::Easy),
            "Medium" => Some(DifficultyLevel::Medium),
            "Hard" => Some(DifficultyLevel::Hard),
            "VeryHard" => Some(DifficultyLevel::VeryHard),
            "SuperHard" => Some(DifficultyLevel::SuperHard),
            "ExtremelyHard" => Some(DifficultyLevel::ExtremelyHard),
            _ => None,
        }
    }

    pub fn targets(&self) -> DifficultyTargets {
        match self {
            Self::Easy => DifficultyTargets {
                target_contrast_ratio: 0.15,
                target_cluster_overlap: 0.00,
                intra_cluster_noise_std: 0.05,
                cluster_density_fraction: 0.01,
            },
            Self::Medium => DifficultyTargets {
                target_contrast_ratio: 0.30,
                target_cluster_overlap: 0.10,
                intra_cluster_noise_std: 0.10,
                cluster_density_fraction: 0.02,
            },
            Self::Hard => DifficultyTargets {
                target_contrast_ratio: 0.50,
                target_cluster_overlap: 0.20,
                intra_cluster_noise_std: 0.15,
                cluster_density_fraction: 0.05,
            },
            Self::VeryHard => DifficultyTargets {
                target_contrast_ratio: 0.65,
                target_cluster_overlap: 0.35,
                intra_cluster_noise_std: 0.20,
                cluster_density_fraction: 0.10,
            },
            Self::SuperHard => DifficultyTargets {
                target_contrast_ratio: 0.75,
                target_cluster_overlap: 0.50,
                intra_cluster_noise_std: 0.25,
                cluster_density_fraction: 0.20,
            },
            Self::ExtremelyHard => DifficultyTargets {
                target_contrast_ratio: 0.85,
                target_cluster_overlap: 0.65,
                intra_cluster_noise_std: 0.30,
                cluster_density_fraction: 0.40,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct DifficultyTargets {
    pub target_contrast_ratio: f64,
    pub target_cluster_overlap: f64,
    pub intra_cluster_noise_std: f64,
    pub cluster_density_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredDifficultyProperties {
    pub contrast_ratio: f64,
    pub mean_lid: f64,
    pub std_lid: f64,
    pub hubness_skewness: f64,
    pub unambiguous_top1_fraction: f64,
}

impl MeasuredDifficultyProperties {
    pub fn measure(
        documents: &[Document],
        queries: &[Query],
        ground_truth: &[Vec<DocumentId>],
        head: HeadType,
        sample_size: usize,
        rng: &mut StdRng,
    ) -> anyhow::Result<Self> {
        let doc_sample: Vec<&Document> = documents
            .choose_multiple(rng, sample_size.min(documents.len()))
            .collect();

        let contrast_ratio = compute_contrast_ratio(&doc_sample, queries, head.clone())?;
        let (mean_lid, std_lid) = compute_lid_approx(&doc_sample, head.clone())?;
        let hubness_skewness = compute_hubness_skewness(&doc_sample, 10, head.clone())?;
        let unambiguous = compute_unambiguous_fraction(queries, ground_truth, documents, head.clone())?;

        Ok(Self {
            contrast_ratio,
            mean_lid,
            std_lid,
            hubness_skewness,
            unambiguous_top1_fraction: unambiguous,
        })
    }
}

fn compute_contrast_ratio(
    docs: &[&Document],
    queries: &[Query],
    head: HeadType,
) -> anyhow::Result<f64> {
    if queries.is_empty() || docs.is_empty() {
        return Ok(0.0);
    }

    let mut true_neighbor_dists = Vec::new();
    let mut random_dists = Vec::new();

    for query in queries.iter().take(200) {
        let q_vec = match query.embeddings.iter().find(|e| e.head_name == head) {
            Some(e) => &e.vector,
            None => continue,
        };

        if let Some(gt_id) = query.ground_truth.first() {
            if let Some(gt_doc) = docs.iter().find(|d| d.id == *gt_id) {
                if let Some(gt_emb) = gt_doc.embeddings.iter().find(|e| e.head_name == head) {
                    true_neighbor_dists.push(euclidean_distance(q_vec, &gt_emb.vector) as f64);
                }
            }
        }

        let rand_idx = query.id.len() % docs.len();
        let rand_doc = docs[rand_idx];
        if let Some(rand_emb) = rand_doc.embeddings.iter().find(|e| e.head_name == head) {
            random_dists.push(euclidean_distance(q_vec, &rand_emb.vector) as f64);
        }
    }

    if true_neighbor_dists.is_empty() || random_dists.is_empty() {
        return Ok(0.5);
    }

    let mean_true: f64 = true_neighbor_dists.iter().sum::<f64>() / true_neighbor_dists.len() as f64;
    let mean_rand: f64 = random_dists.iter().sum::<f64>() / random_dists.len() as f64;
    if mean_rand < f64::EPSILON { return Ok(0.5); }

    Ok(mean_true / mean_rand)
}

fn compute_lid_approx(
    docs: &[&Document],
    head: HeadType,
) -> anyhow::Result<(f64, f64)> {
    let query_sample: Vec<&Document> = docs.iter().take(200).copied().collect();
    let mut lids = Vec::new();

    for q in &query_sample {
        let q_vec = match q.embeddings.iter().find(|e| e.head_name == head) {
            Some(e) => &e.vector,
            None => continue,
        };

        let mut dists: Vec<f64> = docs.iter()
            .filter(|d| d.id != q.id)
            .map(|d| {
                d.embeddings.iter().find(|e| e.head_name == head)
                    .map(|e| euclidean_distance(q_vec, &e.vector) as f64)
                    .unwrap_or(f64::MAX)
            })
            .filter(|&d| d < f64::MAX)
            .collect();

        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if dists.len() >= 10 {
            let k = 10.min(dists.len());
            let r_k = dists[k - 1];
            if r_k > f64::EPSILON {
                let sum_log: f64 = dists[..k - 1].iter()
                    .map(|&r| {
                        let ratio = r / r_k;
                        if ratio < f64::EPSILON { 0.0 } else { ratio.ln() }
                    })
                    .sum();
                if sum_log.abs() > f64::EPSILON {
                    let lid = -((k - 1) as f64) / sum_log;
                    lids.push(lid);
                }
            }
        }
    }

    if lids.is_empty() { return Ok((3.0, 1.0)); }
    let mean: f64 = lids.iter().sum::<f64>() / lids.len() as f64;
    let std: f64 = (lids.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / lids.len() as f64).sqrt();
    Ok((mean, std))
}

fn compute_hubness_skewness(
    docs: &[&Document],
    k: usize,
    head: HeadType,
) -> anyhow::Result<f64> {
    if docs.is_empty() { return Ok(0.0); }

    let mut occurrence_counts = vec![0usize; docs.len()];
    let query_sample: Vec<&Document> = docs.iter().take(200).copied().collect();

    for q in &query_sample {
        let q_vec = match q.embeddings.iter().find(|e| e.head_name == head) {
            Some(e) => &e.vector,
            None => continue,
        };

        let mut dists: Vec<(usize, f32)> = docs.iter()
            .enumerate()
            .map(|(i, d)| {
                let dist = d.embeddings.iter().find(|e| e.head_name == head)
                    .map(|e| euclidean_distance(q_vec, &e.vector))
                    .unwrap_or(f32::MAX);
                (i, dist)
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for &(idx, _) in dists.iter().take(k) {
            occurrence_counts[idx] += 1;
        }
    }

    let counts_f64: Vec<f64> = occurrence_counts.iter().map(|&c| c as f64).collect();
    let n = counts_f64.len() as f64;
    let mean: f64 = counts_f64.iter().sum::<f64>() / n;
    let std: f64 = (counts_f64.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();

    if std < f64::EPSILON { return Ok(0.0); }

    let skewness: f64 = counts_f64.iter()
        .map(|x| ((x - mean) / std).powi(3))
        .sum::<f64>() / n;

    Ok(skewness)
}

fn compute_unambiguous_fraction(
    queries: &[Query],
    ground_truth: &[Vec<DocumentId>],
    documents: &[Document],
    head: HeadType,
) -> anyhow::Result<f64> {
    if queries.is_empty() || ground_truth.is_empty() { return Ok(0.0); }

    let mut unambiguous = 0;
    let total = queries.len().min(ground_truth.len());

    for (query, gt) in queries.iter().zip(ground_truth.iter()).take(total) {
        if gt.len() < 2 {
            unambiguous += 1;
            continue;
        }

        let q_vec = match query.embeddings.iter().find(|e| e.head_name == head) {
            Some(e) => &e.vector,
            None => continue,
        };

        let scores: Vec<f32> = gt.iter().take(2).filter_map(|doc_id| {
            documents.iter().find(|d| d.id == *doc_id)
                .and_then(|d| d.embeddings.iter().find(|e| e.head_name == head))
                .map(|e| cosine_similarity(q_vec, &e.vector))
        }).collect();

        if scores.len() >= 2 && (scores[0] - scores[1]).abs() > 0.05 {
            unambiguous += 1;
        }
    }

    Ok(unambiguous as f64 / total as f64)
}

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { 0.0 } else { data.iter().sum::<f64>() / data.len() as f64 }
}
