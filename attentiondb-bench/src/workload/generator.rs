use crate::workload::{Document, HeadEmbedding, HeadType, Query, DocumentId};
use crate::workload::difficulty::DifficultyLevel;
use crate::workload::failure_modes::{FailureMode, apply_failure_to_document, apply_failure_to_query};
use crate::workload::ground_truth::compute_all_ground_truth;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCorpus {
    pub documents: Vec<Document>,
    pub queries: Vec<Query>,
    pub ground_truth: Vec<Vec<DocumentId>>,
}

pub struct WorkloadGenerator {
    rng: StdRng,
    dimension: usize,
}

impl WorkloadGenerator {
    pub fn new(seed: u64, dimension: usize) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            dimension,
        }
    }

    pub fn generate_corpus(
        &mut self,
        n_docs: usize,
        n_queries: usize,
        difficulty: DifficultyLevel,
        head_combination: Vec<HeadType>,
        failure_mode: Option<FailureMode>,
    ) -> GeneratedCorpus {
        let targets = difficulty.targets();
        let n_clusters = ((n_docs as f64) * targets.cluster_density_fraction).ceil() as usize;
        let n_clusters = n_clusters.max(1);

        let cluster_centers: Vec<Vec<Vec<f32>>> = (0..n_clusters)
            .map(|_| {
                head_combination.iter().map(|_| {
                    self.random_unit_vector()
                }).collect()
            })
            .collect();

        let mut documents = Vec::with_capacity(n_docs);

        for i in 0..n_docs {
            let cluster_idx = i % n_clusters;

            let embeddings: Vec<HeadEmbedding> = head_combination.iter().enumerate().map(|(head_idx, head)| {
                let center = &cluster_centers[cluster_idx][head_idx];
                let mut vector = center.clone();

                for v in vector.iter_mut() {
                    *v += self.rng.gen::<f32>() * targets.intra_cluster_noise_std as f32;
                }

                self.normalize(&mut vector);
                HeadEmbedding { head_name: head.clone(), vector }
            }).collect();

            documents.push(Document {
                id: format!("doc_{}", i),
                embeddings,
                metadata: HashMap::new(),
                timestamp: Some(i as i64),
            });
        }

        if let Some(ref mode) = failure_mode {
            for doc in documents.iter_mut() {
                apply_failure_to_document(doc, mode, &mut self.rng);
            }
        }

        let mut queries: Vec<Query> = (0..n_queries)
            .map(|i| {
                let base_doc = &documents[i % n_docs];
                let embeddings: Vec<HeadEmbedding> = head_combination.iter().map(|head| {
                    let base_vec = base_doc.embeddings.iter()
                        .find(|e| e.head_name == *head)
                        .map(|e| e.vector.clone())
                        .unwrap_or_else(|| self.random_unit_vector());

                    HeadEmbedding { head_name: head.clone(), vector: base_vec }
                }).collect();

                Query {
                    id: format!("query_{}", i),
                    embeddings,
                    enabled_heads: head_combination.clone(),
                    ground_truth: Vec::new(),
                    difficulty,
                    failure_mode: failure_mode.clone(),
                }
            })
            .collect();

        if let Some(ref mode) = failure_mode {
            for query in queries.iter_mut() {
                apply_failure_to_query(query, mode, &mut self.rng);
            }
        }

        let ground_truth = compute_all_ground_truth(&queries, &documents, 100);

        for (i, gt) in ground_truth.iter().enumerate() {
            queries[i].ground_truth = gt.clone();
        }

        GeneratedCorpus { documents, queries, ground_truth }
    }

    pub fn generate_synthetic(
        &mut self,
        n_docs: usize,
        n_queries: usize,
        dimension: usize,
    ) -> GeneratedCorpus {
        let mut documents = Vec::with_capacity(n_docs);
        for i in 0..n_docs {
            let mut vec = (0..dimension)
                .map(|_| self.rng.gen::<f32>() - 0.5)
                .collect::<Vec<f32>>();
            self.normalize(&mut vec);

            documents.push(Document {
                id: format!("doc_{}", i),
                embeddings: vec![HeadEmbedding {
                    head_name: HeadType::Semantic,
                    vector: vec,
                }],
                metadata: HashMap::new(),
                timestamp: None,
            });
        }

        let mut queries = Vec::with_capacity(n_queries);
        for i in 0..n_queries {
            let mut vec = (0..dimension)
                .map(|_| self.rng.gen::<f32>() - 0.5)
                .collect::<Vec<f32>>();
            self.normalize(&mut vec);

            queries.push(Query {
                id: format!("query_{}", i),
                embeddings: vec![HeadEmbedding {
                    head_name: HeadType::Semantic,
                    vector: vec,
                }],
                enabled_heads: vec![HeadType::Semantic],
                ground_truth: Vec::new(),
                difficulty: DifficultyLevel::Easy,
                failure_mode: None,
            });
        }

        let ground_truth = compute_all_ground_truth(&queries, &documents, 100);
        for (i, gt) in ground_truth.iter().enumerate() {
            queries[i].ground_truth = gt.clone();
        }

        GeneratedCorpus { documents, queries, ground_truth }
    }

    fn random_unit_vector(&mut self) -> Vec<f32> {
        let mut vec: Vec<f32> = (0..self.dimension)
            .map(|_| self.rng.gen::<f32>() * 2.0 - 1.0)
            .collect();
        self.normalize(&mut vec);
        vec
    }

    fn normalize(&self, vec: &mut Vec<f32>) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
    }
}
