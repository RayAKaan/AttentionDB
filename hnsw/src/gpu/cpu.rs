use crate::gpu::backend::GpuBackend;
use crate::gpu::error::GpuError;
use std::collections::HashMap;

pub struct CpuBackend;

impl GpuBackend for CpuBackend {
    fn is_available(&self) -> bool {
        true
    }

    fn rerank_exact(
        &self,
        query: &[f32],
        candidates: &[(u64, Vec<f32>)],
        k: usize,
    ) -> Result<Vec<(u64, f32)>, GpuError> {
        let mut scored: Vec<(u64, f32)> = candidates
            .iter()
            .map(|(id, vec)| {
                let score: f32 = query.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
                (*id, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    fn project_batch(
        &self,
        matrix: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, GpuError> {
        let dim = if vectors.is_empty() {
            return Ok(Vec::new());
        } else {
            vectors[0].len()
        };
        let mut results = Vec::with_capacity(vectors.len());

        for vec in vectors {
            let mut output = vec![0.0; dim];
            for i in 0..dim {
                for j in 0..dim {
                    output[i] += matrix[i * dim + j] * vec[j];
                }
            }
            results.push(output);
        }

        Ok(results)
    }

    fn fuse_scores(
        &self,
        head_results: &[(String, Vec<(u64, f32)>)],
        gate_weights: &[f32],
    ) -> Result<Vec<(u64, f32)>, GpuError> {
        if head_results.is_empty() || gate_weights.is_empty() {
            return Ok(vec![]);
        }

        let mut aggregated: HashMap<u64, f32> = HashMap::new();

        for ((_, results), &weight) in head_results.iter().zip(gate_weights.iter()) {
            for (id, score) in results {
                *aggregated.entry(*id).or_insert(0.0) += score * weight;
            }
        }

        let mut final_scores: Vec<_> = aggregated.into_iter().collect();
        final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(final_scores)
    }

    fn run_gating_network(
        &self,
        query_embedding: &[f32],
        weights: &[f32],
        bias: &[f32],
    ) -> Result<Vec<f32>, GpuError> {
        let num_heads = bias.len();
        let dim = query_embedding.len();

        let mut logits = bias.to_vec();

        for h in 0..num_heads {
            let mut sum = 0.0;
            for d in 0..dim {
                sum += weights[h * dim + d] * query_embedding[d];
            }
            logits[h] += sum;
        }

        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
        let sum: f32 = exp.iter().sum();

        Ok(exp.into_iter().map(|x| x / sum).collect())
    }
}
