use crate::gpu::backend::GpuBackend;
use crate::gpu::error::GpuError;

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
        let dim = if vectors.is_empty() { return Ok(Vec::new()); } else { vectors[0].len() };
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
}
