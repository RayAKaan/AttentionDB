use crate::gpu::error::GpuError;

pub trait GpuBackend: Send + Sync {
    /// Returns whether the GPU backend is available on this system
    fn is_available(&self) -> bool;

    /// Performs exact reranking using dot product similarity
    fn rerank_exact(
        &self,
        query: &[f32],
        candidates: &[(u64, Vec<f32>)],
        k: usize,
    ) -> Result<Vec<(u64, f32)>, GpuError>;

    /// Performs batched projection (e.g. for W_Q, W_K, W_V)
    fn project_batch(
        &self,
        matrix: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, GpuError>;
}
