use crate::gpu::backend::GpuBackend;
use crate::gpu::error::GpuError;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;

pub struct CudaBackend {
    device: CudaDevice,
    dot_product_ptx: Ptx,
    matvec_ptx: Ptx,
    fuse_ptx: Ptx,
    gating_ptx: Ptx,
}

impl CudaBackend {
    pub fn new() -> Result<Self, GpuError> {
        let device = CudaDevice::new(0)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let dot_kernel = r#"
            extern "C" __global__ void dot_product(
                const float* __restrict__ query,
                const float* __restrict__ candidates,
                float* __restrict__ scores,
                int dim,
                int num_candidates
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= num_candidates) return;

                float sum = 0.0f;
                const float* cand = candidates + idx * dim;

                for (int i = 0; i < dim; i++) {
                    sum += query[i] * cand[i];
                }
                scores[idx] = sum;
            }
        "#;

        let dot_product_ptx = cudarc::nvrtc::compile_ptx(dot_kernel)
            .map_err(|e| GpuError::Cuda(format!("PTX compilation failed: {:?}", e)))?;

        let matvec_kernel = r#"
            extern "C" __global__ void matvec_batch(
                const float* __restrict__ matrix,
                const float* __restrict__ vectors,
                float* __restrict__ output,
                int dim,
                int num_vectors
            ) {
                int vec_idx = blockIdx.x;
                int row = threadIdx.x;

                if (vec_idx >= num_vectors || row >= dim) return;

                float sum = 0.0f;
                const float* vec = vectors + vec_idx * dim;

                for (int col = 0; col < dim; col++) {
                    sum += matrix[row * dim + col] * vec[col];
                }

                output[vec_idx * dim + row] = sum;
            }
        "#;

        let matvec_ptx = cudarc::nvrtc::compile_ptx(matvec_kernel)
            .map_err(|e| GpuError::Cuda(format!("PTX compilation failed: {:?}", e)))?;

        let fuse_kernel = r#"
            extern "C" __global__ void fuse_weighted(
                const float* __restrict__ head_scores,
                const float* __restrict__ gate_weights,
                float* __restrict__ output,
                int num_heads,
                int num_candidates
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= num_candidates) return;

                float sum = 0.0f;
                for (int h = 0; h < num_heads; h++) {
                    sum += head_scores[h * num_candidates + idx] * gate_weights[h];
                }
                output[idx] = sum;
            }
        "#;

        let fuse_ptx = cudarc::nvrtc::compile_ptx(fuse_kernel)
            .map_err(|e| GpuError::Cuda(format!("PTX fuse kernel error: {:?}", e)))?;

        let gating_kernel = r#"
            extern "C" __global__ void gating_forward(
                const float* __restrict__ query,
                const float* __restrict__ weights,
                const float* __restrict__ bias,
                float* __restrict__ output,
                int dim,
                int num_heads
            ) {
                int h = blockIdx.x * blockDim.x + threadIdx.x;
                if (h >= num_heads) return;

                float sum = bias[h];
                for (int d = 0; d < dim; d++) {
                    sum += weights[h * dim + d] * query[d];
                }
                output[h] = sum;
            }
        "#;

        let gating_ptx = cudarc::nvrtc::compile_ptx(gating_kernel)
            .map_err(|e| GpuError::Cuda(format!("PTX gating kernel error: {:?}", e)))?;

        Ok(Self {
            device,
            dot_product_ptx,
            matvec_ptx,
            fuse_ptx,
            gating_ptx,
        })
    }
}

impl GpuBackend for CudaBackend {
    fn is_available(&self) -> bool {
        true
    }

    fn rerank_exact(
        &self,
        query: &[f32],
        candidates: &[(u64, Vec<f32>)],
        k: usize,
    ) -> Result<Vec<(u64, f32)>, GpuError> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let dim = query.len();
        let num_candidates = candidates.len();

        let mut candidate_vectors = Vec::with_capacity(num_candidates * dim);
        let mut ids = Vec::with_capacity(num_candidates);

        for (id, vec) in candidates {
            if vec.len() != dim {
                return Err(GpuError::InvalidInput("Candidate dimension mismatch".into()));
            }
            candidate_vectors.extend_from_slice(vec);
            ids.push(*id);
        }

        let d_query = self.device.htod_sync_copy(query)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let d_candidates = self.device.htod_sync_copy(&candidate_vectors)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let mut d_scores = self.device.alloc_zeros::<f32>(num_candidates)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let module = self.device.load_ptx(
            self.dot_product_ptx.clone(),
            "dot_product",
            &[],
        ).map_err(|e| GpuError::Cuda(e.to_string()))?;

        let func = module.get_func("dot_product")
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let block_size = 256;
        let grid_size = (num_candidates as u32 + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(config, (&d_query, &d_candidates, &mut d_scores, dim as i32, num_candidates as i32))
                .map_err(|e| GpuError::Cuda(e.to_string()))?;
        }

        let scores = self.device.dtoh_sync_copy(&d_scores)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let mut scored: Vec<(u64, f32)> = ids.into_iter().zip(scores.into_iter()).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored)
    }

    fn project_batch(
        &self,
        matrix: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, GpuError> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let dim = vectors[0].len();
        let num_vectors = vectors.len();

        if matrix.len() != dim * dim {
            return Err(GpuError::InvalidInput(
                format!("Matrix must be dim x dim ({} x {}), got {}", dim, dim, matrix.len())
            ));
        }

        let mut flat_vectors = Vec::with_capacity(num_vectors * dim);
        for vec in &vectors {
            if vec.len() != dim {
                return Err(GpuError::InvalidInput("All vectors must have same dimension".into()));
            }
            flat_vectors.extend_from_slice(vec);
        }

        let d_matrix = self.device.htod_sync_copy(matrix)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let d_vectors = self.device.htod_sync_copy(&flat_vectors)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let mut d_output = self.device.alloc_zeros::<f32>(num_vectors * dim)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let module = self.device.load_ptx(
            self.matvec_ptx.clone(),
            "matvec_batch",
            &[],
        ).map_err(|e| GpuError::Cuda(e.to_string()))?;

        let func = module.get_func("matvec_batch")
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let config = LaunchConfig {
            grid_dim: (num_vectors as u32, 1, 1),
            block_dim: (dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(config, (&d_matrix, &d_vectors, &mut d_output, dim as i32, num_vectors as i32))
                .map_err(|e| GpuError::Cuda(e.to_string()))?;
        }

        let flat_output = self.device.dtoh_sync_copy(&d_output)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let mut results = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let start = i * dim;
            results.push(flat_output[start..start + dim].to_vec());
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

        let num_heads = head_results.len();

        // Collect all unique IDs and build a dense score matrix: heads × candidates
        let mut all_ids: Vec<u64> = Vec::new();
        let mut id_to_col: HashMap<u64, usize> = HashMap::new();
        for (_, results) in head_results.iter() {
            for (id, _) in results {
                if !id_to_col.contains_key(id) {
                    let col = all_ids.len();
                    id_to_col.insert(*id, col);
                    all_ids.push(*id);
                }
            }
        }
        let num_candidates = all_ids.len();

        // Build dense score matrix: [head][candidate]
        let mut flat_scores = vec![0.0f32; num_heads * num_candidates];
        for (h, (_, results)) in head_results.iter().enumerate() {
            for (id, score) in results {
                if let Some(&col) = id_to_col.get(id) {
                    flat_scores[h * num_candidates + col] = *score;
                }
            }
        }

        // Transfer to GPU
        let d_flat_scores = self.device.htod_sync_copy(&flat_scores)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;
        let d_gate_weights = self.device.htod_sync_copy(gate_weights)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;
        let mut d_output = self.device.alloc_zeros::<f32>(num_candidates)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let module = self.device.load_ptx(
            self.fuse_ptx.clone(),
            "fuse_weighted",
            &[],
        ).map_err(|e| GpuError::Cuda(e.to_string()))?;

        let func = module.get_func("fuse_weighted")
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let block_size = 256;
        let grid_size = (num_candidates as u32 + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(config, (
                &d_flat_scores,
                &d_gate_weights,
                &mut d_output,
                num_heads as i32,
                num_candidates as i32,
            )).map_err(|e| GpuError::Cuda(e.to_string()))?;
        }

        let fused_scores = self.device.dtoh_sync_copy(&d_output)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let mut final_scores: Vec<(u64, f32)> = all_ids.into_iter()
            .zip(fused_scores.into_iter())
            .collect();
        final_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(final_scores)
    }

    fn run_gating_network(
        &self,
        query_embedding: &[f32],
        weights: &[f32],
        bias: &[f32],
    ) -> Result<Vec<f32>, GpuError> {
        let dim = query_embedding.len();
        let num_heads = bias.len();

        if weights.len() != num_heads * dim {
            return Err(GpuError::InvalidInput("Weights must be num_heads × dim".into()));
        }

        let d_query = self.device.htod_sync_copy(query_embedding)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;
        let d_weights = self.device.htod_sync_copy(weights)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;
        let d_bias = self.device.htod_sync_copy(bias)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;
        let mut d_logits = self.device.alloc_zeros::<f32>(num_heads)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let module = self.device.load_ptx(
            self.gating_ptx.clone(),
            "gating_forward",
            &[],
        ).map_err(|e| GpuError::Cuda(e.to_string()))?;

        let func = module.get_func("gating_forward")
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let block_size = 256;
        let grid_size = (num_heads as u32 + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(config, (
                &d_query,
                &d_weights,
                &d_bias,
                &mut d_logits,
                dim as i32,
                num_heads as i32,
            )).map_err(|e| GpuError::Cuda(e.to_string()))?;
        }

        let logits = self.device.dtoh_sync_copy(&d_logits)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        // Softmax on CPU (small num_heads, negligible overhead)
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
        let sum: f32 = exp.iter().sum();

        Ok(exp.into_iter().map(|x| x / sum).collect())
    }
}
