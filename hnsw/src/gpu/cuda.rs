use crate::gpu::backend::GpuBackend;
use crate::gpu::error::GpuError;
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

pub struct CudaBackend {
    device: CudaDevice,
    dot_product_ptx: Ptx,
}

impl CudaBackend {
    pub fn new() -> Result<Self, GpuError> {
        let device = CudaDevice::new(0)
            .map_err(|e| GpuError::Cuda(e.to_string()))?;

        let kernel_code = r#"
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

        let ptx = cudarc::nvrtc::compile_ptx(kernel_code)
            .map_err(|e| GpuError::Cuda(format!("PTX compilation failed: {:?}", e)))?;

        Ok(Self {
            device,
            dot_product_ptx: ptx,
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
        _matrix: &[f32],
        _vectors: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, GpuError> {
        Err(GpuError::NotAvailable)
    }
}
