use crate::errors::{AttentionError, Result};
use serde::{Deserialize, Serialize};

/// Per-head alignment projection: maps from retrieval-head dimension d_h to attention dimension d_a.
/// P_h ∈ R^{d_h × d_a}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentProjection {
    pub input_dim: usize,
    pub output_dim: usize,
    /// Row-major weights: [input_dim × output_dim]
    pub weights: Vec<f32>,
    /// Bias: [output_dim]
    pub bias: Vec<f32>,
}

impl AlignmentProjection {
    /// Create a new alignment projection with given weights.
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self> {
        if weights.len() != input_dim * output_dim {
            return Err(AttentionError::InvalidMatrixDimensions(format!(
                "weights len {} != input_dim * output_dim = {}",
                weights.len(),
                input_dim * output_dim
            )));
        }
        if bias.len() != output_dim {
            return Err(AttentionError::DimensionMismatch {
                expected: output_dim,
                found: bias.len(),
            });
        }
        Ok(Self {
            input_dim,
            output_dim,
            weights,
            bias,
        })
    }

    /// Create an identity-like projection where input_dim == output_dim.
    /// Weights = identity matrix, bias = zeros.
    pub fn identity(dim: usize) -> Self {
        let mut weights = vec![0.0; dim * dim];
        for i in 0..dim {
            weights[i * dim + i] = 1.0;
        }
        Self {
            input_dim: dim,
            output_dim: dim,
            weights,
            bias: vec![0.0; dim],
        }
    }

    /// Create a random projection with deterministic xorshift64* RNG.
    pub fn random(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = DetRng::new(seed);
        let scale = 1.0 / (input_dim as f32).sqrt();
        let mut weights = vec![0.0; input_dim * output_dim];
        for w in &mut weights {
            *w = rng.gauss() * scale;
        }
        let bias = vec![0.0; output_dim];
        Self {
            input_dim,
            output_dim,
            weights,
            bias,
        }
    }

    /// Apply the projection: y = x W + b, where x ∈ R^{d_in}, y ∈ R^{d_out}
    pub fn project(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.input_dim {
            return Err(AttentionError::DimensionMismatch {
                expected: self.input_dim,
                found: x.len(),
            });
        }
        let mut y = vec![0.0; self.output_dim];
        for j in 0..self.output_dim {
            let mut sum = self.bias[j];
            for i in 0..self.input_dim {
                sum += x[i] * self.weights[i * self.output_dim + j];
            }
            y[j] = sum;
        }
        Ok(y)
    }

    /// Project a batch of vectors (stacked as rows).
    pub fn project_batch(&self, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        xs.iter().map(|x| self.project(x)).collect()
    }

    /// Get parameter count.
    pub fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

/// Deterministic xorshift64* RNG — matches the repo's convention.
pub struct DetRng(u64);

impl DetRng {
    pub fn new(seed: u64) -> Self {
        DetRng(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}
