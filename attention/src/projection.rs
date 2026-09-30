use crate::alignment::DetRng;
use crate::errors::Result;
use serde::{Deserialize, Serialize};

/// Q/K/V projection triple: W_Q, W_K, W_V each in R^{d_a × d_k/d_v}.
/// These are the core learned parameters for genuine QKV attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QkvProjection {
    pub attention_dim: usize, // d_a
    pub key_dim: usize,       // d_k
    pub value_dim: usize,     // d_v
    /// W_Q ∈ R^{d_a × d_k}, row-major [d_a × d_k]
    pub w_q: Vec<f32>,
    /// W_K ∈ R^{d_a × d_k}, row-major [d_a × d_k]
    pub w_k: Vec<f32>,
    /// W_V ∈ R^{d_a × d_v}, row-major [d_a × d_v]
    pub w_v: Vec<f32>,
}

impl QkvProjection {
    /// Create from explicit weights.
    pub fn new(
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
    ) -> Result<Self> {
        if w_q.len() != attention_dim * key_dim {
            return Err(crate::errors::AttentionError::InvalidMatrixDimensions(
                format!(
                    "w_q len {} != attention_dim * key_dim = {}",
                    w_q.len(),
                    attention_dim * key_dim
                ),
            ));
        }
        if w_k.len() != attention_dim * key_dim {
            return Err(crate::errors::AttentionError::InvalidMatrixDimensions(
                format!(
                    "w_k len {} != attention_dim * key_dim = {}",
                    w_k.len(),
                    attention_dim * key_dim
                ),
            ));
        }
        if w_v.len() != attention_dim * value_dim {
            return Err(crate::errors::AttentionError::InvalidMatrixDimensions(
                format!(
                    "w_v len {} != attention_dim * value_dim = {}",
                    w_v.len(),
                    attention_dim * value_dim
                ),
            ));
        }
        Ok(Self {
            attention_dim,
            key_dim,
            value_dim,
            w_q,
            w_k,
            w_v,
        })
    }

    /// Identity-compatible initialization for C7-D (fixed/identity diagnostic).
    /// W_Q, W_K = identity on shared subspace; W_V = identity.
    /// Requires attention_dim == key_dim == value_dim.
    pub fn identity(dim: usize) -> Self {
        let mut w_q = vec![0.0; dim * dim];
        let mut w_k = vec![0.0; dim * dim];
        let mut w_v = vec![0.0; dim * dim];
        for i in 0..dim {
            w_q[i * dim + i] = 1.0;
            w_k[i * dim + i] = 1.0;
            w_v[i * dim + i] = 1.0;
        }
        Self {
            attention_dim: dim,
            key_dim: dim,
            value_dim: dim,
            w_q,
            w_k,
            w_v,
        }
    }

    /// Random initialization with deterministic seed.
    pub fn random(attention_dim: usize, key_dim: usize, value_dim: usize, seed: u64) -> Self {
        let mut rng = DetRng::new(seed);
        let sq = 1.0 / (attention_dim as f32).sqrt();
        let sk = 1.0 / (attention_dim as f32).sqrt();
        let sv = 1.0 / (attention_dim as f32).sqrt();
        let mut w_q = vec![0.0; attention_dim * key_dim];
        let mut w_k = vec![0.0; attention_dim * key_dim];
        let mut w_v = vec![0.0; attention_dim * value_dim];
        for w in &mut w_q {
            *w = rng.gauss() * sq;
        }
        for w in &mut w_k {
            *w = rng.gauss() * sk;
        }
        for w in &mut w_v {
            *w = rng.gauss() * sv;
        }
        Self {
            attention_dim,
            key_dim,
            value_dim,
            w_q,
            w_k,
            w_v,
        }
    }

    /// Project query: Q = q_a W_Q  →  Q ∈ R^{d_k}
    pub fn project_q(&self, q_a: &[f32]) -> Result<Vec<f32>> {
        if q_a.len() != self.attention_dim {
            return Err(crate::errors::AttentionError::DimensionMismatch {
                expected: self.attention_dim,
                found: q_a.len(),
            });
        }
        let mut q = vec![0.0; self.key_dim];
        for j in 0..self.key_dim {
            let mut sum = 0.0;
            for i in 0..self.attention_dim {
                sum += q_a[i] * self.w_q[i * self.key_dim + j];
            }
            q[j] = sum;
        }
        Ok(q)
    }

    /// Project keys: K_d = Z_d W_K  →  K_d ∈ R^{H × d_k}
    /// Z_d is H × d_a (stacked aligned head vectors).
    pub fn project_k(&self, z_d: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if z_d.is_empty() {
            return Err(crate::errors::AttentionError::EmptyInput(
                "z_d cannot be empty".into(),
            ));
        }
        let h_count = z_d.len();
        let mut k_d = Vec::with_capacity(h_count);
        for h in 0..h_count {
            let z_h = &z_d[h];
            if z_h.len() != self.attention_dim {
                return Err(crate::errors::AttentionError::DimensionMismatch {
                    expected: self.attention_dim,
                    found: z_h.len(),
                });
            }
            let mut k_h = vec![0.0; self.key_dim];
            for j in 0..self.key_dim {
                let mut sum = 0.0;
                for i in 0..self.attention_dim {
                    sum += z_h[i] * self.w_k[i * self.key_dim + j];
                }
                k_h[j] = sum;
            }
            k_d.push(k_h);
        }
        Ok(k_d)
    }

    /// Project values: V_d = Z_d W_V  →  V_d ∈ R^{H × d_v}
    pub fn project_v(&self, z_d: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if z_d.is_empty() {
            return Err(crate::errors::AttentionError::EmptyInput(
                "z_d cannot be empty".into(),
            ));
        }
        let h_count = z_d.len();
        let mut v_d = Vec::with_capacity(h_count);
        for h in 0..h_count {
            let z_h = &z_d[h];
            if z_h.len() != self.attention_dim {
                return Err(crate::errors::AttentionError::DimensionMismatch {
                    expected: self.attention_dim,
                    found: z_h.len(),
                });
            }
            let mut v_h = vec![0.0; self.value_dim];
            for j in 0..self.value_dim {
                let mut sum = 0.0;
                for i in 0..self.attention_dim {
                    sum += z_h[i] * self.w_v[i * self.value_dim + j];
                }
                v_h[j] = sum;
            }
            v_d.push(v_h);
        }
        Ok(v_d)
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len()
    }
}
