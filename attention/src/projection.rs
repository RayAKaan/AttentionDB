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

    /// FNV-1a fingerprint over the geometry and all three matrices, used to
    /// prove that two projections are interchangeable (C8 cache validity).
    pub fn fingerprint(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        let mut push_u64 = |v: u64| {
            for b in v.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        };
        push_u64(self.attention_dim as u64);
        push_u64(self.key_dim as u64);
        push_u64(self.value_dim as u64);
        for w in self
            .w_q
            .iter()
            .chain(self.w_k.iter())
            .chain(self.w_v.iter())
        {
            for b in w.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash
    }
}

/// Deterministic truncated-identity base projection `P ∈ R^{d_a x d_reduced}`
/// (C8 decision Q1):
///
/// ```text
/// P[i][j] = 1 if i == j else 0,   for 0 <= i, j < d_reduced
/// P[i][j] = 0                     otherwise (rows beyond the reduced dimension)
/// ```
///
/// This is deliberately *not* a random or orthonormal projection. C8 exists to
/// isolate the value of a learned *residual correction*, and a random or QR base
/// would inject a second representation transformation whose influence could not
/// be separated from the attention correction. The truncated identity keeps the
/// base exactly as interpretable as the C7 identity diagnostic while making the
/// 384 -> 64 reduction well defined. It is never learned.
///
/// Requires `d_reduced <= d_a`.
pub fn truncated_identity(d_a: usize, d_reduced: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; d_a * d_reduced];
    for i in 0..d_reduced.min(d_a) {
        w[i * d_reduced + i] = 1.0;
    }
    w
}

/// C8 residual Q/K/V projection: `W = W_base + residual_alpha * W_delta`.
///
/// The base `W_base` is frozen (truncated identity by construction) and only
/// `W_delta` is learned. Two modes are supported, which is exactly the C8
/// E-vs-D contrast:
///
/// * `use_residual = true`  — arm E/F/G/H: `W = W0 + alpha * dW`, `dW` starts at
///   zero, so the model starts exactly at the deterministic base and learns only
///   the correction. At initialization the attention output therefore equals the
///   non-degenerate truncated-identity attention, not a degenerate zero state.
/// * `use_residual = false` — arm D (unrestricted learned): `W = dW` alone with
///   `W_base = 0`, i.e. a plain learned low-dimensional projection.
///
/// Because the base is frozen, the gradient with respect to the effective matrix
/// `W` transfers directly to `W_delta` via the chain rule
/// `dL/dW_delta = alpha * dL/dW` (and `= dL/dW` when `use_residual = false`).
/// [`Self::delta_grad_from_effective`] performs exactly that transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualQkvProjection {
    pub attention_dim: usize, // d_a
    pub key_dim: usize,       // d_k
    pub value_dim: usize,     // d_v
    /// `true` => `W = W_base + alpha * W_delta`; `false` => `W = W_delta`.
    pub use_residual: bool,
    /// Scale of the learned correction. Must be finite and >= 0.
    pub residual_alpha: f32,
    /// Frozen base `W_Q^0 ∈ R^{d_a x d_k}` (row-major).
    pub w_q_base: Vec<f32>,
    /// Frozen base `W_K^0 ∈ R^{d_a x d_k}` (row-major).
    pub w_k_base: Vec<f32>,
    /// Frozen base `W_V^0 ∈ R^{d_a x d_v}` (row-major).
    pub w_v_base: Vec<f32>,
    /// Learned correction `W_Q^Δ ∈ R^{d_a x d_k}` (row-major).
    pub w_q_delta: Vec<f32>,
    /// Learned correction `W_K^Δ ∈ R^{d_a x d_k}` (row-major).
    pub w_k_delta: Vec<f32>,
    /// Learned correction `W_V^Δ ∈ R^{d_a x d_v}` (row-major).
    pub w_v_delta: Vec<f32>,
}

impl ResidualQkvProjection {
    /// Create from explicit base and delta matrices. All lengths must match the
    /// declared geometry and `residual_alpha` must be finite and non-negative.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
        use_residual: bool,
        residual_alpha: f32,
        w_q_base: Vec<f32>,
        w_k_base: Vec<f32>,
        w_v_base: Vec<f32>,
        w_q_delta: Vec<f32>,
        w_k_delta: Vec<f32>,
        w_v_delta: Vec<f32>,
    ) -> Result<Self> {
        if attention_dim == 0 || key_dim == 0 || value_dim == 0 {
            return Err(crate::errors::AttentionError::Config(
                "residual qkv dimensions must be > 0".into(),
            ));
        }
        if key_dim > attention_dim || value_dim > attention_dim {
            return Err(crate::errors::AttentionError::Config(format!(
                "reduced dims must not exceed attention_dim ({attention_dim})"
            )));
        }
        if !residual_alpha.is_finite() || residual_alpha < 0.0 {
            return Err(crate::errors::AttentionError::Config(
                "residual_alpha must be finite and >= 0".into(),
            ));
        }
        let nq = attention_dim * key_dim;
        let nv = attention_dim * value_dim;
        for (name, got, want) in [
            ("w_q_base", w_q_base.len(), nq),
            ("w_k_base", w_k_base.len(), nq),
            ("w_q_delta", w_q_delta.len(), nq),
            ("w_k_delta", w_k_delta.len(), nq),
            ("w_v_base", w_v_base.len(), nv),
            ("w_v_delta", w_v_delta.len(), nv),
        ] {
            if got != want {
                return Err(crate::errors::AttentionError::InvalidMatrixDimensions(
                    format!("{name} len {got} != {want}"),
                ));
            }
        }
        for (name, m) in [
            ("w_q_base", &w_q_base),
            ("w_k_base", &w_k_base),
            ("w_v_base", &w_v_base),
            ("w_q_delta", &w_q_delta),
            ("w_k_delta", &w_k_delta),
            ("w_v_delta", &w_v_delta),
        ] {
            if let Some(bad) = m.iter().position(|w| !w.is_finite()) {
                return Err(crate::errors::AttentionError::NonFinite(format!(
                    "{name}[{bad}]"
                )));
            }
        }
        Ok(Self {
            attention_dim,
            key_dim,
            value_dim,
            use_residual,
            residual_alpha,
            w_q_base,
            w_k_base,
            w_v_base,
            w_q_delta,
            w_k_delta,
            w_v_delta,
        })
    }

    /// Arm C/E initialization: deterministic truncated-identity base with a
    /// zero correction. Attention at `delta = 0` is exactly the reduced
    /// truncated-identity attention — never a degenerate all-zero state.
    pub fn truncated_identity_residual(
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
        residual_alpha: f32,
    ) -> Result<Self> {
        Self::new(
            attention_dim,
            key_dim,
            value_dim,
            true,
            residual_alpha,
            truncated_identity(attention_dim, key_dim),
            truncated_identity(attention_dim, key_dim),
            truncated_identity(attention_dim, value_dim),
            vec![0.0; attention_dim * key_dim],
            vec![0.0; attention_dim * key_dim],
            vec![0.0; attention_dim * value_dim],
        )
    }

    /// Residual initialization with a small deterministic random correction.
    /// Used when an arm needs a non-degenerate starting point that is not the
    /// base itself.
    pub fn random_residual(
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
        residual_alpha: f32,
        seed: u64,
    ) -> Result<Self> {
        let mut p =
            Self::truncated_identity_residual(attention_dim, key_dim, value_dim, residual_alpha)?;
        let scale = 1.0 / (attention_dim as f32).sqrt();
        let mut rng = DetRng::new(seed);
        for w in p.w_q_delta.iter_mut() {
            *w = rng.gauss() * scale;
        }
        for w in p.w_k_delta.iter_mut() {
            *w = rng.gauss() * scale;
        }
        for w in p.w_v_delta.iter_mut() {
            *w = rng.gauss() * scale;
        }
        Ok(p)
    }

    /// Arm D initialization: unrestricted learned projection with no base, i.e.
    /// `W = W_delta` alone. Seeded deterministically — a zero start would make
    /// every attention logit identical and kill the gradient entirely.
    pub fn unrestricted(
        attention_dim: usize,
        key_dim: usize,
        value_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut p = Self::truncated_identity_residual(attention_dim, key_dim, value_dim, 0.0)?;
        p.use_residual = false;
        p.residual_alpha = 0.0;
        p.w_q_base = vec![0.0; attention_dim * key_dim];
        p.w_k_base = vec![0.0; attention_dim * key_dim];
        p.w_v_base = vec![0.0; attention_dim * value_dim];
        let scale = 1.0 / (attention_dim as f32).sqrt();
        let mut rng = DetRng::new(seed);
        for w in p.w_q_delta.iter_mut() {
            *w = rng.gauss() * scale;
        }
        for w in p.w_k_delta.iter_mut() {
            *w = rng.gauss() * scale;
        }
        for w in p.w_v_delta.iter_mut() {
            *w = rng.gauss() * scale;
        }
        Ok(p)
    }

    /// Wrap an existing plain (C7) projection as an unrestricted C8 projection.
    /// `W = W_delta`, so the materialized effective matrix is exactly `w_q/w_k/w_v`.
    pub fn from_plain(p: &QkvProjection) -> Result<Self> {
        Self::new(
            p.attention_dim,
            p.key_dim,
            p.value_dim,
            false,
            0.0,
            vec![0.0; p.attention_dim * p.key_dim],
            vec![0.0; p.attention_dim * p.key_dim],
            vec![0.0; p.attention_dim * p.value_dim],
            p.w_q.clone(),
            p.w_k.clone(),
            p.w_v.clone(),
        )
    }

    /// Materialize the effective matrices `W = W_base + alpha * W_delta`
    /// (or `W = W_delta` when `use_residual` is false) as a plain
    /// [`QkvProjection`], which every existing attention routine consumes.
    pub fn to_qkv(&self) -> QkvProjection {
        let combine = |base: &[f32], delta: &[f32]| -> Vec<f32> {
            if !self.use_residual {
                return delta.to_vec();
            }
            let a = self.residual_alpha;
            let mut out = Vec::with_capacity(base.len());
            for i in 0..base.len() {
                out.push(base[i] + a * delta[i]);
            }
            out
        };
        QkvProjection {
            attention_dim: self.attention_dim,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            w_q: combine(&self.w_q_base, &self.w_q_delta),
            w_k: combine(&self.w_k_base, &self.w_k_delta),
            w_v: combine(&self.w_v_base, &self.w_v_delta),
        }
    }

    /// Transfer a gradient with respect to the *effective* matrix to a gradient
    /// with respect to the learned correction.
    ///
    /// `dL/dW_delta = alpha * dL/dW` in residual mode, and `= dL/dW` when
    /// `use_residual` is false (unrestricted training has no base to add).
    ///
    /// Because Adam normalizes per-coordinate, a constant non-zero `alpha` is
    /// absorbed by the optimizer's magnitude normalization, so `alpha` governs
    /// the size of the effective correction rather than the effective step size.
    ///
    /// `alpha = 0` is a true degenerate case, not a pass-through: there
    /// `W = W_base` is *independent* of `W_delta`, so the derivative is exactly
    /// zero and returning the effective gradient unchanged would push the
    /// correction in a direction that provably cannot change the score.
    pub fn delta_grad_from_effective(&self, g_effective: &[f32]) -> Vec<f32> {
        if !self.use_residual {
            return g_effective.to_vec();
        }
        if self.residual_alpha == 0.0 {
            return vec![0.0; g_effective.len()];
        }
        let a = self.residual_alpha;
        g_effective.iter().map(|g| a * g).collect()
    }

    /// Squared L2 norm of the learned corrections only — the quantity the C8
    /// residual regularizer penalizes. The frozen base contributes nothing.
    pub fn delta_sq_norm(&self) -> f32 {
        let mut sum = 0.0f32;
        for m in [&self.w_q_delta, &self.w_k_delta, &self.w_v_delta] {
            for w in m {
                sum += w * w;
            }
        }
        sum
    }

    /// Number of *trainable* parameters (deltas only; the base is frozen).
    pub fn trainable_param_count(&self) -> usize {
        self.w_q_delta.len() + self.w_k_delta.len() + self.w_v_delta.len()
    }

    /// FNV-1a fingerprint over geometry, flags, and every stored matrix.
    /// Changes whenever anything that affects the effective `W` changes, so a
    /// stale model can never share a cache.
    pub fn fingerprint(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        let mut push_u64 = |v: u64| {
            for b in v.to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
        };
        push_u64(self.attention_dim as u64);
        push_u64(self.key_dim as u64);
        push_u64(self.value_dim as u64);
        push_u64(self.use_residual as u64);
        push_u64(self.residual_alpha.to_bits() as u64);
        for m in [
            &self.w_q_base,
            &self.w_k_base,
            &self.w_v_base,
            &self.w_q_delta,
            &self.w_k_delta,
            &self.w_v_delta,
        ] {
            for w in m {
                for b in w.to_le_bytes() {
                    hash ^= b as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_identity_is_first_reduced_basis_vectors() {
        let p = truncated_identity(384, 64);
        assert_eq!(p.len(), 384 * 64);
        // Unit diagonal on the first 64 coordinates...
        for i in 0..64 {
            assert_eq!(p[i * 64 + i], 1.0);
        }
        // ...and nothing anywhere else.
        let ones = p.iter().filter(|&&w| w != 0.0).count();
        assert_eq!(ones, 64);
        assert!(p.iter().all(|&w| w == 0.0 || w == 1.0));
    }

    #[test]
    fn truncated_identity_covers_a_short_projection() {
        let p = truncated_identity(2, 5);
        assert_eq!(p.len(), 10);
        assert_eq!(p[0], 1.0); // (0,0)
        assert_eq!(p[6], 1.0); // (1,1) with d_reduced = 5
        assert_eq!(p.iter().filter(|&&w| w != 0.0).count(), 2);
    }

    #[test]
    fn residual_starts_at_the_deterministic_base() {
        let p = ResidualQkvProjection::truncated_identity_residual(8, 4, 4, 0.1).unwrap();
        let q = p.to_qkv();
        assert_eq!(q.w_q, truncated_identity(8, 4));
        assert_eq!(q.w_k, truncated_identity(8, 4));
        assert_eq!(q.w_v, truncated_identity(8, 4));
        // Attention is non-degenerate at init: keys are non-zero.
        assert!(q.w_k.iter().any(|&w| w != 0.0));
    }

    #[test]
    fn zero_alpha_gives_a_zero_delta_gradient() {
        // `W = W_base + 0 * W_delta` is independent of `W_delta`, so the
        // chain-rule derivative is exactly zero. A pass-through here would
        // move the correction in a direction that cannot change any score.
        let p = ResidualQkvProjection::truncated_identity_residual(4, 2, 2, 0.0).unwrap();
        let g = p.delta_grad_from_effective(&[1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(g, vec![0.0; 8]);
    }

    #[test]
    fn nonzero_alpha_scales_the_delta_gradient() {
        let p = ResidualQkvProjection::truncated_identity_residual(4, 2, 2, 0.25).unwrap();
        let src = vec![1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let g = p.delta_grad_from_effective(&src);
        for (got, want) in g.iter().zip(src.iter()) {
            assert!((got - 0.25 * want).abs() < 1e-6);
        }
    }

    #[test]
    fn unrestricted_gradient_passes_through() {
        // With `use_residual = false` the effective matrix *is* the correction,
        // so the gradient transfers unchanged regardless of alpha.
        // `unrestricted` always sets `residual_alpha = 0`, so this case is
        // exactly the alpha == 0 pass-through branch.
        let p = ResidualQkvProjection::unrestricted(4, 2, 2, 7).unwrap();
        let src = vec![1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(p.delta_grad_from_effective(&src), src);
    }

    #[test]
    fn alpha_scales_the_correction_exactly() {
        let mut p = ResidualQkvProjection::truncated_identity_residual(4, 2, 2, 0.25).unwrap();
        p.w_q_delta = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let q = p.to_qkv();
        for i in 0..p.w_q_base.len() {
            assert!((q.w_q[i] - (p.w_q_base[i] + 0.25 * p.w_q_delta[i])).abs() < 1e-6);
        }
        p.residual_alpha = 1.0;
        let q = p.to_qkv();
        for i in 0..p.w_q_base.len() {
            assert!((q.w_q[i] - (p.w_q_base[i] + p.w_q_delta[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn alpha_zero_reproduces_the_base() {
        let mut p = ResidualQkvProjection::truncated_identity_residual(4, 2, 2, 0.0).unwrap();
        p.w_q_delta = vec![9.0; 8];
        p.w_k_delta = vec![9.0; 8];
        p.w_v_delta = vec![9.0; 8];
        let q = p.to_qkv();
        assert_eq!(q.w_q, truncated_identity(4, 2));
        assert_eq!(q.w_k, truncated_identity(4, 2));
        assert_eq!(q.w_v, truncated_identity(4, 2));
    }

    #[test]
    fn unrestricted_ignores_the_base_entirely() {
        let p = ResidualQkvProjection::unrestricted(4, 2, 2, 42).unwrap();
        assert!(!p.use_residual);
        let q = p.to_qkv();
        assert_eq!(q.w_q, p.w_q_delta);
        assert_eq!(q.w_k, p.w_k_delta);
        assert_eq!(q.w_v, p.w_v_delta);
        assert!(q.w_q.iter().any(|&w| w != 0.0), "must not start degenerate");
    }

    #[test]
    fn unrestricted_is_deterministic_for_a_seed() {
        let a = ResidualQkvProjection::unrestricted(8, 4, 4, 7).unwrap();
        let b = ResidualQkvProjection::unrestricted(8, 4, 4, 7).unwrap();
        assert_eq!(a.fingerprint(), b.fingerprint());
        let c = ResidualQkvProjection::unrestricted(8, 4, 4, 8).unwrap();
        assert_ne!(a.fingerprint(), c.fingerprint());
    }

    #[test]
    fn round_trip_from_plain_projection_is_exact() {
        let plain = QkvProjection::random(6, 3, 3, 11);
        let r = ResidualQkvProjection::from_plain(&plain).unwrap();
        let q = r.to_qkv();
        assert_eq!(q.w_q, plain.w_q);
        assert_eq!(q.w_k, plain.w_k);
        assert_eq!(q.w_v, plain.w_v);
    }

    #[test]
    fn gradient_transfer_applies_the_chain_rule() {
        let p = ResidualQkvProjection::truncated_identity_residual(4, 2, 2, 0.2).unwrap();
        let g = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let d = p.delta_grad_from_effective(&g);
        for i in 0..g.len() {
            assert!((d[i] - 0.2 * g[i]).abs() < 1e-6);
        }
        // Unrestricted mode has no base, so no chain-rule factor.
        let mut u = ResidualQkvProjection::unrestricted(4, 2, 2, 3).unwrap();
        u.residual_alpha = 0.5; // must still be ignored
        assert_eq!(u.delta_grad_from_effective(&g), g);
    }

    #[test]
    fn regularizer_penalizes_only_the_delta() {
        let mut p = ResidualQkvProjection::truncated_identity_residual(4, 2, 2, 1.0).unwrap();
        assert!(p.delta_sq_norm().abs() < 1e-9);
        let mut base_norm = 0.0f32;
        for w in &p.w_q_base {
            base_norm += w * w;
        }
        assert!(
            base_norm > 0.0,
            "base is non-trivial but must not be penalized"
        );
        p.w_q_delta = vec![3.0; 8];
        assert!((p.delta_sq_norm() - 72.0).abs() < 1e-4);
        assert_eq!(p.trainable_param_count(), 8 + 8 + 8);
    }

    #[test]
    fn rejects_bad_geometry_and_alpha() {
        assert!(ResidualQkvProjection::new(
            0,
            2,
            2,
            true,
            1.0,
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![]
        )
        .is_err());
        assert!(ResidualQkvProjection::truncated_identity_residual(4, 8, 8, 1.0).is_err());
        assert!(ResidualQkvProjection::truncated_identity_residual(4, 2, 2, -1.0).is_err());
        assert!(ResidualQkvProjection::truncated_identity_residual(4, 2, 2, f32::NAN).is_err());
        assert!(ResidualQkvProjection::new(
            4,
            2,
            2,
            true,
            1.0,
            vec![0.0; 7],
            vec![0.0; 8],
            vec![0.0; 8],
            vec![0.0; 8],
            vec![0.0; 8],
            vec![0.0; 8]
        )
        .is_err());
        assert!(ResidualQkvProjection::new(
            4,
            2,
            2,
            true,
            1.0,
            vec![f32::NAN; 8],
            vec![0.0; 8],
            vec![0.0; 8],
            vec![0.0; 8],
            vec![0.0; 8],
            vec![0.0; 8]
        )
        .is_err());
    }

    #[test]
    fn fingerprint_detects_every_material_change() {
        let base = ResidualQkvProjection::truncated_identity_residual(4, 2, 2, 0.1).unwrap();
        let mut alpha = base.clone();
        alpha.residual_alpha = 0.2;
        let mut delta = base.clone();
        delta.w_q_delta[0] = 0.5;
        let mut flag = base.clone();
        flag.use_residual = false;
        let f = base.fingerprint();
        assert_ne!(f, alpha.fingerprint());
        assert_ne!(f, delta.fingerprint());
        assert_ne!(f, flag.fingerprint());
        assert_eq!(f, base.clone().fingerprint());
    }
}
