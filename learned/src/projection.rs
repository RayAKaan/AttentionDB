use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self { dim: 256, num_heads: 4, head_dim: 64 }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectionMatrix {
    pub config: ProjectionConfig,
    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,
}

impl ProjectionMatrix {
    pub fn new(config: ProjectionConfig) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (config.dim as f32).sqrt();

        let w_q = Array2::from_shape_fn((config.dim, config.dim), |_| {
            rng.gen::<f32>() * scale - scale / 2.0
        });
        let w_k = Array2::from_shape_fn((config.dim, config.dim), |_| {
            rng.gen::<f32>() * scale - scale / 2.0
        });
        let w_v = Array2::from_shape_fn((config.dim, config.dim), |_| {
            rng.gen::<f32>() * scale - scale / 2.0
        });

        Self { config, w_q, w_k, w_v }
    }

    pub fn project_query(&self, x: &[f32]) -> Vec<f32> {
        let input = Array1::from(x.to_vec());
        self.w_q.dot(&input).to_vec()
    }

    pub fn project_key(&self, x: &[f32]) -> Vec<f32> {
        let input = Array1::from(x.to_vec());
        self.w_k.dot(&input).to_vec()
    }

    pub fn project_value(&self, x: &[f32]) -> Vec<f32> {
        let input = Array1::from(x.to_vec());
        self.w_v.dot(&input).to_vec()
    }

    pub fn update(&mut self, new_w_k: Array2<f32>, new_w_v: Array2<f32>) {
        self.w_k = new_w_k;
        self.w_v = new_w_v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_shapes() {
        let config = ProjectionConfig { dim: 8, num_heads: 2, head_dim: 4 };
        let pm = ProjectionMatrix::new(config);
        assert_eq!(pm.w_q.shape(), &[8, 8]);
        assert_eq!(pm.w_k.shape(), &[8, 8]);
    }

    #[test]
    fn test_project_query_output_len() {
        let config = ProjectionConfig { dim: 16, num_heads: 2, head_dim: 8 };
        let pm = ProjectionMatrix::new(config);
        let input = vec![0.1; 16];
        let output = pm.project_query(&input);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_update_changes_matrices() {
        let config = ProjectionConfig { dim: 4, num_heads: 1, head_dim: 4 };
        let mut pm = ProjectionMatrix::new(config);
        let old_k = pm.w_k.clone();
        let new_k = Array2::from_shape_fn((4, 4), |_| 0.5);
        let new_v = Array2::from_shape_fn((4, 4), |_| 0.5);
        pm.update(new_k, new_v);
        assert!(pm.w_k[[0, 0]] != old_k[[0, 0]]);
    }
}
