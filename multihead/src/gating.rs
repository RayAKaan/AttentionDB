use ndarray::{Array1, Array2};
use rand::Rng;

pub struct GatingNetwork {
    pub input_dim: usize,
    pub num_heads: usize,
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl GatingNetwork {
    pub fn new(input_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = Array2::from_shape_fn((num_heads, input_dim), |_| rng.gen_range(-0.1..0.1));
        let bias = Array1::from_shape_fn(num_heads, |_| rng.gen_range(-0.1..0.1));

        Self { input_dim, num_heads, weights, bias }
    }

    pub fn forward(&self, query_embedding: &[f32]) -> Vec<f32> {
        let input = Array1::from(query_embedding.to_vec());

        let mut logits = self.bias.clone();
        for (i, row) in self.weights.outer_iter().enumerate() {
            logits[i] += input.dot(&row);
        }

        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp: Array1<f32> = logits.mapv(|x| (x - max_logit).exp());
        let sum: f32 = exp.sum();

        exp.mapv(|x| x / sum).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_produces_softmax() {
        let gate = GatingNetwork::new(8, 3);
        let input = vec![0.1; 8];
        let output = gate.forward(&input);
        assert_eq!(output.len(), 3);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_all_weights_positive() {
        let gate = GatingNetwork::new(8, 4);
        let input = vec![0.2; 8];
        let output = gate.forward(&input);
        assert!(output.iter().all(|w| *w > 0.0));
    }
}
