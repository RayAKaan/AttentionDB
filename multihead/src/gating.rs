use ndarray::{Array1, Array2};
use rand::Rng;

pub struct GatingNetwork {
    pub input_dim: usize,
    pub num_heads: usize,
    weights: Array2<f32>,
    bias: Array1<f32>,
    pub learning_rate: f32,
}

impl GatingNetwork {
    pub fn new(input_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = Array2::from_shape_fn((num_heads, input_dim), |_| rng.gen_range(-0.1..0.1));
        let bias = Array1::from_shape_fn(num_heads, |_| rng.gen_range(-0.1..0.1));

        Self { input_dim, num_heads, weights, bias, learning_rate: 0.01 }
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
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

    /// Single SGD step on cross-entropy loss between predicted gating
    /// weights and target weights (e.g., from user feedback).
    /// Returns the loss value before the update.
    pub fn train_step(&mut self, query_embedding: &[f32], target_weights: &[f32]) -> f32 {
        let input = Array1::from(query_embedding.to_vec());
        let target = Array1::from(target_weights.to_vec());

        let mut logits = self.bias.clone();
        for (i, row) in self.weights.outer_iter().enumerate() {
            logits[i] += input.dot(&row);
        }

        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp: Array1<f32> = logits.mapv(|x| (x - max_logit).exp());
        let sum: f32 = exp.sum();
        let softmax: Array1<f32> = exp.mapv(|x| x / sum);

        let loss = -target.iter()
            .zip(softmax.iter())
            .map(|(t, s)| if *t > 0.0 { t * s.ln() } else { 0.0 })
            .sum::<f32>();

        let grad: Array1<f32> = &softmax - &target;

        for (i, mut row) in self.weights.outer_iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                *w -= self.learning_rate * grad[i] * input[j];
            }
        }
        for (i, b) in self.bias.iter_mut().enumerate() {
            *b -= self.learning_rate * grad[i];
        }

        loss
    }

    pub fn save_weights(&self) -> (Vec<f32>, Vec<f32>) {
        (self.weights.iter().copied().collect(), self.bias.iter().copied().collect())
    }

    pub fn load_weights(&mut self, flat_weights: &[f32], flat_bias: &[f32]) {
        let expected_w = self.num_heads * self.input_dim;
        if flat_weights.len() == expected_w {
            if let Some(slice) = self.weights.as_slice_mut() {
                for (i, &v) in flat_weights.iter().enumerate() {
                    slice[i] = v;
                }
            }
        }
        if flat_bias.len() == self.num_heads {
            for (i, &v) in flat_bias.iter().enumerate() {
                self.bias[i] = v;
            }
        }
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

    #[test]
    fn test_train_step_reduces_loss() {
        let mut gate = GatingNetwork::new(8, 2).with_learning_rate(0.1);
        let input = vec![0.5; 8];
        let target = vec![0.8, 0.2];
        let l1 = gate.train_step(&input, &target);
        let l2 = gate.train_step(&input, &target);
        assert!(l1 >= 0.0);
        assert!(l2 < l1 || (l2 - l1).abs() < 1e-6);
    }

    #[test]
    fn test_save_load_weights() {
        let mut gate = GatingNetwork::new(4, 2);
        let (w_orig, b_orig) = gate.save_weights();
        let input = vec![0.3; 4];
        let target = vec![0.6, 0.4];
        gate.train_step(&input, &target);
        gate.load_weights(&w_orig, &b_orig);
        let (w_restored, b_restored) = gate.save_weights();
        assert_eq!(w_orig, w_restored);
        assert_eq!(b_orig, b_restored);
    }
}
