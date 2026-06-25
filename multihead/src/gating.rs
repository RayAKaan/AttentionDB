use ndarray::{Array1, Array2};
use rand::Rng;
use std::fs;
use std::path::Path;

/// Trained gating network with online SGD, momentum, early stopping, and weight persistence.
pub struct GatingNetwork {
    pub input_dim: usize,
    pub num_heads: usize,
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub optimizer: GatingOptimizer,
}

/// Momentum-based SGD optimizer for the gating network.
pub struct GatingOptimizer {
    pub learning_rate: f32,
    pub momentum: f32,
    pub velocity_w: Array2<f32>,
    pub velocity_b: Array1<f32>,
}

impl GatingOptimizer {
    pub fn new(input_dim: usize, num_heads: usize, learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity_w: Array2::zeros((num_heads, input_dim)),
            velocity_b: Array1::zeros(num_heads),
        }
    }
}

/// Training orchestrator with early stopping.
pub struct GatingTrainer {
    pub patience: usize,
    pub min_delta: f32,
}

impl GatingTrainer {
    pub fn new() -> Self {
        Self { patience: 5, min_delta: 0.001 }
    }

    /// Train the network on a batch of (query, target) pairs.
    /// Returns the best loss achieved.
    pub fn train_online(
        &self,
        net: &mut GatingNetwork,
        data: &[(Vec<f32>, Vec<f32>)],
        max_epochs: usize,
    ) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let mut best_loss = f32::INFINITY;
        let mut stall = 0;
        for epoch in 0..max_epochs {
            let mut epoch_loss = 0.0f32;
            for (query, target) in data {
                let loss = net.train_step(query, target);
                epoch_loss += loss;
            }
            epoch_loss /= data.len() as f32;
            if best_loss - epoch_loss > self.min_delta {
                best_loss = epoch_loss;
                stall = 0;
            } else {
                stall += 1;
            }
            if stall >= self.patience {
                break;
            }
        }
        best_loss
    }
}

impl GatingNetwork {
    pub fn new(input_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((num_heads, input_dim), |_| rng.gen_range(-0.1..0.1));
        let bias = Array1::from_shape_fn(num_heads, |_| rng.gen_range(-0.1..0.1));
        let optimizer = GatingOptimizer::new(input_dim, num_heads, 0.01, 0.9);
        Self { input_dim, num_heads, weights, bias, optimizer }
    }

    /// Softmax forward pass.
    pub fn forward(&self, query_embedding: &[f32]) -> Vec<f32> {
        let input = Array1::from(query_embedding.to_vec());
        let logits = self.weights.dot(&input) + &self.bias;
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp = logits.mapv(|x| (x - max_logit).exp());
        let sum = exp.sum();
        exp.mapv(|x| x / sum).to_vec()
    }

    /// Single SGD step with cross-entropy loss and momentum.
    pub fn train_step(&mut self, query_embedding: &[f32], target_weights: &[f32]) -> f32 {
        let input = Array1::from(query_embedding.to_vec());
        let target = Array1::from(target_weights.to_vec());
        let logits = self.weights.dot(&input) + &self.bias;
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp = logits.mapv(|x| (x - max_logit).exp());
        let sum = exp.sum();
        let softmax = exp.mapv(|x| x / sum);
        let loss = -target.iter()
            .zip(softmax.iter())
            .map(|(t, s)| if *t > 0.0 { t * s.ln() } else { 0.0 })
            .sum::<f32>();
        let grad = &softmax - &target;
        for (i, mut row) in self.weights.outer_iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                let g = grad[i] * input[j];
                let v = &mut self.optimizer.velocity_w[[i, j]];
                *v = self.optimizer.momentum * *v - self.optimizer.learning_rate * g;
                *w += *v;
                *w = w.clamp(-10.0, 10.0);
            }
        }
        for (i, b) in self.bias.iter_mut().enumerate() {
            let g = grad[i];
            let v = &mut self.optimizer.velocity_b[i];
            *v = self.optimizer.momentum * *v - self.optimizer.learning_rate * g;
            *b += *v;
            *b = b.clamp(-10.0, 10.0);
        }
        loss
    }

    pub fn save_weights(&self) -> (Vec<f32>, Vec<f32>) {
        (self.weights.iter().copied().collect(), self.bias.iter().copied().collect())
    }

    pub fn load_weights(&mut self, w: &[f32], b: &[f32]) {
        if w.len() == self.num_heads * self.input_dim {
            if let Some(slice) = self.weights.as_slice_mut() {
                slice.copy_from_slice(w);
            }
        }
        if b.len() == self.num_heads {
            self.bias.assign(&Array1::from(b.to_vec()));
        }
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let (w, b) = self.save_weights();
        let encoded = bincode::serialize(&(w, b))?;
        fs::write(path, encoded)?;
        Ok(())
    }

    pub fn load_from_file(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = fs::read(path)?;
        let (w, b): (Vec<f32>, Vec<f32>) = bincode::deserialize(&encoded)?;
        self.load_weights(&w, &b);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_produces_softmax() {
        let gate = GatingNetwork::new(8, 3);
        let output = gate.forward(&vec![0.1; 8]);
        assert_eq!(output.len(), 3);
        assert!((output.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_train_step_reduces_loss() {
        let mut gate = GatingNetwork::new(8, 2);
        gate.optimizer.learning_rate = 0.1;
        let data = vec![(vec![0.5; 8], vec![0.8, 0.2])];
        let trainer = GatingTrainer::new();
        let loss = trainer.train_online(&mut gate, &data, 20);
        assert!(loss < 0.5);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gate.bin");
        let mut g = GatingNetwork::new(4, 2);
        g.train_step(&[0.5; 4], &[0.7, 0.3]);
        g.save_to_file(path.to_str().unwrap()).unwrap();
        let mut g2 = GatingNetwork::new(4, 2);
        g2.load_from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(g.weights, g2.weights);
    }
}