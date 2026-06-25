use ndarray::{Array1, Array2};
use rand::Rng;
use tracing::{info, warn, debug};

/// A learned gating network that learns to weight heads via online cross-entropy training.
pub struct GatingNetwork {
    pub input_dim: usize,
    pub num_heads: usize,
    weights: Array2<f32>,
    bias: Array1<f32>,
    pub learning_rate: f32,
    optimizer: GatingOptimizer,
}

/// Simple SGD with momentum and weight decay for the gating network.
pub struct GatingOptimizer {
    pub momentum: f32,
    pub weight_decay: f32,
    velocity_w: Array2<f32>,
    velocity_b: Array1<f32>,
}

impl GatingOptimizer {
    pub fn new(input_dim: usize, num_heads: usize, momentum: f32, weight_decay: f32) -> Self {
        Self {
            momentum,
            weight_decay,
            velocity_w: Array2::zeros((num_heads, input_dim)),
            velocity_b: Array1::zeros(num_heads),
        }
    }
}

/// A single training sample for the gating network.
#[derive(Debug, Clone)]
pub struct GatingTrainingBatch {
    pub query_embedding: Vec<f32>,
    /// Target head weights (e.g., derived from user click-through or downstream relevance).
    pub target_weights: Vec<f32>,
}

/// Feedback signal for online learning.
#[derive(Debug, Clone)]
pub struct GatingFeedback {
    pub query_embedding: Vec<f32>,
    /// Positive document IDs (e.g., clicked/relevant results).
    pub positive_ids: Vec<u64>,
    /// Negative document IDs (e.g., skipped/irrelevant results).
    pub negative_ids: Vec<u64>,
    /// Per-head scores used to compute target weights.
    pub head_scores: Vec<(String, Vec<(u64, f32)>)>,
}

impl GatingNetwork {
    pub fn new(input_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((num_heads, input_dim), |_| rng.gen_range(-0.1..0.1));
        let bias = Array1::from_shape_fn(num_heads, |_| rng.gen_range(-0.1..0.1));
        let optimizer = GatingOptimizer::new(input_dim, num_heads, 0.9, 1e-4);
        Self { input_dim, num_heads, weights, bias, learning_rate: 0.01, optimizer }
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
        // SGD with momentum and weight decay
        for (i, mut row) in self.weights.outer_iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                let g = grad[i] * input[j] + self.optimizer.weight_decay * *w;
                let v = &mut self.optimizer.velocity_w[[i, j]];
                *v = self.optimizer.momentum * *v - self.learning_rate * g;
                *w += *v;
            }
        }
        for (i, b) in self.bias.iter_mut().enumerate() {
            let g = grad[i] + self.optimizer.weight_decay * *b;
            let v = &mut self.optimizer.velocity_b[i];
            *v = self.optimizer.momentum * *v - self.learning_rate * g;
            *b += *v;
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

    /// Train on a batch of samples with early stopping.
    pub fn train_online(
        &mut self,
        batches: &[GatingTrainingBatch],
        epochs: usize,
        patience: usize,
    ) -> f32 {
        if batches.is_empty() {
            warn!("[GatingNetwork] train_online called with empty batch set");
            return 0.0;
        }
        let mut best_loss = f32::INFINITY;
        let mut stall = 0;
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            for batch in batches {
                let loss = self.train_step(&batch.query_embedding, &batch.target_weights);
                epoch_loss += loss;
            }
            epoch_loss /= batches.len() as f32;
            if epoch_loss < best_loss - 1e-6 {
                best_loss = epoch_loss;
                stall = 0;
            } else {
                stall += 1;
            }
            if epoch % 10 == 0 || epoch == epochs - 1 {
                debug!("[GatingNetwork] epoch {} avg_loss={:.6}", epoch, epoch_loss);
            }
            if stall >= patience {
                info!("[GatingNetwork] Early stopping at epoch {} (loss={:.6})", epoch, epoch_loss);
                break;
            }
        }
        best_loss
    }

    /// Convert feedback signals into training batches and train.
    pub fn train_from_feedback(&mut self, feedback: &[GatingFeedback]) -> f32 {
        let batches: Vec<GatingTrainingBatch> = feedback.iter().map(|fb| {
            let target_weights = self.compute_target_weights(fb);
            GatingTrainingBatch {
                query_embedding: fb.query_embedding.clone(),
                target_weights,
            }
        }).collect();
        self.train_online(&batches, 50, 5)
    }

    fn compute_target_weights(&self, fb: &GatingFeedback) -> Vec<f32> {
        let num_heads = fb.head_scores.len();
        if num_heads == 0 {
            return vec![1.0 / self.num_heads as f32; self.num_heads];
        }
        let mut head_relevance = vec![0.0f32; num_heads];
        for (h, (_, results)) in fb.head_scores.iter().enumerate() {
            let mut pos_score = 0.0f32;
            let mut neg_score = 0.0f32;
            let mut pos_count = 0;
            let mut neg_count = 0;
            for (id, score) in results {
                if fb.positive_ids.contains(id) {
                    pos_score += score;
                    pos_count += 1;
                } else if fb.negative_ids.contains(id) {
                    neg_score += score;
                    neg_count += 1;
                }
            }
            let avg_pos = if pos_count > 0 { pos_score / pos_count as f32 } else { 0.0 };
            let avg_neg = if neg_count > 0 { neg_score / neg_count as f32 } else { 0.0 };
            head_relevance[h] = (avg_pos - avg_neg).max(0.0);
        }
        let sum: f32 = head_relevance.iter().sum();
        if sum > 0.0 {
            head_relevance.iter().map(|r| r / sum).collect()
        } else {
            vec![1.0 / num_heads as f32; num_heads]
        }
    }

    /// Persist weights to a binary file.
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let (w, b) = self.save_weights();
        let data = (w, b);
        let encoded = bincode::serialize(&data)?;
        std::fs::write(path, encoded)?;
        info!("[GatingNetwork] Saved weights to '{}'", path);
        Ok(())
    }

    /// Load weights from a binary file.
    pub fn load_from_file(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = std::fs::read(path)?;
        let (w, b): (Vec<f32>, Vec<f32>) = bincode::deserialize(&encoded)?;
        self.load_weights(&w, &b);
        info!("[GatingNetwork] Loaded weights from '{}'", path);
        Ok(())
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

    #[test]
    fn test_online_training_converges() {
        let mut gate = GatingNetwork::new(8, 2);
        let batches = vec![
            GatingTrainingBatch { query_embedding: vec![0.2; 8], target_weights: vec![0.9, 0.1] },
            GatingTrainingBatch { query_embedding: vec![0.8; 8], target_weights: vec![0.1, 0.9] },
        ];
        let loss = gate.train_online(&batches, 100, 10);
        assert!(loss < 0.5);
    }

    #[test]
    fn test_save_to_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gating_weights.bin");
        let path_str = path.to_str().unwrap();

        let mut gate = GatingNetwork::new(4, 2);
        gate.train_step(&[0.5; 4], &[0.7, 0.3]);
        gate.save_to_file(path_str).unwrap();

        let mut restored = GatingNetwork::new(4, 2);
        restored.load_from_file(path_str).unwrap();

        let (w_orig, b_orig) = gate.save_weights();
        let (w_rest, b_rest) = restored.save_weights();
        assert_eq!(w_orig, w_rest);
        assert_eq!(b_orig, b_rest);
    }
}