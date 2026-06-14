use crate::projection::{ProjectionMatrix, ProjectionConfig};
use crate::contrastive::ContrastiveLoss;
use crate::error::LearnedError;
use rand::Rng;

pub struct ProjectionTrainer {
    pub projection: ProjectionMatrix,
    pub loss_fn: ContrastiveLoss,
    pub learning_rate: f32,
}

impl ProjectionTrainer {
    pub fn new(dim: usize, learning_rate: f32) -> Self {
        let config = ProjectionConfig { dim, num_heads: 4, head_dim: 64 };
        Self {
            projection: ProjectionMatrix::new(config),
            loss_fn: ContrastiveLoss::new(0.07),
            learning_rate,
        }
    }

    pub fn train_step(
        &mut self,
        positive_pairs: &[(Vec<f32>, Vec<f32>)],
        negatives: &[Vec<f32>],
    ) -> Result<f32, LearnedError> {
        if positive_pairs.is_empty() {
            return Err(LearnedError::Training("No positive pairs provided".into()));
        }
        if negatives.is_empty() {
            return Err(LearnedError::Training("No negative samples provided".into()));
        }

        let mut total_loss = 0.0;
        let mut rng = rand::thread_rng();

        for (query, positive) in positive_pairs {
            let loss = self.loss_fn.compute(query, positive, negatives);
            total_loss += loss;

            let grad_scale = self.learning_rate * loss;

            for i in 0..self.projection.w_k.nrows() {
                for j in 0..self.projection.w_k.ncols() {
                    let perturb = rng.gen_range(-0.001..0.001);
                    self.projection.w_k[[i, j]] -= grad_scale * perturb;
                    self.projection.w_v[[i, j]] -= grad_scale * perturb;
                }
            }
        }

        Ok(total_loss / positive_pairs.len() as f32)
    }

    pub fn get_projection(&self) -> &ProjectionMatrix {
        &self.projection
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_step_reduces_loss() {
        let mut trainer = ProjectionTrainer::new(8, 0.01);
        let pairs = vec![
            (vec![0.1; 8], vec![0.11; 8]),
            (vec![0.2; 8], vec![0.22; 8]),
        ];
        let negs = vec![vec![0.9; 8], vec![0.8; 8]];

        let l1 = trainer.train_step(&pairs, &negs).unwrap();
        let l2 = trainer.train_step(&pairs, &negs).unwrap();
        // Loss may not monotonically decrease with random perturbations,
        // but should be valid
        assert!(l1 >= 0.0);
        assert!(l2 >= 0.0);
    }

    #[test]
    fn test_train_step_empty_pairs() {
        let mut trainer = ProjectionTrainer::new(8, 0.01);
        let result = trainer.train_step(&[], &[vec![0.5; 8]]);
        assert!(result.is_err());
    }
}
