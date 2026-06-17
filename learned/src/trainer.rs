use crate::projection::{ProjectionMatrix, ProjectionConfig};
use crate::contrastive::ContrastiveLoss;
use crate::error::LearnedError;
use ndarray::{Array1, Array2};

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
        let dim = self.projection.config.dim;

        let mut batch_grad_w_q = Array2::<f32>::zeros((dim, dim));
        let mut batch_grad_w_k = Array2::<f32>::zeros((dim, dim));

        let tau = self.loss_fn.temperature;

        for (raw_q, raw_pos) in positive_pairs {
            let q = self.projection.project_query(raw_q);
            let pos = self.projection.project_key(raw_pos);
            let negs: Vec<Vec<f32>> = negatives.iter()
                .map(|neg| self.projection.project_key(neg))
                .collect();

            let loss = self.loss_fn.compute(&q, &pos, &negs);
            total_loss += loss;

            let q_arr = Array1::from(q);
            let pos_arr = Array1::from(pos);
            let neg_arrs: Vec<Array1<f32>> = negs.into_iter()
                .map(Array1::from)
                .collect();

            let s_pos = q_arr.dot(&pos_arr) / tau;
            let s_negs: Vec<f32> = neg_arrs.iter()
                .map(|n| q_arr.dot(n) / tau)
                .collect();

            let mut all_s = s_negs.clone();
            all_s.push(s_pos);

            let max_s = all_s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_pos = (s_pos - max_s).exp();
            let exp_negs: Vec<f32> = s_negs.iter().map(|&s| (s - max_s).exp()).collect();
            let exp_sum = exp_pos + exp_negs.iter().sum::<f32>();

            let p_pos = exp_pos / exp_sum;
            let p_negs: Vec<f32> = exp_negs.iter().map(|e| e / exp_sum).collect();

            let dl_dpos = &q_arr * ((p_pos - 1.0) / tau);
            let mut dl_dq = &pos_arr * ((p_pos - 1.0) / tau);

            let mut grad_w_k_sample = Array2::<f32>::zeros((dim, dim));
            let raw_pos_arr = Array1::from(raw_pos.clone());
            for i in 0..dim {
                for j in 0..dim {
                    grad_w_k_sample[[i, j]] += dl_dpos[i] * raw_pos_arr[j];
                }
            }

            for (neg_idx, neg_arr) in neg_arrs.iter().enumerate() {
                let p_neg = p_negs[neg_idx];
                let dl_dneg = &q_arr * (p_neg / tau);
                dl_dq = dl_dq + (neg_arr * (p_neg / tau));

                let raw_neg_arr = Array1::from(negatives[neg_idx].clone());
                for i in 0..dim {
                    for j in 0..dim {
                        grad_w_k_sample[[i, j]] += dl_dneg[i] * raw_neg_arr[j];
                    }
                }
            }

            let raw_q_arr = Array1::from(raw_q.clone());
            let mut grad_w_q_sample = Array2::<f32>::zeros((dim, dim));
            for i in 0..dim {
                for j in 0..dim {
                    grad_w_q_sample[[i, j]] += dl_dq[i] * raw_q_arr[j];
                }
            }

            batch_grad_w_q = batch_grad_w_q + grad_w_q_sample;
            batch_grad_w_k = batch_grad_w_k + grad_w_k_sample;
        }

        let num_pairs = positive_pairs.len() as f32;
        batch_grad_w_q = batch_grad_w_q / num_pairs;
        batch_grad_w_k = batch_grad_w_k / num_pairs;

        self.projection.w_q = &self.projection.w_q - &(&batch_grad_w_q * self.learning_rate);
        self.projection.w_k = &self.projection.w_k - &(&batch_grad_w_k * self.learning_rate);
        self.projection.w_v = &self.projection.w_v - &(&batch_grad_w_k * self.learning_rate);

        Ok(total_loss / num_pairs)
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
        let mut trainer = ProjectionTrainer::new(4, 0.1);
        let pairs = vec![
            (vec![1.0, 0.0, 0.0, 0.0], vec![0.9, 0.1, 0.0, 0.0]),
            (vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.9, 0.1, 0.0]),
        ];
        let negs = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let l1 = trainer.train_step(&pairs, &negs).unwrap();
        let l2 = trainer.train_step(&pairs, &negs).unwrap();

        assert!(l1 >= 0.0);
        assert!(l2 >= 0.0);
        assert!(l2 < l1);
    }

    #[test]
    fn test_train_step_empty_pairs() {
        let mut trainer = ProjectionTrainer::new(4, 0.01);
        let result = trainer.train_step(&[], &[vec![0.5; 4]]);
        assert!(result.is_err());
    }
}
