use ndarray::Array1;

pub struct ContrastiveLoss {
    pub temperature: f32,
}

impl ContrastiveLoss {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    /// InfoNCE-style contrastive loss
    /// query: query vector, positive: matching key, negatives: non-matching keys
    pub fn compute(&self, query: &[f32], positive: &[f32], negatives: &[Vec<f32>]) -> f32 {
        let q = Array1::from(query.to_vec());

        let pos_sim = q.dot(&Array1::from(positive.to_vec())) / self.temperature;

        let mut all_logits: Vec<f32> = negatives
            .iter()
            .map(|neg| q.dot(&Array1::from(neg.clone())) / self.temperature)
            .collect();

        all_logits.push(pos_sim);

        let max_val = all_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = all_logits.iter().map(|x| (x - max_val).exp()).sum();
        let pos_exp = (pos_sim - max_val).exp();

        -(pos_exp / exp_sum).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_non_negative() {
        let loss = ContrastiveLoss::new(0.07);
        let q = vec![0.1; 8];
        let p = vec![0.12; 8];
        let negs = vec![vec![0.5; 8], vec![0.9; 8]];
        let l = loss.compute(&q, &p, &negs);
        assert!(l >= 0.0);
    }

    #[test]
    fn test_loss_improves_with_better_positive() {
        let loss = ContrastiveLoss::new(0.07);
        let q = vec![1.0, 0.0, 0.0];
        let p_good = vec![0.95, 0.05, 0.0];
        let p_bad = vec![0.0, 0.0, 1.0];
        let negs = vec![vec![0.0, 1.0, 0.0]];

        let loss_good = loss.compute(&q, &p_good, &negs);
        let loss_bad = loss.compute(&q, &p_bad, &negs);
        assert!(loss_good < loss_bad);
    }
}
