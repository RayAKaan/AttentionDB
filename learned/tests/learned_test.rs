use attentiondb_learned::{
    ContrastiveLoss, ProjectionConfig, ProjectionMatrix, ProjectionTrainer, ReprojectionJob,
};

#[test]
fn test_projection_creation() {
    let config = ProjectionConfig::default();
    let pm = ProjectionMatrix::new(config);
    assert_eq!(pm.w_q.shape(), &[256, 256]);
    assert_eq!(pm.w_k.shape(), &[256, 256]);
}

#[test]
fn test_project_key_output_len() {
    let config = ProjectionConfig {
        dim: 64,
        num_heads: 4,
        head_dim: 16,
    };
    let pm = ProjectionMatrix::new(config);
    let input = vec![0.1; 64];
    let output = pm.project_key(&input);
    assert_eq!(output.len(), 64);
}

#[test]
fn test_project_value_output_len() {
    let config = ProjectionConfig {
        dim: 64,
        num_heads: 4,
        head_dim: 16,
    };
    let pm = ProjectionMatrix::new(config);
    let input = vec![0.1; 64];
    let output = pm.project_value(&input);
    assert_eq!(output.len(), 64);
}

#[test]
fn test_training_step_returns_loss() {
    let mut trainer = ProjectionTrainer::new(8, 0.01);
    let pairs = vec![(vec![0.1; 8], vec![0.11; 8])];
    let negs = vec![vec![0.9; 8], vec![0.8; 8], vec![0.7; 8]];
    let loss = trainer.train_step(&pairs, &negs).unwrap();
    assert!(loss >= 0.0);
    assert!(loss < 10.0);
}

#[test]
fn test_contrastive_loss_positive_identical() {
    let loss_fn = ContrastiveLoss::new(0.07);
    let q = vec![0.5, 0.5, 0.0, 0.0];
    let p = vec![0.5, 0.5, 0.0, 0.0];
    let negs = vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]];
    let loss = loss_fn.compute(&q, &p, &negs);
    assert!(loss >= 0.0);
}

#[test]
fn test_reprojection_job_run() {
    let config = ProjectionConfig::default();
    let old = ProjectionMatrix::new(config.clone());
    let new = ProjectionMatrix::new(config);
    let job = ReprojectionJob::new("papers", old, new);
    assert!(job.run().is_ok());
}

#[test]
fn test_projection_update() {
    let config = ProjectionConfig {
        dim: 8,
        num_heads: 2,
        head_dim: 4,
    };
    let mut pm = ProjectionMatrix::new(config);
    let new_k = ndarray::Array2::from_shape_fn((8, 8), |_| 0.5);
    let new_v = ndarray::Array2::from_shape_fn((8, 8), |_| 0.5);
    pm.update(new_k, new_v);
    assert!((pm.w_k[[0, 0]] - 0.5).abs() < 1e-5);
}

#[test]
fn test_training_step_no_pairs() {
    let mut trainer = ProjectionTrainer::new(8, 0.01);
    let result = trainer.train_step(&[], &[vec![0.5; 8]]);
    assert!(result.is_err());
}

#[test]
fn test_training_step_no_negs() {
    let mut trainer = ProjectionTrainer::new(8, 0.01);
    let result = trainer.train_step(&[(vec![0.1; 8], vec![0.11; 8])], &[]);
    assert!(result.is_err());
}

#[test]
fn test_different_dims() {
    for dim in [16, 32, 64] {
        let config = ProjectionConfig {
            dim,
            num_heads: 2,
            head_dim: dim / 2,
        };
        let pm = ProjectionMatrix::new(config);
        let input = vec![0.1; dim];
        assert_eq!(pm.project_key(&input).len(), dim);
        assert_eq!(pm.project_value(&input).len(), dim);
    }
}
