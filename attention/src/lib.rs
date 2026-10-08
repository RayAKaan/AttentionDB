//! C7 Genuine Candidate-Level Q/K/V Attention Subsystem.
//!
//! This crate implements the core mathematical operations for candidate-level
//! attention in AttentionDB, following the C7 specification.
#![allow(clippy::needless_range_loop)]
// Explicit index loops are used throughout for mathematical clarity and
// bit-exact determinism (identical loop order everywhere → identical f32
// accumulation order).

pub mod alignment;
pub mod attention;
pub mod c8_training;
pub mod c9_benchmark;
pub mod cache;
pub mod config;
pub mod diagnostics;
pub mod distillation;
pub mod errors;
pub mod model;
pub mod negative_mining;
pub mod projection;
pub mod qkv;
pub mod scorer;
pub mod training;

pub use alignment::{AlignmentProjection, DetRng};
pub use attention::{
    AttentionOutput, AttentionSubsystem, C8AttentionOutput, C8AttentionSubsystem,
    C8AttentionTimings, C8CandidateScore,
};
pub use c8_training::{
    C8LossReport, C8QkvDataset, C8QkvDatasetBuilder, C8QkvExample, C8TrainedModel,
    C8TrainingConfig, ResidualQkvTrainer,
};
pub use c9_benchmark::{benchmark_scalar_vs_batch, C9BenchmarkReport};
pub use cache::{AttentionKVCache, CacheFingerprint, CacheStats, CachedCandidateKV};
pub use config::{config_fingerprint, AttentionConfig, C8AttentionConfig};
pub use diagnostics::AttentionDiagnostics;
pub use distillation::{
    candidate_softmax, distillation_gradient, distillation_loss, kl_divergence,
};
pub use errors::{AttentionError, Result};
pub use model::{C7ModelCard, C8ModelCard, C8ResidualMeta, TrainingMeta};
pub use negative_mining::{
    HardNegative, HardNegativeConfig, HardNegativeMiner, NegSource, PoolEntry,
};
pub use projection::{truncated_identity, QkvProjection, ResidualQkvProjection};
pub use qkv::{AttentionEngine, CandidateAttention};
pub use scorer::{AttentionScorer, ResidualScorer, RetrievalEvidence, MIN_HEADS_FOR_DISAGREEMENT};

#[cfg(test)]
mod tests;
