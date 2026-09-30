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
pub mod config;
pub mod diagnostics;
pub mod errors;
pub mod model;
pub mod projection;
pub mod qkv;
pub mod scorer;
pub mod training;

pub use alignment::{AlignmentProjection, DetRng};
pub use attention::{AttentionOutput, AttentionSubsystem};
pub use config::{config_fingerprint, AttentionConfig};
pub use diagnostics::AttentionDiagnostics;
pub use errors::{AttentionError, Result};
pub use model::{C7ModelCard, TrainingMeta};
pub use projection::QkvProjection;
pub use qkv::{AttentionEngine, CandidateAttention};
pub use scorer::{AttentionScorer, RetrievalEvidence};

#[cfg(test)]
mod tests;
