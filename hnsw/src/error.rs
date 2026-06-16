use thiserror::Error;

#[derive(Error, Debug)]
pub enum HNSWError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HNSW internal error: {0}")]
    Hnsw(String),

    #[error("Index not found for head: {0}")]
    HeadNotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Index not built yet")]
    IndexNotBuilt,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("GPU error: {0}")]
    Gpu(#[from] crate::gpu::GpuError),

    #[error("Persistence error: {0}")]
    Persistence(String),
}
