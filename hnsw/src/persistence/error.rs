use thiserror::Error;

#[derive(Error, Debug)]
pub enum PersistenceError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Index not found at path: {0}")]
    IndexNotFound(String),

    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),
}
