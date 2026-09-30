use thiserror::Error;

#[derive(Debug, Error)]
pub enum AttentionError {
    #[error("invalid dimensions: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    #[error("invalid head count: expected {expected}, found {found}")]
    HeadCountMismatch { expected: usize, found: usize },

    #[error("invalid matrix dimensions: {0}")]
    InvalidMatrixDimensions(String),

    #[error("non-finite value encountered: {0}")]
    NonFinite(String),

    #[error("empty input: {0}")]
    EmptyInput(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("training error: {0}")]
    Training(String),
}

pub type Result<T> = std::result::Result<T, AttentionError>;
