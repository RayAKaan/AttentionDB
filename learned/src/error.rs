use thiserror::Error;

#[derive(Error, Debug)]
pub enum LearnedError {
    #[error("Training error: {0}")]
    Training(String),

    #[error("Projection error: {0}")]
    Projection(String),

    #[error("Reprojection error: {0}")]
    Reprojection(String),

    #[error("Invalid pair: {0}")]
    InvalidPair(String),
}
