use thiserror::Error;

#[derive(Error, Debug)]
pub enum MultiHeadError {
    #[error("Head not found: {0}")]
    HeadNotFound(String),

    #[error("Gating error: {0}")]
    Gating(String),

    #[error("Fusion error: {0}")]
    Fusion(String),

    #[error("Invalid head configuration: {0}")]
    InvalidConfig(String),
}
