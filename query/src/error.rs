use thiserror::Error;

#[derive(Error, Debug)]
pub enum QueryError {
    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Planning error: {0}")]
    Planning(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Invalid head: {0}")]
    InvalidHead(String),

    #[error("Unsupported feature: {0}")]
    Unsupported(String),
}
