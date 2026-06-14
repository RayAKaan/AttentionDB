use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("GPU backend not available")]
    NotAvailable,

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Memory allocation error: {0}")]
    Memory(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}
