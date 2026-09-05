use thiserror::Error;

/// Phase 1 error taxonomy (docs/phase1-final-report.md §error-handling).
/// Expected failures are typed errors — never panics.
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Collection already exists: {0}")]
    CollectionAlreadyExists(String),

    #[error("HNSW error: {0}")]
    Hnsw(#[from] attentiondb_hnsw::HNSWError),

    #[error("Query error: {0}")]
    Query(#[from] attentiondb_query::QueryError),

    #[error("MultiHead error: {0}")]
    MultiHead(#[from] attentiondb_multihead::MultiHeadError),

    #[error("Storage error: {0}")]
    Storage(#[from] attentiondb_storage::StorageError),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Unavailable: {0}")]
    Unavailable(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Corruption detected: {0}")]
    Corruption(String),

    #[error("Recovery failed: {0}")]
    RecoveryFailed(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Transaction error: {0}")]
    Transaction(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<std::io::Error> for CoreError {
    fn from(e: std::io::Error) -> Self {
        CoreError::Storage(attentiondb_storage::StorageError::Io(e))
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(e: serde_json::Error) -> Self {
        CoreError::Internal(e.to_string())
    }
}

impl CoreError {
    /// Stable machine-readable category used by APIs and the checker.
    pub fn category(&self) -> &'static str {
        match self {
            CoreError::CollectionNotFound(_) | CoreError::NotFound(_) => "NotFound",
            CoreError::CollectionAlreadyExists(_) | CoreError::AlreadyExists(_) => "AlreadyExists",
            CoreError::Conflict(_) => "Conflict",
            CoreError::InvalidArgument(_) | CoreError::InvalidConfig(_) => "InvalidArgument",
            CoreError::ResourceExhausted(_) => "ResourceExhausted",
            CoreError::Unavailable(_) => "Unavailable",
            CoreError::Timeout(_) => "Timeout",
            CoreError::Corruption(_) => "Corruption",
            CoreError::RecoveryFailed(_) => "RecoveryFailed",
            CoreError::InvalidOperation(_) => "InvalidOperation",
            CoreError::Transaction(_) => "Transaction",
            _ => "Internal",
        }
    }
}
