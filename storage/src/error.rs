use thiserror::Error;

/// Phase 1 error taxonomy. Expected failures are typed errors, never panics:
/// a malformed database file must not crash the server through an uncontrolled panic.
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("WAL error: {0}")]
    Wal(String),

    #[error("Corruption detected in {file}: {detail}")]
    Corruption { file: String, detail: String },

    #[error("Recovery failed: {0}")]
    RecoveryFailed(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Unsupported format version in {file}: found {found}, supported {supported}")]
    UnsupportedVersion {
        file: String,
        found: u32,
        supported: u32,
    },

    #[error("Record not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Checksum mismatch in {file}")]
    ChecksumMismatch { file: String },

    #[error("SSTable error: {0}")]
    Sstable(String),

    #[error("Projection error: {0}")]
    Projection(String),
}

impl StorageError {
    pub fn corruption(file: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Corruption {
            file: file.into(),
            detail: detail.into(),
        }
    }
}
