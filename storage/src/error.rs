use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("WAL error: {0}")]
    Wal(String),

    #[error("Record not found: {0}")]
    NotFound(String),

    #[error("Checksum mismatch")]
    ChecksumMismatch,

    #[error("SSTable error: {0}")]
    Sstable(String),

    #[error("Projection error: {0}")]
    Projection(String),
}
