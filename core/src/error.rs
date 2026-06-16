use thiserror::Error;

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
}
