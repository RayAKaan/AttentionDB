use thiserror::Error;

#[derive(Error, Debug)]
pub enum DistributedError {
    #[error("Shard error: {0}")]
    Shard(String),

    #[error("Raft consensus error: {0}")]
    Raft(String),

    #[error("Replication error: {0}")]
    Replication(String),

    #[error("Chaos test failure: {0}")]
    Chaos(String),

    #[error("Network error: {0}")]
    Network(String),
}
