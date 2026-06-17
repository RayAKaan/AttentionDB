use crate::persistence::error::PersistenceError;
use std::path::Path;

pub async fn compact_index_async(dir: &Path) -> Result<usize, PersistenceError> {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        crate::persistence::compaction::compact_index(&dir)
    })
    .await
    .map_err(|e| PersistenceError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?
}
