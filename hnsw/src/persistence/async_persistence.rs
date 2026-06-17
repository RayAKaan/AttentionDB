use crate::hnsw_index::HNSWIndex;
use crate::persistence::error::PersistenceError;
use std::path::Path;
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub async fn save_index_async(
    index: &HNSWIndex,
    dir: &Path,
) -> Result<(), PersistenceError> {
    fs::create_dir_all(dir).await?;

    let checksum: u64 = index.vectors
        .iter()
        .flat_map(|(_, vec)| vec)
        .map(|v| v.to_bits() as u64)
        .sum();

    let metadata = crate::persistence::index_persistence::IndexMetadata {
        version: 2,
        head_name: index.head_name.clone(),
        dim: index.dim,
        config: index.config.clone(),
        vector_count: index.len(),
        created_at: chrono::Utc::now().to_rfc3339(),
        checksum: Some(checksum.to_string()),
    };

    let meta_path = dir.join("metadata.json");
    let meta_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
    fs::write(&meta_path, meta_json).await?;

    let vectors_path = dir.join("vectors.bin");
    let mut file = fs::File::create(&vectors_path).await?;

    file.write_all(&(index.len() as u64).to_le_bytes()).await?;

    for (id, vec) in &index.vectors {
        file.write_all(&id.to_le_bytes()).await?;
        file.write_all(&(vec.len() as u32).to_le_bytes()).await?;
        for val in vec {
            file.write_all(&val.to_le_bytes()).await?;
        }
    }

    Ok(())
}
