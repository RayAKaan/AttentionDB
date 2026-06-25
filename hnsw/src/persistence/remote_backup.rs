use crate::persistence::error::PersistenceError;
use std::path::Path;

pub async fn upload_backup(backup_dir: &Path, remote_url: &str) -> Result<(), PersistenceError> {
    if !backup_dir.exists() {
        return Err(PersistenceError::IndexNotFound(
            backup_dir.to_string_lossy().to_string(),
        ));
    }

    let client = reqwest::Client::new();

    let meta_path = backup_dir.join("metadata.json");
    let vectors_path = backup_dir.join("vectors.bin");

    if !meta_path.exists() || !vectors_path.exists() {
        return Err(PersistenceError::IndexNotFound(
            "Backup is missing required files".to_string(),
        ));
    }

    let metadata = tokio::fs::read_to_string(&meta_path)
        .await
        .map_err(PersistenceError::Io)?;

    let backup_name = backup_dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    let form = reqwest::multipart::Form::new()
        .text("metadata", metadata)
        .text("backup_name", backup_name);

    let response = client
        .post(remote_url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| PersistenceError::Io(std::io::Error::other(e.to_string())))?;

    if response.status().is_success() {
        Ok(())
    } else {
        Err(PersistenceError::Io(std::io::Error::other(format!(
            "Remote backup failed with status: {}",
            response.status()
        ))))
    }
}

pub async fn download_backup(_remote_url: &str, _local_dir: &Path) -> Result<(), PersistenceError> {
    Err(PersistenceError::Io(std::io::Error::other(
        "Remote download not yet implemented",
    )))
}
