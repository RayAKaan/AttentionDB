use crate::persistence::error::PersistenceError;
use std::path::{Path, PathBuf};
use std::fs;

pub fn create_backup(dir: &Path) -> Result<PathBuf, PersistenceError> {
    if !dir.exists() {
        return Err(PersistenceError::IndexNotFound(dir.to_string_lossy().to_string()));
    }

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let dir_name = dir.file_name()
        .and_then(|n| Some(n.to_string_lossy().to_string()))
        .unwrap_or_else(|| "attentiondb_backup".to_string());
    let backup_dir = dir.parent()
        .unwrap_or(dir)
        .join(format!("backup_{}_{}", dir_name, timestamp));

    fs::create_dir_all(&backup_dir)?;

    let meta_src = dir.join("metadata.json");
    if meta_src.exists() {
        fs::copy(&meta_src, backup_dir.join("metadata.json"))?;
    }

    let vectors_src = dir.join("vectors.bin");
    if vectors_src.exists() {
        fs::copy(&vectors_src, backup_dir.join("vectors.bin"))?;
    }

    let graph_meta_src = dir.join("graph_metadata.json");
    if graph_meta_src.exists() {
        fs::copy(&graph_meta_src, backup_dir.join("graph_metadata.json"))?;
    }

    Ok(backup_dir)
}

pub fn list_backups(dir: &Path) -> Result<Vec<PathBuf>, PersistenceError> {
    let parent = dir.parent().unwrap_or(dir);
    let prefix = format!(
        "backup_{}_",
        dir.file_name()
            .and_then(|n| Some(n.to_string_lossy().to_string()))
            .unwrap_or_else(|| "attentiondb_backup".to_string())
    );

    let mut backups = Vec::new();
    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) {
                backups.push(entry.path());
            }
        }
    }

    backups.sort();
    Ok(backups)
}
