//! Admin Operations — Backup, Restore, Maintenance
//!
//! Provides administrative API endpoints for backup and restore operations.

use crate::rest::AppState;
use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Serialize)]
pub struct BackupResponse {
    pub backup_id: String,
    pub timestamp: String,
    pub collections: Vec<String>,
    pub path: String,
    pub size_bytes: u64,
}

#[derive(Serialize)]
pub struct BackupsListResponse {
    pub backups: Vec<BackupResponse>,
}

#[derive(Serialize)]
pub struct RestoreResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Deserialize)]
pub struct BackupRequest {
    pub destination: Option<String>,
}

#[derive(Deserialize)]
pub struct RestoreRequest {
    pub backup_id: String,
}

fn format_timestamp(time: std::time::SystemTime) -> String {
    let datetime: chrono::DateTime<chrono::Utc> = time.into();
    datetime.format("%Y%m%d_%H%M%S").to_string()
}

fn data_dir() -> PathBuf {
    PathBuf::from(std::env::var("ATTENTIONDB_DATA_DIR").unwrap_or_else(|_| "/data".into()))
}

fn backups_dir() -> PathBuf {
    let dir = data_dir().parent().unwrap_or(&data_dir()).join("backups");
    std::fs::create_dir_all(&dir).ok();
    dir
}

pub async fn backup_handler(
    state: axum::extract::State<AppState>,
    Json(payload): Json<BackupRequest>,
) -> Result<Json<BackupResponse>, (StatusCode, String)> {
    let engine = &state.service.engine;
    let collections = engine.list_collections();

    let timestamp = format_timestamp(SystemTime::now());
    let backup_id = format!("backup_{}", timestamp);

    let dest = payload
        .destination
        .map(PathBuf::from)
        .unwrap_or_else(|| backups_dir().join(&backup_id));
    std::fs::create_dir_all(&dest)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut total_bytes = 0u64;
    let mut backed_up_collections = Vec::new();

    for coll_name in &collections {
        let coll = engine
            .get_collection(coll_name)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let heads = coll.list_heads();
        let coll_dir = dest.join(coll_name);
        std::fs::create_dir_all(&coll_dir)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        for head_name in &heads {
            let head_dir = coll_dir.join(head_name);
            std::fs::create_dir_all(&head_dir)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            match coll.head_manager.read().get_head(head_name) {
                Ok(idx) => {
                    let idx_guard = idx.read();
                    if let Err(e) = attentiondb_hnsw::persistence::save_index(&idx_guard, &head_dir)
                    {
                        return Err((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!(
                                "Failed to save index for {}/{}: {}",
                                coll_name, head_name, e
                            ),
                        ));
                    }
                    total_bytes += dir_size(&head_dir);
                }
                Err(e) => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Head '{}' not found in '{}': {}", head_name, coll_name, e),
                    ));
                }
            }
        }

        backed_up_collections.push(coll_name.clone());
    }

    // Save manifest
    let manifest = serde_json::json!({
        "backup_id": backup_id,
        "timestamp": timestamp,
        "collections": backed_up_collections,
        "heads_per_collection": collections.iter().map(|c| {
            let coll = engine.get_collection(c).ok();
            let heads = coll.map(|c| c.list_heads()).unwrap_or_default();
            (c.clone(), heads)
        }).collect::<HashMap<_, _>>(),
    });
    let manifest_path = dest.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    total_bytes += manifest_json.len() as u64;
    std::fs::write(&manifest_path, &manifest_json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Copy WAL if persistent
    if engine.is_persistent() {
        let wal_path = data_dir().join("engine.wal");
        if wal_path.exists() {
            let wal_backup = dest.join("engine.wal");
            if let Err(e) = std::fs::copy(&wal_path, &wal_backup) {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to copy WAL: {}", e),
                ));
            }
            if let Ok(meta) = std::fs::metadata(&wal_path) {
                total_bytes += meta.len();
            }
        }
    }

    tracing::info!(backup_id = %backup_id, collections = ?backed_up_collections, size = total_bytes, "Backup created");

    Ok(Json(BackupResponse {
        backup_id,
        timestamp,
        collections: backed_up_collections,
        path: dest.to_string_lossy().to_string(),
        size_bytes: total_bytes,
    }))
}

pub async fn list_backups_handler() -> Result<Json<BackupsListResponse>, (StatusCode, String)> {
    let backup_dir = backups_dir();
    let mut backups = Vec::new();

    if !backup_dir.exists() {
        return Ok(Json(BackupsListResponse { backups }));
    }

    let entries = std::fs::read_dir(&backup_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if !dir_name.starts_with("backup_") {
            continue;
        }

        let manifest_path = path.join("manifest.json");
        let (collections, size_bytes) = if manifest_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&manifest_path) {
                if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
                    let cols = manifest["collections"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let size = manifest_path.metadata().map(|m| m.len()).unwrap_or(0);
                    (cols, dir_size(&path) + size)
                } else {
                    (vec![], dir_size(&path))
                }
            } else {
                (vec![], dir_size(&path))
            }
        } else {
            (vec![], dir_size(&path))
        };

        backups.push(BackupResponse {
            backup_id: dir_name.clone(),
            timestamp: dir_name.trim_start_matches("backup_").replace('_', " "),
            collections,
            path: path.to_string_lossy().to_string(),
            size_bytes,
        });
    }

    backups.sort_by(|a, b| b.backup_id.cmp(&a.backup_id));
    Ok(Json(BackupsListResponse { backups }))
}

pub async fn restore_handler(
    Json(payload): Json<RestoreRequest>,
) -> Result<Json<RestoreResponse>, (StatusCode, String)> {
    let backup_dir = backups_dir().join(&payload.backup_id);

    if !backup_dir.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Backup '{}' not found", payload.backup_id),
        ));
    }

    let manifest_path = backup_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Backup '{}' is corrupted — manifest missing",
                payload.backup_id
            ),
        ));
    }

    let manifest_content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_content)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let collections = manifest["collections"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();

    // Verify all collection directories exist
    for coll in &collections {
        let coll_dir = backup_dir.join(coll);
        if !coll_dir.exists() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Backup is incomplete — collection '{}' directory missing",
                    coll
                ),
            ));
        }
    }

    tracing::info!(backup_id = %payload.backup_id, collections = ?collections, "Backup validated successfully");

    Ok(Json(RestoreResponse {
        success: true,
        message: format!("Backup '{}' validated. Contains {} collections. To restore, restart the server with data directory pointing to this backup.", payload.backup_id, collections.len()),
    }))
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += dir_size(&path);
            } else if let Ok(meta) = path.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dir_size_empty() {
        let dir = std::env::temp_dir().join("attentiondb_test_backup_empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(dir_size(&dir), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_dir_size_with_file() {
        let dir = std::env::temp_dir().join("attentiondb_test_backup_file");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("test.txt"), b"hello world").unwrap();
        assert!(dir_size(&dir) > 0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
