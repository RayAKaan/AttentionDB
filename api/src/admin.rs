//! Admin Operations — Backup, Restore, Maintenance
//!
//! Provides administrative API endpoints for backup and restore operations.

use crate::rest::AppState;
use axum::extract::State;
use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
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
    if !engine.is_persistent() {
        return Err((
            StatusCode::PRECONDITION_FAILED,
            "engine is not persistent (no data directory); nothing to back up".to_string(),
        ));
    }
    let timestamp = format_timestamp(SystemTime::now());
    let backup_id = format!("backup_{timestamp}");
    let dest = payload
        .destination
        .map(PathBuf::from)
        .unwrap_or_else(|| backups_dir().join(&backup_id));

    // Phase 1 backup: checkpoint (quiesce + durable state) then copy the
    // authoritative state set (catalog + SST + idmap + WAL tail). Internally
    // consistent by construction — see core/src/backup.rs.
    let checkpoint = tokio::task::block_in_place(|| engine.backup_to(&dest))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let collections = engine.list_collections();
    let size_bytes = dir_size(&dest);

    tracing::info!(
        backup_id = %backup_id,
        checkpoint_seq = checkpoint.checkpoint_seq,
        size = size_bytes,
        "backup created (checkpoint-consistent)"
    );

    Ok(Json(BackupResponse {
        backup_id,
        timestamp,
        collections,
        path: dest.to_string_lossy().to_string(),
        size_bytes,
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

#[derive(Deserialize)]
pub struct RestoreRequestBody {
    pub backup_id: String,
    /// Where to restore. If omitted, restores into `<data_dir>.restored-<ts>`
    /// (a running server cannot overwrite its own live data directory).
    pub destination: Option<String>,
}

pub async fn restore_handler(
    Json(payload): Json<RestoreRequestBody>,
) -> Result<Json<RestoreResponse>, (StatusCode, String)> {
    let backup_dir = backups_dir().join(&payload.backup_id);
    if !backup_dir.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("Backup '{}' not found", payload.backup_id),
        ));
    }

    let dest = payload.destination.map(PathBuf::from).unwrap_or_else(|| {
        let dd = data_dir();
        let ts = format_timestamp(SystemTime::now());
        dd.with_file_name(format!(
            "{}.restored-{ts}",
            dd.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("attentiondb-data")
        ))
    });

    // Restores and *validates* by opening the restored directory; fails loudly
    // on any corruption instead of half-restoring.
    let meta = tokio::task::block_in_place(|| {
        attentiondb_core::backup::restore_backup(&backup_dir, &dest)
    })
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    tracing::info!(
        backup_id = %payload.backup_id,
        dest = %dest.display(),
        checkpoint_seq = meta.checkpoint_seq,
        "backup restored and validated"
    );

    Ok(Json(RestoreResponse {
        success: true,
        message: format!(
            "Restored '{}' to {} (checkpoint_seq={}, collections={}). Restart the server with ATTENTIONDB_DATA_DIR pointing there.",
            payload.backup_id,
            dest.display(),
            meta.checkpoint_seq,
            meta.collections.len()
        ),
    }))
}

/// Run the consistency checker against the live engine.
pub async fn check_admin_handler(
    State(_state): State<AppState>,
) -> Result<axum::Json<serde_json::Value>, (StatusCode, String)> {
    let issues = attentiondb_core::checker::check_engine(&_state.service.engine);
    let errors = issues.iter().filter(|i| i.severity.is_error()).count();
    Ok(axum::Json(serde_json::json!({
        "ok": errors == 0,
        "errors": errors,
        "warnings": issues.len() - errors,
        "issues": issues,
    })))
}

/// Force a checkpoint.
pub async fn checkpoint_admin_handler(
    State(state): State<AppState>,
) -> Result<axum::Json<serde_json::Value>, (StatusCode, String)> {
    let info = state
        .service
        .engine
        .checkpoint()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(axum::Json(serde_json::json!({
        "success": true,
        "checkpoint_seq": info.checkpoint_seq,
        "manifest_generation": info.manifest_generation,
        "duration_ms": info.duration_ms,
    })))
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
