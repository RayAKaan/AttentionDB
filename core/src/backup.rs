//! Backup & restore (Phase 1: full snapshot backups, no incrementals).
//!
//! A backup is internally consistent because [`crate::engine::AttentionEngine::backup_to`]
//! first *checkpoints*: all state through sequence N is in SSTables + idmap +
//! manifest, WAL segments before N are trimmed, and mutations are blocked by the
//! mutation gate while the copy runs. We therefore never copy a live directory
//! mid-write (the old admin.rs recursively copied during traffic — G18).
//!
//! The backup contains everything required to reconstruct the database:
//! CURRENT + MANIFEST (catalog), sst/ (documents incl. tombstones), META/
//! (idmap), WAL/ (post-checkpoint tail). HNSW graphs are intentionally NOT
//! included — they are deterministically rebuilt at open (docs/indexes.md).

use crate::error::CoreError;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    pub backup_format_version: u32,
    pub database_format_version: u32,
    pub created_at: String,
    pub checkpoint_seq: u64,
    pub collections: Vec<String>,
    pub source_dir: String,
}

pub const BACKUP_FORMAT_VERSION: u32 = 1;
const BACKUP_META_FILE: &str = "backup-meta.json";

/// Copy the authoritative state set from `db_dir` into `dest`. Caller must have
/// quiesced writes (engine.backup_to does this via checkpoint + mutation gate).
pub fn copy_database_dir(db_dir: &Path, dest: &Path) -> Result<(), CoreError> {
    use attentiondb_storage::Catalog;
    std::fs::create_dir_all(dest)?;

    for item in [
        Catalog::current_path(db_dir),
        db_dir.join("MANIFEST"),
        Catalog::sst_dir(db_dir),
        Catalog::meta_dir(db_dir),
        db_dir.join(attentiondb_storage::WAL_DIR_NAME),
    ] {
        copy_recursive(&item, &dest.join(item.file_name().unwrap_or_default()))?;
        attentiondb_storage::crashgate::GATE_BACKUP_MID_COPY.hit();
    }
    attentiondb_storage::crashgate::GATE_BACKUP_AFTER_COPY.hit();

    // Validate the copy actually forms an openable database before declaring success.
    let (catalog, _) = Catalog::load(dest)?;
    let meta = BackupManifest {
        backup_format_version: BACKUP_FORMAT_VERSION,
        database_format_version: catalog.database_format_version,
        created_at: chrono::Utc::now().to_rfc3339(),
        checkpoint_seq: catalog.checkpoint_seq,
        collections: catalog
            .collections
            .iter()
            .filter(|c| !c.dropped)
            .map(|c| c.name.clone())
            .collect(),
        source_dir: db_dir.display().to_string(),
    };
    std::fs::write(
        dest.join(BACKUP_META_FILE),
        serde_json::to_vec_pretty(&meta).map_err(|e| CoreError::Internal(e.to_string()))?,
    )?;
    attentiondb_storage::fsync_dir(dest)?;
    Ok(())
}

fn copy_recursive(src: &Path, dest: &Path) -> Result<(), CoreError> {
    if src.is_dir() {
        std::fs::create_dir_all(dest)?;
        for e in std::fs::read_dir(src)? {
            let e = e?;
            copy_recursive(&e.path(), &dest.join(e.file_name()))?;
        }
    } else if src.exists() {
        std::fs::copy(src, dest)?;
    }
    Ok(())
}

/// Restore a backup previously produced by [`copy_database_dir`] into
/// `dest_db_dir`. Fails if the destination is non-empty (never mixes states).
pub fn restore_backup(src: &Path, dest_db_dir: &Path) -> Result<BackupManifest, CoreError> {
    if !src.exists() {
        return Err(CoreError::NotFound(format!(
            "backup source {} does not exist",
            src.display()
        )));
    }
    if dest_db_dir.exists() {
        let non_empty = std::fs::read_dir(dest_db_dir)
            .map(|d| d.count() > 0)
            .unwrap_or(false);
        if non_empty {
            return Err(CoreError::Conflict(format!(
                "restore destination {} is not empty — refusing to overwrite",
                dest_db_dir.display()
            )));
        }
    }
    std::fs::create_dir_all(dest_db_dir)?;

    // E4: backup-meta.json is written LAST by copy_database_dir (after the
    // copy validates as an openable catalog) — it IS the completion marker.
    // A directory without it is a partially copied backup; restoring it could
    // silently mix an arbitrary prefix of files into a plausible-looking DB.
    let meta_path = src.join(BACKUP_META_FILE);
    if !meta_path.exists() {
        return Err(CoreError::Corruption(format!(
            "backup manifest {} missing — incomplete/invalid backup; refusing restore",
            meta_path.display()
        )));
    }
    let meta: BackupManifest = serde_json::from_slice(&std::fs::read(&meta_path)?)
        .map_err(|e| CoreError::Corruption(format!("backup meta: {e}")))?;
    // E4: never silently reinterpret an unknown backup format.
    if meta.backup_format_version != BACKUP_FORMAT_VERSION {
        return Err(CoreError::Corruption(format!(
            "backup format v{} unsupported (supported v{}) — refusing restore",
            meta.backup_format_version, BACKUP_FORMAT_VERSION
        )));
    }

    for item in [
        "CURRENT",
        "MANIFEST",
        "sst",
        "META",
        attentiondb_storage::WAL_DIR_NAME,
    ] {
        let s = src.join(item);
        if s.exists() {
            copy_recursive(&s, &dest_db_dir.join(item))?;
            attentiondb_storage::crashgate::GATE_RESTORE_MID_COPY.hit();
        }
    }

    // Validate before declaring success: the restored dir must load as a catalog
    // and open cleanly.
    let (catalog, _) = attentiondb_storage::Catalog::load(dest_db_dir)?;
    if meta.database_format_version != 0
        && catalog.database_format_version != meta.database_format_version
    {
        return Err(CoreError::Corruption(format!(
            "backup claims database format v{} but restored catalog says v{}",
            meta.database_format_version, catalog.database_format_version
        )));
    }
    let _ = crate::engine::AttentionEngine::open_dir(
        dest_db_dir,
        attentiondb_storage::Durability::Sync,
    )?;
    attentiondb_storage::fsync_dir(dest_db_dir)?;
    Ok(meta)
}
