//! SSTable Compaction — Merges multiple small SST files into larger ones.
//!
//! Compaction reduces the number of files the storage engine must scan on reads
//! and removes tombstone entries (deleted records) permanently.

use crate::error::StorageError;
use crate::record::Record;
use crate::sstable::{SSTableReader, SSTableWriter};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Configuration for compaction behavior.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Minimum number of SST files before compaction triggers.
    pub min_files_to_compact: usize,
    /// Maximum number of SST files to merge in a single compaction run.
    pub max_files_per_run: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            min_files_to_compact: 4,
            max_files_per_run: 8,
        }
    }
}

/// Result of a compaction run.
#[derive(Debug)]
pub struct CompactionResult {
    /// Number of input SST files merged.
    pub files_merged: usize,
    /// Number of entries in the output file.
    pub output_entries: usize,
    /// Number of tombstone entries removed.
    pub tombstones_removed: usize,
    /// Path of the new merged SST file.
    pub output_path: PathBuf,
    /// Paths of input files that were merged (safe to delete).
    pub merged_paths: Vec<PathBuf>,
}

/// Incremental size-tiered compaction: merge the oldest N SST files into one,
/// deduplicating by key (latest timestamp wins). Tombstones are retained unless
/// the merge happens to cover every file (see `compact_all`).
pub fn compact(
    dir: &Path,
    config: &CompactionConfig,
) -> Result<Option<CompactionResult>, StorageError> {
    let sst_paths = list_sst_files(dir)?;

    if sst_paths.len() < config.min_files_to_compact {
        debug!(
            sst_count = sst_paths.len(),
            threshold = config.min_files_to_compact,
            "Not enough SST files for compaction"
        );
        return Ok(None);
    }

    // Incremental (partial) merge: take the oldest `max_files_per_run` files.
    // Tombstones are RETAINED — an old unmerged file may still hold a pre-delete
    // version of the key, and dropping the tombstone here could resurrect it
    // (INV-12). Tombstones are only reclaimed by `compact_all` (full compaction).
    let total_files = sst_paths.len();
    let files_to_merge: Vec<PathBuf> = sst_paths
        .into_iter()
        .take(config.max_files_per_run)
        .collect();
    let gc_tombstones = files_to_merge.len() == total_files;

    do_compact(dir, files_to_merge, gc_tombstones)
}

/// Full compaction: merge EVERY SSTable in the directory and reclaim tombstones.
/// Safe because no unmerged file can hold an older version of any key.
pub fn compact_all(dir: &Path) -> Result<Option<CompactionResult>, StorageError> {
    let sst_paths = list_sst_files(dir)?;
    if sst_paths.len() < 2 {
        return Ok(None);
    }
    do_compact(dir, sst_paths, true)
}

fn list_sst_files(dir: &Path) -> Result<Vec<PathBuf>, StorageError> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) == Some("sst") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn do_compact(
    dir: &Path,
    files_to_merge: Vec<PathBuf>,
    gc_tombstones: bool,
) -> Result<Option<CompactionResult>, StorageError> {
    info!(
        files = files_to_merge.len(),
        dir = %dir.display(),
        gc_tombstones,
        "Starting SSTable compaction"
    );

    let mut merged: BTreeMap<Vec<u8>, (Vec<u8>, i64, bool)> = BTreeMap::new();

    for path in &files_to_merge {
        let reader = SSTableReader::open(path)?;
        for entry in reader.iter() {
            let is_tombstone = if let Ok(record) = Record::from_msgpack(&entry.value) {
                record.tags.contains(&"__TOMBSTONE__".to_string())
            } else {
                false
            };

            let should_replace = match merged.get(&entry.key) {
                Some((_, existing_ts, _)) => entry.timestamp > *existing_ts,
                None => true,
            };

            if should_replace {
                merged.insert(
                    entry.key.clone(),
                    (entry.value.clone(), entry.timestamp, is_tombstone),
                );
            }
        }
    }

    let before_gc = merged.len();
    if gc_tombstones {
        merged.retain(|_, (_, _, is_tombstone)| !*is_tombstone);
    }
    let total_tombstones = before_gc - merged.len();

    let timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    let output_path = dir.join(format!("compacted_{timestamp}.sst"));
    let mut writer = SSTableWriter::new(&output_path)?;

    // Preserve each entry's ORIGINAL logical timestamp: the output must compete
    // fairly with files outside the merge set (a re-stamped old record could
    // otherwise shadow a newer tombstone/update and resurrect deleted data).
    for (key, (value, ts, _)) in &merged {
        writer.append_with_timestamp(key.clone(), value.clone(), *ts)?;
    }
    writer.flush()?;

    let result = CompactionResult {
        files_merged: files_to_merge.len(),
        output_entries: merged.len(),
        tombstones_removed: total_tombstones,
        output_path,
        merged_paths: files_to_merge,
    };

    info!(
        files_merged = result.files_merged,
        output_entries = result.output_entries,
        tombstones_removed = result.tombstones_removed,
        output = %result.output_path.display(),
        "Compaction complete"
    );

    Ok(Some(result))
}

/// Remove the old SST files after successful compaction.
/// Call this only after the new compacted file is verified.
pub fn cleanup_merged_files(result: &CompactionResult) -> Result<usize, StorageError> {
    let mut removed = 0;
    for path in &result.merged_paths {
        if let Err(e) = std::fs::remove_file(path) {
            tracing::warn!(path = %path.display(), error = %e, "Failed to remove merged SST file");
        } else {
            removed += 1;
        }
    }
    info!(removed = removed, "Cleaned up merged SST files");
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;

    fn make_record(idx: usize, tombstone: bool) -> Record {
        let mut fields = HashMap::new();
        fields.insert("idx".to_string(), serde_json::json!(idx));
        let mut rec = Record::new(fields);
        if tombstone {
            rec.tags.push("__TOMBSTONE__".to_string());
        }
        rec
    }

    #[test]
    fn test_compaction_merges_and_removes_tombstones() {
        let dir = tempdir().unwrap();

        for batch in 0..4 {
            let path = dir.path().join(format!("sstable_{:03}.sst", batch));
            let mut writer = SSTableWriter::new(&path).unwrap();
            for j in 0..10 {
                let idx = batch * 10 + j;
                let tombstone = idx == 5 || idx == 15;
                let rec = make_record(idx, tombstone);
                let key = format!("key_{:04}", idx).into_bytes();
                let value = rec.to_msgpack().unwrap();
                writer.append(key, value).unwrap();
            }
            writer.flush().unwrap();
        }

        let config = CompactionConfig {
            min_files_to_compact: 4,
            max_files_per_run: 4,
        };

        let result = compact(dir.path(), &config).unwrap();
        assert!(result.is_some());

        let result = result.unwrap();
        assert_eq!(result.files_merged, 4);
        assert_eq!(result.tombstones_removed, 2);
        assert_eq!(result.output_entries, 38);

        let reader = SSTableReader::open(&result.output_path).unwrap();
        assert_eq!(reader.len(), 38);

        let removed = cleanup_merged_files(&result).unwrap();
        assert_eq!(removed, 4);
    }

    #[test]
    fn test_compaction_skips_when_too_few_files() {
        let dir = tempdir().unwrap();

        for batch in 0..2 {
            let path = dir.path().join(format!("sstable_{:03}.sst", batch));
            let mut writer = SSTableWriter::new(&path).unwrap();
            writer.append(b"key".to_vec(), b"val".to_vec()).unwrap();
            writer.flush().unwrap();
        }

        let config = CompactionConfig::default();
        let result = compact(dir.path(), &config).unwrap();
        assert!(result.is_none());
    }
}
