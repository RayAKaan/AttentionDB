//! Durable database catalog (MANIFEST/CURRENT) — Phase 1.
//!
//! The catalog is the authoritative persistent record of:
//! - database + manifest format versions
//! - collection definitions (id, name, dim, heads, settings, schema, state)
//! - next collection id / next document numeric id (monotonic across restarts)
//! - the WAL sequence covered by the last checkpoint
//!
//! Crash-safety protocol for every update (`docs/consistency-model.md` §1):
//! 1. serialize + checksum
//! 2. write `<target>.tmp`
//! 3. fsync file
//! 4. atomic rename onto the final name
//! 5. fsync parent directory
//!
//! `CURRENT` holds the name of the authoritative manifest. If CURRENT or the
//! newest manifest is unreadable/corrupt, recovery falls back to the highest
//! generation that validates. A MANIFEST directory that contains files but zero
//! valid manifests is a hard error — we never start "fresh" over existing data.

use crate::error::StorageError;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const DATABASE_FORMAT_VERSION: u32 = 1;
pub const MANIFEST_FORMAT_VERSION: u32 = 1;
pub const IDMAP_FORMAT_VERSION: u32 = 1;

const MANIFEST_MAGIC: u32 = 0x414D_4346; // "AMCF"
const MANIFEST_DIR_NAME: &str = "MANIFEST";
const CURRENT_NAME: &str = "CURRENT";
const META_DIR_NAME: &str = "META";
const IDMAP_MAGIC: u32 = 0x414D_4944; // "AMID"
const SST_DIR_NAME: &str = "sst";
const INDEX_DIR_NAME: &str = "INDEX";

/// A retrieval head definition (per-head HNSW overrides are stored as JSON so the
/// storage crate does not depend on the hnsw crate).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeadMeta {
    pub name: String,
    /// JSON-encoded per-head settings override ("" = inherit collection settings)
    #[serde(default)]
    pub settings_json: String,
}

/// One collection's durable definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CollectionMeta {
    pub id: u32,
    pub name: String,
    pub dim: usize,
    pub heads: Vec<HeadMeta>,
    /// JSON-encoded collection-level settings
    pub settings_json: String,
    /// declared schema: (field name, type) pairs
    pub schema_fields: Vec<(String, String)>,
    /// WAL sequence at which this collection was created
    pub created_seq: u64,
    /// dropped collections are kept (tombstoned) until the checkpoint that GCs them
    pub dropped: bool,
}

/// The database catalog. Versioned, checksummed, atomically installed.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Catalog {
    pub format_version: u32,
    pub database_format_version: u32,
    pub next_collection_id: u32,
    /// global monotonic document-id allocator state
    pub next_doc_numeric_id: u64,
    /// all WAL mutations with seq <= checkpoint_seq are represented in SSTables +
    /// idmap snapshot; recovery only needs to replay seq > checkpoint_seq
    pub checkpoint_seq: u64,
    pub collections: Vec<CollectionMeta>,
}

impl Catalog {
    pub fn fresh() -> Self {
        Self {
            format_version: MANIFEST_FORMAT_VERSION,
            database_format_version: DATABASE_FORMAT_VERSION,
            next_collection_id: 1,
            next_doc_numeric_id: 1,
            checkpoint_seq: 0,
            collections: Vec::new(),
        }
    }

    pub fn live_collection(&self, name: &str) -> Option<&CollectionMeta> {
        self.collections
            .iter()
            .find(|c| c.name == name && !c.dropped)
    }

    pub fn manifest_dir(db_dir: &Path) -> PathBuf {
        db_dir.join(MANIFEST_DIR_NAME)
    }

    pub fn meta_dir(db_dir: &Path) -> PathBuf {
        db_dir.join(META_DIR_NAME)
    }

    pub fn sst_dir(db_dir: &Path) -> PathBuf {
        db_dir.join(SST_DIR_NAME)
    }

    pub fn index_dir(db_dir: &Path) -> PathBuf {
        db_dir.join(INDEX_DIR_NAME)
    }

    pub fn current_path(db_dir: &Path) -> PathBuf {
        db_dir.join(CURRENT_NAME)
    }

    /// Ensure the standard directory skeleton exists.
    pub fn ensure_dirs(db_dir: &Path) -> Result<(), StorageError> {
        for d in [
            Self::manifest_dir(db_dir),
            Self::meta_dir(db_dir),
            Self::sst_dir(db_dir),
            Self::index_dir(db_dir),
            db_dir.join(crate::wal::WAL_DIR_NAME),
        ] {
            std::fs::create_dir_all(&d)?;
        }
        Ok(())
    }

    fn encode(&self) -> Result<Vec<u8>, StorageError> {
        let body =
            bincode::serialize(self).map_err(|e| StorageError::Serialization(e.to_string()))?;
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&body);
        let crc = hasher.finalize();
        let mut out = Vec::with_capacity(body.len() + 16);
        out.extend_from_slice(&MANIFEST_MAGIC.to_be_bytes());
        out.extend_from_slice(&MANIFEST_FORMAT_VERSION.to_be_bytes());
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc.to_be_bytes());
        Ok(out)
    }

    fn decode(bytes: &[u8], file: &str) -> Result<Catalog, StorageError> {
        if bytes.len() < 16 {
            return Err(StorageError::corruption(file, "manifest too short"));
        }
        let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if magic != MANIFEST_MAGIC {
            return Err(StorageError::corruption(file, "bad manifest magic"));
        }
        let version = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != MANIFEST_FORMAT_VERSION {
            return Err(StorageError::UnsupportedVersion {
                file: file.to_string(),
                found: version,
                supported: MANIFEST_FORMAT_VERSION,
            });
        }
        let len = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        if bytes.len() < 12 + len + 4 {
            return Err(StorageError::corruption(file, "manifest truncated"));
        }
        let body = &bytes[12..12 + len];
        let stored_crc = u32::from_be_bytes([
            bytes[12 + len],
            bytes[13 + len],
            bytes[14 + len],
            bytes[15 + len],
        ]);
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(body);
        if hasher.finalize() != stored_crc {
            return Err(StorageError::ChecksumMismatch {
                file: file.to_string(),
            });
        }
        let catalog: Catalog = bincode::deserialize(body)
            .map_err(|e| StorageError::corruption(file, format!("body: {e}")))?;
        Ok(catalog)
    }

    /// Atomically install this catalog as the newest manifest generation and point
    /// CURRENT at it. Returns the generation number installed.
    pub fn save(&self, db_dir: &Path) -> Result<u64, StorageError> {
        let manifest_dir = Self::manifest_dir(db_dir);
        std::fs::create_dir_all(&manifest_dir)?;
        let generation = Self::latest_generation(db_dir)?.unwrap_or(0) + 1;
        let bytes = self.encode()?;
        let final_path = manifest_dir.join(format!("manifest-{:09}", generation));
        let tmp_path = manifest_dir.join(format!("manifest-{:09}.tmp", generation));

        {
            let mut f = File::create(&tmp_path)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp_path, &final_path)?;
        fsync_dir(&manifest_dir)?;

        // Now repoint CURRENT (same crash-safe protocol).
        let cur_tmp = db_dir.join(format!("{CURRENT_NAME}.tmp"));
        {
            let mut f = File::create(&cur_tmp)?;
            let name = final_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();
            writeln!(f, "{name}")?;
            f.sync_all()?;
        }
        std::fs::rename(&cur_tmp, Self::current_path(db_dir))?;
        fsync_dir(db_dir)?;
        Ok(generation)
    }

    fn latest_generation(db_dir: &Path) -> Result<Option<u64>, StorageError> {
        let manifest_dir = Self::manifest_dir(db_dir);
        if !manifest_dir.exists() {
            return Ok(None);
        }
        let mut max: Option<u64> = None;
        for e in std::fs::read_dir(&manifest_dir)? {
            let p = e?.path();
            if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                if let Some(num) = stem.strip_prefix("manifest-") {
                    if let Ok(g) = num.parse::<u64>() {
                        max = Some(max.map_or(g, |m: u64| m.max(g)));
                    }
                }
            }
        }
        Ok(max)
    }

    /// Load the authoritative catalog.
    ///
    /// - `Ok((catalog, false))` — loaded from CURRENT.
    /// - `Ok((catalog, true))` — CURRENT was missing/corrupt; recovered from the
    ///   highest valid manifest generation (reported so startup can log it).
    /// - `Err(Corruption)` — manifests exist but none validate (we refuse to start
    ///   fresh over existing data).
    /// - `Ok((fresh, false))` — no manifest state exists at all (new database).
    pub fn load(db_dir: &Path) -> Result<(Catalog, bool), StorageError> {
        let manifest_dir = Self::manifest_dir(db_dir);
        if !manifest_dir.exists() {
            return Ok((Catalog::fresh(), false));
        }
        let has_files = std::fs::read_dir(&manifest_dir)?
            .filter_map(|e| e.ok())
            .any(|e| e.path().is_file());
        if !has_files {
            return Ok((Catalog::fresh(), false));
        }

        // 1) Try CURRENT.
        let current = Self::current_path(db_dir);
        if current.exists() {
            if let Ok(name) = std::fs::read_to_string(&current) {
                let name = name.trim();
                if !name.is_empty() {
                    let path = manifest_dir.join(name);
                    if let Ok(bytes) = std::fs::read(&path) {
                        match Self::decode(&bytes, &path.display().to_string()) {
                            Ok(c) => return Ok((c, false)),
                            Err(e) => {
                                tracing::error!(
                                    manifest = %name,
                                    error = %e,
                                    "CURRENT manifest unreadable — falling back to previous generation"
                                );
                            }
                        }
                    }
                }
            }
        }

        // 2) Fall back to the highest valid generation.
        let mut gens: Vec<u64> = Vec::new();
        for e in std::fs::read_dir(&manifest_dir)? {
            let p = e?.path();
            if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                if let Some(num) = stem.strip_prefix("manifest-") {
                    if let Ok(g) = num.parse::<u64>() {
                        gens.push(g);
                    }
                }
            }
        }
        gens.sort_unstable_by(|a, b| b.cmp(a));
        for g in gens {
            let path = manifest_dir.join(format!("manifest-{:09}", g));
            if let Ok(bytes) = std::fs::read(&path) {
                if let Ok(c) = Self::decode(&bytes, &path.display().to_string()) {
                    tracing::warn!(
                        generation = g,
                        "recovered catalog from previous manifest generation"
                    );
                    return Ok((c, true));
                }
            }
        }

        Err(StorageError::corruption(
            manifest_dir.display().to_string(),
            "manifest files exist but none are valid; refusing to start over existing data",
        ))
    }
}

/// fsync a directory (best-effort no-op on platforms that don't support it).
pub fn fsync_dir(dir: &Path) -> Result<(), StorageError> {
    #[cfg(unix)]
    {
        let f = File::open(dir)?;
        f.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

/// Snapshot of the IdMapper state (uuid↔numeric-id bijection + allocator + retired ids).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IdMapSnapshot {
    pub format_version: u32,
    pub next_id: u64,
    /// (uuid string, numeric id) pairs — strings keep the format stable
    pub mappings: Vec<(String, u64)>,
    /// retired (deleted/updated-away) numeric ids; never reused
    pub retired: Vec<u64>,
}

impl IdMapSnapshot {
    pub fn save(&self, db_dir: &Path) -> Result<(), StorageError> {
        let meta_dir = Catalog::meta_dir(db_dir);
        std::fs::create_dir_all(&meta_dir)?;
        let mut body =
            bincode::serialize(self).map_err(|e| StorageError::Serialization(e.to_string()))?;
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&body);
        let crc = hasher.finalize();

        let mut bytes = Vec::with_capacity(body.len() + 16);
        bytes.extend_from_slice(&IDMAP_MAGIC.to_be_bytes());
        bytes.extend_from_slice(&IDMAP_FORMAT_VERSION.to_be_bytes());
        bytes.extend_from_slice(&(body.len() as u32).to_be_bytes());
        bytes.append(&mut body);
        bytes.extend_from_slice(&crc.to_be_bytes());

        let final_path = meta_dir.join("idmap.bin");
        let tmp_path = meta_dir.join("idmap.bin.tmp");
        {
            let mut f = File::create(&tmp_path)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp_path, &final_path)?;
        fsync_dir(&meta_dir)?;
        Ok(())
    }

    /// `Ok(None)` = no snapshot (fresh database).
    pub fn load(db_dir: &Path) -> Result<Option<IdMapSnapshot>, StorageError> {
        let path = Catalog::meta_dir(db_dir).join("idmap.bin");
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path)?;
        let file = path.display().to_string();
        if bytes.len() < 16 {
            return Err(StorageError::corruption(file, "idmap too short"));
        }
        let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if magic != IDMAP_MAGIC {
            return Err(StorageError::corruption(file, "bad idmap magic"));
        }
        let version = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != IDMAP_FORMAT_VERSION {
            return Err(StorageError::UnsupportedVersion {
                file,
                found: version,
                supported: IDMAP_FORMAT_VERSION,
            });
        }
        let len = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        if bytes.len() < 12 + len + 4 {
            return Err(StorageError::corruption(file, "idmap truncated"));
        }
        let body = &bytes[12..12 + len];
        let stored_crc = u32::from_be_bytes([
            bytes[12 + len],
            bytes[13 + len],
            bytes[14 + len],
            bytes[15 + len],
        ]);
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(body);
        if hasher.finalize() != stored_crc {
            return Err(StorageError::ChecksumMismatch { file });
        }
        let snap: IdMapSnapshot = bincode::deserialize(body)
            .map_err(|e| StorageError::corruption(file, format!("body: {e}")))?;
        Ok(Some(snap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut cat = Catalog::fresh();
        cat.collections.push(CollectionMeta {
            id: 1,
            name: "papers".into(),
            dim: 128,
            heads: vec![HeadMeta {
                name: "semantic".into(),
                settings_json: String::new(),
            }],
            settings_json: "{\"ef_search\":128}".into(),
            schema_fields: vec![("title".into(), "TEXT".into())],
            created_seq: 3,
            dropped: false,
        });
        cat.next_collection_id = 2;
        cat.next_doc_numeric_id = 42;
        cat.checkpoint_seq = 7;
        let gen = cat.save(dir.path()).unwrap();
        assert_eq!(gen, 1);

        let (loaded, fallback) = Catalog::load(dir.path()).unwrap();
        assert!(!fallback);
        assert_eq!(loaded, cat);
    }

    #[test]
    fn current_corruption_falls_back_to_previous_generation() {
        let dir = tempfile::tempdir().unwrap();
        let mut cat = Catalog::fresh();
        cat.checkpoint_seq = 1;
        cat.save(dir.path()).unwrap();

        let mut cat2 = cat.clone();
        cat2.checkpoint_seq = 2;
        cat2.next_doc_numeric_id = 99;
        let gen2 = cat2.save(dir.path()).unwrap();
        assert_eq!(gen2, 2);

        // Corrupt the generation-2 manifest and point CURRENT at it.
        std::fs::write(
            Catalog::manifest_dir(dir.path()).join("manifest-000000002"),
            b"garbage garbage garbage",
        )
        .unwrap();
        std::fs::write(Catalog::current_path(dir.path()), "manifest-000000002\n").unwrap();

        let (loaded, fallback) = Catalog::load(dir.path()).unwrap();
        assert!(fallback);
        // Generation 1 state: checkpoint_seq=1, default next_doc_numeric_id=1.
        assert_eq!(loaded.checkpoint_seq, 1);
        assert_eq!(loaded.next_doc_numeric_id, 1);
    }

    #[test]
    fn manifest_dir_with_only_garbage_is_fatal() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(Catalog::manifest_dir(dir.path())).unwrap();
        std::fs::write(
            Catalog::manifest_dir(dir.path()).join("manifest-000000001"),
            b"total garbage",
        )
        .unwrap();
        assert!(Catalog::load(dir.path()).is_err());
    }

    #[test]
    fn empty_db_loads_fresh() {
        let dir = tempfile::tempdir().unwrap();
        let (cat, fallback) = Catalog::load(dir.path()).unwrap();
        assert!(!fallback);
        assert_eq!(cat.next_doc_numeric_id, 1);
    }

    #[test]
    fn idmap_snapshot_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let snap = IdMapSnapshot {
            format_version: IDMAP_FORMAT_VERSION,
            next_id: 5,
            mappings: vec![("550e8400-e29b-41d4-a716-446655440000".into(), 1)],
            retired: vec![2, 3],
        };
        snap.save(dir.path()).unwrap();
        let loaded = IdMapSnapshot::load(dir.path()).unwrap().unwrap();
        assert_eq!(loaded.next_id, 5);
        assert_eq!(loaded.retired, vec![2, 3]);
        assert_eq!(loaded.mappings.len(), 1);
    }
}
