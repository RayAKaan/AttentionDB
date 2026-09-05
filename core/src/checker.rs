//! Consistency checker — validates the invariants in docs/consistency-model.md
//! and reports actionable, coded issues (never just "database inconsistent").
//!
//! Two entry points:
//! - [`check_engine`] — live engine (used at startup during VALIDATING, and by
//!   the admin check endpoints).
//! - [`check_db_dir`] — offline check over a database directory (CLI
//!   `attentiondb check <dir>`); replays the WAL in memory without mutating.

use crate::engine::AttentionEngine;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl Severity {
    pub fn is_error(&self) -> bool {
        matches!(self, Severity::Error)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckIssue {
    pub severity: Severity,
    /// stable machine-readable code, e.g. INDEX_ORPHAN_VECTOR
    pub code: &'static str,
    pub collection: String,
    pub detail: String,
}

impl CheckIssue {
    fn error(code: &'static str, collection: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            code,
            collection: collection.into(),
            detail: detail.into(),
        }
    }

    fn warning(
        code: &'static str,
        collection: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            severity: Severity::Warning,
            code,
            collection: collection.into(),
            detail: detail.into(),
        }
    }
}

/// Validate a live engine's invariants. Read-only.
pub fn check_engine(engine: &AttentionEngine) -> Vec<CheckIssue> {
    let mut issues = Vec::new();
    let mapper = engine.id_mapper.read();
    let store = engine.document_store.read();

    // --- ID mapping invariants (INV-6) ------------------------------------
    let next_id = mapper.next_id();
    for (uuid, id) in mapper_snapshot_pairs(engine) {
        if id >= next_id {
            issues.push(CheckIssue::error(
                "MAPPING_ID_OUT_OF_RANGE",
                "-",
                format!("uuid {uuid} maps to id {id} but next_id is {next_id}"),
            ));
        }
        if mapper.is_retired(id) && mapper.uuid_to_id(&uuid).is_some() {
            issues.push(CheckIssue::error(
                "DUPLICATE_MAPPING",
                "-",
                format!("id {id} is retired but still live-mapped to {uuid}"),
            ));
        }
    }

    // --- Document-level invariants -----------------------------------------
    let live_records = store.list_all_records();
    for rec in &live_records {
        let Some(numeric) = mapper.uuid_to_id(&rec.id) else {
            issues.push(CheckIssue::error(
                "MISSING_MAPPING",
                collection_of(rec).unwrap_or("-"),
                format!("document {} has no id mapping (orphan record)", rec.id),
            ));
            continue;
        };
        if mapper.is_retired(numeric) {
            issues.push(CheckIssue::error(
                "MAPPING_RETIRED_LIVE_DOC",
                collection_of(rec).unwrap_or("-"),
                format!("document {} maps to retired id {}", rec.id, numeric),
            ));
        }
        // Vector dimension sanity against its collection(s).
        for tag in &rec.tags {
            if let Some(coll_name) = tag.strip_prefix("collection:") {
                if let Ok(coll) = engine.get_collection(coll_name) {
                    for (head, vec) in &rec.k_vecs {
                        if vec.len() != coll.dim {
                            issues.push(CheckIssue::error(
                                "DIMENSION_MISMATCH",
                                coll_name,
                                format!(
                                    "document {} head '{head}' dim {} != collection dim {}",
                                    rec.id,
                                    vec.len(),
                                    coll.dim
                                ),
                            ));
                        }
                    }
                    if !coll.list_heads().iter().any(|h| rec.k_vecs.contains_key(h))
                        && rec.k_vecs.is_empty()
                    {
                        issues.push(CheckIssue::warning(
                            "DOC_WITHOUT_VECTORS",
                            coll_name,
                            format!("document {} has no vectors in any head", rec.id),
                        ));
                    }
                } else {
                    issues.push(CheckIssue::warning(
                        "INVALID_COLLECTION_TAG",
                        coll_name,
                        format!("document {} tagged for unknown collection", rec.id),
                    ));
                }
            }
        }
    }

    // --- Index ⇄ document invariants (INV-1/INV-2) --------------------------
    let cols = engine.collections.read();
    for (name, coll) in cols.iter() {
        let retired = coll.retired_ids.read();
        let live_ids: std::collections::HashSet<u64> = live_records
            .iter()
            .filter(|r| r.tags.contains(&format!("collection:{name}")))
            .filter_map(|r| mapper.uuid_to_id(&r.id))
            .collect();

        for head in coll.list_heads() {
            let idx = match coll.head_manager.read().get_head(&head) {
                Ok(i) => i,
                Err(e) => {
                    issues.push(CheckIssue::error(
                        "HEAD_MISSING",
                        name,
                        format!("head '{head}' declared but missing: {e}"),
                    ));
                    continue;
                }
            };
            let guard = idx.read();
            for (vid, _) in guard.vector_pairs() {
                if retired.contains(vid) {
                    // stale entry awaiting purge — warning only (filtered at search)
                    issues.push(CheckIssue::warning(
                        "INDEX_RETIRED_VECTOR",
                        name,
                        format!("head '{head}' vector id {vid} is retired (awaiting purge)"),
                    ));
                } else if !live_ids.contains(vid) {
                    issues.push(CheckIssue::error(
                        "INDEX_ORPHAN_VECTOR",
                        name,
                        format!(
                            "head '{head}' vector id {vid} has no live document (reason=document_missing)"
                        ),
                    ));
                }
            }
        }

        // Documents that should be searchable but have no index entry (INV-2).
        for rec in &live_records {
            if !rec.tags.contains(&format!("collection:{name}")) {
                continue;
            }
            let Some(numeric) = mapper.uuid_to_id(&rec.id) else {
                continue;
            };
            for head in rec.k_vecs.keys() {
                if let Ok(idx) = coll.head_manager.read().get_head(head) {
                    if !idx.read().contains_id(numeric) {
                        issues.push(CheckIssue::error(
                            "DOC_MISSING_INDEX_ENTRY",
                            name,
                            format!(
                                "document {} (id {numeric}) has vector for head '{head}' but no index entry",
                                rec.id
                            ),
                        ));
                    }
                }
            }
        }
    }

    issues
}

fn collection_of(rec: &attentiondb_storage::Record) -> Option<&str> {
    rec.tags.iter().find_map(|t| t.strip_prefix("collection:"))
}

/// Snapshot (uuid, id) pairs without exposing IdMapper internals.
fn mapper_snapshot_pairs(engine: &AttentionEngine) -> Vec<(uuid::Uuid, u64)> {
    let mapper = engine.id_mapper.read();
    mapper
        .snapshot()
        .mappings
        .iter()
        .filter_map(|(s, i)| s.parse::<uuid::Uuid>().ok().map(|u| (u, *i)))
        .collect()
}

/// Offline check of a database directory. Loads catalog, replays WAL in memory
/// (no mutation), loads SSTables, and reports structural problems. Returns
/// issues; an empty list means the directory passed all checks.
pub fn check_db_dir(db_dir: &std::path::Path) -> Result<Vec<CheckIssue>, crate::error::CoreError> {
    let mut issues = Vec::new();

    // Catalog.
    let catalog = match attentiondb_storage::Catalog::load(db_dir) {
        Ok((c, fallback)) => {
            if fallback {
                issues.push(CheckIssue::warning(
                    "MANIFEST_FALLBACK",
                    "-",
                    "CURRENT was unusable; catalog recovered from an older generation",
                ));
            }
            c
        }
        Err(e) => {
            return Ok(vec![CheckIssue::error(
                "CATALOG_UNREADABLE",
                "-",
                e.to_string(),
            )]);
        }
    };

    // WAL: full replay from zero via a throwaway engine open is the strongest
    // check; here we do a cheap structural pass (segments parse + sequences).
    let wal_dir = db_dir.join(attentiondb_storage::WAL_DIR_NAME);
    if wal_dir.exists() {
        let mut wal = match attentiondb_storage::Wal::open(
            &wal_dir,
            attentiondb_storage::Durability::Async,
            u64::MAX,
        ) {
            Ok(w) => w,
            Err(e) => {
                return Ok(vec![CheckIssue::error(
                    "WAL_UNOPENABLE",
                    "-",
                    e.to_string(),
                )]);
            }
        };
        match wal.replay(0) {
            Ok(outcome) => {
                if outcome.torn_tail {
                    issues.push(CheckIssue::warning(
                        "WAL_TORN_TAIL",
                        "-",
                        format!(
                            "last sequence {} had a torn (unacknowledged) tail",
                            outcome.last_seq
                        ),
                    ));
                }
                if outcome.last_seq < catalog.checkpoint_seq {
                    issues.push(CheckIssue::error(
                        "WAL_SEQ_INVALID",
                        "-",
                        format!(
                            "checkpoint claims seq {} but WAL ends at {} (segments trimmed too far?)",
                            catalog.checkpoint_seq, outcome.last_seq
                        ),
                    ));
                }
            }
            Err(e) => {
                return Ok(vec![CheckIssue::error("WAL_CORRUPT", "-", e.to_string())]);
            }
        }
    } else if catalog.checkpoint_seq > 0 {
        issues.push(CheckIssue::error(
            "WAL_MISSING",
            "-",
            "catalog references WAL state but the WAL directory is absent",
        ));
    }

    // SSTables: open each and validate readability (errors are already strict).
    let sst_dir = attentiondb_storage::Catalog::sst_dir(db_dir);
    if sst_dir.exists() {
        for e in std::fs::read_dir(&sst_dir)?.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("sst") {
                if let Err(err) = attentiondb_storage::SSTableReader::open(&p) {
                    issues.push(CheckIssue::error(
                        "SST_CORRUPT",
                        "-",
                        format!("{}: {err}", p.display()),
                    ));
                }
            }
        }
    }

    // Idmap snapshot.
    if let Err(e) = attentiondb_storage::IdMapSnapshot::load(db_dir) {
        issues.push(CheckIssue::error("IDMAP_CORRUPT", "-", e.to_string()));
    }

    // Full engine-level check (recovery-equivalent, in-memory).
    let engine =
        crate::engine::AttentionEngine::open_dir(db_dir, attentiondb_storage::Durability::Async)?;
    issues.extend(check_engine(&engine));

    Ok(issues)
}
