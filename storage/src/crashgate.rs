//! E2 durability testing: deterministic in-process crash gates.
//!
//! These gates exist ONLY to let the Phase 3E E2 experiment family abort the
//! process at an exact point of the commit path (between fsync and ack, between
//! WAL append and in-memory apply, ...). They are instrumentation, not
//! behavior: when `PH3E_CRASH_AT` is unset (every production run) the check is
//! one atomic load against a `OnceLock` and the gates never fire.
//!
//! Contract:
//!   PH3E_CRASH_AT  = gate name (see the statics below)
//!   PH3E_CRASH_HIT = 1-based hit number of THAT gate to abort at
//!
//! `std::process::abort()` raises SIGABRT: no destructors run, no buffers are
//! flushed — the same sudden-death model as an external SIGKILL, deliverable
//! at a point no external killer can aim at.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Gate hit inside `Wal::append`, immediately after the frame bytes entered the
/// BufWriter (userspace buffer) and before any flush/fsync of this append.
pub static GATE_AFTER_WRITE: Gate = Gate::new("after_write");
/// Gate hit inside `Wal::append` (GroupCommit mode) after `flush()` returned —
/// frame bytes are in the OS page cache, not yet fsynced.
pub static GATE_AFTER_FLUSH: Gate = Gate::new("after_flush");
/// Gate hit inside `Wal::append` (Sync mode) after `sync_all()` returned —
/// frame bytes are fsynced.
pub static GATE_AFTER_FSYNC: Gate = Gate::new("after_fsync");

/// Engine gate: entered the mutation path, before any WAL append of this
/// operation (insert/delete call sites and the txn COMMIT sequence).
pub static GATE_BEFORE_WAL_APPEND: Gate = Gate::new("before_wal_append");
/// Engine gate: the operation's last WAL append returned (insert: the record;
/// txn: the COMMIT record) — mode durability action complete.
pub static GATE_AFTER_WAL_APPEND: Gate = Gate::new("after_wal_append");
/// Engine gate: in-memory apply of the operation completed (before ack).
pub static GATE_AFTER_APPLY: Gate = Gate::new("after_apply");
/// Engine gate: immediately before returning Ok to the caller (last instant
/// before ACK).
pub static GATE_BEFORE_ACK: Gate = Gate::new("before_ack");

// ---- E3 structural windows ----
/// Wal::rotate: the COMPLETED segment was flushed + fsynced, the new segment
/// does not exist yet (the in-flight frame has NOT been written — rotation
/// precedes the frame write in Wal::append).
pub static GATE_ROTATE_AFTER_OLD_FSYNC: Gate = Gate::new("rotate_after_old_fsync");
/// Wal::open_segment: the durable watermark record (wal-state.json) was
/// written for the newly created segment.
pub static GATE_ROTATE_AFTER_STATE_WRITE: Gate = Gate::new("rotate_after_state_write");
/// checkpoint_locked: WAL fsynced (everything so far machine-boundary durable
/// in ALL modes), nothing else done yet.
pub static GATE_CKPT_AFTER_WAL_FSYNC: Gate = Gate::new("ckpt_after_wal_fsync");
/// checkpoint_locked: memtable flushed to SSTable(s) (written at FINAL names),
/// manifest not yet updated.
pub static GATE_CKPT_AFTER_SST: Gate = Gate::new("ckpt_after_sst");
/// checkpoint_locked: idmap snapshot saved, manifest not yet updated.
pub static GATE_CKPT_AFTER_IDMAP: Gate = Gate::new("ckpt_after_idmap");
/// Catalog::save: manifest-N.tmp fully written + sync_all'd, not renamed.
pub static GATE_MANIFEST_AFTER_TMP_WRITE: Gate = Gate::new("manifest_after_tmp_write");
/// Catalog::save: manifest-N installed + MANIFEST dir fsynced; CURRENT still old.
pub static GATE_MANIFEST_AFTER_MANIFEST_DIRSYNC: Gate =
    Gate::new("manifest_after_manifest_dirsync");
/// Catalog::save: CURRENT.tmp written + sync_all'd; CURRENT still old.
pub static GATE_MANIFEST_AFTER_CURRENT_TMP_WRITE: Gate =
    Gate::new("manifest_after_current_tmp_write");
/// Catalog::save: CURRENT renamed + db dir fsynced — manifest replacement
/// complete.
pub static GATE_MANIFEST_AFTER_CURRENT_RENAME: Gate = Gate::new("manifest_after_current_rename");
/// DocumentStore::flush_memtable: the SSTable was fully written (final name),
/// not yet installed into the reader set.
pub static GATE_SST_AFTER_WRITE: Gate = Gate::new("sst_after_write");
/// checkpoint_locked: manifest installed AND WAL rotated; trim not yet run.
pub static GATE_CKPT_AFTER_ROTATE: Gate = Gate::new("ckpt_after_rotate");
/// checkpoint_locked: trim done — checkpoint fully complete except manifest
/// generation cleanup.
pub static GATE_CKPT_AFTER_TRIM: Gate = Gate::new("ckpt_after_trim");

// ---- E6 transaction commit windows (Engine::commit_transaction) ----
/// COMMIT record not yet appended: WAL holds BEGIN+ops of this txn (the
/// partial-WAL window); a crash here must recover as "transaction never
/// happened". (Staged-but-uncommitted transactions have NO WAL presence at
/// all: staging is in-memory only, so "crash before commit" at the API level
/// needs no gate.)
pub static GATE_TX_BEFORE_COMMIT_WAL: Gate = Gate::new("tx_before_commit_wal");
/// COMMIT record appended (mode durability performed synchronously inside
/// append: Sync=fsync, Group=flush, Async=buffered) and returned; in-memory
/// apply has NOT started. In this implementation the append call IS the
/// persistence step, so a distinct post-fsync gate would be the same code
/// point (documented as coincident; not separately instrumented).
pub static GATE_TX_AFTER_COMMIT_WAL: Gate = Gate::new("tx_after_commit_wal");

// ---- E5 compaction windows (Engine::compact_storage — coordinated path) ----
/// Memtable flushed (flush SST at final name); merge not started.
pub static GATE_COMPACT_BEFORE_MERGE: Gate = Gate::new("compact_before_merge");
/// Merged output SST fsynced + renamed at its final name; inputs still present;
/// reader list still the pre-compaction set.
pub static GATE_COMPACT_AFTER_OUTPUT: Gate = Gate::new("compact_after_output");
/// Input files unlinked (reader-safe: readers materialize entries in memory);
/// reader list still the pre-compaction set.
pub static GATE_COMPACT_AFTER_CLEANUP: Gate = Gate::new("compact_after_cleanup");
/// Reader list swapped to the post-compaction (output-only) set — fully complete.
pub static GATE_COMPACT_AFTER_INSTALL: Gate = Gate::new("compact_after_install");

pub struct Gate {
    name: &'static str,
    hit: AtomicUsize,
}

impl Gate {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            hit: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn hit(&self) {
        if let Some((want, n)) = crash_cfg() {
            if want == self.name {
                let h = self.hit.fetch_add(1, Ordering::SeqCst) + 1;
                if h == n {
                    match crash_model() {
                        // F2-equivalent: announce the window durably, then park.
                        // The controller SIGKILLs the whole process group —
                        // no destructors, no flushes, nothing cleans up.
                        "groupkill" => {
                            if let Some(m) = std::env::var_os("PH3E_CRASH_MARKER") {
                                use std::io::Write;
                                if let Ok(mut f) = std::fs::OpenOptions::new()
                                    .create(true)
                                    .write(true)
                                    .truncate(true)
                                    .open(m)
                                {
                                    let _ = writeln!(f, "{} hit {}", self.name, h);
                                    let _ = f.sync_all();
                                }
                            }
                            loop {
                                std::thread::sleep(std::time::Duration::from_secs(3600));
                            }
                        }
                        // F1: sudden death, like SIGKILL, nothing flushed/dropped.
                        _ => std::process::abort(),
                    }
                }
            }
        }
    }
}

/// Crash model: "abort" (F1, default) or "groupkill" (F2-equivalent: park at
/// the window; the controller kills the whole process group).
fn crash_model() -> &'static str {
    static M: OnceLock<&'static str> = OnceLock::new();
    M.get_or_init(|| match std::env::var("PH3E_CRASH_MODEL").as_deref() {
        Ok("groupkill") => "groupkill",
        _ => "abort",
    })
}

/// Parse (`gate_name`, `hit_number`) once per process; `None` disables all gates.
fn crash_cfg() -> Option<(&'static str, usize)> {
    static CFG: OnceLock<Option<(&'static str, usize)>> = OnceLock::new();
    *CFG.get_or_init(|| {
        let name = std::env::var("PH3E_CRASH_AT").ok()?;
        let n: usize = std::env::var("PH3E_CRASH_HIT").ok()?.parse().ok()?;
        const NAMES: &[&str] = &[
            "after_write",
            "after_flush",
            "after_fsync",
            "before_wal_append",
            "after_wal_append",
            "after_apply",
            "before_ack",
            // E3 structural windows
            "rotate_after_old_fsync",
            "rotate_after_state_write",
            "ckpt_after_wal_fsync",
            "ckpt_after_sst",
            "ckpt_after_idmap",
            "manifest_after_tmp_write",
            "manifest_after_manifest_dirsync",
            "manifest_after_current_tmp_write",
            "manifest_after_current_rename",
            "sst_after_write",
            "ckpt_after_rotate",
            "ckpt_after_trim",
            // E5 compaction windows
            "compact_before_merge",
            "compact_after_output",
            "compact_after_install",
            "compact_after_cleanup",
            // E6 transaction commit windows
            "tx_before_commit_wal",
            "tx_after_commit_wal",
        ];
        let name = NAMES.iter().find(|c| **c == name.as_str())?;
        Some((*name, n.max(1)))
    })
}
