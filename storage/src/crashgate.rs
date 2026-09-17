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
                    // Sudden death: like SIGKILL, nothing is flushed or dropped.
                    std::process::abort();
                }
            }
        }
    }
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
        ];
        let name = NAMES.iter().find(|c| **c == name.as_str())?;
        Some((*name, n.max(1)))
    })
}
