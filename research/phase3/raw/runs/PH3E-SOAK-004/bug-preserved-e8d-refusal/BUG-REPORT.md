# ENGINE DEFECT #2 (E8 soak find) — tombstone shadowing via ghost SST during recovery replay

**Status:** FIXED + regression-tested. Discovered during E8d smoke (pre-official), 2026-09-18.

## Symptom
E8d smoke panicked at the phase-3 reopen: `RecoveryFailed("18 consistency error(s) detected
after recovery — refusing to become READY")` — 18 MISSING_MAPPING orphans, deterministic
(identical uuid sets across runs). This directory preserves the failing state (`db/` =
final SST + idmap + WAL; `rundir-full/` = complete run artifacts).

## Forensics (instrumented builds, since removed)
- idmap.bin: exactly 1000 live mappings (correct). The 18 orphan records are STALE
  RETIRED versions (e.g. key 626: live v37, leaked record v5) inside the final compacted SST.
- Backtrace proof: `apply_insert ← insert_nolog ← apply_replay_op ← open_dir ←
  check_db_dir ← checker_report ← e8_verify` — the mid-run verify's dir-check opened a
  SECOND engine on the LIVE directory; its WAL replay crossed the memtable threshold
  (1000) and the threshold auto-flush wrote a "ghost" SST of partial replay state into
  the live dir. Replayed prefix state re-materialized already-deleted records (inserts
  whose deletes sit later in the WAL) stamped with NEWER timestamps than their real
  tombstones → the next full compaction's tombstone GC could not kill them → the next
  recovery resolved the ghost as the winning version → unmapped record → refusal.
- E7 never hit this: its WALs were too small to cross the threshold during replay, and
  its checker ran on quiescent dirs. E8d's 52k-op WAL + mid-run verifies exposed it.

## Fixes (engine, minimal)
1. `DocumentStore::set_auto_flush(false)` around the WAL replay loop in
   `AttentionEngine::open_dir` (INV-RECOVERY-READONLY: recovery never emits SSTables;
   open persists nothing; the over-threshold memtable flushes on the next live mutation
   or owner-initiated checkpoint).
2. Logically monotonic flush entry timestamps (`last_entry_ts`, INV-C2 corollary):
   `entry_ts = max(now_millis, last+1)` per flush — same-millisecond flushes can no
   longer tie on (ts, file-order), where the lexical `compacted_* < sstable_*` order
   could invert chronology.

## Regression tests (all green)
- core/tests/regression_recovery_no_sst_writes.rs — recovery writes no SSTs; deleted doc stays dead.
- core/tests/regression_tombstone_shadow_leak.rs — engine-level minimal leak repro.
- storage/tests/regression_flush_ts_monotonic.rs — consecutive flushes strictly increase ts; INV-C2 ts-first resolution.
- Full suite: 328 passed / 0 failed (oracle 324 + 4 new). Clippy -D warnings clean.
