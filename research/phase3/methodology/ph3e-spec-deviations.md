# Phase 3E — Specification Deviations (E2)

Scope: E2 (Durability Semantics & Acknowledgment Contract) only. Companion to
`production-contract.md` amendment A2 and `phase3e-e2-final-report.md`.

## 1. Crash-gate instrumentation touches production code

Exact ACK-boundary injection (between fsync and ack, between WAL append and apply) is
impossible to aim at from outside a process. E2 adds `storage/src/crashgate.rs` and nine
one-line gate hits inside `Wal::append` and the engine mutation paths
(`insert_document` / `delete_document` / `commit_transaction`).

- When `PH3E_CRASH_AT` is unset (all production runs) the cost is one atomic load
  against a `OnceLock`-parsed config; no behavior change.
- Gates fire via `std::process::abort()` (SIGABRT): no destructors, no flushes — the
  same sudden-death model as SIGKILL, deliverable at points an external killer cannot
  target.
- This is the "smallest necessary change" under the E2 scope rule ("crash-test
  harnesses required specifically for E2"). Regression evidence that gates change
  nothing: PH3E-WAL-002 (byte-identical to PH3E-WAL-001) and PH3E-REG-002 (all 3D
  families clean).

## 2. Gate-hit arithmetic

WAL-level gates (`after_write` / `after_flush` / `after_fsync`) fire on EVERY
`Wal::append`, including the `CreateCollection` record. Engine-level gates
(`before_wal_append` / `after_wal_append` / `after_apply` / `before_ack`) fire only on
insert/delete/txn-commit call sites. The harness computes hit numbers from the
deterministic record sequence:

- ack suite: engine gates hit 11 (10 baseline + target); WAL gates hit 12
  (collection + 10 baseline + target).
- txn suite: engine gates hit 6 (5 baseline + the commit call); WAL gate
  `after_write`: hit 12 = BEGIN + 4 ops written (mid-ops), hit 18 = COMMIT frame
  written.

## 3. NOT_REACHED cells

`after_flush` exists only inside the GroupCommit branch of `Wal::append`;
`after_fsync` only inside the Sync branch. In other modes the gate is unreachable. The
parent records these cells as `NOT_REACHED` (the §11 vocabulary "NOT IMPLEMENTED /
NOT TESTED" class) instead of spawning: the child's fallback abort would fabricate an
`after_ack`-like observation. 6 of 81 ack-boundary cells are NOT_REACHED.

## 4. Transaction all-or-nothing is judged on txn inserts

A crash cell where the transaction is ABSENT and the two baseline documents it would
have deleted were THEMSELVES lost (Async buffering) leaves "deletes applied" vacuously
true. Calling that PARTIAL would be a harness artifact, not a database fact. The
all-or-nothing invariant is therefore judged on the transaction's INSERT records
(0 or 8, never between); the delete outcome is recorded as a separate fact column and
interpreted only when the deleted baseline documents survived. Observed PARTIAL
count: 0 of 63 cells.

## 5. Async baseline survival is an artifact, not a guarantee

In the txn suite, Async cells show 5/5 baseline survival for gates after the txn ops —
an artifact of the 8 KiB `BufWriter` filling during the txn's op appends
(auto-flushing earlier records to the page cache). The contract (A2) explicitly states
such survival MUST NOT be relied upon; expectations for Async baseline counts are
therefore "ANY" in the generator. The ack-boundary suite (smaller workload) shows the
same records lost 0-survival — both facts preserved.

## 6. `flush_wal()` is flush-only

The public `flush_wal()` flushes userspace buffers to the OS page cache and does NOT
fsync (pre-existing behavior, unchanged). After `flush_wal()`, data survives process
crash but not machine crash — the GroupCommit per-append boundary. The harness's
`explicit_flush` cell (Async + flush_wal → abort) demonstrates exactly this boundary.

## 7. Graceful drop is a durability point (but not a guarantee)

`Wal` has no `Drop` impl; Rust's `BufWriter` flushes on drop, so a graceful process
exit moves Async userspace buffers to the page cache. SIGKILL/SIGABRT skip drops
entirely. The E2 experiments use abort/kill only; graceful-exit behavior is documented
here and in A2 but is NOT part of any crash guarantee.

## 8. GroupCommit naming retained despite semantics

"GroupCommit" suggests coalesced group fsync across writers. Actual semantics: the
engine-wide mutation gate serializes ALL mutations and checkpoints, so appends never
overlap; each append flushes to the page cache before returning — there are no groups.
The name is kept for API compatibility (renaming would be an API break out of E2's
minimal-change scope); A2 documents the real meaning. The group-boundary suite
(PH3E-DUR-004) verifies the multi-writer ack invariant on the real implementation.

## 9. Crash points not instrumented

The §8 list's "before WAL append" is instrumented at the engine call sites
(`before_wal_append`), which is AFTER validation and id registration but BEFORE any
WAL append — the nearest meaningful point (aborting during validation would not test
durability). No gate exists inside fsync itself (mid-syscall): `after_write` /
`after_flush` / `after_fsync` bracket it on both sides; a mid-`sync_all` kill is
equivalent to "fsync never returned" — covered logically by the `after_write`/`after_flush`
cells (data not yet fsynced → Sync-mode loss possible for the un-acked record only;
acked records always have `sync_all` returned before the ack, verified by `after_ack`).

## 10. Latency environment

PH3E-DUR-005 latency/throughput figures are measured on the sandbox overlay filesystem
where fsync is inexpensive. They are used ONLY for the relative default-mode decision
(A2) and are explicitly not production-comparable.

## 11. Out-of-scope notes

- The api crate remains unbuildable (protoc unavailable; established in E0/E1) — the
  A2 "no API change" decision also avoids depending on it.
- Machine/power-loss durability: NOT VERIFIED anywhere in E2 (prompt §16); all such
  wording in artifacts is prospective ("machine boundary"), never a claim.

## 12. Pre-existing stale integration tests found by the E2 full-suite sweep

The §19 requirement (`cargo test`/clippy over everything, `--all-targets`) exposed
three latent breakages that earlier phases never caught because their sweeps only read
the FIRST test-result block (lib tests). Proven pre-existing: all fail identically at
clean HEAD 615407c with the E2 changes stashed. Fixes (test infrastructure only; no
engine behavior changed):

- `core/tests/phase1_compaction.rs` t11: passed the sst dir to `compact_all`, which
  takes the DATABASE ROOT since the earlier subdir-resolution fix → NotFound. Test
  updated to the current documented API.
- `core/tests/golden_lifecycle.rs`: same stale call, previously silenced by
  `let _ =` (compaction silently never ran). Updated to the current API; asserts the
  call succeeds (a post-checkpoint merge is not required — `None` is legitimate).
- query `filter.rs` + storage test lints (`is_multiple_of`, duplicate `#[test]`,
  `match`→`if let`): mechanical all-targets clippy fixes, behavior identical.

## 13. phase1_recovery t13/t14: expectations updated to the E1 contract

Both tests' expectations predate the E1 refusal semantics (they were never re-run in
full after E1 landed — same sweep gap as §12):

- t14 picked the segment as the first non-zero file from an UNSORTED `read_dir(WAL)`,
  which can be `wal-state.json`; appending torn-frame bytes to the integrity sidecar
  now (correctly) refuses with WAL_STATE_CORRUPT. Fixed to select a sorted `.wal`
  segment; the torn-tail-then-repair scenario is unchanged.
- t13 window (b) corrupted the CURRENT manifest and expected the gen-1 fallback to
  open. Under E1 this REFUSES (WAL_SEQ_GAP): the fallback generation predates the
  checkpoint, the seqs that would reconnect the surviving WAL exist only in SSTables
  the destroyed manifest described — opening would replay onto an unprovable base.
  The test now asserts the refusal AND adds the legitimate designed fallback
  (CURRENT write torn, generations intact, fallback generation CONNECTED to the
  surviving WAL → opens, state correct). This is a documented E1 consequence, not a
  regression: two-generation manifest recovery covers a torn CURRENT write, not
  destruction of the newest manifest across a checkpoint+trim boundary.

## 14. PH3E-DUR-003 superseded by PH3E-DUR-006 (restart cycles)

The clippy all-targets pass caught a harness defect: `e2_recover`'s restart loop
contained an unconditional `break`, so PH3E-DUR-003 recorded ONE restart cycle while
the report claimed two. Raw runs are immutable: PH3E-DUR-003 is preserved unchanged
(its single-cycle facts are valid) and PH3E-DUR-006 re-runs the identical suite with
the corrected two-cycle loop (close→open→state-equality→checker, twice per cell).
results/durability-checkpoint.csv is generated from PH3E-DUR-006. All 27 cells:
restarts_equal=true and checker clean across both cycles. (The remaining clippy fix
in the harness, `.chain(target_idx)`, is compiler-verified semantically neutral and
does not affect the recorded PH3E-DUR-001/002/004/005 facts.)
