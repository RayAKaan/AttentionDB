# Phase 3E / E4 — Deviations, Boundary Notes & Post-run Corrections

Experiment: **PH3E-BACKUP-004** (supersedes PH3E-BACKUP-003) — online backup / snapshot
consistency / restore integrity. Date: 2026-09-17. All deviations are documented
addenda; no recorded raw data was altered.

## D1. PH3E-BACKUP-003 superseded by PH3E-BACKUP-004 (harness column shift)

The first recorded run (PH3E-BACKUP-003) had a **harness formatting defect** in two
matrix rows: the `snapshot-transition` and `multi-backup` row templates misplaced the
`checker_clean` field one column left (a stray literal `-` consumed the `backup_us` /
`restore_us` slots). The backup/restore behavior in those rows was correct (state
matched the reference model; the checker ran clean inside the row logic), but the raw
CSV was ambiguous. Per the raw-immutability rule, PH3E-BACKUP-003 is retained untouched
(marked SUPERSEDED in the registry) and the run was repeated as **PH3E-BACKUP-004**
with the corrected templates and an explicit `all_clean` accumulator for the
multi-backup row. PH3E-BACKUP-004 is the sole basis for every E4 claim.

## D2. Crash-child calibration changed between smoke and recorded runs

The crash-during-backup child initially seeded 3 000 docs with a 5 ms kill delay; at
that size the backup completed in ≈1 ms, so the SIGKILL landed **after** the backup
finished and the smoke run showed `done=true` (partial ACCEPTED_COMPLETE) — a harness
timing artifact, not an engine property. Calibration was changed to 20 000 docs with a
500 µs kill delay; the recorded run kills the process group **inside** the copy
(`done=false`, backup-meta.json absent → REFUSED). No registered run used the
uncalibrated child; smoke outputs lived in /tmp and were never registered.

## D3. Delete/reinsert boundary capture moved to backup return

The first smoke run compared the restored backup against the sidecar model read
**after** post-backup source churn (8 vs 13 mismatch). The harness was corrected to
read the fsynced sidecar **immediately at backup return** (boundary = 33 ACK lines =
20 inserts + 10 deletes + 3 reinserts). Recorded in PH3E-BACKUP-004 as 13/13 MATCH.
The engine was never at fault; this was a measurement-order defect.

## D4. Writer-pause measurement honesty (no fabricated straddle)

In PH3E-BACKUP-004 the b3 single-writer cell recorded `ops_overlapping_backup=0` and
max op latency 192 µs: no writer op happened to straddle the 1.37 ms backup window, so
that row alone does NOT measure the pause. The pause is instead measured where it is
deterministic: b5's concurrent **checkpoint** must acquire the same mutation gate and
waited **257 µs** until the backup finished. We report the pause only from the b5
measurement plus the design invariant (backup holds the gate for its whole copy); no
invented straddle latency is claimed.

## D5. Integrity cases ACCEPTED by policy (documented, not silent)

Two corruption cases restore ACCEPTED/clean and are recorded as expected:
- **corrupt-current** (CURRENT → missing generation): the catalog generation-fallback
  selects the latest *valid* generation, which inside a consistent backup is the
  snapshot generation (self-healing; all generations present and intact). Restored
  **state** was verified against the expected document set, not merely checker-clean.
- **garbage appended to the (empty) active WAL segment**: post-checkpoint a backup's
  WAL contains an empty active segment + `wal-state.json`; zero frames are lost, so the
  E1 torn-tail policy accepts the intact prefix. Documented as designed behavior — the
  same policy E1 already established for live databases.
No validation was weakened to produce these ACCEPTED results.

## D6. Post-run clippy fixes (output-identical; recorded runs unaffected)

After PH3E-BACKUP-004 was recorded, `cargo clippy --workspace --all-targets
-D warnings` flagged 5 style lints in the E4 harness: type-complexity of the integrity
setup closure (factored into `E4SetupResult` alias), an unnecessary `u32` cast, a
nested `format!` (precomputed into `det`), `format!("{e}")` → `e.to_string()`, and a
needless borrow in `fs::copy`. Each produces byte-identical output; no recorded CSV
changes and no re-run was required. (Clippy allows continue to require justification
comments; none were added because no allow was needed.)

## D7. Mode assignment across matrix classes

Matrix classes rotate durability modes (sync/group/async) to cover all three across
the 15 cells rather than a full 15×3 cross-product (cost). The full mode dimension is
covered by the dedicated `modes-sync/group/async` cells (300 acked writes each, all
captured). Read-concurrent (b2) runs under group mode; reader behavior is
mode-independent because readers never touch the WAL commit path.

## D8. Crash row column semantics

In `e4-matrix.csv` the crash-during-backup row stores the **source** document count
(20 000/20 000, `match=SOURCE_OK`); the partial backup itself was never restored (the
new meta gate REFUSES it). `partial_backup_refused=REFUSED`,
`early_backup_still_restores=true` carry the crash properties. This is documented here
because the row reuses the shared schema.
