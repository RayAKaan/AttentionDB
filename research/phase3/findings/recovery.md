# PH3D Finding — Recovery

Status: **SUPPORTED** (with one documented undetectable case) · Run: PH3D-WALCORRUPT-001 + recovery legs of PH3D-CRASH-*/PH3D-STATE-*

## Finding
1. **Torn tail**: an incomplete final WAL frame is truncated at open with a WARNING; the
   intact prefix recovers cleanly (23/40 docs after driver-side truncation to 60%; the 17
   lost acks were removed by the driver's surgery, not by the engine — recovery is the
   longest intact prefix, exactly as documented).
2. **Corrupt frames refuse to open**: appended garbage (`garbage_tail`) and an in-frame byte
   flip (`flip_byte`) both produce `ERROR_ON_OPEN` — no silent false recovery, no fabricated
   data. A gapped/duplicated segment name (`duplicate_segment_gap`: file claims start seq 999,
   first record seq 1) is rejected by the continuity check.
3. **Checkpoint-trimmed WALs reopen cleanly**: every restart/compact gate in PH3D-STATE-*
   (≈30 checkpoint/compact gates per run) reopened a rotated+trimmed WAL — after the checker
   fix below.
4. **Checker false positive fixed (product change)**: `check_db_dir` flagged the normal
   post-checkpoint state (`checkpoint_seq > 0`, WAL empty after rotate+trim) as
   `WAL_SEQ_INVALID`. The gap invariant is now: empty WAL is legitimate iff segment files
   exist; real over-trim (`first_seq > checkpoint_seq+1`, or a contiguous WAL ending before
   `checkpoint_seq`) still errors. Invariant preserved: the checker still fails whenever
   recovery could lose acknowledged mutations.
5. **Undetectable case (OPEN)**: deleting the only WAL segment before any checkpoint leaves a
   directory that opens as an *empty database* with a clean checker — the catalog stores no
   WAL high-water mark, so "fresh" and "vandalized" are indistinguishable. Documented as an
   OPEN limitation (§14 detect-and-document; no mitigation attempted).

## Evidence
`results/wal-replay.csv`, raw `PH3D-WALCORRUPT-001/` (walcorrupt.csv + metrics.json),
checker gates inside `PH3D-STATE-*/consistency-*.json`.

## Limitations
Surgery is file-level (truncate/append/flip/unlink); no multi-segment real rotation
sequences were corrupted mid-rotation. The undetectable pre-checkpoint deletion is a format
limitation, not fixed in 3D.
