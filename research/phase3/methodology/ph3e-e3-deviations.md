# Phase 3E — E3 Deviations

Scope: E3 only. Companion to `phase3e-e3-spec.md`, `production-contract.md` A3 and
`phase3e-e3-final-report.md`.

## 1. F2E is NOT a VM/container kill — the honest strongest-available mechanism

The sandbox provides no VM/container termination API, no root, no block devices, and no
power control. The implemented F2-equivalent ("F2E") is: the workload process runs in
its own process group and **parks** at the exact in-engine window gate (marker file
fsynced first); the controller then SIGKILLs the entire process group. No destructors,
no flushes, no orderly cleanup of anything holding DB state; the engine is
single-process, so the group covers every DB-state holder. What F2E does NOT cover:
kernel death, page-cache loss, disk-image state (all survive). F2E results are
therefore process-death-semantics evidence at structural windows — they cannot and do
not claim machine or power-loss durability. F3/F4 are BLOCKED, not simulated.

## 2. Window hit numbers (manifest gates)

`Catalog::save` is called by BOTH `create_collection` (persist_catalog) and
`checkpoint_locked`. The manifest gates therefore fire at hit 1 during collection
creation and hit 2 during the checkpoint's save. The cells use hit 2 (the designed
checkpoint window). The hit-1 behavior was observed serendipitously during harness
bring-up and is documented here: a crash at the creation-save tmp-write window leaves
ONLY an orphan `manifest-000000001.tmp` (no CURRENT, no final manifest) and the open
REFUSES ("manifest files exist but none are valid") — fail-safe on the very first
catalog write; at later creation-save windows the DB opens with zero documents
(create not yet acked-visible). Raw bring-up runs were discarded (harness bring-up,
pre-registry); the documented behavior is re-derivable by running any manifest cell
with PH3E_CRASH_HIT=1.

## 3. Serendipitous finding: rotation-hit arithmetic

With `ATTENTIONDB_WAL_SEGMENT_BYTES=2048`, segment 1 holds collection + 3 records
(frames are larger than first estimated), so the rotation window crashes during the
4th insert (not the 8th as first sketched). The expectation table derives everything
from the ACK sidecar (3 acked; in-flight unacked), never from constants — per the
independent-reference-model rule.

## 4. SST mid-write window is NOT_INSTRUMENTED

`flush_memtable` writes SSTables at their FINAL name (audit finding F-A). A gate was
added at full-write completion (`sst_after_write`, wired but not used as a crash cell:
the adjacent CKPT_AFTER_SST cell covers the same on-disk state). The MID-write torn
case has no gate inside the SSTable frame loop (out of minimal-change scope); its
contract is the existing corruption-fatal behavior (t15c) plus the loader's
directory-scan refusal. Documented as the audit finding F-C: a mid-SST-write crash
REFUSES at the next open (fail-safe, not self-healing).

## 5. Async baseline behavior differs between suites — artifacts preserved

In ack-c, async loses ALL buffered records (0/10) because the workload never fills the
8 KiB BufWriter. In txn-c, async shows baseline 5/5 at post-COMMIT windows because the
14 txn records auto-flush the buffer mid-commit. Both are artifacts of buffer capacity
boundaries, both preserved in the raw run, both covered by the contract sentence
"survival of any particular acked Async write MUST NOT be relied upon" (A2).

## 6. Mixed-c "missing" accounting

The mixed workload's ACK sidecar contains ACK lines for documents that are later
DELETED by the same workload. `missing_acked` therefore legitimately equals 3 in every
passing structural cell (the deleted docs). The generator's expectation encodes
exactly this (survivors = acked inserts − 3 deleted; deleted_still_gone = 3). This is
bookkeeping, not resurrection.

## 7. The stale-binary trap fired once during harness bring-up

A patch-then-run sequence executed against a stale binary (patch had failed an
assertion); the smoke output was misread for one step and then re-examined. The final
PH3E-E3-001 run was produced by the binary built from the final harness source
(rebuilt and verified before the run). No registered run contains stale-binary output.

## 8. Repetition structure

3 repetitions per cell; workloads are deterministic (fixed seeds/records), so
repetitions vary only in filesystem timing, which cannot change the instrumented
window. 210 cells total; counts reported exactly, no percentages (§23 of the prompt).
