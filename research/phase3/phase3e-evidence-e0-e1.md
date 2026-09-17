# Phase 3E — Evidence Report: E0 (Contract Freeze) + E1 (WAL Integrity)

Date: 2026-09-17 · Commit: this commit (re-record lineage: 1204e35 = content of ea4e6b0, parent e8ce38d)
Scope: **E0+E1 only**, per the staged-batch execution model. E2+ explicitly NOT started.
Status: **E0 COMPLETE · E1 COMPLETE** (all E1 acceptance criteria closed, including the two
deferred cases — delete-middle-segment and explicit segment-name gap — via PH3E-WAL-001).

---

## E0 — Production contract freeze

Deliverable: `methodology/production-contract.md`

- G1–G10 guarantees: each names the exact API surface, the experiment that evidences it,
  and its boundary. Written **from the code as it is**, never stronger than the API.
- N1–N9 non-guarantees: distributed operation, serializability, exactly-once delivery,
  crash-safe durability at `Async` level, multi-writer isolation, online compaction (3D
  evidence only), memory bounds, cross-process locking, rollback of `research/phase2`.
- Q1–Q10: direct answers to the ten Phase 3E kickoff questions, including the size
  envelope (Q10) and the durability vocabulary (Committed ≠ Durable unless Sync/GroupCommit).
- Amendment log (§5) is the sanctioned correction mechanism; **A1** is pre-registered for
  the E1 invariant (see below for its final wording).

## E1 — WAL integrity: required-WAL loss must refuse to open

### Gap closed

PH3D-WALCORRUPT-001 (`delete_segment`) showed: delete the only pre-checkpoint WAL segment
→ the database opens as an **apparently-valid empty database**. Root cause: nothing durable
recorded how far the WAL had advanced, so "fresh/trimmed" and "vandalized" were
indistinguishable. This violates the 3E rule: *a fault must never silently produce an
invalid database that believes it is valid.*

### Mechanism (as implemented — see contract §5 A1)

- Durable sidecar `<db>/WAL/wal-state.json` `{format_version, high_watermark, active_start}`.
  Written by `Wal::open_segment` at **every segment creation** (first append + every
  rotation, including checkpoint-rotation) via tmp → rename → fsync dir.
- `high_watermark` = last sequence of **COMPLETED** segments (`next_seq-1` at creation;
  `0` while the first segment is active; `= checkpoint_seq` after checkpoint-rotation).
- `AttentionEngine::open_dir` **refuses** (CoreError) after replay when:
  - `WAL_LOST_SEGMENT` — active segment missing without a newer segment; watermark record
    present with zero `.wal` files; `last_seq < max(high_watermark, checkpoint_seq)` while
    watermark > checkpoint;
  - `WAL_SEQ_GAP` — oldest record seq > checkpoint+1 (uncovered gap);
  - `WAL_STATE_CORRUPT` — sidecar present but unparseable (absent sidecar = legacy DB,
    documented boundary, opens with pre-E1 semantics and adopts the invariant at its next
    segment creation/rotation).
- **Deviation from the original sketch, documented:** no catalog watermark field. The
  catalog is bincode v1 **positional** — adding fields breaks existing manifests — so the
  watermark lives **only** in `WAL/wal-state.json` (never in `META/WAL/`). Contract A1 and
  `phase3e-spec.md` §E1 record this.
- The consistency checker (`core/src/checker.rs`) mirrors the invariant
  (WAL_STATE_CORRUPT / WAL_LOST_SEGMENT ×3 / legacy WAL_MISSING / seq checks).

### Evidence

**PH3E-WAL-001** (raw: `raw/runs/PH3E-WAL-001/`, 11 isolated case DBs, driver-side surgery,
`phase3-bench dbtest walintegrity`; generated: `results/wal-integrity-e1.csv` via
`generate_results_ph3e.py`; **11/11 MATCH, 0 mismatches**):

| case | verdict | observed |
|---|---|---|
| delete_required_segment_pre_checkpoint | REFUSED | WAL_LOST_SEGMENT |
| delete_all_segments_post_checkpoint | REFUSED | WAL_LOST_SEGMENT |
| rename_segment_gap | REFUSED | WAL_REPLAY_CORRUPTION |
| corrupt_wal_state_record | REFUSED | WAL_STATE_CORRUPT |
| delete_wal_state_record_legacy | OPENED | docs=20, checker_clean (LEGITIMATE_LEGACY_NO_SIDECAR) |
| truncate_torn_tail | OPENED | docs=11 intact prefix, checker_clean (LEGITIMATE_TORN_TAIL_PREFIX) |
| corrupt_frame | REFUSED | WAL_REPLAY_CORRUPTION |
| corrupt_current_manifest_fallback | OPENED | docs=20, checker_clean (LEGITIMATE_MANIFEST_FALLBACK) |
| corrupt_all_manifests | REFUSED | MANIFEST_UNREADABLE |
| fresh_no_documents | OPENED | docs=20, checker_clean (LEGITIMATE_FRESH_REOPEN) |
| trimmed_reopen | OPENED | docs=20, checker_clean (LEGITIMATE_POST_CHECKPOINT_TRIM) |

Legitimate-vs-required distinction holds: torn tail recovers the intact prefix (trimmed
prefix bytes only, reported); post-checkpoint trimmed WAL opens; legacy sidecar-less DBs
open; **deleting required history refuses**. Every OPENED case passed the full consistency
checker at open.

**PH3E-REG-001** (raw: `raw/runs/PH3E-REG-001/`) — regression gate against false refusals:
model workload (seed 42, 300 ops) **6/6 passed**; filterx **14/14**; integration **16/16**;
group-durability crash legs (7 points incl. mid_inserts, during_commit_txn, after_compact)
**7/7 verify clean** — acked recovery intact, 0 unacked docs, no resurrection.

Unit tests: 9 engine `wal_integrity_*` + 2 storage sidecar tests + full suites
(core 43, storage 21, query 35, multihead 7, learned 10 — all pass); clippy `-D warnings` clean.

### Consistency gates

`verify_consistency.py` extended (§18): PH3E runs registered with artifacts; generated
results CSV must match the raw run row-for-row; expectation gate (any actual/expected
deviation → MISMATCH → FAIL); registry metrics drift check; regression artifacts present;
evidence-report presence enforced. Result: **PASS** (phase 3 checker), phase 2 checker PASS,
registry = 62 experiments.

## Honest boundaries

- Legacy DBs without a sidecar remain openable (documented migration boundary, A1).
- The watermark anchors at **rotation**, not per-append: loss of the *active* segment
  (since last rotation/checkpoint) is bounded by the size threshold but is NOT caught by
  the invariant if no watermark covers it — documented in A1/N-list; per-append
  watermarking was rejected (manifest rewrite per append is not acceptable).
- `rename_segment_gap` and `corrupt_frame` refuse via the existing replay corruption
  path (reported as WAL_REPLAY_CORRUPTION by the harness's coarse mapping); the
  WAL_SEQ_GAP open-path check is additionally covered by engine unit tests.
- Crash-machine levels for E3, durability API (E2), backup/compaction/txn/concurrency
  work: **NOT STARTED** in this batch.

## Stop point

Per the adopted execution model: **STOP after E0+E1.** Next batch (requires explicit
go-ahead): E2 durability semantics.
