# Phase 3E — Production Hardening: Adopted Spec & Execution Status

Adopted: 2026-09-17 · Author: Rayyan Kaan · Freeze rules honored: no new retrieval
architecture, no distributed/RAFT, no speculative optimization. Contract:
`production-contract.md` (amendments only).

## Execution model (accepted)

Staged execution with evidence gates. This batch executes **E0 → E1** and STOPS with an
evidence report (`phase3e-evidence-e0-e1.md`) before E2. Each later batch: implement →
test → record runs → amend contract if semantics changed → update readiness matrix →
commit. No uncontrolled batches.

## Status board

| Item | Scope | Status |
|---|---|---|
| E0 | production-contract.md + phase3e-spec.md freeze | DONE (this batch) |
| E1 | WAL integrity invariant + refusal semantics + matrix | DONE (this batch) |
| E2 | Durability semantics & acknowledgment contract (audit + PH3E-DUR-001..005; contract A2; no API change — documented distinction) | COMPLETE (phase3e-e2-final-report.md) |
| E3 | Machine/power-loss durability: failure-model hierarchy F0-F4; F2E group-kill at 19 instrumented windows (210 cells); F3/F4 BLOCKED; contract A3 | COMPLETE (phase3e-e3-final-report.md) |
| E4 | Online backup: coordinated snapshot protocol | PENDING |
| E5 | Online compaction — only if storage architecture supports it safely; else document + stop | PENDING |
| E6 | Transaction semantics: decide needed op set (candidate: Insert/Update/Delete/Upsert); atomicity + crash tests | PENDING |
| E7 | Concurrency: choose contract (gate + deterministic order + read consistency vs stronger); history-based checker | PENDING |
| E8 | Soak: 1/6/12/24 h mixed workload with RSS/disk/WAL/latency/corruption tracking | PENDING |
| E9 | Memory instrumentation & attribution (raw/HNSW/store/BM25/idmap/WAL/SST/cache/gating/workspace/allocator) — instrument BEFORE optimizing | PENDING |
| E10 | Production-scale envelope 10K→1M where hardware allows; maximum = max reproducible config | PENDING |
| E11 | Fault injection at WAL append/fsync/rotation, record apply, index ops, checkpoint, manifest, SST write/rename, compaction, backup, restore, txn commit | PENDING |
| E12 | Final certification: regenerate production-readiness.csv + phase3e-final-report.md (matrix, never a score) | PENDING |

## E0 — Production contract freeze (accepted output)

`production-contract.md` answers the ten freeze questions (guarantees, non-guarantees,
ack semantics, process crash, power loss, WAL disappearance, backup, compaction,
isolation, ordering, supported size). Rule: implementation changes that alter any answer
require a dated amendment (A1, A2, …) in the contract.

## E1 — WAL integrity invariant (design, implemented this batch)

**Problem (from PH3D-WALCORRUPT-001 `delete_segment`):** a database whose only pre-
checkpoint WAL segment is deleted opens as an apparently-valid empty database. Root
cause: nothing durable records how far the WAL had advanced, so "fresh/trimmed" and
"vandalized" are indistinguishable.

**Invariant chain (per spec):**
```
manifest (checkpoint_seq; NO watermark field — catalog is bincode v1
positional, adding fields breaks existing manifests)
        ↓
WAL/wal-state.json (durable JSON sidecar: format_version, high_watermark,
active_start)
        ↓
required WAL history = (checkpoint_seq, watermark]
        ↓
open() refuses when required history is absent
```

**Mechanics:**
1. `WAL/wal-state.json` durable sidecar (JSON, tmp→rename→fsync_dir) — NOT a catalog
   field: the catalog is bincode v1 positional and must never gain fields (deviation
   from the original sketch; production-contract §5 amendment A1).
   Written at every segment creation (first append and every rotation):
   `high_watermark` = last seq of COMPLETED segments (`next_seq-1` at creation;
   0 while the first segment is active; = checkpoint_seq after checkpoint-rotation);
   `active_start` = the new segment's first seq.
   Rotation occurs at the size threshold and at every checkpoint, so every COMPLETED
   segment's coverage is durably recorded. Per-append watermarking is rejected: manifest
   rewrite per append is not acceptable (cost), and the rotation anchor bounds the
   exposure to the active (post-last-rotation) segment — documented.
3. `AttentionEngine::open_dir` refuses to open (CoreError) when:
   - replay reports a corrupt frame / segment-name mismatch / sequence gap (existing);
   - oldest WAL record seq > checkpoint_seq + 1 (uncovered gap — moved from checker into
     the open path);
   - WAL last_seq < durable watermark (lost/truncated required segments →
     `WAL_LOST_SEGMENT`);
   - the WAL is entirely absent while a watermark > checkpoint_seq exists.
   An empty WAL with watermark == checkpoint_seq is the legitimate post-checkpoint
   trimmed state and opens normally.
4. Legacy migration (honest boundary): a database that has never checkpointed or rotated
   under the new code has no watermark records; loss from before adoption remains
   undetectable for exactly that pre-adoption history. First checkpoint/rotation makes
   the invariant authoritative. Documented, not hidden.
5. `check_db_dir` mirrors the same invariant so the checker and the open path can never
   disagree.

**Tests (E1 acceptance):**
- delete required WAL → REFUSED; delete middle segment → REFUSED; rename segment →
  REFUSED; gap in segment names → REFUSED; truncate final segment → torn-tail WARNING +
  intact prefix (unchanged); corrupt frame → REFUSED (unchanged); corrupt CURRENT →
  fallback to older manifest generation, both generations corrupt → REFUSED; legitimate
  post-checkpoint trim → OPENS (no false positive); legacy DB (no watermark) → OPENS;
  repeated restart cycles → stable.
- Regression: the full PH3D harness must still pass (its restart/trim flows exercise the
  legitimate-trim path ~30×/run).

**Runs:** PH3E-WAL-001 (engine-level matrix via the harness `walintegrity` subcommand),
plus cargo tests `wal_integrity_*` as preserved regression tests.
