# Systems Limitations (Phase 3D)

This document separates what Phase 3B/3C validated (retrieval), what Phase 3D validated
(database behavior), and what remains open. It is the honest boundary of the system.

## Validated: retrieval (Phase 3B/3C, frozen architecture)
Query → per-head candidates → union → trained gating → weighted fusion → top-K.
Head scaling to 8 heads (gating .71 @5K, .74 @10K H4), candidate-budget flatness
(.7377–.7410 across K=10–200), BM25 channel verification, memory envelopes and leak matrix.

## Validated: database behavior (Phase 3D, this phase)
- Correct mixed-workload state maintenance vs a reference model (0/24 gated checks failed).
- Filter soundness (never returns excluded docs); completeness candidate-bound (recall
  0.967–1.0 measured).
- Acked-write durability per selected mode: GroupCommit/Sync all-acked at 7 crash points;
  Async proper-prefix with explicit committed-txn-loss semantics.
- WAL replay: torn-tail truncation with warning; corrupt frames refuse to open; strict
  sequence continuity; checkpoint rotate+trim interaction.
- Transactions all-or-nothing under crash (21/21 points) and injected validation failure;
  rollback; delete-if-present; NO update op; NO isolation levels.
- Concurrency: parallel readers (zero errors, checker-clean), mixed r/w stability, mutation-
  gate serialization measured; NO ordering/linearizability claim.
- Backup/restore exactness on the quiescent path; no online backup (documented).
- Compaction: offline full-merge + tombstone GC equivalence (after the sst/-path fix);
  automatic incremental compaction at ≥4 files with tombstone retention in partial merges.

## Two product fixes made during 3D (each with regression coverage)
1. `check_db_dir` WAL gap invariant — the post-checkpoint trimmed state is legitimate;
   genuine over-trim still errors. (checker false positive fixed)
2. `compact_all(db_dir)` — resolved the `sst/` subdirectory; offline compaction was
   previously an unreachable silent no-op. (correctness gap closed)

## NOT validated / NOT supported (explicit)
- Machine-crash axis (power loss/VM kill): implemented via fsync, harness only exercised
  process death → durability for Sync is PARTIAL.
- Linearizability / serializability / isolation: not implemented, not claimed, not tested.
- Cross-collection transactions, update ops inside transactions: not implemented.
- Online backup / online compaction: not implemented; quiescence required (documented).
- Pre-checkpoint WAL-segment-loss detection: structurally impossible without a WAL
  high-water mark in the catalog (documented OPEN).
- Async-mode ack durability: explicitly documented as not durable across process death.
- Distributed operation, replication, sharding: out of scope by standing decision.
- Memory optimization: not attempted in 3D (protocol defined for later).

## Known behavioral edges (measured, documented)
- p99 query latency grows with reader count (128 µs @r1 → 60 ms @r32, 4 s windows).
- A single reader cannot hide writer mutation-gate critical sections (r1w1 p50 ≈ 0.8 ms).
- Retired-vector purge backlog under delete-heavy writers (checker WARNINGs, non-fatal).
- 8 heads × 10K × 512 quality run remains OOM-bounded (PH3C preserved failure).
