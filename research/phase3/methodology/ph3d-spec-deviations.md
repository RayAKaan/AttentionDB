# Phase 3D — Spec Deviations & Mapping Addendum (audit closure)

Date: 2026-09-17 · Status: committed record · Supersedes nothing; documents honest deltas
between the Phase 3D spec text and the delivered experiments.

## 1. Experiment identifiers (§3)

The spec names nine families; delivery used compact names for four of them and ran one
family later during audit closure. Mapping (child runs carry `parent-map.json`):

| Spec ID | Delivered as | Status |
|---|---|---|
| PH3D-FILTER-001 | PH3D-FILTER-001 | identical |
| PH3D-MUTATION-001 | child run → PH3D-STATE-001..003 + PH3D-FILTER-001 (fm_*) + PH3D-COMPACT-001 | registered child |
| PH3D-RECOVERY-001 | child run → PH3D-WALCORRUPT-001 + PH3D-CRASH-001..003 + STATE restart gates | registered child |
| PH3D-CRASH-001..003 | PH3D-CRASH-001 (group), -002 (async), -003 (sync) | spec CRASH-001 delivered as a 3-run durability×point matrix |
| PH3D-TX-001 | PH3D-TX-001 | identical |
| PH3D-COMPACTION-001 | PH3D-COMPACT-001 + PH3D-COMPACT-002, registered as child PH3D-COMPACTION-001 | child registered |
| PH3D-CONCURRENCY-001 | PH3D-CONC-001/002/003, registered as child PH3D-CONCURRENCY-001 | child registered |
| PH3D-BACKUP-001 | PH3D-BACKUP-001 (+ PH3D-BACKUP-002 inventory run, audit) | identical + extension |
| PH3D-INTEGRATION-001 | PH3D-INTEGRATION-001 | run during audit closure |

## 2. Crash-point granularity (§15–17)

Spec lists 10 crash points; the harness implements 7 (API-boundary granularity, process
kill via SIGKILL/exit(137)). Mapping: mid_inserts ≈ (2 during append / 4 after storage
mutation / 5 during index update); after_acks ≈ 3; mid_flush ≈ 6; after_flush ≈ 7;
after_checkpoint ≈ 8+10 (post-checkpoint/metadata); after_compact ≈ 9 (post-compaction);
during_commit_txn ≈ 3 on the transaction path. NOT separately instrumented: 1 (before first
append — nothing recoverable), and the mid-checkpoint / mid-compaction windows (the
checkpoint/compaction phases are not interruptible at their internal steps without
instruction-level injection). No claim in the report exceeds the implemented points.

## 3. Transaction-failure injection granularity (§20)

Implemented: pre-commit validation failure (injected wrong-dimension vector) and process
death at the commit boundary (covers the during-WAL-write window, since the commit path is
killable at any point before CommitTxn durability). Not implementable in this harness:
device-level failures (ENOSPC, fsync error) at the WAL device. Documented, not patched.

## 4. Multi-op transaction matrix (§21)

Combos involving update (insert+update, update+update, update+delete, upsert+delete as
update) are **UNSUPPORTED**: `TxnOp = Insert | Delete` only. Delivered combos: insert+insert,
insert+delete, delete+delete, plus the injected-failure mix. Stated in Table 4 and the
contract; not worked around.

## 5. Concurrency tooling (§23)

No TSan/lo stomping harness run in this sandbox (toolchain availability); race safety rests
on the RwLock/Mutex/gate design, zero observed errors across all runs, and clean checker
gates. Noted as a limitation, not evidence of absence.

## 6. Fuzzing scope (§34)

Delivered: WAL parser fuzz (64 random-byte + 32 truncated-valid segments, preserved as
cargo tests) and filter validate/eval fuzz (256 random trees + depth rejection). Not
delivered: cargo-fuzz/LVM coverage-guided targets, metadata/idmap fuzz. Scope recorded.

## 7. Raw-run file mapping (§40)

Crash runs carry crash-point.txt, recovery-state.json, consistency-report.json (added at
audit closure); concurrency runs carry operation-log.jsonl; backup-002 carries
inventory.csv (per-file sha256); latency data lives in results.csv columns (latency.csv
not used as a separate file); memory.csv is not applicable to PH3D families (no memory
optimization run — PH3C memory CSVs remain the memory record).

## 8. Tables/figures (§41–42)

Tables follow the spec numbering 1–8 (guarantees, mutation, crash §17-shape, txn atomicity,
compaction equivalence, concurrency, backup/restore, memory optimization) with three
retained extras as tables 9–11. Figures are PH3D-prefixed (figure-ph3d-1 crash recovery,
-2 concurrent throughput, -3 mixed latency) to avoid colliding with the existing PH3C
figure-1..9 series; §42 Figure 4/5 (memory before/after optimization, scaling after
optimization) are deliberately NOT produced — PH3D-MEM-OPT-001 was not run, and the
consistency checker fails if such a figure appears without the run.

## 9. Multi-collection semantics discovered during INTEGRATION-001

The first INTEGRATION-001 attempt cross-infected collections: the harness reused logical
idx across alpha/gamma, and the engine treats uuid identity as GLOBAL (collections are
membership tags; inserting an existing uuid into another collection re-members it and
replaces the record). This was a harness defect (ids now namespaced per collection) AND a
real semantic finding now documented in database-guarantees.md (§3): multi-collection
 callers must namespace logical ids across collections.
