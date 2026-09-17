# PH3D Finding — Durability

Status: **SUPPORTED for GroupCommit/Sync (process-crash axis); NOT SUPPORTED for Async acks — documented** · Runs: PH3D-CRASH-001/002/003 · Config: 40 acked inserts + 10-insert txn, acked-sidecar protocol, SIGKILL (parked children) or exit(137) (Drop skipped).

## Finding
The three `Durability` modes behave exactly as their code-level contract states, verified at
7 crash points each (21/21 consistent with contract):

- **GroupCommit**: every acknowledged insert survived every process-death point
  (`ALL_ACKED` 7/7) — WAL appends are flushed to the page cache before the ack returns.
- **Sync**: same 7/7 `ALL_ACKED`, plus per-append fsync (machine-crash axis implemented;
  only the process-crash axis could be exercised in this environment).
- **Async**: unflushed acked writes are lost on process death and recovery yields a
  **proper prefix** (35/40, 11/21) with zero unacknowledged documents present and a clean
  checker. After `flush_wal`/`checkpoint`, acks survive (`ALL_ACKED`).

## Key sub-finding (documented, not fixed)
Under **Async**, `commit_transaction` returned success (marker "committed") yet all 10 txn
effects were absent after SIGKILL (`COMMITTED_NOT_DURABLE_ASYNC`). The loss was all-or-nothing
— atomicity held — but the commit acknowledgment does not imply process-crash durability in
this mode. This is the documented meaning of "userspace-only" and is now stated explicitly in
the contract with a run behind it.

## Transactions under crash
`during_commit_txn` (SIGKILL around the commit call): the 10-insert transaction was fully
present (group/sync) or fully absent (async) in every run — never partial (0/21 points).

## Evidence
`results/durability.csv`, `results/crash-recovery.csv` (+ §17-shaped table-3), figure-ph3d-1,
raw
`PH3D-CRASH-001..003/crash-matrix.csv` + per-point `metrics.json`.

Graceful close→reopen durability for insert/update/delete/upsert: PH3D-INTEGRATION-001
(16/16). Backup inventory + independent per-file sha256 integrity: PH3D-BACKUP-002 (6/6).

## Limitations
Process-crash axis only for group/async (SIGKILL); the machine-crash axis (power loss /
VM kill) was not harnessable here — Sync's machine-crash durability is implemented
(fsync per append) but only partially verified. Crash points are parked-process SIGKILLs,
not injected I/O errors at the fsync boundary.
