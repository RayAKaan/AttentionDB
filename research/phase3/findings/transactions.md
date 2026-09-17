# PH3D Finding — Transactions

Status: **SUPPORTED for the implemented scope; scope itself is narrower than the word "transactions" usually implies (documented)** · Runs: PH3D-TX-001, PH3D-CRASH-* (during_commit_txn)

## Finding
The engine's transaction model is: staged `Insert|Delete` ops → commit = mutation-gate →
pre-validate ALL ops → WAL `BeginTxn→TxnOp*→CommitTxn` → idempotent apply. Verified:

- **Commit atomicity**: multi-op commit (5 inserts + 2 deletes in one txn) applies
  completely; state exactly matches the model (`txn_multiop_matrix`).
- **Rollback**: staged ops never become visible; the txn is unstaged
  (`txn_rollback_invisible`, `txn_rollback_unstaged`).
- **Injected commit failure**: a txn containing one pre-validation-failing op (wrong-dim
  vector) rejects the WHOLE commit — neither the bad op nor the valid sibling applied
  (`txn_commit_failure_atomic`). Not patched around; the failure is the expected result.
- **Crash atomicity**: SIGKILL around the commit call at the `during_commit_txn` point
  yields 10/10 or 0/10 txn documents — never partial — across all three durability modes
  (21/21 crash points).
- **Delete-if-present**: deleting an absent uuid inside a committed txn is a clean no-op
  (`txn_delete_missing_noop`), matching the `numeric_id 0 = delete-if-present` WAL encoding.

## Scope statements (so nobody over-reads "transactions")
- `TxnOp = Insert | Delete` — **no update op**; updates are single-op `update_document`
  calls outside txns, or delete+insert compositions.
- One collection per transaction (`begin_transaction(collection)`); no cross-collection txn.
- **No isolation levels, no concurrent-txn scheduling, no serializability claim.** Mutations
  serialize on a single mutation gate; the ACID letters claimable here are A (verified), C
  (per-op validation), D (durability-mode-dependent, verified per mode). I is **not claimed
  and not tested beyond gate serialization**.

## Evidence
`results/transactions.csv` (7/7), `results/crash-recovery.csv` (txn column), raw
`PH3D-TX-001/results.csv`.

## Limitations
Single-collection, insert/delete-only workloads; failure injection is limited to
pre-validation failure and process death (no disk-full/fsync-error injection at the WAL
device level).
