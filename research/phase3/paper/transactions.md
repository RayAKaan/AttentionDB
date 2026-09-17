# Transaction Semantics Validation (Phase 3D)

## Claim
Transactions are all-or-nothing: staged insert/delete sets commit completely — surviving
both process death at the commit boundary and injected pre-validation failures — roll back
cleanly, and never partially apply. The engine's transaction scope is deliberately narrow:
`TxnOp = Insert | Delete`, one collection per transaction, no update op, no isolation
levels. These boundaries are stated, not hidden.

## Method
Direct API probing through the transaction manager (begin/record/commit/rollback/staged
inspection) with state export comparison against the reference model, an injected
failure (a wrong-dimension vector that must fail pre-validation), a delete-of-absent-uuid
no-op probe, and the crash harness's `during_commit_txn` point across all three durability
modes.

## Results (PH3D-TX-001, 7/7)
Commit visible and exact (2 inserts + 1 delete); rollback invisible + unstaged; five-insert
two-delete single-transaction matrix exact; injected validation failure rejected the whole
commit (neither the invalid nor the valid sibling applied); delete-if-present no-op clean;
engine+directory checker clean. Crash legs: 10/10 present (group/sync) or 0/10 absent
(async) at every `during_commit_txn` point — 0 partial outcomes in 21 points.

## What is not claimed
No concurrency/isolation semantics: transactions are serialized by the mutation gate and
the engine exposes no isolation levels, snapshots, or multi-transaction scheduling. No
cross-collection transactions. No write-write conflict detection. The ACID letter "I" is
**not claimed**; "A", "C", "D" are verified within the durability semantics of the selected
mode.

## Limitations
Failure injection covers validation errors and process death; device-level errors (ENOSPC,
fsync failure) are not injectable in this harness and remain untested paths.
