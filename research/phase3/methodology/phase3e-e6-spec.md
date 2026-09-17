# Phase 3E — E6 Specification: Transaction Semantics, Update/Upsert Correctness & Atomic Commit Boundaries

Date: 2026-09-17 · Baseline: cdfd282 (E5 final) · Author: Rayyan Kaan
Run family: `PH3E-TXN-001` (first of family; no prior TXN runs exist)

## 1. Implementation audit (25 answers from actual code)

Code traced: `core/src/transaction.rs` (TransactionManager, Transaction, TxnOp),
`core/src/engine.rs` (begin/record/rollback/commit_transaction, update_document,
upsert_document, apply_replay_record, decode_op, open_dir replay loop,
wal_append), `storage/src/wal.rs` (RecordKind, append, rotation), `storage/src/
crashgate.rs`, plus the existing txn harness (`run_txn`/`txn_child`).

1. **Start:** `begin_transaction(collection)` mints a monotonic txn_id and inserts
   an empty staged Transaction (in-memory HashMap). No WAL record.
2. **Staging:** `record_transaction_operation(txn_id, TxnOp)` appends to the
   in-memory op vector. `TxnOp = Insert(Record) | Delete(Uuid)` ONLY.
3. **Durable storage at stage time?** NO — nothing is written anywhere durable.
4. **In-memory state at stage time?** NO — even in-memory doc state is untouched;
   staging touches only the TransactionManager map.
5. **WAL records written:** only inside `commit_transaction`: BeginTxn, then one
   TxnOp record per op (insert: tagged record msgpack; delete: empty payload),
   then CommitTxn. All under one gate hold.
6. **Identity:** `txn_id` (u64, process-lifetime counter) on every txn record.
7. **Completion:** the CommitTxn record.
8. **COMMIT:** appending the CommitTxn record (gate-held, pre-apply).
9. **COMMIT durable before ACK?** Append performs the mode durability action
   synchronously (Sync: fsync every append; Group: flush; Async: buffered) and
   returns before apply starts; ACK happens after apply. So YES in Sync/Group
   process-death semantics; Async = buffered only (E2 contract unchanged).
10. **COMMIT failure:** `wal_append` error → `?` propagates; staged txn already
    removed from the map (taken at start) — the transaction is lost, nothing was
    applied; partial WAL records (BEGIN/ops without COMMIT) are inert on recovery.
11. **Death before COMMIT:** staged ops were memory-only → transaction never
    happened. Death mid-commit (between BEGIN and COMMIT): partial WAL group →
    recovery discards (buffers without COMMIT never applied).
12. **Death after COMMIT:** WAL has the complete group; recovery applies it
    atomically; if death lands mid-apply, replay re-applies from
    checkpoint/WAL state — apply primitives are idempotent upserts/retires.
13. **Incomplete transactions in recovery:** replay buffers by txn_id; CommitTxn
    flushes the buffer atomically; buffers at end-of-log are DISCARDED with a
    warning ("commit marker = atomicity boundary; aborts are implicit").
14. **Inserts in txns:** yes. 15. **Deletes:** yes. 16. **Updates:** NO — TxnOp
    has no Update variant (update exists only as the standalone gate-held
    `update_document`). 17. **Upserts in txns:** NO (same; standalone
    `upsert_document` = exists ? update : insert).
18. **Span collections:** NO — one Transaction is bound to one collection_name.
19. **Nesting:** NO — begin returns an id; no nesting concept.
20. **Concurrent transactions:** distinct staged txn objects are possible
    (HashMap); each commit takes the mutation gate → commits serialize.
21. **Ordering:** serialized by the mutation gate at commit; staging order is
    irrelevant; commit order = WAL order.
22. **Rollback:** logical (in-memory discard); nothing was ever persisted, so
    there is nothing to undo. AbortTxn WAL kind exists but is never emitted.
23. **Checkpoints with partial txns:** impossible — checkpoint is gate-held and
    WAL writes happen only inside the gate-held commit; a checkpoint can never
    interleave a commit group. checkpoint_seq only advances over appended
    (committed or standalone) records.
24. **Compaction vs txn records:** compaction scans SSTs only; txn staging never
    reaches SSTs; committed txns are applied before compaction can run (gate).
25. **Backup mid-transaction:** backup holds the gate for checkpoint+copy; a
    commit cannot interleave; a staged (uncommitted) txn has no durable presence
    → a backup can never capture half a transaction.

Other facts: `update_document` requires existence (NotFound otherwise), preserves
UUID, bumps `version`, REMAPS the numeric id (old id retired — INV-6), logs
UpdateDocument(old_numeric, record). `upsert_document` = exists ? update : insert
(two gate-held ops — not atomic across the pair; each is individually durable).
MAX_TXN_OPS = 100 000 (replay refuses beyond). WAL_FORMAT_VERSION gate on append.

## 2. Transaction state machine (actual)

NEW → ACTIVE (begin) → COMMITTING (gate; WAL group append) → COMMITTED (ack) —
or — ACTIVE → ROLLED_BACK (in-memory discard). Failure: commit error → txn lost,
nothing applied. Illegal transitions: commit of unknown/committed/rolled-back id →
`Ok(false)` (no-op, never a silent re-commit); rollback after commit → `Ok(false)`.
COMMITTED/ROLLED_BACK are terminal.

## 3. Contract under test

T1 atomicity (all-or-nothing at recovery) · T2 COMMIT = durability/publication
boundary · T3 uncommitted never survives as committed · T4 committed survives per
E2 mode semantics · T5 rollback leaves no visible effects · T6 replay idempotent ·
T7 commit-order serialization (no serializability claim — E7).

## 4. Minimal E6 code changes

1. `GATE_TX_BEFORE_COMMIT_WAL` / `GATE_TX_AFTER_COMMIT_WAL` in
   commit_transaction (+ crashgate allowlist). The generic GATE_AFTER_APPLY /
   GATE_BEFORE_ACK already instrument this exact path (E2/E3) — mapped, not
   duplicated. AFTER_COMMIT_FSYNC is coincident with AFTER_COMMIT_WAL (single
   append call performs the durability action) → documented, not separately
   instrumented.
2. No protocol changes: BEGIN+ops+COMMIT + commit-marker recovery already is the
   smallest coherent protocol, E3-verified at C-windows. No new WAL format fields
   → no version bump (WAL_FORMAT_VERSION unchanged).

## 5. Experiment plan (PH3E-TXN-001) — harness `e6run` + `e6child`

E6a basic (single/multi insert, insert+delete, multi-delete, empty commit) ·
E6b rollback + illegal transitions · E6c multi-op atomicity (5/10/50/100 ops,
mixed) · E6d multiple txns (commit/rollback/commit + crash between) · E6e crash
before commit (staged 1/3/10 ops, groupkill, fresh open) · E6f crash during
commit (2 gates × 3 modes; fresh-process open; async-before-ACK = both-outcomes-
legal, partial never) · E6g post-ACK crash × 3 modes (async loss = contract) ·
E6h durability matrix (modes × windows) · E6i ordering (insert-then-delete;
delete-of-missing; same-uuid double insert across txns = last commit wins) ·
E6j bounded concurrency (3 staged txns interleaved, serialized commits) ·
E6k update/upsert semantics (standalone; in-txn = UNSUPPORTED by type) ·
E6l same-key chains (multi-insert same uuid in one txn; cross-txn; insert+delete
same txn) · E6m tombstones (txn-insert then txn-delete → compact → restart) ·
E6n compaction interaction (both orders) · E6o checkpoint interaction (4 cases
incl. staged-txn + checkpoint → absent) · E6p WAL rotation (100-op txn, 2 KiB
segments) + E1 regression · E6q backup interaction (before/during-staged/after
commit) · E6r recovery idempotence (restart ×3) · E6s WAL corruption (torn tail
mid-txn-group; garbled TxnOp frame). Failure injection: crash gates only
(no write-failure hooks exist; E11 not started).
