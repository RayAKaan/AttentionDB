# E6 Findings — Transaction Semantics (PH3E-TXN-001, 2026-09-17)

Generated from results/e6-transactions.csv + results/e6-crash-atomicity.csv
(49/49 MATCH; 30 in-process cells + 19 fresh-process crash rows).

1. **Atomic commit boundary is real (T1, T2, T3)**: a committed transaction
   recovers, in a fresh process, as a complete unit at every one of the 7
   instrumented windows x 3 durability modes; an uncommitted transaction
   recovers as never-happened. Zero partial recoveries (8 PRESENT-atomic,
   11 ABSENT-class rows, all ATOMIC).
2. **COMMIT = the WAL CommitTxn record** (gate-held group BEGIN -> TxnOp\* ->
   CommitTxn; per-mode durability inside the append; idempotent apply after).
   ACK semantics are exactly E2's: sync/group PRESENT after ACK at every
   post-WAL window; async may lose the buffered append (A2 — ACK != machine
   durability, unchanged).
3. **Rollback is a logical discard** (T5): staged ops never reach memory or
   WAL; the three illegal transitions (double-commit, rollback-after-commit,
   commit-after-rollback) are no-ops. AbortTxn exists in the WAL enum but is
   never emitted; end-of-log uncommitted groups are discarded at replay.
4. **No isolation claims**: staging is concurrent; commits serialize on the
   mutation gate; WAL order = commit order (E6j: 3 staged txns, commits out of
   stage order, all-or-nothing per txn). T7 documented — not invented stronger.
5. **Update/upsert classified, not invented** (E6k): standalone update is
   exists-only/uuid-preserving/fields-replaced-wholesale with old id retired;
   upsert = exists ? update : insert (both branches verified); **in-txn
   update/upsert is UNSUPPORTED by the TxnOp type** (Insert|Delete only) and
   is documented as such.
6. **Interactions hold**: checkpoint never commits a staged txn
   (NEVER-COMMITTED/absent); compaction both orders consistent; backup during
   a staged txn snapshots the baseline (never a partial txn); 100-op txn
   across 2KiB WAL rotations replays with no gap/dup; restart x3 identical.
7. **Corruption never silently converts**: a torn tail through a partial txn
   group discards the whole group atomically; a garbled committed segment
   REFUSES to open (E1 policy).
8. **Test-side root-causes fixed, engine semantics unchanged**: the phase-2
   filter test contradicted the documented two-valued NOT contract (fixed +
   4 deterministic unit tests); two tests asserted exact recall from an
   approximate ANN (replaced with subset/isolation asserts). Proven
   pre-existing on clean cdfd282 via stash test.
