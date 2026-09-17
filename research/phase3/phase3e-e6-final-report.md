# Phase 3E — E6 Final Report: Transaction Semantics, Update/Upsert Correctness & Atomic Commit Boundaries

Run: **PH3E-TXN-001** (raw immutable) · date 2026-09-17 · commit `cdfd2823570…` (E5 final; tree dirty with E6 harness + documented test fixes) · 49/49 cells MATCH · verdict **D — atomic + crash-recoverable + explicitly durable per durability mode, with documented boundaries** (definitions §35).

---

## 1. Executive Summary

E6 establishes, by measurement in fresh processes, exactly what transactional
atomicity AttentionDB has. A transaction's commit boundary is the gate-held
WAL group `BeginTxn → TxnOp* → CommitTxn`; the `CommitTxn` record **is**
COMMIT. Across 7 instrumented crash windows × 3 durability modes (19
fresh-process kill cells) plus 30 in-process cells, every committed
transaction recovered as a complete unit and every uncommitted one recovered
as never-happened — zero partial recoveries. Rollback is a logical discard;
illegal transitions are no-ops. Standalone update/upsert semantics are
classified from code and verified; **in-txn update/upsert is UNSUPPORTED by
type** and documented as such. No isolation/serializability/MVCC claim is
made. Test-side root-causes (D9/D10) were fixed with engine semantics
unchanged, proven by byte-identical E1–E5 regressions. A6 appended to the
production contract. E6 stops here; E7 not started.

## 2. Baseline

Entering E6: HEAD = `cdfd282` (E5 final: coordinated compaction C1 VERIFIED,
verdict B; equal-timestamp tie-break fixed; registry 83). The mutation path
already had: single authoritative mutation WAL; gate-held writes; E1 WAL
integrity policy (refuse on loss/corruption); E2 ACK/COMMIT semantics (A2);
E3 machine-crash windows; E4 coordinated backup; E5 coordinated compaction.
Transactions existed as code (begin/stage/commit/rollback) but had **no
measured crash-safety evidence** — that gap is what E6 closes.

## 3. Environment

Same pinned toolchain as E4/E5 (rustup/cargo under /var/tmp/toolchain; protoc
present). Linux VM, ext4, 2 vCPU. Durability modes exercised: Sync, Group,
Async (env `PH3D_DURABILITY`). Crash model: process-group SIGKILL at
instrumented gates (crashgate `groupkill`; marker-then-park, controller
kills the group — no destructors, no flushes). Fresh-process open is the only
accepted recovery oracle.

## 4. Implementation Audit (pre-change)

Full 25-question audit in `methodology/phase3e-e6-spec.md` (from code, not
intent). Headlines: begin = in-memory txn id (no WAL); stage = memory-only;
commit = pre-validated, gate-held, WAL group append (ids assigned at commit;
deletes log numeric 0 = delete-if-present), then idempotent apply; replay
buffers by `txn_id`, flushes atomically on `CommitTxn`, discards incomplete
tails with a warning; `MAX_TXN_OPS = 100 000` (overflow = Corruption);
`TxnOp = Insert | Delete` only; `AbortTxn` never emitted; one collection per
txn; no nesting; commits serialize on the mutation gate.

## 5. Transaction State Machine

From code: `NEW → ACTIVE → COMMITTING → COMMITTED`; `ACTIVE → ROLLED_BACK`.
Illegal transitions (double-commit, rollback-after-commit,
commit-after-rollback) are no-ops returning false — verified live in E6b
(`c1=true/c2=false/r=false/c3=false` all no-ops, D8 encoding). The state
machine is documentation of actual behavior, not a design aspiration.

## 6. WAL Records & the Commit Boundary

Commit writes ONE gate-held group: `BeginTxn`, one record per `TxnOp`
(inserts bake the collection tag into the payload; deletes log numeric 0 =
delete-if-present), then `CommitTxn`. No separate txn log exists — the
mutation WAL stays singular. Per-mode durability happens inside the append
call, so `tx_after_commit_fsync` is coincident with `tx_after_commit_wal`
(D1; no fabricated row). WAL segments `WAL/<start_seq:020>.wal`;
`WAL_FORMAT_VERSION` enforced on append.

## 7. Invariants (T1–T7)

T1 atomicity — VERIFIED (§8–§15, §19). T2 COMMIT = authoritative boundary —
VERIFIED (§13). T3 uncommitted never survives as committed — VERIFIED (§12,
§19, §22). T4 committed survives per E2 mode — VERIFIED (§13–§15). T5
rollback invisible — VERIFIED (§9). T6 replay idempotent — VERIFIED (§25,
§19). T7 gate serialization preserved (documented, not strengthened) —
VERIFIED (§17, §21–§24).

## 8. E6a — Basic Lifecycle

t1 single insert, t2 multi-insert, t3 insert+delete, t4 multi-delete,
t5 empty txn (committed no-op): all COMMITTED, checker clean, restart-equal,
model MATCH. The empty transaction commits durably and changes nothing —
the boundary is real even for an empty op list.

## 9. E6b — Rollback & Illegal Transitions

Rollback-invisible: staged ops never left memory; state after rollback =
baseline; ROLLED_BACK terminal. Illegal-transitions: all three illegal
attempts are no-ops (D8). Nothing rolled-back ever resurfaced after restart
or compaction.

## 10. E6c — Multi-Op Scale

5 / 10 / 50 / 100-op transactions (mixed inserts+deletes): committed,
restart-identical, full model equality. The 100-op txn (E6p variant) crosses
multiple WAL rotations at 2 KiB segments — 105 documents after restart, no
sequence gap, no duplicate application.

## 11. E6d — Multi-Transaction Interleaving

commit→rollback→commit across three txns: T1 + T3 COMMITTED, T2 ROLLED_BACK;
independent per-txn outcomes; restart equality.

## 12. E6e — Crash Before Commit (fresh-process)

Stage-1/3/10 ops then park (no marker, no commit): parent groupkill.
Fresh-process recovery: state = baseline exactly (ABSENT / ATOMIC / clean /
MATCH ×3). Staging has zero durable or in-memory effect — the strongest
possible form of T3.

## 13. E6f — Crash During Commit (windows)

`tx_before_commit_wal` ×3 modes: ABSENT_ATOMIC (group torn before the commit
record → discarded whole). `tx_after_commit_wal` ×3: sync/group
PRESENT_ATOMIC; async ABSENT_ATOMIC (allowed, A2). `after_apply` (hit 6, D2)
and `before_ack` (hit 7): sync/group PRESENT_ATOMIC; async either-legal
(observed ABSENT). `tx_after_commit_fsync` coincident with
`tx_after_commit_wal` (D1). No window produced a partial txn.

## 14. E6g — Crash After ACK

Child commits (returns Ok = ACK) then `abort()`s, ×3 modes: sync/group
PRESENT_ATOMIC; async ABSENT-or-PRESENT (contract-legal; observed PRESENT).
ACK→COMMIT is durable under sync/group exactly per E2/A2.

## 15. E6h — Durability Matrix (ACK ≠ COMMIT)

The full window×mode grid (19 rows) is the measured durability matrix:
before the commit record → always ABSENT in all modes; after the commit
record → sync/group always PRESENT; async allowed ABSENT until the OS
flushes (never observed PRESENT in before-windows, and that is a loss of the
*buffered append*, not a txn failure — the txn is uncommitted in that case
and recovers as never-happened). No Async machine-durability claim is made.

## 16. E6i — Ordering

Commit-order-wins: same-key reinsert across txns → last COMMIT wins (value
999 survives restart). Delete-missing commits as a successful no-op. Ordering
is a consequence of gate serialization (WAL order = commit order), not a
separate ordering subsystem.

## 17. E6j — Bounded Concurrency (2–3 staged txns)

3 transactions staged concurrently (12 ops), committed out of stage order
(C, A, B): all three recover as complete units; WAL order = commit order.
This is serialization **at commit**, not isolation: no snapshot, no
conflict detection, no serializability claim (T7 documented).

## 18. E6k — Update/Upsert Classification

Standalone update: exists-only (missing/deleted → NotFound), uuid preserved,
fields map REPLACED wholesale (test payloads must carry full identity — D6),
old numeric id retired (no ghost vectors; export loses the stale version —
assert on observable fields, num=4321/cat="t"). Upsert: exists ? update :
insert — both branches verified (doc 150 inserted by upsert with its own
full fields). Restart-stable. **In-txn update/upsert: UNSUPPORTED by the
TxnOp type** (D5; fabrication guard in generator + gate 23).

## 19. E6l — Same-Key Chains Inside One Txn

ins→ins→del→ins on one key within a SINGLE txn: exactly 1 surviving uuid,
final value = the reinserted one (num=160, D8-era expectation), restart
stable. E5's equal-timestamp tie-break regression is included in the suite
and passes (no resurrection).

## 20. E6m — Tombstones

Txn-inserted-then-deleted docs compact away (2 tombstones reclaimed; absent
after compact+restart). A staged (uncommitted) delete rolled back leaves the
doc fully present. Deletion is permanent across txn commit + compaction +
restart.

## 21. E6n — Compaction Interaction

commit→compact and compact→commit both consistent; restart-clean; model
equality. Compaction scans SSTs only and is gate-serialized against commit —
a txn can never be half-compacted (T7).

## 22. E6o — Checkpoint Interaction

txn-then-ckpt and ckpt-then-txn both apply cleanly; **staged-then-ckpt: the
txn is NEVER-COMMITTED and absent after checkpoint + restart** — a checkpoint
can never commit an uncommitted txn (it is gate-serialized behind commit and
WAL writes happen only inside the gate-held commit).

## 23. E6p — WAL Rotation

100-op txn across 2 KiB segments (forced rotation): replays with no gap, no
duplication; 105 docs after restart. E1 ran after rotation in E5 and again
byte-identical in PH3E-WAL-007 (§31).

## 24. E6q — Backup Interaction

Backups taken before / during a staged txn / after commit: during-staged =
baseline snapshot (a snapshot NEVER contains a staged-only partial txn);
after = committed state; restore from each verified (E4 semantics + gate
serialization).

## 25. E6r — Recovery Idempotence

restart×3: state identical after every restart (`recovery(recovery(s)) =
recovery(s)`); no duplicate application of any committed txn (T6). The
replay's txn_id grouping + CommitTxn flush makes double-apply structurally
impossible, and the checker's duplicate-application gate (§38 additions)
would flag it in any case.

## 26. E6s — WAL Corruption

Torn tail through a PARTIAL txn group (real `tx_before_commit_wal` crash
dir): the whole group is discarded atomically — never half-applied (D7
method: drop-without-close, largest segment, ≥64 B). Garbled committed
segment: open REFUSED (E1 policy; ACCEPTED would be UNEXPECTED). Corruption
is never silently converted into a valid transaction.

## 27. Fresh-Process Evidence Discipline

Every crash cell: separate child process (`dbtest e6txn`), gate env
(`PH3E_CRASH_AT/HIT/MODEL/MARKER`), marker-then-groupkill by the parent,
then a NEW process opens the directory and the checker + independent
transaction-aware model judge the recovered state. No crash-safety claim
anywhere in E6 rests on in-process state.

## 28. Blocking / Availability

Commits serialize on the mutation gate; staging does not block. Checkpoint /
compaction / backup vs commit are mutually exclusive critical sections
(measured serializations in §21–§24). No reader was involved in txn cells;
E5 reader-blocking semantics are unchanged (regression §31).

## 29. Performance (observed, non-claims)

In-process txn cells complete in milliseconds at harness scale (10-doc
baseline + ≤100-op txns); the 19 kill cells + 30 live cells ran in ~0.7 s
wall. No throughput/latency claims are made from E6 (out of charter;
correctness > concurrency).

## 30. Bugs Found and Fixed

Engine bugs: **none found in the transaction path** — 49/49 MATCH on the
first official run. Test-side root-causes (found via intermittent CI
failures, proven pre-existing on clean cdfd282 by stash test):
- **E6-D9** phase-2 filter test contradicted the documented two-valued NOT
  contract; test corrected to `AND(IsNotNull(...), ...)` + 4 deterministic
  unit tests added; retrieval semantics UNCHANGED (frozen).
- **E6-D10** two tests asserted exact recall from the approximate ANN
  (t07: 4≠5; executor: 1≠2); replaced with subset/isolation asserts.
  hnsw_rs layer-RNG makes exact recall non-contractual.
- **E6-D11** temporary flake probe removed after diagnosis.
Post-fix: 320 tests ×5 consecutive runs clean; clippy `-D warnings` clean;
E1–E5 regressions byte/classification-identical (fixes touched tests only).

## 31. E1–E5 Regression Results (new run IDs; historical raw untouched)

| regression | new ID | equivalence |
|---|---|---|
| E1 WAL integrity | PH3E-WAL-007 | wal-integrity.csv + metrics.json byte-identical to WAL-001/002/003/006 |
| E2 durability | PH3E-DUR-010 | ack-boundary≡DUR-001, txn-ack≡DUR-002, checkpoint-interaction≡DUR-003, group-boundary≡DUR-004 (all byte-identical, also =008/009); mode-latency timing-only |
| E3 machine crash | PH3E-E3-004 | e3-matrix.csv + metrics.json byte-identical to E3-001/002/003 (210 cells) |
| E4 backup | PH3E-BACKUP-006 | e4-integrity byte-identical; e4-matrix classification-identical (15/15 MATCH) |
| E5 compaction | PH3E-COMPACT-005 | e5-compaction + e5-crash classification-identical (25/25 MATCH) |

All originals preserved; rerun = new ID (immutability rule).

## 32. Production Contract Amendment

**A6** appended (`methodology/production-contract.md` §5): transaction
atomicity semantics and the commit boundary, exactly as measured — including
the UNSUPPORTED-by-type in-txn update/upsert, the AbortTxn-never-emitted
rollback semantics, and the no-isolation boundary. A1–A5 unchanged.

## 33. Capability Matrix (24 rows; VERIFIED / PARTIAL / NOT VERIFIED / UNSUPPORTED / BLOCKED)

| # | capability | status | evidence |
|---|---|---|---|
| 1 | Single-doc insert (durable, all modes) | VERIFIED | E2 (A2), E6a |
| 2 | Delete + tombstone permanence | VERIFIED | E5, E6m |
| 3 | Standalone update (exists-only, uuid-preserving, wholesale fields) | VERIFIED | E6k |
| 4 | Upsert (exists?update:insert) | VERIFIED | E6k |
| 5 | In-txn update/upsert ops | UNSUPPORTED | TxnOp type (D5) |
| 6 | Txn basic lifecycle (begin/stage/commit/rollback) | VERIFIED | E6a/E6b |
| 7 | Txn atomic commit (T1) | VERIFIED | E6a–E6d, PH3E-TXN-001 |
| 8 | Uncommitted = never-happened (T3) | VERIFIED | E6e, E6o, crash rows |
| 9 | Committed survives per mode (T4) | VERIFIED | E6f/E6g/E6h (sync/group) |
| 10 | Async durability = machine-durable at ACK | NOT VERIFIED (contract: allowed loss per A2) | E6h async windows |
| 11 | Rollback invisibility (T5) | VERIFIED | E6b, E6m staged-delete |
| 12 | Replay idempotence (T6) | VERIFIED | E6r, E6l |
| 13 | Multi-op txn crash safety (≤100 ops) | VERIFIED | E6c/E6p |
| 14 | Commit serialization (T7) | VERIFIED | E6j |
| 15 | Isolation / serializability / snapshot | UNSUPPORTED (out of charter, not built) | audit |
| 16 | MVCC / conflict detection | UNSUPPORTED (not built) | audit |
| 17 | Txn × checkpoint | VERIFIED | E6o |
| 18 | Txn × compaction | VERIFIED | E6n |
| 19 | Txn × backup | VERIFIED | E6q |
| 20 | Txn × WAL rotation | VERIFIED | E6p |
| 21 | Txn × WAL corruption (torn/garbled) | VERIFIED | E6s |
| 22 | Machine-crash durability (power-loss class) | BLOCKED | A3 boundary; gate-groupkill model only |
| 23 | Distributed / replicated txns | BLOCKED (standing do-not) | charter |
| 24 | 2PC / cross-collection / nested txns | UNSUPPORTED (not built) | audit (one collection, no nesting) |

## 34. Limitations & Unsupported Claims

Exactly the tested failure model (process-group death at instrumented gates;
fresh-process recovery) is claimed — no power-loss or machine-crash claims
beyond E3's A3 boundary. No isolation/serializability/MVCC/distributed/2PC
claims. No performance claims. Async ACK is not machine durability. In-txn
update/upsert UNSUPPORTED. Readiness is this matrix (§33), never a score.

## 35. Final E6 Verdict

**D — atomic, crash-recoverable, explicitly durable per mode, with
documented boundaries.** Reasoning against the A–E scale: (A) grouping only —
exceeded; (B) atomic — yes (T1–T3, T5–T7 verified); (C) atomic +
crash-recoverable — yes (19/19 fresh-process windows); (D) + explicitly
durable per mode — yes (sync/group PRESENT at every post-commit window; async
exactly per A2 with loss confined to uncommitted state); (E) partial — the
qualifier applies only to the documented boundary set: in-txn update/upsert
UNSUPPORTED, no isolation guarantees, async loss legal — all matrix-visible.
Atomicity ≠ isolation ≠ serializability; only the first is claimed.

**E6 COMPLETE; E7 NOT STARTED.**
