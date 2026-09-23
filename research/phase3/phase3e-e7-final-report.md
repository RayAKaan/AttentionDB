# Phase 3E — E7 Final Report: Concurrency, Isolation Boundaries & Transaction Interleaving

Run: **PH3E-CONC-002** (official; supersedes PH3E-CONC-001, see §34/§29/D13) ·
Date: 2026-09-17 · Commit: `3e65d50f8fd1a81cda339d4608260d6be98ba708` (tree sha16
`7ccdc6e15450bb23`) · Raw: `research/phase3/raw/runs/PH3E-CONC-002/` ·
Spec: `methodology/phase3e-e7-spec.md` · Deviations:
`methodology/phase3e-e7-deviations.md` (D12–D18)

## §1 Executive Summary

E7 measured — from executed schedules, not architecture documents — what
consistency and isolation AttentionDB actually provides under concurrent
readers, writers, and transactions. Result: **48 cells, 42 MATCH + 2
UNSUPPORTED BY API + 4/4 crash rows ATOMIC**. The engine's real contract is:
**commits serialize on a single mutation gate (commit serialization VERIFIED);
transactions are atomic crash units and commit-order-serialized, but they are
NOT visibility units** — a concurrent non-transactional reader can observe
intermediate states of a multi-op commit (mixed pairs captured in E7e/E7f).
There are **no dirty reads**, staged/rolled-back state is never visible,
visibility begins at the apply point inside the commit call, and there is **no
conflict detection** (lost updates occur silently, E7h). Write-skew and phantom
behavior are **UNSUPPORTED BY API** (transactions cannot read: `TxnOp = Insert
| Delete`). No formal isolation level is claimed or claimable; MVCC is
UNSUPPORTED. The strongest earned guarantee is a **single-operation
linearizability subset** (P1/P2/P3 clean on real-time point-register histories)
→ verdict **D** (§35). One genuine engine defect was found and fixed
(§29/D12: recovery-refusing orphan after committed same-uuid [Delete,Insert]
transactions); one suspected defect dissolved into a harness off-by-one
(§29/D13). E1–E6 regressions after the fix are byte- or
classification-identical to their sealed baselines (§30). Commit serialization
is **not** serializability; the absence of observed races is not
linearizability; every claim below carries its exact evidence.

## §2 Baseline

Single-process embedded store; retrieval frozen (Phase 2). Concurrency baseline
entering E7: all mutations (single insert/delete/update **and** whole commit
apply) hold one engine-level mutation gate; reads take shorter RWLock guards
(id mapper, document store, indexes) and are never gate-blocked (E6 bounded
concurrency; E7a re-verifies). Checkpoint/compaction/backup also gate-serialize
with commits. No lock hierarchy, no deadlock detector, no multi-key locks, no
snapshots, no version checks exist — E7's job was to measure what this
architecture actually delivers, including where it is weaker than the mental
model of "serialized commits" (per-op apply = partial-transaction visibility).

## §3 Environment

Linux 6.1.158+ x86_64, 2 vCPUs, rustc 1.98.1 (48a229cea 2026-09-01), tmpfs-backed
workspace filesystem, `Durability::Sync` unless a cell names group/async.
Thread counts: 1–16 readers (E7a), 2–4 workers elsewhere + a sampler thread.
Determinism: fixed seeds (E7x: 11/23/37/52/68/84; E7v: 7/13); harness
randomness is a seeded LCG; failing seeds would be preserved with their raw
dirs (none occurred in the official run). Timing-sensitive cells are classified
by event-log intervals, not wall-clock thresholds (§7).

## §4 Existing Concurrency Architecture (as measured, not as documented)

- One `mutation_gate` mutex serializes: single-doc mutations, entire commit
  apply (WAL append → fsync → apply), checkpoint, compaction, backup.
- Reads (`attend`, `scan_filtered`, `get_document_fields`, mapper probes) take
  RWLock read guards; they never take the mutation gate (E7a: 0 gate blocks
  across 7 440 logged op-slots across all widths (3 840 at 16 threads); p99 latency spikes are scheduler noise).
- Transactions: stage into a TxnManager buffer (invisible to readers), commit
  under the gate: pre-validate → append BEGIN+ops+COMMIT to WAL (durability
  per mode) → apply ops **one at a time** → ACK. Rollback drops the buffer.
- Id mapper: uuid→numeric, mints **from 1**, retires on delete, never reuses
  (INV-6). 0 = n/a sentinel.
- Recovery: WAL replay buffers per txn_id; CommitTxn flushes atomically;
  uncommitted tails discarded; post-recovery consistency checker gates READY.

## §5 Lock/Gate Model

The mutation gate is a **commit-serialization gate, not an isolation
mechanism**: it orders writers and maintenance against each other (E7q/E7r/E7p
serialize correctly) but does nothing for reader atomicity across a multi-op
apply, because readers never take the gate and apply is per-op. The lock/gate
model is therefore: *single-writer-at-a-time, readers-or-writer RWLocks per
structure, no reader snapshots*. A lock is not snapshot isolation — §12/§28
show exactly what this buys and what it does not.

## §6 Reference Model

Per-family independent models, all checked by the generator's expectation
gates (`generate_results_ph3e.py::gen_e7`): static 120-doc census (E7a);
committed-values-only + never-torn (E7b); per-doc monotonicity + pair census
(E7e/f); later-commit-wins-1-doc with restart equality (E7g); both-commits-Ok
final=2 (E7h); absent-then-present ordered semantics (E7k/l); apply-point
visibility with invoke/complete interval membership + post-ACK stability (E7m);
uuid version succession (E7n); no-contamination (E7o); ckpt-never-commits
(E7p); no-partial-no-lost (E7q); pre-or-post-never-partial (E7r); WAL =
commit order (E7s/t); real-time precedence P1/P2/P3 (E7u); serial-by-
construction histories (E7v); all-or-nothing schedule enumeration (E7w);
record-multiset fresh-replay equality (E7x); per-transaction independent crash
judgment (E7y). The generator FAILS on any raw/result drift or expectation
mismatch (gate 24 extends this).

## §7 Event-Log Methodology

Every family logs ops with **invoke/complete timestamps** (monotonic micros
since log start) into `e7-events.csv` (6 042 rows in the official run);
unbounded loops log sampled (1/256) with true timestamps; **E7u logs its full
read history unsampled** because linearizability checking needs every
read's interval. Ordering claims use **barriers/channels only — never
sleep-as-ordering**: staged-window claims use a 3-barrier protocol where the
reader's staged-window reads COMPLETE (barrier b_reads) before the writer
INVOKES commit/rollback (E7k/E7l/E7p). E7m uses a two-barrier protocol and
classifies each rep post-run against the commit's logged [invoke, complete]
interval: early-retire (dirty delete leak — must be 0), window-observed
(retire/appear inside the interval), noticed-after-ack (bounded observation),
plus 50 post-ACK stability reads per rep. Whether a window is *captured* is
scheduling-dependent (D15); the invariants (no early retire, no post-ACK
staleness) are not.

## §8 Concurrent Reader (E7a) — VERIFIED

5 thread counts × 240–3 840 ops on a static 120-doc collection: **errors=0
kinds=0;0;0;0;0** at every width (1/2/4/8/16 threads); attends return only
mapper-valid ids (domain 1..=120, verified by `e7_tiny_probe`), scans return
exactly 120 docs, point fields correct; readers never gate-blocked; p50 ≈
73–80 µs; p99 spikes at ≥8 threads (up to ≈ 24 ms at 16) are scheduler
artifacts on 2 vCPUs, not gate behavior. Checker clean on every final state.

## §9 Reader/Writer (E7b) — VERIFIED

2 readers + 1 writer flipping one key through committed values 1→40: 51
sampled reads observed **only committed values, never torn, never blocked**
(invalid=0); final state = 40. A non-txn reader is always somewhere on the
committed version chain — it just has no guarantee about *which* committed
version relative to a commit's other ops (that is §12's finding).

## §10 Repeated-Read (E7c, expressed via E7e/E7k ordered observations) — PARTIAL

A non-transactional reader re-reading an **unmodified** key sees the same
committed version every time (single visible version per uuid; E7k's
old-missing-while-staged=0 across 30 ordered reps). But there is **no
per-transaction snapshot**: a reader straddling a commit that modified the key
will see the new version on the next read. Repeatable-read in the ANSI sense
is therefore NOT provided (and not claimable — transactions cannot read at
all). VERIFIED: single-version stability. NOT PROVIDED: snapshot repeatability.

## §11 Dirty-Read (E7d, expressed via E7b+E7k) — VERIFIED-ABSENT

No uncommitted value was ever observed: E7b's census contains only committed
values; E7k's 30 ordered reps saw the staged version **0** times and the old
version **exactly** where expected (old-missing-while-staged=0); E7l saw a
rolled-back version **0** times (§18). Dirty-read absence is claimable only
through these instruments, per the anti-gaming rules — and it held.

## §12 Atomic Visibility (E7e/E7f) — PARTIAL (per-doc atomic; NO per-txn snapshot)

30 sequential transactions each rewrite docs A and B (E7e: per-rep distinct
values with monotonic succession; E7f: [Delete A, Insert B]). Per-document
semantics are atomic and monotonic (E7e per-doc-monotonic=YES; E7f
per-doc-monotonic=0 violations; A falls once, B rises once). The (A,B) **pair**
has no snapshot: PH3E-CONC-001's E7e run captured **115 mixed (A,B) pairs**
(out of ~1 400 reads) and PH3E-CONC-002's E7f captured **3 mixed presence
pairs including (absent, absent)** — the transient mid-commit state is real
and observable. CONC-002's E7e captured 0 mixed pairs this run (bounded
observation, D15). Verdict: multi-document transactions are **not** visibility
units for non-transactional readers.

## §13 Write/Write Conflict (E7g) — VERIFIED (commit-order-wins; no detection)

Same key, same uuid, one [Delete U, Insert U] per transaction, both orders ×
sync/group/async (6 cells) + a plain 2-writers×10 case: both commits return Ok
in every cell; the **later commit's value** is final (order-honored, want =
later committer); WAL order = commit order; fresh-process recovery identical.
No error, no abort, no version check — conflicts are not detected, they are
**last-write-won by commit order**. A lock is not snapshot isolation and
commit-order-wins is not conflict detection.

## §14 Lost-Update (E7h) — VERIFIED-OCCURS

The classic read-outside-transaction lost update: two threads blind-stage
`+1`-semantics writes (both read `num=1` outside any txn, stage `2`), both
commit Ok, final = **2**, not 3. Undetected by construction: no conflict
detection exists (§13). Recorded as measured behavior; not fixed (would require
conflict detection = out-of-scope verdict improvement; contract amended
instead, A7.5).

## §15 Write-Skew (E7i) — UNSUPPORTED BY API

`TxnOp = Insert | Delete`: transactions cannot read, so the classic
write-skew invariant (T1 reads x, writes y; T2 reads y, writes x) cannot be
expressed. Row: `not-expressible`, `UNSUPPORTED-NO-TXN-READS`, status `-`.
Not fabricated, not inferred from point reads.

## §16 Phantom/Range (E7j) — UNSUPPORTED BY API

No transactional range/filter/query API exists; a predicate-requery phantom
test cannot be constructed. Row: `not-expressible`,
`UNSUPPORTED-NO-TXN-QUERY`, status `-`.

## §17 Transaction Visibility (E7k) — VERIFIED (ordered)

30 ordered 3-barrier reps: a read that **completes before the commit is
invoked** never sees the staged version (staged-new-visible=0) and always sees
the old committed version (old-missing-while-staged=0); after the commit, the
new version is always there (post-commit-missing=0). Staging is invisible;
visibility begins at apply.

## §18 Rollback Visibility (E7l) — VERIFIED

20 ordered reps, read before staging, during staging, and after rollback:
**ever-visible=0**. A staged-then-rolled-back write never exists for any
reader at any observation point.

## §19 Commit Visibility (E7m) — VERIFIED (visibility = apply point)

40 ordered reps, each classified against the commit's logged invoke/complete
interval: **early-retire=0** (the old version never disappears before the
commit is invoked — no dirty delete leak), **stability-viol=0** (50 post-ACK
reads per rep never see the old version live or the new version absent), and
the retire/appear events land inside the commit interval where the watcher was
fast enough to observe them (this run: 0 window-captured / 39
noticed-after-ack + 1 unclassified — bounded observation, D15; the
mid-commit transient state itself is proven by E7f's (absent,absent) mixed
pairs). Together: **a committed write becomes visible at its apply step,
inside the commit call — possibly before the ACK returns — and never rolls
back or goes stale after.**

## §20 Same-Key Delete/Reinsert (E7n) — VERIFIED

Both orders (delete-then-insert, insert-then-delete-of-old-version) land on
the same final state (uuid v2/v3, old uuids tombstoned); survives compaction +
restart. This family also exposed defect #1 in its same-uuid single-txn form
(§29) — now fixed and regression-pinned.

## §21 Collection Concurrency (E7o) — VERIFIED

Concurrent transactions into disjoint collections (A/B): zero contamination;
read-A/write-B/checkpoint/compaction/backup all isolate; backup restores with
isolation intact.

## §22 Checkpoint (E7p) — VERIFIED

Checkpoint executed entirely inside a staged window (ordered 3-barrier):
`staged-visible-at-ckpt=0`; the checkpoint never commits or exposes staged
state; the old version stays live; the new version appears only via its own
commit.

## §23 Compaction (E7q) — VERIFIED

5 commits racing a compact-in-a-loop (4 compactions this run): commit ×
compaction serialize on the mutation gate; no partial transaction, no lost
write, no resurrection; all 5 docs exact.

## §24 Backup (E7r) — VERIFIED

Backup during staged (snapshot=pre) and during commit (snapshot=post):
**pre-or-post, never partial**; restored state checker-clean both cases.

## §25 Commit Contention (E7t) — VERIFIED

4 barrier-released committers × sync/group/async: all commits Ok, no partial
transactions, no failures; completion orders recorded (e.g. [1,3,0,2] sync,
[2,3,1,0] async); replay order = commit order.

## §26 Linearizability (E7u) — PARTIAL (single-op subset)

Single-key register, 2 writers + 2 readers, **full** invoke/complete history
(1 851 reads / 79 writes): P1 (no future values) = 0, P2 (no
stale-after-completion) = 0, P3 (every value written) = 0; 1 read observed the
transient absent state inside an update commit (counted separately, per-op
apply — the same evidence class as §12; a single-register linearization would
not admit it, so it is NOT folded into P3). Claim: **single-operation
linearizability holds for the tested point-register subset**; NOT a system-wide
claim; transactions are not linearizable as units (§12); invoke/complete
histories preserved in e7-events.csv.

## §27 Serializability (E7v) — PARTIAL (blind-write subset only)

Seeded 3-transaction blind-write histories (seeds 7, 13): commits are atomic
sections on the gate, so every realized history IS a serial execution by
construction — conflict-serializable **for the blind-write subset**. General
serializability is UNTESTABLE (no transactional reads ⇒ no anomaly to search
for) and is NOT claimed. Commit-order histories preserved; no history-analysis
shortcut was taken beyond what the gate-serialized structure proves.

## §28 Isolation Classification (§38 of spec; stated only after all experiments)

**NO FORMAL ISOLATION LEVEL CLAIMED.** Mapping to the ANSI ladder, strictly as
measured: READ UNCOMMITTED anomalies (dirty reads) — absent (§11); READ
COMMITTED anomalies (non-repeatable reads) — present by design for
non-txn readers and untestable inside txns (no txn reads); REPEATABLE READ
anomalies (phantoms) — UNSUPPORTED BY API (§16); SERIALIZABLE — not claimed
(§27 subset only). The engine sits **below a describable ANSI level** because
transactions cannot read: the honest classification is *commit-serialized
atomic units with per-operation visibility, no snapshot, no conflict
detection, no formal isolation level*.

## §29 Bugs Found and Fixed

**Defect #1 (engine, FOUND & FIXED — D12).** Committed transactions containing
[Delete U; Insert same U] applied the reinsert under numeric id 0 with the
mapping retired (`uuid_to_id().unwrap_or(0)` after the txn's own Delete arm).
Live commit Ok; checkpoint persisted an orphan; fresh-process recovery
**refused**: `MISSING_MAPPING … orphan record`. Discovered by the E7g/E7w
same-uuid rewrite; minimized in `core/tests/e7_replay_probe.rs`; failing
on-disk state preserved at `raw/runs/PH3E-CONC-001/bug-MISSING_MAPPING-preserved-dir/`;
fixed by re-registering a fresh numeric id iff the mapping is absent at apply
(double-checked, deadlock-free); regression-pinned (2 tests incl. the WAL
no-close crash path); E1–E6 re-run clean (§30). Severity: data-loss-shaped
(refused recovery after legal committed history) — the highest-value find of
E7.

**Suspected defect #2 (DISSOLVED — D13).** E7a intermittently flagged
`attend` ids ≥ 120. Root cause: harness validity domain off-by-one — the
mapper mints numeric ids **from 1** (proven by `e7_tiny_probe`: doc idx i →
numeric i+1; attend returns exactly the mapper id with correct ordering), so
the legit range is 1..=120 and the check `id >= 120` flagged doc idx 119.
Instrument corrected; **no engine change**; first official run superseded by
PH3E-CONC-002 per the new-run-ID rule.

## §30 E1–E6 Regressions (new IDs; originals sealed)

After the defect-#1 engine change: **PH3E-WAL-008** (E1, 11 cases) wal-
integrity.csv **byte-identical** to WAL-007; **PH3E-DUR-011** (E2, 192 cells)
ack-boundary/txn-ack/checkpoint-interaction/group-boundary **byte-identical**
to DUR-010 (mode-latency: timing-only); **PH3E-E3-005** (E3, 210 cells)
e3-matrix.csv **byte-identical** to E3-004; **PH3E-BACKUP-007** (E4, 25)
integrity byte-identical + matrix classification-identical 15/15;
**PH3E-COMPACT-006** (E5, 25) crash byte-identical + invariants identical
(8/294 timing-only cells); **PH3E-TXN-002** (E6, 49) e6-txn + e6-crash
**byte-identical** to TXN-001. Workspace tests 324 pass; clippy `-D warnings`
clean.

## §31 Production Contract Amendment (A7)

`methodology/production-contract.md` gains **A7 — Concurrency & Isolation
Guarantee** (evidence-warranted per spec §40): A7.1 commit serialization;
A7.2 per-op atomicity, no per-txn visibility unit; A7.3 no dirty reads;
A7.4 visibility = apply point, ACK-after-apply; A7.5 no conflict detection /
lost updates possible; A7.6 single-op linearizability (bounded subset);
A7.7 no formal isolation level, MVCC UNSUPPORTED; A7.8 maintenance isolation
(ckpt/compact/backup never partial). Every line cites its run family. A1–A6
unchanged.

## §32 Limitations

2 vCPUs (window-capture probability, latency spikes); hot-reader evidence is
bounded observation (D15) — invariants, not captures, carry the claims;
schedules exercised are barrier-ordered and spin-reader families at ≤ 16
threads; single-collection transactions; failure model = process-group death
at instrumented gates only (no power-loss claims, A3); HNSW graph randomness
(level RNG) makes top-10 boundary membership vary run-to-run (harmless, but it
is why single-run capture counts vary); E7u's absent-window classification
depends on the update path's per-op apply (documented, not hidden).

## §33 Unsupported Claims (explicit non-claims)

Serializability (beyond the blind-write serial-by-construction subset);
linearizability (beyond the point-register subset); any ANSI isolation level;
MVCC or snapshot semantics; conflict detection / CAS; write-skew or phantom
behavior (UNSUPPORTED BY API); transaction-atomic visibility for non-txn
readers; distributed/replicated anything (out of scope by charter); no
performance claims beyond the recorded latencies of §8.

## §34 Reproducibility

`./target/release/phase3-bench dbtest e7run --out <dir>` regenerates every
CSV from the harness; commit `3e65d50f…`, tree sha16 `7ccdc6e15450bb23`;
seeds fixed (E7x/E7v); events CSV carries the full invoke/complete history;
classification scripts are `generate_results_ph3e.py` (expectation gates) and
`generate_results_e7_tables.py` (tables/figure). PH3E-CONC-001 is preserved
untouched as the superseded first official run (E7a instrument off-by-one,
D13); its non-E7a rows are classification-identical to CONC-002 and its E7e
mixed-pair capture (115 pairs) is cited as evidence where richer than CONC-002's.
Failing-state preservation: `PH3E-CONC-001/bug-MISSING_MAPPING-preserved-dir/`.

## §35 Final E7 Verdict

**Verdict D — single-operation linearizability subset** (strongest guarantee
earned; per spec §50). The engine is a *safe-but-commit-serialized, no-isolation-level*
store: reads never see uncommitted or rolled-back state (VERIFIED-ABSENT),
commits serialize and are crash-atomic (4/4 ATOMIC), single operations are
linearizable in the tested real-time subset, and maintenance never tears state
— but transactions are per-op visible to concurrent readers, conflicts are
silently last-write-won, and the API cannot express the tests (write-skew,
phantom) that would even name a higher isolation level. Commit serialization
is not serializability; the lock/gate model is not snapshot isolation; every
claim above is bounded to its measured schedules. E7 COMPLETE; E8 NOT STARTED.
