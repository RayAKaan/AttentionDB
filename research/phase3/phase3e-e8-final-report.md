# Phase 3E — E8 Final Report: Soak, Stability & Reliability

Author: Rayyan Kaan (RayAKaan). Commit under test: `3e65d50f8fd1a81cda339d4608260d6be98ba708`
(E7-sealed tree; E8 engine changes are the INV-RECOVERY-READONLY guard + INV-C2
monotonic flush timestamp, both regression-sealed). Tree sha16 at E8 close:
`1b524cf3e39dfc6b`. Statuses use the §48 vocabulary: VERIFIED / PARTIAL / FAILED /
INVALIDATED / UNSUPPORTED / BLOCKED — never "PASS" for a long-run claim.

## §1 Executive Summary

E8 asked one question: do the guarantees AttentionDB earned in E0–E7 survive
sustained operation? Nine official soak families (PH3E-SOAK-001..008, 011) ran to
their deterministic op budgets under a lock-step reference model that was never
read back: 841,909 progress-ops, 47,330 transactions (43,581 committed, 3,749
rolled back by design), 183 checkpoints, 67 compactions, 49 backups, 39 restores,
118 restarts, 190 full verification points with **0 verification failures**, and
1,689,856 concurrent-reader contract checks with **0 violations**. The kill-point
matrix R1–R7, the async kill cases, 60 graceful restarts, a 25-cycle full lifecycle
pipeline and a 12-cycle maintenance cadence all verified. One real engine defect
was found mid-soak (DEFECT #2, a recovery-time ghost-SST resurrection), fixed
under the §50 protocol, regression-sealed, and the affected family re-run clean.
Memory telemetry was measured and classified (linear-with-ops in every family;
boundedness NOT established; E9 marker). No E9 optimization, scale expansion, or
fault injection was performed. **E8 verdict: B.** AttentionDB is NOT
production-ready on soak evidence alone, and this report claims no such thing.
E8 COMPLETE; E9 NOT STARTED.

## §2 Baseline and Entering Contract

E7 closed with verdict D (PH3E-CONC-002 sealed): concurrent readers, reader/writer
isolation, per-doc atomic visibility, commit-order-wins conflicts, ordered txn
visibility, rollback/commit visibility, same-uuid delete/reinsert, checkpoint/
compaction/backup under concurrency, single-op linearizability subset; UNSUPPORTED
by API: write-skew and phantom/range conflicts. E8's charter: verify these
SURVIVE sustained operation — drift, growth, leaks, WAL/SST/tombstone
accumulation, ID-map drift, recovery drift, deadlocks, stalls, corruption,
degradation — without changing any frozen semantics.

## §3 Environment

2-vCPU sandbox, Rust release builds, `PH3D_DURABILITY=sync` for all official
runs, toolchain per workspace env. Harness: `dbtest e8run --family a..i`, child
process `dbtest e8child --dir D --phase R1..R7|ASYNC --seed N`, watchdog (120 s
no-progress ⇒ stall dump + abort), telemetry thread (2 s cadence), model
checkpoints every 10k ops and at every maintenance/restart boundary. Known
scheduler noise (2-vCPU p99 spikes from E7) is explicitly not classified as an
engine event.

## §4 Methodology: Anti-Circular Model, Oplog, Verification Points

The independent reference model is mutated in lock-step with every issued op and
NEVER read back from the engine. Append-only `oplog.csv` per op (buffered,
flushed every 2k ops); (seed, budget, cadence) regenerates the workload and the
oplog records the realized result of every op. Verification point = model
equality on every collection + engine checker (catalog, WAL structure, SST
readability, id-map, throwaway-OpenDir replay guard) + S5 every-3rd-live-uuid
mapper check + S6 ≤50 sampled retired-uuid unmapped probes. State hashes are
FNV-1a-64 over canonical sorted `coll|idx|ver|num` lines. FIRST failure is
preserved in raw before any rerun; reruns take new run IDs; INVALIDATED runs are
preserved, never deleted.

## §5 Run Registry and Statuses

| run | family | status | shape |
|---|---|---|---|
| PH3E-SOAK-001 | E8a smoke | VERIFIED | 30k mixed ops, 500 keys, 2 readers |
| PH3E-SOAK-002 | E8b mutation | VERIFIED | 200k ops, 2000 keys + hot set |
| PH3E-SOAK-003 | E8c mixed rw | VERIFIED | 120k writer ops + 3 hot readers |
| PH3E-SOAK-004 | E8d txn | VERIFIED | ladder 1..100 + 15k txns (25% rollback) |
| PH3E-SOAK-005 | E8e maintenance | VERIFIED | 12 cycles, 2 KiB WAL segments |
| PH3E-SOAK-006 | E8f restart/recovery | VERIFIED | 60 graceful + R1–R7×2 + async A/B |
| PH3E-SOAK-007 | E8g lifecycle | VERIFIED | 25 full pipeline cycles, 3 collections |
| PH3E-SOAK-008 | E8h resource | VERIFIED | 240 s steady-state, 800 keys ≈600 live |
| PH3E-SOAK-009 | E8i extended | INVALIDATED | harness cadence/restart/staging bugs |
| PH3E-SOAK-010 | E8i retry | INVALIDATED | window counter + txn model recompute |
| PH3E-SOAK-011 | E8i extended | VERIFIED | integrated 469 s, full cadence, reader |

Registry: PH3E-SOAK-001..011 (108 index entries total). Full numbers in
`results/e8-soak.csv` and `tables/table-e8-soak.md`.

## §6 E8a Smoke — VERIFIED

30,000 ops in 50 s; maintenance cadence fired (ckpt 6, compact 3, backup 1,
restore 1, restart 1); 12 verification points, 0 failures; two concurrent
readers: 92,892 checks, 0 violations. Establishes the harness end-to-end before
any long family.

## §7 E8b Mutation Soak — VERIFIED

200,000 ops in 246 s on 2,000 keys with a 40-key hot set (2%) under 70/15/15
update/insert/delete mixing; 8 checkpoints, 4 compactions, 1 backup+restore,
2 restarts; 23 verifications, 0 failures. Hot-key churn exercised the same-uuid
delete/reinsert path continuously; tombstone counts stayed bounded through
compaction.

## §8 E8c Mixed RW — VERIFIED

120,000 writer ops in 589 s across disjoint ranges with 3 hot concurrent readers
validating the E7 visibility contract: 746,988 reader checks, 0 violations.
No maintenance cadence BY DESIGN (deviation D25): the db dir grows monotonically
(0.87 MB → 65.4 MB) — the documented no-trim shape. Census taken at 2 quiesced
barriers; exact.

## §9 E8d Transactions — VERIFIED (after DEFECT #2 fix)

Ladder txns 1/2/5/10/25/50/100 × 40 commits each + 15,000 mixed txns at 25%
scheduled rollback + single ops: 58,764 progress-ops, 17,280 txns (13,531
committed / 3,749 rolled back), 10 verification points, 0 failures, 1 restart
between phases. Mid-run the FIRST attempt surfaced ENGINE DEFECT #2 (§25) —
the run was preserved, the engine fixed under §50, and the family re-run clean
under the same official ID slot with the failed state preserved at
`raw/runs/PH3E-SOAK-004/bug-preserved-e8d-refusal/`.

## §10 E8e Maintenance Cadence — VERIFIED

12 cycles × 1,000 ops at 2 KiB WAL segments with per-cycle
checkpoint→rotation→compaction→backup→restart: 12/12/12/12/12 and 36
verification points (3 per cycle), 0 failures. WAL trimmed at every checkpoint;
SST count bounded by policy; backup generations never crossed.

## §11 E8f Restart/Recovery — VERIFIED

60 graceful restart cycles + kill-point matrix R1–R7 × 2 reps (child process,
SIGKILL to the process group at deterministic points) + async Case A (graceful ⇒
all survives) and Case B (unflushed kill ⇒ documented bounded loss). 76 restarts
total, 23 verification points, 0 failures. Async rows: async-0 289/289 with
losses=1; async-1 286/285 with losses=0 — the 286/285 row is a lost ACKED DELETE
under the buffered WAL tail (document survives at its pre-delete version):
prefix-consistent, counted as documented loss, NOT a failure (deviation D22).
No torn documents, no future-version reads, no WAL sequence damage.

## §12 E8g Lifecycle Pipeline — VERIFIED

25 cycles of the full §39 pipeline (INSERT→UPDATE→DELETE→REINSERT→TXN→CKPT→
ROTATE→COMPACT→BACKUP→RESTART→VERIFY) across 3 collections with per-collection
independent model comparison: 50 verified txn commits, 25 of each maintenance
op, 50 verification points, 0 failures. Same-uuid delete→reinsert under
sustained cycles: no resurrection, no remap.

## §13 E8h Resource Stability — VERIFIED

240 s steady state, FIXED 800-key space with live count pinned ≈600 by balanced
churn (delete-oldest / insert-new): 150,000 ops in 214 s, 15 checkpoints,
5 compactions, 2 verification points, 0 failures. With the key space pinned,
RSS still tracked op count (S16 finding, §16) while db bytes stayed bounded
(1.25 MB → 1.00 MB).

## §14 E8i Integrated — VERIFIED

The full integration: multi-collection txns + maintenance cadence + a
continuously-running concurrent reader + telemetry, 150,002 progress-ops in
469 s (op budget completed before the 600 s cap): 30,000 txns ALL committed,
ckpt 50, compact 16, backup 10, restart 1 (after reader join, per the
never-rebind-under-a-live-reader constraint), 32 verification points 0
failures, reader 545,690 checks 0 violations. Two prior attempts are preserved
as INVALIDATED (PH3E-SOAK-009/010; deviation D23) — both harness defects, the
engine exonerated each time.

## §15 S-Property Scorecard

S1–S15, S17, S18: VERIFIED (matrix §24 below). S16: VERIFIED as
measurement+classification (the spec's requirement); the observed class is
linear-with-ops and boundedness beyond tested budgets is NOT established (§16).
No S-property is FAILED, BLOCKED, or UNSUPPORTED within its tested scope.

## §16 Memory Classification (S16) — E9 MARKER

Every official family: RSS sampled every 2 s, classified against op progress.
Result: linear-with-ops in all 9 families (Pearson r 0.869–1.000; slopes
~0.4–1.2 MB per 1k progress-ops on the long families; peak 310,980 KB at E8i).
Classification honesty: this is NOT adjudged a leak; boundedness beyond the
tested budgets is NOT established; RSS-at-entry (~13–28 MB empty engine) is the
floor. FD count 8–11 and thread count 2–9 stayed flat in every family — no
handle growth, no thread leak. Per the frozen E8 mandate this finding is
measured, reproduced across families, localized to per-op growth, and
documented: **E9 owns root-cause and remediation.** Data:
`results/e8-memory.csv`, figure `figures/e8-memory-classification.svg`.

## §17 File-Growth Classification (S15)

With the full lifecycle active (E8a/b/d/e/g/i), WAL bytes trim at checkpoints
and SST bytes stay bounded by compaction policy: db dirs measured stable
(e.g. E8h 1.25→1.00 MB; E8i 1.48→1.52 MB across 423k progress-ops). E8c runs
no maintenance BY DESIGN and grows monotonically to 65.4 MB — the documented
no-trim shape, not a defect. No unexplained file accumulation anywhere; no
SST-count explosion; tombstones GC on full merge as documented.

## §18 FD/Thread Stability (S14)

Across 118 restarts, 183 checkpoints, 67 compactions, 49 backups, 39 restores
and every kill/restore cycle: fds min-max 8–11, threads 2–9 across all runs
(child processes have their own small budgets). No stepwise FD growth, no
thread accumulation, no descriptor leak signal. Classification: stable.

## §19 Recovery Stability & Restart Equality (S8/S17)

Every restart (graceful and post-kill) recovered to the exact modeled state:
190/190 verification points green including post-restart points; R1–R7 with the
same logical prefix + same kill point recovered to identical state hashes in
both reps. Recovery never degraded across the 118 restarts. Fresh-process
recovery remains the authoritative instrument for crash claims (E6/E7 residue).

## §20 WAL/Checkpoint Stability (S3/S9)

2 KiB rotation segments (E8e/E8g) and default segments elsewhere: sequence
continuity held across every rotation and restart; no duplicate or missing
committed records after any recovery; WAL size bounded by checkpoint trim.
Recovery refused nothing in any official run.

## §21 Compaction & Tombstone Stability (S6/S10)

67 compactions; no resurrection anywhere after DEFECT #2's fix (INV-C2
monotonic flush timestamps + INV-RECOVERY-READONLY replay guard, both
regression-sealed); sampled retired (idx, version) pairs stayed unmapped at
every verification point; SST counts bounded by policy through repeated
compaction cycles.

## §22 Backup/Restore Stability (S11)

49 backups, 39 restores, every restore model-snapshot-equal; generations never
cross-contaminated (E8e's 12 generations and E8g's 25 are the dense tests).
E8i exercises backup only, to keep its reader's engine Arc stable for the full
budget (deviation D28).

## §23 Concurrency Progress & Deadlock Watch (S12/S13)

All reader/writer workers completed their budgets; the watchdog (120 s
no-progress) fired ZERO times across 1,871 s of officials. No stall dumps, no
deadlock classification, no suspected-stuck states. Latency spikes alone were
never treated as engine events (2-vCPU scheduler noise, per E7).

## §24 Capability Matrix §63 (25 rows)

| # | capability / claim | status | evidence |
|---|---|---|---|
| 1 | S1 state correctness at every verify point (in-proc + fresh-proc) | VERIFIED | 190/190 points, 0 failures |
| 2 | S2 reference-model equality, all collections | VERIFIED | 190/190; model never read back |
| 3 | S3 WAL integrity across rotation/restart | VERIFIED | §20; 0 refusals, 0 seq damage |
| 4 | S4 SST integrity (checker + compaction output loads) | VERIFIED | 67 compactions clean |
| 5 | S5 ID-map integrity (live mapped; retired never remap) | VERIFIED | S5 every-3rd + S6 samples at 190 points |
| 6 | S6 tombstone correctness (deleted stay absent) | VERIFIED | §21; export + read probes |
| 7 | S7 txn atomicity across graceful + groupkill restarts | VERIFIED | E8d + E8f; never partial |
| 8 | S8 recovery stability (restart hash equality, no degradation) | VERIFIED | §19; 118 restarts |
| 9 | S9 checkpoint stability (WAL trim, no unbounded growth) | VERIFIED | 183 ckpts; §20 |
| 10 | S10 compaction stability (no resurrection, bounded SSTs) | VERIFIED | §21 post-INV-C2 |
| 11 | S11 backup/restore generation equality | VERIFIED | 49/39 cycles, 0 cross-contamination |
| 12 | S12 concurrency progress (op budgets complete) | VERIFIED | all workers finished |
| 13 | S13 deadlock absence (watchdog-classified) | VERIFIED in scope | 0 stalls in 1,871 s |
| 14 | S14 FD/thread stability | VERIFIED | 8–11 fds, 2–9 threads, flat |
| 15 | S15 bounded file growth under lifecycle | VERIFIED | §17; E8c no-trim by design |
| 16 | S16 memory behavior | VERIFIED (classified) | linear-with-ops r≥0.87; boundedness NOT established; E9 marker |
| 17 | S17 deterministic restart behavior (R1–R7) | VERIFIED | identical state hashes, 2 reps each |
| 18 | S18 long-run retrieval correctness | VERIFIED | deleted never returned; live addressable |
| 19 | async kill durability boundary (Case B documented loss) | VERIFIED (documented) | 289/289 l=1; 286/285 l=0; no torn/future reads |
| 20 | same-uuid delete/reinsert + hot-key churn under sustained ops | VERIFIED | E8b/E8g/E8i continuous |
| 21 | 2 KiB WAL rotation stability | VERIFIED | E8e/E8g |
| 22 | multi-collection equality under concurrent txns | VERIFIED | E8g/E8i per-collection model |
| 23 | linearizability beyond E7's single-op subset | NOT VERIFIED | unchanged entering contract; out of E8 scope |
| 24 | serializability / snapshot isolation / skew+phantom detection | UNSUPPORTED BY API | E7 sealed; unchanged |
| 25 | production-readiness, leak-freedom, boundedness-beyond-budget, distributed | BLOCKED / NOT CLAIMED | soak ≠ production-ready; E9 memory; E10 scale; E11 faults; no distributed work ever |

## §25 Defects Found and Fixed

- **ENGINE DEFECT #2 (the only engine defect):** mid-E8d recovery refusal.
  A throwaway checker engine opened on the LIVE dir replayed the WAL across
  memtable_threshold with auto-flush on (guard placed pre-construction = no-op)
  → partial-replay ghost SST → compaction emitted newer last_entry_ts than live
  tombstones → 18 dead records resurrected → next fresh recovery REFUSED.
  Fixed: INV-RECOVERY-READONLY (replay loop runs write-free by guard) +
  INV-C2 (one monotonic flush timestamp = max(now, last+1)). Preserved:
  `PH3E-SOAK-004/bug-preserved-e8d-refusal/` with BUG-REPORT.md and the 18
  orphans enumerated. Regressions: `regression_flush_ts_monotonic` (2),
  `regression_tombstone_shadow_leak` (1), `regression_recovery_no_sst_writes`
  (1; weak signal — deviation D26). Family d re-run clean.
- **Harness defects (engine exonerated):** E8a reader-window tolerance
  recalibration (settled 0); E8f async judge semantics (older-coherent = loss
  counter, future/torn = failure); E8i × 3 across SOAK-009/010 (cadence
  modulo starvation + restart-under-live-reader + cross-collection staging;
  global-vs-bench window counter + txn duplicate-key model recompute).

## §26 INVALIDATED Runs (preserved)

PH3E-SOAK-009 (ops 125,492, 600 s): maintenance never fired; reader violations
218,051 from the restart-under-reader defect. PH3E-SOAK-010 (ops 143,477,
601 s): 30/30 verify failures + 251,024/505,232 reader violations from the two
remaining harness defects. Both dirs preserved intact with INVALIDATED.md;
engine never at fault (no refusals, checker clean, no corruption). Official
family i = PH3E-SOAK-011.

## §27 E1–E7 Regression Coverage

Engine changed only by INV-RECOVERY-READONLY + INV-C2. Full workspace suite at
the final build: **328 passed / 0 failed** (`cargo test --release --workspace`),
including the sealed E1–E7 chains byte-identical, the E6 byte-identical
regression set, and the three new E8 regression files. No old test was weakened
or deleted; nothing was unsealed.

## §28 Production Contract Amendment Status

**No A8 amendment.** E8 measured stability/persistence; it did not change
observable transaction or concurrency semantics, so §61's evidence rule for an
amendment was never triggered. A1–A7 stand as sealed. The async durability
boundary (documented loss of an ACKED DELETE only under a killed process with
buffered WAL tail, sync-durability unset) is DOCUMENTED behavior already
covered by the tested failure-model vocabulary; it adds no new contract term.

## §29 Limitations and Explicit Non-Claims

Soak is NOT production-readiness evidence: AttentionDB is NOT production-ready
on E8 evidence alone. Explicit non-claims: no leak-freedom claim (S16
boundedness unestablished — E9); no boundedness-beyond-tested-budget claim; no
power-loss claim (the tested failure model is SIGKILL + PH3D_DURABILITY
semantics, nothing more); no multi-writer linearizability beyond the E7
single-op subset; no serializability/snapshot/phantom/skew detection
(UNSUPPORTED BY API); no distributed/sharding/replication claims (never
built, never tested); no scale-envelope push (30K/60K/100K/1M doc envelope
untouched); no performance claims from soak telemetry.

## §30 Reproducibility

Every family is (seed, budget, cadence)-deterministic: `dbtest e8run --family
{a..i} --out RAWDIR` regenerates the workload; `oplog.csv` + model checkpoints
regenerate every intermediate state; `resource.csv` reproduces the telemetry
tables; `generate_results_ph3e.py` regenerates results/tables/figures from raw
(never hand-edited); `verify_consistency.py` gate 25 FAILS on any drift between
registry ↔ raw dirs ↔ results ↔ this report. INVALIDATED runs are preserved so
the failure path is reproducible too.

## §31 Final E8 Verdict

**Verdict: B.** All in-scope S-properties (S1–S18) verified within their tested
scopes with zero verification failures and zero reader violations across
841,909 progress-ops; one real engine defect was found, preserved, fixed under
§50, regression-sealed, and survived the full suite (328/0); the verdict is B
rather than A because of documented scope caveats, not failures: S16 is
classified as linear-with-ops with boundedness unestablished (E9 marker), E8c's
no-trim growth is by-design exclusion from S15's lifecycle claim, E8i required
two INVALIDATED harness attempts before its official pass,
`regression_recovery_no_sst_writes` remains a weak signal (D26), and the
linearizability claim stays scoped to E7's single-op subset. §69 stop condition
met: families complete, defects fixed and re-run under new IDs, registry and
deviations closed, no open engine defect. **HARD STOP honored — E8 COMPLETE;
E9 NOT STARTED.**

## ADDENDUM — Artifact restoration (2026-09-22, documented; no results changed)

A platform workspace-snapshot cap silently dropped top-level evidence
(counts/resource/verif/oplog/reader) of PH3E-SOAK-002/003/006/008/011 and all of
PH3E-SOAK-010 after a sandbox reset; PH3E-SOAK-009 (INVALIDATED failure run) and
all crash-test child dirs of 006 survived intact. Nothing was hand-edited. Per
the immutability protocol the originals stand untouched; deterministic
same-seed restoration executions under NEW run IDs (PH3E-SOAK-012=b, 013=c,
014=f, 015=i, 016=h) reproduce the artifact sets. Every restored run matches
its original's registry metrics EXACTLY (ops, checkpoints, compactions,
restarts, verifications, failures — only wall-clock differs, as expected for
time-bounded loops on shared 2-vCPU infrastructure), which is itself independent
confirmation of E8's determinism claim (§30). PH3E-SOAK-010 (INVALIDATED) is not
re-creatable — its harness bugs were fixed — and is recorded as documented loss;
its INVALIDATED summary survives in the registry, this report §26, and
raw/runs/TRIAGE-2026-09-22.md. Regenerable backup/restore generation dirs of
green runs were slimmed to fit the persistence cap; failure evidence
(PH3E-SOAK-004/bug-preserved-e8d-refusal, PH3E-SOAK-009) is untouched.
E8 COMPLETE; E9 NOT STARTED.
