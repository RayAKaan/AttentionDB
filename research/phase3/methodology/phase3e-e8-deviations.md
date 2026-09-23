# Phase 3E — E8 Deviations & Adaptations (all documented before or at measurement)

Author: Rayyan Kaan (RayAKaan). Vocabulary per the standing anti-gaming rules:
a VERIFIED soak run means exactly what its artifacts show — no more. Soak is
NOT production-readiness evidence.

## D19 — Duration model: op budgets, not uncontrolled 24h wall-clock (§44)
The E8 families use deterministic operation budgets (30k/200k/120k+3 readers/
ladder+15k/12 cycles/60+R1–R7+async/25 cycles/240s/600s) with wall-clock as a
secondary observable. This replaces any uncontrolled 24-hour run: shorter,
reproducible, seed+oplog-regenerable runs were preferred per the research
mandate (raw runs immutable; a rerun takes a NEW run ID). No family was cut
short; every family completed its budget except E8i, which hit its 150,002-op
progress cap at 469 s of its 600 s allowance (budget satisfied by ops).

## D20 — E8a bring-up: reader-window tolerance reclassified
During E8a harness bring-up the concurrent-reader scan-window check was
recalibrated twice (777 → 93 → 0 tolerance) before the official run. The
official PH3E-SOAK-001 ran with tolerance 0 and reader violations 0/92,892.
Bring-up runs were never registered as official; the settled tolerance is 0
and is not re-flagged.

## D21 — E8d: ENGINE DEFECT #2 found, fixed, sealed (§50 protocol)
Mid-E8d, a fresh-process recovery REFUSED (`db refused: recovery` on a live
soak dir). Full §50 protocol executed:
- Preserve: raw/runs/PH3E-SOAK-004/bug-preserved-e8d-refusal/{BUG-REPORT.md, db, rundir-full}.
- Diagnose: mid-run `check_db_dir` opened a throwaway engine on the LIVE dir;
  its WAL replay crossed `memtable_threshold=1000` with `auto_flush` on (the
  replay-loop guard had been installed pre-construction = no-op) → a
  partial-replay-state ghost SST was flushed; internal compaction then emitted
  `compacted_*` with a NEWER `last_entry_ts` than real tombstones → 18 dead
  records resurrected → next recovery refused. The 18 orphans are enumerated
  in BUG-REPORT.md for audit.
- Fix: (1) INV-RECOVERY-READONLY — `set_auto_flush(false)/(true)` wraps the
  replay loop in `open_dir`, making the recovery path write-free by guard, not
  by convention; (2) INV-C2 — flush stamps ONE monotonic timestamp
  `max(now, last_entry_ts+1)`, so compacted output can never outrun tombstones.
- Regression: `core/tests/regression_flush_ts_monotonic.rs` (2),
  `core/tests/regression_tombstone_shadow_leak.rs` (1),
  `core/tests/regression_recovery_no_sst_writes.rs` (1, currently a weak
  signal — see D26). Full suite 328/0 at the final build.
- Rerun: family d completed clean as the official PH3E-SOAK-004
  (58,764 ops; txns 17,280 = 13,531c/3,749r; verify 10/0; unmapped=0).

## D22 — E8f: async kill-case judge semantics; documented bounded loss
The original async judge treated ANY version mismatch after an unflushed
groupkill as torn = failure, contradicting the frozen Case B contract
(documented bounded loss ≠ failure). Judge fixed and frozen BEFORE the
official run: recovered older-coherent-version or missing doc ⇒ counted as
documented loss (`ASYNC-LOSS-OK`); a FUTURE version or torn content ⇒ failure.
Official result: async-0 289/289 losses=1; async-1 286/285 losses=0. The
286/285 row is a lost-ACKED-DELETE under the buffered-WAL tail (document
survives at its pre-delete version): prefix-consistent, counted, NOT a
correctness failure, and recorded as a durability boundary for the contract
(ACK ⇒ durable under PH3D_DURABILITY=sync only; async window loss is
documented engine behavior).

## D23 — E8i: two INVALIDATED runs, preserved; third attempt official
Family i took three harness attempts; the engine was exonerated each time
(no refuses, checker clean, no corruption):
- PH3E-SOAK-009 (INVALIDATED, preserved): (a) maintenance cadence used
  `seq % N` checks after a `%5` txn block that adds +2 to seq — every cadence
  multiple (3000/5000/9000/15000/20000) is a multiple of 5, so the +2 always
  skipped the boundary ⇒ ckpt=0, verify=0 across 125,492 ops; (b) the
  seq%20000 restart rebound the engine Arc WHILE the long-running reader held
  a clone ⇒ reader polled a closed engine ⇒ 218,051 violations; (c) txn keys
  could be drawn from different collection blocks but were staged into
  keys[0]'s collection (membership mis-tagging).
- PH3E-SOAK-010 (INVALIDATED, preserved): (a) the reader scans "bench" only
  but the scan-window counter carried the GLOBAL 3-collection live count
  (~3× too large) ⇒ every window check failed; (b) the txn commit updated the
  model by recomputing `nv = maxver+1` per key instead of applying the staged
  plan — duplicate keys within one txn advanced the model to X+2 while the
  engine held X+1 (mapper divergence, `mapper_ok=false`).
- PH3E-SOAK-011 (OFFICIAL, VERIFIED): counter-based cadence
  (`if seq >= next_*`), restart moved AFTER reader-join (E8's own constraint:
  the engine Arc is never rebound under a live reader), bench-scoped window
  counter, plan-based txn model apply. Result: 150,002 progress-ops, 30,000
  txns all committed, ckpt=50, compact=16, backup=10, restart=1, verify=32/0,
  reader 545,690 checks / 0 violations.

## D24 — S16 memory classification: linear-with-ops in every family (E9 marker)
Telemetry (RSS every 2 s) shows RSS growth correlated with operation progress
in ALL nine official families (Pearson r 0.869–1.000; class
`linear-with-ops`; slopes 0.4–1.2 MB per 1k progress-ops; peak 310,980 KB at
E8i's 423k progress-ops). Per the E8 mandate this is MEASURED and CLASSIFIED
only: boundedness beyond the tested budgets is NOT established, this is NOT
adjudged a leak, and no optimization was attempted in E8. E9 owns root-cause
and remediation. FDs (8–11) and threads (2–9) stayed flat in every family;
watchdog fired zero times (no stalls, no deadlocks classified).

## D25 — E8c runs without maintenance cadence (by design)
E8c (mixed rw) runs writers + hot readers with NO checkpoint/compaction
cadence by design, so its db dir grows monotonically (65,379,275 bytes at the
end vs 870,136 at start) — this is the documented no-trim shape, not a
defect; every other family exercises the trim lifecycle and stays bounded.

## D26 — regression_recovery_no_sst_writes is a weak signal (to strengthen)
The INV-RECOVERY-READONLY regression currently passes vacuously (close()
writes a checkpoint SST, so reopen replays ~nothing). It stays in the suite
but is NOT counted as proof; strengthening (force a >1000-op replay via a
second open_dir on a live dir and assert zero SST writes at open) is queued
with the E8 follow-ups. The INV itself is additionally protected by the
family-d soak (10 fresh-process verifies, unmapped=0).

## D27 — No A8 amendment
E8 measured stability and persistence behavior; it did not change observable
transaction/concurrency semantics. The evidence does not warrant an A8
amendment; A1–A7 stand as sealed. (A8 candidates remain listed in the E8
spec; none was triggered per §61's evidence rule.)

## D28 — E8i restore cadence
E8i exercises backup (10) but not backup→restore; restore is exercised and
verified in E8a (1), E8b (1), E8e (12), E8g (25) — every restore
model-snapshot-equal. No restore was attempted in E8i to keep the reader's
engine Arc stable for the full budget.

## D29 — Registry growth
Registry PH3E-SOAK-001..011 (9 official + 2 INVALIDATED preserved). The
original plan said 001..009; the INVALIDATED-preserved protocol adds 010/011.
experiment-index.json updated accordingly (108 entries total).
