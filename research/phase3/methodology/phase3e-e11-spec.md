# Phase 3E — E11 Spec: Fault Injection, Crash-Recovery Verification & Single-Node Reliability Closure

FROZEN before the first official E11 run (2026-09-24). Any post-freeze change
is a documented deviation. Failure model: sudden process death only (A3) —
see §13 non-claims on power-loss.

## M1 Environment & resource class
Same host class as E9/E10 (2 vCPU, 1.9 GiB RAM, Linux 6.1, ext2/ext3,
rustc 1.98.1). All fault injection runs in disposable directories under
`research/phase3/raw/runs/<RUN_ID>/` (db state in a `db/` subdir) created and
destroyed by the harness — never the developer tree, never git metadata.
Per-run wall-clock guard 600 s (F09: 1800 s); memory guard inherited from the
E10 harness discipline (workloads bounded far below the E10 envelope).

## M2 Fault-injection mechanisms (in order of preference)
1. **In-process instrumented gates** (storage/src/crashgate.rs, existing E2/E3
   infrastructure): env `PH3E_CRASH_AT=<gate>` + `PH3E_CRASH_HIT=<n>` arms a
   named gate; `PH3E_CRASH_MODEL=abort` (default) → `std::process::abort()`
   (SIGABRT, no destructors); `groupkill` writes `PH3E_CRASH_MARKER` then
   parks, controller SIGKILLs the process group. Gates are disabled unless the
   env is set (one atomic load / OnceLock when unset — existing inertness
   tests cover ordinary runs). **Three new gates added for E11** (narrowly
   scoped, same contract): `backup_mid_copy` (inside backup copy loop, after
   the first file), `backup_after_copy` (copy complete, before the
   `backup-meta.json` completion marker), `rebuild_mid` (inside the
   deterministic recovery rebuild loop, after the first record insert of the
   first collection).
2. **Harness-level termination points** (E6 pattern): child parks or
   `exit(137)`/aborts after an explicit workload stage; sidecar ack log
   records every acknowledged operation before the fault.
3. **File-level tamper** (F01): deterministic byte edits of closed-database
   WAL artifacts (truncate tail, flip body byte, zero magic, bump sequence
   gap, remove segment, regress sidecar watermark), applied by the harness to
   the disposable copy, with pre-tamper inventory + sha256 recorded.
No timing-based kills: a fault is either at a named gate or a named stage.
Proof that the intended boundary was reached = `fault-reached.json` written
by the controller containing the crash marker (groupkill), the child's
signal/exit status, or the gate's abort signature (exit code 134/SIGABRT).

## M3 Gate catalog (points exercised)
WAL append: after_write, after_flush, after_fsync, before/after_wal_append,
after_apply, before_ack. Rotation: rotate_after_old_fsync,
rotate_after_state_write. Checkpoint: ckpt_after_wal_fsync, ckpt_after_sst,
ckpt_after_idmap, ckpt_after_rotate, ckpt_after_trim, manifest_after_tmp_write,
manifest_after_manifest_dirsync, manifest_after_current_tmp_write,
manifest_after_current_rename, sst_after_write. Txn: tx_before_commit_wal,
tx_after_commit_wal. Compaction: compact_before_merge, compact_after_output,
compact_after_cleanup, compact_after_install. E11 additions: backup_mid_copy,
backup_after_copy, rebuild_mid.

## M4 Run-ID allocation & registry
`PH3E-FAULT-001..NNN`, allocated strictly sequentially; registered in
`research/phase3/raw/experiment-index.json` + `run-manifest.json` with family,
scenario, classification, corrective links. Raw runs immutable; rerun = new ID
linking the original. Registry currently ends at PH3E-SCALE-021 (164 entries).

## M5 Workload & dataset generation
Deterministic hash-derived records (same generator discipline as E10: FNV
uuid scheme, `idx`-keyed fields, unit-basis vectors, dataset hash recorded).
Sizes are deliberately SMALL (bounded by the workload script per family:
F01 20 docs; F02 200; F03 30+txn; F04 300; F05 500; F06 400+150 deletes;
F07 3×800 multi-head; F08 40k base + 300 writer ops + paced reader; F09
40,000 docs). F09 scale is justified by A10 (40k = VERIFIED tier with full
gate battery). Durability mode recorded per run.

## M6 Reference model (independent, harness-side)
The writer process maintains an in-harness model: uuid → (fields hash,
vector checksum), txn table (id, ops, commit status, ack status), acked-op
list — computed from the HARNESS's own workload semantics, never from engine
internals. The model + ack log are flushed to disk (page-cache durable,
survives process death) BEFORE any fault point can fire. Verification
(read-side only) in a fresh recovery process: exact live-set comparison
(uuid set + per-doc field/vector checksum) when the tier allows (all E11
tiers except F09 use full comparison; F09 uses full uuid-set + full
field-hash comparison — still full state, no sampling), transaction
atomicity (all-or-nothing per txn), duplicate-live-id absence, deleted-doc
absence from filtered scans, engine consistency checker clean, retrieval
self-hit on a frozen 50-query set (gate ≥95%) where retrieval applies, and
deterministic recovery = a SECOND fresh open in the recovery process plus a
third open after graceful close must produce byte-identical logical state
hashes. Expected-recoverable sets per mode: Sync ⇒ acked ⊆ recovered (extras
= durable-unacked, reported, allowed); GroupCommit ⇒ same under process
death; Async ⇒ recovered ⊇ last-promoted prefix, loss ≤ unflushed tail,
zero unacknowledged leakage.

## M7 Acknowledgment tracking
An `ACK <opid> <uuid>` line is written+flushed to `ack-log.jsonl`
immediately after the engine call returns Ok. An op is "acknowledged" iff
its line reached the log before the fault. Engine-internal commit state is
NOT consulted for acks (the harness cannot see it; that is the point).

## M8 Recovery validation
Recovery child: opens the db dir fresh (refusal recorded as evidence with the
error text), exports the full logical state via public read APIs, compares to
model per M6, runs the checker, runs retrieval gates, repeats the open, and
writes `recovery-verification.json` (verdict: PASS/FAIL/REFUSED-AS-EXPECTED
with per-check rows). The recovery process NEVER receives the fault env.

## M9 Timeouts, stop rules, safety
Controller per-child timeout (writer 600 s, recovery 600 s, F09 1800 s);
on timeout the child process group is SIGKILLed and the run is classified per
outcome (a parked groupkill fault is SIGKILLed BY DESIGN after the marker
appears — this is the fault, not a timeout). Uncontrolled OOM experiments
forbidden; no fault may target the host, other processes, or git metadata.
If a fault run wedges without reaching its marker within the guard, classify
INVALIDATED (harness) and preserve evidence.

## M10 Evidence retention
Every run keeps: config.json, environment.json, fault-plan.json,
ack-log.jsonl, reference-model.json, fault-reached.json, exit-status.json,
recovery-verification.json, summary.json, stdout.log/stderr.log (both
children), db-inventory.txt + checksums.sha256 of the recovered db dir (and
of the tampered WAL state for F01). Db payloads may be trimmed only AFTER
inventory+checksum, per the E10 TRIAGE-11 policy; failure-state payloads are
compressed and inventoried, never silently deleted.

## M11 Failure classification
Exactly one primary class per run (VERIFIED / SUPPORTED / OBSERVED_LIMIT /
FAILED / INVALIDATED / BLOCKED / UNSUPPORTED) per §11 of the prompt. An async
documented-tail loss is SUPPORTED (boundary behavior confirmed), never
FAILED. A Sync acked-op loss IS FAILED (contract C3) → §10 defect protocol.

## M12 Rerun & invalidation rules
Any harness/config defect → INVALIDATED with reason + corrective link; the
corrective run takes a new ID. Engine defects → preserve run, minimize,
regression-test, smallest in-scope fix, E1–E10 regression chain, rerun under
new ID, update contract matrix row. No threshold or gate may be changed to
make a run pass (tuning ban).

## M13 Statistics / repetition
Fault classes are deterministic by construction (named gates): one run per
scenario point, with repeated-restart determinism checks INSIDE each run
(M6). Where a scenario depends on scheduling (F08), the run records the
observed schedule summary; no timing comparisons across runs are made.

## M14 Completion criteria
All families F01–F09 executed or individually BLOCKED/UNSUPPORTED with
evidence; every official run registered with proof-of-boundary-reached;
reference-model comparisons complete for every correctness claim; all
in-scope defects resolved or E11 explicitly INCOMPLETE; final gates green
(suite, clippy, both checkers, E1–E10 regressions, E11 regressions); final
report + reliability verdict + deviations delivered; §16 checklist satisfied.

## M15 Artifact-policy note
"Where applicable" artifacts: telemetry.csv applies only to long runs (F09);
a short fault run that never opens a db has no recovery telemetry — the
absence is marked in summary.json (`artifacts.skipped`) rather than by empty
files.
