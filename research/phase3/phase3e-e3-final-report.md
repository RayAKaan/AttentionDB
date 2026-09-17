# Phase 3E — E3 Final Report

## 1. Executive Summary

E3 asked what AttentionDB can truthfully guarantee when everything holding database
state dies abruptly, rather than one process being signaled. Answer, exactly bounded:

- **Tested (F2E — environment-termination-equivalent):** abrupt termination of the
  entire process group (SIGKILL, no destructors, no flushes) at **19 exactly
  instrumented in-engine windows** across 5 workload classes and 3 modes — **210
  cells**: 177 RECOVERED_EXPECTED, 21 DATA_LOSS_ALLOWED_BY_CONTRACT (Async, exactly per
  the E2 contract), 12 NOT_REACHED (mode-unreachable gates), **0 refusals, 0
  corruption, 0 partial transactions, 0 unsafe opens**. The Sync fsync boundary holds
  under group-kill *before* the ack (fsynced-but-unacked record survives 3/3);
  checkpoint mid-flight windows recover full acked state in every mode (checkpoint-
  start fsync); manifest replacement windows — including the orphan-`manifest-N.tmp`
  state — recover via generation fallback + full WAL replay; WAL-rotation windows hold
  the E1 watermark contract in every mode (rotation precedes the frame write; the
  in-flight record is absent, all acked records survive).
- **Not testable here (BLOCKED, not simulated):** filesystem/cache disruption (F3 — no
  root/block devices) and physical power loss (F4 — no power mechanism). Machine-crash
  durability beyond process-death semantics remains **NOT VERIFIED** for every mode.
- E1 and E2 are untouched: E1 matrix byte-identical (PH3E-WAL-003); all three E2
  durability suites byte-identical (PH3E-DUR-007). Contract amendment **A3** records
  the failure-model boundary.

## 2. Scope and Baseline

E3 only. Baseline commit 5c56ffe (E2 final), tree clean, branch `main` — verified
before any change (standing protocol). No E4+ work started; no changes to transaction,
concurrency, retrieval, or memory subsystems; the only code change is the E3 window
instrumentation (12 new zero-cost-when-disabled gates + a group-kill crash model) and
the harness.

## 3. Git / Toolchain / Environment

- Git: HEAD 5c56ffe at start; branch main; tree clean; no rollback this session.
- Toolchain: rustup/cargo 1.98.1 (reinstalled after sandbox reset, same configuration
  as prior phases), rustc 1.98.1; Python 3.13.
- Environment (part of the evidence): Linux 6.1.158+ x86_64 container on a VM; 2 vCPU
  Intel Xeon @ 2.60 GHz; 2 GB RAM; root + test filesystem **ext4 (rw, relatime,
  discard)** on a virtualized disk layer; NOT tmpfs. libc 0.2 added to the bench crate
  (process-group kill).

## 4. Implementation Audit

Full fs-layer crash-consistency map in `methodology/phase3e-e3-spec.md` §2. Headlines:
manifest/CURRENT/idmap/wal-state all use tmp→sync_all→rename→fsync_dir (atomic);
**SSTables are written at their final name without tmp→rename and without a directory
fsync** (findings F-A/F-B — new); a fully-written orphan `manifest-N.tmp` is counted by
generation scans and can be selected by fallback (benign: CRC-valid complete manifest;
torn tmps fail CRC and are skipped); a mid-SST-write crash leaves a torn file that the
directory-scanning loader REFUSES at open (finding F-C: fail-safe, not self-healing).
WAL: fsync-before-ack under Sync; flush-to-page-cache under GroupCommit; userspace
buffer under Async; rotation fsyncs the completed segment BEFORE creating the next
(and before writing the in-flight frame); wal-state.json is tmp→rename→fsync_dir at
every segment creation.

## 5. Failure Model

- F0 graceful close — control. F1 process crash — E2 evidence, not re-presented.
- **F2E** (implemented): park at in-engine window gate (marker fsynced) → controller
  SIGKILLs the entire process group. Strongest available here; NOT a VM kill; the page
  cache and disk survive. Single-process engine ⇒ the group covers every DB-state holder.
- F3 filesystem disruption — **BLOCKED** (no root, no fsfreeze/dm tooling).
- F4 physical power loss — **BLOCKED** (no power mechanism). Never simulated, never
  substituted.

## 6. Experimental Methodology

Controller/child design: child performs a deterministic workload, records every ACK to
an fsynced sidecar, and parks at the requested window via the crash gate
(`PH3E_CRASH_MODEL=groupkill`); the controller polls the durable marker, kills the
group, then independently reopens the database and records facts (recovered keys,
checker, wal-state, restart stability ×2). Expected state is derived ONLY from the
sidecar + the contract table in `generate_results_ph3e.py` — never from the database.
3 repetitions per cell; 210 cells; counts exact, no percentages.

## 7. Sync Results

Baseline acked records 10/10 at every window. Target record: ABSENT at before_wal_append
and after_write (userspace buffer); **PRESENT from after_apply onward — including
after_fsync where the record is durable BEFORE its own ack (3/3)**. Interpretation: the
Sync acknowledgment boundary (ack ⇒ fsynced) holds under the strongest available abrupt
termination; the fsynced state itself survives process-group death on ext4. Machine-axis
(power-loss) durability: NOT VERIFIED (F4 BLOCKED).

## 8. GroupCommit Results

Baseline 10/10 at every window. Target: ABSENT at before_wal_append / after_write;
**PRESENT from after_apply and at after_flush (3/3)** — the page-cache frame survives
group-kill (as the process-crash contract predicts; F2E cannot distinguish this from
Sync because the cache survives). Machine-axis durability of page-cached frames:
NOT VERIFIED; the contract is NOT upgraded (§12 of the prompt followed).

## 9. Async Results

Baseline **0/10 at every pre-structural window** (buffer lost on death — deterministic
here); acked target lost at after_ack (3/3). Loss classification:
DATA_LOSS_ALLOWED_BY_CONTRACT in 21 cells. Where structural points intervened (mixed
checkpoint windows), recovery is complete (9/9 = all non-deleted acked inserts). In the
txn suite the baseline shows 5/5 at post-COMMIT windows — an 8 KiB BufWriter
capacity-flush artifact, preserved and documented (deviations §5); the acked
transaction itself vanished whole (ABSENT), consistent with E2. Zero partial
transactions anywhere; zero unacked-document resurrection anywhere.

## 10. WAL Rotation Results

Crash INSIDE rotation (2 KiB segments; segment 1 = collection + 3 records;
in-flight 4th insert): all 3 acked records survive in ALL modes at both windows
(rotate_after_old_fsync: completed segment fsynced; rotate_after_state_write:
wal-state.json durably repointed, hwm 0→4 exactly per E1 semantics). The in-flight
record is ABSENT in all modes (rotation precedes the frame write — audited and now
proven). E1 invariants held at every open; restart-stable ×2.

## 11. Checkpoint Results

Crash at 9 mid-checkpoint windows (after WAL fsync / after SST / after idmap / 4
manifest-replacement sub-windows / after rotate / after trim): **16/16 acked records
recovered in EVERY mode at EVERY window**; checker clean; state a fixed point across 2
restarts. The valid outcomes predicted by the audit (old-cp + full replay, or new-cp +
trimmed WAL) are exactly what was observed; no invalid hybrid; checkpoint_seq and
high_watermark consistent in every recovered dir (W=0 pre-rotation windows — first
segment still active; W=17 post-rotate/trim).

## 12. Manifest / SST Crash Results

Manifest replacement (4 sub-windows): all recover — incl. the orphan
`manifest-N.tmp` window, where CURRENT points at a not-yet-existing final name and the
loader falls back to generation N−1 + full WAL replay (16/16). The audited risk (tmp
stem parsed as a generation) is benign: a complete tmp decodes as a valid manifest
(equivalent to a completed rename); a torn tmp fails CRC and is skipped. SST: the
post-write window is covered by ckpt_after_sst (16/16); the MID-write torn-SST case is
NOT_INSTRUMENTED (no gate inside the frame loop); its contract is corruption-fatal
refusal (t15c) — fail-safe, not self-healing (finding F-C).

## 13. Recovery Classification

Per §19 vocabulary, generated from raw facts (results/e3-recovery-classification.csv):
RECOVERED_EXPECTED 177 · DATA_LOSS_ALLOWED_BY_CONTRACT 21 · NOT_REACHED 12 ·
REFUSED_SAFELY 0 · CORRUPTION_DETECTED 0 · UNEXPECTED_DATA_LOSS 0 ·
UNEXPECTED_PARTIAL_STATE 0 · UNSAFE_OPEN 0 · HARNESS_FAILURE 0 · BLOCKED 0 (F3/F4 are
blocked at the INFRASTRUCTURE level, recorded in the spec and metrics, not as cells).

## 14. Independent Reference-Model Validation

Every cell's expected state was computed from the fsynced ACK sidecar + the window/mode
contract table (generator), never from the database. All 210 rows matched (0 MISMATCH;
generator exits non-zero on any mismatch — the phase-3 checker re-runs it).

## 15. E1/E2 Regression Results

- E1: PH3E-WAL-003 **byte-identical** to PH3E-WAL-001/002 (6 REFUSED / 5 legitimate).
- E2: PH3E-DUR-007 (ack-boundary 81, txn-ack 63, checkpoint-interaction 27 cells)
  **byte-identical** to PH3E-DUR-001/002/006. Sync/GroupCommit/Async semantics and ACK
  classification unchanged. A1/A2 not altered; A3 added (see §18).

## 16. Machine-Crash Evidence

Strongest available (F2E): VERIFIED for process-death semantics at all instrumented
commit-path and structural windows (see §1). This is NOT VM-termination evidence: the
kernel, page cache, and disk image survived every experiment. A real VM/container
termination additionally destroys the kernel and (with it) nothing the database wrote
through fsync, but we could not run that experiment; F2E is documented as the exact
mechanism used, with its limits.

## 17. Physical Power-Loss Evidence

**BLOCKED.** No controllable power mechanism exists in this environment. Nothing was
simulated and labeled power-loss. Consequently: "fsync() returned" is verified as an
implementation/runtime observation; "physical media survived power removal" is NOT
VERIFIED for any mode (N6/Q3 unchanged; A3 records the boundary).

## 18. Production Contract Amendment

**A3** appended (append-only): durability claims indexed by failure model
(F0/F1 VERIFIED · F2E VERIFIED with exact mechanism and scope · F3/F4 BLOCKED); the
strongest true statement recorded verbatim; environment recorded; experiments cited;
E1/E2 wording untouched. A3 was required (evidence expanded to structural windows +
failure-model taxonomy); A1/A2 unchanged (E3 regressions prove no behavioral drift).

## 19. Limitations

F2E ≠ VM kill ≠ power loss (page cache intact in all experiments). ext4/VM-disk only —
results must not be generalized to other filesystems or enterprise storage. Mid-SST
write window NOT_INSTRUMENTED (corruption-fatal contract applies). 3 repetitions of a
deterministic workload per cell — window timing is deterministic, so repetitions bound
fs-timing variance only. The `sst_after_write` gate is wired but not a crash cell
(adjacent window covers the same state). Latency/performance out of scope.

## 20. Unsupported Claims

No machine-crash durability claim beyond F2E. No power-loss claim for any mode. No
filesystem-generalization claim. No statistical durability percentage. No claim that
Async is safe (it is not, and its observed losses are documented as contract). No
claim that unacked writes never survive (none were observed; the contract allows them
between WAL-durability and ack).

## 21. Reproducibility

`phase3-bench dbtest e3run --out <dir>` (env-free; all modes exercised internally).
Raw: `research/phase3/raw/runs/PH3E-E3-001/` (+ PH3E-WAL-003, PH3E-DUR-007).
Results: `results/e3-recovery-classification.csv` via `generate_results_ph3e.py`
(MISMATCH → exit 1; re-run by the phase-3 checker). Gate 20 in
`research/phase3/verify_consistency.py` enforces: registry presence, artifacts,
failure_model column, no POWER_LOSS rows, classification match, byte-identical E1/E2
regressions, report + deviations presence.

## 22. Final E3 Verdict

| Mode | Process crash (F1) | Container/VM abrupt (strongest avail.: F2E) | Filesystem disruption (F3) | Physical power loss (F4) | ACK guarantee |
|---|---|---|---|---|---|
| Sync | VERIFIED (E2+E3) | VERIFIED (F2E, 19 windows incl. pre-ack fsync) | BLOCKED | BLOCKED / NOT VERIFIED | ACK ⇒ fsynced (machine boundary designed; unproven at F4) |
| GroupCommit | VERIFIED (E2+E3) | VERIFIED (F2E; page-cache frames survive) | BLOCKED | BLOCKED / NOT VERIFIED | ACK ⇒ page-cache flush (process boundary only) |
| Async | VERIFIED LOSS per contract (E2+E3) | VERIFIED LOSS per contract (F2E) | BLOCKED | BLOCKED | ACK ⇒ committed only; loss observed + documented |

Structural points (checkpoint/rotation/close): recover-all in every mode at every
instrumented window (F2E VERIFIED). **E3 COMPLETE; E4 NOT STARTED.**
