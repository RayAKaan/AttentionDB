# Phase 3E — E11 Final Report
## Fault Injection, Crash-Recovery Verification & Single-Node Reliability Closure

Author: Rayyan Kaan (RayAKaan). Entry baseline: commit `2df0591`, E10 final tree
sha16 `35ea3b7e5c7c0427` (verified at entry, §2). Failure model for every fault
test: sudden process death (SIGABRT at instrumented gates / process-group
SIGKILL at marker-parked stages) — the A3 axis. Physical power-loss is NOT
tested and is NOT claimed (§17).

## 1. Executive Summary

E11 executed 44 registered fault-injection runs (PH3E-FAULT-001..044) across
all nine mandatory families on the frozen methodology
(`phase3e-e11-spec.md` M1–M15) against the recovered E1–E10 contract
(`phase3e-e11-contract-matrix.md` C1–C15). Result: **every sealed single-node
contract survived controlled interruption at its documented boundaries — zero
FAILED runs, zero engine defects.** 34 runs VERIFIED, 3 SUPPORTED (the
documented Async tail boundary and the bounded concurrency subset), 7
INVALIDATED — every one a harness or expectation defect (D49–D57), each
preserved and each closed by a successful corrective run under a new ID.
Highlights: acknowledged Sync/GroupCommit writes survived death at every
acknowledgment boundary; transactions stayed all-or-nothing at every commit
boundary; partial backups are refused at restore; interrupted checkpoints,
compactions, backups, restores, and recovery rebuilds all recovered to the
contract state with byte-identical repeat restarts; the fresh multi-head
checkpoint performs no spurious rebuild (D40 e2e: 3.1 ms); and the integrated
40k lifecycle (build → churn → 200 transactions → checkpoint → compaction →
coordinated backup → controlled interruption → recovery) verified with exact
full-state equality, clean checker, 48/50 retrieval self-hit, and
restore-equals-source. One measured tolerance was added to the contract by
evidence (A11): a regressed `wal-state.json` watermark opens and fully
recovers — WAL records are authoritative. Final gates: suite 334/0, clippy
clean, both consistency checkers PASS with the new E11 evidence gates.
Terminal status with the completion checklist: **PHASE 3E E11 COMPLETE —
FINAL RELIABILITY VERDICT ISSUED** (§21).

## 2. Baseline and Final Repository State

Entry: `git status` clean; HEAD `2df0591` (main); tree sha16 recomputed by the
E10 method = `35ea3b7e5c7c0427` — exact match, no baseline deviation. E10
report/spec/registry/defect notes verified present; registry at entry = 164
entries ending PH3E-SCALE-021. Sandbox reset #4 had wiped the toolchain;
restored (rustc 1.98.1 — identical version — + protoc 28.2) and the full
entry-gate battery run before any official fault test (§3). Final state: HEAD
=this commit (§18), registry 208 entries (164 + 44 E11), workspace 134 MB /
3,191 files after the documented retention pass (D58).

## 3. Entry Gates (all captured in raw logs)

Verified before official runs: E10 tree sha ✓; suite 330/0 and clippy `-D
warnings` 0 (run at entry; the count later grew to 330→334 only via the four
new E11 regression tests, §18); phase-2 and phase-3 consistency checkers PASS;
E10 generator + evidence gates green; E9 hygiene regression tests present and
passing (the restored-attribute repair from E10 close re-verified: both
`e9_hygiene_bounds…` and `e9_hygiene_multi_head…` execute). One entry
blocker found and fixed before runs: sandbox-reset loss of the toolchain
(restored, version-identical). No gate was suppressed or weakened.

## 4. Recovered E1–E10 Contract

`phase3e-e11-contract-matrix.md` recovers G1–G10 plus amendments A1–A10 into
15 testable rows C1–C15 (source citations, expected fault-boundary behavior,
evidence, pass conditions, non-claims), plus three recorded ambiguities
(AMB-0 resolved by measurement — see §6; AMB-1/2/3 preserved as recorded).
Nothing was inferred into a stronger promise; the E10 scale envelope (C14)
and the process-death failure model (C15) are binding constraints on E11
itself.

## 5. Frozen Fault-Injection Methodology

`phase3e-e11-spec.md` frozen before the first official run: M1 environment
(2 vCPU/1.9 GiB class, disposable run dirs); M2 mechanisms (in-process
crashgate instrumentation — SIGABRT at named gates, groupkill at marker-parked
stages, file-level WAL tamper for F01; three→four new narrowly scoped gates
with the same inertness contract); M3 gate catalog (30 named boundary points);
M4 run-ID allocation; M5 deterministic workloads (sizes bounded far below the
E10 envelope; F09 at the A10 40k VERIFIED tier); M6 independent harness-side
reference model (never engine internals; flushed to disk BEFORE any fault
point); M7 acknowledgment tracking (ack-log line + flush per completed call);
M8 recovery validation in a fresh process; M9 timeouts/stop rules (no
uncontrolled OOM; nothing outside disposable dirs touched); M10 evidence
retention; M11 classification vocabulary; M12 rerun/invalidation rules (new
IDs, corrective links); M13 repetition/determinism treatment; M14 completion
criteria; M15 artifact applicability. Post-freeze changes are deviations
D49–D58 only; no threshold was changed to make a run pass.

## 6. WAL Integrity/Replay Findings (C1/C2 — F01, runs 001–008, 038)

Corrupt frame body (003), replaced frame magic (004), +7 sequence gap (005),
and missing segment (006) all REFUSE to open — G5's refusal behavior intact
under byte-level attack. Truncated tail (002) recovers the intact prefix with
the loss bounded by the destroyed bytes and deterministic across restarts
(pinned by `e11_torn_tail_recovers_intact_prefix`). Absent sidecar (008) —
the documented legacy condition — is tolerated (A1). MEASURED TOLERANCE
(AMB-0): a regressed sidecar watermark OPENS and recovers the full state
(007 preserved INVALIDATED as an expectation defect; 038 VERIFIED the
corrected expectation; pinned by
`e11_regressed_sidecar_opens_with_full_recovery`). The sidecar is derived
bookkeeping; the records are authoritative. No E1 refusal behavior was
weakened.

## 7. Durability-Mode Findings (C3/C4/C5 — F02, runs 009–014)

At gates after_write (userspace buffer), after_flush (page cache), after_fsync
(post-fsync/pre-return), and before_ack, with 200-insert sync workloads:
every ACKNOWLEDGED write survived process death in Sync (009/010/011) and
GroupCommit (012); the at-most-one durable-but-unacked in-flight op appeared
exactly where the contract permits it (durable_unacked recorded per run,
zero leakage). Async (013/014): loss confined to the un-promoted tail with
zero unacknowledged leakage; the promote boundary (`flush_wal()` after 100
acks) held — everything promoted survived, everything unpromoted was lost,
exactly the documented buffered-WAL-tail boundary (E8f confirmed at the
boundary level). Classified SUPPORTED, not VERIFIED-durability, per M11.
GroupCommit semantics observed from the gate placement: ack follows a page
cache flush per append — no coalescing claim is made (AMB-1).

## 8. Transaction Atomicity Findings (C6/C7 — F03, runs 015–018)

10-insert transactions interrupted at `tx_before_commit_wal` (015: zero ops
recover), `tx_after_commit_wal` (016: ALL ten ops recover — the commit
boundary is the WAL marker, exactly A6), parked pre-commit (017: zero ops),
and rollback-then-death (018: zero ops). All-or-nothing held at every point;
no partial transaction ever became visible; rollback discards durably. The
documented scope stands: Insert+Delete, single collection, no isolation
levels, no in-transaction update/upsert op (UNSUPPORTED by type, unchanged).

## 9. Checkpoint/Recovery Findings (C8 — F04, runs 019–023, 039)

Checkpoints interrupted after SST creation (019), after WAL trim (020),
after rotation (021), after the current-manifest rename (039), and mid-SST
write (023): every leg recovered all 300 Sync-acked writes, checker clean,
triple-restart byte-identical. 022 (hit=1 of the manifest gate) is preserved
INVALIDATED — the hit was consumed by the setup-time catalog write, so the
fault landed before the workload (proof: absent model + no CKPT ack note);
039 (hit=2) is the corrected run. The contract that intermediate states need
not be openable was not tested beyond what the gates reached; where an
interior state WAS opened (039), recovery was exact.

## 10. Backup/Restore Findings (C9 — F05, runs 024–026, 040)

A crash inside the backup copy loop (040, corrective for 024) leaves a
partial destination with no `backup-meta.json`; restore REFUSES it
(Corruption: "incomplete/invalid backup") and the SOURCE reopens intact with
a clean checker. A crash after the copy but before the manifest (025) leaves
a complete-looking directory that restore also REFUSES — the completion
marker rule (A4) is enforced, not incidental. Clean backup → restore → open
(026) equals the captured logical state exactly (counts + state hash +
checker), and restore refuses a non-empty destination. Backup is quiescent
by the mutation gate; no nonblocking claim is made.

## 11. Compaction Findings (C10 — F06, runs 027–031)

Compaction interrupted before merge (027), after output creation (028),
after old-file cleanup (029), and after install (030) — plus the clean
reclamation control (031: SST 3→1, real tombstone reclamation at 400→250
docs) — every leg recovered exactly 250 live documents with all 150 deletes
absent from filtered scans, checker clean, restart-deterministic. Deletions
survive interruption at every compaction interior point (anti-resurrection
holds). Blocking readers/writers per the existing contract is not treated as
a defect; no performance claim.

## 12. Index-Hygiene Findings (C11 — F07, runs 032–035, 041–044)

The D40 head-count-aware predicate is intact end-to-end: a fresh multi-head
checkpoint (800 docs × 3 heads, zero dead) performs NO rebuild — ckpt 3.1 ms
(042; contrast the pre-D40 78–157 s) — while a dead-dominated checkpoint
(25k→1k docs) legitimately fires purge + deterministic rebuild (11.2 ms →
427 ms, 033). Recovery rebuilds interrupted at `rebuild_mid` (041: leg-1
SIGABRT inside the rebuild, leg-2 full green) complete deterministically on
the next open with exact counts, no duplicate live ids, and identical state
hashes. The hygiene checkpoint interrupted after SST creation (044) recovers
the exact 1,000-doc post-purge state. Retrieval on the corrected generator:
50/50 self-hit (042). The frozen E10 gate (≥95%) applied wherever retrieval
was measured. No threshold was tuned (D40's predicate fix remains the E10
correctness fix; E11 only measured it).

## 13. Concurrency/Recovery Findings (C12/C13 — F08, run 036)

Writer + paced reader (2 ms cadence, E10 finding) with the writer crashed at
`before_ack` (op 150 of 300): recovered state matched the model within the
Sync contract (acked ⊆ recovered, the in-flight op as durable-unacked, zero
leakage), reader log shows zero errors, and repeat restarts are
byte-identical. Consistent with A7's bounded subset: commit serialization,
no dirty reads observed, per-key LWW; NO linearizability/serializability
claim is made (A7.5 lost-update semantics unchanged). The E10 scan-heavy
reader bottleneck was characterized, not optimized (§2.6 of the charter).

## 14. Integrated Lifecycle Results (C14 — F09, runs 037, 043)

40,000 documents (A10 VERIFIED tier), Sync durability: deterministic build →
5,000 updates + 2,000 deletes + 1,000 asserted delete-then-reinserts → 200
committed transactions (2 inserts + 1 delete each) → checkpoint →
compaction (3→1 SSTs, 301.8 ms) → coordinated backup → controlled
interruption at `ckpt_after_sst` during the final checkpoint → fresh-process
recovery. PH3E-FAULT-043 (corrective for 037, D56): recovered state equals
the independent model EXACTLY — 38,200/38,200 documents, full uuid-set +
per-doc content comparison (no sampling), zero missing acked (37,800-strong
acked live set), zero unacknowledged leakage, no duplicate live ids, checker
clean, retrieval self-hit 48/50 (≥95% gate), restore-from-backup equals
source (38,200/38,200 + clean checker), and THREE consecutive restarts
byte-identical (state hashes equal). 037 is preserved INVALIDATED (D56: the
harness "reinsert" minted second live uuids — the E10 SCALE-014 trap — and
the model's acked set went stale; the engine's state was verified exact even
there). The complete single-node lifecycle remains within the documented
contract at the largest fully-verified tier.

## 15. Defects Found, Fixed, and Unresolved

Engine defects: **ZERO.** No sealed contract was violated by the
implementation under any injected fault. Harness/expectation defects: seven,
all preserved INVALIDATED with corrective new-ID reruns that VERIFIED —
007 expectation defect (D49, → 038), 022 gate-hit consumption (D51, → 039),
024 crashgate allowlist missing E11 names (D53, → 040), 032 degenerate
2-sparse dataset defeating the retrieval gate (D54, → 042), 034 missing F07C
dispatch (D55, → 041), 037 reinsert-without-delete + stale acked set (D56,
→ 043), 035 model saved before the delete stage (D57, → 044). One tolerance
was measured and AMENDED into the contract (A11): regressed-sidecar opens
with full recovery. Unresolved in-scope defects: **none** (E11 is not
incomplete on any contract). New regressions pin the closure: gate inertness
without env, sidecar-regress tolerance, torn-tail prefix recovery, and the
dual-live-uuid documented semantics (4 tests; suite 334/0).

## 16. Invalidated and Blocked Runs

INVALIDATED (7): 007, 022, 024, 032, 034, 035, 037 — all harness or
expectation defects (D49–D57), each linked to its successful corrective run
(038, 039, 040, 042, 041, 044, 043 respectively). FAILED: none. BLOCKED:
none — every family executed. UNSUPPORTED as test classes: physical
power-loss (no controlled source; §17). The failure register table is
generated from raw summaries (`tables/table-e11-failure-register.md`).

## 17. Power-Loss and Hardware-Failure Limitations

Mandatory statement: every E11 fault test used PROCESS-DEATH semantics
(SIGABRT / process-group SIGKILL). No physical power-loss, no machine crash,
no VM reset, and no storage-fault-injection hardware was used. fsync calls
were exercised at the syscall level (the after_fsync gate boundary is real),
but fsync semantics were NOT validated against every hardware/filesystem
failure mode. Sync durability beyond process death remains PARTIALLY VERIFIED
exactly as A3 recorded it, and is here recorded as UNSUPPORTED-to-claim
beyond the tested model. No E11 text labels any process-kill test a
power-loss test (checker-enforced).

## 18. Final Regression Results

At the final tree: `cargo test --release --workspace` → **334 passed / 0
failed** (330 sealed E1–E10 + 4 new E11 regressions:
`e11_fault_gates_inert_without_env`,
`e11_regressed_sidecar_opens_with_full_recovery`,
`e11_torn_tail_recovers_intact_prefix`,
`e11_upsert_dual_live_uuid_is_documented_semantics`); `cargo clippy
--release --workspace --all-targets -- -D warnings` → 0 errors; phase-2
checker PASS; phase-3 checker PASS including the new gate 28 (E11 evidence:
registry↔raw-dir bijection, per-run required artifacts, classification
vocabulary, corrective links, report drift, terminal-status line, A11
presence, E11 regression file presence); E10 generator + gates green
(gate 27). Exact final HEAD, tree hash, and commands are recorded in the
commit message and §21 of the verdict document.

## 19. Evidence Traceability

44 runs, each with config.json, environment.json, fault-plan.json,
ack-log.jsonl, reference-model.json, exit-status.json,
recovery-verification.json, summary.json, stdout/stderr logs, checksums.sha256
(metadata manifest), and payload inventories + checksums; F01 additionally
keeps pre/post tamper inventories. Every fault carries proof its boundary was
reached (SIGABRT exit with no clean-end marker, or the groupkill marker + 
controller SIGKILL; F05/F06 additionally require the fault to land inside the
call — completion notes must be absent). All classifications are
artifact-derived (the register recomputes fault-landing strictly; summary
disagreements produce classification-correction.json records — raw summaries
were never rewritten). Tables/figure regenerate via
`generate_results_ph3e.py::gen_e11` from raw artifacts only; runs re-execute
via `research/phase3/e11_drive.py`. Registry: 208 entries; index ↔ run-dir
bijection checker-enforced.

## 20. Explicit Non-Claims

No power-loss, machine-crash, or VM-reset durability claim. No claim beyond
the tested fault points (30 named boundaries; unreached interior states are
not thereby verified). No isolation, linearizability, serializability, MVCC,
or distributed claim. No claim that Async preserves acked writes across
process death. No production-readiness claim (readiness remains the A1–A11
capability matrix, never a score). No performance claims from E11. The E10
scale envelope is unchanged and was not extrapolated. GroupCommit batching/
coalescing behavior is unclaimed (AMB-1). A partial-backup refusal is
verified for the tested tamper shapes, not for arbitrary byte-level backup
corruption.

## 21. E11 Verdict and Stop-Condition Checklist

All §16 completion criteria verified: E10 baseline verified (§2); contracts
recovered and cited (§4); methodology + matrix frozen pre-run (§5); all nine
families executed — none blocked (§6–§14); every official run registered and
immutable (§19); every injected fault has boundary proof (§19); acks
independently tracked (M7); reference-model comparisons completed for every
correctness claim (full-state where claimed); every in-scope defect resolved
(none existed in the engine; all harness defects corrected with new-ID
reruns, §15); invalidated runs preserved (§16); corrective runs use new IDs;
durability claims match the process-death model (§7, §17); power-loss
limitations preserved (§17); E1–E10 regressions green (334/0 total, §18);
new E11 regressions pass; both checkers PASS (§18); final report, verdict,
and deviations complete; tables/figures regenerate from raw evidence (§19);
no unsupported production-readiness or hardware-durability claim anywhere
(§20); final repository state and commands recorded (commit message + verdict
§21).

**PHASE 3E E11 COMPLETE — FINAL RELIABILITY VERDICT ISSUED**
