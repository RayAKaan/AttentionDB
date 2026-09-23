# Phase 3E — Final Reliability Verdict (E11 Closure)

Author: Rayyan Kaan (RayAKaan). This is the terminal Phase 3E closure
document. Evidence: 208 registered runs (PH3-QUAL → PH3E-FAULT-044), sealed
contracts A1–A11, reports E0–E11, and both consistency checkers PASS at the
final tree.

## What AttentionDB's single-node contracts ARE — verified, with the exact failure model

| Capability | Classification | Verified failure model & tier | Evidence (primary) |
|---|---|---|---|
| Logical state correctness vs independent model | VERIFIED | mixed op streams + restarts + compaction; E11: full-state equality after fault-recovery at 20–40k docs | G1; PH3D-STATE-001..003; PH3E-FAULT-043 |
| WAL integrity: refuse corruption | VERIFIED | byte-level tamper: corrupt body, bad magic, seq gap, missing segment → open REFUSES | G5/A1; PH3E-FAULT-003..006 |
| WAL torn tail | VERIFIED | intact prefix recovers, loss = destroyed bytes, deterministic | G4; PH3E-FAULT-002 |
| WAL legacy / derived-bookkeeping tolerance | VERIFIED (measured tolerance, A11) | absent sidecar tolerated; regressed sidecar watermark opens + full recovery (records authoritative) | A11; PH3E-FAULT-007/008/038 |
| Sync durability (process death) | VERIFIED | SIGABRT/SIGKILL at after_write/after_flush/after_fsync/before_ack; acked ⊆ recovered always | G3/A2; PH3E-FAULT-009..011 |
| GroupCommit durability (process death) | VERIFIED | ack ⇒ page-cache flush survived process death | G3/A2; PH3E-FAULT-012 |
| Async durability boundary | SUPPORTED (documented boundary) | loss confined to un-promoted tail; zero unacked leakage; flush/checkpoint promote | G3/A2/E8f; PH3E-FAULT-013/014 |
| Transaction atomicity | VERIFIED | commit boundary = WAL marker; all-or-nothing at 4 fault points; rollback discards | G6/A6; PH3E-FAULT-015..018 |
| Checkpoint crash safety | VERIFIED | 5 interior gates (SST/trim/rotate/manifest-rename/SST-write); acked state + checker + deterministic restarts | G1/G4; PH3E-FAULT-019..023/039 |
| Backup: partial never valid; restore exact | VERIFIED | crash mid-copy and pre-manifest → restore REFUSES; clean restore = exact state; source intact | G8/A4; PH3E-FAULT-024..026/040 |
| Compaction crash safety | VERIFIED | 4 interior gates + clean reclamation; deletions survive every point | G9/A5; PH3E-FAULT-027..031 |
| Index hygiene & D40 predicate | VERIFIED | fresh multi-head ckpt: no rebuild (3.1 ms); dead-dominated: legitimate rebuild (427 ms); interrupted recovery rebuild completes deterministically | A9/D40; PH3E-FAULT-032..035/041..044 |
| Concurrency (bounded) | SUPPORTED | commit serialization, no dirty reads, LWW; crash mid-write recovery exact | A7; PH3E-FAULT-036 |
| Recovery determinism | VERIFIED | triple-restart byte-identical state hashes on every fault leg | G1/G4; all recovery runs |
| Integrated lifecycle @40k | VERIFIED | build→churn→200 txns→ckpt→compact→backup→crash→recover: exact 38,200/38,200, checker, self-hit 48/50, restore = source | A10 tier; PH3E-FAULT-043 |
| Scale envelope | VERIFIED (A10, unchanged) | 40k primary / 60k slim+integrated / 80k with caveats; ≥100k blocked on the tested host | A10; PH3E-SCALE-001..021 |

## Under which failure model

Process/container-VM abrupt termination at instrumented boundaries (A3):
SIGABRT at 30 named persistence gates and process-group SIGKILL at
marker-parked stages; plus deterministic file-level WAL tampering (F01).
PHYSICAL POWER-LOSS, machine crash, and VM-reset equivalence: **UNSUPPORTED
— not tested, not claimed** (A3/A11; checker-enforced language).

## At what tested resource envelope

2 vCPU / 1.9 GiB RAM host (Linux 6.1, ext2/ext3, rustc 1.98.1). Fault-tier
workloads: 20–25k docs per family; integrated lifecycle 40k (A10 VERIFIED
tier). The E10 scale envelope is preserved unchanged and bounds all claims.

## Which boundaries remain

- Memory ceiling ~95–100k docs on the tested host (E10; kernel-OOM evidence).
- Async acked-write loss across process death (documented, boundary-verified).
- 80k grown-graph recall nondeterminism at the 95% gate, restored by rebuilds.
- Reopen cost grows with doc count (deterministic-rebuild recovery contract:
  ~0.7–1.05 ms/doc measured).
- Scan-heavy reader throttles the writer ~30× at 60k (measured, not optimized).
- Backup is quiescent (mutation gate), not nonblocking.
- Sidecar watermark regression is tolerated (A11) — safe by measurement, but
  the sidecar must not be treated as an integrity authority.

## Which known defects remain unresolved

**None in scope.** E11 found zero engine defects. Seven harness/expectation
defects were found, preserved INVALIDATED, and closed by corrective VERIFIED
reruns (D49–D57). The suite is 334/0 with the four new E11 regression pins.

## Which fault classes remain untested or unsupported

Physical power-loss / machine crash / storage-hardware faults (UNSUPPORTED —
no controlled source). Unreached interior states of multi-phase operations
(e.g., mid-fsync instants between named gates) are not thereby verified.
Arbitrary byte-level corruption of backups (only manifest-less partials and
WAL tamper shapes tested). VM-reset equivalence untested.

## What may AttentionDB accurately claim after Phase 3E

Within the tested resource class and the process-death failure model:
acknowledged Sync/GroupCommit writes survive sudden death at every tested
acknowledgment boundary; transactions never appear partially applied;
corruption refuses rather than fabricates; torn tails recover the intact
prefix; interrupted checkpoints/compactions/backups/restores/recovery-
rebuilds recover to the contract state deterministically; partial backups
are refused; the hygiene predicate behaves exactly as specified; and the
full single-node lifecycle holds at the 40k verified tier with full-state
equality. Claims are bounded by the capability matrix above — never a score,
never "production-ready" (no evidence-backed production-readiness standard
exists in this repository, so none is declared).

## What must explicitly NOT be claimed

Physical power-loss or hardware-level durability; any claim beyond 80k docs /
260k ops / dim 256 / 4 heads on this host class; isolation levels,
linearizability, serializability, MVCC, distribution; Async ack survival
across process death; recovery-latency bounds beyond measured tiers;
production readiness as a binary verdict; any behavior at untested fault
points.

## Final verification record (exact commands, final tree)

- `cargo test --release --workspace` → 334 passed / 0 failed
- `cargo clippy --release --workspace --all-targets -- -D warnings` → 0 errors
- `python3 research/phase2/verify_consistency.py` → PASS
- `python3 research/phase3/verify_consistency.py` → PASS (gates 1–28 incl. E11 evidence gates)
- `python3 research/phase3/generate_results_ph3e.py` → regenerates all results/tables/figures from raw evidence
- Registry: 208 entries; E11 runs PH3E-FAULT-001..044; final HEAD/tree sha recorded in the closing commit message.

**PHASE 3E E11 COMPLETE — FINAL RELIABILITY VERDICT ISSUED**
