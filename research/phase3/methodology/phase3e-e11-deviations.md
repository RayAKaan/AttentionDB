# Phase 3E — E11 Deviations & Adaptations

## D49 — PH3E-FAULT-007 INVALIDATED: pre-run expectation defect (sidecar regress)
The run pre-declared "refuse" for a regressed `wal-state.json` high-watermark;
measurement showed the engine opens and fully recovers (records are
authoritative; the sidecar is derived bookkeeping). Per the immutability rule
the run is preserved INVALIDATED (wrong expectation, not wrong engine), the
contract-matrix ambiguity AMB-0 was resolved BY MEASUREMENT, and the
corrective run PH3E-FAULT-038 (expectation: open + exact recovery) VERIFIED.

## D50 — F01 sidecar cases require a checkpoint in the build
`wal-state.json` only exists after a rotation/checkpoint and lives in `WAL/`
(not the db root; initial path bug panicked — fixed). Sidecar tamper cases
therefore build with one checkpoint + post-checkpoint records, so both the
sidecar and a live WAL exist (runs 007/008/038).

## D51 — PH3E-FAULT-022 INVALIDATED: gate hit consumed by setup-time manifest write
`manifest_after_current_rename` hit=1 fired during the catalog manifest write
at collection creation — BEFORE the workload and model flush — so the fault
did not land at the checkpoint boundary (proof: no CKPT ack note, model file
absent). Preserved as INVALIDATED; corrective PH3E-FAULT-039 uses hit=2 with
controller-verified crash position (model present + full ack count).

## D52 — Harness iteration before the first OFFICIAL registration
The smoke phase (pre-registration) fixed five harness defects: model-flush
discipline (model must be on disk BEFORE any fault point), number-vs-string
field read, in-flight-op allowance (`allowed_unacked`, single-threaded writer
⇒ at most one op may be durable-but-unacked), torn-tail intact-prefix
comparison shape, and stale-binary discipline (build AFTER every harness
edit, verified by the GATE_ARMED line). No official run IDs were consumed by
smoke testing (IDs 001-014 were re-executed cleanly after the fixes).

## D53 — PH3E-FAULT-024 INVALIDATED: crashgate allowlist predates the E11 gates
`crash_cfg()` validates the requested gate name against the pre-E11 allowlist;
the four E11 gate names were absent, so the gate never armed and 024 ran clean
(BACKUP_OK in the ack log proves the boundary was passed, not crashed). Fixed
by extending the allowlist (test-only list, same contract); strict
fault-position verification added to the controller AND the register (a
planned-fault run whose gate never fired is INVALIDATED by artifact evidence,
never by the summary's own claim). Corrective run: PH3E-FAULT-040.
PH3E-FAULT-025 (killed mid-run when the batch was stopped, never registered)
was re-executed under its own ID once the fix was live.

## D54 — Dataset defect: degenerate 2-sparse vectors (fixed before official gate runs)
The initial E11 generator used 2-sparse unit vectors (two nonzero coordinates
from a 32-dim basis); with 800+ documents this makes top-1 self-hit a
bucket-tie lottery (all bucket-mates are near-identical). PH3E-FAULT-032
measured 39/50 — recorded as the detection evidence — and the run is
preserved INVALIDATED (harness dataset defect). The generator was replaced
with the E10 clustered scheme (50 clusters, 0.8·centroid + 0.2·deterministic
noise) BEFORE the retrieval-gated official runs (037 F09, 042 corrective).
Runs that never gated on retrieval (counts/durability/atomicity) are
unaffected by the change.

## D55 — PH3E-FAULT-034 INVALIDATED: F07C missing from the workload dispatch
The writer panicked `unknown family F07C` (the F07C arm was added to the
recovery design but never to the workload match); recovery legs then found no
model file. Preserved INVALIDATED; corrective PH3E-FAULT-041.

## D56 — PH3E-FAULT-037 INVALIDATED: reinsert-without-delete + stale acked set (harness)
F09's "reinsert" stage inserted ver-3 documents on the still-live ver-1 idxs
(1,000 second-live-uuid pairs — the E10 SCALE-014 trap, engine exonerated
there by the D41 minimal probe), and the model failed to prune
`acked_live_idx` on churn/txn deletes (the phantom 2,200 "missing acked").
Counts were exact (38,200==38,200), checker clean, zero leakage, self-hit
49/50 — the failures were entirely model-bookkeeping. The compaction/backup
evidence from 037 stands (301 ms compact 3→1 SSTs). Fixed harness (asserted
delete-before-reinsert; acked-set pruning), rerun as PH3E-FAULT-043. The
engine's dual-live-uuid semantics are now pinned by
`e11_upsert_dual_live_uuid_is_documented_semantics` so they cannot silently
change.

## D57 — PH3E-FAULT-035 INVALIDATED: model saved before the delete stage
F07D's reference model was flushed before the 24k deletes, and the ckpt2
crash (the scenario's fault) prevented the final save — recovery compared a
correct 1,000-doc recovered state against a stale 25,000-doc model. The
engine's recovered state was CORRECT (the observed failure is the mirror of
D52's lesson: the on-disk model must reflect the state at the fault point).
Model save moved after the deletes; corrective PH3E-FAULT-044.

## D58 — Evidence-retention pass under the workspace cap
Raw E11 db payloads were retained in full where they ARE the evidence (F01
tamper states, all INVALIDATED runs at registration time), inventoried and
checksummed per directory (M10). After registration reclassified five runs
VERIFIED via artifact-derived corrections, their regenerable payloads were
trimmed with in-run notes; the two 25k/40k-class failure payloads (037/041)
and the duplicated restored snapshot of 037 were compressed/removed with
per-run notes after per-file inventory + checksum capture. Sealed E9/E10 run
dirs were NOT touched. Workspace at E11 close: 134 MB / 3,191 files — 6 MB
over the ~128 MB snapshot guidance, the remainder being .git history (65 MB,
includes the E10-committed-then-pruned phase2b/c/d data), frozen phase2
(16 MB), and sealed E9/E10 evidence; file count is far under the 10k cap.

## D59 — Post-run lint repairs (no behavior change)
After the last official run (044) closed, two clippy lints in the D54
generator function were repaired (loop style + vec allowance). The change is
semantically identical; the recorded noise sequence is unchanged. The official
runs' provenance is their registration-time tree; the final report quotes the
post-repair tree.
