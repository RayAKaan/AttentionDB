# C3 — Closure Decision

Study `comparative-study-001` · C3 pilot stage · protocol v1.0.0

## What was decided

1. **Pilot closed as PASS.** All 7 registered pilot runs executed under the
   frozen plan (5 fresh-process reps, warmup 20, seeded order, paired subset),
   all `PASS`, all registered in `raw/RUN-INDEX.yaml`. Gate audit (see
   `c3-gate-audit.md`) METs every applicable gate with a single disclosed,
   bounded deviation (SciFact own-hashes).

2. **Deviation carried by user decision (host-as-execution-envelope, SciFact
   on-own-hashes):** no blocker; recorded, disclosed, accepted for this stage.
   The cross-env drift bound (~2–3e-7) does not affect recall/latency read-outs
   at the reported precision; the study remains valid on-own-hashes.

3. **Confirmed-scale execution (C4) is NOT started.** The pilot established
   feasibility, the execution machinery (driver `c2pilot` + orchestrators),
   honest ceilings, and per-mode pilot signals; expansion to the full eligibility
   matrix requires a new approved plan/goal, not an extrapolation of this one.

## Pilot signals carried forward (see `c3-execution-report.md`)

- B2 (FixedFusion) beats the single-view exact ceiling on NFCorpus qrels
  (0.168 vs 0.142): multi-view union is worth pursuing (RQ2/RQ6).
- B7 (Full default) is the laggard at highest latency on this dataset
  (0.155 @ 2.2 ms) — "production default evaluated as-is."
- B3 (LearnedGating) transfers the NFCorpus-derived LODO card to SciFact-headline
  with §15-invariant enforcement clean (0.813 qrels recall) — strongest RQ3
  signal.
- Synthetic diagnostics flagged no generator/harness soundness defect (GT vs
  oracle 200/200 in both cells).

## Guardrail / resource standing

All runs fit comfortably inside the host envelope (peak child RSS ≤ 304 MB vs
~13.5 GB threshold). No OBSERVED-LIMIT/OOM classification anywhere.
Harness corrections (sampler semantics, 3-head collection config) are preserved
pre-measurement and ledgered; the engine's §15 refusal caught the B3 config bug
(as designed), which validated INV-8 in situ.

## Artifacts finalized by this decision

- `c3/c3-run-plan.csv` → statuses updated to PASS.
- `c3/c3-execution-report.md`, `c3/c3-gate-audit.md`, `c3/run-reconciliation.csv`.
- `raw/C3-W01-NFC-B0-001|B1|B2|B7|C3-W03-SCI-B3-001|C3-W08-SYN-B1-001|C3-W10-SYN-DUP-B0-001/`
  (7 run dirs, each with environment.yaml, config, 5 per-rep JSON, metrics,
  status.txt). `raw/C3-*_ABORT-ATTEMPT-*` / `*_CONFIG-CORRECTION-0` evidence dirs.
- `raw/RUN-INDEX.yaml` append-only ledger reflects all of the above.

## C4 boundary (do not cross without new intent)

No further execution without an explicit new task from the operator. If a
confirmed-scale run is later approved, it must pre-register fresh run IDs,
validation-split treatment for ±0.01 recall claims, and paired-bootstrap
analysis — none implied by this pilot.