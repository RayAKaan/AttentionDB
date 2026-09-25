# C3 — Gate Audit (executed evidence)

Study `comparative-study-001` · C3 pilot stage · protocol v1.0.0
Audit conducted against the executed pilot evidence, `raw/RUN-INDEX.yaml`,
`c3/c3-run-plan.csv` (now PASS-ed), and per-run `environment.yaml`/`metrics.json`.

## Audit items

### Gate G1 — Hunt eligibility / requirements of protocol §6 (datasets)
- DS-NFCORPUS: inputs re-verified on this host by `C3DATA-NFC-VERIFY-001` (PASS;
  7/7 checks: corpus, qrels/queries, 5 embeddings vs C2 manifests). ELIGIBLE.
- DS-SCIFACT: inputs = `C3-EMBED-SCIFACT-001`, valid ON-OWN-HASHES; cross-env
  float drift vs C2 manifest disclosed and bounded (max_abs ≈ 2–3e-7) per the
  re-procurement decision. ELIGIBLE-ON-OWN-HASHES.
- DS-SYNTH-PH2B: in-repo deterministic generator (`c2probe corpus`); generator
  GT verified against a Python-side brute-force oracle 200/200 (both cells).
  DIAGNOSTIC-LABEL only.
Result: MET (gate passes with disclosed deviation on SciFact hashes).

### Gate G2 — Resource guardrails (`c1/environment-guardrails.md`)
- Preflight snapshot present in every run's `environment.yaml` (MemAvailable,
  mem load, disk not part of pilot inputs).
- Live guardrail: 500 ms sampler aborts on child process-tree RSS ≥ 85% of
  preflight MemAvailable. Observed peaks 2–304 MB vs ~13.5 GB budget → no
  resource-boundary classification anywhere. No OOM, no abort during
  measurement (as confirmed by per-rep full completion).
- PH3E crash env vars not set by harness (harness writes its own env; no
  crashgates activated — verified in initial C3 preflight).
Result: MET. (One harness defect during bring-up — sampler read host-global
memory load, corrected; evidence preserved as ABORT-ATTEMPT-*, see §6 of the
execution report. No run was classified OBSERVED-LIMIT / OOM-LIMIT.)

### Gate G3 — Fair comparison (statistical plan + charters)
- Paired per query: identical seeded subsample (100) across all four NFCorpus
  cells; identical seeded per-process query order; warmup 20 queries executed
  but excluded from latency stats. Verified byte-identical config.subsample.
- 5 fresh-process reps per real-data cell; seeded order identical across reps
  → variance is per-process, not order-driven.
- Same engine build (`c2pilot` sha in each environment.yaml), same input view,
  same k. Approximate-index tail variance (~0.92 Jaccard top-10 across reps,
  recall means tight 0.800–0.825) is real measurable variance, captured by the
  rep protocol — not a fairness skew.
Result: MET.

### Gate G4 — Soundness / silence-wrong-result
- Synthetic cells: generator-GT vs brute-force oracle exact equality 200/200 in
  both cells → no silent wrong-result indicator.
- All reps exit 0, `ok:true`, per-query rows finite.
- B0 exact oracle cross-used as reference by every engine cell
  (`recall10_exact`).
Result: MET.

### Gate G5 — Recv/quality floor (recall ±0.01 on VALIDATION)
- This is a pilot; validation-split ±0.01 assessment is a confirmed-scale
  activity. Pilot provides ceiling (B0 exact) and mode rows but does NOT claim
  validation discrimination. No trial-model-selection or advantage claim made
  from pilot numbers.
Result: N/A at pilot scale (documented; no false claim).

### Gate G6 — Budget caps
- 12-run / 180-min budget slice; 7 pilot runs executed; wall time under an hour.
Result: MET.

## Corrections ledger (all pre-measurement, evidence preserved in raw/)
1. Sampler guardrail semantics fixed during bring-up (host-load → child RSS).
   Attempts: `C3-W01-NFC-B0-001_ABORT-ATTEMPT-{0,1,2}`.
2. B3 4-head collection rejected by engine (§15) → 3-head collection.
   Attempt: `C3-W03-SCI-B3-001_ABORT-ATTEMPT-0`.
3. B2/B7 config-corrected to 3-head collection (retrieval unaffected; attended
   heads were already H=3). Evidence: `*_CONFIG-CORRECTION-0`.

## Verdict
All gates for the pilot stage PASS with one disclosed deviation (SciFact
own-hashes). No resource boundary, no silent-wrong-result, no fairness skew.

Additional note: attentiondb engine `install_gating_card` enforces card-vs-
collection head-count equality (§15 refusal) — the no-silent-wrong-weights
invariant performed exactly as audited in C1 (INV-8).