# C2 — Harness Build, Dataset Materialization, Baseline Feasibility, B3 Verification

Study `comparative-study-001`, protocol v1.0.0 (C1 content commit `963170b`,
registration stamp `7788067`). C2 converts the preregistered C1 protocol into
an executable, validated harness and determines what is actually runnable on
the declared host (2 vCPU / ~1.9 GiB RAM / ~20 GB disk). **C2 is NOT the main
comparative benchmark** — no sweeps, no comparative conclusions, no rankings.

## Layout

- `harness/` — python harness (env capture, run registry, resource sampler,
  oracle + validation, BEIR materializer, embedding pipeline, ANN
  materializer, B3 orchestration, conditional gates, external smokes, reports)
- `probe/` — standalone Rust crate (`c2probe`) driving the REAL engine:
  mode correctness battery (TEST-C2-002..009, B4 evidence, B3 activation,
  card rejection), oracle agreement, DS-SYNTH-PH2B corpus export, gating
  dataset construction (exact per-head oracle targets), training via the
  in-repo `learned::gating_v2::train_gating`
- `oracle/` — B0 exact-oracle documentation + validation report
- `b3/` — training config (preregistered grid), datasets, modelcards,
  training validation
- `adapters/` — per-baseline readiness reports
- `smoke/` — smoke registry + results
- `c2-validation-report.md` — the §33/§34 report (A–S)
- Raw, immutable runs: `../raw/<run_id>/` (+ `../raw/RUN-INDEX.yaml`);
  datasets + downloads: `../raw/datasets/`

## Key C2 facts (see c2-validation-report.md for full evidence)

- B0 oracle validated (numpy, independent of engine): 7/7 battery PASS;
  full-recompute NN-GT verification on GloVe-25/50 (set equality 1.0).
- Engine mode battery (C2-MODES-TEST-001, current-probe re-run 2026-09-24):
  **5/8 PASS** — dense paths deterministic (score desc, id asc); failures are
  documented, not hidden: **TEST-C2-008 (B5 hybrid)** = BM25 tie-ordering
  nondeterministic per call (core/src/bm25.rs sorts with `unwrap_or(Equal)` and
  no id tiebreaker over a per-call HashMap); **TEST-C2-003/009** = strict-1e-5
  score-tolerance mismatches (~1e-3 deltas) with 100% top-k ID agreement —
  harness-tolerance observations, not engine result defects. (Earlier draft
  text claimed "7/8 PASS"; superseded by the registered run's terminal status.)
- BEIR qrels are positives-only (SciFact binary score=1; NFCorpus graded
  1/2). C1's "graded 0/1/2" for SciFact described the vocabulary, not the
  shipped files — recorded as a C1-vs-data discrepancy; acceptance rule
  unaffected.
- B3: trained with the in-repo trainer on preregistered sources only
  (synthetic PH2B multiview + LODO NFCorpus-derived), INV-L1..L5 negative
  tests PASS; engine activation validated (install, inspect, weights==softmax,
  name→head mapping, deactivate fallback uniform).

## Discipline

Run dirs are never overwritten; reruns get new IDs; failed/aborted runs are
preserved with reasons. `PH3E_CRASH_*` asserted unset in every run preflight.
No benchmark sweeps were executed in C2.
