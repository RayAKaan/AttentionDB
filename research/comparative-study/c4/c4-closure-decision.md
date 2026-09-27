# C4.7 — Closure Decision

Date: 2026-09-26
Branch: `comparative-study/c4-confirmed-benchmark`

## Decision: COMPLETE

C4 confirmed-scale benchmark execution closes as **COMPLETE** (not BLOCKED,
not PARTIAL). Engine: c2pilot `3ec8c193…` (deadline-capable deterministic
/Brepro build; C4-BINVERIFY-003 PASS 4/4).

## What was delivered

- 13 confirmed-scale TEST cells (9 primary engine cells + B0 oracles + 2
  Track-A/B cells) + 2 boundary cells (MEM 3-leg, TIME 3-leg), all registered
  in RUN-INDEX as PASS and reconcilable against the frozen plan
  (`run-reconciliation.csv` 15/15 OK).
- Preregistered statistical analysis (paired per-query, mean-of-5-reps
  estimator to absorb fresh-process variance, 10k seeded bootstrap CIs,
  Wilcoxon, Holm within family, Cohen's dz) written to
  `c4-statistical-analysis.md`.
- Budget matching (C4.3): candidate / memory / time budgets never exceeded;
  BUDGET-CAND:500, BUDGET-MEM:512MiB, MEM legs, TIME deadlines all clean.
- Reports and gate audit: `c4-execution-report.md`, `c4-gate-audit.md`
  (20/20 gates PASS).

## Headline findings (no-winner-labels framework)

| dataset | B0 | B1 | B2 | B3 | B4 | B7 |
|---------|----|----|----|----|----|----|
| SCI recall@10 | 0.7833 | 0.7783 | 0.7920 | 0.7911 | 0.7918 | 0.7421 |
| NFC recall@10 | 0.1550 | 0.1491 | 0.1597 | 0.1593 | — | 0.1488 |

- B2-vs-B7: significant on both datasets (SCI +0.0498, p_holm 0.0004; NFC
  +0.0109, p_holm 0.0133), B2 higher.
- B2-vs-B3 and B3-vs-B4: negligible paired differences on both datasets
  (|diff| ≤ 0.0037) — supports treating B3 ≈ B2 and B4 ≈ B3 within the
  negligible band.
- B1-vs-B2: not significant after Holm (SCI p_holm 0.7051, NFC p_holm 0.4584)
  but CI overlaps ±0; effect sizes small; power noted.
- B4 tested only on SciFact (frozen no-op-arm scope); NFC B3-vs-B4 recorded
  NOT-EXECUTABLE, never interpolated.
- 5 external-system cells (X-*) remain BLOCKED in the plan (not runnable in
  this environment); they are **out of scope** for C4 and do not gate closure.

## Integrity statements

- No TEST/candidate tuning happened on test splits; all configs matched on
  validation (C4.2) and frozen before C4.4.
- Every PASS cell has immutable per-query JSON, metrics.json, environment.yaml
  (binary sha), status.txt; RUN-INDEX is append-only (RUN → PASS pairs).
- No outlier removal; the B7-TRKB 0.6822-in-5 rep and its sd 0.0262 are
  reported transparently.
- One data-loss incident (eligibility CSV truncation) was fully recovered
  from git + reconstruction; documented in the execution report §8.

## Remaining downstream (handoffs, recorded not executed here)

- C0/Phase3E: unaffected (no design-leak rebuild; deadline arm additive only).
- Fairlight/hardware/scale phases: future work, require new plan cells;
  not drafted here.

## Committed artifacts

Final commit contains: engine c2pilot.rs (B4 identity-init + additive
`deadline_us` arch), `.cargo/config.toml` (/Brepro), Cargo.toml comment,
harness scripts (`c4_test_run.py`, `c4_statistical_analysis.py`), raw evidence
(run dirs + RUN-INDEX.yaml), docs (`c4-*.md`, `c4-system-eligibility.csv`,
`run-reconciliation.csv`).