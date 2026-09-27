# C4.6 — Gate Audit (C4-G1..C4-G20)

Audit date: 2026-09-26. Evidence base: immutable raw runs (`raw/C4-*`),
RUN-INDEX.yaml (append-only), plan CSV, eligibility/validation docs, c2pilot
binary `3ec8c193…` (deadline-capable deterministic /Brepro build).

| Gate | Requirement | Status | Evidence |
|------|-------------|--------|----------|
| C4-G1 | C3 closure provenance OK | PASS | branch cut from C3 commit `5195133ca5…` (tree `93c1ccdef076aeca`), recorded in `c4-plan.md` §1 |
| C4-G2 | Host/toolchain capture | PASS | `c4-environment.md` (rustc/cargo 1.96.0, /Brepro determinism, host, mem snapshot per run) |
| C4-G3 | System eligibility with evidence | PASS | `c4-system-eligibility.csv` (SYS-ADB / B0 / B3 ELIGIBLE; externals CONDITIONAL/BLOCKED) |
| C4-G4 | Dataset/embedding hashes | PASS | `C4DATA-VERIFY-001` PASS 18/18; `c4-dataset-validation.md` |
| C4-G5 | Exact oracle + metric validated | PASS | C2 oracle battery + C3 0/100 exact-top10; B0 TEST oracles run 2026-09-26 (SCI 0.7833, NFC 0.1550, exact_top10 hashes recorded) |
| C4-G6 | Track A/B reporting separated | PASS | TRKB cells run and reported separately (`C4-W07-SCI-B7-TRKB-001`, `C4-W01-NFC-B1-TRKB-001`); never merged with Track A table |
| C4-G7 | Confirmed workloads from C1 registry | PASS | W-01,W-02,W-03,W-04,W-07,W-19 executed as frozen plan rows; 5 X-* cells remain BLOCKED (external systems) in plan |
| C4-G8 | Parameter matching/fairness | PASS | Matched configs frozen in `c4-validation-tuning-results.md`; reflected verbatim into `c4_test_run.py` REFLECT table; TARGET-UNREACHABLE (SCI-B2, SCI-B7) run with plan defaults, flagged, never interpolated |
| C4-G9 | Resource guardrails enforced | PASS | per-run RSS sampler (500 ms, abort when child-tree RSS ≥ 85% preflight MemAvailable); no aborts fired; runtime sha256 recorded in every `environment.yaml` (15/15 `3ec8c193…`) |
| C4-G10 | Pre-execution registration | PASS | RUN-INDEX append-only; each run's `status: RUN` entry precedes measurement; all 15 test cells have RUN→PASS pairs |
| C4-G11 | Failed runs preserved, new-ID reruns | PASS | no run rewritten; the one aborted bootstrapping attempt (`C4-W01-SCI-B0-001` pre-measurement key error deletion) removed before measurement; no partial PASS artifacts lost |
| C4-G12 | Valid artifacts + hashes per PASS cell | PASS | 15/15 cells: environment.yaml + metrics.json + status.txt PASS + per-rep JSON; 15/15 sha256 == `3ec8c193…` |
| C4-G13 | Statistical pipeline reproducible | PASS | `c4_statistical_analysis.py` derives from raw per-query rows only; seeded bootstrap; Holm within family; `c4-statistical-analysis.md` |
| C4-G14 | Reporting verifier rejects incomplete evidence | PASS | all 13 engine/oracle TEST cells + 2 boundary cells PASS before any status written; reconciliation `run-reconciliation.csv` 15/15 OK |
| C4-G15 | C0/Phase3E untouched; B3 leak guards | PASS | only c2pilot deadline arm (additive `deadline_us`) built; B4 identity-init arm unchanged; B3 card frozen `b3-lodo-nf-v2-s20260925-h32-lr0.01.json`; TEST splits never tuned |
| C4-G16 | Auditable go/no-go from pilot evidence | PASS | C4.2 matched configurations carried forward; go decision documented in `c4-validation-tuning-results.md` |
| C4-G17 | Confirmed-scale query sets frozen (300/323) | PASS | SciFact qrels/test.tsv = 300 distinct qids; NFCorpus qrels/test.tsv = 323 distinct qids; every TEST cell ran its full frozen set (checked in reconciliation: nq == plan query_count) |
| C4-G18 | Per-query schema + multi-head cost accounting | PASS | every row records candidate_count/heads_present/latency_us/recall10_* ; multi-head cells emit head-level fields |
| C4-G19 | Equal-budget/equal-quality tables honest | PASS | BUDGET-CAND:500 never exceeded (max candidate_count 128, mean 50–97); BUDGET-MEM:512MiB respected (max peak RSS 305.1 MiB); MEM legs {512MiB,1.0GiB,1.6GiB} all within budget; TIME legs {1,10,100ms} 0 deadline misses; UNREACHABLE rows flagged |
| C4-G20 | Closure decision with evidence | PASS | `c4-closure-decision.md` |

## Terminal observations

- Per-query latency p50 across engine cells: 0.77–2.17 ms; p95 1.11–4.15 ms
  (B7 Track-A/B at p95 up to 4.15 ms from B7-TRKB rep variance).
- `C4-W07-SCI-B7-TRKB-001` rep-level sd 0.0262 is the largest within-cell
  spread (single fresh-process rep at 0.6822). Per C1 rule no outlier removal;
  flagged for the report and C4.5 power notes.
- All B0 exact oracles stable across independent processes (repeatable).
- Total wall-clock for C4.4 execution ≈ 51 min (first RUN registration → last
  PASS completion in RUN-INDEX; sum of per-cell durations ≈ 49 min).

Audit conclusion: 20/20 gates PASS. No open defects blocking closure.