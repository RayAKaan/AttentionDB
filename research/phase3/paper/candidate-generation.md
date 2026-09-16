# Paper section: candidate generation and budgets (PH3C)

## Setup
Budgets {10,25,50,100,200} as exact per-head pool prefixes from K_max=200
generation; gate trained once per seed at K=100 (documented), evaluated at
all budgets; K never tuned on test. Runs: PH3C-HEAD-001-{H1,H2,H4,H8}-S5K
+ PH3C-HEAD-001-H4 (10K) + PH3C-HEAD-001-{H1,H2} (10K).

## Findings (canonical: results/candidate-budget.csv,
results/candidate-decomposition.csv; figures 7–8)
- Gating R@10 is FLAT across K=10..200 (≤0.004 band at 4h/10K): view
  selection, not pool depth, binds the frozen system on this workload.
- Candidate recall climbs 0.787→0.877 over the same range while final
  quality does not — strong evidence the gate uses the head top-10s.
- Miss decomposition (H4/10K): 34.2% of GT misses never entered any pool
  (candidate generation); 65.9% were pooled but ranked out; 18.9% of all
  misses were in the defining head's top-10 (gating-attributable view-
  selection subset).
- Oracle-head headroom (gating 0.71–0.74 vs oracle 0.76–0.78) is capped
  by pool coverage (0.86–0.89): both candidate quality and view selection
  must improve to approach the ceiling; neither alone suffices.
- Val-optimal small default (K=25–50/head) is supported by flatness but
  the formal default selection is deferred to the systems phase.
