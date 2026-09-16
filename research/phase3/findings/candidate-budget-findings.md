# PH3C candidate-budget findings (PROVISIONAL)

Scope: AG News ladder runs with pools generated at K_max=200 and budgets
{10,25,50,100,200} taken as exact per-head prefixes; the gate is trained
once per seed at K=100 pools (the §11 default) and evaluated at every
budget (protocol documented in the run configs). K was never tuned on
test (§16).

## CB-1 (SUPPORTED, scope: tested configs) — Gating quality plateaus IMMEDIATELY: K=10 pools already deliver ≈ full gating quality

Trained-gating R@10 across K=10/25/50/100/200 (4-head 10K run): 0.7393 /
0.7393 / 0.7377 / 0.7410 / 0.7407 — a ≤ 0.004 band. Candidate recall still
climbs steeply (0.787 → 0.877) while final quality does not: the gate is
view-SELECTION-bound, not pool-depth-bound, on this workload.

## CB-2 (SUPPORTED, scope: tested configs) — The residual quality gap decomposes into candidate absence (34%) and in-pool ranking (66%), with view-selection errors a measurable sub-part (19% of all GT misses)

PH3C-HEAD-001-H4 (10K, K=100, seed-42 gate): of 164 GT misses, 56 (34.2%)
were absent from ALL head pools (candidate generation), 108 (65.9%) were
pooled but ranked outside the fused top-10, and 31 (18.9% of all misses)
were present in the DEFINING head's top-10 — i.e., a perfect view-selector
would have surfaced them (gating-attributable subset). The three-way
decomposition (§17) is emitted per run in candidate_decomposition.csv.

## CB-3 (SUPPORTED, scope: tested configs) — Oracle-head headroom remains large (gating ≈ 0.71–0.74 vs oracle 0.76–0.78 at 5K/10K), and candidate recall bounds it from above

Even a perfect per-query view selector cannot exceed pool coverage
(0.86–0.89 at K=100): closing the gap therefore requires BOTH candidate
improvement (§25 systems work) AND better per-query view selection —
neither alone reaches the ceiling. This quantifies the §21 separation.

## OPEN QUESTION

- The val-optimal production default K: val data shows gating flat, so
  the default can be small (K=25–50 candidates per head) with negligible
  measured quality cost and ~proportional latency savings — but a formal
  val-based selection across more configs is pending (§16 "use validation
  to select any proposed default"; the measured flatness is the evidence,
  the default itself is deferred to the systems phase).
