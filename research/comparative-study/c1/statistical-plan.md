# C1 — Statistical Analysis Plan (preregistered before any C2 result)

Study `comparative-study-001`, protocol v1.0.0.

## Experimental unit & pairing

- Unit: QUERY (paired across modes/systems — identical query sets and
  embeddings per Track A; per-dataset query counts are fixed: SciFact 300,
  NFCorpus 323, ann-benchmarks 10,000 with the preregistered
  validation/test split of the query set).
- Latency unit: per-query latency observation; repetitions = 5 independent
  harness runs (fresh processes) for latency workloads; retrieval-quality
  metrics are deterministic per config (repeated only across seeds where
  training is involved: 3 seeds for B3).

## Randomization & ordering

- Query order randomized (seeded) per run; each system receives the SAME
  order per paired comparison. Warm-up: first 20 queries discarded (cached
  steady state); cold-start runs are separate workloads (W-18) with no
  warm-up discard (by design).

## Bootstrap CIs

- 10,000-resample percentile bootstrap (seeded) for means of recall/nDCG/
  MRR/latency; 95% CI; per-query resampling (paired bootstrap for between-
  mode/system differences on the same queries).

## Tests & correction

- Primary contrasts (family, preregistered): B1 vs B2, B2 vs B3, B3 vs B4,
  B2 vs B7 on each primary dataset (SciFact, NFCorpus) → 8 primary paired
  comparisons. Method: paired bootstrap difference CI + two-sided
  Wilcoxon signed-rank on per-query metric values; multiple-comparison
  correction: Holm step-down within the family. Secondary (cross-system
  Track A/B comparisons): Benjamini-Hochberg at q=0.10, reported as
  exploratory unless promoted by amendment before outcomes.
- Effect sizes: mean paired difference with 95% CI (primary); Cohen's dz
  reported where the paired design supports it. Practical-significance
  threshold declared per metric: retrieval-quality differences < 0.01
  absolute are reported as negligible regardless of p-value; latency
  differences < 5% are negligible.

## Sample size & power

- SciFact n=300: with paired per-query binary outcomes (recall@10), the
  study can detect (α=0.05 two-sided, Holm-adjusted, power 0.8) a paired
  difference of roughly ≥0.08-0.10 proportion depending on discordance —
  computed exactly post-hoc from observed discordant counts and REPORTED
  with every non-significant result (no "no difference" claims from
  underpowered cells; those are reported inconclusive).
- Latency: 5 repetitions × full query set; run-to-run variance reported as
  the between-run std of p50/p95.

## Outliers, failures, missing data

- No outlier removal. Timeouts/deadline misses are FAILURES: reported as
  error rate, excluded from latency aggregates ONLY with the exclusion
  count in the same cell (preregistered, uniform rule). OOM/abort → run
  classified OBSERVED-LIMIT/FAILED per guardrails; never dropped silently.
- Missing metric (e.g., candidate recall not exposed by an external API):
  n/a with reason; no imputation.

## Stopping rules & raw preservation

- No optional stopping: each preregistered workload runs its full planned
  repetitions unless an abort criterion fires (guardrails) — in which case
  the partial run is preserved and classified.
- Raw per-query rows (query id, per-metric value, latency, timestamps,
  errors) are immutable run artifacts; all aggregates are derived by the
  analysis scripts from these rows (charter §10/§11). Any rerun = new run
  ID; unfavorable runs are never excluded (INV rule: post-hoc exclusion
  without a preregistered rule invalidates the analysis).

## Implementation note

The existing `attentiondb-bench/src/stats/` stack (bootstrap CI, Welch +
Holm/BH helpers) is reused where its semantics match this plan; the analysis
scripts must emit their derivation from raw rows (reproducible), and the C8
audit recomputes headline numbers independently.
