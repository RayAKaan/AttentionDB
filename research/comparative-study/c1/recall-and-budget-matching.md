# C1 — Recall-Matched & Budget-Matched Comparison Protocols

Study `comparative-study-001`, protocol v1.0.0.

## A. Quality-matched comparison (Rule 5)

- **Matching metric:** Recall@10 against B0 exact ground truth on the
  dataset's NN-GT view (efficiency datasets) OR qrels-based proxy
  (Relevance-Recall@10 = fraction of graded-relevant (≥1) judged docs
  retrieved) for semantic datasets — the matching metric is per-dataset
  fixed in the C2 run configs before any outcome is seen.
- **Matching set:** VALIDATION queries only (ann-benchmarks: a fixed 2,000-
  query subsample of the official test split preregistered as
  "validation-for-tuning" with the REMAINING queries as test; semantic
  datasets: training/transfer corpora validation queries — official test
  queries are never used for tuning).
- **Tolerance:** matched configs must fall within ±0.01 Recall@10 of the
  target on the matching set. If ≥2 configs are inside tolerance, the
  FASTEST p50 is selected (selection recorded).
- **Sweep discipline:** per-system preregistered grids (recorded verbatim in
  C2 configs): AttentionDB {ef_search × candidate_multiplier ×
  min_candidates_per_head}; Qdrant {ef (hnsw) or exact/rescore};
  pgvector {hnsw.ef_search, lists/probes for ivfflat or hnsw.m/ef};
  Elasticsearch {num_candidates}; others analogous. Same grid-size budget
  per system (D5).
- **Reporting:** frozen matched configs are then run on TEST queries; tables
  report test latency (p50/p95/p99) at matched quality + the matching-set
  evidence.
- **Unreachable target:** if a system cannot reach the target within its
  grid/host limits → report the full quality–latency curve, mark
  TARGET-UNREACHABLE (with the best achieved recall), never interpolate a
  fake match.
- **Pareto-incomparable configs:** reported as incomparable with both
  points visible; no dominance claim without the curve.

## B. Budget-matched comparison

Preregistered budget axes (fixed BEFORE outcomes; each comparison uses ONE
axis as the constraint):

| Budget axis | Levels | Applies to |
|---|---|---|
| BUDGET-CAND | candidate pool {50, 100, 200, 500} (AttentionDB candidate_budget / per-head k×H equivalents per system-documented semantics) | Track A quality at fixed budget |
| BUDGET-EF | ef_search {16, 32, 64, 128} (AttentionDB) ↔ each system's documented closest knob (Qdrant ef, pgvector hnsw.ef_search, ES num_candidates) — recorded as ANALOGOUS-KNOB, never universal equivalence | recall-latency curves |
| BUDGET-MEM | RSS cap {512 MiB, 1.0 GiB, 1.6 GiB} enforced by sampler-abort (guardrails) | Track B practical sizing |
| BUDGET-TIME | per-query deadline {1 ms, 10 ms, 100 ms} (AttentionDB deadline API; external: client-side timeout) | tail-behavior comparison |

## Failure/limit handling

- Systems that OOM/timeout at a level: recorded as OBSERVED-LIMIT at that
  level with logs preserved; the level is NOT silently skipped for that
  system in the summary table (absence must be explained).
- Exact-match impossibility (e.g., a system exposes no ef-like knob):
  documented knob-mapping table + full-curve reporting; no invented
  equivalence (charter §9).
- All sweeps run on the SAME machine, same dataset hash, same query order
  policy (statistical plan); reruns take new run IDs.
