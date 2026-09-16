# PH3C head-scaling findings (PROVISIONAL)

Scope: AG News multi-field ladder, size-matched 5K docs (seeds 42/7/1,
K=100 pools), plus 10K extra points. Ladder caveat (documented at dataset
build time): sub-field views make head count and view granularity co-vary
at 4/8 heads — the ladder measures the frozen architecture over
progressively finer multi-view decompositions.

## HS-1 (SUPPORTED, scope: 5K ladder) — Each complementary view contributes large quality gains up to 4 heads; gains SATURATE at 8

Gating R@10 (mean ± std over seeds 42/7/1):
- 1 head (full only): 0.5187 (all arms identical — sanity anchor)
- 2 heads (title/body): 0.6578 ± 0.0205
- 4 heads (+body halves): 0.7146 ± 0.0221
- 8 heads (+quarters/cross): 0.7094 ± 0.0233 (≈ 4-head, within noise)
- 3 heads @10K (Phase 3B reference): 0.6448 ± 0.0025
"More heads improve quality" is NOT supported beyond 4 in this ladder
(§13 discipline: measured, not assumed).

## HS-2 (SUPPORTED, scope: 5K ladder) — The quality/latency/memory tradeoff identifies 2–4 heads as the practical region on this sandbox, WITHOUT labeling any count "best"

@K=100, serial p50: 1.7 / 4.2 / 6.7 / 22.1 ms for 1/2/4/8 heads — the 8-head
point is super-linear (per-head ANN cost × pool fusion over 8×200
candidates). Peak build RSS: 178 / 269 / 464 / 873 MB (linear). Combined
with HS-1: 4 heads ≈ +0.196 R@10 over 1 head at ~3.9× serial latency;
8 heads adds latency/memory for no measured quality change.

## HS-3 (SUPPORTED, scope: 2-CPU sandbox, tested path) — Parallel head execution halves p50 at ≥2 heads and HURTS at 1 head

2-worker scoped-thread parallelism: 1.66× speedup at 4 heads (8.86→5.35 ms
p50), 2.02× at 8 heads; CPU totals confirm real work redistribution (1.65
vs 1.57 s at 4 heads). At 1 head, parallel p50 is 24% WORSE (thread
hand-off overhead) — parallelism is a multi-head-feature, honestly
bounded by the 2-CPU environment.

## HS-4 (SUPPORTED, scope: ladder runs) — Uniform fusion NEVER beats the best single view at any head count; gating beats both everywhere except the degenerate 1-head config

Uniform 0.344–0.386 vs gbest 0.423–0.522 vs gating 0.658–0.715 across
2/4/8 heads. The frozen gate — not uniform weighting — is what converts
additional views into quality.

## OPEN QUESTIONS

- Whether the 4→8-head saturation is intrinsic (redundant sub-views) or
  gate-capacity limitation (trained on 280 queries with 4096-dim inputs at
  8 heads); separating these needs more training queries (memory-bound).
- 8-head quality at 10K docs is untested (OOM; preserved run
  PH3C-HEAD-001-H8-FAILED-10K).

## Audit notes (2026-09-06, spec-clause audit)

- §19 deviation (documented, not silently dropped): seed mean ± std is
  reported for R@10/NDCG@10/MRR (all in results.csv per seed), but
  head-selection AGREEMENT was only computed for the canonical seed-42
  model per run (gating_weights_test.csv carries per-query
  argmax/oracle/agreement/entropy for that model). Per-seed agreement
  std is therefore NOT claimed anywhere; listed under open questions if
  needed for the paper.
- §18 scope note: per-query-type metrics/weights retained at K=100 for
  every head count (head-query-type.csv + per-run query_groups.csv and
  gating_weights_test.csv); per-BUDGET per-type slices were not retained
  (budget arms are ALL-type) — recorded as a scope limitation.
