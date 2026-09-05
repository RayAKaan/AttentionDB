# Phase 3 harness corrections

## HC-P3-1: exact-reference arm used raw-dot ranking instead of cosine
- **Behavior**: the exact/brute-force reference ranked the corpus by raw
  f32 dot products with unnormalized query vectors, while the ground truth
  is unit-normalized cosine — the arm read R@10 ≈ 0.001.
- **Detection**: pre-registered harness anchor "exact_reference must be
  1.0" (it defines the GT). Fired immediately on the first Tier-S run.
- **Correction**: normalize corpus + query full-view vectors (cosine).
- **Invalidated**: none — caught before any recording.

## HC-P3-2: exact-reference arm compared corpus indices against engine-id GT
- **Behavior**: after fixing HC-P3-1, the arm still read 0.0013: rankings
  were keyed by corpus doc index while `q.ground_truth` holds ENGINE ids.
- **Detection**: same anchor.
- **Correction**: the exact arm evaluates against the doc-index GT
  (`loaded.gt[qid]`); document the two id spaces in the harness.
- **Invalidated**: none — caught before recording.

## HC-P3-3: pool-reachability over-assertion
- **Behavior**: the pool-generation step hard-asserted every GT doc appears
  in ≥1 head pool of every query. On real data HNSW (ef_search=64,
  pool=100) legitimately misses true neighbors — candidate recall is a
  measured quantity (spec §8/§11), not an invariant.
- **Correction**: record pool coverage as metrics; keep hard assertions
  only for structural breakage (empty pools, id linkage).
- **Invalidated**: none (assertion fired pre-training; converted to a
  measurement, then re-run).
