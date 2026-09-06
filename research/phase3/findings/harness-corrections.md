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

## HC-P3-4: argpartition breaks boundary ties arbitrarily (dataset builder)
- **Behavior**: the AG News GT builder used argpartition for top-k; with
  1e-4-rounded cosines, boundary ties are common in hashed text space and
  argpartition selects among tied docs arbitrarily — the independent
  full-argsort validation failed on 47/270 sampled rows.
- **Correction**: chunked STABLE argsort on rounded scores (deterministic
  score-desc, idx-asc in every path); rust exact arm uses matching
  round-ties-even at 1e-4 (numpy np.rint parity). Exact-reference anchor
  = 1.0000 on every recorded run.
- **Invalidated**: none (caught pre-registration).

## HC-P3-5: ALL-row aggregation averaged non-test queries
- **Behavior**: the textquality harness computed ALL-subset metrics for
  per-query arms by averaging over ALL queries with zero placeholders for
  non-test rows — deflating those arms by (n_test/n_total) ≈ 0.152.
- **Detection**: cross-check between results.csv ALL rows and the
  by-type rows (which were correct) during the pre-recording smoke run.
- **Correction**: ALL rows aggregate test queries only (test_avg).
- **Invalidated**: none — caught in /tmp smoke runs before any recording.

## HC-P3-6: bm25_top10.csv diagnostic id-space mismatch
- **Behavior**: the dump labeled rows with the test-row position while the
  query text was selected by the global per-type index; the independent
  verifier initially read the qid as a per-type index — producing
  meaningless 0.0 verification numbers (rare-token hit 0.0, overlap 0.0).
- **Detection**: the PH3B-BM25-001 independent verification itself; a toy
  6-document engine test proved the BM25 channel correct, isolating the
  bug to the diagnostic mapping chain.
- **Correction**: verifier reconstructs the global per-type index from
  ds.json (split + query_group) + dataset meta.json block offsets; the
  rust dump now emits the global per-type index for future runs.
- **Invalidated**: none — BM25 ARM metrics never used the diagnostic CSV;
  post-fix verification: containment 0.9902, rare-token hit 0.7556,
  overlap 0.5852 (bm25_verify.json).
- **Operational addendum**: SIGKILLed runs leak tmpfs engine dirs
  (632 MB found); engine dirs default to disk (/var/tmp) now; the
  tmpfs-era failed run is preserved and labeled
  (PH3B-COMP-002-D256-FAILED-TMPFS).
