# C7 Dataset Report

Datasets are the frozen C6/C7 views (3-head decomposition of the canonical 384-d embeddings: HEAD-TITLE / HEAD-BODY / HEAD-CITE + canonical QUERIES).

## 1. Materialized views
- Vectors: per-dataset `.f32` files in `raw/C7-SHARED-{SCI,NFC}/` — `CANONICAL.f32`, `HEAD-TITLE.f32`, `HEAD-BODY.f32`, `HEAD-CITE.f32`, `QUERIES.f32` (each 5,580,288 / 4,972,032 bytes SciFact; equal sizes NFCorpus).
- Hashes of these inputs are frozen into each run's `environment.yaml` (`input_hashes`).

## 2. Splits and sizes

| Dataset | Validation | TEST (executed) | Test-time qids |
|---|---|---|---|
| SciFact | 368 qids | 300 | 300 |
| NFCorpus | 328 qids | 323 | 323 |

Split provenance is unchanged from C4/C5/C6 (no TEST queries touched during TUNE — gate #5 verified by harness; qrels/query vectors for VALIDATION test-rows are excluded from training artifacts by construction).

## 3. Data cleanliness checks
- exact-oracle parity (gate #2): arm A (CANONICAL, k=10) vs full-collection exact top-K computed from raw embeddings in `c7_verify.py` `exact_heights`; **no discrepancies** (all_pass).
- EffectiveRecall for multi-head arms measured against qrels; n_rel = 0 queries excluded from recall denominators.

## 4. Observed difficulty
- SciFact: fixed-fusion B recall@10 ≈ 0.757, nDCG@10 ≈ 0.610 — retrievable dataset (union 500/500 saturated).
- NFCorpus: recall@10 ≈ 0.157, nDCG@10 ≈ 0.320 — hard, sparse-query retrieval; candidate budget 500 is 100% saturated; relevant docs frequently sit at/near the candidate boundary (drives EF128 > EF16 recall gains, and larger attention effects).