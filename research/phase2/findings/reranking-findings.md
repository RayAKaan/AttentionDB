# Reranking Findings (§32, F9)

Phase 2 measured mode E (pipeline exact rerank, equal heads) at 0.558
R@10 vs mode D 0.672. Question: is exact similarity harmful, or is the
fusion weighting harmful?

## Offline decomposition (PH2C-RERANK-001/002/003, cached test candidates)

| method | controlled | noise | multiview |
|---|---|---|---|
| norm fusion, uniform | 0.6244 | 0.7378 | 0.2128 |
| exact fusion, uniform | 0.6511 | 0.7133 | 0.2406 |
| exact fusion, best head | 0.5356 | 0.8378 | 0.3428 |
| exact fusion, oracle head | 0.9533 | 0.8400 | 0.9933 |

## Reading (INTERPRETATION unless noted)

1. Exact scores are NOT intrinsically harmful: on controlled, uniform
   exact fusion beats uniform norm fusion (0.6511 > 0.6244); on multiview
   same direction (0.2406 > 0.2128).
2. The Phase 2 regression reproduces only on the globally-dominated
   corpus: there, uniform exact fusion (0.7133) < uniform norm fusion
   (0.7378) — noisy heads' champions carry high exact cosine against the
   query and outvote the clean head. This matches the Phase 2 mode E
   mechanism (MEASURED direction; exact pipeline arithmetic differs —
   mode E mixes exact scores with attention+mhs fusion weights, so this
   is an attribution, not an exact replication).
3. Head weighting dominates the score source: oracle-weighted exact
   fusion equals the oracle on every corpus (0.9533/0.8400/0.9933).
   The lever is WHICH heads, not WHICH scores.
4. Caveat (ledger N3): the pipeline-level fix (learned weights inside
   mode E) is untested; do not claim the regression is fixed.

Caption draft: "Table. Exact-score fusion under uniform, best-head and
oracle head weighting on held-out test candidates. Equal weighting of
unequal heads explains the Phase 2 exact-rerank regression; oracle
weighting of exact scores attains the oracle bound."
