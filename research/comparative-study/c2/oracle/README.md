# B0 — Harness-side Exact Oracle (C2)

C1 defines B0 as the harness-side exact oracle: for each query, exact
similarity against every eligible vector of a view, deterministic top-k
(score DESC, id ASC — the engine's documented tie order), written to
immutable artifacts. C0 found no valid exact endpoint inside any candidate
system; the oracle is therefore implemented OUTSIDE all systems, in the
harness (numpy), and cross-checked against independent implementations.

## Implementations used

1. `harness/oracle.py` — numpy `exact_topk` (cosine/dot/euclidean).
2. `pure_python_reference` — same math, no vectorization (independent).
3. `probe` (Rust) brute-force — third implementation for engine-side checks.

## Two ground-truth layers (never mixed)

- **NN-GT**: exact nearest neighbors of a vector view (candidate quality,
  ANN correctness, efficiency track). Generated here; GloVe ships official
  NN-GT which C2 re-verifies by full recompute.
- **Human qrels**: semantic relevance (BEIR graded labels) — the ONLY basis
  for quality metrics (nDCG@10 etc.). NN-GT is never presented as relevance.

## Validation results

See `validation-report.md` (battery: hand-check, independent cross-check,
determinism, ties, boundaries, empty input, monotonicity), the engine-side
`C2-ORACLE-AGREE-001` run, and the full NN-GT recomputes recorded inside
`C2-DATA-GLOVE25-003` / `C2-DATA-GLOVE50-003` (set equality 1.0).
