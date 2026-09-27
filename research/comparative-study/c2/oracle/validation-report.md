# B0 Exact Oracle — Validation Report (C2)

Study `comparative-study-001`, protocol v1.0.0 (commit `7788067`).
Interfaces documented in `README.md`; this report records the validation
evidence. Three independent implementations: `harness/oracle.py` (numpy),
`pure_python_reference` (unvectorized), and the Rust `probe` brute-force
(engine-side checks).

## Battery — [C2-ORACLE-TESTS-002 PASS]

| # | Test | Status | Evidence |
|---|---|---|---|
| 1 | hand-check tiny | PASS | expected [0,1,3], got [0,1,3]; scores [1.0, 0.993884, 0.707107] |
| 2 | independent cross-check numpy vs pure-python | PASS | 9 queries |
| 3 | deterministic repeat | PASS | — |
| 4 | tie order id-ascending | PASS | got [0,1,2,3,4,5] |
| 5 | top-k boundary (0, n, k>n) | PASS | — |
| 6 | empty corpus input | PASS | — |
| 7 | scores monotonically non-increasing | PASS | n=5000 |

`n_pass = 7 / n_total = 7`.

## Full NN-GT recompute vs shipped GloVe neighbors (set equality)

- GloVe-25 [C2-DATA-GLOVE25-003]: 10,000 queries, full exact recompute,
  top100_set_mismatches 0, exact_boundary_tie_swaps 2, match_rate 1.0.
- GloVe-50 [C2-DATA-GLOVE50-003]: 10,000 queries, full exact recompute,
  top100_set_mismatches 0, exact_boundary_tie_swaps 1, match_rate 1.0.

Method: B0 full exact recompute vs shipped neighbors with set equality per
query (float association may reorder only exact ties — recorded, not masked).

## Engine-side oracle agreement — [C2-ORACLE-AGREE-001 PASS]

Fresh run (2026-09-24, current `c2probe` source, git `7788067`; first
registration of this run_id): engine mode A score/top-k vs Rust brute force
over 20 in-process queries.

- returned_scores_consistent_with_stored_vectors: true (the correctness assertion)
- set_equality_rate 1.0, order_equality_rate 1.0
- note: "engine mode A is approximate; agreement rate is recorded as
  diagnostics, score consistency is the correctness assertion"

## Two ground-truth layers (never mixed)

- NN-GT (candidate quality / ANN correctness / efficiency): generated here
  (GloVe official NN-GT re-verified).
- Human qrels (semantic relevance): the ONLY basis for quality metrics
  (nDCG@10 etc.). NN-GT is never presented as relevance.

## Verdict

B0 oracle validated: 7/7 battery PASS; NN-GT set-equality 1.0 on both ANN
datasets; engine agreement consistent on the real engine. C2-ORACLE-TESTS-001
(INVALID-STARTUP, superseded) is preserved in `../raw/` and the RUN-INDEX.