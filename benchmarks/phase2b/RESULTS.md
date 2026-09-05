# Phase 2B Interim Results — Learned Gating (§23–29)

Date: 2026-09-03. Protocol: cached dataset per corpus (HNSW runs once, §4);
seeded 70/15/15 splits (§3); grid of {objective × (hidden, lr, batch)} selected
ENTIRELY on validation R@10 of the temperature-calibrated model (§6/§19 — the
temperature is fitted on validation only); the test split was evaluated once
per corpus after selection. No test-set tuning occurred. Dataset hashes and
full run metadata are in each corpus directory's `run_info.txt` / `dataset.json`.

Honesty notes (§43): two harness flaws were found and fixed on the way, and
both failures are part of the record: (1) ground truth expressed in corpus
hint-ids instead of engine numeric ids (IdMapper starts at 1) silently
mis-scored everything — the first "controlled" run was partly an artifact;
(2) the first multiview design gave queries full-fidelity representations in
ALL views, making the best view statistically undetectable (measured corr
−0.34 — the metrics correctly refused to fake progress). Both were fixed by
changing the harness to be well-defined, never by relaxing honesty.

## Central table (§29) — held-out TEST splits, R@10 / NDCG@10

| approach | CONTROLLED | NOISE | MULTIVIEW |
|---|---|---|---|
| uniform multi-head | 0.624 / 0.688 | 0.738 / 0.803 | 0.213 / 0.280 |
| global best single head | 0.536 / 0.566 | 0.838 / 0.885 | 0.343 / 0.352 |
| trained gating (calibrated) | **0.953 / 0.970** | 0.849 / 0.894 | 0.532 / 0.574 |
| RRF k=60 | 0.669 / 0.729 | 0.780 / 0.839 | 0.262 / 0.311 |
| oracle per-query head | 0.953 / 0.970 | 0.840 / 0.886 | 0.993 / 0.996 |

## Verdict per corpus

**CONTROLLED (best head = f(query group); the §8/§9 sanity test):**
Trained gating equals the per-query oracle — R@10 0.9533 vs oracle 0.9533,
NDCG identical at 0.9700. It recovers **100% of the uniform→oracle gap** and
beats RRF by +28.4pp. Head-entropy ≈ 0.003 (confident per-query selection;
per-group averages stay balanced because each query selects ITS group's head —
selection frequency matches the group distribution). This is the definitive
YES to §8: the gating mechanism can learn query→useful-head.

**NOISE ladder (one globally dominant head; §7 collapse investigation):**
The model collapses to head 0 — average weight 0.894, selection frequency
100% — and this is the CORRECT behavior: with no query-conditional structure
the oracle (0.840) is barely above the global best single head (0.838).
Gating (0.849) lands on the right answer, slightly above oracle via residual
hedging on boundary queries, +11.1pp over uniform. Reported as-is: on
globally-dominated corpora, gating degenerates to "pick the best head",
which is exactly what the data supports. No artificial diversity was forced.

**MULTIVIEW (semantic / lexical / metadata views; query arrives in one
modality; §34–37):** Gating wins over every non-oracle baseline — 0.532 vs
uniform 0.213 (+31.9pp), RRF 0.262 (+27.0pp), global best 0.343 (+18.9pp) —
but recovers only **41% of the uniform→oracle gap** (oracle 0.993). Verdict
per §41 vocabulary: **MODEST WIN**. The path here is itself a finding: at 210
training queries gating learned NOTHING (corr −0.34, weights pinned at
uniform); at 840 it learned decisive preferences. The bottleneck is training
sample-efficiency in the 192-dim concatenated representation, not capacity —
a 2× hidden/lr/batch grid moved val R@10 by <2pp while data ×4 moved it +36pp.

## Per-query-type check (§35/§37)

`by_group.csv` per corpus: on CONTROLLED, the trained model's argmax head
matches the query's group head (the oracle-type baseline is recovered by
inference from the representation alone, §36-compliant — the label is never
an input). On MULTIVIEW, selection frequency [0.311, 0.489, 0.200] vs true
1/3–1/3–1/3 — all three views get selected, with a lexical skew, consistent
with the partial (41%) gap recovery.

## Head diversity AND utility (§21/§22)

`diversity.csv` (pairwise Jaccard@10) and `utility.csv` (per-head recall) are
written per corpus. On CONTROLLED, heads are simultaneously diverse (low
cross-head overlap on each query's pool) and conditionally useful; on NOISE,
six heads are high-diversity/low-utility — diversity did NOT save them, and
the trained model correctly discounts them. Diversity ≠ utility, explicitly.

## Overfitting chain (§26)

`overfit_chain.csv`: for every corpus, train ≥ val ≥ test quality in the
expected order; no corpus shows train-up/test-flat divergence. Early stopping
fired on nearly every run (val-loss patience), and the temperature selection
always used validation.

## Answer to the Phase 2B central question (§29)

"Can learned query-dependent weighting recover the gap between fixed
multi-head and oracle head selection?"

- When query-conditional head quality exists and the query representation
  identifies it: **YES — up to 100% (controlled), 41% (multiview)**.
- When it does not exist (noise ladder): gating correctly reduces to the
  global best single head; there is no gap to recover.
- Against RRF at identical candidates: gating wins on all three corpora
  (+28.4 / +6.9 / +27.0pp R@10).

## What remains for Phase 2B completion

Trained QK attention (§11–13) — gated on this gating verdict, which is now
positive; exact-rerank regression investigation (§32) — the cached datasets
already store per-head exact scores for the offline weighting experiments;
multi-seed reproducibility runs; formal in-bench latency rows (§38; the
engine integration test currently bounds gating overhead at <5ms/query,
actual ≈ µs); `docs/phase2b-final-report.md` (§41) and the §42 checklist.
