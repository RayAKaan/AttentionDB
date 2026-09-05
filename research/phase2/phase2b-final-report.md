# Phase 2B Final Report — Learned Query-Dependent Head Selection

Status: Phase 2B core complete (gating learned, validated, integrated,
benchmarked). Phase 2C items explicitly pending: trained QK attention,
pipeline-level rerank re-weighting. This report follows the §41 structure;
verdicts use the fixed vocabulary. All numbers trace to
`raw/experiment-index.json`; methodology in `methodology/`.

## 1. Original problem

Phase 2 established that quality-blind fusion is the bottleneck: multi-head
fixed fusion (0.672 R@10) lost to the best single head (0.763), untrained
gating and identity-QK were exact no-ops, and equal-weight exact rerank
regressed (0.558). Phase 2B's question: can the system learn which heads
are useful for a given query, and does that improve retrieval?

## 2. Phase 2 benchmark results

Frozen accepted baseline: `results/ablation.csv`, `tables/table-ablation.md`
(single best 0.763 / fixed 0.672 / untrained gating 0.672 / identity-QK
0.672 / exact rerank 0.558; parallel 1.39–1.72× with identical recall).
Not numerically comparable to 2B (ground-truth correction HC-4) — ordering
evidence only.

## 3. Training methodology

Cached datasets (HNSW runs once per run ID), seeded 70/15/15 splits,
GatingMlp (input → ReLU 32/64 → softmax heads), three objectives
(quality-MSE / soft-target CE τ=0.1 / pairwise margin 0.05), Adam
(lr 0.01/0.003, batch 32/16, L2 1e-4, patience 15), per-head Recall@10
targets, grid selected on validation R@10 of the temperature-calibrated
model; test evaluated once. Full protocol: `methodology/experimental-protocol.md`.

## 4. Training dataset

Per-run `dataset.json`: per query — engine-id ground truth (top-10 by
exact cosine over generator vectors), gating input (query or concatenated
view vectors), split, and per head — top-100 candidates with raw,
per-head-MinMax-normalized, and exact scores plus Recall@10/NDCG@10/MRR.
`methodology/datasets.md` documents all three corpora; hashes in the registry.

## 5. Loss function

All three implemented and compared on validation; no universal winner —
controlled converges under any objective; multiview at 300q favored
pairwise, at 120q soft_target/pairwise tied. Selected objective per run is
recorded in run metadata (§6 of the spec satisfied; selection by validation).

## 6. Model architecture

One-hidden-layer MLP, hidden 32 or 64, stable softmax output; 1188 params
/ 20 288 bytes at the trained controlled configuration. Inference
0.87 µs/query CPU (`PH2B-LATENCY-001`). Deterministic (bit-identical
retraining, unit-tested). Persisted as validating JSON model cards
(format/arch/dims/head-names/training metadata), loaded through a
per-collection registry with atomic activation and an active-file delete
guard; incompatible cards (head count/coverage) are rejected with typed
errors; no model ⇒ deterministic fallback fusion. Hot-swap verified
without restart.

## 7. Validation methodology

Validation-only selection at three levels (objective+hyperparameters,
early stopping, temperature). Test touched once per run. Multi-seed:
controlled 0.9556 ± 0.0000; multiview 0.4365 ± 0.0361
(`tables/table-multiseed.md`). Overfitting chain train ≥ val ≥ test held
everywhere (`results/*-overfit-chain.csv`).

## 8. Gating behavior

Controlled: per-query one-hot post-calibration (entropy 0.003), routing
matches the query group without ever seeing the label
(fig8). Noise: collapse to head 0 (avg weight 0.894) — correct on a
globally-dominated corpus. Multiview: real but imperfect routing
([0.311, 0.489, 0.200]). `findings/gating-findings.md`.

## 9. Head-quality correlation

Pearson r (pooled test examples, weight vs head recall): controlled 0.903,
noise 0.609, multiview 0.269. Calibration was addressed at the
decision level (temperature fitted on validation); probability calibration
was not studied (per §19, ordering is what matters).

## 10. Head diversity

Pairwise Jaccard@10 per corpus (`results/*-head-diversity.csv`) +
utility per head (`results/*-head-utility.csv`). F5 stands: diverse ≠
useful; the trained gate discounts diverse-but-weak heads.

## 11. Exact rerank investigation

`findings/reranking-findings.md` + PH2C-RERANK-001/002/003: exact scores
are not harmful (uniform exact ≥ uniform norm on 2/3 corpora; oracle
exact = oracle bound everywhere); equal head weighting explains the Phase 2
regression. Pipeline-level fix remains untested (ledger N3, Q3).

## 12. RRF comparison

Trained gating beat RRF k=60 on all three corpora (+28.4 / +6.9 / +27.0 pp
R@10 single-run; multiview multi-seed mean still +17.4pp). Limitation: no
k sweep (ledger C3).

## 13. QK attention results

Not yet trained — Phase 2C. Identity-QK = no-op (measured, Phase 2).
`results/qk-attention.csv` and figure 10 are explicit pending markers.

## 14. Real multi-view benchmark

Synthetic 3-view corpus with one-modality queries (§34 allows a labeled
synthetic construction when no public dataset is ingested; no external
dataset was used). `findings/multiview-findings.md`.

## 15. Held-out test results

`tables/table-final-comparison.md` — generated from raw CSVs, verified by
`verify_consistency.py`.

## 16. Latency

Gating 0.87 µs/query (CPU); pipeline context from the frozen Phase 2
corpus (p50 186–1892 µs depending on heads). `tables/table-latency.md`.

## 17. Memory

Model 20 288 bytes serialized / 1188 params. Corpus RSS: 367 MB (Phase 2
8-head 10k-doc run, recorded in run_info).

## 18. Model size

1188 parameters; JSON card 20 KB (uncompressed; fine at this scale).

## 19. Failure cases

`findings/negative-results.md` + `findings/failure-analysis.md` + registry
INVALIDATED entries (GT id shift, tie-lottery GT, all-view queries,
210-query non-learning, seed variance).

## 20. Limitations

`paper/phase2-limitations.md` (synthetic data, scale, seeds, hardware, no
drift/production evaluation).

## Verdicts (§41, fixed vocabulary)

**Does learned query-dependent head selection improve retrieval?**
YES on the evaluated corpora — **CLEAR WIN on controlled** (gating =
oracle, +33pp over uniform, +28pp over RRF), **MODEST WIN on multiview**
(+32pp single-run / +22pp multiseed-mean over uniform, 41%-of-gap
single-run / 28–34% multiseed, beats RRF), and **correct degenerate
behavior** (equal-or-better than every baseline) on the noise ladder.
Scoped to synthetic corpora with known head structure.

**Does candidate-level attention improve retrieval beyond learned
gating?** UNKNOWN — not yet trained (Phase 2C). Prior identity-init
results carry no evidence either way.

**Does the complete pipeline beat strong non-neural baselines?** Against
RRF and best-single-head on identical candidates: yes on all three
evaluated corpora. Against the per-query ORACLE: fully matches on
controlled, matches on noise, not on multiview. Against the frozen Phase 2
"best single head" heuristic on real head-quality structure: that
comparison remains open until gating is evaluated on a corpus where head
quality is neither engineered-grouped nor globally constant
(HYPOTHESIS: real multi-view data).
