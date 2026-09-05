# PH2C-QK-002 — QK vs gating: main-corpus analysis (PROVISIONAL → F11–F15 assigned)

## Verdict (Rule Zero): NO SIGNIFICANT WIN — trained candidate-level QK LOSES to trained gating on the only corpus where a valid paired comparison is possible.

## Observations [PH2C-QK-002-MULTIVIEW; test split; seeds 42/7/1]

| arm | R@10 (agg) | NDCG@10 | MRR |
|---|---|---|---|
| uniform | 0.2128 | 0.2803 | 0.6435 |
| global_best | 0.3428 | 0.3521 | 0.4706 |
| RRF k=60 | 0.2622 | 0.3114 | 0.6805 |
| trained gating | **0.4983** (per-seed .5050/.4939/.4961) | 0.5289 | 0.7009 |
| trained QK | 0.1113 (per-seed .1100/.1078/.1156) | 0.1151 | 0.2944 |
| gating+QK (RRF-60) | 0.2769 | 0.3187 | 0.6153 |
| oracle | 0.9933 | 0.9957 | 1.0000 |

- Reference arms reproduce the registered Phase 2B values EXACTLY
  (uniform/gbest/RRF/oracle) — pipeline cross-validation.
- **Candidate recall is NOT limiting**: GT-in-union fraction 0.9975, 100% of
  queries have ≥1 relevant in the pool. The gap is pure candidate ORDERING.
- **QK trained, then lost**: ‖ΔW‖=106.2 (gradients flow), train loss
  decreases (trainlog_qk.csv), score std mean 8.26 min 2.76 (non-degenerate),
  untrained baseline 0.0339 → trained 0.1113 (it learned *something*, but
  far less useful than per-head cosine fusion), τ(QK,uniform)=0.459 — QK
  reorders substantially and WRONGLY.
- **Latency**: QK p50 310.5 µs vs gating 13.0 µs (24×) — the loss is not
  even cheap. gating+QK (0.2769) is worse than gating alone: blending in the
  QK ranking damages a good ranking.
- **Scope of valid comparison**: multiview only. controlled/noise QK arms
  are NOT_EXECUTABLE on the frozen caches (HC-6: content linkage lost,
  refused pre-training per §4). Their reference arms are the canonical 2B
  results. On controlled, gating already equals oracle (0.9533) — QK had
  ~no headroom there; on noise, gating collapses to the only reliable head —
  also ~no headroom. Multiview was the corpus with real headroom (gating
  recovers only ~41% of the oracle gap) — and QK still loses.

## Interpretation

The representational capability demonstrated on the sanity set
(PH2C-QK-001: QK can express query–candidate interaction gating cannot)
does NOT correspond to exploitable structure in the real multiview pools:
whatever ordering-relevant signal exists in the candidate content, a linear
bilinear model trained by InfoNCE on 840 queries extracts less of it than
per-head cosine fusion with learned head weights. Consistent with the
corpus design: multiview relevance is defined VIEW-WISE (the query's
modality) — head-level selection captures nearly all of it; per-candidate
cross-view interaction adds noise, not signal.

## Findings assignment (F11–F15; numbered only now that the main comparison is validated)

- **F11** [PH2C-QK-002-MULTIVIEW]: Trained candidate-level QK does NOT
  improve over trained gating on multiview (0.1113 vs 0.4983 R@10) — it
  loses to uniform fusion; candidate recall is not limiting. NOT SUPPORTED:
  "candidate-level QK improves retrieval" (multiview; controlled/noise
  untestable on frozen caches, HC-6).
- **F12** [PH2C-QK-002-MULTIVIEW]: QK produces large but harmful ranking
  changes (τ=0.459 vs uniform; R@10 below untrained-fusion baselines).
- **F13** [PH2C-QK-002-MULTIVIEW]: No evidence that diversity or any query
  group benefits: QK loses in EVERY query group (by_query_type.csv).
- **F14** [PH2C-QK-002-MULTIVIEW + PH2C-RERANK-001..003]: exact rerank
  quality tracks the WEIGHTING, not the scores: gating-weighted exact
  fusion is the best exact variant; QK-weighted exact does not rescue it.
- **F15** [PH2C-QK-002-MULTIVIEW]: quality/latency frontier: QK is
  dominated (worse quality AND 24× latency); gating+QK is dominated by
  gating.

## Unresolved / out of scope

- controlled/noise QK on REGENERATED corpora (new pools; would be a new
  experiment family, not the frozen-pool paired comparison). Not run:
  Rule Zero outcome is already determined by the valid multiview result
  plus the structural headroom argument above.
- Deeper QK architectures: FORBIDDEN by §29 STOP condition.

## Figures

Figure slots for the QK main comparison are PENDING (no figure generated in
this pass; all numbers are tabulated above from raw CSVs — no figure was
fabricated to fill the slot).
