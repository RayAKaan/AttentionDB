# C8 Statistical Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`
Analysis: `harness/c8_analyze.py` · Output: `analysis/statistical_results.json`

## 1. Design (frozen in c8-protocol.md §9)

- Unit of analysis: **query**. Per-query metric = mean over 5 TEST reps.
- Primary metric: **nDCG@10**; secondary: recall@10, MRR.
- Primary contrasts: **B vs {C, D, E, F, G, H, I} × {SciFact, NFCorpus} = 14**.
- Paired bootstrap, **10,000** resamples, seed **20260925**, percentile 95% CI.
- Wilcoxon signed-rank (two-sided) on per-query means; **Holm–Bonferroni** across the
  14 primary nDCG@10 contrasts.
- Effect size: Cohen's dz on the paired per-query difference.
- Negligibility threshold: **|Δ nDCG@10| < 0.01** ⇒ `negligible = true` regardless of p.

## 2. Current status

TRK-A TEST cells (`C8-TEST-{SCI,NFC}-{A..I}-001`) are **not present locally**; they
are produced on Ubuntu CI. `c8_analyze.py` therefore reports the 18 missing cells
and exits without writing `statistical_results.json` (by design, to avoid emitting
statistics over VALIDATION data). Once CI artifacts land, the script emits:

- `primary_contrasts.{DS}_{A}{B}.ndcg10_qrels` = `{mean_diff, ci95, wilcoxon_stat,
  wilcoxon_p, cohens_dz, negligible, holm_alpha, holm_reject}`;
- `means.{DS}_{arm}.{ndcg10_qrels,recall10_qrels,mrr}`;
- `observables.{DS}_{arm}` = mean entropy, mean Spearman(`delta`,`final`), mean
  `|applied_correction|`.

## 3. Directional VALIDATION signal (not inferential)

The SMOKE sample (report 1) suggests three hypotheses for TEST to adjudicate:

1. **λ=0 identity** — C ≡ B (proven bit-exact; TEST contrast must be flat, Δ=0).
2. **Residual helps or is neutral vs B** — E/F/G/H/I sit slightly above B on the
   SciFact SMOKE sample (`0.5008` vs `0.4911`) and at parity on NFCorpus
   (`0.2654` vs `0.2651`); both within the noise of 20 qids.
3. **Unrestricted D may overfit** — D's SMOKE nDCG is the highest (`0.609`/`0.286`)
   but its correction anti-correlates with the baseline (Spearman(`delta`,`baseline`)
   `-0.127`/`-0.077`), the signature of a projection that is not aligned with
   retrieval utility. TEST will show whether this generalizes or is VALIDATION noise.

No winner is claimed pending TEST.
