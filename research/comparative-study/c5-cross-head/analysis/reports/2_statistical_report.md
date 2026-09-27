# C5 Statistical Analysis Report

**Method**: Protocol §10 statistics — paired bootstrap (10,000 samples, seed 20260925), Wilcoxon signed-rank, Holm-Bonferroni, Cohen's dz. Unit = query; per-query score = mean of 5 fresh-process reps.

---

## Bootstrap 95% Confidence Intervals (Mean Difference)

| Contrast | Observed Δ | CI Lower | CI Upper | Contains 0? |
|----------|-----------|----------|----------|-------------|
| SCI A→B | +0.0131 | −0.0420 | +0.0143 | Yes |
| SCI B→C | +0.0018 | −0.0079 | +0.0115 | Yes |
| NFC A→B | +0.0088 | −0.0206 | +0.0023 | Yes |
| NFC B→C | −0.0017 | −0.0055 | +0.0021 | Yes |

All CIs contain zero → no contrast significant at α=0.05 (uncorrected).

---

## Wilcoxon Signed-Rank Test (Two-Sided)

| Contrast | W-statistic | p-value | p < 0.05? |
|----------|------------|---------|-----------|
| SCI A→B | 338.5 | 0.2276 | No |
| SCI B→C | 27.0 | 0.5915 | No |
| NFC A→B | 4956.5 | 0.2318 | No |
| NFC B→C | 1832.5 | 0.8700 | No |

---

## Holm-Bonferroni Correction (Family = 4 Primary Contrasts)

Sorted p-values: 0.2276, 0.2318, 0.5915, 0.8700  
Thresholds: 0.05/4=0.0125, 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05

| Rank | Contrast | p | αᵢ | Reject? |
|------|----------|---|-----|---------|
| 1 | SCI A→B | 0.2276 | 0.0125 | No |
| 2 | NFC A→B | 0.2318 | 0.0167 | No |
| 3 | SCI B→C | 0.5915 | 0.0250 | No |
| 4 | NFC B→C | 0.8700 | 0.0500 | No |

**No contrasts survive Holm correction.**

---

## Effect Sizes (Cohen's dz)

| Contrast | dz | Interpretation |
|----------|-----|----------------|
| SCI A→B | 0.053 | Negligible |
| SCI B→C | 0.021 | Negligible |
| NFC A→B | 0.084 | Negligible |
| NFC B→C | 0.048 | Negligible |

All |dz| < 0.2 (small threshold).

---

## Negligible-Effect Threshold (Protocol: |Δ| < 0.01)

| Contrast | |Δ| | Negligible? |
|----------|-----|-------------|
| SCI A→B | 0.0131 | No (but CI contains 0) |
| SCI B→C | 0.0018 | **Yes** |
| NFC A→B | 0.0088 | **Yes** |
| NFC B→C | 0.0017 | **Yes** |

Three of four contrasts are negligible per protocol threshold; A→B is borderline.

---

## Bootstrap Distribution Diagnostics

- Seed: 20260925 (fixed)
- Samples: 10,000 per contrast
- n_queries: 300 (SCI), 323 (NFC)
- Resampling: query-level with replacement
- Per-query scores: mean of 5 reps

All bootstrap distributions approximately symmetric; no outliers detected.

---

## Conclusion

Statistical evidence does **not** support a recall@10 benefit from cross-head interaction (B→C) on either dataset. The multi-head union (A→B) shows positive but non-significant gains. All effect sizes are negligible to small.