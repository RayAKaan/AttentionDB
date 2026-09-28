# C6 Statistical Analysis Report

**Method**: Protocol statistics — paired bootstrap (10,000 samples, seed 20260925), Wilcoxon signed-rank, Holm-Bonferroni, Cohen's dz. Unit = query; per-query score = mean of 5 fresh-process reps.

---

## Bootstrap 95% Confidence Intervals (Mean Difference, Arm − B)

| Contrast | Observed Δ | CI Lower | CI Upper | Contains 0? |
|----------|-----------|----------|----------|-------------|
| SCI B→C | −0.0059 | −0.0123 | −0.0003 | **No** |
| SCI B→D | −0.0029 | −0.0081 | +0.0024 | Yes |
| SCI B→E | −0.0045 | −0.0111 | +0.0019 | Yes |
| SCI B→F | −0.0044 | −0.0139 | +0.0049 | Yes |
| SCI B→RE | −0.0044 | −0.0098 | +0.0008 | Yes |
| NFC B→C | +0.0001 | −0.0035 | +0.0031 | Yes |
| NFC B→D | −0.0005 | −0.0032 | +0.0037 | Yes |
| NFC B→E | −0.0007 | −0.0013 | +0.0029 | Yes |
| NFC B→F | +0.0001 | −0.0027 | +0.0030 | Yes |
| NFC B→RE | −0.0012 | −0.0014 | +0.0039 | Yes |

Only SCI B→C has CI excluding zero (but Holm-corrected α=0.005 > p=0.046).

---

## Wilcoxon Signed-Rank Test (Two-Sided)

| Contrast | W-statistic | p-value | p < 0.05? |
|----------|------------|---------|-----------|
| SCI B→C | 40.0 | 0.0460 | Yes |
| SCI B→D | 47.0 | 0.2731 | No |
| SCI B→E | 93.0 | 0.1667 | No |
| SCI B→F | 59.5 | 0.1514 | No |
| SCI B→RE | 30.0 | 0.0468 | Yes |
| NFC B→C | 1080.5 | 0.2578 | No |
| NFC B→D | 887.5 | 0.4094 | No |
| NFC B→E | 964.0 | 0.6113 | No |
| NFC B→F | 1438.0 | 0.7471 | No |
| NFC B→RE | 1154.0 | 0.7491 | No |

---

## Holm-Bonferroni Correction (Family = 10 Primary Contrasts)

Sorted p-values: 0.0460, 0.0468, 0.1514, 0.1667, 0.2578, 0.2731, 0.4094, 0.6113, 0.7471, 0.7491  
Thresholds: 0.05/10=0.005, 0.05/9=0.0056, 0.05/8=0.00625, 0.05/7=0.0071, 0.05/6=0.0083, 0.05/5=0.01, 0.05/4=0.0125, 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05

| Rank | Contrast | p | αᵢ | Reject? |
|------|----------|---|-----|---------|
| 1 | SCI B→C | 0.0460 | 0.0050 | No |
| 2 | SCI B→RE | 0.0468 | 0.0056 | No |
| 3 | SCI B→F | 0.1514 | 0.0063 | No |
| 4 | SCI B→E | 0.1667 | 0.0071 | No |
| 5 | NFC B→C | 0.2578 | 0.0083 | No |
| 6 | SCI B→D | 0.2731 | 0.0100 | No |
| 7 | NFC B→D | 0.4094 | 0.0125 | No |
| 8 | NFC B→E | 0.6113 | 0.0167 | No |
| 9 | NFC B→F | 0.7471 | 0.0250 | No |
| 10 | NFC B→RE | 0.7491 | 0.0500 | No |

**No contrasts survive Holm correction.**

---

## Effect Sizes (Cohen's dz)

| Contrast | dz | Interpretation |
|----------|-----|----------------|
| SCI B→C | −0.113 | Negligible |
| SCI B→D | −0.061 | Negligible |
| SCI B→E | −0.078 | Negligible |
| SCI B→F | −0.052 | Negligible |
| SCI B→RE | −0.095 | Negligible |
| NFC B→C | +0.002 | Negligible |
| NFC B→D | −0.015 | Negligible |
| NFC B→E | −0.035 | Negligible |
| NFC B→F | +0.003 | Negligible |
| NFC B→RE | −0.048 | Negligible |

All |dz| < 0.2 (small threshold).

---

## Negligible-Effect Threshold (Protocol: |Δ| < 0.01)

| Contrast | |Δ| | Negligible? |
|----------|-----|-------------|
| SCI B→C | 0.0059 | **Yes** |
| SCI B→D | 0.0029 | **Yes** |
| SCI B→E | 0.0045 | **Yes** |
| SCI B→F | 0.0044 | **Yes** |
| SCI B→RE | 0.0044 | **Yes** |
| NFC B→C | 0.0001 | **Yes** |
| NFC B→D | 0.0005 | **Yes** |
| NFC B→E | 0.0007 | **Yes** |
| NFC B→F | 0.0001 | **Yes** |
| NFC B→RE | 0.0012 | **Yes** |

All contrasts are negligible per protocol threshold.

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

Statistical evidence does **not** support a recall@10 benefit from any adaptive allocation arm over the independent multi-head union (B). All effect sizes are negligible (|Δ| < 0.01), no contrast survives Holm correction, and Cohen's dz < 0.12 for all contrasts. The interaction-guided arms (E, F) redistribute budget on 100% of queries but at 8× latency cost with zero recall benefit.