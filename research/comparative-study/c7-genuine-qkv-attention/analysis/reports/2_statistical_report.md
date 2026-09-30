# C7 Statistical Report

Method: per protocol §9 — unit = query; per-query metric = mean of 5 reps; paired bootstrap 10,000 (seed 20260925); two-sided Wilcoxon signed-rank; Holm–Bonferroni across the 8 primary contrasts (B vs C/D/E/F × 2 datasets); Cohen's dz; |Δ| < 0.01 → negligible. Primary metric nDCG@10; secondary R@10, MRR. Artifact: `analysis/statistical_results.json`.

## 1. Dataset sizes
- SciFact TEST: 300 query ids; NFCorpus TEST: 323 query ids. Each arm ran 5 reps (reps seeded SEED + (rep−1)·REP_MIX; all deterministic after the hnsw_rs layer-seed fix).

## 2. Primary contrasts — nDCG@10 (Δ = mean of B − mean of target, CEI symmetric 2.5–97.5 bootstrap)

| Key | Δ | CI95 low | CI95 high | Wilcoxon stat | p | Holm α | Reject | dz | Negligible |
|---|---|---|---|---|---|---|---|---|---|
| SCI_BC | −0.00708 | −0.01089 | −0.00335 | – | 0.00020 | 0.01000 | yes | −0.212 | yes |
| SCI_BD | −0.00177 | −0.00364 | 0.00004 | – | 0.06500 | 0.01250 | no | −0.107 | yes |
| SCI_BE | 0.06278 | 0.04352 | 0.08169 | – | 0.00000 | 0.00833 | yes | 0.371 | no |
| SCI_BF | 0.00670 | −0.00918 | 0.02213 | – | 0.19857 | 0.02500 | no | 0.048 | yes |
| NFC_BC | −0.00081 | −0.00182 | 0.00021 | – | 0.12081 | 0.01667 | no | −0.085 | yes |
| NFC_BD | 0.00002 | −0.00035 | 0.00039 | – | 0.30278 | 0.05000 | no | 0.006 | yes |
| NFC_BE | 0.09291 | 0.08358 | 0.10265 | – | 0.00000 | 0.00625 | yes | 1.065 | no |
| NFC_BF | 0.04919 | 0.04126 | 0.05738 | – | 0.00000 | 0.00714 | yes | 0.670 | no |

Notes:
- The three significant degradations (SCI_E, NFC_E, NFC_F) all survive Holm–Bonferroni.
- SCI_BC is "significant" by Wilcoxon (p=0.0002, Δ=−0.007) but the effect is below the protocol's 0.01 negligibility bound; not treated as a material difference.
- C≈D≈B by construction of effect; E≠F (evidence weighting) on NFC only.

## 3. Secondary metrics — mean differences (B − target)

| Key | Δ R@10 | Δ MRR |
|---|---|---|
| SCI_BC | −0.0105 | −0.0058 |
| SCI_BD | −0.0033 | −0.0015 |
| SCI_BE | +0.0619 | +0.0591 |
| SCI_BF | −0.0085 | +0.0079 |
| NFC_BC | +0.0010 | −0.0004 |
| NFC_BD | +0.0002 | 0.0000 |
| NFC_BE | +0.0451 | +0.0950 |
| NFC_BF | +0.0269 | +0.0527 |

Secondary metrics follow the primary pattern (no contradictions).

## 4. Means table (TEST, mean over reps)

| | SciFact | | | NFCorpus | | |
|---|---|---|---|---|---|---|
| Arm | nDCG@10 | R@10 | MRR | nDCG@10 | R@10 | MRR |
| A | 0.6359 | 0.7700 | 0.5967 | 0.3154 | 0.1540 | 0.5078 |
| B | 0.6097 | 0.7567 | 0.5678 | 0.3201 | 0.1569 | 0.5185 |
| C | 0.6168 | 0.7672 | 0.5736 | 0.3209 | 0.1559 | 0.5189 |
| D | 0.6115 | 0.7600 | 0.5693 | 0.3201 | 0.1567 | 0.5185 |
| E | 0.5469 | 0.6948 | 0.5087 | 0.2272 | 0.1118 | 0.4235 |
| F | 0.6030 | 0.7652 | 0.5599 | 0.2709 | 0.1300 | 0.4658 |

## 5. Statistical-soundness notes
- Per-rep per-query values are bit-identical across reps within an arm (deterministic engine), so rep means carry no additional sampling variance beyond the query unit; CIs are query-resampling CIs.
- Bootstrap seed fixed (20260925); no post-hoc exclusions; no winner labels.