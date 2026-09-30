# C7 Results Report

Phase: **C7 — Genuine candidate-level Q/K/V attention** · Branch: `comparative-study/c7-genuine-qkv-attention`
Date: 2026-09-30 · Evidence: `research/comparative-study/raw/C7-*` (RUN-INDEX lines 1626+) · Verify: `all_pass: true, checks: 0`

## 1. Summary of findings

- All six arms (A–F) ran on the frozen TEST splits (SciFact 300 qids, NFCorpus 323 qids) × 5 reps with one shared in-process engine. The B/C/D/E/F candidate union is **identical by construction** (union identity gate: 0 mismatches over every rep × dataset).
- **Fixed-fusion (B) remains the quality reference.** Identity attention (C, D) reproduces B within |Δ|=0.01 nDCG@10 (negligible). **Learned candidate-level QKV (E) significantly degrades ranking quality** on both datasets. Evidence-supervized learning (F) mitigates the degradation on NFCorpus but stays below B; on SciFact it is negligible vs B.
- The attention channel is **genuine**: `attention` is `Some` for every candidate in C/D/E/F, weight vectors vary across candidates/queries, and Spearman(attention, multi_head_similarity) is well below 0.999 (D 0.72/0.56, E 0.13/0.08, F 0.33/0.23 on SCI/NFC).
- The ef knob is now a real lever (search_k cap): EF16 vs EF128 differ in candidate counts (~30 vs ~234) and latency (≈0.9 ms vs ≈2.4 ms).

## 2. Headline numbers (TEST, mean of 5 reps)

| Dataset | Arm | nDCG@10 | R@10 | MRR | p50 latency (µs) |
|---|---|---|---|---|---|
| SciFact | A single-head | 0.6359 | 0.7700 | 0.5967 | 3442 |
| SciFact | B fixed fusion (ref) | 0.6097 | 0.7567 | 0.5678 | 12793 |
| SciFact | C identity + gate g*=1.0 | 0.6168 | 0.7672 | 0.5736 | 829221 |
| SciFact | D identity QKV | 0.6115 | 0.7600 | 0.5693 | 828652 |
| SciFact | E learned QKV | 0.5469 | 0.6948 | 0.5087 | 850612 |
| SciFact | F learned QKV + evidence | 0.6030 | 0.7652 | 0.5599 | 850383 |
| NFCorpus | A single-head | 0.3154 | 0.1540 | 0.5078 | 3029 |
| NFCorpus | B fixed fusion (ref) | 0.3201 | 0.1569 | 0.5185 | 10713 |
| NFCorpus | C identity + gate g*=0.9 | 0.3209 | 0.1559 | 0.5189 | 821749 |
| NFCorpus | D identity QKV | 0.3201 | 0.1567 | 0.5185 | 821158 |
| NFCorpus | E learned QKV | 0.2272 | 0.1118 | 0.4235 | 842992 |
| NFCorpus | F learned QKV + evidence | 0.2709 | 0.1300 | 0.4658 | 842696 |

Legacy A (single canonical head, k=10) top-scoring on SciFact is expected: it uses the full canonical 384-d vector without the budgeted per-head decomposition; it is not a candidate-level attention arm (attention absent, union n/a).

## 3. Primary contrasts (Δ = B − target, higher is worse for E/F)

| Contrast | Δ nDCG@10 | 95% CI | Wilcoxon p | Holm reject | Cohen's dz | |Δ|<0.01 |
|---|---|---|---|---|---|---|---|
| SCI B vs C | −0.00708 | [−0.01089, −0.00335] | 0.00020 | yes | −0.212 | yes |
| SCI B vs D | −0.00177 | [−0.00364, 0.00004] | 0.06500 | no | −0.107 | yes |
| SCI B vs E | +0.06278 | [0.04352, 0.08169] | <1e-5 | yes | 0.371 | no |
| SCI B vs F | +0.00670 | [−0.00918, 0.02213] | 0.19857 | no | 0.048 | yes |
| NFC B vs C | −0.00081 | [−0.00182, 0.00021] | 0.12081 | no | −0.085 | yes |
| NFC B vs D | +0.00002 | [−0.00035, 0.00039] | 0.30278 | no | 0.006 | yes |
| NFC B vs E | +0.09291 | [0.08358, 0.10265] | <1e-5 | yes | 1.065 | no |
| NFC B vs F | +0.04919 | [0.04126, 0.05738] | <1e-5 | yes | 0.670 | no |

Interpretation (no winner labels): E is a statistically significant, non-negligible **degradation** vs B on both datasets (Cohen's dz 0.37 / 1.07). F degrades NFC by +0.049 (significant, dz 0.67) but is within noise on SciFact. C/D are statistically/effectively equivalent to B.

## 4. Gate and TUNE results

- EFPROBE (ef knob real): candidate mean counts EF16→EF128 = 30→231 (SCI), 34→242 (NFC); recall@10 0.675→0.725 (SCI), 0.1004→0.1158 (NFC).
- Gate scan: SciFact best_g = 1.0 (nDCG@10 0.5861 on VALIDATION, monotone in g); NFCorpus best_g = 0.9 (0.3012, peaked at 0.9).
- TUNE training: SciFact E loss 4.7e-4, F 4.5e-4; NFC E 7.4e-3, F 7.8e-3 (5 epochs, 192/303 contrastive examples).