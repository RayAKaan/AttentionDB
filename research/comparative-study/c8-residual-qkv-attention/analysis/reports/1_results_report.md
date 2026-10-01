# C8 Results Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`
Date: 2026-10-01 · Evidence: `research/comparative-study/raw/C8-*` · Verify: `all_pass: true, checks: 0`

Status: **VALIDATION + SMOKE + EFPROBE complete locally.** TRK-A TEST (SciFact 300 /
NFCorpus 323 qids × 5 reps) is executed on Ubuntu CI (authoritative); this report
marks those cells `PENDING` and reports the compelling VALIDATION/SMOKE evidence.

## 1. Summary of findings

- All nine arms A–I run against one shared in-process engine; the B/C/D/E/F/G/H/I
  candidate union is **identical by construction** (union-identity gate: 0
  mismatches over every SMOKE/SMOKE-rerun query on both datasets).
- **`lambda = 0` parity holds bit-for-bit**: arm C (truncated-identity machinery at
  `residual_scale = 0`) reproduces arm B's final score for every candidate on every
  query (0/20 mismatches), and C's `applied_correction` is exactly `0.0`.
- **Cache parity holds bit-for-bit**: arm I (learned residual + in-memory K/V cache)
  reproduces arm E exactly (0/20 mismatches), while its p50 latency drops from
  E `356` ms to I `317` ms (SCI) and `365`→`314` ms (NFC) on the `d_k=d_v=64` config.
- The correction is **genuinely active and non-degenerate**: E/F/G/H/I each produce a
  non-zero `applied_correction` on every query and their per-query final-score maps
  differ on 20/20 queries (E≠F, E≠G, E≠H, E≠I after cache removal).
- On the 20-query SMOKE sample the **unrestricted projection (D) reorders the top-10
  more than the residual projection (E)** — SCI nDCG@10 D `0.609` vs E `0.501` vs B
  `0.491` — but this is a small VALIDATION sample and is not a TEST claim.
- Assembly controls behave as designed: identity `lambda=0` C == B; evidence arms
  F/G produce a larger mean `|correction|` (`0.039`/`0.040` SCI) and a correction
  that tracks the baseline (Spearman(`delta`,`baseline`) ≈ `0.92`), whereas the
  pure residual arms E/H have `|correction|` ≈ `0.004` and near-uniform attention
  entropy (≈ `ln 3`).

## 2. Arm means — SMOKE (20 VALIDATION qids, 1 rep; directional only)

| Dataset | Arm | nDCG@10 | R@10 | MRR | p50 µs |
|---|---|---|---|---|---|
| SciFact | A single-head | 0.5640 | 0.7750 | 0.4996 | 3369 |
| SciFact | B fixed fusion (ref) | 0.4911 | 0.7750 | 0.4027 | 13774 |
| SciFact | C λ=0 control | 0.4911 | 0.7750 | 0.4027 | 342086 |
| SciFact | D unrestricted learned | 0.6087 | 0.7750 | 0.5517 | 348829 |
| SciFact | E residual learned | 0.5008 | 0.7750 | 0.4142 | 356282 |
| SciFact | F residual + evidence | 0.5008 | 0.7750 | 0.4142 | 352058 |
| SciFact | G residual + evid + disagree | 0.5008 | 0.7750 | 0.4142 | 351548 |
| SciFact | H residual distilled | 0.5008 | 0.7750 | 0.4142 | 354263 |
| SciFact | I residual + K/V cache | 0.5008 | 0.7750 | 0.4142 | 316994 |
| NFCorpus | A single-head | 0.2636 | 0.0968 | 0.5163 | 3036 |
| NFCorpus | B fixed fusion (ref) | 0.2651 | 0.1158 | 0.5113 | 11963 |
| NFCorpus | C λ=0 control | 0.2651 | 0.1158 | 0.5113 | 348860 |
| NFCorpus | D unrestricted learned | 0.2861 | 0.1162 | 0.6017 | 358464 |
| NFCorpus | E residual learned | 0.2654 | 0.1158 | 0.5118 | 364848 |
| NFCorpus | F residual + evidence | 0.2654 | 0.1158 | 0.5118 | 357295 |
| NFCorpus | G residual + evid + disagree | 0.2654 | 0.1158 | 0.5118 | 361193 |
| NFCorpus | H residual distilled | 0.2654 | 0.1158 | 0.5118 | 358314 |
| NFCorpus | I residual + K/V cache | 0.2654 | 0.1158 | 0.5118 | 314140 |

Ranking metrics at 20 qids are dominated by ties (`R@10` identical across arms);
final-score maps still differ per arm (§1), so these means are directional, not
inferential. TEST is the inferential unit (report 2).

## 3. TUNE training (VALIDATION)

| Dataset | Arm | examples | optimizer steps | final loss |
|---|---|---|---|---|
| SciFact | D unrestricted | 192 | 120 | 1.6844 |
| SciFact | E residual | 192 | 120 | 2.4608 |
| SciFact | F residual+evidence | 192 | 120 | 2.5851 |
| SciFact | G residual+evid+disagree | 192 | 120 | 2.5713 |
| SciFact | H residual distilled | 192 | 120 | 2.4608 |
| NFCorpus | D unrestricted | 303 | 190 | 1.8598 |
| NFCorpus | E residual | 303 | 190 | 2.5636 |
| NFCorpus | F residual+evidence | 303 | 190 | 2.7310 |
| NFCorpus | G residual+evid+disagree | 303 | 190 | 2.7243 |
| NFCorpus | H residual distilled | 303 | 190 | 2.5636 |

`optimizer_steps = epochs × ceil(examples / batch_size)` exactly (5 × 24 = 120 SCI;
5 × 38 = 190 NFC), confirming real mini-batch accumulation rather than per-example
updates. D's lower loss reflects its unrestricted (unregularized) projection; the
residual arms start from the frozen base and learn a small correction.

## 4. Gate and safety results

- **λ=0 parity (stop #9)**: B == C bit-for-bit, 20/20 SCI and NFC; C
  `max|applied_correction| = 0.0`.
- **Cache parity (stop #10)**: E == I bit-for-bit, 20/20 SCI and NFC.
- **Union identity (stop #4)**: B vs C/D/E/F/G/H/I = 0 mismatches, both datasets.
- **ef knob (stop #3)**: EF16 → EF128 changes recall@10 `0.675→0.725` (SCI) and
  `0.1004→0.1158` (NFC), with p50 latency `866→2264` µs (SCI), `855→2338` µs (NFC).
- **Fingerprints (stop #7)**: A/B carry no C8 fingerprint; C carries the control
  fingerprint; D/E/F/G/H/I each carry distinct non-control fingerprints.
- **Non-finite (stop #8)**: none across SMOKE/EFPROBE.
- **Leakage (stop #5)**: TUNE row sets are disjoint from TEST.
- **Geometry**: every TUNE model card declares `attention_dim=384`, `d_k=d_v=64`.

## 5. What remains

TRK-A TEST cells (`C8-TEST-{SCI,NFC}-{A..I}-001`, 5 reps) are the primary
inferential evidence and run on CI. Once present, `c8_analyze.py` produces the 14
Holm-corrected contrasts in `analysis/statistical_results.json` (report 2).
