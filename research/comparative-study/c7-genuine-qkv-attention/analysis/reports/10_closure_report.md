# C7 Closure Report

## 1. Status
C7 (genuine candidate-level Q/K/V attention) executed to completion on branch `comparative-study/c7-genuine-qkv-attention`, with **all 12 stop conditions clear**, verification `all_pass: true`, and the 10 reports delivered.

## 2. Stop-condition ledger (protocol §8)
| # | Condition | Status |
|---|---|---|
| 1 | Anchor hash mismatch (B0) | Pass |
| 2 | Exact-oracle discrepancy | Pass (0) |
| 3 | ef-sensitivity probe (real knob) | Pass (EF16 vs EF128 differ, search_k cap) |
| 4 | B/C/D/E/F candidate-set mismatch | Pass (0 mismatches, all reps × datasets) |
| 5 | TEST leakage during TUNE | Pass (VALIDATION-only) |
| 6 | Non-determinism on re-run | Pass (bit-identical fresh-process reruns after fixed layer seed; original violation documented in 8/9) |
| 7 | Attention fingerprint collision | Pass (distinct fingerprints) |
| 8 | NaN/Inf/non-finite | Pass |
| 9 | Raw evidence overwritten | Pass (immutable dirs enforced; superseded evidence quarantined, never overwritten) |
| 10 | 3 consecutive harness crashes | Pass |
| 11 | Irreproducible statistics | Pass (bootstrap CIs non-empty) |
| 12 | RUN-INDEX consistency | Pass |

## 3. Cells executed
All 34 plan cells registered `PASS` in `raw/RUN-INDEX.yaml`: EFPROBE 4, TUNE 6, SMOKE 12, TEST 12.

## 4. Scientific outcome (no winner labels)
- Fixed-fusion baseline (B) is the quality reference: identity attention (C, D) matches it within |Δ|<0.01; **learned candidate-level QKV (E) significantly degrades ranking quality** on both datasets (SCI Δ=+0.063, NFC Δ=+0.093 nDCG@10, Holm-significant); evidence-supervized F degrades NFC only (Δ=+0.049) and is negligible on SciFact.
- The attention mechanism is genuine (present for every candidate; varies; Spearman vs mhs 0.08–0.72, nowhere near 0.999).
- Under the frozen budgets/hyperparameters, 3-head per-candidate attention adds ~0.83 s/query-arm (≈ 100× latency) without improving ranking; the studied route does not pay for itself in this configuration.

## 5. Deliverables (protocol §10)
1. `c7-protocol.md` ✓ (frozen)
2. `c7-run-plan.csv` ✓ (34 cells)
3. `probe/` c7pilot (reproducible build; patched hnsw_rs vendored at `AttentionDB/vendor/hnsw_rs`) ✓
4. `harness/c7_test_run.py` ✓
5. `harness/c7_analyze.py` ✓
6. `harness/c7_verify.py` ✓
7. 10 reports in `analysis/reports/` (this is report 10) ✓
8. Raw evidence in `research/comparative-study/raw/C7-*` immutably registered ✓

## 6. Open work
- Commit the C7 phase (evidence + harness/analysis changes + vendored dependency) and optionally open a PR — **not done**, awaiting explicit instruction.
- The `.C7-OSSEED-EVIDENCE/` quarantine (superseded pre-fix runs) can be removed once the deterministic evidence is committed; retained for auditability meanwhile.