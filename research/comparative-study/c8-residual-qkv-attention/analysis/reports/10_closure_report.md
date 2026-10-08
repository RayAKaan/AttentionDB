# C8 Closure Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`
Date: 2026-10-01 · HEAD: `4ca4f05ce4a7d4c684287cc1bf436554acfd4c16`

## 1. What is complete

- **Implementation** (attention + core + probe): C8 residual candidate-level QKV
  attention, applied post-fusion (stage 8b), `S_final = S_base + λ·ΔS_attention`.
  `attentiondb-attention` 90 tests, `attentiondb-core` 76 tests, both crates
  clippy-clean; `cargo fmt --all` clean.
- **Protocol & plan**: `c8-protocol.md` v1.0.0 (9 arms, 13 stop conditions,
  14 primary contrasts); `c8-run-plan.csv` (66 cells × 22 cols).
- **Harness**: `c8_test_run.py`, `c8_verify.py`, `c8_analyze.py`.
- **Evidence produced locally**: TUNE (10 cells), SMOKE (18 cells) ×2 runs,
  EFPROBE (4 cells) for both datasets.
- **Verification**: `c8_verify.py` → `all_pass: true, checks: 0`.

## 2. Key verified claims

| Invariant | Result |
|---|---|
| Union identity B vs C..I | 0 mismatches (SCI, NFC) |
| λ=0 parity C == B bit-exact | 20/20 queries, both datasets |
| Cache parity I == E bit-exact | 20/20 queries, both datasets |
| C `applied_correction` | exactly 0.0 |
| E/F/G/H/I correction active | non-zero, distinct per query |
| ef knob real (stop #3) | recall 0.675→0.725 (SCI), 0.1004→0.1158 (NFC) |
| fingerprints (stop #7) | A/B none, C control, learned ≠ C |
| leakage (stop #5) | TUNE ∩ TEST = ∅ |

## 3. What remains (boundary of this closure)

1. **TRK-A TEST** (18 cells × 5 reps) on Ubuntu CI via
   `.github/workflows/c8.yml` → then `c8_analyze.py` →
   `analysis/statistical_results.json` with the 14 Holm-corrected contrasts.
   The CI driver `harness/c8_ci.py` runs PROBE → TUNE → SMOKE → SUPPORT → TEST
   in dependency order, is idempotent (skips complete cells), and invokes the
   gates + analysis; the resulting bundle is uploaded as a workflow artifact.
2. **SUPPORT** (16 cells) dimension/depth sweep.
3. Final study-level REPORT.md once TEST statistics exist.
4. The repository now includes `harness/c8_finalize.py`, a strict closure gate that refuses to declare C8 complete until all 18 TEST cells, all 16 SUPPORT cells, five TEST repetitions per dataset, a passing verification report, and statistical results are present.

## 4. Stop-condition status

Stop conditions #1–#13 are satisfied on available evidence except those that
inherently require TEST (#1 primary-result, #2 effect-size direction, #11/#12
report-level). No condition has **failed**; TEST is pending, not negative.

## 5. Recommendation

Run the authoritative Ubuntu C8 workflow. It executes PROBE → TUNE → SMOKE → SUPPORT → TEST, then verification and statistical analysis, and finally the strict `c8_finalize.py` closure gate. C8 is not scientifically closed until that gate reports `READY`.

The C8 implementation itself is already integrated into `main`; this closure work is deliberately limited to research reproducibility, validation gating, and documentation. No validation result is promoted to a paper claim before TEST-scale inference is available.
