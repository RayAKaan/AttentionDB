# C8 Closure Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`
Date: 2026-10-02
Study code HEAD when the local evidence below was produced: `4ca4f05`
Branch HEAD as of this revision: `893777e` (C8 evidence, plus harness
portability/robustness fixes for the Linux CI runs)

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
| leakage (stop #5) | TUNE ∩ TEST = ∅ (verified explicitly: VAL 200/324 vs TEST 300/323 qrels, disjoint; every TUNE cell used exactly its validation rows) |

## 3. What remains (boundary of this closure)

1. **TRK-A TEST** (18 cells × 5 reps) on Ubuntu CI via
   `.github/workflows/c8.yml` → then `c8_analyze.py` →
   `analysis/statistical_results.json` with the 14 Holm-corrected contrasts.
   The CI driver `harness/c8_ci.py` runs PROBE → TUNE → SMOKE → SUPPORT → TEST
   in dependency order, is idempotent (skips complete cells), and invokes the
   gates + analysis; the resulting bundle is uploaded as a workflow artifact.
   Local execution is not viable for TEST: the probe needs ~4 GB for 300
   queries × 9 arms and this host has ~3.8 GB free, so the harness RAM
   guardrail correctly aborts. Ubuntu CI is the authoritative platform.
2. **SUPPORT** (16 cells) dimension/depth sweep.
3. Final REPORT.md at the study root once TEST statistics exist.

### CI harness defects found and fixed

Both were latent: they only execute on Linux / in the SUPPORT path, so neither
was reachable from the local Windows runs.

- `mem_status()` selected its Win32 `MEMORYSTATUSEX` branch whenever
  `import ctypes` succeeded, but `ctypes.windll` does not exist on Linux, so
  every cell aborted before the probe started (run `36899118011`). Now selected
  on `platform.system()`, with a `psutil`-backed object exposing the same
  attribute names so `env.json` still records availability on Linux.
- The SUPPORT metrics block read `mean_abs_correction` off `c8_aggregate`,
  which only carries entropy/spearman summaries, so every SUPPORT cell died
  with `KeyError` after training and the sweep had already completed (run
  `36900463929`). Now computed from the per-query `c8_trace` candidates, the
  same way `c8_analyze.py` does.

No protocol, configuration, arm, or scoring change was made for either fix.

## 4. Stop-condition status

Stop conditions #1–#13 are satisfied on available evidence except those that
inherently require TEST (#1 primary-result, #2 effect-size direction, #11/#12
report-level). No condition has **failed**; TEST is pending, not negative.

## 5. Recommendation

Proceed with the CI TEST/SUPPORT run, then `c8_analyze.py`, then write the
study-level REPORT.md and classify the outcome.

Repository policy in force for this phase: work lands on
`comparative-study/c8-residual-qkv-attention` as logical commits pushed to
`origin`, with the study PR opened/updated to match. `main` is not modified.
C7's PR #7 is a separate, still-open PR that this phase does not merge — it is
tracked on its own, not as a C8 prerequisite.
