# C8 Reproducibility Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`

## 1. Determinism

- Frozen seed **20260925**; rep mix `0x9E3779B9`; candidate budget **500**;
  `ef_search` **64**; `k=10`; warmup **20**; `dim=384`; fusion **0.3/0.5/0.2**.
- **SMOKE rerun determinism gate** (stop #6): re-running SMOKE reproduces the first
  run bit-for-bit (`c8_verify.py` passes with 0 violations).
- **λ=0 parity**: arm C == arm B bit-for-bit, 20/20 queries, both datasets.
- **Cache parity**: arm I == arm E bit-for-bit, 20/20 queries, both datasets.

## 2. Reproduce locally

```powershell
# from research/comparative-study/c8-residual-qkv-attention
cargo build --release -p c8pilot            # or the probe manifest path
python harness/c8_test_run.py --run-id C8-SMOKE-SCI-A-001     # 9-arm group
python harness/c8_test_run.py --run-id C8-EFPROBE-SCI-001
python harness/c8_test_run.py --run-id C8-TUNE-SCI-E-001      # per TUNE arm
python harness/c8_verify.py                  # invariant gates -> verification_report.json
python harness/c8_analyze.py                 # requires TEST cells (CI)
```

## 3. Frozen artifacts

- Protocol: `c8-protocol.md` v1.0.0.
- Plan: `c8-run-plan.csv` — 66 cells × 22 columns (SAFETY 22 = 4 EFPROBE + 18
  SMOKE; TUNE 10; TRK-A 18; SUPPORT 16).
- Evidence: `research/comparative-study/raw/C8-*`, registered additively in
  `raw/RUN-INDEX.yaml`.
- Verification: `analysis/verification_report.json` (`all_pass: true`).

## 4. Training reproducibility

Example counts and dataset hashes are content-derived and replay exactly from the
seed (report 5). `optimizer_steps` is deterministic (`epochs × ceil(N/batch)`).
TUNE model cards declare `attention_dim=384`, `d_k=d_v=64`.

## 5. Known non-reproducibility sources

Floating-point reduction order across platforms is the only expected source of
last-bit variation; all bit-exact claims above are within a single platform/session
and are gated by exact comparison on the same machine.
