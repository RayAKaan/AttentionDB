# C7 Environment Report

Host: `Originator` (win32) · 15.93 GB physical RAM.
Evidence dirs record per-run `environment.yaml` (host, OS, python, c7pilot_sha256, input hashes, guardrail snapshot).

## 1. Runtime stack
- Rust: `c7pilot` binary (Cargo workspace `probe/`, release profile `lto=thin, codegen-units=1, opt-level=3, panic=abort`).
- Pilot SHA-256 (all C7 runs, deterministic build): `c14f9418…` (mirrored in every `artifacts/environment.yaml`).
- Python: 3.14 harness (`harness/c7_test_run.py` orchestrator, `c7_verify.py` gates, `c7_analyze.py` stats).
- Dependency note: `hnsw_rs` is **vendored** at `AttentionDB/vendor/hnsw_rs` with a fixed layer-RNG seed and patched into the probe via `[patch.crates-io]` in `probe/Cargo.toml` (see 9_reproducibility_report).

## 2. Guardrail / memory policy
- Policy: sampler (500 ms) aborts the child probe when its process RSS ≥ 85% of MemAvailable measured at rep preflight; per-probe timeout 7200 s (train 2400 s).
- Preflight snapshots at TEST: SciFact MemAvailable ≈ 1.76 GB; NFCorpus ≈ 4.40 GB. Child peak RSS during TEST ≈ 1.2–1.4 GB.
- This policy was the *cause* of the only executed-run interruption (NFC TEST rep5 aborted twice when the harness parent retained ~2 GB of accumulated 390 MB multi-JSONs); fixed by per-rep streaming free (see 8_failure_report).

## 3. Build reproducibility
- Probe build used an out-of-tree CARGO_TARGET_DIR (temp) with `--offline`; the resulting binary is pinned by `c7pilot_sha256` in every cell's `environment.yaml`.
- Input hashes recorded per run (`input_hashes` in `environment.yaml`): shared materialized `.f32` vectors (CANONICAL/HEAD-*/QUERIES) per dataset, written once into `C7-SHARED-{SCI,NFC}`.

## 4. Config surface (frozen)
Per `c7-protocol.md` §3–§4: candidate budget 500, per-head min 20 / max 300, ef_search 64, k=10, WARMUP 20, dim 384, 3 heads, fusion fixed 0.3/0.5/0.2 (attention/multi-head-similarity/bm25); gate g tuned on VALIDATION only; QKV training 5 epochs, lr 1e-2, τ 0.07, l2 1e-4, batch 8, 8 negatives/query, seed 20260925.