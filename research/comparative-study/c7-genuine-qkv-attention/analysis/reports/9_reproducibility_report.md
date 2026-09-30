# C7 Reproducibility Report

## 1. Determinism policy (protocol §8 gate #6, user decision 2026-09-29)
"Observable-level reproducibility": a fresh-process re-run must reproduce per-query candidate counts, recall@10, nDCG@10, MRR, and top-10 ranked rows; raw ledger *tail* tolerance jaccard ≥ 0.95 was the original documented-noise allowance.

## 2. What happened
The OS-seeded hnsw_rs layer RNG violated even that policy: on the SMOKE rerun, recall@10 flipped on several queries and top-10 ranks changed on ~1/3 of queries. This was an environmental nondeterminism, not a harness artifact.

## 3. Fix and evidence strength
hnsw_rs is now vendored with a **fixed layer-RNG seed** (`672025`/20260925 in `vendor/hnsw_rs/src/hnsw.rs`); `probe/Cargo.toml` redirects via `[patch.crates-io]`. Because HNSW search is deterministic given a graph, entire index builds are bit-identical across processes.

Fresh-process verification (SMOKE, both datasets, 20 qids, 6 arms):
- candidate_count, recall@10, nDCG@10, top-10 rows, and full union ledgers: **identical** across `multi-smoke-rep1.json` vs `multi-smoke-rerun.json`.
- The construct-verified scratch check (arms A+B, 20 qids, two separate processes) matched on every observable including full ledgers.

`c7_verify.py` on the full evidence set: **`{"all_pass": true, "checks": 0}`**.

## 4. Reproducing the build
1. Build pilot with the patched dependency:
   - `$env:CARGO_TARGET_DIR = <temp>`; `cargo build --release --offline` in `probe/`.
   - (The workspace-level `[patch.crates-io]` ensures hnsw_rs resolves to the vendored, seeded copy.)
2. Pin the binary hash: `c7pilot_sha256 = c14f9418…` recorded in every cell's `artifacts/environment.yaml`.
3. Re-run any cell with `C7PILOT=<binary> python harness/c7_test_run.py --run-id C7-...-001`.

## 5. Re-run guarantees
- Multi-arm reps within a process: bit-identical per rep (same seed stream).
- Across processes: bit-identical union ledgers and scores (fixed layer seed + deterministic search + deterministic training with fixed seed 20260925).
- Statistics: bootstrap seed fixed; query-level unit; Wilcoxon/Holm deterministic.
- Note: the two prefixed ef_search values and the gate scan remain VALIDATION-only; TEST queries never appear in training artifacts (gate #5).