# C7 Implementation Report

## 1. Pilot (`probe/`)
`c7pilot` (bin crate, own workspace `[workspace]`, deps on attentiondb-core / attentiondb-storage / attentiondb-attention). CLI: `c7pilot --config <cfg.json> [--out <path>]`; subcommands `run` (multi-arm `Replicate`) and `train` (contrastive QKV). Config JSON mirrors the C6 pilot contract with C7 extensions per arm:

- `mode` A/B/C/D/E/F; `attend_heads`, `fusion` (weights + optional learned gate g), `attention` block (`enabled`, `arch: identity-qkv | learned-qkv`, `qkv_model`, `scorer(w_attn, w_evidence, bias)`, `use_evidence`), `ef_search`, `search_k` (new; caps per-head search k so the ef knob is real), `configuration_id`, `qkv_model` path.
- Emits per-query: `union_ledger` (row, head_sims, mhs, attention, final, rank), `candidate_count`, recall/ndcg vs qrels and exact, `latency_us` (measured from a per-arm timer — fixed, see failures report), `attention_fingerprint`, `c7_trace`, plus `c7_aggregate` (mean entropy over queries, mean Spearman attention↔final, per-head means).
- Multi-arm output streams arm-by-arm when `--out` is given (keeps peak RSS flat on large TEST JSONs).

## 2. Core engine changes (`core/`)
- `RetrievalConfig.search_k: Option<usize>` — when set, caps `top_k`/`per_head_k` in the MODE-A/B/C/D/E/F candidate fetch so `ef_search` exercises the real hnsw_rs beam (hnsw_rs clamps beam to max(ef, k)).

## 3. Vendored dependency (determinism fix)
`hnsw_rs` is vendored at `AttentionDB/vendor/hnsw_rs` and redirected via `[patch.crates-io] hnsw_rs = { path = "../../../../vendor/hnsw_rs" }` in `probe/Cargo.toml`. The only change: `LayerGenerator` seeds its RNG with the fixed constant `20260925` instead of OS entropy (`StdRng::seed_from_u64(HNSW_DETERMINISTIC_SEED)` in `src/hnsw.rs`). Search is deterministic given a graph, so full index builds become bit-identical across processes.

## 4. Harness (`harness/c7_test_run.py`)
- Orchestrates the 34-cell plan; probes `raw/<run_id>` immutability; materializes shared vectors; runs multi-arm probes per cell group; splits per-arm artifacts; writes `metrics.json`, `environment.yaml`, registers `RUN-INDEX.yaml`.
- Memory-safe rep loop: each rep's union check + per-arm split happens before the next rep launches, and the `multi` object is freed (avoids parent-RSS creep tripping the child guardrail).
- Determinism rerun: SMOKE emits `multi-smoke-rerun.json` from a **second fresh process** for gate #6.

## 5. Gate/analysis code
- `c7_verify.py`: gates 1–12 (anchor hashes, exact-oracle parity, ef knob, union identity, TEST-leak, determinism on rerun, fingerprint collision, finite checks, RUN-INDEX integrity; run-index now accepts per-track artifacts `metrics.json` vs `gate_choice.json`/`model.json` for TUNE).
- `c7_analyze.py`: paired bootstrap 10k (seed 20260925), Wilcoxon, Holm–Bonferroni (8 contrasts), Cohen's dz, per-arm means + observables → `analysis/statistical_results.json`.