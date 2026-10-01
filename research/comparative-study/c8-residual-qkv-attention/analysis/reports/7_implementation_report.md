# C8 Implementation Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`

## 1. Attention crate — `attentiondb-attention`

New C8 modules: `config.rs`, `cache.rs`, `distillation.rs`, `negative_mining.rs`,
`projection.rs`, `scorer.rs`, `qkv.rs`, `attention.rs`, `c8_training.rs`, `model.rs`,
`lib.rs`.

- `C8AttentionConfig` owns geometry (`attention_dim`, `key_dim`, `value_dim`,
  `n_heads`), `use_residual`, `residual_scale` (`λ`), evidence/disagreement toggles,
  `distillation_temperature`, and the frozen training hyperparameters. `λ` lives in
  the attention config, **not** the fusion config (protocol Q2).
- `with_residual_scale(λ)` builder added.
- `attention_dim = 0` disables C8 and builds an inert placeholder projection.
- Unit tests (90 passing) include cache retention/purge and the frozen
  truncated-identity projection.

## 2. Core crate — `attentiondb-core`

- `retrieval.rs`: `C8CandidateTrace` (row, baseline, delta, correction, final,
  per-head attention, entropy) and `C8Trace`; `verify_residual`.
- `collection.rs`: `RetrievalConfig.c8_attention`; `Collection.c8_attention` +
  `Collection.c8_kv_cache`; **stage 8b** applies the residual **after**
  `fuse_candidate`; metrics `attentiondb_c8_residual_micros` and
  `attentiondb_c8_kv_projection_micros`; `subsystem_key`; `attend_detailed_c8`
  (with C8) and `attend_detailed_c7` (frozen reference).
- `lib.rs` exports the new handles.
- 76 core unit tests passing; `cargo fmt --all` clean.

Design invariant: stage 8b runs *after* fusion so that at `λ=0` the present-weight
renormalization of fusion is untouched — this is what makes C ≡ B bit-for-bit.

## 3. Probe — `probe/src/main.rs` (`c8pilot`)

- `run` and `train` subcommands; arms A–I; JSON output contract with per-query C8
  observability (entropy, Spearman(delta,final), mean |correction|).
- Streaming multi-arm writer emits one `ARM-<X>-repN.json` per arm per rep with a
  shared candidate union.
- Hard-negative mining + `ResidualQkvTrainer`; `train` emits a **full report** whose
  model card is nested under `model_card`.
- Fixes applied during bring-up:
  1. train path reads `key_dim`/`value_dim` (default `= dim`, asserted in
     `(0, dim]`) and threads them into `unrestricted(...)` /
     `truncated_identity_residual(...)`;
  2. the truncated-identity branch now sets `c.residual_scale = λ`, forcing arm C to
     `λ=0` regardless of external flags.

## 4. Harness

- `harness/c8_test_run.py` (~790 lines): orchestrates EFPROBE / SMOKE / TUNE / SUPPORT
  / TEST; builds C8 config blocks per arm; union-identity check (B..I); RAM guardrail;
  environment + hash capture; RUN-INDEX registration. `run_train` extracts
  `report["model_card"]` to `model.json` and keeps the full report as
  `train_report.json`.
- `harness/c8_verify.py`: invariant gates (union identity, non-finite, λ=0 parity,
  cache parity, correction-active, determinism, ef-knob, fingerprints, no-leakage,
  geometry, RUN-INDEX); writes `analysis/verification_report.json`.
- `harness/c8_analyze.py`: 14 Holm-corrected contrasts (report 2).
