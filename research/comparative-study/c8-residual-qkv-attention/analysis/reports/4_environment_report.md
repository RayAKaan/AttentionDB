# C8 Environment Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`

## 1. Repository state

| Item | Value |
|---|---|
| Branch | `comparative-study/c8-residual-qkv-attention` |
| Base branch | `comparative-study/c7-genuine-qkv-attention` |
| HEAD | `4ca4f05ce4a7d4c684287cc1bf436554acfd4c16` |
| Entry commit (C0) | `fe4f92b2187bf631125e975a5db8a202b5316d00` |
| C7 PR | #7 — **not merged, not modified** |

Untracked vendor sentinels `vendor/hnsw_rs/.cargo-ok` and
`vendor/hnsw_rs/.cargo_vcs_info.json` are left untouched.

## 2. Toolchains

| Tool | Version |
|---|---|
| rustc | 1.96.0 (ac68faa20 2026-05-25) |
| cargo | 1.96.0 |
| clippy | 0.1.96 |
| Python | 3.14.4 (MSC v.1944, 64-bit) |
| numpy / scipy | available (analysis harness) |

## 3. Platform

Local runs are Windows (`win32`, PowerShell 5.1). Consequences:

- `cargo` progress on stderr surfaces as PowerShell `NativeCommandError`; builds
  still succeed (cosmetic only).
- Full-workspace `--all-targets` testing/clippy is limited by Unix-only
  `phase3-bench` APIs; the authoritative full gate is **Ubuntu CI**.
- TRK-A TEST is therefore executed on CI via `.github/workflows/c8.yml`
  (`workflow_dispatch`), which builds `c8pilot` (release), runs
  `harness/c8_ci.py --verify --analyze`, and uploads the evidence bundle. The
  frozen datasets (20 files) and C2/C3 embeddings (19 files) are tracked in git,
  so the runner needs no download step.

## 4. Pre-existing clippy blocker (unrelated to C8)

`cargo clippy --workspace --all-targets -- -D warnings` fails at
`hnsw/benches/recall_bench.rs:146` on the removed lint
`clippy::chunks_exact_to_as_chunks` (removed in clippy 1.96). `hnsw/` is unmodified
by C8; the C8 crates (`attentiondb-attention`, `attentiondb-core`) are clippy-clean
individually (`-D warnings`).
