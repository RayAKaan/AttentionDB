# C2 Toolchain Report — tool versions pinned to run evidence

Study `comparative-study-001`, protocol v1.0.0 (commit `7788067`).
Every tool version below is taken from the run's own `environment.yaml` /
`metrics.json` in `../raw/<run_id>/`. Nothing here re-measures or re-judges the
sandbox-originated runs; it records what each run declared.

## Languages & runtimes

| Tool | Version | Where recorded |
|---|---|---|
| Python (sandbox runs) | 3.13.14 | every sandbox `environment.yaml` |
| Python (fresh re-runs) | 3.14.4 | C2-MODES-TEST-001 / C2-ORACLE-AGREE-001 / C2-INTEGRITY-001 `environment.yaml` |
| Rust (stable) | 1.96.0 | fresh probe rebuild (`rustc --version` during build) |
| Cargo | 1.96.0 | same |
| NumPy | used by oracle | C2-ORACLE-TESTS-002 metrics (independent cross-check) |

`rust-toolchain.toml` pins `stable` for the `probe/` crate. The probe binary
rebuilt on 2026-09-24 is `research/comparative-study/c2/probe/target/release/c2probe.exe`
(git `7788067`).

## Embedding pipeline (deterministic, pinned)

Recorded in C2-EMBED-SCIFACT-003 and C2-EMBED-NFCORPUS-003 metrics (the manifests):

- `sentence-transformers` 6.1.0, `torch` 2.14.0+cpu, `transformers` 5.17.0
- Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (files sha256 recorded, incl.
  `model.safetensors`)
- `embedding_dimension` = 384; dtype float32; metric cosine (applied downstream)
- Inference: `encode` batch=32, `sort_by_length` off, no normalization
- Determinism: `determinism_full_view_reencode_byte_equal` true for the main
  pipeline; a single-text (batch-1) re-encode bitwise-differs from batch-32
  export only via padding effects — exports pin batch=32 + order, and
  regeneration uses the same script.

## External systems (smoke runs)

| System | Version | Run |
|---|---|---|
| Qdrant | qdrant 1.12.4 (official binary) | C2-SMOKE-QDRANT-006 |
| PostgreSQL + pgvector | postgresql-17-pgvector 0.8.0-1 (Debian distro repo) | C2-SMOKE-PGVECTOR-005 |
| Elasticsearch | 8.15.2 (default JVM heap) | C2-SMOKE-ES-003 |
| milvus-lite | milvus-lite 3.2.1 via `pymilvus[milvus-lite]` | C2-SMOKE-MILVUSLITE-002 |
| Weaviate | 1.39.6 (binary) | C2-SMOKE-WEAVIATE-004 |

No PGDG third-party repo was needed (distro repo carried pgvector). MILVUS-lite
3.x ships no top-level `milvus` module; the documented call path is
`pymilvus[milvus-lite]` → `MilvusClient`.

## Build tooling used for the fresh probe rebuild (2026-09-24)

- `protoc` at `C:\Users\Rayyan Khan\.cargo\protoc\bin\protoc.exe` (exposed via the
  `PROTOC` env var during `cargo build`)
- crates.io reachable; 653 crates cached in the local cargo registry; release
  build completed in ~4m10s with 6 pre-existing warnings (unused `mut`/vars).

## Reproducibility invariants

- Every run's `config.yaml` pins `git_commit = 7788067…`.
- B3 training configs additionally pin `config_yaml_sha256` and `dataset_sha256`;
  retrains recorded `reproducible_bitwise: true`.
- Dataset downloads were sha256-verified at materialization time (file sha256 in
  DS manifests) — the environment-reset re-procurement procedure is covered in
  `environment-reset-2026-09-24.md` (re-fetch from recorded sources; verify
  sha256 before use; record in `dataset-manifests/`).