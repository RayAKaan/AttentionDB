# C6 Environment Report

**Host**: Windows 11 (PowerShell 5.1)  
**CPU**: AMD Ryzen 9 7950X (16C/32T)  
**RAM**: 64 GiB DDR5-5600  
**OS**: Windows 11 Pro 23H2  
**Rust**: 1.82.0 (MSVC toolchain)  
**Python**: 3.12.4  

---

## Software Versions

| Component | Version | Notes |
|-----------|---------|-------|
| attentiondb-core | 0.1.0 (local) | Engine with C6 adaptive |
| attentiondb-storage | 0.1.0 (local) | HNSW index |
| c6pilot (probe) | 0.1.0 | Built with `/Brepro` (reproducible) |
| Cargo | 1.82.0 | Lockfile pinned |
| NumPy | 1.26.4 | Vector loading |
| SciPy | 1.13.0 | Statistics (wilcoxon, bootstrap) |

---

## Build Configuration

- **Profile**: `release` (full optimization, LTO=thin)
- **Reproducibility**: `CARGO_INCREMENTAL=0`, `RUSTFLAGS="-Ctarget-cpu=native -Ccodegen-units=1"`, `/Brepro` linker flag
- **Target**: `x86_64-pc-windows-msvc`
- **c6pilot binary hash**: recorded per-run in `environment.yaml`

---

## Memory Guardrails (Per-Run)

- Sampler interval: 500 ms
- Abort threshold: 85% of `MemAvailable` at preflight
- Timeout: 1800 s per rep
- Peak RSS observed: 270–305 MiB (well under 512 MiB budget)

---

## Random Seeds

| Purpose | Seed |
|---------|------|
| Global protocol seed | 20260925 |
| Query order shuffle (per rep) | SEED + rep × 0x9E3779B9 |
| VAL split (SciFact train-sample) | SEED + 0x5EED |
| Bootstrap | 20260925 |
| EFPROBE/SMOKE subsample | 20260925 |
| Randomized control (RAND) | 20260925 |

All seeds fixed and recorded; no adaptive seed selection.

---

## Storage & Artifacts

- Raw evidence: `raw/C6-*/artifacts/` (immutable, append-only)
- Vector artifacts: `.f32` (little-endian float32, row-major)
- Metrics: `metrics.json` per run
- Per-rep outputs: `RUN-rep{1..5}.json` with per-query ledger
- Run index: `raw/RUN-INDEX.yaml` (append-only, YAML)
- No evidence modified after registration

---

## CI Parity

- Ubuntu 22.04 GitHub Actions covers full workspace test suite
- Windows local checks use `--exclude phase3-bench` (pre-existing Unix-gated bench)
- MSVC `/Brepro` ensures bit-for-bit binary reproducibility