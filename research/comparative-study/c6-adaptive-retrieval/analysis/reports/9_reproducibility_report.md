# C6 Reproducibility Report

**Goal**: Exact re-execution of all C6 TEST cells with identical results.

---

## Frozen Inputs (Immutable)

| Artifact | Location | Hash Method |
|----------|----------|-------------|
| Embedding vectors (SciFact) | `raw/C3-EMBED-SCIFACT-001/artifacts/*.npy` | SHA256 per file |
| Embedding vectors (NFCorpus) | `raw/C2-EMBED-NFCORPUS-003/artifacts/*.npy` | SHA256 per file |
| Qrels (TEST) | `raw/datasets/beir/{scifact,nfcorpus}/qrels/test.tsv` | SHA256 |
| Queries/corpus JSONL | `raw/datasets/beir/{scifact,nfcorpus}/*.jsonl` | SHA256 |
| Protocol | `c6-protocol.md` | SHA256 |
| Run plan | `c6-run-plan.csv` | SHA256 |
| Engine binary | `probe/target/release/c6pilot.exe` | SHA256 in `environment.yaml` |

All input hashes recorded per-run in `environment.yaml`.

---

## Determinism Guarantees

| Source | Seed | Scope |
|--------|------|-------|
| Global protocol | 20260925 | All stochastic choices |
| Query order (per rep) | SEED + rep × 0x9E3779B9 | Shuffle before search |
| VAL split (SciFact) | SEED + 0x5EED | Train-sample shuffle |
| PROBE/SMOKE subsample | SEED | Subsample from VAL |
| Bootstrap | 20260925 | Resampling |
| HNSW build | (frozen in C2/C3) | Index construction |
| Randomized control | 20260925 | RandomizedPolicy seed |

No adaptive/random seeds outside this table.

---

## Binary Reproducibility (MSVC)

- Build: `cargo build --release -p c6pilot` with `/Brepro` (linker) + `CARGO_INCREMENTAL=0` + `RUSTFLAGS="-Ccodegen-units=1"`
- Result: Bit-identical `c6pilot.exe` across builds on same toolchain
- Verification: `sha256sum c6pilot.exe` stable across rebuilds

---

## Re-execution Instructions

```bash
# 1. Checkout branch
git checkout comparative-study/c6-adaptive-retrieval

# 2. Build probe (reproducible)
cd research/comparative-study/c6-adaptive-retrieval/probe
cargo build --release

# 3. Run any TEST cell (example)
cd ../../harness
python c6_test_run.py --run-id C6-TEST-SCI-B-001

# 4. Verify metrics match
# Compare generated metrics.json with archived raw/C6-TEST-SCI-B-001/metrics.json
```

---

## Expected Determinism

- Per-query recall@10 identical across reps (5 fresh processes, same seed → same query order → same HNSW traversal)
- Per-rep mean recall identical across re-runs (same binary, same inputs, same seed)
- `adaptive_changed_queries`, `adaptive_redistributed_queries` identical (deterministic interaction)

Any deviation → environment mismatch (record in failure report).

---

## CI Verification

- Ubuntu 22.04 GitHub Actions: full workspace test suite (61 core tests + integration)
- Windows local: `--exclude phase3-bench` (pre-existing)
- No flaky tests observed in 30 harness runs