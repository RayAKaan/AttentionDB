# PH3-REFERENCE-001 — Frozen reference implementation (Phase 3 spec §2)

**Identity:** the Phase 2 final architecture, frozen. Code: commit
`86de099` lineage (worktree `benchmarks/phase3/`, `learned/`, `core/`,
`hnsw/`); Rust 1.98.0; build profile `release` (opt-level 3, codegen 16
units default); CARGO_INCREMENTAL=0.

## Configuration of record

| Item | Value |
|---|---|
| Retrieval path | per-head ANN → candidate union → trained gating → weighted fusion → top-K |
| Collections | `bench` (Phase 3 harness), 5 heads: `full,q0,q1,q2,q3` |
| Embedding dimension | 196 per head (quadrant heads zero-padded; see datasets.md) |
| HNSW parameters | max_nb_connection M=16, ef_construction=400, ef_search=64, store_vectors=true (engine defaults) |
| Candidate budget | POOL=100 per head (union ≤ 500) |
| top-K | 10 (GT_K) |
| Normalization | per-head minmax over pool scores (pipeline stage 3, unchanged from Phase 2B) |
| Fusion | learned-weight sum over per-head normalized scores |
| Gating model | GatingMlp input 980 (=5×196) → hidden → 5 logits; protocol = Phase 2B shipped grid (3 objectives × lr{0.01,0.003} × hidden{32,64}), validation-only selection + validation temperature ∈ {0.25,0.5,1,2} |
| Gating checkpoint | `raw/runs/PH3-QUAL-FM-S/models/gating_s{42,7,1}.json` (ModelCard schema) |
| Durability (quality runs) | `Durability::Async` on a session tempdir (in-session engine; persistence/restart behavior is measured by dedicated PH3 persistence runs, not this one) |
| Filters | none (filtered retrieval is a separate §12 experiment family) |
| Hardware of record | sandbox VM: 2 CPUs, 1984 MB RAM, 20 GB disk |

## Reproduction

```
python3 research/phase3/datasets/build_fashion.py S      # or T30/M (regenerable; sha256 in meta.json)
cargo build --release -p phase3-bench
./target/release/phase3-bench quality --tier S --data /tmp/phase3/fashion-S \
  --out research/phase3/raw/runs/PH3-QUAL-FM-S --seeds 42,7,1
```

Any change to HNSW parameters, pool size, head set, gating protocol, or GT
definition creates a NEW reference config and a NEW experiment id — never a
silent edit of this one.
