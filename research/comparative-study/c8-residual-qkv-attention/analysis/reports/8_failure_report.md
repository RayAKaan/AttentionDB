# C8 Failure / Limitations Report

Phase: **C8 — Residual candidate-level Q/K/V attention** · Branch: `comparative-study/c8-residual-qkv-attention`

## 1. Open items

1. **TRK-A TEST not run locally.** The 18 primary TEST cells are executed on Ubuntu
   CI; local analysis is intentionally gated so no statistics are computed over
   VALIDATION data. Consequence: no inferential claim is made in this report set yet.
2. **Pre-existing clippy blocker.** Workspace `--all-targets` clippy fails at
   `hnsw/benches/recall_bench.rs:146` (removed lint in clippy 1.96). Not caused by
   and not fixed in C8; C8 crates are individually clean.
3. **Windows parity gap.** `phase3-bench` uses Unix-only APIs, so the local
   full-workspace gate cannot run on Windows; CI is authoritative.

## 2. Honest design costs

- **`λ=0` is not performance-free.** Arm C still computes the projection and per-
  candidate trace, then multiplies by zero (`~342–349 ms` p50 vs B `~12–14 ms`). The
  bit-exact `λ=0` parity guarantee was chosen deliberately over a fast-path that
  would special-case `λ=0`; the cost is documented, not hidden.
- **Near-uniform learned attention.** Mean attention entropy for E/F/G/H/I is
  ≈ `1.099` (`ln 3`), i.e. learned weights are essentially uniform; the correction
  is driven by the value/output projection and (for F/G) the evidence term rather
  than by sharp per-head reweighting. This limits the mechanistic interpretation and
  is reported as a limitation.
- **H ≈ E.** Distilled arm H's loss (`2.46080`) and observables are extremely close to
  E (`2.46078`): the distillation term does not move the model much at
  `distillation_temperature = 0.5` and the frozen contrastive weight. H is retained as
  a distinct arm (20/20 score maps differ) but its practical effect is small at this
  scale.

## 3. VALIDATION noise caveat

The SMOKE sample is 20 queries / 1 rep with heavy metric ties; its arm ordering
(e.g. D > E on SciFact) is **not** evidence and may invert on TEST. It is included
only as a pipeline sanity signal.

## 4. Not-yet-run cells

SUPPORT (16 cells, dimension/depth sweep artifact generation) remains optional
pending; if skipped locally it must be produced on CI to satisfy the SUPPORT
deliverable.
