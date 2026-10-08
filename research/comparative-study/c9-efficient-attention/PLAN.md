# C9 — Efficient Attention Serving and Projection Scaling

Status: implementation and protocol landed on `research/c9-efficient-attention`; authoritative benchmark evidence is pending CI execution.

## Purpose

C8 established residual Q/K/V attention and an in-memory document-side K/V cache. C9 investigates serving cost without changing retrieval semantics, and also follows the C9 pre-registration from the C8 follow-up: evaluate unanchored/wider projections rather than treating C8's residual parameterization as the only architecture.

## C9 subphases

| Subphase | Deliverable | Acceptance gate |
|---|---|---|
| C9.1 | Separate scalar reference and batch-serving paths; establish deterministic microbenchmark | Same input/model and candidate order |
| C9.2 | Reuse query projection across a batch | Exact scalar/batch output parity |
| C9.3 | Preserve independent candidate softmax and score semantics | Weights, logits, output, entropy match bit-for-bit |
| C9.4 | Exercise empty batches and malformed candidates | Defined empty result; invalid candidate returns error |
| C9.5 | Sweep candidate depth: 32, 64, 128, 256, 500, 1000, 2000 | Report p50/p95, throughput, parity |
| C9.6 | Sweep attention/key/value dimensions: d_a 64/128/256/384/512; d_k 32/64/128; d_v 32/64/128 | Record geometry and memory estimates; no unsupported combinations |
| C9.7 | Cache warm/cold and concurrency profiles | Compare cache-on/off rankings and lifecycle behavior |
| C9.8 | Reproducible report and CI gate | Correctness gates pass before interpreting timing |

## Serving paths

- **Scalar reference:** one `attend` call per candidate. This is the semantic oracle.
- **C9 batch path:** project query once, then project candidate K/V and perform the same per-candidate attention operation in original order. Candidate softmaxes remain independent; no candidate is dropped, merged, or reordered.
- **C8 K/V cache:** existing cache path remains the document-side projection optimization. This change does not bypass cache fingerprints or claim durable caching.
- **Wider/unanchored projection study:** registered as a separate experimental arm. It must be compared against the C8 residual arm on the same query/candidate union, with identical splits/seeds and exact candidate-membership checks.

## Benchmark matrix (authoritative runs)

- Arms: baseline retrieval, C8 scalar cache-off, C8 scalar cache-on, C9 batch cache-off, C9 batch cache-on, and the pre-registered unanchored/wider projection arm.
- Candidate depths: 32, 64, 128, 256, 500, 1000, 2000 (where dataset/index size supports the depth).
- Cache states: cold, warm, mixed; report cache hits/misses and resident entries.
- Concurrency: 1, 2, 4, 8, 16, 32, 64 workers where the runner can sustain them.
- Datasets: preserve the C8 datasets and disjoint validation/test split. No test-driven tuning.
- Repetitions: at least 5 for timing summaries; report p50/p95 and throughput. Timing claims require a named runner and commit SHA.

## Mandatory correctness gates

1. Batch and scalar outputs are bit-identical for weights, logits, output, and entropy.
2. Candidate order and count are preserved; empty input yields empty output.
3. Invalid candidate input fails explicitly.
4. Cache-on/off final scores and ranking are equivalent for the same model and candidate union.
5. Fingerprints invalidate cache after model, projection, dimension, or ordered-head changes.
6. The HNSW candidate generator is unchanged by C9 serving code.
7. Benchmark records include configuration, seed, commit, runner, warm-up, repetitions, p50/p95, throughput, and correctness status.
8. A faster path is not promoted based on a noisy single run; performance regressions are reported rather than hidden.

## Interpretation boundary

The code change removes repeated query projection from batch inference, an exact and bounded optimization. It does **not** by itself establish a speedup magnitude, implement SIMD/GPU kernels, or complete the wider-projection scientific comparison. Those require measurements from the authoritative CI/benchmark runner. A PR passing unit/CI checks is implementation validation, not a scientific performance result.
