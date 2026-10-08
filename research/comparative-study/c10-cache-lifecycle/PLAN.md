# C10 — Cache Lifecycle and Projection-Cost Benchmark

Status: implementation landed on `research/c10-cache-benchmark`; CI and authoritative measurements are pending. This phase provides a reusable, correctness-gated kernel benchmark. It does not claim a measured speedup or end-to-end retrieval improvement.

## Research question

When the same candidate representations recur across queries, what are the measured costs of recomputing document-side K/V projections, filling the in-memory cache, and looking up warm cached projections? Does cache reuse preserve the exact projected K/V payload and cache accounting under a frozen projection?

## Scope

C10 is deliberately narrower than an end-to-end retrieval benchmark. It isolates the cache lifecycle that C8 introduced and provides an experimental primitive for later C11 integrated workload evaluation.

### Implemented

- `attention/src/c10_cache_benchmark.rs`: `benchmark_cache_lifecycle`.
- Three measured paths over identical candidate inputs:
  1. uncached K/V projection,
  2. cold cache fill,
  3. warm cache lookup.
- Two discarded warm-up rounds; minimum three measured repetitions.
- Nearest-rank p50/p95 timing summaries, cache hit/miss counts, hit rate, resident entries, and estimated K/V payload bytes.
- Exact K/V equality checks between the reference projections, uncached projections, and cold-filled cache entries.
- Input gates: non-empty candidates, minimum repetitions, unique candidate IDs, and valid candidate dimensions.
- Cache geometry validation and fingerprint checks.
- Unit tests for parity, report metrics, malformed inputs, duplicate IDs, and model/geometry fingerprint invalidation.

## Frozen measurement protocol

- Use a single frozen `QkvProjection` and model fingerprint across all three paths.
- Feed the same ordered candidate IDs and aligned head representations to every path.
- Do not change projection weights, candidate contents, head order, or dimensions between paths.
- Run at least 5 measured repetitions for authoritative results (the API accepts 3 for fast correctness tests).
- Record the full serialized report, repository commit SHA, machine/CPU, Rust version, build profile, input-generation seed, candidate count, head count, and dimensions alongside each run.
- Run cold-fill with a newly empty cache each repetition. Warm lookup uses a cache filled with the same inputs.
- Preserve all raw runs; reruns get new run IDs. Report neutral or unfavorable measurements as-is.
- Do not compare microseconds from different machines as if they were controlled paired observations.

## Required experiment matrix

| Factor | Required values |
|---|---|
| Candidate count | 32, 64, 128, 256, 500, 1,000, 2,000 (subject to memory/runtime) |
| Head count | 1, 3, 8 |
| Attention dimension | 64, 128, 384 |
| Key dimension | 32, 64, 128 |
| Value dimension | 32, 64, 128 |
| Cache path | uncached projection, cold fill, warm lookup |
| Repetitions | at least 5 measured per cell, after warm-up |
| Correctness | exact K/V parity, 100% expected warm hits, zero warm misses, valid cache geometry |

Only use supported dimension combinations and report skipped cells with reasons. The configurable `run_cache_lifecycle_sweep(&C10SweepConfig::default())` API executes the Cartesian matrix and returns one seed-keyed report per cell. The caller must persist the returned cells as immutable JSONL raw runs. End-to-end retrieval evaluation remains C11.

## Acceptance gates

1. Workspace CI compiles and tests the new API.
2. All unit tests pass, including malformed-input and fingerprint checks.
3. Every measured cell reports exact K/V parity.
4. Warm lookups yield one hit per requested candidate per measured repetition, zero misses, and the expected resident-entry count.
5. The report clearly labels payload memory as an estimate excluding allocator/HashMap overhead.
6. No performance or end-to-end latency claim is made without authoritative measurements.
7. Existing C8 and C9 semantics remain unchanged.

## Interpretation boundary

A faster warm lookup is expected to measure less work than a fresh K/V projection, but the magnitude depends on dimensions, candidate size, machine, and cache working set. The helper's timings do not include attention scoring, HNSW, serialization, storage, network, or complete query latency. C11 should integrate these measurements into a reproducible end-to-end workload and analysis pipeline.
