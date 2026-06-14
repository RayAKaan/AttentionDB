# AttentionDB Phase 2 — HNSW Index Layer

High-performance per-head HNSW graphs for approximate attention scoring.

## Architecture

```
HeadIndexManager
├── HNSWIndex (semantic)     — ef_search=64, vector storage enabled
├── HNSWIndex (temporal)     — temporal decay support ready
├── HNSWIndex (structural)   — schema similarity
└── ...
```

## Quick Start

```bash
cd phase2
cargo run --release   # Run demo
cargo test            # Run 15 unit tests
cargo bench           # Run performance benchmarks
```

## Key Features

- **One HNSW graph per attention head** — independent indexes per head
- **Configurable ef_search** — trade recall vs. latency per query
- **Exact reranking** — `rerank_exact()` and `search_with_rerank()` for hybrid search
- **Multi-head weighted fusion** — `search_multi_weighted()` with per-head importance weights
- **Versioned binary persistence** — version byte + JSON config + length-prefixed vectors
- **Performance metrics** — atomic counters for insert count, search count, avg latency
- **17 unit tests** — comprehensive coverage of all paths

## API Overview

```rust
// Create with custom config
let config = HNSWConfig::new()
    .with_ef_search(64)
    .with_vector_storage(true);
let mut index = HNSWIndex::new("semantic", 256, config);

// Insert & search
index.insert(42, &vector)?;
let results = index.search(&query, 10, None)?;

// With exact reranking
let reranked = index.search_with_rerank(&query, 5, None)?;

// Multi-head weighted search
let manager = HeadIndexManager::new(256);
manager.add_head_with_config("semantic", config.clone());
let fused = manager.search_multi_weighted(
    &[("semantic", 1.0), ("temporal", 0.7)], &query, 10, None
)?;

// Metrics
let m = index.metrics();
println!("Avg search latency: {:.3}ms", m.avg_search_latency_ms);
```

## Performance Targets

| Metric | Target | Status |
|---|---|---|
| Vectors per head | 1,000,000 | Ready (configurable via `with_max_elements`) |
| Recall@10 | > 95% | Tunable via ef_search |
| p99 latency | < 5ms | Benchmarked |

## File Structure

```
src/hnsw_index.rs   — Core HNSW wrapper (HNSWConfig, HNSWMetrics, HNSWIndex)
src/head_index.rs   — Multi-head manager (HeadIndexManager)
src/persistence.rs  — Versioned persistence helpers
benches/hnsw_bench.rs  — Criterion benchmark (100k vectors, 5 ef values)
tests/hnsw_test.rs     — 17 unit tests
```

## Next Phase

**Phase 3 — Query Engine:** AQL parser, query planner, and executor on top of these indexes.
