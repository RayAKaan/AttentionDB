# AttentionDB

A database built on the mathematics of attention.

AttentionDB replaces traditional binary match/no-match retrieval with continuous relevance scoring using scaled dot-product attention (Q·Kᵀ / √dₖ + softmax). It is designed as a hybrid between vector databases and relational databases, with a focus on semantic relevance, multi-dimensional scoring, and weighted aggregation.

## Project Structure

```
AttentionDB/
├── storage/      # Storage Engine (WAL, DocumentStore, ProjectionStore)
├── hnsw/         # HNSW Index Layer + GPU Reranking + Recall Benchmark
├── query/        # AQL Parser + Query Planner + Executor
├── multihead/    # Multi-Head Architecture + Gating Network + Score Fusion
├── api/          # gRPC (tonic) + REST (axum) Server & Client SDK
├── learned/      # Contrastive Training for Learned Projections (W_Q, W_K, W_V)
└── distributed/  # Sharding, Raft Replication, Read Replicas, K8s Operator
```

## Key Achievements

### Recall Performance on Real Data (GloVe 300d, 100k vectors)

| Config | ef | Recall@10 | MRR |
|---|---|---|---|
| Balanced | 256 | 90.8% | 1.000 |
| HighQuality | 256 | 94.8% | 1.000 |
| MaxQuality | 256 | 95.9% | 1.000 |

**Critical improvement:** Insert-time L2 normalization increased recall from ~19% to 71%+ on the Balanced configuration.

### End-to-End Demo

The project includes a working demo that demonstrates:

- Multi-head indexing (semantic, temporal, structural)
- Multi-head search with weighted score fusion
- Documents ranking high due to consensus across heads (even when scoring 0 in one head)

Run it with:

```bash
cargo run --example end_to_end_demo -p attentiondb-hnsw --release
```

## Building the Project

```bash
# Build entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Run all benchmarks
cargo bench --workspace
```

## Research Highlights

- **Recall Benchmark** (`hnsw/benches/recall_bench.rs`): CLI-driven, supports real `.fvecs` datasets, tests multiple graph quality configurations, exports JSON results.
- **Learned Projections** (`learned/`): Contrastive training pipeline for W_Q, W_K, W_V matrices.
- **Tunable Retrieval Parameters** (Design): Collection-level control over `ef_search`, `ef_construction`, `max_nb_connection`, and `similarity_metric`.

## Vision

AttentionDB introduces a new retrieval paradigm where relevance is computed as a soft attention operation rather than exact matching or nearest-neighbor search. The project aims to provide:

- Workload-specific optimization through tunable parameters
- Strong multi-dimensional relevance via multi-head architecture
- GPU acceleration for high-performance reranking and training
- Production-ready distributed capabilities

## Status

All 7 modules are implemented. The core retrieval layer (hnsw) has been validated on real embedding data with excellent recall. The project is ready for integration, further GPU work, and research experimentation.

## License

MIT License

---

*Built with the belief that retrieval should be soft, weighted, and semantically aware.*
