# AttentionDB

A database built on the mathematics of attention.

## Project Structure

```
AttentionDB/
├── storage/      # Storage Engine (WAL, DocumentStore, ProjectionStore)
├── hnsw/         # HNSW Index Layer (per-head graphs + reranking)
├── query/        # Query Engine + AQL Parser
├── multihead/    # Multi-Head Architecture + Score Fusion
├── api/          # gRPC + REST API + SDKs
├── learned/      # Learned Projections (Contrastive Training)
└── distributed/  # Distributed Mode + Replication
```

## Building

```bash
cargo build --workspace
cargo test --workspace
cargo bench --workspace
```

## License

MIT