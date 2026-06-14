# AttentionDB — Phase 1: Storage Engine

**Goal:** Build the physical foundation (LSM-style document store, WAL, columnar projection store) before adding any query logic.

> **Note for Agents / Future Developers:**
> This phase deliberately keeps soft queries out. Focus only on durability and physical storage. The next phase will add HNSW indexes.

## Components

- **record.rs** — Core record with `id`, `version`, `fields`, `k_vecs`, `v_vecs`, `t_embed`
- **wal.rs** — Write-ahead log with CRC32 + durability modes (SYNC / GROUP_COMMIT / ASYNC)
- **projection_store.rs** — Columnar K/V vector storage
- **document_store.rs** — In-memory memtable + optional WAL-backed store
- **error.rs** — Storage error types

## Build & Test

```bash
cargo build
cargo test
```

## Run CLI (requires cli feature)

```bash
cargo run --features cli -- insert "Rayyan"
cargo run --features cli -- stats
```

## Current Status

- [x] Record serialization (MessagePack)
- [x] Basic WAL with append + CRC32
- [x] DocumentStore with CRUD + optional WAL
- [x] ProjectionStore skeleton
- [ ] Full LSM-tree (SSTables + compaction) — deferred
- [ ] On-disk projection store with memory-mapping

## Next Phase

Phase 2 will add HNSW indexes on top of this storage layer.
