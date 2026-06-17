<div align="center">
  <h1>🌌 AttentionDB</h1>
  <p><strong>A database that retrieves by multi-head attention, not just vector distance.</strong></p>

  [![Build](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=rust)](https://github.com/RayAKaan/AttentionDB)
  [![Tests](https://img.shields.io/badge/tests-185%20passing-brightgreen?style=for-the-badge)](https://github.com/RayAKaan/AttentionDB)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
  [![Rust 1.96+](https://img.shields.io/badge/rust-1.96%2B-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org)
</div>

---

AttentionDB scores documents across **independent embedding spaces** (semantic, temporal, structural, or any custom head) and fuses the results with a learned gating network. Documents that rank high across multiple heads surface to the top — even if they're mediocre in any single head.

```
ATTEND TO research_papers
WHERE QUERY "0.12, 0.85, ..."
HEADS [semantic, temporal, structural]
TOP_K 10
TEMPORAL_DECAY 0.35;
```

No other vector database does this. Qdrant, Milvus, Weaviate, and pgvector search one vector space at a time. Combining signals requires three separate collections, three queries, and application-level fusion with manually-tuned weights. AttentionDB does it inside the engine with query-adaptive softmax gating.

---

## What's in the box

8 Rust crates, ~13,700 lines, 185 tests (all passing), zero warnings.

```
AttentionDB/
├── core/         Attention engine — collections, documents, AQL execution
├── hnsw/         HNSW index — search, reranking, GPU backend, persistence
├── multihead/    Gating network (softmax), weighted score fusion
├── learned/      InfoNCE contrastive training for W_Q / W_K projections
├── query/        AQL parser (PEG/pest), query planner
├── storage/      WAL (CRC32), SSTables (CRC32), memtable, compaction
├── api/          gRPC + REST, auth, TLS, validation, observability
└── distributed/  Raft state machine, consistent hash ring, K8s CRD operator
```

### What works (verified by tests, not marketing)

| Capability | Evidence |
|:---|:---|
| Multi-head attention fusion | Consensus candidate (close in 3 heads) scores 2.97 vs 0.99 for single-vector. Tested in stress suite. |
| HNSW recall | 99–100% recall@10 verified against brute-force ground truth (1K–3K vectors). |
| InfoNCE training | Loss reduces 90%+ over 30 epochs across dims 4, 8, 16, 32. W_Q and W_K projections converge. |
| Crash recovery | 100/100 WAL entries recovered after simulated crash, all CRC32 valid. |
| SSTable integrity | CRC32 on every SST file. Single-bit-flip detected. v1 (no checksum) backward-compatible. |
| Compaction | Triggered automatically after SSTable flush. Merges files, removes tombstones, reloads reader list. |
| Concurrency | 20 threads × 500 mixed read/write ops, zero errors, zero deadlocks. |
| Auth | API key auth on both gRPC (metadata check) and REST (middleware). SHA-256 hashed, env-configured. |
| TLS | REST: rustls via axum-server (file-based or self-signed). gRPC: tonic TLS with file-based certs. |
| Validation | Collection names, vector dimensions (max 4096), top_k (max 10K), field sizes, request body (10MB). On every handler. |
| Observability | `tracing` structured logging + Prometheus metrics (counters, histograms, gauges). `/metrics` scrape endpoint. |

### What doesn't work yet

| Gap | Reality |
|:---|:---|
| Distributed | Raft state machine passes tests in single-process simulation. `replicate_to_peers()` returns `Ok(peers.len())` without network I/O. ShardManager is not wired into the engine. No distributed queries. |
| gRPC TLS (self-signed) | File-based gRPC TLS works. Self-signed mode falls back to plaintext gRPC (REST self-signed works fine). |

---

## Quickstart

```bash
git clone https://github.com/RayAKaan/AttentionDB.git
cd AttentionDB
cargo build --workspace --release
cargo test --workspace                    # 185 tests
cargo run --bin attentiondb-server --release
```

The server starts gRPC on `:7400` and REST on `:8080`.

### Configuration

All via environment variables — no config files needed.

| Variable | Default | What it does |
|:---|:---|:---|
| `ATTENTIONDB_GRPC_PORT` | `7400` | gRPC listen port |
| `ATTENTIONDB_REST_PORT` | `8080` | REST listen port |
| `ATTENTIONDB_DATA_DIR` | `/data` | WAL and SSTable directory |
| `ATTENTIONDB_API_KEYS` | *(unset = open)* | Comma-separated API keys. When set, all endpoints except `/health` and `/metrics` require `Authorization: Bearer <key>` or `X-API-Key` header. |
| `ATTENTIONDB_TLS_CERT` | *(unset)* | Path to PEM certificate for production TLS |
| `ATTENTIONDB_TLS_KEY` | *(unset)* | Path to PEM private key for production TLS |
| `ATTENTIONDB_TLS_SELF_SIGNED` | `false` | Set to `true` for auto-generated dev certificate |
| `RUST_LOG` | `attentiondb=info` | Log level filter (`debug`, `trace`, etc.) |

### Examples

```bash
# With authentication
ATTENTIONDB_API_KEYS="my-secret,admin-key" cargo run --bin attentiondb-server --release

# With TLS (production)
ATTENTIONDB_TLS_CERT=/etc/tls/cert.pem ATTENTIONDB_TLS_KEY=/etc/tls/key.pem \
  cargo run --bin attentiondb-server --release

# With TLS (development, auto-generated self-signed cert)
ATTENTIONDB_TLS_SELF_SIGNED=true cargo run --bin attentiondb-server --release

# With debug logging
RUST_LOG=attentiondb=debug cargo run --bin attentiondb-server --release
```

### Docker

```bash
docker build -t attentiondb .
docker run -p 7400:7400 -p 8080:8080 \
  -e ATTENTIONDB_API_KEYS="my-secret" \
  attentiondb
```

### With GPU (NVIDIA CUDA)

```bash
cargo build --workspace --release --features attentiondb-hnsw/cuda
```

Four CUDA kernels compile to PTX at runtime via `cudarc`: `dot_product` (reranking), `matvec_batch` (W_Q/W_K/W_V projection), `fuse_weighted` (multi-head score fusion), `gating_forward` (softmax head weights). Falls back to CPU transparently.

---

## API

### gRPC (port 7400)

7 methods, all authenticated (except HealthCheck), all validated, all instrumented.

| Method | What it does |
|:---|:---|
| `Attend` | Multi-head search with pagination (`offset`, `top_k`). Returns `results`, `total_count`, `has_more`. |
| `Insert` | Insert document with vector embeddings per head. |
| `Delete` | Delete document by ID. |
| `HealthCheck` | Returns collection/head/vector counts. Unauthenticated. |
| `CreateCollection` | Create with configurable `dimension`, HNSW settings, per-head overrides. |
| `GetCollectionSettings` | Read current settings. |
| `AlterCollection` | Update settings at runtime. |

### REST (port 8080)

| Endpoint | Method | What it does |
|:---|:---|:---|
| `/v1/attend` | POST | Multi-head search with pagination |
| `/v1/insert` | POST | Insert document |
| `/v1/collections` | POST | Create collection (with `dimension` field) |
| `/v1/collections/{name}` | PUT | Alter collection settings |
| `/health` | GET | Health check (unauthenticated) |
| `/metrics` | GET | Prometheus metrics scrape (unauthenticated) |

### REST example

```bash
# Create a 128-dimensional collection
curl -X POST http://localhost:8080/v1/collections \
  -H "Authorization: Bearer my-secret" \
  -H "Content-Type: application/json" \
  -d '{"collection": "papers", "fields": [], "dimension": 128}'

# Insert a document
curl -X POST http://localhost:8080/v1/insert \
  -H "Authorization: Bearer my-secret" \
  -H "Content-Type: application/json" \
  -d '{"collection": "papers", "fields": {"semantic": "0.1,0.2,0.3,..."}}'

# Search with pagination
curl -X POST http://localhost:8080/v1/attend \
  -H "Authorization: Bearer my-secret" \
  -H "Content-Type: application/json" \
  -d '{"collection": "papers", "query": "0.1,0.2,0.3,...", "top_k": 10, "offset": 0}'
```

---

## AQL (Attention Query Language)

A PEG grammar (`aql.pest`) for expressing multi-head retrieval.

```sql
-- Create a collection with per-head HNSW overrides
CREATE COLLECTION papers (title TEXT, abstract TEXT)
WITH (
    ef_search = 128,
    ef_construction = 400,
    max_connections = 32,
    similarity = "cosine",
    exact_rerank = true,
    semantic.ef_search = 256,
    temporal.ef_search = 64
);

-- Multi-head retrieval with temporal decay
ATTEND TO papers
WHERE QUERY "0.12, 0.85, ..."
HEADS [semantic, temporal, structural]
TOP_K 10
MIN_WEIGHT 0.05
TEMPORAL_DECAY 0.35;

-- Alter settings at runtime
ALTER COLLECTION papers SET (ef_search = 256, similarity = "cosine");
```

The query planner auto-scales `ef` based on `TOP_K`: 3→32, 15→64, 50→128.

**Note:** `ATTEND` queries require a pre-computed query vector (the engine has no built-in embedding model). `execute_aql_with_vector(aql, Some(&vector))` accepts the vector; `execute_aql(aql)` returns an error for ATTEND queries to prevent silently searching with zeros.

---

## How it works

```
Client request (gRPC or REST)
  │
  ├─ Auth check (API key via header/metadata)
  ├─ Input validation (names, dimensions, limits)
  │
  ▼
AQL Parser (pest PEG) or direct API call
  │
  ▼
Query Planner → Physical plan (auto-scaled ef, overfetch, rerank)
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  Per-head HNSW search                                │
│  ┌──────────┐   ┌──────────┐  ┌────────────┐         │
│  │ Semantic │   │ Temporal │  │ Structural │  ...    │
│  │   HNSW   │   │   HNSW   │  │    HNSW    │         │
│  └────┬─────┘   └────┬─────┘  └─────┬──────┘         │
│       └──────────────┼──────────────┘                │
│                      ▼                               │
│  Gating network: softmax weights per head            │
│                      ▼                               │
│  Weighted score fusion (GPU fuse_weighted / CPU)     │
│                      ▼                               │
│  Optional: GPU exact reranking (dot_product kernel)  │
└──────────────────────────────────────────────────────┘
  │
  ├─ Pagination (offset + take)
  ├─ Metrics recording (latency, counts)
  ├─ Structured logging (tracing)
  │
  ▼
Response (with total_count, offset, has_more)
```

---

## Storage

- **WAL**: CRC32-checksummed, configurable durability (Sync / GroupCommit / Async).
- **SSTables**: v2 format with `[ASST magic][CRC32][bincode payload]`. Single-bit-flip detected on read. v1 files (raw bincode) remain readable for backward compatibility.
- **Memtable**: In-memory buffer, flushed to SSTable at configurable threshold.
- **Compaction**: Size-tiered. Triggered automatically after each flush. Merges small SST files, permanently removes tombstone entries, reloads reader list from disk.
- **Crash recovery**: On open, replays WAL, merges SSTables, garbage-collects tombstones.

---

## GPU acceleration

Four CUDA C kernels compiled to PTX at runtime via `cudarc`:

| Kernel | Operation | Parallelism |
|:---|:---|:---|
| `dot_product` | Exact reranking | 1 thread per candidate, 256 threads/block |
| `matvec_batch` | W_Q/W_K/W_V projection | 1 block per vector, dim threads/block |
| `fuse_weighted` | Multi-head score fusion | Per-candidate weighted sum |
| `gating_forward` | Gating network MLP | Per-head logit computation |

Enable with `--features attentiondb-hnsw/cuda`. The `GpuBackend` trait always falls back to `CpuBackend` — every operation works identically without CUDA, just slower.

Competitors that use GPU (Milvus, OpenSearch) accelerate **index building**. AttentionDB accelerates **query-time operations** — the hot path that determines response latency.

---

## Observability

- **Logging**: `tracing` with env-filter via `RUST_LOG`. File, line, thread ID, target in every log line.
- **Metrics**: Prometheus counters (`attentiondb_attend_total`, `_insert_total`, `_delete_total`, `_errors_total`), histograms (`_attend_latency_ms`, `_insert_latency_ms`), gauges (`_collections_count`, `_heads_count`, `_vectors_count`).
- **Scrape**: `GET /metrics` returns Prometheus text exposition format.
- **Per-request**: `LatencyTimer` RAII guard records timing on every handler.

---

## Comparison to other vector databases

Tested on identical data (2,000 vectors × 32d, 50 queries, same HNSW parameters):

| System | Recall@10 | p50 latency | What it can't do |
|:---|:---:|:---:|:---|
| Qdrant/Pinecone | 98.4% | 1,981 μs | No multi-signal fusion |
| Weaviate | — | 2,847 μs | No learned gating |
| Milvus | 99.6% | 9,272 μs | No attention mechanism |
| pgvector | 96.0% | 1,225 μs | No multi-head search |
| **AttentionDB** | **96.8%** | **6,394 μs** | **No BM25 hybrid, no ACID, no billion-scale distributed (yet)** |

AttentionDB's latency scales linearly with the number of heads (~2ms/head). The GPU layer targets exactly this overhead.

---

## Known limitations

- **Distributed layer is a simulation.** Raft state machine works in-process; no network transport; shard manager not wired to engine. Multi-node deployment is not available.
- **No BM25/keyword search.** Weaviate and Qdrant offer hybrid BM25+vector. AttentionDB is dense-vector only.
- **No ACID transactions.** pgvector has SQL joins and transactions. AttentionDB does not.
- **gRPC self-signed TLS falls back to plaintext.** File-based TLS works on both protocols.
- **REPL is limited.** The REPL calls `execute_aql()` without a vector, which now correctly errors on ATTEND queries.

---

## Building and testing

```bash
cargo build --workspace --release          # Build all 8 crates
cargo test --workspace                     # 185 tests
cargo run -p attentiondb-stress-tests      # 51 stress tests (Easy → Extremely Hard)
cargo run -p attentiondb-comparison-bench   # Comparative benchmark vs 5 paradigms
```

---

## License

[MIT](LICENSE)

---

<p align="center"><em>Retrieval should be soft, weighted, and multi-signal aware.</em></p>
