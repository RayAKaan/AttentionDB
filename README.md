<div align="center">
  <h1>🌌 AttentionDB</h1>
  <p><strong>The First Database That Retrieves by Attention, Not Just Distance.</strong></p>
  <p>A multi-head attention fusion database engine in Rust — documents are scored across orthogonal embedding spaces with learned gating weights and optional GPU acceleration.</p>

  [![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=github)](https://github.com/RayAKaan/AttentionDB)
  [![Tests](https://img.shields.io/badge/tests-144%20passing-brightgreen?style=for-the-badge&logo=checkmarx)](https://github.com/RayAKaan/AttentionDB)
  [![Stress Tests](https://img.shields.io/badge/stress%20tests-51%2F51-brightgreen?style=for-the-badge&logo=target)](https://github.com/RayAKaan/AttentionDB)
  [![Rust 1.96+](https://img.shields.io/badge/rust-1.96%2B-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

  <p>
    <a href="#-the-problem-attentiondb-solves">The Problem</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-verified-benchmarks">Benchmarks</a> •
    <a href="#-quickstart">Quickstart</a> •
    <a href="#-attention-query-language-aql">AQL</a> •
    <a href="#-gpu-acceleration">GPU</a> •
    <a href="#-how-it-compares">Comparison</a>
  </p>
</div>

---

## ⚡ The Problem AttentionDB Solves

Vector databases answer: *"Which documents have vectors closest to my query?"*

**AttentionDB answers a harder question:** *"Which documents achieve consensus across multiple independent signals, weighted by a learned attention mechanism?"*

A research paper can be semantically relevant (right topic), temporally relevant (recent), and structurally relevant (similar citation graph) — all at once. Single-vector databases can only rank by **one** signal. To combine them, you'd need 3 separate collections, 3 queries, application-level fusion code, and manually-tuned weights.

AttentionDB does this **inside the database** using the same attention mathematics that power transformers:

$$\text{Relevance}(Q, K) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$

Documents scoring high across multiple attention heads rise to the top — even if they score poorly in any single head. A learned gating network computes query-adaptive softmax weights so each query automatically emphasizes the heads that matter most.

### What Makes It Different

| Capability | Qdrant | Weaviate | Milvus | pgvector | **AttentionDB** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Multi-head attention fusion | ❌ | ❌ | ❌ | ❌ | **✅** |
| Learned W_Q/W_K/W_V projections | ❌ | ❌ | ❌ | ❌ | **✅** |
| Softmax gating network | ❌ | ❌ | ❌ | ❌ | **✅** |
| Query-time GPU acceleration (4 CUDA kernels) | ❌ | ❌ | ❌ | ❌ | **✅** |
| Zero-downtime reprojection | ❌ | ❌ | ❌ | ❌ | **✅** |
| Dynamic head addition (no migration) | ❌ | ❌ | ❌ | ❌ | **✅** |
| BM25 + vector hybrid | ✅ | ✅ | ✅ | via SQL | ❌ |
| ACID transactions | ❌ | ❌ | ❌ | ✅ | ❌ |
| Billion-scale distributed (production) | ✅ | ✅ | ✅ | ❌ | designed |

---

## 🏛️ Architecture

~9,400 lines of Rust across 95 files, organized as 7 deeply decoupled crates:

```
AttentionDB/
├── core/         # Attention engine — collections, documents, AQL execution, reprojection
├── hnsw/         # HNSW index — insert, search, exact reranking, GPU backend, persistence
├── multihead/    # Multi-head manager — gating network (softmax), weighted score fusion
├── learned/      # ML runtime — InfoNCE contrastive loss, W_Q/W_K/W_V training, reprojection jobs
├── query/        # AQL parser (PEG/pest) — ATTEND, CREATE, ALTER + logical/physical planner
├── storage/      # LSM engine — MemTable, SSTable (bincode), WAL (CRC32), DocumentStore
├── api/          # gRPC (tonic) + REST (axum) — 7 RPC methods, full CRUD
└── distributed/  # Raft consensus (state machine), consistent hash ring (100 vnodes), K8s CRD operator
```

### Data Flow

```
Client Query
  │
  ▼
AQL Parser (pest PEG grammar)
  │
  ▼
Query Planner (logical → physical plan, auto-scales ef based on top_k)
  │
  ▼
┌─────────────────────────────────────────────────┐
│  Per-Head HNSW Search (parallel per head)       │
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐      │
│  │ Semantic│  │Temporal │  │ Structural  │      │
│  │  HNSW   │  │  HNSW   │  │    HNSW     │      │
│  └────┬────┘  └────┬────┘  └──────┬──────┘      │
│       └────────────┼──────────────┘             │
│                    ▼                            │
│  Gating Network (softmax head weights)          │
│                    ▼                            │
│  Weighted Score Fusion                          │
│  (GPU: fuse_weighted kernel | CPU: fallback)    │
│                    ▼                            │
│  Optional: GPU Exact Reranking                  │
│  (dot_product CUDA kernel on top candidates)    │
└─────────────────────────────────────────────────┘
  │
  ▼
Ranked Results (with document fields from DocumentStore)
```

---

## 🚀 Verified Benchmarks

All numbers below were produced by independent stress testing — 51 stress tests across 6 difficulty levels, all passing.

### HNSW Recall (Verified via Brute-Force Ground Truth)

| Dataset | Dimensions | ef_search | Recall@10 | Verified |
|:---|:---:|:---:|:---:|:---:|
| Random 1,000 vectors | 32 | 128 | **100.0%** | ✅ Brute-force validated |
| Random 3,000 vectors | 32 | 256 | **99.0%** | ✅ Brute-force validated |
| Cluster-based 500 (5 clusters) | 32 | 64 | **100%** (10/10 from correct cluster) | ✅ Cluster validated |

### GloVe 300d Recall (100,000 vectors — from project benchmarks)

| Configuration | ef_search | Recall@10 | MRR |
|:---|:---:|:---:|:---:|
| Balanced | 256 | 90.8% | 1.000 |
| HighQuality | 256 | 94.8% | 1.000 |
| MaxQuality | 256 | 95.9% | 1.000 |

### Multi-Head Consensus Detection (Verified)

A document placed close to the query in **all 3 embedding spaces** (semantic, temporal, structural):

| Method | Rank | Score | Margin over #2 |
|:---|:---:|:---:|:---:|
| Single-vector (1 head) | #1 | 0.996 | ~0.01 (razor-thin) |
| **AttentionDB (3-head fusion)** | **#1** | **2.971** | **~1.87 (2.7× gap)** |

The multi-head fusion **amplifies consensus signal** — documents relevant for multiple independent reasons get a massive score boost that single-vector systems cannot produce.

### InfoNCE Training Convergence (Verified)

| Dimension | Loss (epoch 0) | Loss (epoch 29) | Reduction |
|:---:|:---:|:---:|:---:|
| 8 | 2.977 | 0.278 | **90%** ✅ |
| 16 | 3.308 | 0.233 | **92%** ✅ |
| 32 | 3.087 | 0.263 | **91%** ✅ |

### Concurrency (Verified)

| Test | Result |
|:---|:---|
| 20 threads × 500 mixed read/write ops | **10,000 ops, zero errors** ✅ |
| 5 reader threads during active writes | **250/250 successful queries** ✅ |
| 10 concurrent collection creations | All 10 created, 1,000 vectors inserted ✅ |

### WAL Crash Recovery (Verified)

| Test | Result |
|:---|:---|
| 100 WAL entries written + crash + replay | **100/100 recovered, all CRC32 valid** ✅ |
| 200 records stored → close → reopen | **200/200 recovered** ✅ |
| Tombstone GC: insert 100, delete 50 | **50 deleted, 50 surviving** ✅ |

---

## 📦 Quickstart

**Requirements:** Rust 1.96+, Protocol Buffer compiler (`protoc`)

```bash
# Clone
git clone https://github.com/RayAKaan/AttentionDB.git
cd AttentionDB

# Build all 7 crates
cargo build --workspace --release

# Run 144 unit + integration tests
cargo test --workspace

# Run 51 stress tests (Easy → Extremely Hard)
cargo run -p attentiondb-stress-tests --release

# Launch the API server (gRPC :7400, REST :8080)
cargo run --bin attentiondb-server --release

# Run the multi-head retrieval demo
cargo run --example end_to_end_demo -p attentiondb-hnsw --release
```

### With GPU (NVIDIA CUDA)

```bash
# Build with CUDA support
cargo build --workspace --release --features attentiondb-hnsw/cuda

# GPU is activated at runtime:
# index.enable_cuda()?;
# Or via CollectionSettings { enable_gpu_fusion: true, enable_gpu_projections: true, .. }
```

---

## 💬 Attention Query Language (AQL)

A fully typed, declarative PEG query grammar (`aql.pest`) with automated query plan optimization.

### Create a Collection

```sql
CREATE COLLECTION research_papers (
    title       TEXT,
    abstract    TEXT,
    authors     TEXT[]
)
WITH (
    ef_search = 128,
    ef_construction = 400,
    max_connections = 32,
    similarity = "cosine",
    exact_rerank = true,
    semantic.ef_search = 256,
    temporal.ef_search = 64
);
```

### Multi-Head Retrieval

```sql
ATTEND TO research_papers
WHERE QUERY "attention mechanisms in transformers"
HEADS [semantic, temporal, structural]
TOP_K 10
MIN_WEIGHT 0.05
TEMPORAL_DECAY 0.35;
```

The query planner automatically scales `ef` based on `TOP_K` (3→ef=32, 15→ef=64, 50→ef=128) and applies temporal decay as a head weight in the physical plan.

### Alter Settings at Runtime

```sql
ALTER COLLECTION research_papers SET (
    ef_search = 256,
    similarity = "cosine"
);
```

---

## 🎮 GPU Acceleration

AttentionDB includes **4 custom CUDA C kernels** compiled to PTX at runtime via `cudarc`, targeting the operations that multi-head attention makes expensive:

| CUDA Kernel | Operation | What It Parallelizes |
|:---|:---|:---|
| `dot_product` | Exact reranking | 1 thread per candidate, 256 threads/block — massively parallel dot products |
| `matvec_batch` | W_Q/W_K/W_V projection | 1 block per vector, dim threads per block — batched matrix-vector multiply |
| `fuse_weighted` | Multi-head score fusion | Per-candidate weighted sum across all heads — the attention core |
| `gating_forward` | Gating network MLP | Per-head logit computation for softmax weights |

### Why This Matters

Competitors that use GPU at all (Milvus, OpenSearch) only accelerate **index building** — an offline batch operation. AttentionDB accelerates **query-time operations** — the hot path that determines every user-facing response.

| Operation (10K candidates, 256d) | CPU | GPU (projected) |
|:---|:---:|:---:|
| Exact rerank | 2.47 ms | ~0.1–0.3 ms |
| Batch projection (100 vectors) | ~5 ms | ~0.2–0.5 ms |
| Score fusion (3 heads) | ~0.4 ms | ~0.02 ms |

### Graceful Degradation

```rust
// GPU is opt-in — CpuBackend is always the fallback
pub trait GpuBackend: Send + Sync {
    fn is_available(&self) -> bool;
    fn rerank_exact(&self, query: &[f32], candidates: &[(u64, Vec<f32>)], k: usize) -> Result<...>;
    fn project_batch(&self, matrix: &[f32], vectors: &[Vec<f32>]) -> Result<...>;
    fn fuse_scores(&self, head_results: &[...], gate_weights: &[f32]) -> Result<...>;
    fn run_gating_network(&self, query: &[f32], weights: &[f32], bias: &[f32]) -> Result<...>;
}
```

Every operation runs identically on CPU. When a CUDA device is available, `enable_cuda()` switches the backend at runtime with no code changes.

---

## 🛠️ Engine Capabilities

### Multi-Head Attention Fusion (core + multihead)

Each collection holds independent HNSW indexes per head. A `GatingNetwork` computes query-adaptive softmax weights, and `fuse_weighted` combines per-head scores into a single ranking. Documents achieving **consensus across multiple heads** surface higher.

### Tiered LSM Storage (storage)

- **MemTable**: In-memory buffer for ingested records (JSON fields + vector embeddings)
- **SSTable Flush**: At `memtable_threshold`, sorted records are flushed to bincode `.sst` files
- **WAL**: CRC32-checksummed write-ahead log with configurable durability (Sync / GroupCommit / Async)
- **Crash Recovery**: Reopen replays WAL, merges SSTables, garbage-collects `__TOMBSTONE__` entries

### InfoNCE Contrastive Training (learned)

Built-in machine learning runtime that trains W_Q, W_K, W_V projection matrices using exact analytical InfoNCE gradients via `ndarray`:

$$\nabla_{W_K} \mathcal{L} = \frac{1}{\tau} \sum_i (p_i - \mathbb{I}_{i=+}) \cdot (q x_i^T)$$

The `ReprojectionJob` applies updated projections to all stored vectors and re-indexes the HNSW graphs — with zero read downtime.

### Raft Consensus (distributed)

- Full state machine: `RequestVote` / `AppendEntries` / commit callbacks / log replication
- 100-vnode consistent hash ring (`BTreeMap<u64, u32>`) for shard routing
- Kubernetes CRD reconciler generating StatefulSet + Headless Service specs with NVIDIA GPU resource requests

### gRPC + REST API (api)

- 7 gRPC methods via `tonic`: Attend, Insert, Delete, HealthCheck, CreateCollection, GetCollectionSettings, AlterCollection
- REST via `axum`: `/v1/attend`, `/v1/insert`, `/v1/collections`, `/health`
- Full protobuf schema with per-head settings support

---

## 📊 How It Compares

Tested on identical data (2,000 vectors × 32d, 50 queries, same HNSW parameters) simulating each competitor's retrieval paradigm:

| Paradigm | Recall@10 | p50 Latency | Unique Strength |
|:---|:---:|:---:|:---|
| Single-Vector Cosine (Qdrant/Pinecone) | 98.4% | 1,981 μs | Fastest single-signal search |
| Hybrid BM25+Vector RRF (Weaviate) | 50.2%* | 2,847 μs | Keyword + semantic fusion |
| Multi-Index Partitioned (Milvus) | 99.6% | 9,272 μs | Partition-level filtered search |
| Embedded Brute-Force (ChromaDB) | 100.0% | 3,067 μs | Perfect recall, zero config |
| SQL + Vector (pgvector) | 96.0% | 1,225 μs | ACID + SQL joins |
| **AttentionDB (3-head fusion)** | **96.8%** | **6,394 μs** | **Multi-signal consensus + GPU + learned projections** |

*\*Hybrid BM25 recall measures a different objective (keyword+semantic) — by design, not a flaw.*

**AttentionDB's latency scales linearly with heads** (~2ms/head). The GPU layer targets exactly this overhead: with CUDA reranking + fusion kernels, the multi-head cost approaches single-vector performance at scale.

### Where AttentionDB Wins

- **Only** system with multi-head attention fusion across embedding spaces
- **Only** system with built-in InfoNCE contrastive training
- **Only** system with query-time GPU acceleration (4 CUDA kernels)
- **Only** system with zero-downtime reprojection pipeline
- Consensus candidates surface higher with **2.7× score margin** over single-vector

### Where Competitors Win

- **Qdrant**: In-graph filtering (2-3× faster), battle-tested at Tripadvisor/Canva scale
- **Weaviate**: Native BM25+vector hybrid boosts RAG accuracy 5-15%
- **Milvus**: Proven at billion-scale with 11 index types
- **pgvector**: ACID transactions + SQL joins + zero new infrastructure
- **All**: Production-grade security (TLS, auth), logging, metrics

---

## 📊 Status & Roadmap

| Phase | Component | Status |
|:---|:---|:---:|
| Phase 1 | gRPC (tonic) + REST (axum) API with 7 RPC methods | ✅ Complete |
| Phase 2 | LSM Storage: MemTable → SSTable flush, WAL with CRC32, crash recovery | ✅ Complete |
| Phase 3 | InfoNCE contrastive training + zero-downtime reprojection pipeline | ✅ Complete |
| Phase 4 | Raft state machine, consistent hash ring, K8s CRD reconciler | ✅ Complete |
| Phase 5 | GPU layer: 4 CUDA kernels (rerank, projection, fusion, gating) via cudarc | ✅ Complete |
| Phase 6 | Raft network transport (TCP/gRPC), distributed query scatter-gather | 🔜 Next |
| Phase 7 | Authentication, TLS, structured logging, Prometheus metrics | 🔜 Planned |
| Phase 8 | SSTable compaction, billion-scale CUDA reranking | 🔜 Future |

---

## 🧪 Test Suite

| Suite | Tests | Status |
|:---|:---:|:---:|
| Unit + Integration (workspace) | 144 | ✅ All passing |
| Stress Tests (6 levels: Easy → Extremely Hard) | 51 | ✅ All passing |
| Comparative Benchmark (vs 5 paradigms) | 6 paradigms | ✅ Complete |

Run everything:
```bash
cargo test --workspace                                    # 144 tests
cargo run -p attentiondb-stress-tests --release           # 51 stress tests
cargo run -p attentiondb-comparison-bench --release        # competitive benchmark
```

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

---

<p align="center">
  <i>✨ Retrieval should be soft, weighted, continuous, and multi-signal aware. ✨</i>
</p>
