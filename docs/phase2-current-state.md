# Phase 2 — Current Retrieval State (Audit)

Audited at Phase 2 start, from code (not README), at commit `7631c94`.
Scope: every stage a query touches, from API entry to serialized results.

---

## 1. Entry points and the two disconnected query paths

There are **two query paths**, and they do not share execution code:

**Path A — durable engine (the real one).**
`POST /v1/attend` (api/src/rest.rs `attend_handler`) → `AttentionEngine::attend`
(core/src/engine.rs) → `Collection::attend` (core/src/collection.rs). This is the
only path that touches durable collections, WAL-recovered indexes, retired-id
filtering, and BM25 built from durable records. Everything below describes
Path A unless stated otherwise.

**Path B — AQL planner/executor demo path (query crate).**
`parse_aql` (query/src/parser.rs, pest grammar query/src/aql.pest) →
`plan_query` (query/src/planner.rs: LogicalPlan → PhysicalPlan) →
`QueryExecutor` (query/src/executor.rs) operating on **standalone in-memory
`HNSWIndex`/`MultiHeadManager` maps** — not the engine's collections. Used by
the REPL/demo binaries. The durable engine never consults `plan_query`; the
PhysicalPlan types exist but are not executed against durable state.
`exact_filters` is parsed into `AQLQuery` and then ignored everywhere.

## 2. Current query grammar (AQL, Path B only)

```
ATTEND TO <collection> WHERE QUERY "<text>"
           [HEADS [h1, h2, ...]] [TOP_K n] [MIN_WEIGHT f] [TEMPORAL_DECAY f]
CREATE COLLECTION <name> [(field TYPE, ...)] [WITH (key = value, head.key = value, ...)]
ALTER COLLECTION <name> SET (key = value, ...)
```

Settings keys: `ef_search`, `ef_construction`, `max_connections`,
`similarity` ("cosine" | "dot_product" | "l2"), `exact_rerank`,
`enable_gpu_fusion`, plus `head.key` per-head overrides.

AST (`AQLQuery`): collection, query_text, heads, top_k (default 10),
min_weight (default 0.01), temporal_decay, exact_filters (unused).
There is **no filter AST**; no WHERE-comparison grammar exists.

REST (Path A) accepts a JSON body: `{collection, query|query_vector, heads,
top_k, query_text?}` — no filters, no pagination, no explain.

## 3. Path A pipeline as actually implemented

```
engine.attend(collection, heads, q, top_k)
  └─ Collection::attend:
       1. effective_top_k = max(top_k × OVERFETCH_MULTIPLIER(5), 20)
       2. for each head (SEQUENTIAL, in request order):
            HNSWIndex::search(q, effective_top_k, ef=None)
              → hnsw_rs search; score = 1.0 − distance   (distance = metric of
                the hnsw_rs build; settings.similarity_metric is stored but the
                Collection path never maps it to a distance function)
            normalize_head_scores: divide by per-head max if max > 0
       3. gate weights: GatingNetwork::forward(q) = softmax(W·q + b) if a
          network was loaded; otherwise uniform 1/N   (default: uniform)
       4. fuse_weighted_with_gating → fusion::fuse_scores:
            HashMap<u64, f32>, sum over heads of normalized_score × gate
            sort desc by partial_cmp (ties → Equal → arbitrary due to
            HashMap iteration order)
       5. filter_retired (numeric ids retired by delete/update)
       6. truncate to top_k
```

`attend_weighted`: same but static `(head, weight)` pairs instead of gating.
`attend_hybrid`: dense = attend(effective_top_k) → retired filter →
BM25 search(effective_top_k) → retired filter →
`bm25::reciprocal_rank_fusion(dense, sparse, top_k)` — RRF with k=0
(`1/rank`, not the standard `1/(k+rank)`), HashMap-aggregated, unstable ties.

## 4. Stage classification (per Phase 2 spec §1)

| Stage | Current implementation | Status |
|---|---|---|
| Candidate generation | per-head `hnsw_rs` search, ×5 overfetch, sequential | exists, unparameterized from caller |
| Candidate union | implicit `HashMap` sum in `fuse_scores`; no provenance/rank retained | naive |
| Scoring | per-head max-normalization; weighted sum; RRF (hybrid only) | partially defined |
| Attention | **head-level only**: `GatingNetwork` softmax(query) → per-head weights; uniform by default (network is never loaded automatically) | mislabeled as "attention"; no candidate-level scoring |
| Exact reranking | `HNSWIndex::search_with_rerank`/`rerank_exact` exist (store_vectors=true) but **`Collection::attend` never calls them**; `settings.enable_exact_reranking` is ignored in the engine path | dead code on the durable path |
| Filtering | none. Only the retired-id (tombstone) filter | missing |
| Ranking | sort desc, unstable ties, truncate | weak determinism |
| Output | `Vec<(u64, f32)>`; no breakdown, no explain, no pagination | minimal |

## 5. Head architecture

- `HeadIndexManager`: `HashMap<String, Arc<RwLock<HNSWIndex>>>`; one hnsw_rs
  graph + exact-vector store per head.
- `MultiHeadManager` (multihead crate): static per-head `weight` metadata +
  `fuse_*` helpers that delegate to `fusion.rs`.
- `GatingNetwork` (multihead/src/gating.rs): single linear layer
  `softmax(W·q+b)`, online SGD trainer with momentum + early stopping,
  weights persistable via serde. **No W_Q/W_K/W_V, no candidate interaction.**
- Heads are per-collection; every head must have the collection dim; heads
  found in data but missing from the catalog are auto-created at rebuild with
  default config.

## 6. Gating architecture

`get_gated_weights(query, heads)`: if `gating_network` is `Some` and its
output length matches, use it; else uniform. Nothing in Phase 1 ever loads a
network (no training loop wired to traffic; `load_gating_network_from` exists
for tests/tools). So **production behavior today = uniform head weights**.

## 7. Score normalization

- Per-head `normalize_head_scores`: `score /= max(head)` when `max > 0`.
  - all-negative head (possible if a metric yields negative similarity):
    max < 0 → **no normalization**, raw values fused.
  - all-equal head: unchanged (each already equal) — fine.
  - NaN: `fold(0.0, f32::max)` with NaN → max may propagate NaN → NaN fused
    into final scores. No guard anywhere.
- RRF: `1/rank` (k=0); rank-based so scale-free, but nonstandard and
  overweighting rank 1 vs rank 2 relative to the literature default k=60.

## 8. BM25 path

`Bm25Index` (core/src/bm25.rs): Okapi BM25 over whitespace/lowercase tokens,
per-collection, rebuilt deterministically from durable record text at recovery;
`search(query, top_k)` returns id→score. `search_phrase` exists (unused on the
engine path). BM25 scores are **unbounded** (not a similarity) — currently only
consumed through RRF, which is correct; any linear fusion with cosine would be
wrong without normalization.

## 9. Filtering

None. There is no filter representation, parser, or evaluator. The only
post-search filter is `filter_retired` (INV-3/INV-4 tombstone semantics).
REST/AQL cannot express metadata predicates.

## 10. Reranking

`rerank_exact` (hnsw crate): computes exact similarity from the per-head
exact-vector store for candidate ids. Correct but **not wired into
`Collection::attend`/`attend_hybrid`**; `enable_exact_reranking` (default true)
is respected nowhere on Path A. This is a gap between claims and behavior.

## 11. Pagination

None. No offset, no cursor. Repeat queries re-run everything.

## 12. Concurrency

- `Collection::attend` holds `head_manager.read()` across all head searches;
  head searches are **sequential** on the caller's thread; tokio handlers call
  the engine directly (attend is CPU-bound sync work inside async context —
  rest.rs uses `block_in_place` for backup; attend does not, relying on the
  multi-threaded runtime).
- Mutation/checkpoint serialization is via `mutation_gate` (Phase 1).
- No query timeouts, no cancellation, no candidate budgets beyond the fixed
  ×5 multiplier.

## 13. GPU path

`enable_gpu_fusion` setting + `gpu` cargo feature (cudarc). Fusion-side GPU
helper exists (`MultiHeadManager::is_gpu_fusion_enabled`), feature-gated;
requires CUDA toolkit to build (unavailable in the current environment — CPU
is the correctness reference). No GPU test comparing CPU/GPU results exists.

## 14. Limits, safety, determinism

- No top_k cap: `top_k = 10_000_000` → ×5 overfetch → hnsw_rs allocation and
  a fused vector of that size. Unbounded text query length into BM25.
- Determinism: hnsw_rs seeds its layer RNG per process (build-time
  nondeterminism across restarts — documented in Phase 1); **query-time**
  nondeterminism exists in tie ordering (HashMap iteration + `sort_by` with
  `Equal` fallback) and in f32 summation order (fixed head order — actually
  deterministic per call; only ties are unstable).
- Float safety: no NaN/Inf guards on any scoring path; zero-vector cosine
  behavior delegated to hnsw_rs internals.
- `min_weight`, `temporal_decay`: parsed; unused on Path A (temporal_decay
  influences only the unused Path B planner weights).

## 15. Known correctness / performance issues (Phase 2 targets)

1. **No filtering** (spec §11–13) — biggest functional gap.
2. **`enable_exact_reranking` ignored** (spec §10) — the "precise ordering"
   stage is missing; HNSW approximate scores are final.
3. **No candidate-level attention** (spec §3) — "attention" is head-weight
   softmax only; cannot justify the name yet.
4. Sequential multi-head search (spec §5) — no parallelism.
5. Union loses provenance (no per-head scores/ranks retained → no explain,
   no proper RRF over multi-head unions).
6. RRF uses k=0 (nonstandard); ties unstable (HashMap + Equal fallback).
7. No NaN/Inf guards; max-normalization degenerate cases unhandled.
8. No query limits/cancellation/timeouts (spec §18–19).
9. No pagination (spec §20).
10. Path B (AQL planner) disconnected from the durable engine; `plan_query`
    output is never executed against durable collections (spec §16).
11. No retrieval metrics beyond request-level REST timers (spec §43).
12. `similarity_metric` setting is validated but the engine attend path never
    selects a distance function by it (hnsw_rs is built with its default
    metric; exact rerank uses its own formula) — metric semantics are
    inconsistent across HNSW/exact/normalize (spec §10/§39).

## 16. What is fine (do not break)

- WAL-first durability, recovery gate, checker, retired-id semantics (Phase 1).
- BM25 determinism (rebuilt from records at recovery).
- Per-head stores with exact-vector side store (`store_vectors=true`) —
  the substrate exact reranking needs already exists.
- GatingNetwork training machinery (SGD, early stop, persistence) — reuse,
  do not duplicate.

## 17. Phase 2 plan of record (short form)

1. `retrieval` module in core: staged pipeline types (CandidateSet,
   CandidateFeatures, RankedCandidate), union with budgets, normalization
   (min-max/z-score/softmax/rank) with float safety, RRF (k=60 default),
   exact rerank wiring, Q/K candidate attention, ablation modes A–E.
2. Filter AST + typed evaluator + planner integration (pre/post heuristics).
3. Parallel head search (rayon, bounded) with parallel≡serial tests.
4. Planner on the durable path + EXPLAIN + limits + timeout/cancellation +
   cursor pagination.
5. Metrics/tracing spans; REST/gRPC semantic equivalence tests.
6. Benchmark harness (benchmarks/phase2/): ablation A–E, scale 1K→1M,
   filter selectivity, hybrid variants; recall/latency regression baselines.
7. docs/retrieval/* (architecture, attention, multi-head, scoring, hybrid,
   filtering, planner, benchmarks, explain) + phase2-final-report.md with
   WIN / NO SIGNIFICANT WIN / TRADEOFF per technique.
