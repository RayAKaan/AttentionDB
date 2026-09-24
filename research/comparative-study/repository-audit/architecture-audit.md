# C0 — Architecture Audit (Comparative Benchmarking Study)

Initiative: research-grade comparative benchmarking of AttentionDB.
Audit commit: `fe4f92b2187bf631125e975a5db8a202b5316d00` (main @ E11 close),
tree sha16 `d12fa093bc294977` (established sha16 procedure), working tree
clean, audit branch `comparative-study/c0-audit`. Phase 3E artifacts were NOT
modified.

## 1. Workspace map (verified against Cargo manifests)

| Crate | Role | Evidence |
|---|---|---|
| `storage/` | WAL (framed segments + sidecar), SSTables, DocumentStore + BlockCache(50k), catalog, crashgate instrumentation | `storage/src/{wal,sstable,document_store,catalog,compaction,crashgate}.rs` |
| `hnsw/` | Wrapper over the `hnsw_rs` library: per-head `HNSWIndex`, `similarity()` (cosine default; dot_product; l2-as-negative-squared-distance), CollectionSettings (ef_search/ef_construction/metric/max_elements=100_000 default), persistence (graph persistence, async compaction, backup) | `hnsw/src/hnsw_index.rs:5-72`, `hnsw/src/settings.rs`, `hnsw/src/persistence/` |
| `query/` | AQL grammar (pest), parser, planner, filter expressions, executor | `query/src/{aql.pest,parser,planner,filter,executor}.rs` |
| `multihead/` | HeadConfig/HeadType metadata, legacy linear+softmax GatingNetwork (+SGD trainer), MultiHeadManager | `multihead/src/{head,gating,manager}.rs` |
| `learned/` | Phase 2B gating v2: `GatingDataset` (train/val/test + content hash), `GatingMlp` (hidden layer, softmax, seeded deterministic init), `ModelCard` + `TrainingMeta`, model registry (active pointer, hot-swap), projection/reprojection trainers, contrastive + eval helpers; `main.rs` is a synthetic demo | `learned/src/{gating_v2,registry,projection,reprojection,contrastive,eval,trainer}.rs` |
| `core/` | Engine, Collection (retrieval pipeline lives HERE), retrieval stage helpers (union/normalize/fusion/AttentionScorer/RRF/top-k), BM25 index, transactions, backup, checker | `core/src/{engine,collection,retrieval,bm25,transaction,backup,checker}.rs` |
| `api/` | REST server + OpenAPI + auth/rate-limit/observability | `api/src/` |
| `distributed/` | ~1.2k lines; OUT OF SCOPE (Phase 3E sealed non-claims: no distribution/replication claims) | `distributed/src/` |
| `attentiondb-bench/` | Pre-existing comparative harness: adapter trait + AttentionDB(CPU/GPU), Qdrant, pgvector, Elasticsearch adapters (real clients); workload generator/difficulty/failure-modes/ground-truth; stats (bootstrap CI, Welch + BH, power, hubness, LID); metrics (latency/quality/resource/energy/pareto); reporting (json/csv/latex/ANN-benchmark style); server manager; prior smoke results from a DIFFERENT machine (Windows/16GB, 2026-06-18, git dca4b91) | `attentiondb-bench/src/`, `attentiondb-bench/results/` |
| `benchmarks/phase2, phase2b, phase3/` | Historical phase research harnesses (phase3 = the sealed E1–E11 evidence; DO NOT MODIFY) | `benchmarks/`, `research/phase2*`, `research/phase3/` |
| `sdk/python`, `examples/`, `design/gpu/` | SDK, demos (incl. `master_research_lab_benchmark.rs`), GPU design report | — |

## 2. Direct answers to the mandate's questions A–O

**A. What does a retrieval head actually represent?**
A named, independent HNSW index (`attentiondb_hnsw::HNSWIndex` via
`HeadIndexManager`, `core/src/collection.rs:150-152`, `hnsw/src/head_index.rs`)
over f32 vectors of the collection's dimension. Each document carries a map of
per-head vectors (`Record.k_vecs: HashMap<String, Vec<f32>>`,
`core/src/engine.rs:842`); insert writes each head's vector into that head's
index. A `HeadType` enum (Semantic/Structural/Temporal/Relational/
FieldSpecific/Custom, `multihead/src/head.rs:4-10`) exists as DESCRIPTIVE
METADATA only — retrieval treats every head identically.

**B. Separate embedding / index / metric / transformation / candidate method?**
Separate INDEX: yes (one HNSW graph per head). Separate EMBEDDING: only in the
sense that the CALLER may supply different vectors per head — the engine has
no embedding models. Separate METRIC: NO — one `similarity_metric` per
collection (`hnsw/src/settings.rs:21-22`, default "cosine"). No per-head
transformation (learned projections exist in `learned/src/projection.rs` but
are NOT wired into the query path). Same candidate-generation method for all
heads (HNSW search, identical ef parameters).

**C. How does the GatingNetwork produce head weights?**
Three tiered sources at `core/src/collection.rs:536-546`:
1. `gate_override` (explicit caller weights);
2. `FixedFusion` mode → uniform 1/n;
3. otherwise: trained `ModelCard` MLP if one is ACTIVE
   (`get_trained_weights`, `collection.rs:202-220`: MLP forward on the QUERY
   VECTOR, weights picked by head name from the card's `head_names` order)
   → legacy `GatingNetwork` (single linear layer + softmax over the query,
   `multihead/src/gating.rs:103-111`) → uniform fallback. Both networks
   softmax; profile is renormalized to sum 1 (`collection.rs:548-556`).

**D. At what stage are weights applied?**
FUSION (stage 8 of `attend_detailed_inner`): the multi-head similarity is
`Σ_h gate_h · norm_h(id)` over min-max-normalized per-head scores (or exact
similarities in mode E), `core/src/collection.rs:652-686`. Gates do NOT
affect candidate generation (every present head is searched with the same
per_head_k regardless of its gate). The gate profile also forms the attention
query vector (stage 7).

**E. Does AttentionDB perform candidate-level attention?**
A bounded form: `AttentionScorer` (`core/src/retrieval.rs:346-428`) computes,
per candidate, a tanh-squashed dot product between the head-gate profile
(identity `W_q`) and the candidate's per-head feature vector
[normalized sims | reciprocal ranks] (identity `W_k`, rank weight 0.1).
O(C·(F+D)), NOT softmax over candidates, NO cross-candidate interaction
(explicit design note, `retrieval.rs:330-334`), and NOT learned — weights are
identity-initialized with NO training path wired (confirmed by
`research/phase2/findings/attention-findings.md`: measured mathematical no-op
vs mode C, identical R@10/NDCG to 3 decimals).

**F. Cross-candidate interaction?** None (by design, for page-stable scores).

**G. How are candidates from multiple heads merged?**
Stage 1: each PRESENT head searches HNSW with `per_head_k = max(top_k × 5,
20)` (default config). Stage 2: `CandidateSet` union with provenance, clamped
to `candidate_budget = 500`. Stage 3: per-head min-max normalization. Stage
8: gated sum (+ attention 0.3 / BM25 0.2 weights when present; weights
renormalize over present components, `fuse_candidate`, `retrieval.rs:472`).
Stage 9: deterministic top-k. Hybrid default is RRF (k=60) over the union of
vector candidates + the BM25 channel (`attend_hybrid`, `collection.rs:789+`).

**H. Duplicate candidates?**
The same numeric id from several heads is ONE candidate with per-head
features; `deterministic_top_k` dedupes keeping the best score
(`retrieval.rs:288-303`, "duplicate results impossible" §46).

**I. Head returns no candidates?**
That head is dropped from `present_heads` before gating/normalization —
weights renormalize over contributing heads only. ALL heads empty → empty
result (`collection.rs:508-513`). A head missing from the collection is
silently skipped in multi-head modes (`.ok()` at stage 1) but errors in
SingleHead mode.

**J. Tie resolution?**
`rank_comparator` (`retrieval.rs:279-283`): score DESC, then numeric id ASC.
Fully deterministic and thread-count independent (parallel ≡ serial, tested).

**K. Trained, pretrained, random, or manual gating?**
NO model ships with the repo: `gating_model: RwLock<Option<…>>` starts `None`
(`collection.rs:156`). Unless a ModelCard is ACTIVATED per collection
(`engine.activate_gating_model`, `engine.rs:1322-1374`, via the file registry
with an atomic `active` pointer, `learned/src/registry.rs`), the default
configuration runs with UNIFORM effective weights even in modes C/D/E. The
legacy network initializes weights randomly (`thread_rng`) and has an online
SGD trainer; v2 models are trained offline from `GatingDataset` JSON
(explicit train/val/test splits, content hash, `TrainingMeta`).

**L. Cost of gating evaluation?**
One MLP forward per query: measured 0.87 µs/query (1,188 params, 20,288 B
serialized; Phase 2B PH2B-LATENCY-001, claim C4 in
`research/phase2/findings/claim-ledger.md`).

**M. Cost of querying each head?**
One HNSW search per head with identical k. Phase 2 ablation
(`benchmarks/phase2/ablation.csv`): single head p50 239 µs → 4-head union
p50 1,734 µs on that corpus (~7×; parallel search recovers 1.39–1.72× at
4/8 heads, C6). Cost scales ≈ linearly with contributing heads.

**N. Which operations scan all candidates?**
Mode E exact rerank: `similarity()` computed for every union candidate per
head where the vector is stored (`collection.rs:597-620`) — O(union × heads
× dim). `scan_filtered` scans all records by design (filter soundness
contract). BM25 search traverses postings for query terms. The checker is
full-state. ANN candidate generation itself is graph-bounded (ef_search).

**O. Experimental / incomplete / unsupported parts?**
See `known-limitations.md`: identity-init QK has no training path; no
pretrained gating model ships; `learned/main.rs` is a synthetic demo; GPU
adapter + `design/gpu` untested in this environment; Milvus/Weaviate/Pinecone
adapter FILES are missing (feature-gated declarations in
`attentiondb-bench/src/adapters/mod.rs:11-15` reference nonexistent modules —
enabling those features fails to compile); `master_comparator_results.json`
contains external-system rows whose provenance is NOT reproducible in-repo
(single Windows-machine smoke run on 2026-06-18 with only an AttentionDB row
present); `distributed/` is out of scope; `hnsw_rs` caps `max_elements`
(default 100k/head; E10 measured the boundary).

## 3. What the system is NOT

It does not implement transformer attention over candidates (no softmax
attention across the candidate set, no learned Q/K/V projections, no
multi-head *self*-attention between documents). "Multi-head" = multi-INDEX
score fusion with optional query→head-weight gating. Any comparative-study
language must use these audited terms, not the README's marketing framing
(`README.md` claims "query-adaptive softmax gating" — true only of the
gating tiers, and only when a model is activated).
