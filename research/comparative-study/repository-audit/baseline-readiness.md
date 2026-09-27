# C0 — Baseline-Readiness Audit (B0–B7 + harness + external systems)

Commit `fe4f92b`, branch `comparative-study/c0-audit`.

## B0–B7 readiness against the actual code

| Baseline | Status | Where / what is missing |
|---|---|---|
| B0 Exact single-vector search | **PARTIAL — needs implementation** | No brute-force/exact kNN endpoint exists. Candidates: (a) `scan_filtered` + client-side ranking (slow but exact; O(N·dim) per query); (b) a small exact-search helper in the HARNESS (engine-external, anti-circular) reading exported vectors; (c) HNSW `Full` mode's exact RERANK is exact only over the ANN candidate union — NOT exact search. Decide in C1 (recommend (b): harness-side exact oracle shared across ALL systems). |
| B1 Approximate single-vector search | **EXISTS** | `RetrievalMode::SingleHead` (`collection.rs:418`) — raw per-head HNSW with tuned ef possible via `CollectionSettings` (ef_search/ef_construction exposed). |
| B2 Multi-head, fixed equal weights | **EXISTS** | `RetrievalMode::FixedFusion` (uniform) or `attend_weighted` with explicit weights (`collection.rs:742`). |
| B3 Multi-head, learned query-dependent gating | **EXISTS + REQUIRES TRAINING** | `RetrievalMode::LearnedGating` + `engine.activate_gating_model` with a `ModelCard` trained via `learned/` (GatingDataset JSON → GatingMlp). No model ships; training data generation from a TRAIN split is C2/C3 work. Legacy net (multihead/gating.rs) is random-init + SGD — usable but v2 card path is the documented one. |
| B4 Multi-head without learned gating | **EXISTS (≡B2)** | Mode C with NO active model = uniform (audited fallback). The C/D distinction in the code ladder (QK attention) is identity-init and a measured no-op — B4 as "no gating at all" ≡ B2; the meaningful B4 variant is "gating ON, attention OFF" = mode C, already covered. C1 must define B4 precisely to avoid a vacuous arm. |
| B5 Hybrid lexical+semantic | **EXISTS (AttentionDB side)** | `attend_hybrid` RRF(k=60) / Fusion strategy + `attend_hybrid_filtered`. Requires TEXT fields (BM25 index built from record string fields at insert). External hybrid baselines (Weaviate/Elasticsearch/pgvector+BM25) need adapter/config work — Elasticsearch adapter exists in attentiondb-bench. |
| B6 Multi-vector / reranking baseline | **PARTIAL** | AttentionDB's multi-head IS a multi-vector design (one vector per head per doc) — B6 should be an EXTERNAL multi-vector/rerank system (e.g., Qdrant multi-vectors or a ColBERT-style reranker) — adapter/config work in C2/C4; no reranker exists in-engine. |
| B7 Production-intended configuration | **EXISTS = mode Full (E)** | `RetrievalConfig::default()` (`collection.rs:87`). NOTE: prior evidence shows E UNDERPERFORMED B/C on the phase-2 corpus — the study must report B7 as-is, favorable or not (Rule 10). |

## Harness readiness (`attentiondb-bench/`)

- Adapter trait `DatabaseAdapter` (connect/health/insert/query/capabilities)
  with REAL clients: Qdrant (`qdrant-client` gRPC, health-checked),
  pgvector (`sqlx` postgres), Elasticsearch (`reqwest`); AttentionDB CPU +
  GPU adapters; `translation.rs` for query mapping.
- **Declared-but-missing adapters:** milvus, weaviate, pinecone —
  `adapters/mod.rs:11-15` reference nonexistent files; enabling those cargo
  features FAILS to compile. Must be implemented (C2) or the systems
  classified blocked/deferred.
- Executor: orchestrator, server_manager (can launch/stop external servers),
  warmup, isolation, checkpoint, network_overhead.
- Workload: generator (synthetic), difficulty levels, failure_modes,
  ground_truth (exact cosine/euclidean/dot; uniform + per-head; semantic
  degeneration verifier), dataset_loader.
- Stats: bootstrap CI, Welch t-test + Benjamini-Hochberg, power analysis,
  hubness, LID — matches §11 requirements largely as-is.
- Metrics: latency percentiles, quality (recall/MRR), resource, energy,
  pareto, attentiondb_specific; reporting: JSON/CSV/LaTeX/ANN-style +
  pareto plots.
- Prior runs: `results/{smoke,smoke2,quick}` = AttentionDB(CPU)-only smoke on
  a DIFFERENT machine (Windows, 16 GB, git dca4b91, 2026-06-18, synthetic
  1k). `master_comparator_results.json` (repo root) lists Qdrant/Milvus/
  Weaviate/pgvector/Pinecone/Elasticsearch rows **whose provenance is not
  reproducible in-repo** — treat as UNSOURCED for this study (no raw-run
  registration, no registry entry). C1 must re-baseline everything on the
  documented C0/C1 machine.

## Dataset readiness

- In-repo corpora are ALL synthetic generators: `benchmarks/phase2b/src/
  corpora.rs` (controlled + multiview, deterministic seeds),
  `attentiondb-bench/src/workload/generator.rs`. NO public dataset
  (SIFT/ANN-Discount/TREC/MS MARCO etc.) is vendored — C1/C2 must add a
  dataset acquisition + hashing protocol (`datasets/` namespace) with
  documented licenses.
- AttentionDB ingest shape requires PER-HEAD vectors: public single-vector
  sets map to 1 head natively; multi-view sets (or view-generating
  transforms, e.g., disjoint projections of one embedding) are needed for
  honest multi-head evaluation. Synthetic multiview generators exist but
  Track A must ALSO include at least one dataset where head views have
  externally-defined semantics (C1 design decision).

## External-system feasibility (this sandbox: 2 vCPU, 1.9 GiB RAM, ~20 GB disk, no Docker/systemd; network available)

| System | Feasibility here | Notes |
|---|---|---|
| PostgreSQL + pgvector | **Likely feasible** (apt install postgresql + pgvector; needs ~100-300MB RAM) | adapter exists (sqlx); version/capability audit in C1 |
| Qdrant | **Feasible to attempt** (single static binary; RAM-guarded at this host's 1.9 GiB) | adapter exists (gRPC); HNSW memmap config may be required at 60-80k docs |
| Elasticsearch | **Risky at 1.9 GiB** (JVM heavy; default heap likely OOM) | adapter exists; classify OBSERVED_LIMIT if it cannot run under host constraints |
| Milvus (standalone) | **Not feasible here** (etcd+MinIO deps, memory footprint); milvus-lite (pip) is a possible embedded fallback — verify license/version in C1 | adapter file MISSING |
| Weaviate | **Uncertain** (single binary exists; RAM footprint at 1.9 GiB to verify) | adapter file MISSING |
| Pinecone | **Blocked without credentials/budget** (managed only; §14: no cloud spend without authorization) | adapter file MISSING; record blocked-unauthorized |
| MongoDB Atlas Vector Search | **Blocked** (Atlas-managed; community mongod lacks $vectorSearch) | no adapter; record blocked |

All feasibility entries are PRELIMINARY (documentation + resource reasoning);
C1 must confirm with version-pinned capability audits and a C2 smoke run per
system before any comparative claim.
