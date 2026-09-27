# C1 — Fairness & Resource Accounting

Study `comparative-study-001`, protocol v1.0.0. Implements charter Rules 1–15
with the C0-audited cost structure.

## Cost model: what is counted (charter §8)

Every comparative table carries, per system/mode, a CONFIG-FACTS block:

| Field | Definition |
|---|---|
| vectors_per_record | number of indexed vectors per record (AttentionDB H-head modes: H; single-vector systems: 1; B6: H) |
| dim_per_head | exported embedding dimension (384 text / 512 CLIP / dataset-native ANN dims) |
| total_indexed_dims | vectors_per_record × dim_per_head (headline disclosure column) |
| representation_model | embedding model id + version + hash, or "dataset-native" (ANN sets) |
| representation_cost | wall/CPU time to produce embeddings, when applicable and measurable (shared across systems in Track A; attributed to AttentionDB multi-head rows in the end-to-end view) |
| index_build_time | wall time, documented parallelism |
| ingestion_throughput | records/s (records, not vectors — vector/s also recorded) |
| raw_data_size | exported dataset bytes ingested |
| index_size | on-disk index/collection size after build (system-reported where available, du-verified) |
| storage_amplification | index_size / raw_data_size |
| peak_rss / steady_rss | sampler (500 ms) over the server/adaptor process tree |
| disk_footprint | total (raw + index + WAL/temp) |
| query_latency | per-query distribution (metric-registry.yaml boundaries) |
| cand_gen_latency / rerank_latency | AttentionDB via PipelineStats boundaries; external systems: channel timings only where APIs expose them (else n/a — never imputed) |
| cpu_utilization | /proc sampling where measurable; n/a flagged otherwise |
| failure_restart_overhead | only in W-lifecycle workloads (opt-in, preregistered) |

## Two reporting views

1. **Retrieval-only:** assumes representations exist; compares query-time
   behavior (latency/recall/resources at query time). The default view for
   Track A quality work.
2. **End-to-end:** adds representation generation + ingestion + index build
   (and training time for B3) — the default view for Track B and for all
   multi-head-vs-single-vector ingest comparisons.

## Non-negotiable disclosures

- D1: any AttentionDB multi-head mode vs any 1-vector-per-record system
  carries the ×H factors in the SAME table (ingest, storage, memory, index
  build) — no prose-only disclosure.
- D2: the same embeddings (byte-identical exports, hash-manifested) reach
  every Track A system; Track B may use system-native preprocessing and says
  so per row.
- D3: metric substitution (e.g., L2 vs cosine) is flagged in the table
  header; cosine is used everywhere it exists.
- D4: managed systems (if ever authorized) are reported in a separate
  sub-table with documented service tier and pricing-capture date — never
  ranked jointly with self-hosted rows (Rule 12).
- D5: tuning effort symmetry: AttentionDB receives a preregistered parameter
  grid of the SAME size and search discipline as competitors (Rule 8); any
  asymmetry is a labeled deviation.
- D6: top_k parity does NOT imply fairness (charter §8) — fairness claims
  rest on the CONFIG-FACTS block, matched embeddings, and matched recall
  protocols, not on equal k.

## Comparability impossibilities (declared up front)

- Per-head metric isolation (AttentionDB: one metric per collection) vs
  systems with per-field metrics — controlled by using one metric everywhere.
- Cross-candidate interaction: AttentionDB modes have none (C0); late-
  interaction baselines do — such comparisons are labeled architecture-
  asymmetric and confined to Track B.
- Concurrency semantics (A7: no isolation/MVCC) vs transactional systems —
  W13 compares throughput/latency under the documented semantics only.
