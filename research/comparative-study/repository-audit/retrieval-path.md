# C0 — Retrieval-Path Audit

Commit `fe4f92b`, branch `comparative-study/c0-audit`. Every claim cites
file:line at this commit. The public surface (all under
`core/src/engine.rs:1102-1230` unless noted): `attend`, `attend_weighted`,
`attend_hybrid`, `attend_filtered`, `attend_filtered_with_deadline`,
`attend_hybrid_filtered_with_deadline`, `scan_filtered` — all delegating to
`Collection::attend_detailed_inner` (`core/src/collection.rs:380-730`) or the
filter pipeline.

## Execution stages (default config, `RetrievalConfig::default()`,
`collection.rs:85-118`)

| # | Stage | Implementation | Default parameters |
|---|---|---|---|
| 0 | Config validation | `cfg.validate()` (`collection.rs:119-134`); deadline checks per stage (§19) | mode=Full, normalization=MinMax, mult=5, min/head=20, budget=500, fusion={0.3,0.5,0.2}, rrf_k=60, hybrid=Rrf, parallel=true |
| A | SingleHead shortcut | raw HNSW scores, retired+non-finite filtered (`collection.rs:418-450`) | mode A only |
| 1 | Candidate generation | per-head HNSW search, k clamped to head length; parallel via scoped threads, results reassembled in head order (`collection.rs:452-506`) | per_head_k = max(top_k×5, 20); max_search_threads = available_parallelism |
| — | Empty-head handling | non-contributing heads dropped (`present_heads`); all-empty → empty result | `collection.rs:497-513` |
| 2 | Candidate union | `CandidateSet` (provenance per head) + budget clamp (`candidate_union`, `retrieval.rs`) | budget 500 |
| 3 | Per-head normalization | min-max per head over that head's returned candidates | MinMax |
| 4 | Gate profile | override > uniform(FixedFusion) > active ModelCard MLP > legacy linear+softmax net > uniform; renormalized Σ=1 | `collection.rs:536-556` |
| 5 | BM25 channel | optional `bm25_raw` min-max normalized (`collection.rs:558-566`); `bm25_channel` filters retired ids | absent unless hybrid |
| 6 | Exact rerank (mode E) | `similarity(metric, q, v)` per union candidate per head with stored vector (`collection.rs:597-620`) | mode Full |
| 7 | Candidate attention (mode ≥ D) | `AttentionScorer::score`: q=gate profile (identity W_q), k=[norm sims | 1/(1+rank)] (identity W_k, rank weight 0.1); logit·(1/√d) → 0.5(tanh+1) | `retrieval.rs:346-428` |
| 8 | Fusion | `final = Σ w_i·f_i / Σ w_i` over present features {attention 0.3, multi_head_similarity 0.5, bm25 0.2}; mhs = Σ_h gate_h·norm_h (absent head ⇒ 0, no renormalization — documented at `collection.rs:668-672`); mode E replaces norm with exact sims | `retrieval.rs:472-495` |
| 9 | Deterministic top-k | dedupe by id (max score), sort (score DESC, id ASC), truncate | `retrieval.rs:279-303` |

`PipelineStats {union_size, rerank_size, heads_present}` exposed via
`attend_detailed_with_stats` (`collection.rs:356`) — the harness's hook for
candidate-vs-final quality separation (RQ metrics requirement).

## Ablation ladder (exists in code — maps directly to B0–B4)

`RetrievalMode` (`collection.rs:20-37`): `SingleHead` (A/B0-ish),
`FixedFusion` (B/B2), `LearnedGating` (C/B3 — uniform until a model is
ACTIVE), `QKAttention` (D/B4-adjacent), `Full` (E = default production mode).
Mode overrides are per-call (`mode_override`), so a harness can run all five
modes against ONE engine without reconfiguration. `attend_weighted`
(`collection.rs:742-762`) = FixedFusion with explicit caller weights
(normalized; all-zero → uniform).

**Audit caveats for the ablation design:**
1. The default `Full` mode was measured WORSE than B/C on the Phase 2 corpus
   (R@10 0.558 vs 0.672, `benchmarks/phase2/ablation.csv`, E row) while
   costing ~2.1 ms p50 vs 239 µs single-head — the study must treat
   "AttentionDB default" and each ablation mode as SEPARATE configurations.
2. Mode C without an activated model ≡ mode B (uniform). The central
   B2-vs-B3 comparison therefore REQUIRES training + activating a gating
   model per dataset (train split only) — the training path exists
   (`learned/` GatingDataset/GatingMlp/trainer + registry) but no model
   ships.
3. Identity-init QK attention (D) is a measured no-op vs C
   (`research/phase2/findings/attention-findings.md`); a TRAINED-QK
   experiment would need new wiring (out of C0 scope; note for C1 decision).

## Filtered path

`attend_filtered_with_deadline` (`engine.rs:1148-1160`): post-filter over the
staged pipeline's candidates with bounded pool expansion; HARD guarantee:
every returned doc satisfies the filter; deadline miss → `CoreError::Timeout`,
never a partial unfiltered result. Filter evaluation is exact expression
evaluation on document fields (`query/src/filter.rs`); `scan_filtered` scans
and sorts by id (`engine.rs:1470-1500`).

## Hybrid path

`attend_hybrid` (`collection.rs:789+`): BM25 channel (in-memory inverted
index with phrase search, `core/src/bm25.rs:291-480`) + vector candidates;
default strategy RRF(k=60); `HybridStrategy::Fusion` routes through the
staged pipeline with bm25 feature. Filtered-hybrid filters the SPARSE channel
before fusion (`engine.rs:1222+`).

## Prior measured evidence for this path (in-repo, audited provenance)

- `benchmarks/phase2/ablation.csv`: A 0.763 / B 0.672 / C 0.672 / D 0.672 /
  E 0.558 R@10; p50 239 µs (A) → 1.7-2.1 ms (B-E) on the phase-2 synthetic
  corpus; per-head ablations A0-A7 (default head 0.763; semantic 0.562;
  lexical 0.348; graph 0.249; h4-h7 0.487→0.194).
- Phase 2B claim ledger (`research/phase2/findings/claim-ledger.md`): trained
  gating +33pp R@10 over fixed fusion on controlled corpora (0.9533 vs
  0.6244, PH2B-GATING-004); +32pp on multiview; multiseed 0.4365±0.0361 vs
  0.2128; gating ≈ oracle head selection on the controlled corpus; RRF loses
  to trained gating by 6.9-28.4pp (k=60 fixed, no sweep); gating cost
  0.87 µs/query.
- Limitations of ALL prior retrieval evidence: synthetic corpora, small query
  counts, single splits, ONE machine — the comparative study must redo this
  on public datasets under the C1 protocol. This is the single most important
  input to RQ1-RQ4.
