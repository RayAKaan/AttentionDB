# C5 — Cross-Head Candidate-Generation Architecture Audit

Status: DRAFT (input to `c5-protocol.md`).
Branch: `comparative-study/c5-cross-head` (@ `cc661615c854fb5f8c6e6a35b0287cdf1d2a80f7`, main).
C5 base = `main` exactly as merged through C4 (`cc66161`). C4 evidence immutable.

## 1. Question and null hypothesis

AttentionDB's multi-head search performs one **independent** approximate search per
head on the **same query vector**, then unions candidates and fuses scores. The
`attend_detailed_inner` path (`core/src/collection.rs:380`) never lets one head's
candidate set influence another head's *candidate generation*. So the research
question for C5 is causal and architectural:

> Does adding a genuine **cross-head interaction step that changes the candidate
> set** (not just scores) improve retrieval, after equalizing candidate budget,
> per-head budget, ef effort, union size, rerank budget, memory, and latency?

- **H0**: after budget/latency/memory matching, cross-head interaction gives no
  meaningful improvement (AttentionDB is effectively independent per-head ANN +
  fusion).
- **H1**: cross-head interaction changes candidate generation and improves
  candidate recall / final recall / oracle alignment.
- **H2**: any H1 improvement concentrates on queries whose relevant docs are
  spread across multiple embedding spaces (multi-space-distributed queries).

## 2. The audited query path (what C5 must change or leave alone)

Stages in `Collection::attend_detailed_inner` (`core/src/collection.rs:380-727`):

```text
[entry] check deadline, validate dim, clone RetrievalConfig
MODE A single-head          → direct idx.search(query, top_k, None)   (retrieval.rs helper)
MODE B..E multi-head:
  1. candidate generation   per HEAD: idx.search(query, k, None), k = max(top_k*mult, min_cand)
                                  (parallel, chunked over max_search_threads; results head-ordered)
  2. candidate_union        provenance-preserving union, capped to candidate_budget.max(top_k)
                                  (core/src/retrieval.rs:72)
  3. per-head normalize     MinMax|ZScore|Softmax|Rank of each head's raw scores
  4. head gating profile    fixed | trained card | legacy net | uniform (sums to 1)
  5. BM25 normalize         (optional channel; MinMax)
  6. exact rerank (MODE E)  gated exact similarities over union (settings.similarity_metric)
  7. attention              AttentionScorer: Q=Wq·profile, K=Wk·[sims|ranks]; tanh bounded (identity-init)
  8. fusion                 S = α·attention + β·mhs + γ·bm25 over PRESENT components (renormalized)
  9. top-k                  deterministic (score DESC, id ASC)
```

Key audit-points that determine the causal lever for C5:

- **Candidate generation is the ONLY stage that uses the HNSW graph.** Everything
  after step 2 touches only the union of hits. So "cross-head interaction that
  changes candidate generation" must modify step 1 (what ids each head returns)
  — NOT scoring (steps 3-9 already do cross-head scoring).
- Step 1 calls `idx.read().search(query, k, None)` (`collection.rs:466`): the
  `ef` argument is `None`, so HNSW search falls back to the *per-head index*
  `settings.ef_search` (`hnsw/src/hnsw_index.rs:229`). The collection-level
  `Coll.settings` (written by c2pilot/c5pilot) is **read only for
  `similarity_metric`** (`collection.rs:598`) and `retrieval_config`. Per-head
  indexes are created from `HNSWConfig::default()` (ef 64) via
  `add_default_head`/`add_head_with_config`; `update_settings` has zero callers
  in the engine path. **Consequence: as built on main, `ef_search` is not a live
  per-head knob through the engine API.** Any C5 ef-control that must be real
  (budget axis EF) must therefore be applied per-head (e.g. an additive
  `update_settings`/search-with-ef path) and verified with an ef-sensitivity
  check in the C5 impl-audit.
- The multi-head manager (`multihead/src/manager.rs`) is present on `Collection`,
  but `attend_detailed_inner` does NOT call `MultiHeadManager::fuse` /
  `fuse_scores` from `multihead/src/fusion.rs`; fusion is done inline in
  `retrieval.rs`. The `multihead` crate is used mainly by the REPL/CLI and API
  paths (`query/src/executor.rs`). C5 must make its interaction deterministic
  regardless of which caller drives queries. The pilot driver (`c2pilot.rs`) uses
  `AttentionEngine` → `Collection::attend_detailed_with_stats` directly.
- Exact rerank (MODE E, mode==`Full`) is metric-consistent (`similarity(&metric,
  query, v)`), replaces *approximate* union scores with exact ones for the union,
  and is the C4 B7 behavior (`recall10 = 0.7421 SCI / 0.1488 NFC`), which
  SIGNIFICANTLY UNDERPERFORMED B2 (0.7920 / 0.1597) on both datasets. This is
  C4's most important finding and must be carried: C5's interaction must not be
  confounded with the rerank stage.
- Candidate-budget clamp lives in `candidate_union(set, budget)` — bounded; the
  budget is the same knob both arms must respect (`candidate_budget` default 500,
  `candidate_multiplier` 5, `min_candidates_per_head` 20).
- Per-head k = `(top_k * cfg.candidate_multiplier).max(cfg.min_candidates_per_head)`,
  clamped to head `len()` (`collection.rs:454-465`).

## 3. C1/C4 measurement protocol (reused, not redefined)

- Latency: C1 protocol = 5 fresh-process reps, warmup 20, seeded paired order
  (seed 20260925), per-query timing, p50/mean post-warmup.
- Quality: per-query recall@10 vs qrels and vs exact oracle; per-query mean over
  5 reps is the stable estimator (fresh-process execution is not bit-stable;
  sd ≤ 0.006 for C4 non-TRKB cells).
- Statistics (preregistered in C1, reused verbatim): paired per-query; paired
  bootstrap 10,000 (seed 20260925) 95% CI on the mean paired difference;
  two-sided Wilcoxon signed-rank; Holm step-down within family; Cohen's dz;
  |diff| < 0.01 absolute ⇒ "negligible" regardless of p.
- Fidelity anchors: B0 exact oracle in-process; `recall10_exact` vs canonical
  brute-force top-10; exact-top10 hashes.
- Datasets: DS-SCIFACT (300 test qids, n_docs 5183, dim 384, LE f32) and
  DS-NFCORPUS (323 test qids, n_docs 3633, dim 384), same C2/C3/C4 materialized
  embeddings + qrels splits (verified by C4DATA-VERIFY-001/`c4-dataset-validation.md`).
  Heads used by C4: TITLE / BODY / CITE (+ CANONICAL oracle reference only).

## 4. Design constraints (hard, from the C5 mandate)

1. HNSW stays the candidate generator. No replacement of the ANN.
2. No generic transformer, no LLM, no external reranker, no opaque learned box.
3. The cross-head mechanism must be deterministic at inference, inspectable,
   reproducible, bounded, Rust-compatible, and ablatable.
4. Any learning (C5-E) must be separated from inference and never change the
   inference path's determinism; C5-E only if the simple mechanisms justify it.
5. Budget axes equalized between arms: total candidate budget, per-head budget,
   ef effort, union size, rerank budget, memory, latency. Plus a same-latency
   comparison. Never silently give the interaction more budget.
6. Must falsify H1: measure per-head candidates pre/post interaction, additions,
   removals, overlap, final union, final top-k, per-head effort. "Did cross-head
   interaction cause exploration independent retrieval would not have done?"
7. Ablation ladder C5-A..E must be a strict stepwise ladder like C4 modes.

## 5. Candidate interaction mechanisms (licensed by the audit)

The graph `hnsw_rs::Hnsw` fields are private; only `search(query, k, ef)` +
`get_vector(id)` are exposed. Therefore in-place graph surgery is out; licit
levers are:

- **C5-A** single-head (`RetrievalMode::SingleHead`) — C4 B1 equivalent.
- **C5-B** independent multi-head + union + fixed-fusion gating (identity
  attention, no exact rerank) — C4 B2 candidate-generation-equivalent control.
- **C5-C** ONE cross-head interaction step. Two mandated candidates, both
  changing **step 1** only:
  1. **Cross-head query refinement (self-gated, "retrieve→interact→retrieve")**:
     after the first per-head search, derive a per-head *query offset* from the
     OTHER heads' top candidates (e.g. a normalized blend of the query with the
     centroid / sparse signals of candidates that only appeared under other
     heads), re-search each head once with the refined per-head query, and union
     the two rounds per head. Interaction is visible as "head H returned ids X
     only because head O surfaced their anchors".
  2. **Cross-head candidate injection / propagate (bounded, "beam across heads")**:
     take the union of round-1 per-head hits, and for candidates that were
     missing from a head's own list, run a *targeted exact/approximate* check in
     that head's space (via `get_vector`) and promote the survivors into that
     head's candidate list with a provenance tag `cross` vs `own`. Adds new ids
     into the union that independent per-head search would never have emitted.
- **C5-D** multiple / adaptive interaction steps (iterate C5-C with decaying
  budget) — gated on C5-C showing a signal.
- **C5-E** learned variant (learn which cross-head signals to trust) — gated;
  unlikely to be justified before C5-D evidence.

Both C5-C variants run with k/ef/union/rerank budgets **identical** to C5-B, the
interaction step itself consuming from the same per-query effort envelope, and
exact rerank OFF in the primary ablation (to keep the C4 B2-vs-B7 lesson clear);
a secondary arm may turn exact rerank on to check no interaction-only degradation.

## 6. Observable per-query ledger (the falsification kit)

For every interaction arm we must record, per query:

- round-1 per-head candidate lists (ids, raw scores, ranks)
- round-2 / injected ids with source head and tag (own|cross)
- union size pre/post interaction, additions, removals (removals only if we
  replace, otherwise additions-only), overlap across heads
- final top-k ids + scores, candidate recall vs oracle, recall@10 vs qrels,
  oracle-miss rate
- per-head effort (k used, ef used), total latency, union/rerank counts
- configuration_id for every cell

These are the exact rows the retrospective causal analysis (C5 section) and the
statistical pipeline consume, mirroring the C4 per-query schema.

## 7. What main actually is (base + immutability)

- Current HEAD: `cc661615c854fb5f8c6e6a35b0287cdf1d2a80f7` (merged C4 closure).
- C4 files `research/comparative-study/c4/**` are final; C5 adds a parallel
  sibling area and does not edit C4.
- The engine code C5 modifies (core/src/retrieval.rs, possibly a gated hook in
  collection.rs + hnsw search-with-ef) must stay CI-clean (fmt + clippy -D
  warnings on storage/core/query/api) — the full-workspace clippy stays CI-only
  (`dbtest.rs` is unix-gated).
- Any engine change is additive and off by default (interaction arm selected by
  config), so existing C1/C4 behaviors are unchanged unless a config opts in.