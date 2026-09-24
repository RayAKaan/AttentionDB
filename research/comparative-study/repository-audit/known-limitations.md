# C0 — Known Limitations (implementation, evidence, environment)

Commit `fe4f92b`, branch `comparative-study/c0-audit`. These limitations are
inputs to C1's methodology (each must either be controlled for, disclosed, or
anchored as an explicit non-claim).

## Implementation limitations

1. **Heads are caller-supplied vectors, not learned spaces.** The engine does
   not embed; "independent embedding spaces" holds only if the DATASET
   provides genuinely different per-head views. Any study claim about
   multi-head benefit is conditional on view quality (RQ6/RQ10).
2. **No pretrained gating model ships.** Default effective gating = uniform
   (`collection.rs:156`, fallback chain 536-546). The B3 arm requires
   training on a train split; without it, "learned gating" claims are
   unsupported.
3. **Identity-init QK attention is untrained and measured a no-op** vs mode C
   (`research/phase2/findings/attention-findings.md`); no training path is
   wired. Mode D must be labeled as such; trained-QK would be NEW engineering
   (C1 decision; prompt §5 permits isolated, tested implementation on this
   branch if needed).
4. **Default production mode (E/Full) was measured WORSE than B/C** on the
   phase-2 corpus (R@10 0.558 vs 0.672) at ~9× single-head latency
   (`benchmarks/phase2/ablation.csv`). The exact-rerank stage recomputes
   similarity over the whole union (O(union×heads×dim)) — a latency/recall
   trade-off that must be measured per workload, not assumed.
5. **One metric per collection** (`hnsw/src/settings.rs:21`); no per-head
   metric/transformation in the query path (learned projections exist but are
   unwired, `learned/src/projection.rs`).
6. **hnsw_rs constraints:** max_elements default 100k/head (E10 boundary);
   no physical deletes (retired-id filtering; graph nodes persist until
   hygiene rebuilds the store) — update/delete-heavy workloads (W12) measure
   this design as-is.
7. **Concurrency model:** single-writer semantics under a mutation gate;
   LWW, no isolation levels (A7). W13 must use the established paced-reader
   pattern; no linearizability comparisons with MVCC systems are like-for-
   like and must be labeled.
8. **BM25 is an in-memory inverted index built from record string fields** —
   hybrid quality depends on field text quality; no external analyzer
   ecosystem (stemming etc.) — compare against systems' native BM25 with
   disclosure.
9. **Gating training data format** (`GatingDataset`) requires per-query head
   targets/examples — generating honest training data WITHOUT test leakage
   is a C1 protocol item (prompt §6: never train on test queries/labels).

## Evidence limitations (prior art — usable, with caveats)

10. All in-repo retrieval-quality evidence is SYNTHETIC-corpus, small-query,
    single-split, single-machine (phase2/phase2b, claim ledger C1-C7). It
    motivates the study; it cannot BE the study.
11. `master_comparator_results.json` external rows are unsourced in-repo
    (single Windows smoke run, AttentionDB-only raw row present). Not
    admissible as comparative evidence; re-baseline in C2+.
12. Phase 3E is a single-system reliability/scale program — its runs are NOT
    head-to-head evidence (§2 of the mandate) and stay sealed.

## Environment limitations

13. Benchmark host: 2 vCPU, 1.9 GiB RAM, ~20 GB disk, no Docker/systemd.
    Elasticsearch/Milvus-standalone likely infeasible at this memory class;
    scale ladder beyond ~80k docs (AttentionDB) is host-blocked per A10.
    JVM/Go runtimes must be sized carefully; every external run needs a
    feasibility smoke + memory guard (§12, §15).
14. Pinecone/MongoDB Atlas: managed-only, blocked absent credentials +
    explicit budget authorization (§15). Record as blocked/deferred, never
    fabricate.
15. No CI runner access for GPU/Numpy-heavy tooling beyond what's installable
    via cargo/apt/pip in-sandbox; dataset downloads must be license-checked
    and hash-recorded (C1 dataset protocol).

## Comparability risks to control in C1

16. AttentionDB default mode ≠ competitor defaults (Full/E includes exact
    rerank + attention): Track A must run matched-mode ablations; Track B
    must compare each system's DOCUMENTED production configuration with
    tuning budgets recorded (Rules 5-8).
17. Recall targets: AttentionDB ef_search + candidate_multiplier +
    candidate_budget vs competitors' ef/hnsw_m/nprobe — define a matched-
    recall sweep protocol (Rule 5) with per-system parameter grids recorded
    BEFORE outcomes are seen (preregistration).
18. Per-head vectors multiply ingest cost ×heads (independent HNSW graphs) —
    RQ4/RQ9 must measure and report this explicitly.
