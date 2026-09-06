# PH3B complementary multi-view + text/hybrid analysis (PROVISIONAL)

Status vocabulary per spec §20: SUPPORTED / NOT SUPPORTED / OPEN QUESTION.
No permanent finding numbers are assigned yet (§20); F-P3-1..F-P3-4 remain
provisional. Experiment IDs: PH3B-COMP-001 (primary, seeds 42/7/1),
PH3B-COMP-002-M30/-M20 (FAILED, preserved), PH3B-COMP-002-D256 (medium
probe), PH3B-COMP-003 (reproduction), PH3B-BM25-001 (BM25 verification),
PH3B-HYBRID-001 (k-sensitivity recorded inside COMP-001 artifacts:
hybrid_k_val.csv; primary k=60 was pre-registered, val shows k=20 marginally
higher — k was NOT re-tuned on test).

## Outcome classification (spec §20 menu): Outcome 1 + elements of Outcome 4/5 context

### C-1 (SUPPORTED, scope: PH3B-DS-AG tier S + M20D256) — Trained query-dependent gating provides a large, reproducible advantage when views are complementary and query-dependent

- Primary comparison (R@10, TEST, n=61): **gating 0.6448 ± 0.0025** vs
  uniform 0.3885 vs global-best single view 0.4033 vs oracle head 0.9672
  (seeds 42/7/1). Gap recovered = (gating−uniform)/(oracle−uniform) =
  **0.443** (Phase 2B-comparable metric).
- Medium scale replicates (PH3B-COMP-002-D256, 20K docs, dim 256):
  gating 0.6021 ± 0.0161 vs uniform 0.3580 vs gbest 0.4333; gap recovered
  ≈ 0.47; candidate recall 0.9580.
- Reproduction run (PH3B-COMP-003): gating seed-42 0.6492 vs 0.6426
  (+0.0066, within the documented OS-seeded pool wobble); BM25 identical.
- Mechanism evidence (not assumed, §7/§8): mean gating weight concentrates
  on each query type's DEFINING head (title→0.568, body→0.638,
  mixed→0.645) vs the alternatives; per-query oracle-head agreement is
  0.55–0.62 — the gate selects the useful view on average but is per-query
  noisy (trained on 280 queries, 1536-dim hashed inputs). Gating beats
  both static baselines on EVERY query type.
- Scope/limitations: relevance = exact cosine in the query type's defining
  head space (documented, deterministic, hashed, leakage-filtered,
  cross-validated; 401/450 queries kept). Test n=61 — small; seed std
  0.0025 but CI not computed. AG News titles/descriptions are short news
  text; findings may not transfer to long-document fields.

### C-2 (SUPPORTED, scope: PH3B-DS-AG) — Uniform multi-view fusion does NOT beat the best single view here

0.3885 vs 0.4033 (within noise of each other; both far below gating).
Replicates the Phase 3 image-corpus direction (uniform ≈ single when no
complementarity exploitation) — but here heads ARE complementary, so the
correct reading is: uniform weighting cannot exploit complementarity;
query-dependent selection can.

### C-3 (SUPPORTED, scope: PH3B-DS-AG) — BM25 is a verified-correct but weaker channel against this semantic ground truth; hybrid does not close the gap

- BM25 R@10 0.1787 (NDCG 0.2245, MRR 0.4533); hybrid RRF(BM25+full) k=60
  0.3131; engine hybrid channel 0.3197 — all below uniform/gbest/gating.
- BM25 channel itself verified INDEPENDENT (PH3B-BM25-001,
  bm25_verify.json): term containment in top-10 = 0.9902; rare-token
  known-answer recovery = 0.7556 (45 probes); overlap@10 with a
  same-parameters independent BM25 under a deliberately different
  tokenizer = 0.5852. The weakness is a property of the benchmark's
  semantic-space relevance definition (§21 separation), NOT an engine bug.
- Val k-sensitivity (never tuned on test): k=20 0.350 / k=60 0.330 /
  k=120 0.325 — pre-registered k=60 retained as primary.
- Honest reading (Outcome 4/5 context): on workloads where relevance is
  lexical, BM25/hybrid may dominate; this benchmark cannot measure that.
  What it DOES establish: AttentionDB's learned multi-view retrieval adds
  value beyond conventional hybrid search ON THIS WORKLOAD (0.645 vs
  0.313).

### C-4 (SUPPORTED, measurement) — Candidate generation, not gating, bounds the ceiling

Pool-union candidate recall 0.9839 (S) / 0.9580 (D256). Oracle-arm ≈
0.967 ≈ candidate recall → after pooling, fusion ranking loses little;
the gate's remaining headroom (0.645→0.967) splits into view-selection
error (agreement 0.55–0.62) and candidate misses. §9 discipline applied:
final-ranking failures were never attributed to gating when the document
was absent from pools.

### C-5 (SUPPORTED, ops/§14) — Memory remains the scaling wall for text at dim 512

30K and 20K docs at dim 512 OOM (exit 137; runs preserved); the dim-256
20K probe completes at 967 MB peak (engine multiplier ≈ 14–16× raw vector
bytes, text+BM25). One additional operational finding: SIGKILLed runs
leak tmpfs engine dirs (632 MB found) — engine dirs now default to disk;
the tmpfs-era failed run is preserved and labeled.

### OPEN QUESTIONS

- Does per-query view selection sharpen with more training queries (test
  n=61, train n=280 here)? A larger query set would separate gate
  capacity from data scarcity. [requires bigger medium tier — memory-bound]
- Does the advantage persist when relevance is defined by human/external
  judgments rather than a defined embedding space? (Dataset family
  limitation; recorded in datasets-text.md.)
- Head-count scaling on text (1/2/4/8 fields) — pending.
- Latency comparison S(tmpfs engine dir) vs D256(disk) is confounded by
  the backend; stage-level gating-forward cost is backend-independent.
