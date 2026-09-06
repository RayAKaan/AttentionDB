# Phase 3 final report (IN PROGRESS — updated as experiment families complete)

## Phase 3B addendum (2026-09-06): complementary multi-view + text/hybrid — central question

**Does trained query-dependent gating help when views are genuinely
complementary and query-dependent? YES on this workload.**

- PH3B-COMP-001 (AG News multi-field, 10K docs, seeds 42/7/1):
  gating **0.6448 ± 0.0025** vs uniform 0.3885 vs global-best single view
  0.4033 vs oracle head 0.9672 → **gap recovered 0.443** (2B-comparable).
  Gating beats both static baselines on EVERY query type (title/body/
  mixed); mean gate weight concentrates on each type's defining head
  (0.568/0.638/0.645) with per-query oracle agreement 0.55–0.62.
- PH3B-COMP-002-D256 (20K docs, dim 256): gating 0.6021 ± 0.0161 vs
  uniform 0.3580 / gbest 0.4333 — advantage replicates at 5× scale.
- PH3B-COMP-002-M30/-M20: 512-dim medium corpora OOM on the 2 GB sandbox
  (preserved; one additional tmpfs-hygiene failure preserved and labeled).
- PH3B-BM25-001: engine BM25 independently verified correct
  (containment 0.9902, rare-token known-answer 0.7556, overlap 0.5852 vs
  independent BM25) — yet scores 0.1787 vs the semantic-space GT;
  hybrid RRF k=60 0.3131 < gating 0.6448. Learned multi-view retrieval
  adds value beyond conventional hybrid search on this workload.
- Harness corrections HC-P3-4 (argpartition tie nondeterminism),
  HC-P3-5 (ALL-row aggregation), HC-P3-6 (diagnostic id mapping) — all
  caught pre-recording or in diagnostics; no canonical number invalidated.
- Verdict vs §20 outcome menu: **Outcome 1** (gating significantly
  improves over uniform AND global-best) + honest Outcome-4/5 context
  (BM25/hybrid trail on this semantic GT; lexical-relevance workloads
  untested).
- Decision gate (§22): trained gating demonstrates a reproducible gain on
  complementary real-world data at two scales. This supports treating
  trained gating as the validated contribution — contingent on the
  remaining systems/reliability families (§12–§26) and the §40
  classification, which stay open.

See findings/complementary-analysis.md for the full provisional analysis.

## Answered so far

**Retrieval (§6, PH3-DS-FM-S)**
- Trained gating vs single-vector ANN: NO improvement on this corpus
  (0.9853 vs 0.9860 R@10) — the gate collapses to the dominant full-image
  view (w(full)=1.0). This is correct gate behavior on dominated heads,
  not a defect. [F-P3-1]
- Trained gating vs uniform multi-head: +35.6pp (0.9853 vs 0.6293) —
  uniform fusion is actively harmful with dominated views.
- Candidate recall 0.9992; exact ceiling 1.0 (harness anchor).
- Seed robustness: gating 0.9840/0.9860/0.9860 (seeds 42/7/1).

**Systems (§9/§16/§19, partial)**
- Latency p50 (2-CPU sandbox, warm, release): single-head 766 µs; 5-head
  + fusion 2544 µs; gating forward 36 µs (≈1.4% of pipeline); exact
  reference 3567 µs. [F-P3-3]
- Memory: 10K docs → 571 MB peak; 30K and 60K → OOM (exit 137). The §19
  wall is the dominant open limitation. [F-P3-4]

## Pending experiment families (defined, not yet run)

scaling brackets below 30K; head-scaling serial/parallel (§10); candidate
budgets (§11); filtered retrieval (§12); mutations (§13); restart/recovery
(§14); crash recovery (§15); transactions (§16); checker integration (§17);
compaction (§18); storage footprint (§20); concurrency (§21); determinism
(§22); corruption handling (§23); backup/restore (§24); API workflow (§25);
observability (§26); text-dataset family with BM25/hybrid baselines (§5 D/E).

## Architecture classification (§40) — PREMATURE to assign

Evidence so far covers one real corpus at 10K scale plus the memory wall.
Provisional reading: **B — research prototype** (retrieval works and is
harness-anchored; systems limits — memory scaling — are unresolved). The
classification will be finalized only after the reliability families run.
