# Phase 3 final report (IN PROGRESS — updated as experiment families complete)

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
