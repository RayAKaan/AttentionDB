# Phase 3E — E9 Final Report
## Memory Instrumentation, Root-Cause Analysis & Optimization

Author: Rayyan Kaan (RayAKaan). Baseline: commit `3e65d50f8fd1a81cda339d4608260d6be98ba708`
with the E8 working tree (tree sha16 `1b524cf3e39dfc6b`); E9 engine change is
exactly one mechanism — INV-E9-HYGIENE (checkpoint-time index hygiene) — plus
read-only census accessors and the feature-free e9 harness.

## 1. Executive Summary

E8 left one open question: RSS grows ~linearly with progress operations
(r 0.869–1.000, slope ~0.4–1.2 MB/1k ops) — measured but unexplained, explicitly
NOT classified as a leak. E9 reproduced the signal independently (PH3E-MEM-001:
r=0.992, slope 660 vs 698 KB/1k ops), separated RSS from PSS and anonymous from
file-backed memory, attributed every candidate component with a read-only
census, ran 16 control experiments, and root-caused the growth: **dead-version
retention in the per-collection HNSW structures** (graph nodes that hnsw_rs
cannot delete, exact-rerank vector store purged only at startup) plus the
INV-6-required retired-id sets — all application-owned anonymous heap (PSS≈RSS,
uordblks-tracking before rebuilds), none of it allocator or page-cache
artifacts. The fix — INV-E9-HYGIENE: purge retired ids and deterministically
rebuild indexes (the sealed recovery-path operation) at checkpoints under a
dead-entry/insert-budget policy — cut the churn workload's peak RSS 58.3%
(205,368→85,600 KB) and its growth slope 8.5× (1,125→132 KB/1k ops), restored
retrieval recall that dead nodes had degraded, left doc-proportional workloads
correctly unchanged, and passed the full 329-test suite plus count-identical
E8 regression families h/d/i/f under new IDs. Root cause confidence: HIGH.
Decision tree outcome: **B (identifiable retained structures, minimally
bounded) with C (doc-proportional amplification measurable and bounded) for
the growth component.** E9 COMPLETE; E10 NOT STARTED.

## 2. Baseline

git HEAD `3e65d50f…` (branch main), E8 working-tree state uncommitted as during
E8; suite 328/0 at entry (329 after the E9 regression test), clippy clean
(rustc 1.98.1 — toolchain reinstalled after the sandbox reset; deviation D30).
Environment: 2-vCPU sandbox, 1.9 GiB RAM, ext2/ext3, kernel 6.1.158+. No
engine changes before the reproduction run (M1 discipline).

## 3. E8 Memory Signal Reproduction

PH3E-MEM-001 re-ran the exact E8 family-h harness: ops=150,000, ckpt=15,
compact=5, verify=2, failures=0 — counts identical to PH3E-SOAK-016 — with RSS
signal r=0.992, first 17,384 KB / peak 168,364 KB / slope 660 KB/1k ops vs the
E8 official's 698 KB/1k ops (within 5%). M1 SATISFIED: the E8 observation
stands as the starting hypothesis and is real, durable, and
environment-stable.

## 4. Environment

Recorded per §4 of the E9 charter: commit, dirty-tree sha16, toolchain
1.98.1, kernel, 2 CPUs, 1.9 GiB RAM, ext2/ext3 filesystem, durability modes
per run (D31), WAL segment defaults (2,048 B where the walrot experiment
overrides).

## 5. Measurement Methodology

Inline 2-second telemetry: VmRSS/VmSize/RssAnon/RssFile (/proc/self/status),
Pss/Pss_Anon/Pss_File (/proc/self/smaps_rollup), fd and thread counts,
glibc mallinfo2, db-dir walk (SST bytes/files, db bytes), and the read-only
engine component census (mapper lens, DocumentStore memtable/flushed-cache/
SST-readers/block-cache, txn staging, per-collection vector-store/retired/
BM25 lens). Terminology per spec §7; every claimed number is MEASURED,
every estimate is labeled; mallinfo columns are flagged unreliable under
rebuild workloads (D32); WAL columns NA (D35).

## 6. RSS/PSS Analysis

PSS tracks RSS within ~1.5% everywhere sampled (e.g. optimized end: RSS
85,600 / PSS 83,881 KB); anonymous memory dominates (optimized end anon
80,760 vs file 4,840 KB). The growth is resident ANONYMOUS memory — heap-like
— never file-backed/page-cache. Pre-optimization, heap in-use (uordblks)
tracked RSS (MEM-002: 174.8 MB in-use at 205 MB RSS), ruling out allocator
retention as the growth mechanism; post-restart, ~99 MB of freed heap stays
allocator-retained (M4 separation, MEM-015).

## 7. Component Memory Accounting

Census at the 150k-op churn end (601 live docs): vector_store_len 112,991
(≈188× live), mapper_retired 83,164 (×2 sets, INV-6), collection.retired_ids
83,164, flushed_records 2,599 (≈ live — bounded), BM25 2,600 lens/5,198
postings (bounded), block_cache ≤1 (bounded 50,000 cap), memtable 0, staged
txns 0. hnsw_rs graph nodes are not census-visible (UNKNOWN label; bounded
only by inserts-since-rebuild). Accounting ladder (MEM-014): ~9.5 KB per
document all-in at 10k docs (record + vector + index + read cache), stable
across restart after cache repopulation. Estimated decomposition of the
201 MB churn growth: HNSW graph+store dead versions ~70–80%, retired sets
~8%, live-ish caches ~5%, allocator/arena overhead ~10% (ESTIMATED, census-
anchored).

## 8. Control Experiments

Sixteen controls (results/e9-controls.csv): read-only plateaus (+2.7 MB/40k
ops — REFUTES query/cache growth); fixed-cardinality updates grow linearly
(240 MB — INDICTS per-mutation retention); insert/delete churn at 1k keys
grows (63 MB — same); no-compaction vs compacted shapes match (compaction
exonerated); compaction-heavy shows no retained compaction memory; 2 KiB WAL
rotation exonerated; 150 checkpoints flat; 25 backups flat; transactions
return staged state to zero (txn_staged_end=0); query-path scratch transient
(+3 MB/45k ops); mapper churn shows INV-6 persistence across restart (by
design, ~small); idle decay: none (state is referenced, not transient);
restart-reset: dead vector entries purged at startup but RSS stays elevated
(allocator retention) and growth resumes — process-local retention, not
persistent corruption.

## 9. Root-Cause Tree

RSS growth (churn)
├── Live application memory — SUPPORTED (the mechanism)
│   ├── HNSW graph nodes per version-insert, never removable — SUPPORTED (dominant)
│   ├── HNSW exact-rerank store dead entries (+_arc_refs double-store) — SUPPORTED (purged only at startup pre-E9)
│   ├── IdMapper.retired + Collection.retired_ids — SUPPORTED (INV-6-required, by design, minor slope)
│   ├── DocumentStore.flushed_records — REFUTED as growth driver (bounded at fixed cardinality)
│   ├── BM25 postings/lengths — REFUTED (remove() retains out deletions)
│   ├── transaction staging — REFUTED (returns to 0)
│   └── query scratch — REFUTED (transient)
├── Allocator retention — SUPPORTED as post-restart effect only (~99 MB ford), REFUTED as the growth mechanism
├── File-backed/page-cache residency — REFUTED (anon dominates)
├── Temporary peak memory (compaction/backup) — REFUTED (flat traces)
└── Measurement artifact — REFUTED (two independent instruments + reproduction)

## 10. Root-Cause Findings

H1 HNSW dead-version retention: SUPPORTED, confidence HIGH (census 112,991 vs
601 live; update-minted duplicates 110,000 vs 10,000; restart purge + rebuild
removes exactly this). H2 INV-6 retired sets: SUPPORTED (HIGH, by-design
persistence; ~small slope, retained for correctness). H3 allocator retention:
SUPPORTED (HIGH) as a restart-time RSS floor effect only. H4 flushed_records:
REFUTED (HIGH). H5 WAL/BM25/txn/backup/query: REFUTED (HIGH, five controls).
H6 page cache: REFUTED (HIGH). NEW finding: retrieval-recall degradation at
≥95% dead-node ratio (nondeterministic exact top-k misses), restored by
hygiene rebuilds (D36) — this strengthens S18.

## 11. Memory Budget

RSS(churn, at 150k ops) ≈ base process (~13–28 MB) + HNSW graph+store dead
versions (MEASURED lens × ESTIMATED ~0.35–1 KB/entry; dominant) + retired id
sets (MEASURED 83,164 ×2; ESTIMATED ~50–100 B/entry) + live records + read
caches (MEASURED, bounded) + allocator arenas (MEASURED RSS−anon gap, small
pre-restart). Doc-proportional floor measured at ~9.5 KB/doc all-in (10k-doc
rung). Optimized steady state at the same workload: 85.6 MB with slope
132 KB/1k ops.

## 12. Optimization Candidates

Ranked per the spec's order: (1) unbounded retention — HNSW dead versions
[CHOSEN]; (2) duplication — hnsw store_vectors/_arc_refs double-store and
DocumentStore record caching [DOCUMENTED as future work per §41 — removal
would touch retrieval architecture; not forced into E9]; (3) long-lived
buffers [none found unbounded]; (4) capacity retention [HashMap/Vec slack —
subsumed by (1)]; (5) caches [already bounded 50k]; (6) temporary peaks
[measured small]; (7) allocator [rejected — D32 shows its accounting, and
swapping allocators for a prettier graph is forbidden].

## 13. Implemented Optimizations

**INV-E9-HYGIENE** (engine.rs, one mechanism): at every checkpoint — under
the existing mutation gate — (a) purge retired ids from every head's
exact-rerank store (the startup purge, now also incremental) and (b) if
pre-purge dead entries dominate (vstore > 1.5× mapped-live AND > 20,000) OR
index insertions since the last rebuild exceed max(20,000, 4× live-mapped),
run `rebuild_all_indexes()` — byte-for-byte the recovery-path operation every
open already performs (E1–E8 sealed), so retrieval semantics are preserved by
construction. Counter resets on rebuild. O1-only (purge without rebuild) was
measured insufficient (PH3E-MEM-018..021, preserved) because graph nodes
dominate.

## 14. Before/After Results

Paired, same workload/seed/mode (results/e9-before-after.csv,
figures/e9-before-after.svg): churn repro 205,368→85,600 KB peak (58.3%),
slope 1,125→132 KB/1k ops (8.5×); restart-reset 165,632→105,496 KB (36.3%);
update churn at 10k docs 240,448→202,856 KB (15.6% — the 95 MB doc load
floor dominates; the churn increment collapsed); growing 80k docs
494,412→496,928 KB (−0.5%, correct: no dead nodes exist in an insert-only
workload — hygiene does not touch legitimate scaling).

## 15. Correctness Validation

Full workspace suite 329/0 including the new
`regression_e9_index_hygiene` (dead entries accumulate → checkpoint →
vstore == mapped-live → exact-match retrieval restored → checker clean →
fresh-process recovery intact). E1–E7 sealed suites byte-identical chains
green. Regression soaks under sync with new IDs, count-identical metrics:
family h (PH3E-MEM-026: 15/5/2/0), family d (027: 17,280 txns 13,531c/3,749r,
10/0), family i (028: 30,000 txns, verify 32/0, reader 0 violations),
family f (PH3E-SOAK-017: 76 restarts, R1–R7, async A/B, 23/0).

## 16. Performance Guardrails

Same-workload elapsed: repro 88→85 s (faster — purge shrinks rerank scans);
update churn 121→123 s (+1.6%); growing 87→93 s (+7% — rebuild pauses);
restart-reset 46→54 s (+17% — rebuild at checkpoints; bounded amortized
cost). Compaction/backup/checkpoint durations flat in controls. No metric
collapsed; both sides of every trade reported (§50 honored).

## 17. Resource Stability

Post-optimization: fds 8, threads 3 (flat); SST counts bounded by compaction
policy (max observed 3 sampled / ≤41 pre-compaction in compactheavy);
no stall files; watchdog unused; no new file leaks (TRIAGE documents the
opposite — evidence retention discipline).

## 18. E1–E8 Regression Results

E1–E7: sealed in-workspace suites green (329/0, includes crash-window,
durability, transaction byte-identical, concurrency chains). E8: four fresh
sync soak families under new IDs (PH3E-MEM-026/027/028, PH3E-SOAK-017) — all
count-identical to their sealed originals with zero verification failures and
zero reader violations.

## 19. Bugs Found

No new engine defect codes. One quantified engine-behavior finding: recall
degradation at extreme dead-node ratio (D36) — consequence of the documented
hnsw_rs no-delete limitation; mitigated by INV-E9-HYGIENE and regression-
sealed. One mallinfo2 accounting anomaly under rebuild workloads (D32 —
instrument, not engine). One e9-harness telemetry defect (WAL columns, D35).
One author trim error with deterministic recovery (D33). O1-insufficiency is
recorded as a superseded candidate (D34), not a bug.

## 20. Invalidated Runs

None. All 30 E9 runs (PH3E-MEM-001..029 + PH3E-SOAK-017) completed and are
preserved; O1-partial runs are superseded candidates retained as evidence.
Platform-loss restorations are documented in TRIAGE-2026-09-22.md and the
E8 report addendum.

## 21. Production Contract Amendment

**A9 — Resource Hygiene Guarantee** added (evidence-backed wording): bounded
retained index state at checkpoint boundaries with the measured reductions;
explicit non-claims: leak-freedom is neither established nor claimed; bounded RSS at rest (allocator
retention), document-proportional components excluded. A leak-freedom
claim is NOT made, and none is needed.

## 22. Limitations

Doc-proportional memory (~9.5 KB/doc all-in, including the known ≥2× vector
duplication inside the HNSW wrapper and the read-through record cache) is
bounded-by-construction only in the sense of scaling with legitimate state;
reducing the constant requires architecture work explicitly deferred (§41).
Allocator retention leaves RSS elevated after restarts (~99 MB observed)
until process exit. The insert budget trades periodic O(live·log) rebuild
pauses at checkpoints for bounded retention (+7–17% elapsed on
rebuild-sensitive workloads). Long-duration behavior beyond the tested
budgets (≤150k ops/80k docs) is extrapolation, not measurement.

## 23. Unsupported Claims

Explicitly NOT claimed: leak-freedom; "memory usage bounded" in general;
bounded RSS at rest; power-loss anything; production-readiness from any of
this; no allocator-swap or page-cache claims; no E10-scale memory claims;
serializability/isolation unchanged (untouched); no claim that mallinfo2
figures are exact allocation sizes (ESTIMATED where used, NA where
unreliable).

## 24. Reproducibility

Every experiment is (seed, budget)-deterministic: `dbtest e9run --exp NAME
--out DIR`; telemetry/census CSVs + summary.json per run;
`generate_results_ph3e.py` regenerates results/tables/figures;
`verify_consistency.py` gate 26 FAILS on registry/raw/report drift; the
optimization is a single named invariant (INV-E9-HYGIENE) with a dedicated
regression test.

## 25. Final E9 Verdict

**Outcome B+C.** The E8 linear RSS growth was caused by identifiable,
application-owned retained structures (dead HNSW versions + INV-6 retired
sets); the dominant component is now bounded by a minimal, recovery-path-
preserving checkpoint hygiene with 58.3% peak / 8.5× slope reduction on the
canonical churn workload, recall restored, zero correctness regressions, and
count-identical E8 regression soaks. The remaining growth is legitimate
doc-proportional scaling (measured, bounded by live state) plus a measured
post-restart allocator-retention floor. Confidence HIGH on the root cause
(two independent instruments, sixteen controls, one decisive
restart-reset experiment, and a before/after that moved exactly the
mechanism's prediction). E9 stop condition met; checklist complete.
**E9 COMPLETE; E10 NOT STARTED.**
