# Phase 3C final report (§30)

Scope statement: every number below is measured on a 2 GB RAM / 2-CPU
cgroup sandbox (Debian 13, rustc 1.98.1, release profile), disk-backed
engine dirs, Async durability, engine-default HNSW (M=16, efC=400,
efS=64, store_vectors=true). "Maximum" always means "maximum reproduced
under this hardware/configuration" (§12).

## 1. What causes the memory wall?
The document-insertion loop (inline HNSW construction): ≈64 MB RSS per
1,000 docs ≈ 10.9× the raw vector rate at 3×512 (PH3C-MEM-001 checkpoints).
Engine init, collection creation, flush, BM25, and the gating model are
negligible. Vectors are stored at least twice in RAM (HNSW store_vectors +
document-store records); a ~128 MB build transient sits above steady
state; only ~82 MB is released at engine drop (allocator retention).
On-disk footprint is just 1.4× raw — this is an in-RAM phenomenon.

## 2. How does memory scale with documents?
≈ linearly at ~10.9× raw: 507/860/1182 MB at 5K/10K/15K (3×512); wall
between 15K and 20K (OOM, last peak 1410 MB). [Fig 1]

## 3. How does memory scale with heads?
≈ linearly: 351/596/860/1092 MB at 1/2/3/4 heads (10K×512, ~245–260
MB/head); 8×10K OOMs, 8×5K completes at 960 MB. At dim 256: 242/385/532/
656/1211 MB for 1/2/3/4/8. [Fig 2]

## 4. How does memory scale with dimension?
≈ linearly over a fixed ~130–160 MB base: 361/532/860 MB at 128/256/512
(3h/10K). The multiplier falls with scale (24.7×→14.7×) as the base
dilutes. [Fig 3]

## 5. Build-time vs steady-state?
Peak 859.5 vs steady 649.7 MB at the reference config — ~128 MB build
transient (~15%); post-drop RSS retains most engine allocation
(731→650 MB). Build workspace is NOT returned to the OS.

## 6. Is vector duplication occurring?
YES (bounded estimate): vectors live in both HNSW (store_vectors=true)
and document-store records; ≥2× raw is directly attributable; the
remaining ~6–8× resists outside-process attribution (open question;
engine-side instrumentation required). No invented component values.

## 7. What is the tmpfs failure mechanism?
Unclean death (SIGKILL / exit without destructors) never runs the Drop
guards that remove engine dirs → leak ∝ corpus (632 MB found in 3B); on
tmpfs the leak is RAM. Verified with clean/exit/SIGKILL modes
(PH3C-MEM-003). Mitigation in place: disk-backed engine dirs; leaked
temps are never counted as DB size.

## 8. What head count gives what retrieval quality? (5K ladder, K=100, seeds 42/7/1)
| heads | gbest | uniform | gating | oracle | cand-recall |
|---|---|---|---|---|---|
| 1 | 0.519 | 0.519 | 0.519 | 0.519 | 0.672 |
| 2 | 0.423 | 0.344 | 0.658±.021 | 0.691 | 0.859 |
| 4 | 0.522 | 0.378 | 0.715±.022 | 0.759 | 0.877 |
| 8 | 0.522 | 0.386 | 0.709±.023 | 0.758 | 0.886 |
(3 heads @10K reference: gating 0.6448±0.0025.) [Fig 4]

## 9. Latency cost per head? (serial p50, K=100)
1.7 / 4.2 / 6.7 / 22.1 ms at 1/2/4/8 heads — ~linear then super-linear
at 8 (fusion over 8×200 candidates). [Fig 5]

## 10. Does parallelism remain beneficial?
YES at ≥2 heads (2 workers: 1.66× at 4 heads, 2.02× at 8, p50) and NO at
1 head (24% worse). Bounded by the 2-CPU sandbox.

## 11. What candidate budget reaches a quality plateau?
Immediately: gating R@10 varies ≤0.004 across K=10..200 while candidate
recall climbs 0.787→0.877. Small budgets (25–50/head) are measured-safe;
formal val-based default deferred to the systems phase. [Fig 7]

## 12. Candidate generation or gating — dominant remaining limitation?
Both, separated (§17): 34.2% of GT misses never entered any pool;
65.9% were pooled but ranked out; 18.9% of all misses were in the
defining head's top-10 (view-selection-attributable). Oracle headroom is
capped by pool coverage — fixing either alone cannot reach the ceiling.

## 13. SUPPORTED
M-1..M-5 (memory model, linearity, duplication bounds, budget envelope,
tmpfs mechanism); HS-1..HS-4 (head ladder, tradeoff, parallelism, uniform
never wins); CB-1..CB-3 (budget plateau, decomposition, headroom caps).

## 14. NOT SUPPORTED
- "More heads always improve quality": 8 heads ≈ 4 heads (within noise),
  at higher memory + super-linear latency.
- "Memory explodes nonlinearly with scale": linear in docs/heads/dims at
  the tested envelope.
- Implicit "8-head config is better": measured tradeoff says otherwise.

## 15. OPEN QUESTIONS
- Exact in-engine component attribution (needs engine instrumentation).
- Whether dedup + workspace release recovers ~3–4× memory (the designated
  §25 experiment: baseline → change → identical benchmark → measured).
- 4→8-head saturation cause (redundancy vs gate capacity at 280 training
  queries).
- 8-head quality at 10K (OOM; preserved run).

## Stop-condition status (per spec)
Retrieval mechanism validated (3B) ✔; memory model quantified ✔;
head-count tradeoff quantified ✔; candidate-budget tradeoff quantified ✔;
remaining engineering bottlenecks listed (memory duplication/retention,
candidate recall, view-selection training data size) ✔. Next: database
correctness/reliability families (filtering, concurrent mutations,
restart/recovery, transactions, compaction, concurrency, backup/restore).

## Audit addendum (2026-09-06 — spec re-audit after sandbox snapshot rollback)

- **Snapshot rollback incident**: the original Phase 3C commit did not
  survive the sandbox snapshot (HEAD rolled back to the Phase 3B commit);
  ALL working-tree artifacts survived (26 run dirs, 9 canonical CSVs,
  figures 1–9, findings, paper sections; both consistency checkers PASS;
  Phase 2 untouched). Re-recorded as a single recovery commit; nothing
  was re-run, no number changed.
- **§11 completed**: SIGTERM added to the leak matrix (exit 143, dir
  leaked on tmpfs — identical to SIGKILL; clean drop removes). §11 matrix
  now {clean, exit(137), SIGTERM, SIGKILL} × {disk, tmpfs}.
- **§1 completed**: PH3C-HEAD-002 registered as an analysis run
  (serial-vs-parallel decomposition + CPU utilization over
  PH3C-HEAD-001-* latency artifacts; speedups 0.81×/2.07×/1.53×/2.02× at
  1/2/4/8 heads).
- **§19/§18 documented deviations**: per-seed head-selection agreement
  std NOT computed (agreement recorded for the canonical seed-42 model
  per run); per-budget per-type slices not retained (per-type at K=100
  only). Both noted in head-scaling-findings.md — not silently dropped.
- **§7 scope**: the 25K/30K corpus points were not attempted (wall
  bracketed 15K OK / 20K OOM; larger points cannot complete on this
  sandbox — consistent with the §7 "where feasible" rule).
