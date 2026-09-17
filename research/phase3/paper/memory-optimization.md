# Memory Optimization (Phase 3D Status)

## Claim
**None.** Phase 3D ran no memory optimization experiment; the optional PH3D-MEM-OPT-001
family was not executed. There is no before/after memory result, no optimized-scaling
figure, and no performance-after-optimization smoke in this phase — by design, since §36/37
require one-change-at-a-time with full regression acceptance, and the phase's mandate was
database validation.

## Standing baselines (Phase 3C, unchanged)
- Duplication ≥2× raw document bytes (HNSW `store_vectors` + document store); disk ≈1.4×.
- Engine drop releases only ~82 MB of the ≈128 MB transient build peak.
- Envelope @512-dim: ≈30K head-docs (1h/30K 941MB · 2h/20K 1152MB · 4h/10K 1092MB · 8h/5K 960MB).
- Any unhandled termination leaks the engine directory; clean Drop removes it.

## Two product fixes in PH3D — neither is a memory change
1. `check_db_dir` WAL-gap invariant (correctness of the consistency gate).
2. `compact_all` resolves `db_dir/sst` (offline compaction was an unreachable no-op).
A side observation relevant to future memory work: `compact_all`'s tombstone GC can now
actually be exercised offline, which is the precondition for measuring space reclamation.

## Forward protocol (unchanged)
Dedup first; workspace release second; never combined. After each change: identical
benchmark re-run + full regression acceptance (§37) + §38 perf smoke (R@10, NDCG@10, p50/
95/99, QPS, RSS). Any serious correctness failure stops optimization work until fixed (§51).
