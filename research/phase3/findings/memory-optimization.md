# PH3D Finding — Memory Optimization

Status: **NOT ATTEMPTED (optional family not run)** · Runs: none (PH3D-MEM-OPT-001 reserved)

## Statement
Phase 3D did not attempt a memory optimization. The optional PH3D-MEM-OPT-001 family
(dedup first, then workspace release — one change at a time, baseline → change → identical
benchmark → measure, per §36–37) was **not executed**; there is therefore **no before/after
memory claim, no figure, and no scaling claim** for Phase 3D. Memory behavior baselines
remain those of Phase 3C (duplication ≥2× raw, engine drop releases ~82MB of ~128MB
transient, leak matrix).

## Rationale
The phase's mandate was correctness/durability/recovery/transactions/concurrency/backup
validation. Two product fixes were made (checker WAL-gap invariant; compact_all sst/ path
resolution) — both are correctness fixes with regression tests, not memory changes, and
neither alters allocation behavior measurably (no new allocations on hot paths).

## Rule going forward
If PH3D-MEM-OPT-001 is attempted later: dedup FIRST, workspace-release SECOND, never
combined; full regression acceptance (§37) after each change; §38 perf smoke
(R@10/NDCG@10/p50/95/99/QPS/RSS) after any fix; any serious correctness failure stops
benchmarking until fixed (§51).
