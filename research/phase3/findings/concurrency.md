# PH3D Finding — Concurrency

Status: **SUPPORTED for parallel readers and mixed r/w stability; NO ordering/isolation claim** · Runs: PH3D-CONC-001 (read ladder), PH3D-CONC-002 (mixed matrix)

## Findings
1. **Parallel readers scale to a saturation point, then plateau.** Pure-read QPS:
   13.5K (r1) → 21.6K (r2) → ~19–22K for r4–r32; p50 stays ~68–93 µs across the ladder.
   Beyond ~2–4 readers, throughput is bounded by something other than reader parallelism
   (per-query work dominates); zero query errors and zero checker errors at every rung.
2. **p99 tail grows with reader count** (pure read): 128 µs (r1) → 60 ms (r32). Honest
   tail behavior under many concurrent readers — measured, reported, not hidden.
3. **Mixed read/write is stable**: 6 combos (r1w1 → r16w4) with a deterministic
   insert→update→delete→flush writer cycle: **0 errors everywhere**, post-window state
   checker-clean in all runs, live-doc counts tracked (e.g. r4w1: 1132 docs, 73K queries).
4. **Writer blocks readers at low reader parallelism (mutation gate serialization)**:
   r1w1 p50 ≈ 754–826 µs vs r4w1 p50 ≈ 65–93 µs — a single reader queues behind writer
   mutations; more readers overlap the gate windows. Documented as the actual mechanism;
   no isolation claim is derived from it.
5. **Retired-vector purge backlog under sustained writes**: checker WARNINGS
   (`INDEX_RETIRED_VECTOR`) accumulate with delete-heavy writers (e.g. 3.4–3.7K warnings in
   4 s mixed runs). These are the documented lazy tombstones — deletions are immediately
   invisible; the index entries await purge. Reported as a measurable backlog, not an error.

## What is NOT claimed
No linearizability or serializability (§25): no history-based ordering test was run.
The deterministic per-writer op logs (`operation-log.jsonl`) are recorded to enable such a
test later, but the claim is **NOT VERIFIED** in PH3D.

## Evidence
`results/concurrency.csv`, tables 7, figures 11–12, raw run dirs incl. per-combo op logs.

## Limitations
4 s windows; single process (threads, not multi-client); writers use distinct logical ids
(no write-write contention on the same key — that is coverage for a future family).
