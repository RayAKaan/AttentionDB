# Concurrency Behavior Validation (Phase 3D)

## Claim
AttentionDB sustains parallel query load and mixed query/mutation load with zero errors and
checker-clean state; its concurrency design point is a single mutation gate serializing
writers against stage boundaries — a property that is measured and documented here, without
any linearizability or isolation claim.

## Method
Thread-based load harness, 4-second windows: (i) a pure-read ladder with 1, 2, 4, 8, 16, 32
reader threads over a 300-doc collection; (ii) a mixed matrix (r1w1, r2w1, r4w1, r4w2,
r8w2, r16w4) where each writer runs a deterministic insert → update → delete → flush cycle
with a per-writer JSONL op log. Post-window: live-doc export, duplicate-id scan, and the
engine+directory consistency gate. Latency percentiles are taken per query inside the
reader threads.

## Results
Pure read: QPS 13.5K (r1) → 21.6K (r2) → ~19–22K plateau (r4–r32); p50 68–93 µs flat;
**p99 tail rises from 128 µs (r1) to 60 ms (r32)** — reported, not smoothed. Mixed: 0 query
errors in every combo; checker-clean after every run; writer-throughput dominated runs
(r1w1 ≈ 1.4K QPS with p50 ≈ 754 µs) expose the gate serialization: one reader cannot hide
writer-critical sections, four can. Retired-vector warnings accumulate under delete-heavy
writers (≈3.5K in 4 s) — the documented lazy tombstone backlog; deletions themselves are
immediately invisible.

## What is not claimed
No linearizability (no history-based test, §25), no serializability, no multi-client
distribution, no write-write contention coverage (writers touch disjoint logical ids).
The op logs are retained precisely so a history checker can be added later.

## Limitations
Single process, thread-level concurrency, 4 s windows; tail behavior beyond 32 readers and
sustained-hours behavior not measured.
