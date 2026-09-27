# Qdrant — Readiness Report

Study `comparative-study-001` · §14 (B6 feasibility only; NOT a benchmark, NOT an
AttentionDB adapter).

## Status

**SMOKE-PASS (feasibility: IMPLEMENTED/VALIDATED at API level).**
Not BENCHMARK-READY by this evidence alone.

## Evidence — C2-SMOKE-QDRANT-006 (terminal, PASS)

- binary_version `qdrant 1.12.4`; startup_healthy true (startup 0.51 s)
- create_collection / upsert_100 / search_5 all HTTP 200
- search_returns_exact_self_top1 true
- named-vector collection `c2named` (2×384-d): create/upsert/search 200,
  named_search_self_top1 true, body recorded in metrics
- guarded sampler attached: server_peak_rss_kb 59,604; 4 samples; no abort;
  server_exit_code -9 (sampler SIGKILL after clean shutdown of the test)
- superseded ledger (registry): -001 PASS (no sampler), -002 INVALID-STARTUP
  (harness fake-thread defect), -003 INVALID-STARTUP (rest() on 400), -004/-005
  FAILED (named-search payload shape)

## B6 note

The named-vector collection success confirms the **feasibility of the proposed
B6 Qdrant configuration** (named vectors per head + prefetch fusion mapping).
This is a configuration proposal with no benchmark semantics; it is not an
AttentionDB adapter and is never reported as one.

## Limitations

- No index-type/quantization tuning, no latency/recall sweep — those belong to a
  later phase, gated by the ladder discipline.
- Envelope fit confirmed at 100 vectors under sampler; larger rungs require the
  ≥20%-headroom rule.

## Verdict

Feasible on this host. Ready-to-progress only under a later-phase benchmark plan
that re-states the discipline; C2 asserts nothing about quality or speed.