# C7 Failure Report

Log of defects encountered, their root causes, and dispositions. All were resolved; none invalidated final evidence (the OS-seeded evidence determined before the fix was quarantined to `raw/.C7-OSSEED-EVIDENCE/` and the pipeline re-run with the deterministic binary).

## 1. hnsw_rs OS-entropy layer seeding → cross-process nondeterminism (CRITICAL, gate #6)
- Symptom: SMOKE determinism gate failed; on a fresh-process rerun, top-10 ranks changed on ~1/3 of queries and recall@10 even flipped 1.0→0.0 (e.g. arm A q0). candidate_count stable, so it was not a config mismatch.
- Root cause: hnsw_rs builds per-head HNSW graphs with `StdRng::from_os_rng()`; every process builds a different graph, so union membership flips near top-k boundaries.
- Fix: vendored hnsw_rs + fixed layer seed (probe `[patch.crates-io]`). Verified: two separate processes now produce bit-identical ledgers (candidate_count, recall, nDCG, top-10, full union).
- Cost: full C7 pipeline (EFPROBE, TUNE, SMOKE) re-run + TEST; prior evidence quarantined.

## 2. Inert ef knob (gate #3)
- Symptom: EFPROBE EF16 vs EF128 produced identical candidate counts (~470) and similar metrics because hnsw_rs clamps its beam to `max(ef, k)` and per-head k=500.
- Fix: `RetrievalConfig.search_k` cap applied per head; EFPROBE arms pass `search_k` = ef. Post-fix: EF16 ~30 / EF128 ~234 candidates with strictly higher recall at EF128.

## 3. Probe per-arm latency was cumulative
- Symptom: later arms reported inflated latency (EF128 ≈ 2× EF16 purely from measurement order).
- Root cause: `lat_us = t.elapsed()` used the query-level clock, accumulating prior arms' time.
- Fix: per-arm `Instant::now()` timer.

## 4. Harness parent RSS creep → guardrail aborts (execution interruption)
- Symptom: NFC TEST aborted twice at rep5 (`child RSS >= 85% preflight MemAvailable`), wasting ~1 h each.
- Root cause: harness accumulated all five 390 MB multi-JSONs in `all_reps` in the parent Python process (~+2 GB); standalone 20-qid runs were unaffected.
- Fix: per-rep union-check + per-arm artifact split inside the rep loop; `del multi`. Post-fix run completed cleanly.

## 5. Harness dry-run side effects
- TUNE `--dry-run` (and earlier PROBE/SMOKE) created evidence dirs / register entries. Fixed: dry-run short-circuits before materialize/registration for every track.

## 6. Verify bugs
- `fail()` returned `None` when called without a message, silently nulling the whole gate (`checks` list); fixed to return a `_Sink`.
- `gate_determinism` read the wrong dict key (`b[key]` vs `p[label]`), under-reporting rerun diffs; fixed.
- TUNE cells wrote `gate_choice.json`/`model.json` but the run-index gate demanded `metrics.json`; gate now accepts per-track artifacts.

## 7. Harness timeout
- First TUNE NFC gate scan exceeded the 2400 s probe timeout; raised to 7200 s (NFC gate scan runs 11 arms × 328 qids with attention, ≈ 50 min).

## 8. Original EFPROBE evidence invalid (superseded)
- Initial EFPROBE registers predated fixes #2/#3; superseded and quarantined, then re-run with same run ids.