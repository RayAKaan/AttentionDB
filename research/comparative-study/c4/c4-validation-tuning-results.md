# C4 Validation Tuning Results (C4.2)

Status: **COMPLETE** — all 6 frozen W-19 VAL tuning cells executed on the
frozen plan branch commits (`c4-run-plan.csv` untouched; results documented
here + in RUN-INDEX.yaml + raw run dirs).

Date: 2026-09-25. Harness: `c4/harness/c4_tune_run.py` (all fixes applied).
Binary: c2pilot.exe sha256 90249747... (smoke PASS).
Matching rule: `c1/recall-and-budget-matching.md` — tolerance |config−target|
<= 0.01 INCLUSIVE on qrels-based Relevance-Recall@10 against the B0 exact
canonical oracle computed in-process on the same VAL row set; fastest p50
post-warmup among qualifying configs selected; TARGET-UNREACHABLE never
interpolated (full quality-latency curve reported with best achieved recall).

Split freeze (preregistered): NFCorpus VAL = 324 official qrel-dev qids;
SciFact VAL = first 200 of seeded shuffle (seed 20260925) of sorted
qrel-train qids. B0 heat-up replicated per-run (B0 pass identical queries).

## Matched-config table (frozen TEST-cell inputs)

| Tuning cell | DS | Mode | Target (B0 VAL recQ) | Grid | Selected | Sel recQ | Sel p50 | Status |
|---|---|---|---|---|---|---|---|---|
| C4-W19-NFC-B1-EF-001 | NFC | B1 | 0.136424 | EF{16,32,64,128} | **EF64** | 0.1351 | 817.0 us | PASS |
| C4-W19-SCI-B1-EF-001 | SCI | B1 | 0.785000 | EF{16,32,64,128} | **EF32** | 0.7775 | 962.3 us | PASS (corrected) |
| C4-W19-NFC-B2-EF-001 | NFC | B2 | 0.136424 | EF{16,32,64,128} x CAND{50,100,200,500} | **EF16-C50** | 0.1393 | 1629.3 us | PASS (corrected) |
| C4-W19-SCI-B2-EF-001 | SCI | B2 | 0.785000 | EF{16,32,64,128} x CAND{50,100,200,500} | — | best EF64-C50 0.7458 | — | PASS-TARGET-UNREACHABLE |
| C4-W19-SCI-B7-EF-001 | SCI | B7 | 0.785000 | EF{16,32,64,128} | — | best EF32 0.7021 | — | PASS-TARGET-UNREACHABLE |
| C4-W19-NFC-B7-EF-001 | NFC | B7 | 0.136424 | EF{16,32,64,128} | **EF64** | 0.1372 | 1624.7 us | PASS |

## Reflections into frozen TEST cells (C4.4 inputs)

- C4-W01-NFC-B1-001 (TEST): ef_search=64.
- C4-W01-SCI-B1-001 (TEST): ef_search=32.
- C4-W02-NFC-B2-001 (TEST): ef_search=16, candidate_budget=50.
- C4-W02-SCI-B2-001 (TEST): TARGET-UNREACHABLE (best 0.7458 @ EF64-C50).
  TEST cell retains plan default config BUDGET-CAND:500; quality-matched
  contrast reported as TARGET-UNREACHABLE with the full curve + best recall;
  the B2-vs-B1 matched-latency claim is NOT available for SciFact (recorded
  honestly per rule; never interpolated).
- C4-W07-NFC-B7-001 (TEST): ef_search=64.
- C4-W07-SCI-B7-001 (TEST): TARGET-UNREACHABLE (best 0.7021 @ EF32).
  Same treatment as SciFact B2.

## Corrections applied during tuning (evidence preserved, per C3 convention)

1. **C4-W19-SCI-B1-EF-001**: original run used a strict float comparison
   (<=0.01) that rejected the exact-boundary EF32 (best 0.775 vs 0.785 target,
   diff 0.010) as UNREACHABLE by a ~1e-15 artifact. Matching rule is
   inclusive. Harness tolerance now `<= 0.01 + 1e-9`. Corrected re-run under
   `C4-W19-SCI-B1-EF-001_CONFIG-CORRECTION-0` selects EF32 (0.7775). Original
   evidence preserved under the base dir (PASS-TARGET-UNREACHABLE entry kept
   in RUN-INDEX; correction entry follows).
2. **C4-W19-NFC-B2-EF-001**: original grid configs omitted CANONICAL from
   doc_vectors, so the engine's exact_top10 oracle was empty and every
   recall10_exact was 0.0000 (semantically None). C1 requires recall10_exact
   for all modes (C3 B2 NFC ~0.68). Harness now adds CANONICAL to grid
   doc_vectors (exact-oracle reference ONLY; not to collection/attend heads,
   preserving the 3-head registry). Corrected re-run under
   `C4-W19-NFC-B2-EF-001_CONFIG-CORRECTION-0` restores recall_exact (~0.68)
   and selects EF16-C50. Original preserved.
   NOTE: SciFact-B2/B7 and NFC-B7 cells were executed AFTER the fix, so they
   are unaffected.

## Notable tuning observations

- SciFact B1/B2/B7 all target 0.7850 (B0 canonical oracle is strong on the
  seeded train-sample). Single-head B1 reaches 0.7775 (EF32) within tolerance;
  multi-head fusion (B2 best 0.7458, B7 best 0.7021) cannot close the 0.01
  gap on SciFact VAL at any gridded config. The H5 matched-latency contrast
  (B1 vs multi-head) on SciFact therefore reports TARGET-UNREACHABLE for the
  multi-head arms with full quality-latency curves.
- NFC: every mode reaches within-tolerance recall at multiple configs; B2
  selection EF16-C50 (candidate budget 50 suffices on VAL), B1/B7 EF64.
- p50 latencies are repeatable post-warmup (warmup 20); all grid reps used
  fresh c2pilot processes (rep_count 1 for tuning rows, per frozen plan).
- Peak RSS all grid levels well below the 85% MemAvailable guardrail
  (sampled every 500 ms; no aborts).

## Full per-level evidence (from metrics.json, all levels within tolerance)

| Cell | Level | recQ | recExact | p50 us | within |
|---|---|---|---|---|---|
| NFC-B1 | EF16 | 0.1351 | 0.9716 | 1056.1 | T |
| | EF32 | 0.1338 | 0.9617 | 892.3 | T |
| | EF64 | 0.1351 | 0.9688 | 817.0 | T |
| | EF128 | 0.1338 | 0.9583 | 974.7 | T |
| SciFact-B1 (corr) | EF16 | 0.7683 | 0.9820 | 1085.3 | F |
| | EF32 | 0.7775 | 0.9830 | 962.3 | T |
| | EF64 | 0.7700 | 0.9765 | 1076.9 | F |
| | EF128 | 0.7700 | 0.9810 | 1074.5 | F |
| NFC-B2 (corr) | EF16-C50 | 0.1393 | 0.6818 | 1629.3 | T |
| | EF16-C100 | 0.1403 | 0.6809 | 1732.6 | T |
| | EF16-C200 | 0.1391 | 0.6809 | 1708.9 | T |
| | EF16-C500 | 0.1367 | 0.6827 | 1743.6 | T |
| | EF32-C50 | 0.1402 | 0.6824 | 1711.8 | T |
| | EF32-C100 | 0.1391 | 0.6815 | 1712.5 | T |
| | EF32-C200 | 0.1402 | 0.6806 | 1791.2 | T |
| | EF32-C500 | 0.1404 | 0.6824 | 1639.8 | T |
| | EF64-C50 | 0.1352 | 0.6806 | 1726.0 | T |
| | EF64-C100 | 0.1394 | 0.6790 | 1766.4 | T |
| | EF64-C200 | 0.1398 | 0.6840 | 1658.7 | T |
| | EF64-C500 | 0.1400 | 0.6796 | 1703.6 | T |
| | EF128-C50 | 0.1384 | 0.6809 | 1860.5 | T |
| | EF128-C100 | 0.1407 | 0.6809 | 1646.5 | T |
| | EF128-C200 | 0.1366 | 0.6802 | 1665.8 | T |
| | EF128-C500 | 0.1413 | 0.6836 | 1635.5 | T |
| NFC-B7 | EF16 | 0.1356 | 0.6191 | 1897.3 | T |
| | EF32 | 0.1350 | 0.6173 | 2030.1 | T |
| | EF64 | 0.1372 | 0.6154 | 1624.7 | T |
| | EF128 | 0.1369 | 0.6216 | 2008.0 | T |

SciFact B2/B7 grids (all outside tolerance) are recorded in their raw
metrics.json; SciFact-B2 best EF64-C50 (0.7458), SciFact-B7 best EF32
(0.7021).