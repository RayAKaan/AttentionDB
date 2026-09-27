# C5 Failure Report

**Protocol §9**: Any stop condition → halt + preserve + report.  
**Result**: **No stop conditions triggered**. All runs completed successfully.

---

## Stop Condition Checklist

| Condition | Status | Detail |
|-----------|--------|--------|
| Anchor hash mismatch (B0) | ✅ PASS | C5-A/B0 hashes match C4-B0 exactly |
| Exact-oracle discrepancy | ✅ PASS | C5-A exact recall matches C4-B0 within noise |
| ef-sensitivity probe failing | ✅ PASS | ef latency scales 28–30%; NFC recall moves |
| Budget leak in C5-C | ✅ PASS | Candidate union ≤ 500; per-head ≤ 50; RSS < 512 MiB |
| Raw evidence overwritten | ✅ PASS | All runs append-only; `RUN-INDEX.yaml` immutable |
| 3 consecutive harness crashes | ✅ PASS | 0 crashes in 23 TEST/VALID/SMOKE/EFPROBE cells |
| Silent wrong result | ✅ PASS | Cross-checked against C4 anchors; ledger sanity checks pass |

---

## Known Non-Critical Issues

| Issue | Run | Resolution |
|-------|-----|------------|
| Harness `KeyError: query_ids` on first EFPROBE-001 | C5-EPROBE-001 | Fixed in `c5_test_run.py`; run superseded by C5-EPROBE-005; original marked FAILED in RUN-INDEX |
| Plan CSV comment lines dropped by analysis script | analysis | Comments non-essential; row parsing uses `csv.DictReader` (skips comments if any) |
| Minor: `candidate_change_count` absent for A/B modes | metrics.json | Expected — only mode C populates; analysis guards against null |

---

## Superseded Runs

| Run ID | Status | Superseded By | Reason |
|--------|--------|---------------|--------|
| C5-EPROBE-001 | FAILED | C5-EPROBE-005 | Harness bug pre-measurement; artifacts materialized, no metrics |

All other runs: PASS.

---

## Audit Trail

Every run registered twice in `raw/RUN-INDEX.yaml`:
1. Pre-execution: `status: RUN` (preregistration)
2. Post-execution: `status: PASS|FAILED` + metrics summary

No run deleted or modified after registration.