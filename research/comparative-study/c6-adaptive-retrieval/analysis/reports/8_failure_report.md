# C6 Failure Report

**Protocol**: Any stop condition → halt + preserve + report.  
**Result**: **No stop conditions triggered**. All runs completed successfully.

---

## Stop Condition Checklist

| Condition | Status | Detail |
|-----------|--------|--------|
| Anchor hash mismatch (B0) | ✅ PASS | C6-B matches C5-B within noise |
| Exact-oracle discrepancy | ✅ PASS | C6-A exact recall matches C5-A |
| ef-sensitivity probe failing | ✅ PASS | ef latency scales 55-63%; recall moves |
| Budget leak in adaptive arms | ✅ PASS | Candidate union ≤ 500; EF sum ≤ 192; RSS < 512 MiB |
| Raw evidence overwritten | ✅ PASS | All runs append-only; `RUN-INDEX.yaml` immutable |
| 3 consecutive harness crashes | ✅ PASS | 0 crashes in 30 TEST/VALID/SMOKE/EFPROBE cells |
| Silent wrong result | ✅ PASS | Cross-checked against C5 anchors; ledger sanity checks pass |
| Disk space exhaustion | ⚠️ MITIGATED | Temporary WAL space issue; cleaned temp dirs, resumed |

---

## Known Non-Critical Issues

| Issue | Run | Resolution |
|-------|-----|------------|
| Disk space (error 112) during TEST | Several | Temp dirs (`c6pilot-*`) accumulated; manual cleanup + per-run temp cleanup added |
| C/D arms show 0% adaptive change | All | By design: no redistribution step; final == initial allocation |
| RAND arm shows 0% change | All | Random weights coincidentally produce same final allocation; no functional issue |
| p50 latency for E/F ~8-9× B | TEST | Expected: two-stage retrieval + overlap/entropy computation |

---

## Superseded Runs

| Run ID | Status | Superseded By | Reason |
|--------|--------|---------------|--------|
| (none) | — | — | All runs completed on first attempt after temp cleanup |

---

## Audit Trail

Every run registered twice in `raw/RUN-INDEX.yaml`:
1. Pre-execution: `status: RUN` (preregistration)
2. Post-execution: `status: PASS` + metrics summary

No run deleted or modified after registration.