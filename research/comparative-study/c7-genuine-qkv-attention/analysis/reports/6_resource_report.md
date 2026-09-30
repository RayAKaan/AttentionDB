# C7 Resource Report

## 1. Compute budget (wall clock, this machine)

| Workload | Duration |
|---|---|
| EFPROBE (2 datasets, 1 rep, 20 qids) | ≈ 1 min each |
| TUNE gate scan (SciFact, 11 × 368 qids) | ≈ 36 min |
| TUNE gate scan (NFCorpus, 11 × 328 qids) | ≈ 51 min |
| TUNE trains E/F (2 datasets, 5 epochs) | ≈ 1–2 min each |
| SMOKE (2 datasets, 20 qids, +fresh-process rerun) | ≈ 2 min each |
| TEST SciFact (300 qids × 6 arms × 5 reps) | ≈ 73 min |
| TEST NFCorpus (323 qids × 6 arms × 5 reps) | ≈ 78 min |
| **Total pilot CPU (execution)** | **≈ 4.5 h** |
| Build (probe, release, offline) | ≈ 2–3 min incremental |

## 2. Memory
- Peak child RSS ≈ 1.2–1.4 GB during TEST (attention arms dominate: 6 arms × 500 candidate attention matrices in flight per query).
- Promise: single shared engine per process (6 arms evaluated in-process) bounds peak vs. per-arm processes.
- Guardrail: 85% of preflight MemAvailable; see 4_environment_report for the two triggered aborts and fix.

## 3. Disk
- `C7-SHARED-{SCI,NFC}/`: shared vectors ~20 MB each; multi-probe/smoke JSONs small; TEST multi JSON ≈ 390 MB per rep × 5 reps per dataset (~3.9 GB across both datasets, streamed to disk to keep RAM flat).
- Quarantine of pre-fix evidence: `raw/.C7-OSSEED-EVIDENCE/` holds the invalidated OS-seeded runs (~same magnitude).

## 4. Latency profile (TEST, p50 over queries, mean of reps)

| Arm | SciFact p50 | NFC p50 |
|---|---|---|
| A (attention OFF, 1 head) | 3.4 ms | 3.0 ms |
| B (attention OFF, 3 head union) | 12.8 ms | 10.7 ms |
| C / D / E / F (attention ON) | ≈ 0.83–0.85 s | ≈ 0.82–0.84 s |

Attention ON costs ≈ 800–850 ms/query-arm: a per-candidate 3-head QKV attention scoring stage over up to 500 candidates (plus softmax + gate fusion). This is the dominant, inherent cost of candidate-level attention in this budget regime. Projected/measured TEST total ≈ 75 min/dataset is dominated by it.