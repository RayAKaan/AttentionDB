| family | run | ops | elapsed s | txns (commit/rollback) | ckpt | compact | backup | restore | restart | verify (fail) | reader checks (viol) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E8a | PH3E-SOAK-001 | 30000 | 50 | 0/0 | 6 | 3 | 1 | 1 | 1 | 12 (0) | 92892 (0) |
| E8b | PH3E-SOAK-002 | 200000 | 260 | 0/0 | 8 | 4 | 1 | 1 | 2 | 23 (0) | 0 (0) |
| E8c | PH3E-SOAK-003 | 120000 | 555 | 0/0 | 0 | 0 | 0 | 0 | 0 | 2 (0) | 778248 (0) |
| E8d | PH3E-SOAK-004 | 58764 | 106 | 13531/3749 | 7 | 2 | 0 | 0 | 1 | 10 (0) | 0 (0) |
| E8e | PH3E-SOAK-005 | 12000 | 22 | 0/0 | 12 | 12 | 12 | 12 | 12 | 36 (0) | 0 (0) |
| E8f | PH3E-SOAK-006 | 120000 | 172 | 0/0 | 60 | 0 | 0 | 0 | 76 | 23 (0) | 0 (0) |
| E8g | PH3E-SOAK-007 | 1143 | 6 | 50/0 | 25 | 25 | 25 | 25 | 25 | 50 (0) | 0 (0) |
| E8h | PH3E-SOAK-008 | 150000 | 190 | 0/0 | 15 | 5 | 0 | 0 | 0 | 2 (0) | 303076 (0) |
| E8i | PH3E-SOAK-011 | 150002 | 446 | 30000/0 | 50 | 16 | 10 | 0 | 1 | 32 (0) | 540716 (0) |

Memory classification (S16): see results/e8-memory.csv — every family measured linear-with-ops RSS growth over its tested budget (corr E8a 0.983, E8b 0.880, E8c 0.971, E8d 0.888, E8e 0.996, E8f 0.985, E8g 1.000, E8h 0.993, E8i 0.992); boundedness beyond tested budgets NOT established; E9 marker.

Invalidated runs (preserved in raw):

| run | family | ops | reason |
|---|---|---|---|
| PH3E-SOAK-009 | E8i | 125492 | PH3E-SOAK-009 — INVALIDATED (harness defect, engine exonerated) |
| PH3E-SOAK-010 | E8i | 143477 | artifacts lost to platform snapshot cap (TRIAGE-2026-09-22.md); INVALIDATED record preserved in registry + E8 report s26 |
