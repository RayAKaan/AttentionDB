# PH3E-SOAK-010 — final disposition (platform loss, chain complete)

PH3E-SOAK-010 (E8i INVALIDATED harness run) lost its top-level evidence in the
first platform snapshot-cap loss (TRIAGE entry 1, 2026-09-22). Its remaining
directory contents were regenerable backup-* generation dirs, removed under
TRIAGE entry 2; the emptied directory shell itself was then dropped by a later
workspace snapshot. This note recreates the directory so the run remains
present in raw. The INVALIDATED record is fully preserved in:
- raw/experiment-index.json (status INVALIDATED, metrics ops=143477, elapsed 601s)
- results/e8-invalidated.csv (documented-loss row)
- phase3e-e8-final-report.md §26 and the restoration ADDENDUM
- TRIAGE-2026-09-22.md
The run is not re-creatable (its harness defects were fixed); no part of any
official result depends on its artifacts.
