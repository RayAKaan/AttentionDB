# PH3D Finding — Backup/Restore

Status: **SUPPORTED for the quiescent path; online backup UNSUPPORTED (documented)** · Run: PH3D-BACKUP-001

## Finding
- **Capture → restore equivalence**: populate 150 docs → flush → checkpoint →
  `copy_database_dir` → restore into a fresh dir → mutate the ORIGINAL (insert doc 150,
  delete doc 1) → the restored database equals the backup state exactly (150 docs, doc 1
  present, doc 150 absent), with unique store ids, a clean dir+engine checker, and a present
  manifest (6/6 checks).
- **Live-idle copy** (engine open, no writers): restore opened checker-clean (181 docs) —
  the copy is consistent when the database is open but quiescent.
- **Copy under ACTIVE writers** (2 writer threads during the copy; 665 docs written during
  the window): the restored snapshot happened to open checker-clean in this single sample.
  This is **documented behavior, not a guarantee**: `copy_database_dir` does not coordinate
  with the mutation gate, no online backup exists, and the supported path is a quiescent
  database. Stated in the contract and the production-readiness matrix as UNSUPPORTED with
  quiescence requirement.

## Evidence
`results/backup-restore.csv`, raw `PH3D-BACKUP-001/results.csv` (incl. the
`live_writer_backup_documented` row with the writer count and observed consistency).

Audit closure adds PH3D-BACKUP-002 (6/6): per-file size + sha256 inventory
(`inventory.csv`), independent integrity verification at two levels (source==backup and
backup==restored, 0 mismatches; `backup-meta.json` is the copy manifest the tool itself
writes), state equality and a clean checker on the restored copy.

## Limitations
Single sample for the active-writer probe (its purpose is documentation of the failure
mode, not statistical confidence). No incremental/PITR backup exists; backup cost scales
with full directory size.
