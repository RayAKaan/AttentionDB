| case | mode | sst before | sst after | tombs removed | classification | match |
|---|---|---|---|---|---|---|
| c0-offline-control | sync | 3 | 1 | 0 | COMPACTION_VERIFIED | MATCH |
| multigen-version-resolution | sync | 3 | 1 | 0 | COMPACTION_VERIFIED | MATCH |
| tombstone-gc-basic | sync | 2 | 1 | 1 | COMPACTION_VERIFIED | MATCH |
| tombstone-deep-reinsert | sync | - | - | 1 | COMPACTION_VERIFIED | MATCH |
| delete-reinsert-delete-x3 | sync | - | - | 3 | COMPACTION_VERIFIED | MATCH |
| partial-merge-retains-tombstones | storage-api | 5 | - | 0 | COMPACTION_VERIFIED | MATCH |
| repeated-compaction-x4 | sync | - | - | 0 | COMPACTION_VERIFIED | MATCH |
| readers-during-compaction | group | 2 | 1 | 10 | COMPACTION_VERIFIED | MATCH |
| writer-during-compaction | group | 2 | 1 | 5 | COMPACTION_VERIFIED | MATCH |
| multiwriter-during-compaction | group | - | - | 0 | COMPACTION_VERIFIED | MATCH |
| ckpt-then-compact | sync | - | - | 0 | COMPACTION_VERIFIED | MATCH |
| compact-then-ckpt | sync | - | - | 0 | COMPACTION_VERIFIED | MATCH |
| wal-rotation-during-compaction | group | - | - | 0 | COMPACTION_VERIFIED | MATCH |
| bkp-after-compact | sync | - | - | 0 | COMPACTION_VERIFIED | MATCH |
| bkp-before-compact | sync | - | - | 0 | COMPACTION_VERIFIED | MATCH |
| compact-backup-restore-restart-compact | sync | - | - | 0 | COMPACTION_VERIFIED | MATCH |
| manifest-fallback-after-compact | sync | - | - | 0 | COMPACTION_VERIFIED | MATCH |
| partial-artifact-tmp | sync | - | - | 0 | COMPACTION_VERIFIED | MATCH |
| partial-artifact-garbage-sst | sync | - | - | 0 | REFUSED_BY_POLICY | MATCH |
| auto-scheduler-observation | sync | 3 | 1 | 0 | COMPACTION_VERIFIED | MATCH |
| collections-isolation-compaction | group | - | - | 2 | COMPACTION_VERIFIED | MATCH |

Crash windows (fresh-process recovery):
| window | observed state | match |
|---|---|---|
| compact_before_merge | MATCH | MATCH |
| compact_after_output | MATCH | MATCH |
| compact_after_install | MATCH | MATCH |
| compact_after_cleanup | MATCH | MATCH |

Source: results/e5-compaction.csv + results/e5-crash-windows.csv (generated from raw/runs/PH3E-COMPACT-004 — no manual transcription).
