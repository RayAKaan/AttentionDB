# Backup and Restore Validation (Phase 3D)

## Claim
`copy_database_dir` + `restore_backup` produce an exact, checker-clean copy of a quiescent
database, and the restored database contains none of the original's post-backup mutations.
There is **no online backup**: copies are uncoordinated with the mutation gate, the
supported path is a quiescent database, and live-copy behavior is documented (not
certified).

## Method
Populate 150 documents → flush → checkpoint → capture → restore into a fresh directory →
mutate the original (insert doc 150, delete doc 1) → export and compare the restored
logical state to the backup state (equality, uniqueness, mutation isolation), run the
engine+directory consistency gate, verify the manifest. Two additional probes document
live behavior: copy with the engine open but idle; copy with two writer threads active
during the capture.

## Results (PH3D-BACKUP-001, 6/6)
Restored == backup state (150 docs; doc 1 present, doc 150 absent); unique store ids;
clean checker; manifest present. Live-idle copy: restored checker-clean (181 docs).
Active-writer copy (665 docs written during capture): the single sample restored
checker-clean — recorded as an observation with the quiescence requirement unchanged.

## Limitations
The active-writer probe is documentation of the failure mode, not a distribution over
outcomes. No incremental backup, no point-in-time recovery, no backup integrity signing
beyond the manifest; restore cost is a full directory copy.
