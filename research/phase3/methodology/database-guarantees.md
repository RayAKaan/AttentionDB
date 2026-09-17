# AttentionDB — Database Guarantees Contract (Phase 3D §2)

Date: 2026-09-17 · Status basis: commit at HEAD (Phase 3D start tree) + Phase 3D experiments.
Author: Rayyan Kaan.

This document states what the database **actually implements and what the Phase 3D
experiments verified**, from reading the code (`core/src/engine.rs`,
`core/src/transaction.rs`, `core/src/checker.rs`, `storage/src/{wal,compaction,document_store,sstable,catalog}.rs`,
`query/src/filter.rs`, `core/src/backup.rs`) and running the PH3D families. It never claims
stronger semantics than the API exposes. Verdict vocabulary: **VERIFIED** (implementation +
experiment that exercises the failure mode), **PARTIAL**, **NOT VERIFIED**, **UNSUPPORTED**,
**BLOCKED** (possible but not implemented), **OPEN** (unknown).

## 1. Durability modes (`AttentionEngine::open_dir(db_dir, Durability)`)

| Mode | WAL flush discipline | Protects against | Verified by |
|---|---|---|---|
| `Sync` | fsync per WAL append | machine crash | PH3D-CRASH-003 (process-crash points all ALL_ACKED) |
| `GroupCommit` | page-cache flush per append (no fsync) | process crash | PH3D-CRASH-001 (all 7 points ALL_ACKED) |
| `Async` | userspace buffer only | nothing beyond clean operation | PH3D-CRASH-002 (proper-prefix recovery; committed txn lost after commit returned — `COMMITTED_NOT_DURABLE_ASYNC`) |

VERIFIED (process-crash axis). The machine-crash axis (power loss) is exercised only insofar
as fsync is actually called; no VM-kill/power-cut harness was run → machine-crash durability
is **PARTIAL** (code path verified, hardware-fault axis untested).

## 2. Operation contract matrix

| Operation | Intended guarantee | Implemented? | Tested? | Evidence |
|---|---|---|---|---|
| `insert_document` | durable append of a new uuid→numeric mapping + record | yes | VERIFIED | PH3D-STATE-001..003 (1000-op model suite, 0 failures); crash family ALL_ACKED |
| `update_document` | retire old vector id, mint new numeric, replace fields/vectors; **uuid identity preserved**; old representation never returned | yes | VERIFIED | PH3D-STATE-* (model upsert/update lifecycle v1→vN; restart+compact ⇒ latest content, `updated_content_survives`); PH3D-COMPACT-001 |
| `delete_document` | retire numeric id, remove from BM25, tombstone in store; **immediate invisibility**; index purge is lazy (retired vectors await purge → checker WARNING `INDEX_RETIRED_VECTOR`) | yes | VERIFIED (tombstones documented as lazy) | PH3D-STATE-* `delete_immediate`, `final_dead_not_retrievable`; PH3D-COMPACT-001 `deleted_stay_deleted` after tombstone GC |
| uuid→numeric mapping | `register` idempotent per uuid; `retire` unmaps; re-insert of a retired uuid mints a NEW numeric (deterministic across restart/compact via WAL/idmap snapshot) | yes | VERIFIED | PH3D-STATE-* restart/compact equivalence with delete+reinsert cycles |
| `attend` (unfiltered top-K) | deterministic candidate generation + gating + fusion; only live docs | yes | VERIFIED | PH3D-STATE-* `inv_live_only`, `query_determinism` (≈120–134 queries/run) |
| `attend_filtered` | **soundness**: a doc not matching `FilterExpr` can never be returned; **completeness NOT guaranteed** — ranking runs over the ANN candidate pool with ≤3 adaptive expansion rounds, so recall is bounded by candidate coverage | yes | VERIFIED (soundness + determinism; recall measured 0.967–1.0 on 60–120-doc sets) | PH3D-FILTER-001 (14/14), PH3D-STATE-* `inv_filter_leak`, `filter_zero_match` |
| Transactions: `begin/record/commit/rollback` | staged Insert/Delete ops; commit = mutation-gate → pre-validate ALL ops → WAL BeginTxn→TxnOp\*→CommitTxn → idempotent apply; **all-or-nothing**; rollback discards; **no update op**; single-collection per txn; **no isolation levels** (mutations serialize on the mutation gate; no snapshot isolation claim) | yes (as scoped) | VERIFIED (scoped: atomicity incl. crash + injected failure; isolation NOT TESTED beyond gate serialization) | PH3D-TX-001 7/7 incl. `txn_commit_failure_atomic` (wrong-dim insert aborts whole commit), `txn_delete_missing_noop`; CRASH family `during_commit_txn` (never partial: 10/10 or 0/10, 21/21 points) |
| `flush_wal` | move userspace buffer into page cache (Async→GroupCommit-equivalent visibility) | yes | VERIFIED | PH3D-CRASH-002 `after_flush`/`after_checkpoint` ALL_ACKED vs `after_acks` PREFIX |
| `checkpoint` | flush memtable → SST + idmap snapshot + manifest (crash-safe two-phase install); rotate WAL; trim segments fully covered by `checkpoint_seq` | yes | VERIFIED | PH3D-STATE-* restart equivalence after checkpoint; CRASH family `after_checkpoint` |
| WAL replay on open | full replay from `checkpoint_seq`; strict seq continuity; torn (incomplete) tail truncated with WARNING; checksum/format-corrupt frame → **refuse to open** (no silent recovery) | yes | VERIFIED | PH3D-WALCORRUPT-001: truncate→prefix recovery (clean); garbage tail + flipped byte→ERROR_ON_OPEN; duplicate/gapped segment name→ERROR_ON_OPEN |
| `compaction::compact_all(dir)` | full merge of all SSTs + tombstone GC; **dir-level, offline** (engine must be closed) | yes (after PH3D fix, see §5) | VERIFIED | PH3D-COMPACT-001 (state equivalence, updated content survives, deleted stay deleted) |
| automatic compaction | `document_store.flush` self-compacts when ≥4 SST files (partial merges **retain tombstones** — anti-resurrection INV-12; full GC only via `compact_all`) | yes | PARTIAL (trigger observed; threshold not swept) | code + PH3D-COMPACT-002 probe (3 files → no auto-merge, correct per `min_files_to_compact=4`) |
| `copy_database_dir` / `restore_backup` | directory copy + manifest; restore reproduces the captured state; **no online backup** — copy is uncoordinated with the mutation gate | yes | VERIFIED for quiescent; live probes DOCUMENTED | PH3D-BACKUP-001: restore == backup state, no post-backup mutations, checker-clean, manifest present; copy during ACTIVE writers observed checker-clean in a single sample (not a guarantee — quiescent backup is the supported path) |
| `check_engine` / `check_db_dir` | structural + recovery-equivalent consistency gate; ERROR-severity issues = fail; WARNING (retired vectors awaiting purge) = documented non-fatal | yes | VERIFIED | PH3D families run it as mandatory gate (model: initial/after-each-compact/final; txn/compact/backup/concur/crash-verify) |

## 3. Documented semantics (not obvious from names — verified behavior)

1. **Identity rule**: a document's uuid is assigned at insert and never changes
   (`update_document` clones the old record). The engine-internal `record.version` increments
   per update; the caller-visible content (`fields["version"]` in PH3D docs) is whatever the
   caller writes. PH3D harness pins uuid = `uuid_for(idx, 1)` per logical doc.
2. **No update op in transactions**: `TxnOp = Insert | Delete` only. Updates inside a txn are
   expressed as delete+insert (or are simply absent — both documented; PH3D uses insert-only +
   delete txns).
3. **Delete-if-present**: `TxnOp::Delete` of an absent uuid commits successfully as a no-op
   (PH3D-TX-001 `txn_delete_missing_noop`).
4. **Lazy index purge**: deletes retire vector ids immediately (never returned) but the HNSW
   graph retains them until purge; the checker emits `INDEX_RETIRED_VECTOR` WARNINGs. This is
   the documented tombstone mechanism, not a bug.
5. **Filter completeness is best-effort**: `attend_filtered` never leaks non-matching docs
   (soundness), but which matching docs are found depends on ANN candidate coverage and the
   ≤3-round expansion loop. Measured recall 0.967–1.0 on PH3D-FILTER-001 shapes.
6. **Readers vs writers**: queries do not block on the mutation gate except at stage
   boundaries; a single concurrent writer measurably inflates reader latency at reader
   parallelism 1–2 (PH3D-CONC-002 r1w1 p50 826 µs vs r4w1 65 µs). No isolation-level contract
   is claimed or tested.
7. **Checkpoint/compaction interaction**: checkpoint rotates + trims WAL segments it fully
   covers; a directory right after checkpoint legitimately has an empty active WAL segment
   while `checkpoint_seq > 0` (see §5, checker fix).
8. **Collection membership & global uuid identity (discovered in PH3D-INTEGRATION-001)**:
   document identity (uuid → numeric id) is GLOBAL across collections; a collection is a
   membership tag on the record. Inserting a uuid that already exists — even under a
   different collection — replaces the stored record and re-members it (the new collection's
   tag is added; the record body is replaced). Callers using multiple collections MUST
   namespace their logical ids per collection. Verified: with namespaced ids, no
   cross-collection leakage occurs across live, restart, compaction, and backup/restore
   (PH3D-INTEGRATION-001 16/16).
9. **Graceful shutdown**: `close()` checkpoints before returning; insert, update, delete and
   upsert all survive close→reopen (PH3D-INTEGRATION-001).
10. **Concurrent mutation visibility model (PH3D-CONC-003)**: writers serialize on the
    mutation gate; per-key last-write-wins; records are never torn (each record reflects
    exactly one writer op — verified by a fields-version==fields-num probe over 100
    contended keys × 150 rounds); no ordering guarantee ACROSS keys; no linearizability
    claim.

Spec-mapping and granularity notes: `ph3d-spec-deviations.md` (crash-point mapping §15,
txn-failure injection granularity, §21 UNSUPPORTED combos, §34 fuzz scope, §41–42
numbering).

## 4. WAL architecture (Phase 3D §12)

- **What is logged**: every mutation — InsertDocument, DeleteDocument (tombstone), txn
  BeginTxn / TxnOp\* / CommitTxn markers — with monotonically increasing sequence numbers.
  Checkpoints are NOT in the WAL; they install SST + idmap + manifest out of band.
- **One authoritative sequence?** Yes. The segmented WAL under `<db_dir>/WAL/` is the single
  authoritative mutation sequence. Segments are named by the seq of their first record
  (20-digit zero-padded); replay validates strict +1 continuity across and within segments.
- **Replay**: `Wal::replay(min_seq_exclusive)` decodes frames (version-gated format, CRC32);
  recovery replays seq > `checkpoint_seq` into the engine.
- **Duplicate replay**: apply paths are idempotent (`apply_insert`, retire-if-present, txn
  markers dedup via txn buffers); replay of the same segment range twice yields the same state
  (exercised implicitly by every restart-equivalence check in PH3D-STATE-*).
- **Torn records**: an incomplete final frame is truncated at open with a WARNING
  (`WAL_TORN_TAIL`); the intact prefix recovers (PH3D-WALCORRUPT-001 `truncate_torn`: 23-doc
  prefix clean).
- **Corrupt records**: a complete frame failing version/CRC → replay errors → open refuses
  (`ERROR_ON_OPEN`; `garbage_tail`, `flip_byte`, `duplicate_segment_gap` modes). No silent
  false recovery; no data fabrication.
- **Checkpoint/compaction interaction**: after checkpoint, covered segments are removed; the
  active segment starts at `checkpoint_seq + 1`. Compaction itself never rewrites the WAL.

## 5. Phase 3D product changes (each names the invariant it preserves/closes)

1. **`checker::check_db_dir` WAL gap invariant (fix)** — the old check
   `last_seq < checkpoint_seq ⇒ WAL_SEQ_INVALID` false-errored on the normal post-checkpoint
   trimmed state (empty active WAL). New rule: an empty WAL is legitimate iff segments exist;
   over-trim is `first_seq > checkpoint_seq + 1` (uncovered gap) or a contiguous WAL ending
   before `checkpoint_seq`. Invariant preserved: the checker still fails on any state where
   recovery could lose acknowledged mutations. Regression coverage via PH3D runs (every
   post-checkpoint gate) + storage tests.
2. **`compaction::compact_all` path resolution (fix)** — `compact_all(db_dir)` scanned the DB
   ROOT for `.sst` files, but SSTables live in `db_dir/sst/`; offline compaction therefore
   always no-opped (`Ok(None)`). Fix: resolve `Catalog::sst_dir(db_dir)`. Invariant closed:
   "full compaction is reachable through the public dir-level API". Regression test
   `compact_all_resolves_sst_subdir_from_db_root` (storage tests). Behavior verified by
   PH3D-COMPACT-001 (merge + tombstone GC + equivalence).
3. **Harness-only fixes (not product)**: deterministic uuid stamping in the PH3D harness,
   collection-membership filtering in state export, soundness/recall oracle split for
   filters, durability-aware crash verifier, errors-only checker gate with warnings reported
   separately.

## 6. Known limitations (documented, not hidden)

- **WAL segment deletion before the first checkpoint is structurally undetectable**: the
  catalog stores no WAL high-water mark, so a vandalized (segment-deleted) pre-checkpoint dir
  opens as an empty database with a clean checker (PH3D-WALCORRUPT-001 `delete_segment`).
  Mitigation would require a seq high-water mark in the manifest — out of 3D scope, recorded
  as OPEN.
- **No online backup / no online (engine-open) compaction**: `copy_database_dir` and
  `compact_all` do not coordinate with the mutation gate. Both are supported only against a
  quiescent/closed database; live-invocation behavior is documented (PH3D-COMPACT-002,
  PH3D-BACKUP-001), not certified.
- **Async durability acks**: `commit_transaction` returning Ok under `Durability::Async` does
  not imply the effects survive process death (PH3D-CRASH-002). Atomicity still holds
  (all-or-nothing loss).
- **No isolation levels / no linearizability claims**: mutations serialize on a single gate;
  multi-writer ordering under contention is deterministic but no history-based linearizability
  test was run (§25) — no claim made.
- **No distributed/replication semantics** (standing scope exclusion).
