# Phase 3E — E5 Specification: Online Compaction, Tombstone Safety & Concurrent Storage Lifecycle

Date: 2026-09-17 · Baseline: cf2be27 (E4 final) · Author: Rayyan Kaan
Run family: `PH3E-COMPACT-003` (PH3D-COMPACT-001/002, PH3D-COMPACTION-001 untouched)

## 1. Implementation audit (read before any change)

Code traced: `storage/src/compaction.rs` (`compact`, `compact_all`, `do_compact`,
`cleanup_merged_files`), `storage/src/document_store.rs` (`flush_memtable`,
`apply_insert`/`apply_delete`, `get`/`get_record`, `open_inner`, `load_sstables`),
`storage/src/sstable.rs` (writer/reader), `storage/src/catalog.rs` (manifest),
`storage/src/crashgate.rs`, `core/src/engine.rs` (`checkpoint_locked`,
`insert_document`, `delete_document`, `backup_to`), `core/src/backup.rs`.

1. **What is compacted?** Document-store SSTables in `<db>/sst/`: key = document UUID
   bytes, value = msgpack Record (tombstones carry the `__TOMBSTONE__` tag). HNSW
   index persistence has a separate `compact_index` (frozen retrieval subsystem —
   OUT of E5 scope; the api tracing `.compact()` is a log formatter, unrelated).
2. **Collection-level or DB-level?** The document store is database-level (one
   shared store, collection membership is a record tag); compaction therefore merges
   across collections. Collection isolation (INV-C5) is at the record/tag level.
3. **How are generations identified?** No manifest SST list: recovery SCANS `sst/`
   (sorted paths) and resolves per-key versions by (entry timestamp, file order).
   Flush files `sstable_<ms>.sst`; compaction outputs `compacted_<ns>.sst`
   ('c' sorts after digits ⇒ after older flush files, before later-named ones —
   consistent with content recency since a merge only consumes OLDER files).
4. **Latest-version selection?** Open: `timestamp` then file index, higher wins
   (`open_inner`). Read path (`get_record`): reverse file order, first hit wins.
   Compaction (`do_compact`): **strictly-greater timestamp replaces — BUG, see §1.13.**
5. **Tombstone representation & GC condition?** A tombstone is a Record with tag
   `__TOMBSTONE__` stored like any record. GC condition (proven in code, to be
   verified experimentally): tombstones are dropped ONLY when the merge covers
   EVERY SST file (`compact_all`, or `compact` when merge set == all files) —
   sound because deletes are by UUID and UUIDs are never reused (retired ids +
   fresh reinsert UUIDs), so once all older versions are in the merge set no older
   version can become visible again. Partial merges retain tombstones (INV-C3).
6. **SST immutability?** Yes — files are written once via tmp→`sync_all`→rename→
   `fsync_dir` and never edited; readers materialize all entries in memory at open
   (`SSTableReader.entries: Vec`), so no reader holds a file handle after open.
7. **Do readers hold references to SSTs?** Only through `document_store`
   RwLock-scoped borrows of the in-memory reader list; the list is swapped under
   the write lock. An external directory reader (backup) is serialized by the
   mutation gate (backup holds it across checkpoint+copy).
8. **When can old SSTs be deleted?** After the merge output is fsynced+renamed and
   the reader list reloaded; `cleanup_merged_files` unlinks inputs (unlink failures
   are warnings — extra files are harmless because open resolves versions by
   content). No reference counting needed: in-memory readers + lock scoping.
9. **Manifest relationship?** The catalog manifest names NO SSTs (checkpoint_seq,
   catalog metas, next_doc_numeric_id; idmap snapshot separate). Compaction
   therefore requires NO manifest update and no new generation (INV-C6 restated:
   recovery must never NEED a deleted file — holds because compaction preserves
   logical state and only supersedes inputs; backup can never interleave).
10. **WAL interaction?** None: compaction does not touch WAL, checkpoint_seq, or
    the high-water mark (INV-C10). Post-compaction state = SSTs contain every acked
    op flushed so far (possibly beyond checkpoint_seq); restart replays WAL >
    checkpoint_seq idempotently (re-apply of acked ops is order-preserving).
11. **Does compaction take the mutation gate today?** YES, by construction: the only
    compaction call sites are inside `flush_memtable`, which runs only inside
    gate-held engine paths (`insert_document`/`delete_document` threshold flushes,
    `checkpoint_locked`). No ungated path reaches compaction.
12. **Can compaction race backup/checkpoint?** No — both are gate-held; all
    compaction is inside gate-held paths. Readers: block on the document_store
    write lock for the flush+merge window (no errors; window is ms-scale).
13. **BUG FOUND (latent INV-C2/INV-C1 violation):** `do_compact` replaces a version
    only on *strictly greater* timestamp, so for two versions of the same key with
    EQUAL entry timestamps the OLDER file wins. `append()` stamps **milliseconds**,
    so two flushes of the same key within one millisecond collide; `open_inner`
    resolves the same tie the OPPOSITE way (higher file index wins). Compaction
    could therefore publish v_old where open would serve v_new (lost update /
    resurrection risk). E5 fix: make `do_compact` tie-break by file index exactly
    like `open_inner` (replace when `ts > cur || (ts == cur && file_idx > cur)`).
14. **Pre-existing verified behavior reused:** Phase-1 `phase1_compaction.rs`
    (TEST 11: no resurrection after full compaction) is the C0 control baseline;
    `open_inner` deletes stray `.tmp` files (partial artifacts never valid state).

## 2. Compaction modes (terminology used consistently)

- **C0 offline:** quiesced DB, then compact. Already demonstrated (Phase-1 tests);
  reproduced as E5 control.
- **C1 coordinated:** mutation gate held for boundary+publish; writers pause for the
  compaction window; readers pause (block, zero errors) for the merge window under
  the store write lock. This is what E5 implements and verifies.
- **C2 fully online:** readers AND writers continue with minimal interference.
  **UNSUPPORTED by the current architecture** (single gate + store write lock); not
  claimed. C1 is never called "fully online".

## 3. Invariants (verified experimentally)

INV-C1 no resurrection · INV-C2 latest version survives · INV-C3 tombstone GC only
under the full-merge proof · INV-C4 id-map/retired-id preservation · INV-C5
collection isolation · INV-C6 recovery never needs a deleted/absent SST (no
manifest→SST edges exist) · INV-C7 atomic publication (tmp→rename; open sees old-or-
new) · INV-C8 reader safety (lock-scoped in-memory readers) · INV-C9 writer safety
(gate serialization; no lost acked write vs reference model) · INV-C10 WAL/
checkpoint_seq/high-watermark untouched by compaction.

## 4. Minimal E5 changes (each names the invariant it serves)

1. **do_compact tie-break fix** (§1.13) — serves INV-C1/C2. + unit test.
2. **`DocumentStore::reload_sstables`** public method — publish step for explicit
   compaction (reader-list swap under the caller's write lock) — serves INV-C7/C8.
3. **`Engine::compact_storage()`** — coordinated full compaction: mutation gate →
   WAL fsync → memtable flush → full merge with tombstone GC → reader-list swap →
   input unlink → stats (files before/after, entries, tombstones removed, bytes,
   duration). Serves C1 classification; gate = the same mechanism E3/E4 verified.
4. **Crash gates** (storage/crashgate.rs): `GATE_COMPACT_BEFORE_MERGE`,
   `GATE_COMPACT_AFTER_OUTPUT` (new SST fsynced+renamed, inputs present, readers
   still old list), `GATE_COMPACT_AFTER_CLEANUP` (inputs unlinked — reader-safe
   because readers materialize entries in memory), `GATE_COMPACT_AFTER_INSTALL`
   (reader list swapped to the output-only set — fully complete). Publication
   order is output → cleanup → install so the published reader list always
   equals the on-disk set. Instrumentation only (same
   contract as E2/E3 gates: env-gated, one atomic load in production). Boundaries
   that cannot be instrumented are reported NOT_INSTRUMENTED (§21 of the prompt).

## 5. Experiment plan (PH3E-COMPACT-003)

Harness `phase3-bench dbtest e5run` (+`e5child` for crash windows), one CSV per
family, independent fsynced ACK-sidecar reference model as in E4:

- E5a control/offline: multi-generation build → compact → model equality + checker.
- E5b multi-generation version resolution (SST1..4 overlapping keys incl. the
  prompt's A/B/C/D/E shape; expected state computed by the sidecar model, not
  hard-coded).
- E5c tombstone GC: insert/flush/delete/flush/compact (+restart); deeper histories;
  delete→reinsert→delete × repeated compactions; partial-merge tombstone retention.
- E5d reinsert/ID mapping: delete → compact → reinsert → compact → restart; retired
  ids excluded; inverted search correct.
- E5e repeated compaction (write/compact ×4) with model+checker after every step.
- E5f concurrent readers + compaction (error count, latency, served-from semantics).
- E5g single writer + compaction (background compaction thread; sidecar model).
- E5h multi-writer (3 writers) + compaction (mutation-ordering semantics, no loss).
- E5i checkpoint×compaction orders (ckpt→compact, compact→ckpt, overlapping via
  auto-flush compaction inside checkpoint).
- E5j WAL rotation (2 KiB segments) × compaction; E1 regression after.
- E5k backup×compaction (writer+backup+compaction; backup restores own snapshot;
  no backup references an unlinked SST — restore verified from backup dir copy).
- E5l crash windows (e5child): 4 instrumented gates × {restart, verify old-or-new
  valid state, model equality, checker}; boundaries not instrumentable =
  NOT_INSTRUMENTED.
- E5m manifest fallback × compaction: corrupt CURRENT / manifest gen after
  compaction → recover to valid generation or refuse; never hybrid.
- E5n partial artifacts: garbage `.tmp` (must be ignored/deleted), garbage `.sst`
  (must refuse loudly — documented policy).
- E5o scheduling: existing auto-compaction documented (trigger: flush with ≥4 SSTs,
  ≤8 per run; no new scheduler — minimal safe default already exists).
- E5p measurements: compaction duration, writer max pause (gate wait), reader
  errors/latency, SST count + logical/physical bytes + tombstones before/after.
- E5q lifecycle: compact → backup → restore → restart → compact again.

Failure/abort injection (§25): no injectable write-failure hooks exist in the
compaction path; crash gates (process death at exact windows) are the strongest
safe injection — documented as such; no E11 fault matrix.

## 6. Acceptance mapping

Every prompt §41 criterion maps to a family above; regressions E1/E2/E3/E4 re-run
under NEW run IDs and byte-compared; both checkers must pass; A5 appended ONLY if
the coordinated guarantee verifies.
