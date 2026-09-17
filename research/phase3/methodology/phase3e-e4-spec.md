# Phase 3E — E4 Specification: Online Backup, Snapshot Consistency & Restore Integrity

Status: IMPLEMENTED. Baseline: 2be1faf (E3 final). Scope: E4 ONLY.

## 1. Backup implementation audit (at 2be1faf, before changes)

`AttentionEngine::backup_to(dest)` (core/src/engine.rs): acquires the **mutation gate**
(held across BOTH phases), runs `checkpoint_locked()` (WAL fsync → SST flush → idmap →
manifest+CURRENT install → rotate → trim → manifest-GC), then
`backup::copy_database_dir(db, dest)`.

`copy_database_dir` (core/src/backup.rs): copies exactly
CURRENT + MANIFEST/ + sst/ + META/ + WAL/ (recursive `fs::copy`), then loads the COPIED
catalog to validate openability, then writes **backup-meta.json LAST** (format version,
db format version, timestamp, checkpoint_seq, collections, source_dir) + fsync_dir.
HNSW graphs excluded (rebuilt deterministically at open — documented).

`restore_backup(src, dest)`: refuses non-empty destination; parses backup-meta.json
(pre-E4: silently substituted a default meta if ABSENT); copies the five items; loads
catalog; cross-checks database_format_version; opens the restored DB (Sync) before
declaring success.

Audit answers (§4 of the prompt):
1. A backup = byte-for-byte copy of the authoritative set (CURRENT, MANIFEST/, sst/,
   META/, WAL/) + backup-meta.json completion record.
2. Yes — file-level copy of the live directory's authoritative files.
3. Yes — internally consistent (see 4/8).
4. The checkpoint INSIDE backup_to, serialized against mutations by the gate: after it,
   all acked state through seq N is in SSTs+idmap+manifest; the copy runs on a QUIESCENT
   directory.
5. Yes — checkpoint is mandatory in backup_to (it IS the snapshot boundary).
6. All WAL segments present at copy time: post-checkpoint the WAL holds the new active
   segment (created by the checkpoint's rotate) — copied; pre-checkpoint covered
   segments were trimmed by the same checkpoint. If a trim window retained a covered
   segment it is copied too (harmless: covered by checkpoint).
7. Segments fully covered by checkpoint_seq could be omitted (replay skips ≤cp) — an
   optimization, not needed for correctness; current code copies what exists.
8. Not while the copy runs: the mutation gate blocks all mutations/checkpoints (which
   are the ONLY writers of those files) for the entire backup. E3 established these are
   the sole mutation paths; compaction runs inside flush (gated) and compact_all
   (offline tool).
9. No (consequence of 8). Published SSTs are immutable in practice: flush writes
   <ts>.sst at unique timestamped names and compaction (gated) replaces the SET only
   while the gate is held; backup copies under the same gate → no replacement can
   interleave.
10/11/12. Yes; backup_to acquires it; writers BLOCK for the backup's full duration
   (checkpoint + copy). This is **coordinated backup with bounded writer pause**, not
   non-blocking. Readers (RwLock read) never block.
13. Yes — CURRENT/MANIFEST copied while quiescent; both were installed atomically
   (tmp→rename→dirsync) by the checkpoint.
14. Yes for SSTs at final names (unique <timestamp>.sst; content never edited in
   place). WAL ACTIVE segment is appended to by writers — but not while the gate is
   held.
15. Pre-E4: checksums existed only at the EVIDENCE layer (3D BACKUP-002 verified
     source==backup==restored trees with SHA-256 from the harness; engine-level CRCs
     on WAL frames + SST records refuse corruption at open). No per-file hashes inside
     backup-meta.json — E4 keeps it that way (§17: no fields without a correctness
     need; corruption detection exists at restore via catalog load + full engine open).
16. Failure halfway: dest has a partial tree WITHOUT backup-meta.json (it is written
     last) → detectable.
17. **PRE-E4 BUG (fixed in E4, minimal change):** restore on a meta-less directory
     substituted a DEFAULT meta and proceeded — a partial backup could restore. Fixed:
     restore REFUSES when backup-meta.json is absent (it is the completion marker).
18. Source crash during backup: the source dir is never mutated by the copy (reads
     only); its own checkpoint already completed atomically → source recoverable (E3).
19. Source mutation after a file is copied: impossible while the gate is held; after
     backup releases the gate the copy is complete (meta written before release).
20. Snapshot identity: yes — backup-meta.json records checkpoint_seq + timestamp +
     collections; the restored DB's catalog checkpoint_seq matches. (WAL high-watermark
     is implicitly present via the copied wal-state.json + E1 open enforcement.)

## 2. Snapshot model (definition used by E4)

**Boundary** = acquisition of the mutation gate by backup_to (all ops whose ACK is
visible before backup_to RETURNS are inside; ops attempted during backup block and
ACK only after it returns — no in-flight state can straddle the boundary).
**Content** = the post-checkpoint durable state at that boundary, in ALL durability
modes: the internal checkpoint fsyncs the WAL first, so even Async-buffered ACKed
writes are captured (E2/E3 semantics composed with backup; verified in modes matrix).
**Isolation** = post-backup source mutations never enter the copy (gate + copy-before-
release).

## 3. Classification used for the verdict

- Quiescent backup — writers stopped by the caller.
- Coordinated backup — writers continue issuing ops; the gate PAUSES them for the
  backup duration (bounded pause; no op lost, no op partially captured).
- Fully online (non-blocking) backup — UNSUPPORTED by design (gate); documented, not
  forced.

## 4. E4 code changes (minimal, in-scope)

restore_backup: (1) refuse missing backup-meta.json; (2) refuse
backup_format_version != 1. Nothing else changed — copy/checkpoint/gate paths are
already correct by construction (verified by the experiments below).

## 5. Experiments

PH3E-BACKUP-003 (driver `e4run`, in-process controller): B1 quiescent, B2 read-
concurrent, B3 single-writer, B4 multi-writer, B5 writer+checkpoint, B6 writer+rotation
(2 KiB segments), B7 writer+checkpoint+rotation (async), snapshot-transition (v1→v2
update_document), delete-reinsert, 3-collection isolation, post-backup isolation,
multi-backup B1<B2<B3, durability-mode matrix (sync/group/async writer-backup),
blocking instrumentation (per-op latency, pause = backup duration). Independent
reference model: fsynced per-op ACK sidecar; expected snapshot = acks visible at
backup-return; restored state compared against that model (never against final source
state; checker necessary-but-insufficient).

PH3E-BACKUP-004 (integrity): partial backup (meta removed) refuses; truncated SST
refuses; corrupt WAL byte refuses; corrupt CURRENT refuses/records; malformed meta
refuses; unsupported format version refuses; restore into non-empty dest refuses;
source-dir safety (source reopens, all acks present, no unexpected source mutations).

PH3E-BACKUP-005 (crash during backup): child process performs backup; controller
group-kills mid-copy; asserts (a) source reopens checker-clean with all acks, (b) the
partial backup is REFUSED by restore (no meta), (c) a pre-crash completed backup still
restores exactly.
