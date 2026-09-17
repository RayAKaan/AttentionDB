# Phase 3E — E3 Specification: Machine-Crash / Power-Loss Durability

Status: IMPLEMENTED (this file is the E3 design + implementation audit record)
Baseline: 5c56ffe (E2 final). Scope: E3 ONLY.

## 1. Environment record (part of the evidence)

- OS/kernel: Linux 6.1.158+ SMP PREEMPT_DYNAMIC x86_64 (Debian-based container)
- CPU/RAM: 2 vCPU Intel Xeon @ 2.60 GHz, 2 GB RAM
- Storage: root filesystem **ext4** (rw, relatime, discard), virtualized VM disk layer
  (sandbox container on a VM); `/tmp` (test DB dirs) is on the same root fs
- NOT tmpfs; journaling ext4 with default data ordering (data=ordered)
- Isolation: this is a container on a VM. There is NO host power control, NO block-device
  layer access (no root), NO fsfreeze/dm-flakey/dm-log-writes tooling.

## 2. Implementation audit (fs-layer crash-consistency map)

Paths relied upon, traced at commit 5c56ffe (audit only — unchanged by E3):

| structure | write protocol | atomic? | dir fsync? |
|---|---|---|---|
| WAL frames | append into `BufWriter` on segment file; durability per mode (Sync=flush+fsync per append; GroupCommit=flush; Async=none) | n/a (torn tail = truncation, E-policy) | segment created via `open(…, create+append)`; `wal-state.json` written tmp→rename→fsync_dir(WAL) at every segment creation |
| `wal-state.json` | tmp file + sync_all → rename → fsync_dir | YES | YES |
| SSTables | `flush_memtable`: `sstable_<ts>.sst` written **directly at final name** (`SSTableWriter` + `w.flush()`), then opened in place; then optional in-place compaction + reload | **NO tmp→rename** (audit finding F-A) | NO dir fsync of sst/ (audit finding F-B) |
| idmap.bin | tmp + sync_all → rename → fsync_dir (same protocol as manifest) | YES | YES |
| manifest-N | tmp (write+sync_all) → rename → fsync_dir(MANIFEST) | YES | YES |
| CURRENT | tmp (write+sync) → rename → fsync_dir(db) | YES | YES |
| trim | `retain_from`: `remove_file` of fully-covered segments AFTER manifest install | n/a | (unlinked-but-open semantics irrelevant: no other process) |

Audit answers to the §4 question list:

- File data flushed before rename? YES for manifest/CURRENT/idmap/wal-state (`sync_all`
  on the tmp before rename). NO for SSTables (written at final name; `SSTableWriter`
  flush is a userspace+file sync? — `w.flush()` closes frames; `sync_all` NOT called — F-A).
- Directory fsynced after rename? YES (manifest dir, db dir for CURRENT, WAL dir for
  wal-state, meta dir for idmap). NO for sst/ (F-B).
- Manifest fsynced? YES (file tmp sync_all + dir fsync).
- WAL fsynced before ACK under Sync? YES (`sync_all` per append; E2-gated).
- GroupCommit flushes exactly: userspace `BufWriter` → OS page cache, per append.
- Async buffers: 8 KiB `BufWriter` userspace; lost on process death unless auto-flushed
  at capacity or moved by structural points.
- Process dies after write before rename (manifest path): CURRENT still points at the
  OLD generation; orphan `manifest-N.tmp` remains. `latest_generation`/fallback scan
  parse the tmp's stem as generation N — a FULLY written tmp decodes (CRC ok) and can be
  selected by fallback (benign: content is a complete valid manifest; behaviorally
  equivalent to a completed rename — E3 TESTS this via MANIFEST_AFTER_TMP_WRITE); a torn
  tmp fails CRC and is skipped (next save uses gen N+1 because latest_generation counted
  the tmp — no collision).
- Machine dies after rename before dir fsync: the rename may or may not survive. Recovery
  rule: CURRENT(old)+new manifest, or CURRENT(new)+manifest — all four combinations are
  handled by CURRENT-first-then-fallback decode. E3 tests each reachable window.
- Machine dies during checkpoint: see §5 windows; valid outcomes = old-cp + full WAL
  replay, or new-cp + trimmed WAL, or REFUSED (E1) — never a hybrid that opens wrong.
- Machine dies between SST install and WAL trimming: manifest(N) installed, WAL intact
  (untrimmed) → open replays from N+1; trim is idempotent at next checkpoint.
- wal-state durable but segment not: E1 gate (1)/(2) — missing active segment with no
  newer segment ⇒ WAL_LOST_SEGMENT refuse; with newer segment (post-rotation state write
  crashed) ⇒ tolerated by design (A1).
- Manifest points at SST whose contents are incomplete: SST files carry CRC-framed
  records; a torn/corrupt SST is fatal at open (t15c). Because flush_memtable writes at
  the final name, a mid-SST-write crash can leave a torn `sstable_*.sst` that the loader
  REFUSES (fail-safe: the pre-checkpoint state is still fully in the WAL, but the loader
  scans the directory and refuses before replay). Audit finding F-C: this window is
  REFUSE-not-RECOVER; safe (no silent corruption) but not self-healing. E3 probes the
  adjacent windows (before/after full SST write); the mid-write torn case is covered by
  the existing t15c corruption-fatal contract and is documented, not newly instrumented
  (no gate inside SSTableWriter frame loop in E3 scope).

## 3. Failure model (hierarchy actually testable HERE)

- **F0 graceful close** — control only.
- **F1 process crash (SIGKILL/SIGABRT)** — E2 evidence; referenced, NOT re-presented as E3.
- **F2E — environment-termination-equivalent (strongest available here)**: the workload
  process runs in its OWN process group; it *parks* at the exact in-engine window gate
  (marker file fsynced first); the controller then SIGKILLs the ENTIRE process group —
  no destructors, no flushes, no orderly cleanup of anything holding DB state. The
  engine is single-process (no helpers), so group-kill covers every process that holds
  database state. This is the honest maximum this infrastructure offers: it is NOT a VM
  termination (the kernel, page cache, and disk image survive untouched); it is
  "every DB-state-holding process dies simultaneously without cleanup, mid-operation".
- **F3 filesystem/cache disruption** — **BLOCKED**: requires root + block-device
  infrastructure (fsfreeze/dm-flakey/dm-log-writes/QEMU); unavailable in this sandbox.
- **F4 physical power loss** — **BLOCKED**: no controllable power mechanism. Not
  simulated, not substituted (§25).

Page-cache note: F2E (like F1) leaves the page cache intact, so F2E CANNOT distinguish
Sync from GroupCommit at the machine axis — both survive; only Async-classified data
(userspace-buffered) is at risk. The Sync-vs-GroupCommit distinction at the MACHINE axis
(fsyntax of power loss) remains NOT VERIFIED / BLOCKED. This is stated, not hidden.

## 4. Crash windows (gates; zero-cost when env unset; reuse E2 gates where they exist)

C1 before_wal_append · C2 after_write · C3 after_apply · C4 after_flush (group branch)
· C5 after_fsync (sync branch) · C6 before_ack · C7 after_ack (harness-level park) —
E2 gates reused. NEW structural gates (E3): CKPT_AFTER_WAL_FSYNC, CKPT_AFTER_SST,
CKPT_AFTER_IDMAP, MANIFEST_AFTER_TMP_WRITE, MANIFEST_AFTER_MANIFEST_DIRSYNC,
MANIFEST_AFTER_CURRENT_TMP_WRITE, MANIFEST_AFTER_CURRENT_RENAME (inside
`Catalog::save`/`checkpoint_locked`), SST_AFTER_WRITE (inside `flush_memtable`),
ROTATE_AFTER_OLD_FSYNC, ROTATE_AFTER_STATE_WRITE (inside `Wal::rotate`/`open_segment`),
CKPT_AFTER_ROTATE, CKPT_AFTER_TRIM. Unreachable-in-mode windows are recorded
NOT_REACHED; windows with no gate: NOT_INSTRUMENTED (only the mid-SST-frame write and
inside-fsync itself).

## 5. Workloads and expected-state model

Independent reference model (mandatory §21): the controller derives expected state from
the fsynced ACK sidecar + window/mode contract table — never from the database. The
generator encodes the contract table; any fact/contract deviation → MISMATCH → fail.

- A ack-inserts (10 baseline + 1 target): per-window survival per E2 contract
  (C2/C4-class pre-durability → ABSENT; C5/post → PRESENT; async baseline ANY/ABSENT).
- B mixed mutations (inserts + deletes + inserts + checkpoint window): exact set
  membership incl. deleted-stay-deleted (no resurrection).
- C transactions (E2 shapes): ACK⇒all-or-nothing; 0 partial ever.
- D checkpoint boundary (16 acked docs, crash INSIDE checkpoint at 9 windows):
  all modes ⇒ all acked PRESENT at every window (checkpoint fsyncs WAL first; every
  partially-installed checkpoint state recovers to old-cp+replay or new-cp), restart
  equality ×2.
- E WAL-rotation boundary (2 KiB segments; crash inside the rotation of the in-flight
  append): rotation PRECEDES the frame write ⇒ in-flight record ABSENT in ALL modes;
  all previously acked records PRESENT (rotation fsyncs the completed segment); E1
  watermark invariants hold at every open.
- F0 close/reopen control.

## 6. Repetitions

3 repetitions per cell (deterministic workloads; run-to-run variance limited to fs
timing, which cannot change the instrumented window). Total spawned cells ≈ 192
(± NOT_REACHED rows). Counts recorded exactly; no percentages (§23).

## 7. Artifacts

raw: `PH3E-E3-001` (A/D/E/B/C/F0 suites in one driver, one CSV per suite), supersede
policy per E2 precedent. results: `e3-*.csv` generated by `generate_results_ph3e.py`
(expectation gate, MISMATCH→exit 1). Registry: PH3E-E3-001..00N. Checker: gate 20
(failure_model column mandatory; POWER_LOSS rows must be BLOCKED; no VERIFIED without
raw run). Amendment A3 only if evidence requires it (it does — see final report §18).
