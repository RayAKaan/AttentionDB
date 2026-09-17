# AttentionDB — Production Contract (Phase 3E, E0 freeze)

Date frozen: 2026-09-17 · Author: Rayyan Kaan · Basis: Phase 3B/3C retrieval validation,
Phase 3D database validation (runs PH3D-*), code at commit 1204e35 (E0 freeze point).
Amendments to this contract are appended as dated sections (A1, A2, …) — semantics never
change silently. Companion: `phase3e-spec.md` (execution plan).

Vocabulary: **VERIFIED** (implementation + experiment exercising the failure mode),
**PARTIALLY VERIFIED**, **NOT VERIFIED**, **UNSUPPORTED**, **BLOCKED**, **OPEN**.

---

## 1. What AttentionDB guarantees

G1. **Logical state correctness** — VERIFIED (PH3D-STATE-001..003, INTEGRATION-001):
under mixed insert/update/upsert/delete/query/filter/checkpoint/compact/restart streams,
the observable logical state equals a reference model after every restart, compaction and
at the end; dead documents are never returned; results are deterministic.

G2. **Filter soundness** — VERIFIED (PH3D-FILTER-001): a document not matching the filter
expression can never be returned. Completeness is NOT guaranteed (candidate-bound;
measured recall 0.967–1.0 on tested shapes).

G3. **Durability per selected mode** — VERIFIED on the process-crash axis
(PH3D-CRASH-001..003, 21/21 points):
   - `Durability::Sync`: fsync per WAL append. An acknowledged write survives process
     death; by implementation also machine failure, but the machine axis is PARTIALLY
     VERIFIED (§3).
   - `Durability::GroupCommit`: WAL appends flushed to the OS page cache. An acknowledged
     write survives process death; it does NOT necessarily survive OS/power failure.
   - `Durability::Async`: WAL appends buffered in userspace. An acknowledged write is NOT
     guaranteed to survive process death (documented, run-backed:
     `COMMITTED_NOT_DURABLE_ASYNC`). `flush_wal()` / `checkpoint()` promote acks.

G4. **Crash recovery** — VERIFIED for the tested points: recovery yields all acknowledged
writes (Sync/GroupCommit) or the intact acknowledged prefix with zero unacknowledged
leakage (Async); transactions are all-or-nothing (never partial) at every tested point;
a torn WAL tail is truncated with a WARNING and the intact prefix recovers.

G5. **WAL corruption refusal** — VERIFIED (PH3D-WALCORRUPT-001): a corrupt frame, a gapped
or misnamed segment, or a broken sequence continuum refuses to open. The database never
silently fabricates data. (Pre-E1 exception: §2/Amendment A1.)

G6. **Transaction atomicity** — VERIFIED (PH3D-TX-001; crash legs): commits apply
completely or not at all, under validation failure and process death; rollback discards.
Scope: `TxnOp = Insert | Delete`, single collection per transaction. **No update txn op,
no cross-collection transactions, no isolation levels (UNSUPPORTED).**

G7. **Concurrent stability** — VERIFIED (PH3D-CONC-001..003): parallel readers and mixed
read/write runs complete with zero errors and checker-clean state; mutation records are
never torn; per-key last-write-wins under gate serialization. **No linearizability or
isolation claim.**

G8. **Backup/restore (quiescent)** — VERIFIED (PH3D-BACKUP-001/002): a copy taken from a
closed/quiescent database restores to exactly the captured logical state, with per-file
sha256 integrity. Restore requires the manifest (`backup-meta.json`).

G9. **Compaction** — VERIFIED offline (PH3D-COMPACT-001): full merge + tombstone GC
preserves logical state, latest versions, and deletions. Automatic incremental compaction
triggers at ≥4 SST files and RETAINS tombstones in partial merges (anti-resurrection).

G10. **Multi-collection isolation** — VERIFIED with namespaced ids (PH3D-INTEGRATION-001):
uuid identity is GLOBAL; a collection is a membership tag. Callers must namespace logical
ids per collection (documented semantics).

## 2. What AttentionDB does NOT guarantee (as of freeze)

N1. No isolation levels, no serializability, no linearizability (not implemented; no
history-based claim). Operation ordering = single mutation gate + per-key last-write-wins.
N2. No update operation, no upsert op, and no cross-collection transactions.
N3. No online backup: `copy_database_dir` requires a quiescent database (UNSUPPORTED
under active writers; the one observed consistent sample is documentation, not support).
N4. No online (engine-open) compaction: `compact_all` is dir-level, offline (UNSUPPORTED
under load; BLOCKED by design).
N5. Async-mode acknowledged durability across process death (UNSUPPORTED by design; use
GroupCommit/Sync).
N6. Machine/power-loss durability: PARTIALLY VERIFIED (implemented via fsync in Sync;
no power-cut harness — see §3 and Phase 3E E3).
N7. **Pre-E1 hole**: deleting the only pre-checkpoint WAL segment let a database open as
apparently-valid and empty. **Closed by Amendment A1 (E1).**
N8. No distributed operation, replication, or sharding (out of scope by standing decision).
N9. No memory optimization claims (PH3D-MEM-OPT-001 not run; Phase 3C baselines stand).

## 3. Operational semantics (the ten E0 questions)

Q1 **Write acknowledged → what happened?** The mutation is in the WAL and applied to
in-memory state; visibility to subsequent reads is immediate (read-your-writes within the
process). Persistence depends on the durability mode (G3): Sync = fsync'd; GroupCommit =
page-cache flushed; Async = userspace buffer only.

Q2 **Process crash?** Reopen recovers per G3/G4: all acked (Sync/GroupCommit), intact
prefix (Async), transactions all-or-nothing, torn tail warned+truncated, corrupt WAL
refuses to open. Any unhandled termination leaks the engine directory on disk (PH3C
finding) — operational nuisance, never a correctness event.

Q3 **Power loss?** NOT TESTED. Sync's fsync-per-append is designed for it; GroupCommit/
Async are not. Verdict stays PARTIALLY VERIFIED until a machine-crash harness exists (E3).

Q4 **WAL files disappear?** Pre-E1: post-checkpoint deletion of required history was
detectable only partially (gap check); a deleted pre-checkpoint segment opened as a valid
empty database (the N5/E1 hole). Amendment A1 makes required-WAL loss a refusal to open.

Q5 **Backup during activity?** UNSUPPORTED: the copy does not coordinate with the mutation
gate; supported path is quiescent. Documented (PH3D-BACKUP-001 live probes).

Q6 **Compaction during activity?** UNSUPPORTED: `compact_all` is offline (close engine →
compact → reopen). Invoking it while an engine holds the dir merges files without cleanup
(PH3D-COMPACT-002, documented). Automatic incremental compaction during operation is
internal to flush and retains tombstones (safe by construction, G9).

Q7 **Transaction isolation?** None. Transactions serialize on the mutation gate with all
other mutations; there is no snapshot, no conflict detection, no concurrent-transaction
scheduling.

Q8 **Concurrency ordering?** Mutations: total order via the mutation gate (single
process). Reads: consistent per-call; readers may observe the state between gate
sections. No cross-key ordering guarantees are exposed; per-key writes are never torn.

Q9 **Supported database size?** Maximum = maximum reproducible configuration under a
stated envelope (Phase 3C principle). Measured envelope @DIM 512, ~2 GB sandbox:
≈30K head-docs (1h/30K ≈ 941 MB · 2h/20K ≈ 1152 MB · 4h/10K ≈ 1092 MB · 8h/5K ≈ 960 MB);
8h×10K×512 OOMs (preserved). Memory is the binding constraint (duplication ≥2× raw;
~6–8× unexplained — E9 will attribute). Phase 3D validated correctness up to ~1.8K live
docs in multi-collection/multi-head setups; larger corpora untested for DB semantics.

Q10 **Acknowledgment API distinction (planned, E2)?** Currently `commit()`/mutation
results do not distinguish Committed (applied+logged) from Durable (survives failure
classes). This distinction is scheduled for E2 and will be an API-visible contract
change recorded as Amendment A2. Until then, the mode table in G3 IS the ack contract.

## 4. Consistency gate

`check_engine`/`check_db_dir` remain the mandatory gate: ERROR-severity issues fail a run;
WARNINGs (retired-vector purge backlog) are documented non-fatal. The gate runs after
build, mutations, restart, replay, crash-recovery, compaction, concurrency, and restore.

## 5. Amendment log

- **A1 (2026-09-17, E1)** — WAL integrity invariant: a durable rotation-time WAL
  high-water record in the `WAL/wal-state.json` sidecar makes required-WAL loss a
  REFUSAL to open (`WAL_LOST_SEGMENT` / `WAL_SEQ_GAP`); unparseable sidecar records
  refuse (`WAL_STATE_CORRUPT`); absent sidecar = legacy database, opens with current
  semantics and adopts the invariant at its next segment creation/rotation. The
  catalog gains NO watermark field (bincode v1 positional — durable sidecar files
  only). Details in `phase3e-spec.md` §E1 and the E1 evidence report.

- **A2 (2026-09-17, E2) — Durability/Acknowledgment semantics.**
  - *Old wording:* G3 (mode table) plus Q10: "commit()/mutation results do not
    distinguish Committed from Durable ... scheduled for E2 ... will be an API-visible
    contract change recorded as Amendment A2. Until then, the mode table in G3 IS the
    ack contract."
  - *New wording:* the E2 audit + experiments ESTABLISH the relationship and adopt the
    following vocabulary as THE acknowledgment contract. **No API surface change** (see
    decision below).
    - **Committed** — the database accepted the operation/transaction through its
      logical commit path (mutation applied to in-memory state; WAL record(s) appended;
      for transactions: the COMMIT record appended).
    - **Durable** — the durability mechanism required by the selected mode has
      completed: Sync = frame fsynced (`sync_all`); GroupCommit = frame flushed to the
      OS page cache; Async = nothing beyond the userspace `BufWriter` write.
    - **Acknowledged** — the API returned `Ok` to the caller (always AFTER both the WAL
      append and the in-memory apply; never before).
    - Established relationships (evidence PH3E-DUR-001..004):
      - `Sync`: **Acknowledged ⇒ Durable(machine boundary) ⇒ survives process crash.**
        Crash gate `after_fsync` (frame fsynced, not yet acked): record always
        recovered; acked records never lost in any of 27 cells.
      - `GroupCommit`: **Acknowledged ⇒ Durable(process boundary) ⇒ survives process
        crash, NOT machine crash.** Gate `after_flush` (page cache, not fsynced):
        record recovered after SIGABRT. Despite the name there is NO coalescing/group
        formation: the engine-wide mutation gate serializes all mutations; every append
        flushes before it returns (before the ack). The name is retained for
        compatibility; the semantics are "flush-to-OS before ack".
      - `Async`: **Acknowledged ⇒ Committed only; NOT durable.** Acked single writes
        LOST at the `after_ack` gate (all reps); an acked 10-op transaction LOST whole
        (documented); concurrent acked writes: 4/75 lost (userspace buffer tail).
        Loss is bounded ONLY by structural durability points: WAL rotation (completed
        segments), `checkpoint()` and `close()` (everything), `flush_wal()` (page
        cache). Survival of any particular acked Async write is an implementation
        artifact of the 8 KiB `BufWriter` boundary and MUST NOT be relied upon.
    - **Structural durability points (mode-independent, verified PH3E-DUR-003):**
      rotation (flush+fsync of completed segments), checkpoint (WAL fsync → SSTables →
      idmap → manifest → rotate → trim), graceful close (= checkpoint). After any of
      these, everything appended so far is machine-boundary durable in ALL modes.
      Rotation covers completed segments only: in Async the active-segment records
      remain loss-exposed until the next structural point (observed 27/30 preserved at
      2 KiB segments — exactly the E1 watermark boundary).
    - **Transactions:** atomicity boundary = presence of the COMMIT record in the WAL;
      durability boundary = the mode's action on that append. An acknowledged
      transaction NEVER recovers partially (0 PARTIAL in 63 cells): crash before the
      COMMIT flush discards the whole transaction even in Sync (gate `commit_written`:
      ops already fsynced, COMMIT still userspace-buffered → whole txn ABSENT).
    - **Multiple restarts:** the recovered state is a fixed point — restart/restart
      state equality + clean checker in 27/27 cells (PH3E-DUR-003).
    - *API decision:* the existing single-ack API is kept. Making Committed-vs-Durable
      API-visible would touch the api crate (unbuildable in this environment — protoc
      unavailable) for no semantic gain: the ack contract is fully determined by the
      selected mode per the table above. Q10 is answered by this amendment; the
      distinction is documented, not an API return-value change.
    - *Production default recommendation:* **GroupCommit** for production-oriented use
      (acknowledged writes survive process crash at measured cost parity with Async on
      the test filesystem — PH3E-DUR-005: mean ack 262 µs vs 259 µs; Sync for
      loss-of-no-acked-write-until-machine-failure requirements). The engine keeps NO
      implicit default: `open_dir` requires the caller to choose. Machine/power-loss
      durability remains NOT VERIFIED for every mode until E3 (N6 unchanged).
    - *Limitations:* process-crash axis only (SIGABRT/SIGKILL); machine-crash and
      power-loss claims are NOT VERIFIED (E3 scope); latency figures are
      sandbox-filesystem-specific and not production-comparable; unacknowledged
      in-flight writes MAY survive recovery (allowed, observed 0 times in 189 cells).
    - *Experiments:* PH3E-DUR-001 (81 cells), PH3E-DUR-002 (63), PH3E-DUR-003 (27),
      PH3E-DUR-004 (18), PH3E-DUR-005 (3), regressions PH3E-WAL-002 (identical to
      PH3E-WAL-001) and PH3E-REG-002 (all families clean).
    - N5 is superseded by this amendment: Async acknowledged loss is no longer merely
      "unsupported by design" — it is OBSERVED and characterized (single writes, whole
      transactions, buffer-tail batches). The guidance stands: use GroupCommit/Sync.

- **A3 (2026-09-17, E3) — Machine-crash durability semantics (failure-model boundary).**
  - *Old wording:* G3/N6/Q3: process-crash axis VERIFIED per mode; "Machine/power-loss
    durability: PARTIALLY VERIFIED (implemented via fsync in Sync ...)" and "Power loss?
    NOT TESTED." E2 left every machine-axis claim NOT VERIFIED.
  - *New wording:* durability claims are indexed by failure model:
    - **F0 graceful close / F1 process crash** — VERIFIED (E2 + E3 regression, unchanged).
    - **F2E environment-termination-equivalent** (process-group SIGKILL at exactly
      instrumented in-engine windows; single-process engine; no cleanup) — VERIFIED for
      the process-death-semantics axis: Sync fsync-before-ack boundary holds at
      C2..C7 windows (fsynced-but-unacked record SURVIVES: PH3E-E3-001 ack-c sync
      after_fsync 3/3); GroupCommit flush boundary holds (after_flush 3/3); structural
      points (checkpoint 9 windows incl. manifest-tmp orphan fallback; WAL rotation 2
      windows; mixed-mutation checkpoint) recover all acked state in ALL modes with
      checker-clean fixed-point restarts; zero refusals, zero corruption, zero partial
      transactions in 210 cells.
    - **F3 filesystem/cache disruption** — BLOCKED (no root / block-device tooling).
    - **F4 physical power loss** — BLOCKED (no power mechanism). NEVER simulated.
    - Consequently: "machine-crash durability" beyond process-death semantics remains
      **NOT VERIFIED** for every mode; the E2 boundary sentence stands unchanged. The
      strongest true statement is: *all acknowledged Sync/GroupCommit state survives
      abrupt termination of every process holding database state, at every instrumented
      commit-path and structural window, on ext4 (VM disk), with the page cache intact.*
  - *Environment:* Linux 6.1.158 container on VM, 2 vCPU Xeon, 2 GB RAM, ext4
    (rw,relatime,discard). Results MUST NOT be generalized to physical power loss on
    enterprise storage.
  - *Experiments:* PH3E-E3-001 (210 cells) + regressions PH3E-WAL-003, PH3E-DUR-007
    (byte-identical to E1/E2 originals).
  - *Limitations:* F2E ≠ VM kill ≠ power loss; mid-SST-write window NOT_INSTRUMENTED
    (refuses per corruption-fatal contract); ext4/VM-disk only.

- **A4 (2026-09-17, E4) — Backup / restore: coordinated snapshot semantics and restore
  validation.**
  - *Old wording:* backup existed (`Engine::backup_to` → directory copy: CURRENT,
    MANIFEST, sst/, META/, WAL/) but the contract made no backup claim; `restore_backup`
    silently synthesized a default meta when `backup-meta.json` was missing and accepted
    any `backup_format_version` (restore-integrity gap, fixed in E4, see §1.13 of
    `phase3e-e4-spec.md`).
  - *New wording (exactly as verified, no stronger):*
    - **Snapshot boundary.** `backup_to` acquires the engine mutation gate for the
      duration of the copy and runs a checkpoint inside that critical section, then
      copies CURRENT + MANIFEST + sst/ + META/ + WAL/ and writes `backup-meta.json`
      LAST (fsynced directory). The backup therefore represents exactly the logical
      state in which every operation acknowledged **before** the backup acquired the
      gate is present and no operation acknowledged **after** backup return is present.
      Operations concurrent with the backup are serialized against it (they pause for
      the backup's duration, measured 0.36–1.54 ms at the tested sizes).
    - **Coordination class: COORDINATED, not non-blocking.** Verified with concurrent
      readers (0 errors) and concurrent writers (single, 3-way, checkpoint, WAL
      rotation): every restored backup matched an independent fsynced reference model
      read at backup return (PH3E-BACKUP-004, 15/15 cells). A **fully online
      (zero-writer-pause) backup is NOT supported and NOT claimed**; writers colliding
      with a backup pause until it completes (measured: checkpoint waited 257 µs).
    - **Async durability composition.** A backup's internal checkpoint fsyncs the WAL;
      therefore every write ACKNOWLEDGED before the backup began is contained in the
      backup even under the Async mode (verified: modes-async 300/300 against the
      reference model). This does NOT change A2/A3: an Async ack still only implies
      "committed"; it is the backup's checkpoint — not the ack — that establishes
      durability of the captured state inside the backup.
    - **Restore validation (E4 gates).** `restore_backup` now REFUSES: (1) a backup
      directory without `backup-meta.json` (completion marker ⇒ crash-truncated or
      in-progress backups are never restorable: group-SIGKILL mid-copy produced a
      partial directory that was refused while the pre-crash backup still restored);
      (2) `backup_format_version != 1`; (3) a non-empty destination. Restore remains
      into an EMPTY destination only. Restoring a corrupt backup fails or recovers per
      the E1 documented corruption policy (torn-tail WAL accepted with intact prefix;
      damaged SST / `wal-state.json` / CURRENT refused via catalog fallback rules).
    - **Format.** `backup_format_version = 1`; unknown versions are refused, never
      mis-parsed. No correctness-necessary manifest fields were added in E4; the
      marker + version gate are validation-only.
  - *Evidence:* PH3E-BACKUP-004 (15/15 snapshot cells + 10/10 integrity cases),
    superseding PH3E-BACKUP-003 (harness defect, behavior identical); regressions
    PH3E-WAL-005 ≡ WAL-001, PH3E-DUR-008 ≡ DUR-001/002/003/004/006/007 (byte-identical),
    PH3E-E3-002 ≡ E3-001 (byte-identical). Report:
    `research/phase3/phase3e-e4-final-report.md`.
  - *Boundary:* claims hold for single-process local backups on ext4/VM at the tested
    sizes (≤ 20 000 docs). No incremental/remote/concurrent-backup-pairs claims; no
    machine-durability claim is derived from backup tests (A3 boundary unchanged).

- **A5 (2026-09-17, E5) — Compaction semantics (coordinated) and compaction safety.**
  - *Old wording:* document-store SST compaction existed as an automatic post-flush
    maintenance step (`compact`, full merge when all files are in the merge set,
    tombstone GC only then) and `compact_all` (full merge from the db root). No
    compaction claim was made in the contract; version resolution at compaction
    differed from open on equal-timestamp ties (latent bug, fixed in E5).
  - *New wording (exactly as verified, no stronger):*
    - **Class: C1 coordinated.** `Engine::compact_storage()` holds the engine mutation
      gate for the whole boundary→publish window; it is also the path used by the
      existing automatic trigger (post-flush, when ≥ 4 SST files exist, ≤ 8 files per
      merge — unchanged). Writers pause for the compaction window (bounded, ms-scale
      at the tested sizes; measured max writer op 1.8–2.3 ms including gate wait).
      Readers block under the document-store write lock for the merge/install window
      and never error (measured p50 53 µs / p99 116 µs across 604 concurrent attends
      during a real merge). **Fully online (C2) compaction is UNSUPPORTED and not
      claimed.**
    - **Version/tombstone semantics.** Compaction resolves overlapping versions with
      the SAME rule as open/recovery: (entry timestamp, file order), higher wins —
      including equal-timestamp ties (E5 fix). Tombstones are reclaimed only when the
      merge covers EVERY SST file (full merge proves no older version can become
      visible again, given that document UUIDs are never reused); partial merges
      retain tombstones. Deletion is permanent across compaction + restart.
    - **Publication & crash semantics.** Publication order is output SST
      (tmp→fsync→rename) → unlink superseded inputs → reader-list swap under the
      store write lock. Recovery never requires a removed SST (the catalog manifest
      names no SSTs; open scans sst/ and resolves versions by content). Group-kill
      at each instrumented window (before merge / after output / after cleanup /
      after install) restarts, in a fresh process, to exactly the pre-compaction or
      post-compaction logical state — never a hybrid — with zero stray .tmp
      artifacts; a later compaction then succeeds. Garbage `.tmp` files are ignored
      and removed; garbage `.sst` files are refused loudly (recovery fails — the
      documented corruption policy).
    - **Checkpoint / WAL / backup interaction.** Compaction does not touch the WAL,
      checkpoint_seq, wal-state.json, or manifest generations (no new generation is
      published). Checkpoint→compact and compact→checkpoint are both verified
      consistent; WAL rotation during compaction causes no sequence gap or watermark
      damage; backup and compaction are serialized by the mutation gate — a backup
      never references an SST that compaction later removes, and compaction never
      mutates an existing backup directory.
    - **Scope/boundary.** Verified for the single-process local document store at
      tested sizes (≤ 50 000 docs, ms-scale merges) on ext4/VM. No incremental/
      background/throttled scheduler beyond the existing flush trigger; no
      cross-process or distributed compaction; no claim beyond the tested failure
      model (process-group death; A3 boundary unchanged).
  - *Evidence:* PH3E-COMPACT-004 (25/25 cells incl. 4 fresh-process crash windows;
    supersedes PH3E-COMPACT-003); regressions PH3E-WAL-006 ≡ WAL-001,
    PH3E-DUR-009 ≡ DUR-001/002/003/004/008, PH3E-E3-003 ≡ E3-001/002
    (byte-identical), PH3E-BACKUP-005 (classification-identical, integrity
    byte-identical). Report: `research/phase3/phase3e-e5-final-report.md`.
