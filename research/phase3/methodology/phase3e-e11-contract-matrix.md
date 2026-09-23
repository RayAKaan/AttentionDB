# Phase 3E — E11 Contract Matrix (fault-injection targets)

Recovered from the actual repository documents before any official E11 run.
Every row names its source; missing/ambiguous guarantees are recorded as
ambiguities, never converted into stronger promises. Failure model for ALL
rows: sudden process death (`std::process::abort()` / process-group SIGKILL)
at instrumented gates — the A3 model. Physical power-loss is OUT OF SCOPE
(§13 of the E11 prompt; A3 boundary preserved).

| ID | Area | Source (doc §) | Expected behavior under fault | Fault boundary to test | Observable evidence | Pass condition | Explicit non-claims |
|----|------|----------------|-------------------------------|------------------------|---------------------|----------------|---------------------|
| C1 | WAL integrity — refusal | production-contract G5; A1; storage/src/wal.rs (framing `WAL2` magic+len+body+crc32; segments `WAL/<start_seq:020>.wal`; `wal-state.json` sidecar) | Corrupt frame, gapped/misnamed segment, broken sequence continuum → open REFUSES; database never fabricates data | File-level tamper: truncated tail, corrupted body, bad magic, sequence gap, missing segment, sidecar mismatch | open() error kind/message; pre/post dir inventory + sha256 | Corrupt/gapped/missing → refused; torn tail → truncated with warning + intact prefix recovers; legacy (no sidecar) tolerated per A1 | No claim about media-level bit rot beyond the tested byte edits |
| C2 | WAL integrity — rotation | A1 (durable rotation-time invariant); GATE_ROTATE_* | Rotation never loses the sealed old segment before the new state is durable | `rotate_after_old_fsync`, `rotate_after_state_write` gates | Ack log vs recovered set; wal-state.json present | Acked (Sync) writes survive; reopen clean | — |
| C3 | Sync durability | G3; A2 | Ack ⇒ record fsynced before acknowledgment | `after_write` (userspace buf), `after_flush` (page cache), `after_fsync`, `before_ack` gates | ack-log.jsonl vs recovered set | EVERY acked op recovered; durable-but-unacked extras allowed (reported) | fsync ≠ validated against every hardware failure mode (A3) |
| C4 | GroupCommit durability | G3; A2 | Ack ⇒ flushed to OS page cache (NOT coalesced batching — verify actual semantics from implementation) | same gates as C3, group mode | ack-log vs recovered set | Every acked op recovered under process death | No OS/power-failure claim |
| C5 | Async durability | G3; A2; E8 report (buffered-WAL-tail boundary); E8f pattern | Acked-but-buffered tail may be lost on process death; `flush_wal()`/`checkpoint()` promote acks | `after_write`/`after_flush` gates mid-run; and graceful-reopen-without-checkpoint leg | loss count = acked − recovered; which ops lost | Loss ⊆ unflushed tail; zero unacknowledged leakage; loss is DOCUMENTED boundary, not defect | No claim that async preserves acked writes across process death |
| C6 | Transaction atomicity | G6; A6; E6 report (commit boundary) | A txn is committed only at the CommitTxn boundary; recovery is ALL-or-NOTHING; no partial visibility | `tx_before_commit_wal`, `tx_after_commit_wal`, `after_apply`, `before_ack` on commit_transaction; rollback leg | model txn table vs recovered state | Committed ⇒ all ops visible; not-crossed ⇒ zero ops; never partial | No isolation/MVCC/cross-collection/update-txn claims (G6 scope: Insert+Delete, single collection) |
| C7 | Rollback | G6 | Rollback discards staged ops; unstaged txn unusable | rollback then crash legs | recovered state | Rolled-back ops absent after recovery | — |
| C8 | Checkpoint | G1/G4; ckpt gate docs; A5/A8 maintenance isolation | Checkpoint flushes memtable → SST → id-map → manifest → trims WAL; interrupted at ANY interior point → reopen recovers acked state (or refuses per contract) | `ckpt_after_wal_fsync`, `ckpt_after_sst`, `ckpt_after_idmap`, `ckpt_after_rotate`, `ckpt_after_trim`, `manifest_after_tmp_write`, `manifest_after_manifest_dirsync`, `manifest_after_current_tmp_write`, `manifest_after_current_rename`, `sst_after_write` | ack log vs recovered set; checker; repeat-restart equality | All Sync-acked ops recovered; checker clean; repeated restart deterministic | Intermediate states need not be openable if contract requires refusal — absence of refusal must match contract |
| C9 | Backup/restore | G8; A4 (coordinated snapshot; `backup-meta.json` written LAST = completion marker; restore requires manifest) | Backup under mutation gate is consistent; PARTIAL backup (no manifest) must never be reported valid | `backup_mid_copy`, `backup_after_copy` (pre-manifest) gates; interrupted-restore leg | dest inventory: manifest present?; open-from-backup outcome | Full backup restores exactly (counts+model+checker); partial dest → refused/invalid, source intact | No nonblocking-backup claim (mutation gate pauses writers) |
| C10 | Compaction | G9; A5 (coordinated compaction; crash windows; tombstone retention in partial merges) | Compaction preserves logical state incl. deletions at every crash point; readers/writers may block per contract | `compact_before_merge`, `compact_after_output`, `compact_after_cleanup`, `compact_after_install` | recovered state vs model; checker | State + deletions preserved; checker clean; restart after compaction deterministic | Blocking is not a defect; no performance claim |
| C11 | Index hygiene | A9 + D40 (head-count-aware deadness: `dead = vstore − mapped×heads`); INV-E9-HYGIENE | Fresh multi-head checkpoint does NOT spuriously rebuild; legitimate dead-entry cleanup fires; interrupted rebuild → deterministic re-reconstruction on recovery; no duplicate live ids | `rebuild_mid` (new E11 gate) during recovery rebuild; ckpt-with-hygiene crash legs | rebuild-occurrence evidence; counts; retrieval self-hit (frozen E10 gate ≥95% where measured) | Multi-head fresh ckpt: zero rebuilds; post-interruption reopen: exact counts, no duplicates, deterministic | No threshold tuning; interrupted in-RAM rebuild need not resume mid-position (contract promises deterministic reconstruction) |
| C12 | Concurrency | G7; A7.1–A7.8 | Per-key LWW under gate serialization; no dirty reads; committed visible at commit point; maintenance isolated; NO isolation-level claim | Bounded writer+reader with mid-run fault (`before_ack`/`after_write`), paced reader (E10 finding) | reader observation log; ack log; recovered state | Zero reader errors; no uncommitted data observed; recovered state matches model within contract (Sync: all acked) | No linearizability/serializability claim (A7.5 lost-updates-possible preserved) |
| C13 | Recovery/rebuild | G1/G4; recovery-path rebuild (sealed) | Deterministic index reconstruction from durable records; repeated restart identical | any crash leg + double restart | repeat-restart state hashes equal | Identical state + checker across restarts | No recovery-latency claim beyond measured E10 tiers |
| C14 | Scale envelope | A10 | E11 must not exceed the verified envelope: integrated lifecycle ≤ 60k; principal F09 target 40k per prompt | n/a (constraint) | config.json of every run | No E11 run exceeds 60k docs or claims beyond A10 | E10 boundaries (80k caveats, ≥100k blocked) stand unchanged |
| C15 | Failure model | A3; §13 of prompt | Process-death axis ONLY | n/a | methodology + report language | No power-loss/physical-crash language anywhere | Physical power-loss: UNSUPPORTED (no controlled source); VM-reset equivalence NOT claimed |

Ambiguities found during recovery (recorded, not strengthened):
- AMB-0 (RESOLVED BY MEASUREMENT, PH3E-FAULT-007/038): a REGRESSED
  `wal-state.json` high_watermark is TOLERATED on open (full state recovers;
  the WAL records are authoritative). Refusal applies to record-level
  corruption and sequence-continuum breaks, not to derived-bookkeeping
  regression. C1 expectation corrected accordingly.
- AMB-1: G3 says GroupCommit = "flushed to OS page cache"; the wal.rs doc for
  GATE_AFTER_FLUSH matches. No claim is made about grouping/coalescing
  behavior; E11 reports what the implementation actually does (flush per
  append observed via gate placement), no stronger wording.
- AMB-2: For a partial backup directory the contract says restore "requires
  the manifest"; E11 treats open/restore refusal of a manifest-less copy as
  the expected evidence and does not demand a specific error string.
- AMB-3: Async graceful-reopen loss (E8f) is a documented boundary; its exact
  loss count is workload-dependent. E11 asserts the boundary shape (loss ≤
  unflushed tail; zero leakage), not a specific count.
