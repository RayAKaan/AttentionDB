# Phase 3E — E5 Final Report: Online Compaction, Tombstone Safety & Concurrent Storage Lifecycle

Experiment: **PH3E-COMPACT-004** (supersedes PH3E-COMPACT-003, harness gap D1) ·
Date: 2026-09-17 · Baseline: cf2be27 (E4 final) · Author: Rayyan Kaan

---

## 1. Executive Summary

AttentionDB **can safely compact a live database** — with a bounded writer pause.
The coordinated compaction entry point `Engine::compact_storage()` (C1) was verified
in 25/25 cells against an independent fsynced reference model: version resolution
(including the equal-timestamp tie case whose latent bug E5 found and fixed),
tombstone GC that is provably safe only under full merge, delete/reinsert/ID-map
correctness, repeated compaction, concurrent readers (0 errors; p50 53 µs / p99
116 µs while blocked), single- and multi-writer agreement, checkpoint/WAL-rotation/
backup interaction, manifest fallback, partial-artifact policy, and fresh-process
crash recovery at all four instrumented publication windows. Fully online (C2)
compaction remains **UNSUPPORTED** by the mutation-gate architecture and is not
claimed. E1–E4 regressions are byte-identical (or classification-identical).

**Verdict (§32): B — Coordinated compaction with bounded writer pause** (and C-class
reader availability: concurrent readers + coordinated writers).

## 2. Baseline

E4 final commit **cf2be27**, tree clean, branch `main`. Prior artifacts untouched:
PH3D-COMPACT-001/002, PH3D-COMPACTION-001, all PH3E-\* runs. Registry at E5 close:
83 experiments.

## 3. Environment

rustc/cargo 1.98.1 (stable, /var/tmp/toolchain), Python 3.13.14, Linux 6.1.158
container on VM, 2 vCPU, 1.9 GiB RAM, ext4. protoc installed for the workspace
build. Same environment family as E2–E4; no power-loss or machine-crash claims
anywhere (A3 boundary).

## 4. Implementation Audit (pre-change, full detail in spec §1)

- **What is compacted:** document-store SSTables in `<db>/sst/` (key = UUID bytes,
  value = msgpack Record; tombstones carry `__TOMBSTONE__`). DB-level store;
  collection membership is a record tag. HNSW `compact_index` is the frozen
  retrieval subsystem (out of scope).
- **Generations:** none named — recovery SCANS `sst/` (sorted) and resolves per-key
  versions by (entry timestamp, file order). Flush files `sstable_<ms>.sst`;
  merged outputs `compacted_<ns>.sst`.
- **Tombstones:** ordinary records with the tag; reclaimed **only** when the merge
  covers every file (`compact_all`, or `compact` when merge set == all files) —
  sound because UUIDs are never reused (retire + fresh reinsert UUIDs).
- **Immutability/readers:** SSTs are written once (tmp→`sync_all`→rename→
  `fsync_dir`) and never edited; `SSTableReader` materializes all entries in memory
  at open, so no reader holds a file handle; readers borrow the reader list only
  inside `document_store` RwLock scopes.
- **Manifest/WAL:** the catalog names NO SSTs (checkpoint_seq, collection metas,
  next_doc_numeric_id); compaction touches neither manifest generations nor WAL
  state (checkpoint_seq, high-water mark, wal-state.json).
- **Locks:** every compaction call site (auto post-flush and the new entry point)
  runs under the engine mutation gate; the reader list swaps under the store write
  lock. Backup holds the mutation gate across checkpoint+copy ⇒ **backup cannot
  race compaction**; checkpoint cannot race compaction; writers cannot race
  compaction; readers can only block.
- **Pre-existing verified behavior:** Phase-1 TEST 11 (no resurrection after full
  compaction) = the C0 control baseline; `open_inner` deletes stray `.tmp` files.

## 5. Compaction Model

C0 offline (quiesced; control) · C1 coordinated (gate-held boundary→publish;
writers pause; readers block, zero errors) · C2 fully online (UNSUPPORTED — single
mutation gate + store write lock; not implemented, not claimed). E5 implements and
verifies C1 only. The automatic trigger (flush with ≥ 4 SSTs, ≤ 8 per merge) is the
same coordinated path.

## 6. Invariants

INV-C1 no resurrection · INV-C2 latest version survives (resolution identical to
open, including ties) · INV-C3 tombstone GC only under the full-merge proof ·
INV-C4 id-map/retired-id preservation (checker bijectivity + export iss + attend
probes) · INV-C5 collection isolation (db-level merge preserves per-collection
state) · INV-C6 recovery never needs a removed SST (scan-based recovery) ·
INV-C7 atomic publication (old-or-new, never hybrid — crash-verified) · INV-C8
reader safety (in-memory readers + lock scoping) · INV-C9 writer safety (no lost
acked write vs sidecar model) · INV-C10 WAL/checkpoint_seq/watermark untouched.

## 7. Offline Control (E5a)

`c0-offline-control` (sync, 3 SST generations, 30 docs): pre-compaction state ==
independent model; compaction 3→1 SSTs, 18650→18610 bytes, 190 µs; full value-model
equality after; checker clean; fresh reopen equals model. MATCH.

## 8. Multi-Generation Version Resolution (E5b)

`multigen-version-resolution`: four overlapping generations built via
insert/checkpoint cycles including same-UUID updates (A: 100→200→300), two deletes
(B, C), two inserts (D, E). Expected state derived from the operation log, not
hard-coded. Result: exactly A=300, B absent, C absent, D, E — live AND after fresh
restart. No v-mixtures. MATCH (3→1 SSTs, 3573→1884 bytes).

## 9. Equal-Timestamp Resolution (E5b, deterministic)

**Bug found and fixed (see §26):** `do_compact` replaced a version only on
*strictly greater* timestamp, while `open_inner` breaks (equal-ts) ties toward the
LATER file; `append()` stamps milliseconds, so same-key versions flushed within one
millisecond collide. Under the old rule, a full merge could republish a stale value
or — worse — let a LIVE value beat a newer tombstone and then be GC-published as
"the" version (resurrection). Fix: `ts > cur || (ts == cur && file_idx > cur)` —
exactly `open_inner`'s rule. Regression test
`test_compaction_timestamp_tie_breaks_to_later_file` writes two equal-ts versions
plus an equal-ts live/tombstone pair, compacts all four files, and asserts the
later-file version wins and the tombstone wins + is GCed. The test FAILS under the
old rule (verified: the live value won the tie) and PASSES after the fix. Boundary:
engine-level same-millisecond collisions are timing-dependent and are covered by
the model-checked update chains of §8 (deviation D2).

## 10. Tombstone / Deletion Results (E5c)

- `tombstone-gc-basic`: ins→ckpt→del→ckpt→compact→restart: document absent, 1
  tombstone removed, checker clean. MATCH.
- `tombstone-deep-reinsert`: ins→upd(same UUID)→del→compact→restart (absent) →
  reinsert (fresh UUID v2) → compact → restart (present exactly once, v2).
  MATCH.
- `delete-reinsert-delete-x3`: three rounds of insert/delete/compact on the same
  logical key (fresh UUIDs) — absent after every round and after restart; 3
  tombstones reclaimed. MATCH.
- `partial-merge-retains-tombstones` (storage API): 5 SSTs, merge oldest 4 —
  `tombstones_removed = 0` and the tombstones are present in the OUTPUT (retention
  proven); full-merge-only GC stands. MATCH.

## 11. Reinsert / ID Mapping (E5d)

Covered by `tombstone-deep-reinsert` and `delete-reinsert-delete-x3`: fresh-UUID
semantics, retired-ID exclusion (checker idmap bijectivity + export `iss` empty +
attend probes), no stale pre-delete version, inverted-search correctness via the
checker. MATCH in all.

## 12. Repeated Compaction (E5e)

`repeated-compaction-x4`: four write→checkpoint→compact rounds (incl. a delete in
round 2), full value-model equality AND checker after EVERY round; per-round
diagnostics recorded (`r0..r3 ok/ok, files 1≤1, 2≤1, 2≤1, 2≤1`); fresh-restart
equality (19 docs). A compaction chain never corrupted the assumptions of the next.
MATCH.

## 13. Concurrent Readers (E5f, C-class availability)

`readers-during-compaction` (group mode): 3 threads × 200 attends run DURING a real
merge (2→1 SSTs, 10 tombstones removed, 654 µs). **0 errors, 604 measured attends,
p50 = 53 µs, p99 = 116 µs** — readers block for the merge window and then succeed
("blocked, not errored"). Final state = exact model (50 docs: 60 inserted, 10
deleted pre-compaction). Readers never observe invalid intermediate state (INV-C8).
MATCH.

## 14. Single Writer (E5g)

`writer-during-compaction` (group): pre-built 2-generation store with tombstones
(5000-namespace), live writer (200-namespace) throughout; compaction executes
against live writes; fsynced sidecar model (ACK/DEL lines) compared at full value
level: 182/182 exact. Max writer op latency 1815–2278 µs across runs (includes the
gate wait when the writer collided with the merge). No lost acked write, no stale
overwrite, no resurrection. MATCH.

## 15. Multi-Writer (E5h)

`multiwriter-during-compaction` (group): 3 writers × 60 ops + compaction mid-stream;
gate serializes writers (documented — no linearizability claim); sidecar model
equality 133/133; max op latency 1926–2061 µs. No acknowledged-write loss, no
corruption, no impossible state. MATCH.

## 16. Checkpoint Interaction (E5i)

`ckpt-then-compact` and `compact-then-ckpt` (both orders on the same database,
plus restart): catalog/checkpoint_seq/SST/WAL/wal-state.json stay consistent
(checker clean both times; model equality 20/20). Overlap case: the automatic
post-flush compaction inside `checkpoint_locked`'s flush is the existing overlap —
exercised by every multi-generation cell. MATCH ×2.

## 17. WAL Interaction (E5j)

`wal-rotation-during-compaction` (group, 2 KiB segments): rotations before/during/
after the merge; sidecar model equality 155/155; no sequence gap, no watermark
damage, no duplicate replay (restart-equality via the E1 machinery). E1 regression
after E5: PH3E-WAL-006 byte-identical to PH3E-WAL-001. MATCH.

## 18. Backup Interaction (E5k)

`bkp-after-compact` (compact→backup→restore) and `bkp-before-compact`
(backup→compact→verify-restore): backup always restores its own snapshot; compaction
never mutates the backup directory; a backup never depends on an SST that compaction
removed (restores verified from the copied backup dir, not from the source).
Gate serialization makes backup/compaction mutually exclusive by construction.
Full lifecycle `compact-backup-restore-restart-compact` (incl. compacting the
RESTORED database): MATCH. E4 regression: PH3E-BACKUP-005 classification-identical
(e4-integrity byte-identical).

## 19. Crash-During-Compaction (E5l)

Four crash gates added to `crashgate.rs` (+ allowlist): `compact_before_merge`,
`compact_after_output`, `compact_after_cleanup`, `compact_after_install`. Child
process builds three generations (insert/update/delete + checkpoints), then calls
`compact_storage()`; the configured gate parks the child (groupkill model, durable
marker); the controller SIGKILLs the process group at the window. All four windows
fired (marker-verified; deviation D5: no other failure injection exists — E11 not
started).

## 20. Fresh-Process Recovery Evidence (E5l/§22 of the spec)

After each kill, the CONTROLLER (a fresh process) opens the directory from disk:
**all four windows restart to exactly the same valid logical state** (updated idx1
present, deleted idx2 absent — never a hybrid), checker clean, zero `.tmp`
artifacts, and a SECOND compaction on the recovered store succeeds (model-clean).
The dangerous windows are covered explicitly: `compact_after_output` (new SST +
old SSTs + old reader list) and `compact_after_cleanup` (inputs physically removed
while the old reader list was active) — recovery requires no removed SST (INV-C6)
and accepts no half-published generation (INV-C7). In-process assertions in the
crashed child were never used as evidence (deviation D8).

## 21. Manifest / SST Retirement Safety (E5m + §30)

`manifest-fallback-after-compact`: corrupt CURRENT (→ missing generation) after a
compaction → the catalog falls back to the latest VALID generation, which contains
the compacted state (model equality 20/20, checker clean). Old-SST retirement is
safe because (a) readers materialize entries in memory at open, (b) the reader list
is swapped under the store write lock, (c) recovery scans the directory and resolves
by content — verified experimentally by the crash windows and by readers-during-
compaction. No reference counting needed; none added.

## 22. Partial Artifact Handling (E5n)

`partial-artifact-tmp`: garbage `.tmp` next to a compacted store → ignored AND
deleted by open (state = exact model; remaining tmp = 0).
`partial-artifact-garbage-sst`: garbage `.sst` → **REFUSED loudly**
(`RecoveryFailed: SSTable ... bad SSTable magic`) — arbitrary `.sst` files are never
silently accepted; the documented corruption policy stands. MATCH / REFUSED.

## 23. Automatic Compaction (E5o)

Existing trigger audited and preserved: every memtable flush checks the threshold
(≥ 4 SST files) and merges the oldest ≤ 8 files (tombstone GC only when the merge
covers all files). `auto-scheduler-observation`: 6× 1000-insert flushes bounded the
on-disk file count to [1..3] (auto-merges fired); explicit compaction → 1 file.
The trigger runs inside gate-held paths, so automatic compaction is C1 by
construction. No new scheduler built (spec §26/§5o). MATCH.

## 24. Blocking / Availability (§27/§28 of the spec)

Writers: pause for the whole compaction window (mutation gate) — bounded, measured
max op 1.8–2.3 ms at tested sizes; the deterministic gate-wait evidence is the b5-
style checkpoint wait analogue in E4 and the writer-latency maxima here. Readers:
block for the merge/install window under the store write lock, 0 errors, p50 53 µs
/ p99 116 µs at n=604. C1 ≠ C2: **no zero-pause claim** is made anywhere.

## 25. Performance (§27/§32 of the spec)

Compaction duration 190–660 µs at tested sizes (30–50 live docs, 2–5 input SSTs);
SST counts 3→1 / 2→1; physical bytes shrink with redundancy (41 442→31 102 with
tombstones; 3 573→1 884 multi-gen) and grow when a live writer flushes a large
memtable into the merge window (27 055→112 998 — the writer's own flush SST
dominates; noted, not normalized). Tombstones removed 1–10 per merge where present.
Single-run numbers on a 2-vCPU VM; no extrapolation, no optimization performed
(memory untouched — E9 scope).

## 26. Bugs Found and Fixed

1. **Equal-timestamp version-resolution divergence (latent INV-C1/C2 violation)** —
   `do_compact` strictly-greater rule vs `open_inner` (ts, file_idx) rule; fixed to
   the identical rule; regression test added (fails before, passes after; see §9).
2. **Results-generator defects (post-E4 discovery, D6):** `_gen_e4` mismatch counter
   passed by value (E4 mismatches could never fail the pipeline) and `gen_e5`
   returning row count instead of mismatch count. Fixed; generator fails on any
   mismatch. No recorded outcome changed (all recorded cells MATCH).
Harness (not engine) defects during development were fixed before registration and
are documented in deviations D1–D3.

## 27. E1/E2/E3/E4 Regression Results (new run IDs; historical raw untouched)

| Suite | New run | Result |
|---|---|---|
| E1 WAL integrity | PH3E-WAL-006 | byte-identical to PH3E-WAL-001 (11 cases, 6 refused / 5 opened) |
| E2 durability | PH3E-DUR-009 | ack-boundary ≡ DUR-001/008, txn-ack ≡ DUR-002/008, checkpoint-interaction ≡ DUR-003/008, group-boundary ≡ DUR-004/008 (byte-identical); mode-latency timing-only |
| E3 crash matrix | PH3E-E3-003 | e3-matrix.csv + metrics.json byte-identical to PH3E-E3-001/002 (210 cells) |
| E4 backup | PH3E-BACKUP-005 | e4-integrity.csv byte-identical; e4-matrix classification-identical (all 15 MATCH; timing/notes columns differ) |

## 28. Production Contract Amendment

**A5 — Compaction Semantics** appended to `methodology/production-contract.md` §5
(evidence warrants it): class C1 coordinated; writer pause bounded + measured;
reader block-not-error with measured percentiles; full-merge-only tombstone GC;
publication order + crash semantics (fresh-process verified); no WAL/checkpoint/
manifest interaction; backup serialization; scope ≤ tested sizes; **fully online
compaction = UNSUPPORTED**. Wording is no stronger than the evidence.

## 29. Limitations

Single-process local store; ext4/VM; ms-scale merges at ≤ 50 000 docs; single-run
timings (2-vCPU VM); equal-ms collisions proven at the storage API (deterministic)
with engine-level coverage via model-checked update chains (D2); compaction failure
injection limited to process-death crash gates (general fault injection = E11);
no incremental/background scheduling beyond the existing flush trigger.

## 30. Unsupported Claims

Fully online (C2) compaction — UNSUPPORTED. Zero writer pause — not claimed.
Linearizability of concurrent writers — not claimed (gate serialization documented).
Power-loss/machine-crash durability of compaction — never tested (A3 boundary).
Compaction performance at production scale — not measured (E10 scope). Memory
behavior — not optimized, not claimed (E9 scope).

## 31. Reproducibility

`phase3-bench dbtest e5run --out <dir>` (25 cells: 21 in-process + 4 crash windows
with fresh-process recovery). Raw: `research/phase3/raw/runs/PH3E-COMPACT-004/`
(+ superseded PH3E-COMPACT-003, preserved). Results via
`generate_results_ph3e.py` (fails on any mismatch); tables/figure via
`generate_tables_figs_e5.py`; enforced by `verify_consistency.py` gates 1–22
(including: no coordinated claim from offline-only cells, crash-window evidence,
refusal policy, registry presence). Registry: 83 experiments. Workspace: 316 tests
pass; `cargo clippy --workspace --all-targets -D warnings` clean; phase-2 and
phase-3 checkers both PASS.

## 32. Final E5 Verdict

**The strongest compaction guarantee AttentionDB has actually earned is B —
coordinated compaction with bounded writer pause** (with C-class reader
availability: concurrent readers continue, block briefly, and never error).
Offline compaction (A) remains verified as the control; fully online compaction
(D) is UNSUPPORTED by the mutation-gate architecture and was not implemented to
game the verdict; nothing partial or blocked (E) remains in the coordinated path.

| Capability | Status | Evidence | Boundary |
|---|---|---|---|
| Offline compaction | VERIFIED | c0-control + Phase-1 TEST 11 | quiesced DB |
| Coordinated compaction | VERIFIED | PH3E-COMPACT-004 25/25 | gate-held; ms-scale at tested sizes |
| Fully online compaction | UNSUPPORTED | gate architecture (§4) | not implemented, not claimed |
| Latest-version preservation | VERIFIED | multigen + sidecar model | full value model |
| Equal-timestamp resolution | VERIFIED | storage-API regression test (fails pre-fix) | API-level deterministic; engine-level model-checked |
| Tombstone correctness | VERIFIED | basic/deep/partial-retention cells | full-merge-only GC |
| No resurrection | VERIFIED | 0 RESURRECT across all cells + restarts | tested failure model only |
| Delete/reinsert correctness | VERIFIED | deep-reinsert + x3 loop | fresh-UUID semantics |
| ID-map correctness | VERIFIED | checker bijectivity + iss + attend probes | per exported state |
| Collection isolation | VERIFIED | collections-isolation cell (14/14 + 14/14) | db-level merge |
| Reader safety | VERIFIED | readers cell 0 errors; crash windows | block-not-error |
| Writer safety | VERIFIED | writer/multi-writer sidecar equality | gate serialization |
| Multi-writer behavior | VERIFIED | 3 writers, no loss vs model | no linearizability claim |
| Checkpoint interaction | VERIFIED | both orders + overlap via flush path | same-DB sequences |
| WAL rotation interaction | VERIFIED | 2 KiB segments + E1 regression identical | tested segment size |
| Backup interaction | VERIFIED | both orders + full lifecycle | gate serialization |
| Crash during compaction | VERIFIED | 4 windows, fresh-process recovery | instrumented windows only |
| Manifest/recovery safety | VERIFIED | corrupt-CURRENT fallback post-compact | scan-based recovery |
| Old-SST retirement | VERIFIED | crash windows + in-memory readers | no refcount needed (proven) |
| Repeated compaction | VERIFIED | x4 rounds, per-round model+checker | tested chain length |
| Automatic compaction | VERIFIED | trigger audited + bounded file count | existing ≥4/≤8 trigger preserved |

**E5 COMPLETE; E6 NOT STARTED.**
