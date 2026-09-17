# Phase 3E — E4 Final Report: Online Backup, Snapshot Consistency & Restore Integrity

Experiment: **PH3E-BACKUP-004** (supersedes PH3E-BACKUP-003, harness defect D1) ·
Date: 2026-09-17 · Repo: AttentionDB @ 2be1faf + E4 changes · Author: Rayyan Kaan

---

## 1. Executive Summary

AttentionDB can create a **consistent, restorable backup while the database is
actively serving reads and writers** — verified experimentally in 15/15 matrix cells
against an independent reference model, including concurrent readers, single- and
multi-writer contention, overlapping checkpoints, WAL rotation, and the Async
durability mode. The guarantee is **COORDINATED, not non-blocking**: `backup_to` holds
the engine mutation gate for the copy, so writers colliding with a backup pause for
its duration (measured 257 µs for a gate-seeking checkpoint at the tested size). A
fully online (zero-writer-pause) backup is **UNSUPPORTED and not claimed**. Restore
integrity is enforced by two new gates (completion marker + format version) plus the
existing destination/corruption policy: 10/10 integrity cases behaved as specified,
and a group-SIGKILL **inside** the copy produced a partial backup directory that
restore REFUSES while the pre-crash backup still restores and the source stays intact.

**Verdict (§24): A — Backup/restore VERIFIED as a coordinated snapshot mechanism;
non-blocking online backup UNSUPPORTED, documented as such.**

## 2. Scope and Non-Scope

**In scope:** E4 only — `backup_to`/`restore_backup` semantics, snapshot consistency
under concurrency, restore validation, crash-during-backup, backup corruption policy,
blocking measurement. Two minimal restore-validation code changes (§6).

**Out of scope (unchanged):** E5–E11, replication, sharding, Raft, Kubernetes, new
storage engines, HNSW/retrieval changes (architecture frozen), GPU paths, distributed
anything, machine power-loss claims (A3 boundary).

## 3. Central Question

*"Can AttentionDB create a consistent backup while the database is actively changing?"*
**Answer: YES — with a bounded writer pause.** Every backup taken during concurrent
activity restored to exactly the logical state at the backup boundary (independent
reference model, §9); no impossible mixtures of pre/post-backup state were ever
observed (§12).

## 4. Prior State (E0–E3 + PH3D-BACKUP-001/002)

E0–E3 closed: contract from code (A1), durability per mode + machine-crash boundary
(A2/A3), 210-cell crash-recovery matrix (all recover-all, byte-identical regressions
here: PH3E-WAL-005, PH3E-DUR-008, PH3E-E3-002). PH3D-BACKUP-001/002 (Phase 3D backup
existence/round-trip) untouched — no prior run IDs overwritten; E4 adds
PH3E-BACKUP-003 (superseded) and PH3E-BACKUP-004 to the family. Registry: 78 entries.

## 5. Implementation Audit (cond; full 20 answers in spec §1)

`backup_to` = acquire mutation gate → checkpoint **inside** the gate → copy CURRENT +
MANIFEST + sst/ + META/ + WAL/ → write `backup-meta.json` LAST (fsync_dir). SSTs are
immutable and uniquely named (`<timestamp>.sst`); flush/compaction are gate-serialized,
so the copied set is internally consistent. Backup WAL after the internal checkpoint =
empty active segment + `wal-state.json` (state lives in SSTs). **Pre-E4 restore gaps
found by audit:** (1) missing `backup-meta.json` silently synthesized a default meta —
a crash-truncated backup restored as if complete; (2) `backup_format_version` was never
checked. Both closed in E4 (§6).

## 6. Code Changes (minimal; invariants named)

1. **`restore_backup` refuses a backup directory without `backup-meta.json`**
   (`NO_META`). Invariant closed: *a backup is restorable only if it is provably
   complete* — the marker is written last+fsynced, so its absence proves a torn copy.
2. **`restore_backup` refuses `backup_format_version != 1`** (`BAD_VERSION`).
   Invariant closed: *unknown future formats are refused, never mis-parsed.*
Copy/checkpoint/gate paths untouched. Both changes are pure validation; no backup
validation was weakened anywhere (checker enforces this class of regression, §23).

## 7. Guarantee Classification

Per spec §3: **quiescent** (no activity) / **coordinated** (writers gated, bounded
pause) / **fully online** (zero interference). E4 result: quiescent VERIFIED (control),
**coordinated VERIFIED** (B2–B7), fully online **UNSUPPORTED** (gate design; pause
measured §17). The claim vocabulary is exactly this three-way split; "online backup"
without qualification is never used for a passing claim.

## 8. Snapshot Boundary Mechanism

Boundary = **mutation-gate acquisition**. Everything acked before the gate is in;
nothing acked after backup return is in; concurrent ops serialize against the gate.
The internal checkpoint fsyncs the WAL **inside** the gate, which is what lets the
Async mode capture acked-but-unfstateed writes (§16). Completion marker
(`backup-meta.json`) written last ⇒ a directory without it is provably incomplete.

## 9. Independent Reference Model & Methodology

Mandatory per prompt: a **fsynced per-op ACK sidecar** (append `ACK idx` / `DEL idx` /
`UPD idx idx'` lines, fsync per line) records exactly what the engine acknowledged.
Expected snapshot = replay of the sidecar **read immediately at backup return** —
never the final source state. Restore always into a fresh empty directory; every
restore opened by a fresh engine and passed the full consistency checker (necessary,
not sufficient — restored document sets are compared explicitly). ID mapping: reinserts
create fresh UUIDs; the checker's idmap bijectivity + retired-id exclusion covers
mapping integrity; uniqueness via inverted-search emptiness.

## 10. Experiment Design (B1–B7 + integrity + crash)

PH3E-BACKUP-004 = 15 matrix cells (B1 quiescent; B2 read-concurrent; B3 single
writer; B4 3-writer; B5 writer+checkpoint; B6 writer+WAL-rotation at 2 KiB segments;
B7 writer+checkpoint+rotation under Async; snapshot v1→v2 transition; delete/reinsert;
3-collection isolation; multi-backup chain; durability-mode ×3; crash-during-backup)
+ 10 integrity cases. Harness: `phase3-bench dbtest e4run`; crash child = separate
process group, SIGKILL 500 µs after backup starts on a 20 000-doc DB (D2).

## 11. Results: B1–B7 Matrix (raw: e4-matrix.csv)

| Cell | Mode | Snapshot | Restored | Checker | Note |
|---|---|---|---|---|---|
| b1-quiescent | sync | 30/30 | MATCH | clean | 357 µs backup, 1.75 ms restore |
| b2-readers | group | 40/40 | MATCH | clean | **0 reader errors** during backup |
| b3-single-writer | sync | 300/300 | MATCH | clean | source 400 (post +100 excluded); max op 192 µs |
| b4-multi-writer | group | 300/300 | MATCH | clean | 3 writers |
| b5-writer-ckpt | sync | 200/200 | MATCH | clean | checkpoint waited **257 µs** on gate, then succeeded |
| b6-writer-rotation | group | 200/200 | MATCH | clean | rotations before/during(internal ckpt)/after |
| b7-writer-ckpt-rotation | async | 200/200 | MATCH | clean | all composites together |

15/15 SNAPSHOT_MATCH; every restore checker-clean.

## 12. Results: Snapshot-Transition (no impossible mixtures)

Six backups taken while documents flip v1→v2: every restore is a **pure** v1-or-v2
state (6/6 MATCH) — never a mixture of old and new values. This is the direct
test of "ONE valid logical state, never SST@T1+manifest@T2+WAL@T3".

## 13. Results: Delete/Reinsert + ID Mapping

20 inserts → 10 deletes → backup → 3 reinserts (fresh UUIDs). Boundary sidecar = 33
lines; model = 13 docs; restored = **13/13 MATCH**, checker-clean (idmap bijective,
retired ids excluded, uniqueness holds). Post-backup source churn (+5 delete/reinsert)
stayed out of the backup.

## 14. Results: Multi-Collection Isolation

`alpha`/`bench`/`beta` written concurrently, backup mid-stream: 80/80 per collection
MATCH, **no cross-collection contamination** (each collection's restored set equals
its boundary model).

## 15. Results: Post-Backup Mutation Isolation & Multi-Backup

Backups taken at three sizes (20/40/60 acked docs across quiescent/read-concurrent/
writer phases) each restore independently to their own boundary state (B1=20/20,
B2=40/40, B3=60/60): backup N is not disturbed by backup N+1, and source mutations
after backup N never leak into it.

## 16. Results: Durability-Mode Matrix (Async composition)

300 acked writes per mode, backup at the end: sync 300/300, group 300/300,
**async 300/300** — every acked write captured in all three modes. Mechanism: the
backup's internal checkpoint fsyncs the WAL inside the gate, so Async's buffered acks
become durable *inside the backup*. This is the experimentally verified mechanism the
prompt requires before Async can be claimed captured; it does not alter A2/A3
(an Async ack alone still implies only "committed").

## 17. Blocking Measurement (pause / throughput)

Backup durations at tested sizes: 0.36–1.54 ms (small DBs), gate-held for the whole
copy. Deterministic pause evidence: b5's concurrent checkpoint waited **257 µs** until
backup completion, then proceeded normally. b3's writer showed max op latency 192 µs
with 0 ops straddling the window this run (D4 — no straddle latency is claimed from
that row). Reads are never blocked (0 reader errors; read path does not take the
mutation gate). **No zero-pause claim is made.**

## 18. Results: Crash During Backup

Child process seeds 20 000 docs, starts `backup_to`, and the controller SIGKILLs the
whole process group **inside** the copy (done=false, marker never written):
- **Source: intact and reopenable — 20 000/20 000, checker clean.**
- **Partial backup directory: REFUSED on restore** (`NO_META` — new gate).
- **Pre-crash backup in the same engine: still restores exactly** (early-backup control).
Exact tested failure model: process-group death mid-copy with page cache intact (A3
boundary; no power-loss claim).

## 19. Results: Restore Integrity Suite (raw: e4-integrity.csv)

| Case | Action | Result |
|---|---|---|
| valid-control | none | ACCEPTED, clean, state verified |
| partial-no-meta | rm backup-meta.json | **REFUSED NO_META** (E4 gate) |
| truncated-sst | halve first SST | REFUSED (catalog/open) |
| corrupt-wal-state | garble wal-state.json | REFUSED (E1 policy) |
| garbage-active-segment | append zeros to empty segment | ACCEPTED — torn-tail policy, zero loss (D5) |
| corrupt-current | CURRENT → missing gen | ACCEPTED — fallback gen intact; **restored state verified** (D5) |
| malformed-meta | invalid JSON | REFUSED META_PARSE |
| bad-format-version | v99 | **REFUSED BAD_VERSION** (E4 gate) |
| nonempty-dest | restore over existing | REFUSED DEST_NONEMPTY |
| source-after-backup | open source post-backup | INTACT (30/30) |

10/10 as specified. No validation weakened to accommodate the implementation.

## 20. Manifest Sufficiency

No correctness-necessary manifest fields were added. The two E4 changes are
*validation* of existing fields (marker presence, `backup_format_version==1`); the
manifest content itself was already sufficient for the verified semantics.

## 21. SST Immutability & Format Versioning

Proven from implementation (spec §1): SSTs are written once at unique
`<timestamp>.sst` names and never edited; flush and compaction hold the same mutation
gate as backup, so the copied set cannot be mutated mid-copy — the structural basis of
snapshot consistency. Format versioning: v1 enforced; unknown versions refused
(§6/§19).

## 22. Threats to Validity / Blocked Categories

Single-process, local-disk (ext4 on VM), ≤ 20 000 docs, µs-scale backups — pause
figures do not extrapolate to large DBs (linear in copy size; not measured at scale).
Blocked: remote/incremental backups, backup-of-backup chains, filesystem-disruption
and power-loss axes (A3), true non-blocking backup (unsupported by design, §7).
b2/b4/b6 run under group mode (D7); the full mode dimension is covered by §16's
dedicated cells. Timing numbers are single-run (raw immutability); classifications are
deterministic.

## 23. Deviations

All documented in `methodology/ph3e-e4-deviations.md`: D1 supersede 003→004 (harness
column shift), D2 crash-child calibration, D3 boundary-capture order fix, D4
no-fabricated-straddle rule, D5 accepted-by-policy integrity cases with restored-state
verification, D6 output-identical post-run clippy fixes, D7 mode rotation, D8 crash-row
column semantics. The phase-3 consistency checker (gate 21) now rejects: E4 matrix
violations, missing concurrent-backup classes (an "online" claim from quiescent-only
cells), integrity mismatches, unregistered/missing raw runs, and absent report/deviations.

## 24. Final Capability Matrix (14 rows)

| # | Capability | Status |
|---|---|---|
| 1 | WAL durability — Sync (ack ⇒ fsynced record) | VERIFIED (E1–E3) |
| 2 | WAL durability — GroupCommit (ack ⇒ page-cache frames) | VERIFIED (E2/E3) |
| 3 | WAL durability — Async (ack ⇒ committed; loss allowed/observed) | VERIFIED LOSS per contract (E2/E3) |
| 4 | Process-crash recovery (F1, all modes/structural windows) | VERIFIED (E3; 210 cells) |
| 5 | Container/VM abrupt termination (F2E, strongest available) | VERIFIED (E3) |
| 6 | Filesystem disruption (F3) | BLOCKED (no block-device tooling) |
| 7 | Physical power loss (F4) | BLOCKED / NOT VERIFIED (never simulated) |
| 8 | Checkpoint (all modes × instrumented windows) | VERIFIED (E2/E3) |
| 9 | WAL segment rotation | VERIFIED (E3) |
| 10 | Corrupted-WAL refuse-or-warn + torn-tail policy | VERIFIED (E1) |
| 11 | **Backup = consistent snapshot under concurrent activity (coordinated)** | **VERIFIED (E4; 15/15)** |
| 12 | **Restore integrity (marker/version/corruption/dest refusal; crash-truncated rejected)** | **VERIFIED (E4; 10/10 + crash)** |
| 13 | Fully non-blocking online backup (zero writer pause) | **UNSUPPORTED** (gate-held copy; pause measured) |
| 14 | Machine durability of backups beyond process-death model | NOT VERIFIED (A3 boundary; out of scope) |

## 25. Final Verdict, A4 Decision & Reproducibility

**Verdict A — Backup/restore verified as a coordinated snapshot mechanism with
enforced restore integrity; non-blocking online backup explicitly UNSUPPORTED.**
A4 **earned and appended** to `methodology/production-contract.md` §5 (backup/restore
contract: boundary semantics, coordination class, Async composition, restore gates,
format, tested-scope boundary). The prompt's fallback sentence ("online backup =
UNSUPPORTED; quiescent = VERIFIED") is NOT used because the stronger coordinated
guarantee was actually demonstrated — while its non-blocking limitation is recorded
with equal prominence.

Reproducibility: `phase3-bench dbtest e4run --out <dir>`; raw at
`research/phase3/raw/runs/PH3E-BACKUP-004/` (+ superseded PH3E-BACKUP-003; regressions
PH3E-WAL-005, PH3E-DUR-008, PH3E-E3-002 all byte-identical to their E1/E2/E3
baselines). Results via `generate_results_ph3e.py` (transcription-free); enforced by
`verify_consistency.py` gates 1–21. Registry: 78 experiments. Workspace tests: all
pass; `cargo clippy -D warnings` clean.

**E4 COMPLETE; E5 NOT STARTED.**
