# Phase 3E — E2 Final Report: Durability Semantics & Acknowledgment Contract

Date: 2026-09-17 · Base commit: 615407c (E0+E1) · Scope: **E2 ONLY** (no E3+ started)

---

## 1. Executive summary

E2 establishes, by code audit plus 192 exactly-instrumented crash cells, what an
acknowledgment means in each durability mode:

- **Sync** — *acknowledged ⇒ durable (machine boundary)*: the WAL frame is fsynced
  before the API returns. Acked writes survived **every** crash cell (0 losses / 81).
- **GroupCommit** — *acknowledged ⇒ durable (process boundary)*: the frame is flushed
  to the OS page cache before the API returns. Survives process death; NOT machine
  durable. Despite the name there is **no grouping/coalescing** (the engine-wide
  mutation gate serializes all mutations; every append flushes). Acked losses: 0/81.
- **Async** — *acknowledged ⇒ committed only, NOT durable*: acked writes and even
  whole acked transactions were **repeatedly observed to vanish** after process death;
  loss is bounded only by structural durability points (rotation / checkpoint / close /
  flush_wal). This is its documented contract, now experimentally characterized — not
  hidden.

An acknowledged transaction never recovers as a partial transaction (0 PARTIAL in 63
txn cells). The production-contract amendment **A2** records the vocabulary
(Committed / Durable / Acknowledged), the per-mode contract, and the default-mode
recommendation (**GroupCommit**). The E1 WAL-integrity invariant is untouched
(PH3E-WAL-002 byte-identical). Machine/power-loss durability: **NOT VERIFIED** (E3).

## 2. Scope

E2 only: durability-mode implementation audit, acknowledgment semantics, ACK-boundary
and transaction-durability experiments, checkpoint/WAL-rotation interaction, multi-
restart stability, contract amendment, default-mode decision. Not started: E3 (machine
crash), E4–E11. The smallest necessary code change was instrumentation
(`storage/src/crashgate.rs` + 9 gate hits) — see `methodology/ph3e-spec-deviations.md`
§1. No durability-mode semantics were changed: the audit found the implementation
already matched its documented design; E2's job was to *prove and codify* it.

## 3. Existing implementation audit (the real timeline)

Traced for insert / delete / transaction commit (code refs: `storage/src/wal.rs`
`Wal::append`, `core/src/engine.rs`):

```
T0  mutation_gate acquired (engine-global; serializes ALL mutations AND checkpoints —
    appends never overlap; this is WHY there are no commit "groups")
T1  validate + id registration (idmap; durable only via later snapshots)
T2  wal_append → Wal::append:
      T2a seq assign; frame = magic|len|bincode(body)|crc32
      T2b rotate-if-full: flush + fsync of the COMPLETED segment  [mode-independent]
      T2c open_segment-if-needed: wal-state.json tmp→rename→fsync [mode-independent]
      T2d BufWriter::write_all(frame)        [8 KiB USERSPACE buffer]
      T2e mode action:
            Sync        → flush() + sync_all()   [machine-durable boundary]
            GroupCommit → flush()                [OS page cache = process boundary]
            Async       → (nothing)              [userspace buffer only]
T3  in-memory apply (DocumentStore/HNSW/BM25; txn: all ops AFTER the COMMIT record)
T4  return Ok  →  ACK (T4 always follows T2e: ack never precedes the mode action)
```

Structural durability points independent of mode: **rotation** (completed segment
fsync), **checkpoint** (WAL fsync → memtable→SSTables → idmap snapshot → manifest →
rotate → trim), **close** (= checkpoint). `flush_wal()` = page-cache flush only (no
fsync). Replay order guarantees COMMIT-record atomicity (BEGIN → ops → COMMIT are
written sequentially through one `BufWriter`; a torn tail can only truncate the tail
frame, so a discarded COMMIT discards the whole transaction — partial recovery is
structurally unreachable).

## 4. Durability vocabulary (contract A2)

- **Committed** — accepted through the logical commit path (applied + WAL record(s)
  appended; for txns: COMMIT record appended).
- **Durable** — the selected mode's durability mechanism completed (Sync: fsynced;
  GroupCommit: page-cache-flushed; Async: none beyond the userspace buffer).
- **Acknowledged** — the API returned `Ok` (always after both T2e and T3).

Equivalences established: Sync ACK⇒Durable(machine boundary); GroupCommit
ACK⇒Durable(process boundary); Async ACK⇒Committed only.

## 5. Sync semantics

1. Committed at T2d; durable at T2e (`sync_all` returned); ack at T4.
2. Process death immediately after success: everything acked recovered (27/27 relevant
   cells; also the pre-existing 3D crash family, 7/7 legs at sync... see §15).
3. Acked data disappearing: never observed in any mode at any gate (0/81 ack-boundary
   cells; 0/63 txn cells; 0/18 group cells; 27/27 checkpoint cells).
4. Across rotation: rotate fsyncs the completed segment (redundant with per-append
   fsync). Across checkpoint: checkpoint fsync + SSTables.
5. NOT guaranteed: machine/power-loss survival is DESIGNED for (fsync) but NOT
   VERIFIED until E3 (N6 unchanged).

## 6. GroupCommit semantics

1–2. Committed T2d; durable T2e (page cache); ack T4. **No coalescing, no group
   boundary, no background flusher** — each append's `flush()` is its own boundary;
   writers are serialized by the mutation gate, so no cross-writer buffering exists.
3. Process death after success: acked writes recovered (after_flush gate: frame in
   page cache, SIGABRT → recovered every rep).
4. One writer's failure affects another: N/A structurally (serialization); verified
   concurrently in PH3E-DUR-004 (3 threads, 75 acks, kill -9 → 0 lost).
5. NOT guaranteed: machine-crash survival (page cache lost on power failure).
6. Name retained; real semantics documented (deviations §8).

## 7. Async semantics

1. Committed T2d; durable: NEVER by mode (only via structural points).
2. Process death after success: **acked writes CAN disappear** — observed at the
   `after_ack` gate (single insert: lost in 3/3 reps), whole acked txn lost (3/3), and
   4/75 acked multi-writer writes lost (buffer tail).
3. Loss shape: NOT a clean prefix — it is "everything still in the userspace 8 KiB
   BufWriter" at death; with larger workloads the buffer auto-flushes at capacity
   boundaries, so survival of earlier acked records is an artifact (txn suite showed
   5/5 baseline survival from this; ack-boundary showed 0/10). Deterministic per
   workload, but an implementation artifact — contract says it MUST NOT be relied on.
4. Structural points bound the loss: rotation → completed segments only (27/30
   preserved at 2 KiB segments — exactly the E1 watermark boundary); checkpoint/close →
   everything; flush_wal → page cache (`explicit_flush` cell: acked write preserved).
5. Verdict: acceptable *documented* tradeoff, not an API-contract violation — the mode
   is explicitly named Async and A2/N5 make the loss semantics unmistakable. It is NOT
   a safe default for production (§13).

## 8. ACK-boundary experiments (PH3E-DUR-001, 81 cells)

Gates (exact injection points, SIGABRT in-process): `before_wal_append`, `after_write`,
`after_flush` (GroupCommit branch only), `after_fsync` (Sync branch only),
`after_wal_append`, `after_apply`, `before_ack`, harness-level `after_ack`,
`explicit_flush` (Async + flush_wal). 3 modes × 3 reps × 9 gates. Facts: recovered
state + ack sidecar + full checker. 6 cells statically NOT_REACHED (gate lives in
another mode's branch — deviations §3). Key rows (3/3 reps each):

| mode | gate | acked baseline | target (unacked unless noted) |
|---|---|---|---|
| sync | before_wal_append | 10/10 | ABSENT |
| sync | after_write | 10/10 | ABSENT (frame in userspace buffer) |
| sync | after_fsync | 10/10 | PRESENT (fsynced) |
| sync | after_ack | 10/10 | PRESENT |
| group | after_write | 10/10 | ABSENT |
| group | after_flush | 10/10 | PRESENT (page cache survived SIGABRT) |
| group | after_ack | 10/10 | PRESENT |
| async | before_wal_append | **0/10** | ABSENT |
| async | after_ack | **0/10** | **ABSENT — ACKED WRITE LOST** |
| async | explicit_flush | 10/10 | PRESENT |

No resurrection (extra_count=0) and clean checker in 81/81 cells.

## 9. Transaction durability (PH3E-DUR-002, 63 cells)

TXN = 8 inserts + 2 deletes of baseline docs; 7 crash points. All-or-nothing judged on
txn inserts (deviations §4); 0 PARTIAL cells. Landmarks:

- **sync/group `commit_written`** (COMMIT frame written, pre-flush/fsync): txn ABSENT
  even though the ops were already fsynced/flushed — the atomicity boundary (COMMIT
  record present) and the durability boundary (mode action) are DIFFERENT points, and
  crash between them discards the whole transaction. Exactly the safe behavior.
- **sync/group ≥ COMMIT durable**: COMPLETE (inserts + deletes), incl. `after_ack`.
- **async pre-checkpoint**: txn ABSENT at every gate INCLUDING `after_ack` (acked
  transaction vanished whole) — documented Async contract.
- **async `ack_ckpt`** (ack → checkpoint → abort): COMPLETE in all modes — the
  structural point makes even Async durable (high_watermark=31 recorded).

## 10. Checkpoint / WAL-rotation interaction (PH3E-DUR-003 + superseding PH3E-DUR-006, 27 cells)

Write 30 acked docs → structural point (checkpoint / checkpoint+trim / rotation via
`ATTENTIONDB_WAL_SEGMENT_BYTES=2048`) → acked target insert → SIGABRT → reopen →
restart ×2 (two close/open cycles per cell — PH3E-DUR-006, which supersedes 003's
single-cycle restart evidence after a harness loop defect was caught by the clippy
all-targets pass; 003 is preserved unchanged per raw-run immutability). Results:
baseline 30/30 in ALL modes after checkpoint/checkpoint+trim;
rotation-only: 30/30 in sync/group, **27/30 in async** (3 active-segment records lost;
survivors = watermark−1 exactly — completed segments only, matching the E1 boundary);
target: PRESENT in sync/group, ABSENT in async (post-point append). restarts_equal=true
and restart checker clean in 27/27 across BOTH cycles (PH3E-DUR-006). E1 invariants
verified at every open.

## 11. Recovery behavior / multiple restarts

Recovery is replay + (torn-tail truncation | refusal); the checker ran on every open of
every cell (189 spawned cells + restarts): 0 errors. Restart twice more per PH3E-DUR-003
cell: state is a fixed point — no recovery-induced mutation creates new logical state.
Unacknowledged in-flight writes: never observed surviving (0 in 189 cells) though the
contract allows it between T2 and T3.

## 12. Crash matrix (summary — full facts in results/durability-*.csv)

| capability | sync | group | async |
|---|---|---|---|
| acked single write survives process crash | SUPPORTED+VERIFIED | SUPPORTED+VERIFIED | **NOT — loss VERIFIED** |
| acked txn survives process crash whole | SUPPORTED+VERIFIED | SUPPORTED+VERIFIED | NOT — loss VERIFIED (checkpoint restores) |
| unacked write never survives | observed 0/189 (allowed by contract) | same | same |
| acked txn partial recovery | impossible (0/63) | impossible (0/63) | impossible (0/63) |
| structural point ⇒ durable | VERIFIED | VERIFIED | VERIFIED |
| machine/power-loss | NOT VERIFIED (E3) | NOT VERIFIED (E3) | NOT VERIFIED (E3) |
| gates unreachable in mode | after_flush NOT_REACHED | after_fsync NOT_REACHED | both NOT_REACHED |

## 13. Production-default decision

**Recommendation: GroupCommit** as the production-oriented mode; Sync where no acked
write may ever be lost before machine failure; **Async never as a production default**
(acked-write and acked-txn loss are demonstrated, not hypothetical). Rationale:
data-loss semantics (GroupCommit = process-crash-safe acks), latency/throughput parity
on the test FS (PH3E-DUR-005: mean ack 262 µs group vs 259 µs async vs 264 µs sync —
~3800 ops/s all modes; sandbox caveat), atomicity equal across modes, recovery equal,
API clarity (A2). The engine takes NO implicit default — `open_dir` requires an
explicit choice; the recommendation lives in the contract and docs.

## 14. Contract amendment

**A2** appended to `methodology/production-contract.md` (vocabulary, per-mode contract,
structural points, txn boundaries, API decision [no API change — Q10 answered as
documentation-level distinction], default recommendation, limitations, experiment IDs).
N5 superseded (Async loss now OBSERVED, not merely unsupported-by-design). A1 unchanged
(E1 invariants re-verified: PH3E-WAL-002 byte-identical to PH3E-WAL-001).

## 15. Regression testing

- Unit/integration (ALL targets this time — 262 tests across core/storage/query/
  multihead/learned incl. every integration-test binary): all pass. clippy
  `--all-targets -D warnings` clean. (api crate excluded: unbuildable without protoc —
  documented since E0; not in E2 scope.)
- The full-targets sweep exposed three PRE-EXISTING stale test expectations
  (proven failing at clean HEAD with E2 stashed; deviations §12–13): t11/golden
  `compact_all` db-root contract drift (silently no-op'ing compaction in golden),
  t14's unsorted WAL-dir file pick, and t13's pre-E1 manifest-fallback expectation —
  now updated to the E1 contract with the original failures documented here.
- 3D families (PH3E-REG-002): model 6/6, filterx 14/14, integration 16/16, crash legs
  group-durability 7/7 (acked intact, no resurrection, txn atomic).
- E1 matrix (PH3E-WAL-002): byte-identical to PH3E-WAL-001.

## 16. Known limitations

- Process-crash axis only (SIGABRT/SIGKILL). Machine crash, power loss, FS-level
  torn-page behavior: NOT VERIFIED — E3.
- Latency figures are sandbox-FS-specific; fsync cost here (~equal across modes) is
  NOT representative of production NVMe/_network FS; the relative decision may shift
  with real hardware and should be re-checked at E10/E12.
- `before_wal_append` is instrumented at the engine call site (after validation/id
  registration, before any append) — the nearest meaningful point (deviations §9).
- No gate inside `sync_all` itself; bracketed on both sides instead (deviations §9).
- `mid` group variant kills after all 75 acks had landed (writers faster than the poll)
  — the acked⊆recovered invariant is unaffected; noted for honesty.

## 17. Unsupported claims (explicitly NOT made)

No machine-loss or power-loss durability claim for ANY mode; no "fsync ⇒ durable on
real hardware" claim; no latency/throughput superiority claim; no linearizability,
serializability, exactly-once, or ACID vocabulary; no claim that Async loss is bounded
in size (only that structural points bound it); no claim that unacked writes never
survive (only that none were observed).

## 18. Reproducibility

`phase3-bench dbtest durability --suite {ack-boundary,txn-ack,checkpoint-interaction,
group-boundary,mode-latency} --out <dir>` with `PH3D_DURABILITY` unset (all modes).
Raw: `research/phase3/raw/runs/PH3E-DUR-001..006/` (+ PH3E-WAL-002, PH3E-REG-002);
tables: `results/durability-*.csv` generated by `generate_results_ph3e.py` (raw facts
in, expectations applied, MISMATCH → exit 1 — rerunnable, no hand-typed numbers);
checker gate 19 in `verify_consistency.py`. Every run dir carries run_info.txt +
config.json with the commit hash.

## 19. Final E2 verdict

| Capability | Status | Evidence | Boundary |
|---|---|---|---|
| Sync acknowledgment durability (process) | VERIFIED | PH3E-DUR-001/002/004 (0 losses) | machine axis → E3 |
| GroupCommit acknowledgment durability (process) | VERIFIED | PH3E-DUR-001/004 (0 losses) | no coalescing (name); machine axis → E3 |
| Async semantics | VERIFIED (documented loss) | PH3E-DUR-001/002/003/004 | acked data CAN vanish; bounded by structural points |
| Transaction durability (all-or-nothing under crash) | VERIFIED | PH3E-DUR-002 (0 PARTIAL/63) | — |
| ACK→durable relationship (contract A2) | VERIFIED | audit + DUR-001 gates | — |
| Crash recovery integrity | VERIFIED | checker clean 189/189 cells | — |
| Checkpoint interaction | VERIFIED | PH3E-DUR-003/006 (30/30 all modes) | — |
| WAL rotation interaction | VERIFIED | PH3E-DUR-003/006 (27/30 async = E1 watermark boundary) | active segment in Async |
| Repeated-restart stability | VERIFIED | PH3E-DUR-006 (27/27 across 2 cycles) | — |
| GroupCommit multi-writer ack invariant | VERIFIED | PH3E-DUR-004 (3 writers, kill -9) | single engine process |
| Machine-crash durability | NOT VERIFIED | — | E3 |
| Power-loss durability | NOT VERIFIED | — | E3 |

**E2 COMPLETE. STOP — E3 not started, per the staged-batch model.**
