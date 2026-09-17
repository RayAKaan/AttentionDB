# Phase 3D Final Report — Production Database Validation

Date: 2026-09-17 · Author: Rayyan Kaan · Base commit: e8ce38d (Phase 3D work tree) ·
Contract: `methodology/database-guarantees.md` · Raw: `raw/runs/PH3D-*` (22 run dirs,
incl. 4 registered child runs mapping spec §3 IDs to delivered families) ·
Results: `results/*.csv` · Tables 1–8 (+ extras 9–11) in spec §41 numbering ·
Figures ph3d-1..3 in spec §42 semantics · Deviations:
`methodology/ph3d-spec-deviations.md` · Both consistency checkers PASS.
Audit closure added PH3D-INTEGRATION-001, PH3D-CONC-003, PH3D-BACKUP-002.

**Verdict in one sentence:** within the tested envelope (single node, ≤ ~1800 live docs,
DIM 32, 4-second concurrency windows, process-crash axis), AttentionDB behaves as a correct,
recoverable, all-or-nothing-transactional document store with explicit, verified durability
modes — and it is *not* production-ready in the operational sense (no online backup/compaction,
no isolation levels, Async ack loss, one undetectable corruption case, untested machine-crash
axis), every one of which is documented rather than hidden.

## Correctness (§49 Q1–6)

1. **Does the engine maintain exact logical state under mixed workloads?** Yes — 3 × 1000-op
   + 1 × 100-op seeded runs (upsert/update/delete/query/filter/checkpoint/compact/restart):
   observed state == reference model after every restart and compaction gate
   (PH3D-STATE-001..004; 0/24 gated checks failed).
2. **Can a deleted document be returned?** Never observed: deletes vanish immediately
   (`delete_immediate`), `inv_live_only` and `final_dead_not_retrievable` hold everywhere;
   compaction tombstone GC keeps them gone (`deleted_stay_deleted`). Retired index vectors
   linger as documented WARNINGs until purge (lazy tombstones).
3. **Do filters ever return excluded documents?** Not once (~380 probes + 12 filterx
   checks): soundness held at every shape, zero-match always empty. Completeness is
   candidate-bound by design: measured recall 0.967–1.0.
4. **Is retrieval deterministic?** Yes — repeated queries identical (`query_determinism`,
   `filter_determinism`), which the frozen Phase 2 architecture requires.
5. **Do collections leak into each other?** No (`collection_isolation`); an early +1-doc
   artifact was a harness export bug (fixed), not an engine leak.
6. **Were any correctness bugs found and fixed?** Yes, two product fixes: (a) `compact_all`
   scanned the DB root instead of `db_dir/sst` — offline compaction was an unreachable
   silent no-op; fixed + regression test + PH3D-COMPACT-001; (b) `check_db_dir` flagged the
   normal post-checkpoint trimmed WAL as `WAL_SEQ_INVALID` — replaced by a precise gap
   invariant that still errors on genuine over-trim. Both preserve-or-close named invariants.

## Durability (Q7–9)

7. **Do acknowledged writes survive crashes?** Per selected mode: GroupCommit 7/7 crash
   points ALL_ACKED; Sync 7/7 (process axis); Async yields a proper prefix while unflushed
   (35/40, 11/21) and ALL_ACKED after flush/checkpoint (PH3D-CRASH-001..003, 21/21
   contract-consistent, zero resurrection).
8. **What exactly does a commit acknowledgment mean?** Under Sync/GroupCommit: the txn (and
   prior acks) survive process death. Under Async: an acknowledged commit may vanish with
   the process — observed as `COMMITTED_NOT_DURABLE_ASYNC` (all-or-nothing: 0/10, never
   partial). This is now an explicit, run-backed contract statement.
9. **Is machine-crash (power-loss) durability verified?** Only PARTIAL: fsync-per-append is
   implemented and exercised on the process axis; no power-cut/VM-kill harness ran.

## Recovery (Q10–13)

10. **What does open do with a torn/corrupt WAL?** Torn tail → truncate with WARNING, intact
    prefix recovers (23/40 after driver surgery). Corrupt frame (garbage tail, byte flip) or
    gapped segment name → **refuses to open**. No silent false recovery, no fabricated data.
11. **Is replay idempotent / bounded?** Replay re-applies through idempotent apply paths;
    every restart-equivalence gate (≈30/run × 4 runs) is an implicit double-replay check
    with equal observed state.
12. **How does checkpoint interact with the WAL?** Checkpoint installs SST+idmap+manifest
    (two-phase, crash-safe), rotates the WAL, trims fully covered segments; the empty
    post-checkpoint WAL is legitimate — and was the exact case the old checker mis-flagged.
13. **What corruption is undetectable?** Deleting the only pre-checkpoint WAL segment: the
    DB opens as empty with a clean checker (no WAL high-water mark in the catalog). Documented
    OPEN limitation; no mitigation attempted in 3D. The spec's 10 abstract crash points map
    to the 7 implemented points (mid_inserts covers the append/apply/index windows);
    pre-append and mid-checkpoint/mid-compaction windows are not separately instrumented —
    mapping in `methodology/ph3d-spec-deviations.md`.

## Transactions (Q14–17)

14. **Are transactions all-or-nothing?** Yes — multi-op commits exact; injected
    pre-validation failure rejects the whole commit (no partial apply); SIGKILL at the
    commit boundary yields 10/10 or 0/10, never partial, across all durability modes.
15. **Do rollbacks leak?** No — staged ops never visible; txn unstaged.
16. **What is the transaction scope?** `Insert|Delete` ops, single collection, delete-if-
    present no-op semantics. **No update op, no isolation levels, no cross-collection txns.**
17. **Are isolation/serializability claims justified?** No such claim is made; mutations
    serialize on a single gate; no history-based test ran (§25 discipline).

## Compaction (Q18–20)

18. **Does offline compaction preserve state?** Yes — PH3D-COMPACT-001: 310 docs across 2
    SST generations with 150 updates + 50 deletes: state equal after merge+GC, updated
    content survives at latest version, deleted docs stay deleted, checker clean.
19. **What about compaction under load?** Not supported online: `compact_all` is dir-level
    and uncoordinated; invoking it while an idle engine holds the dir merges files but is
    never cleaned up (PH3D-COMPACT-002, documented). Under load → BLOCKED by design.
20. **Does automatic compaction risk resurrection?** Partial merges retain tombstones
    (anti-resurrection INV-12); only full `compact_all` GCs — verified by code and the
    post-compact `deleted_stay_deleted` check.

## Concurrency (Q21–24)

21. **Do parallel readers scale?** To a plateau: 13.5K (r1) → 21.6K (r2) QPS, ~19–22K for
    r4–r32; zero errors; checker clean at every rung (PH3D-CONC-001).
22. **What are the tail characteristics?** p50 flat (68–93 µs); p99 grows 128 µs (r1) →
    60 ms (r32). Reported unsmoothed.
23. **Is mixed read/write stable?** Yes — 6 combos, deterministic writer cycles (ins→upd→
    del→flush): 0 errors, checker-clean state, live-doc counts tracked; per-writer op logs
    recorded (PH3D-CONC-002). Audit closure adds PH3D-CONC-003: the merged deterministic
    op logs REPLAY to the exact observed state (450 docs) and same-key contention
    (100 keys × 150 rounds × 2 writers) never produced a torn record — per-key
    last-write-wins under the mutation gate, documented; no linearizability claim.
24. **Is there a linearizability claim?** No. Gate serialization measured (r1w1 p50 ≈
    0.75–0.83 ms vs r4w1 ≈ 65 µs); no history-based ordering test; op logs retained to
    enable one later.

## Backup (Q25–27)

25. **Does restore reproduce the backup exactly?** Yes — 150-doc capture, post-backup
    mutations on the original (insert 150, delete 1): restored == backup state, mutation-
    isolated, unique ids, clean checker, manifest present (PH3D-BACKUP-001 6/6).
26. **Is there online backup?** No. Copies are uncoordinated with the mutation gate; the
    live-idle copy restored checker-clean; the copy-under-active-writers sample also opened
    clean (665 docs written during capture) — documented as a single observation; the
    quiescent database remains the only supported path.
27. **What does the manifest provide?** Capture metadata required by `restore_backup`
    (present and required for restore success — `restore_manifest`). Audit closure adds
    PH3D-BACKUP-002: per-file size+sha256 inventory with INDEPENDENT integrity
    verification (source==backup and backup==restored, 0 hash mismatches).

## Memory (Q28–30)

28. **Was memory optimized in 3D?** No — optional PH3D-MEM-OPT-001 not run; no before/after
    claim, no figures (checker enforces figure-without-run cannot appear).
29. **What are the standing memory facts?** PH3C baselines unchanged: duplication ≥2× raw,
    drop releases ~82 MB of ~128 MB transient, ≈30K head-docs @512 envelope, unhandled
    termination leaks the engine dir (operational, never a correctness event — all crashed
    dirs reopened contract-consistently in 3D).
30. **Did the two 3D product fixes change memory behavior?** No — both are correctness fixes
    (checker invariant; compaction path resolution) with no hot-path allocations.

## Production status (Q31–35)

31. **Is AttentionDB production-ready?** As a single score: **the question is disallowed**
    (§45). As a matrix (`results/production-readiness.csv`, 25 capabilities): core storage,
    retrieval soundness, GroupCommit/Sync durability, recovery, transaction atomicity,
    parallel/mixed concurrency, quiescent backup/restore = VERIFIED; machine-crash axis,
    filter completeness, auto-compaction trigger sweep = PARTIAL; Async ack durability =
    NOT VERIFIED (documented loss semantics); online backup, online compaction, txn update
    op, isolation levels, pre-checkpoint segment-loss detection = UNSUPPORTED (documented);
    memory optimization = NOT ATTEMPTED. Audit-closure additions: cross-feature integration,
concurrent-log replay, and backup checksum integrity are VERIFIED (INTEGRATION-001 16/16,
CONC-003 4/4, BACKUP-002 6/6).
32. **What would block production use today?** Operational gaps: no online backup/compaction
    (quiescence required), Async-by-default ack semantics, undetectable pre-checkpoint WAL
    loss, untested power-loss axis, p99 growth at high reader counts.
33. **What is explicitly out of scope?** Distribution/replication/sharding (standing), and
    retrieval-quality work (frozen architecture; validated separately in 3B/3C).
34. **Does any claim in this report lack a run behind it?** None intended; every VERIFIED
    row cites run IDs; PARTIAL/UNSUPPORTED/NOT-ATTEMPTED rows say why. The one single-sample
    observation (live-writer backup) is labeled as such.
35. **Per-subsystem classification (§50).** Retrieval: VERIFIED (3B/3C quality; 3D
    live-only/determinism/soundness). Storage: VERIFIED (lifecycle, ID mapping, compaction).
    Durability: PARTIAL (process axis VERIFIED; machine axis untested; Async documented
    loss). Recovery: VERIFIED (with the documented pre-checkpoint deletion gap). Transactions:
    PARTIAL (atomicity/durability VERIFIED; isolation UNSUPPORTED). Concurrency: PARTIAL
    (stability VERIFIED; ordering NOT VERIFIED, no claim). Backup: PARTIAL (quiescent
    VERIFIED; online UNSUPPORTED). Scaling: VERIFIED for the frozen single-node head ladder
    (3C); multi-node out of scope. Memory: OPEN (baselines only; optimization not attempted).

## Reproducibility
Rebuild: `cargo build --release -p phase3-bench`. Families via `phase3-bench dbtest
<model|filterx|txn|crashchild|verify|walcorrupt|concur|concurreplay|backup|backupinv|
integration|compact|compactlive>`; crash
matrix driver and recorder preserved in session tooling; results/tables/figures regenerate
from raw via `tables/generate_results_ph3d.py`, `tables/generate_tables_ph3d.py`,
`figures/generate_figures_ph3d.py`; gates: `research/phase2/verify_consistency.py` +
`research/phase3/verify_consistency.py` (both PASS; the PH3D section fails on any raw↔CSV↔
registry drift, unregistered runs, resurrection rows, score columns, or figure-without-run).
