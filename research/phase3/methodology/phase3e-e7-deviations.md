# Phase 3E — E7 Deviations (PH3E-CONC-001/-002)

Status: CLOSED with E7. Every deviation names the invariant it touches.
Numbering continues the E6 deviations convention (D1–D11 in ph3e-e6-deviations.md);
E7 deviations are D12+.

## D12 — Engine defect #1 (FOUND & FIXED): MISSING_MAPPING orphan after committed same-uuid [Delete, Insert] transactions
- **Invariant:** INV-REPLAY-CONVERGENCE (fresh-process replay must reproduce the live committed state) + INV-CHECKER-CLEAN (post-recovery consistency checker must pass on any state produced by committed transactions).
- **Discovery:** E7g/E7w harness rewrite used same-uuid [Delete U; Insert U] per transaction. `e6_open` refused recovery: `Recovery failed: 1 consistency error(s)`; checker code `MISSING_MAPPING` (orphan record, uuid `00000000-0000-17d4-0000-000000000000` = uuid (6100<<64)|0).
- **Root cause:** commit-apply `TxnOp::Insert` (core/src/engine.rs) resolved `numeric = uuid_to_id(&record.id).unwrap_or(0)` AFTER the same transaction's Delete arm had already retired the uuid→numeric mapping. The reinsert was applied under numeric id **0** with **no live mapping**. Live commit returned Ok; a later checkpoint persisted the orphan; fresh open refused. Additional latent hazard: all such inserts collide in the vector space under id 0.
- **Evidence preserved:** `research/phase3/raw/runs/PH3E-CONC-001/bug-MISSING_MAPPING-preserved-dir/` (exact failing on-disk state + README). Minimal repro: two committed same-uuid [Delete,Insert] txns after a plain insert + checkpoint.
- **Fix:** re-register a fresh numeric id when (and only when) the mapping is absent at apply time (double-checked write-path; read-then-write with bound guard to avoid RwLock self-deadlock). Normal inserts (mapping present) are unchanged — the WAL-write path already registered them.
- **Regression:** `core/tests/e7_replay_probe.rs` — 2 tests (checkpoint close path + WAL no-close crash path).
- **Protocol trail:** failing state preserved → diagnosed → fixed → regression pinned → E1–E6 regression re-runs under NEW IDs (all byte/classification-identical: PH3E-WAL-008, PH3E-DUR-011, PH3E-E3-005, PH3E-BACKUP-007, PH3E-COMPACT-006, PH3E-TXN-002) → this document.

## D13 — Suspected defect #2 DISSOLVED: E7a "attend returned id ≥ 120" was a harness off-by-one
- **Invariant:** precise claim vocabulary (an instrument defect must never be reported as an engine defect).
- **Course of events:** E7a rows intermittently reported `errors>0` with `attend_bad_id` dominance under ≥8 reader threads. Forensics (kinds breakdown + bad-id capture) isolated id **120** exactly, reproducible single-threaded for e23-direction queries.
- **Root cause:** the id mapper mints numeric ids **from 1** (proven by `core/tests/e7_tiny_probe.rs`: doc idx i → numeric i+1; attend returns exactly the mapper id with correct top-hit ordering). A 120-doc collection owns numeric ids **1..=120**; the harness validity check `id >= 120` flagged the legitimate doc idx 119 (numeric 120, direction e23 — a top hit for e23-type queries; surfacing varied with HNSW graph randomness and thread timing).
- **Resolution:** instrument corrected to `id == 0 || id > 120` (0 = n/a sentinel); attend-probe regression corrected and pinned (`core/tests/e7_attend_probe.rs`). **No engine change.** PH3E-CONC-001 (affected instrument) superseded by PH3E-CONC-002 per the new-run-ID rule; 001 preserved untouched.

## D14 — E7c (repeatable-read) and E7d (dirty-read) family mapping
- The spec table lists E7c/E7d as named families; the harness expresses them as follows: E7d (dirty reads) is covered by E7b (committed-only census over a hot flip stream) **and** E7k (ordered staged-window reads: no staged value ever visible). E7c (repeatable-read) is covered by E7e/E7k single-version-key ordered observations: a non-txn reader re-reading an unmodified key always sees the same committed version; there is **no** per-txn snapshot, so a stronger repeatable-read claim is NOT made. Recorded here so no family is silently skipped (anti-gaming rule: inexpressible ⇒ UNSUPPORTED BY API; partially expressible ⇒ measured subset).

## D15 — Bounded-observation honesty for hot-reader families (E7e/E7f/E7m/E7u)
- Whether a hot reader *captures* a mixed pair or an intra-commit window is scheduling-dependent. PH3E-CONC-002: E7e captured 0 mixed pairs (its reader completed whole (A,B) rounds faster than the commit window this run); E7f captured 3 mixed pairs including the (A,B)=(absent,absent) state; E7m classified 39/40 retire-notices as noticed-after-ACK with 0 post-ACK stability violations; E7u saw 1 absent-window observation. PH3E-CONC-001 (superseded for E7a only) captured 115 mixed (A,B) pairs in E7e. All raw counts are reported as-is; absence of a captured window is **bounded observation**, never evidence of absence — the claims in the report rest on the invariant (stability reads, monotonicity, ordered barriers), with the captured windows as supporting evidence.

## D16 — E7i/E7j recorded as UNSUPPORTED BY API (not failures)
- `TxnOp = Insert | Delete`; transactions cannot read or query. Write-skew and phantom tests are therefore not expressible; the rows record `not-expressible` + `UNSUPPORTED-NO-TXN-READS/-QUERY` with status `-`. Nothing was inferred from point reads (anti-gaming rule honored).

## D17 — Engine instrumentation and debug helpers added during diagnosis
- `dbtest e7dbg --dir X` gained an env-gated (`PH3E_E7DBG_WAL`) WAL dumper; temporary `PH3E_REPLAY_DEBUG` tracing in engine.rs was **removed** after diagnosis (verified absent from the final tree). `core/Cargo.toml` gained `tracing-subscriber` as a dev-dependency for the probe's ERROR-level checker output.

## D18 — Stale-binary and sandbox-reset events during the segment
- Two stale-binary traps were hit and corrected by policy (confirm build success + binary mtime before every run). A sandbox reset mid-segment wiped /var/tmp/toolchain and protoc; both were reinstalled before any further measured run. No raw run was produced by a stale binary: PH3E-CONC-001/-002 and all six regression runs were executed after explicit rebuilds.
