# Phase 3E — E9 Deviations & Adaptations

Author: Rayyan Kaan (RayAKaan). All deviations documented at or before
measurement; raw runs immutable; failures preserved.

## D30 — Toolchain drift
The sandbox reset between E8 and E9 wiped the pinned toolchain; E9 ran on
rustc 1.98.1 (E8 used the 1.90-era pin). The new clippy flagged pre-existing
E8 harness patterns (`is_multiple_of`, `too_many_arguments`, counter-loop);
machine-applied fixes + three targeted `#[allow]`s on E8-frozen harness
functions. Engine semantics untouched; full suite green before and after.

## D31 — Durability mode of the E9 control runs
PH3E-MEM-002..017 ran with the sandbox-default durability (async) because the
batch shell did not export `PH3D_DURABILITY`. The growth phenomenon is
durability-independent (MEM-001, the sync-context E8-harness reproduction,
matches the async-context census runs in shape and slope class), and every
before/after PAIR (002/018/022 etc.) ran under the identical mode, so paired
comparisons are valid. The E9 regression soaks (PH3E-MEM-026..029,
PH3E-SOAK-017) ran sync, matching E8 officials.

## D32 — mallinfo2 accounting unreliable under rebuild workloads
After INV-E9-HYGIENE rebuilds, glibc `mallinfo2` reported uordblks+fordblks
totals exceeding process RSS (impossible for real mappings) — the allocator's
arena accounting diverges from the true resident set under the rebuild
allocation pattern. All application-attribution claims therefore rest on
/proc status + smaps_rollup (RSS/PSS/RssAnon/RssFile) and the engine census,
which are mutually consistent. mallinfo columns are retained in raw telemetry
and labeled unreliable-under-rebuild.

## D33 — Persistence-cap triage (second incident)
The workspace snapshot cap silently dropped E8 evidence after the first
reset; E9 additionally generated ~300 MB of run residue. Triage documented in
raw/runs/TRIAGE-2026-09-22.md: restoration executions for the platform loss
(PH3E-SOAK-012..016), regenerable green-run db/backup/restore removal,
git-recoverable phase-2b bulk removal (checker reads only small summaries),
crash-child binary removal for green soaks (deterministically regenerable
from seed+kill-point+prefix; text evidence retained). The phase-3 claim-ledger
gate correctly FORBADE phase-2 working-tree changes; those deletions were
fully reverted via git checkout. One author error: a `-size +200k` cleanup
over-deleted PH3E-SOAK-014/oplog.csv.gz and PH3E-MEM-029/oplog.csv.gz;
family f was re-executed with full artifacts as PH3E-SOAK-017 (count-identical
to 006/014/029). No failure evidence was ever touched:
PH3E-SOAK-004/bug-preserved-e8d-refusal and PH3E-SOAK-009 are byte-intact.

## D34 — O1 (purge-only) was insufficient and is preserved as evidence
The first optimization candidate (checkpoint-time vector-store purge only,
PH3E-MEM-018..021) reduced dead store entries 54,378→~6,000 but moved peak
RSS ≤3% — the dominant retention is hnsw_rs GRAPH nodes, invisible to the
census and unpurgeable. This negative result motivated O2 (index-insert
budget forcing deterministic rebuilds). O1-only runs are retained in raw.

## D35 — E9 harness telemetry defect (WAL columns)
The e9 harness dir-walk recorded wal_bytes=0 (wrong subdir name match);
WAL-size maxima for E9 runs are NA in the E9 tables. The E8 harness's WAL
telemetry (resource.csv, used for MEM-001) is unaffected. db_bytes and
sst_bytes are valid.

## D36 — Retrieval-recall degradation is a new MEASURED finding
At ≥95% dead-node ratio (long-lived process, heavy churn, no hygiene) exact
top-k recall degraded nondeterministically (0/5 vs 5/5 hits across runs) —
a quantified consequence of the documented hnsw_rs no-delete limitation, not
a new engine defect code path. INV-E9-HYGIENE rebuilds restore recall
(regression-sealed). E8's S18 evidence is not contradicted: E8 verified
addressability via export/model equality and shape-based reader contracts
within its tested budgets; E9 strengthens S18 by making recall
self-restoring.

## D37 — No E10-class work
No scale-envelope expansion, no allocator replacement, no HNSW redesign
(hygiene uses the existing sealed recovery-path rebuild), no page-cache
manipulation, no retrieval/isolation semantics change. E11 fault injection
not started. HARD STOP honored.
