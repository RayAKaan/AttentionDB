# Phase 3E — E9 Specification: Memory Instrumentation, Root-Cause Analysis & Optimization
(FROZEN before measurement)

Date frozen: 2026-09-22. Baseline: commit `3e65d50f8fd1a81cda339d4608260d6be98ba708`
with the E8 working-tree state (INV-RECOVERY-READONLY + INV-C2; tree sha16
`1b524cf3e39dfc6b`). Entering hypothesis from E8 (NOT proof): RSS grows
approximately linearly with progress operations (Pearson r 0.869–1.000, slope
~0.4–1.2 MB/1k progress-ops, peak 310,980 KB), classified
`linear-with-ops`, boundedness NOT established, E9 marker.

## Method markers

- M1 — reproducibility: reproduce the E8 RSS-vs-ops signal with the UNCHANGED
  E8 harness (same family, seed discipline, durability mode, filesystem) under
  a NEW run ID before any instrumentation. If it fails to reproduce,
  investigate environment divergence; do not declare the E8 observation invalid.
- M2 — component attribution: feature-gated read-only census (`mem-census`)
  exposing per-component counts/capacities/estimated bytes (DocumentStore,
  memtable, SST readers, HNSW, ID mapper, WAL, txn manager, caches). Census
  only READS state; it never mutates engine behavior. Estimates are labeled
  MEASURED / ESTIMATED / UNKNOWN and never presented as exact allocations.
- M3 — live-object accounting: separate application-owned live memory
  (census counts × per-entry payload estimates) from total process RSS.
- M4 — allocator-retention separation: glibc `mallinfo2` sampled from the
  HARNESS process only (uordblks = in-use heap bytes, fordblks = free heap
  bytes retained by allocator). No allocator replacement.
- M5 — filesystem/page-cache separation: RssAnon vs RssFile (Vm status) and
  Pss/Pss_Anon/Pss_File (smaps_rollup) sampled every 2 s. File-backed
  residency is never classified as heap leakage.
- M6 — lifecycle release: restart-reset, idle-decay and drop-lifecycle
  experiments decide whether growth is process-local retained state or
  persistent on-disk-proportional state.
- M7 — boundedness: for each candidate source, growth is fit against
  documents, operations, SST count, WAL rotations and compaction count
  (R² + residuals, not Pearson alone).
- M8 — optimization effectiveness: paired before/after on identical
  (workload, seed, environment): peak RSS, steady RSS, slope, plus the
  performance guardrail metrics (§50 of the E9 prompt).
- M9 — correctness preservation: every optimization re-proves E1–E8 via the
  existing sealed suites (new run IDs where raw runs are produced), the full
  workspace test suite, and the consistency checkers.
- M10 — regression safety: resource-leak regression (FDs/threads/files) after
  every accepted optimization; no memory win may trade for another leak.

## Registry

PH3E-MEM-001.. N. Raw runs immutable; INVALIDATED/failed runs preserved.
Memory profiles store: run_id, commit, environment, workload, seed, duration,
operations, live_documents, RSS, PSS, RssAnon, RssFile, heap metrics, WAL
bytes, SST bytes, SST count, FD count, thread count — `NA` where a metric is
not available, never invented.

## Root-cause discipline

Every hypothesis branch ends SUPPORTED / REFUTED / UNRESOLVED with evidence
and confidence (HIGH/MEDIUM/LOW/UNRESOLVED). "Root cause" is claimed only
when the evidence establishes mechanism, not correlation. Optimization only
after the source of growth is established; smallest robust fix first
(unbounded retention → duplication → long-lived buffers → capacity retention
→ caches → temporary peaks → allocator). No HNSW redesign; no retrieval or
isolation changes; no allocator swap for a prettier RSS graph; no page-cache
manipulation. E10 scale work, E11 fault injection, distributed/replication/
MVCC remain forbidden. HARD STOP after E9.

## Terminology (binding for the report)

RSS, PSS, virtual address space, heap allocated bytes, heap live bytes,
resident anonymous memory, resident file-backed memory, page cache,
allocator-retained memory, application-owned live memory are DISTINCT terms
and used accordingly. "Memory leak" is claimed only for demonstrated
application-owned retained allocations that are never released.
