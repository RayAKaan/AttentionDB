# C0 — Storage-Path Audit

Commit `fe4f92b`. The storage layer was verified in depth during Phase 3E
(E1–E11, sealed); this summary cites the implementation and carries the
sealed findings forward as constraints for comparative benchmarking.

## Write path

- `insert_document` / update / delete / upsert → validation → WAL append →
  in-memory apply (`core/src/engine.rs:794+`, `wal_append` at
  `engine.rs:1946-1962`). Durability modes: Sync (fsync per append),
  GroupCommit (flush to page cache), Async (userspace buffer; `flush_wal()` /
  `checkpoint()` promote) — `storage/src/wal.rs`, contract A2.
- WAL: framed records ("WAL2" magic + len + body + crc32), segmented
  `WAL/<start_seq:020>.wal`, `WAL/wal-state.json` sidecar (derived
  bookkeeping — regressed watermark tolerated, A11), torn-tail truncation,
  corruption refusal (G5/A1).
- Transactions: staged, all-or-nothing at the CommitTxn WAL marker (A6);
  TxnOp = Insert|Delete, single collection.

## Read/storage structures

- DocumentStore: memtable + SSTables + `BlockCache` (bounded 50k blocks,
  `storage/src/document_store.rs:64-210`); read-through record cache
  (unbounded by design == live docs — E10 finding: ~16.5-17.2 KB/doc peak
  RSS marginal, payload-independent).
- SSTables + tombstones; automatic incremental compaction at ≥4 SSTs;
  coordinated `compact_storage` (A5). Backup: mutation-gated consistent
  snapshot, `backup-meta.json` written LAST = completion marker; restore
  refuses manifest-less partials (A4). Checkpoint: flush → SST → id-map →
  manifest → WAL trim (A8 windows).
- HNSW persistence: `hnsw/src/persistence/` (graph persistence, async
  compaction, backup); recovery = deterministic index rebuild (the E10
  measured cost: ~0.7-1.05 ms/doc, 44.6-83.7 s at 60-80k).
- Hygiene: checkpoint-time purge of retired ids from the exact-rerank vector
  store + deterministic rebuild when dead entries dominate
  (`collection.rs:232-254`; A9 + D40 head-count-aware predicate, E10-fixed,
  regression-sealed).

## Verified envelope and behaviors that BENCHMARKS MUST RESPECT

From sealed Phase 3E (do not re-litigate; design around):
- 2 vCPU / 1.9 GiB host class: verified 40k docs (primary config, full gate
  battery), 60k slim + 60k integrated lifecycle, 80k reproduced with caveats
  (memory guard + grown-graph recall variance), ≥100k blocked; hnsw_rs
  max_elements default 100k/head (E10, A10).
- Async acked writes may be lost on process death (documented boundary);
  seal pattern: final checkpoint before graceful reopen (E8f/E11-confirmed).
- Recovery rebuild is deterministic and dominant in restart time.
- Under a scan-heavy reader the writer degrades ~30× (measured, E10-019;
  reader pacing 2ms is the established harness pattern for W13-type
  workloads).
- Concurrency semantics: per-key last-write-wins under gate serialization,
  NO isolation levels / linearizability (A7) — comparative workloads must not
  assume multi-version concurrency.

## Implications for the comparative study

1. Memory is the binding resource on this host: external systems must be
   sized to the SAME host constraints (Track B) or the mismatch documented
   (Track A on identical embeddings/queries only).
2. Lifecycle measurements (build, checkpoint, compact, backup, recovery) are
   already engine-first-class (A5/A8/A4/A9) and map cleanly to the study's
   LIFECYCLE metric family.
3. The `crashgate` instrumentation should stay INERT during comparative runs
   (it is env-gated; regression-pinned) — the harness must not set
   PH3E_CRASH_* outside dedicated fault experiments.
4. Dataset ingestion uses `insert_document` with per-head vectors; multi-head
   datasets therefore cost N× index build time and memory vs single-head
   (per-head independent HNSW graphs) — a first-order RQ4/RQ9 cost item to
   measure, not assume.
