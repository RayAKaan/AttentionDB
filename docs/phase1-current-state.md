# Phase 1 Audit — Current State of AttentionDB

**Date:** 2026-09-03 · **Baseline commit:** `9cc29d6` (main, 2026-06-25) · **Auditor:** Phase 1 investigation
**Method:** direct source inspection of all 8 crates. The implementation — not the README — is treated as the source of truth. Every claim below cites files/functions.

---

## 1. Current architecture (as implemented)

```
                    ┌──────────────────────────────────────────────────┐
   REST :8080       │  api/src/bin/server.rs                           │
   gRPC  :7400 ───► │  AttentionDBService ──► AttentionEngine (core)   │
                    └──────────────┬───────────────────────────────────┘
                                   │
        ┌──────────────┬───────────┼────────────────┬───────────────┐
        ▼              ▼           ▼                ▼               ▼
  collections:     IdMapper    DocumentStore    HeadIndexManager   Bm25Index
  HashMap<RAM>     (RAM)       (storage)        (hnsw, per-head)   (core, RAM)
        │              │        │ memtable          │ HNSW graphs    │ postings
        │              │        │ flushed_records   │ vectors store  │ doc_len
        │              │        │ block_cache (LRU) │ id_to_idx      │ avgdl
        │              │        │ SSTables (*.sst)  ▼                ▼
        │              │        │ store.wal ◄─ SECOND WAL (replayed)
        ▼              ▼        ▼
   nothing        nothing   sstable_N.sst
   persisted      persisted
                                   │
                                   ▼
                         engine.wal ◄─ FIRST WAL (appended, NEVER replayed)
```

### Component authority matrix (who is authoritative today)

| State | Authoritative source | Survives restart? | Survives kill -9? |
|---|---|---|---|
| Documents (fields, vectors, tags) | `DocumentStore` (memtable → store.wal → SSTables) | ✅ yes (SST + store.wal replay) | ✅ mostly (page cache) |
| Collections (existence, dim, heads) | **RAM only** (`AttentionEngine.collections`) | ❌ **NO** | ❌ NO |
| Collection settings (`CollectionSettings`) | RAM (`collection.settings`) | ❌ NO (only `alter` RAM write, `engine.rs:191`) | ❌ NO |
| ID mapping UUID↔u64 (`IdMapper`) | RAM (`core/engine.rs:17`) | ❌ NO — `next_id` resets to 1 | ❌ NO |
| HNSW graphs + per-head vector stores | RAM (`HeadIndexManager`) | ❌ NO (save/load code exists, **never called** — `grep save_all core/ api/ query/` = empty) | ❌ NO |
| BM25 postings | RAM (`core/bm25.rs`) | ❌ NO — rebuilt never | ❌ NO |
| Schema fields | `Record.fields` per-document only; no collection schema object | ❌ N/A | — |
| Transactions | RAM staging map (`core/transaction.rs`) | ❌ NO | ❌ NO |

**Net effect:** after any restart, `engine.open()` returns an engine whose `collections` map is
empty. Every subsequent `ATTEND`, insert, or search on a pre-restart collection fails with
`CollectionNotFound`. Documents are physically on disk but logically unreachable. The README's
claim "On open, replays WAL, merges SSTables, garbage-collects tombstones" is **false for the
logical database** — only `DocumentStore`'s raw records survive.

## 2. Current persistence model

- **Directory layout** (flat, created lazily by `DocumentStore::open`):
  - `<data>/engine.wal` — engine-level WAL (`server.rs:49`)
  - `<data>/store.wal` — DocumentStore-internal WAL (`document_store.rs:230`)
  - `<data>/*.sst` — SSTables, filename `sstable_<unix_nanos>.sst` / `compacted_<unix_nanos>.sst`
  - No catalog, no manifest, no CURRENT pointer, no index files, no version markers.
- **Catalog:** does not exist. `create_collection` writes nothing to disk (`engine.rs:186-204`).
- **Manifest/atomic replacement:** does not exist. `SSTableWriter::new` does `File::create(path)`
  **directly at the final path** (`sstable.rs:27`) — a crash mid-flush leaves a partial `.sst`.
- **Format versioning:** none. `WalEntry` has no version field; SST payload is unversioned bincode.

## 3. Current WAL model — TWO independent WALs

### WAL #1 — `storage/src/wal.rs` ("engine.wal")
- Framing: raw `bincode(WalEntry)` back-to-back, **no length prefix, no magic**. Recovery can only
  guess entry boundaries (`replay()` deserializes at every offset until error).
- `crc32` covers **only `data`** (`wal.rs:87-90`) — a flipped bit in `lsn`, `op`, `collection`, or
  `record_id` is **undetected**.
- No format version. No segmentation/rotation → unbounded single file.
- `next_lsn` starts at 1 in `Wal::new` (`wal.rs:47`); **`engine.open()` never calls `replay()`**
  (`engine.rs:145-153`) → after restart the WAL contains duplicate LSNs starting again at 1.
- `replay()` exists but: (a) is never called by the engine path, (b) **silently drops** entries that
  fail CRC or deserialization (`wal.rs:120-133`) — silent data loss, (c) cannot distinguish torn
  tail from mid-log corruption.
- Durability: `Sync` = flush+`sync_all` per append; `GroupCommit` = userspace flush only
  (page-cache durable, **not** fsync-durable); `Async` = buffered. Undocumented.

### WAL #2 — DocumentStore's internal `store.wal` (`document_store.rs:228-232, 254-263, 386-390`)
- Same `Wal` type, separate file. Replayed inside `DocumentStore::open`.
- Records re-declared `OpType::Insert/Delete` with `collection` hardcoded `"default"` — collection
  association is **lost**.
- Consequence: **every document insert is written to two logs** with different schemas and
  different lifetimes, and only one of them is ever read. This is the classic dual-log divergence
  hazard: the engine's WAL says "insert into collection X", the store's WAL says "insert into
  default". Neither is authoritative for the logical database.

## 4. Current recovery model

`AttentionEngine::open(wal_path, durability)` (`engine.rs:144-160`):
1. `Wal::new(engine.wal)` — creates/truncates nothing; LSN=1; **no replay**.
2. `DocumentStore::open(dir)` — loads `*.sst` **sorted by filename**, last-record-wins into
   `flushed_records`; then replays `store.wal` into `memtable`.
3. Collections, IdMapper, HNSW, BM25: **freshly empty**.

Failure modes inside recovery (all silent):
- `SSTableReader::open` failure → `if let Ok(r)` → **file silently skipped** (`document_store.rs:227`).
- Corrupt WAL entry → skipped silently (`wal.rs:126-131`).
- `server.rs:53-59`: if `engine.open` fails entirely, the server **starts an in-memory engine** and
  serves traffic on an empty database — a catastrophic data-loss footgun disguised as resilience.

### Kill -9 semantics today
- OS page cache survives process death (machine alive), so `GroupCommit`'s `write()` calls
  survive `kill -9`. Machine/power crash is only covered by `Durability::Sync`.
- Everything RAM-only (collections, mappings, indexes, BM25, staged txns) is lost — see §1.
- Rust drop handlers do not run; nothing today depends on them, but nothing today is checkpointed either.

## 5. Current index lifecycle

- **HNSW:** `Collection.insert_vector` → `HeadIndexManager.insert` → `HNSWIndex::insert`
  (`hnsw_index.rs:144`). `hnsw_rs`'s `Hnsw` has **no deletion**; `HNSWIndex` keeps a side vector
  store (`vectors: Vec<(u64, Vec<f32>)>` + `id_to_idx`) used for exact rerank.
  - `save()`/`load()`/`save_graph()` and `HeadIndexManager::save_all()` exist
    (`head_index.rs:96`) but **nothing in core/api/query calls them** → dead code in the live path.
  - `max_elements` defaults to 100_000 (`hnsw_index.rs:29`) — silent insert failure beyond that
    (`hnsw_rs` returns Err on overflow, which `Collection::attend`'s `filter_map(.ok())` would
    also swallow at search time).
- **BM25:** `insert`/`search`/`search_phrase`/`reciprocal_rank_fusion` (`core/bm25.rs`). There is
  **no `remove()`** — a deleted document remains in postings forever; re-inserting the same doc id
  double-counts it (term frequencies inflated).
- **Rebuild:** none. Nothing reconstructs indexes at startup.

## 6. Current transaction model

`core/transaction.rs` + `engine.rs:294-316`:
- `begin_transaction` → in-RAM staging; `commit_transaction` **applies each op sequentially through
  the normal insert/delete paths**. Properties:
  - No WAL records for begin/ops/commit → crash mid-commit leaves a **partially applied
    transaction** (e.g., A inserted, B missing, C deleted) with no way to roll forward or back.
  - Staged ops are lost on crash (acceptable) but committed-not-yet-applied is impossible to
    distinguish from applied — because commit and apply are interleaved without a durable marker.
  - **It is not atomic. It must not be called ACID.**
- Single-op writes: insert writes engine WAL + store WAL; delete writes **only** store WAL (see §7).

## 7. Known correctness gaps (each with responsible files)

| # | Gap | Severity | Responsible code |
|---|---|---|---|
| G1 | Collections/config/heads not durable; restart orphans all data | **Critical** | `core/engine.rs` (collections map), `api/src/bin/server.rs:49` |
| G2 | IdMapper not durable; `next_id` resets → re-registering a survived doc mints a **new** numeric id → duplicate vectors under two ids for one logical doc | **Critical** | `core/engine.rs:17-133`, `document_store.rs` |
| G3 | Delete touches DocumentStore only: HNSW, BM25, IdMapper, engine WAL keep the doc → deleted docs remain searchable; delete not in engine WAL → replay can resurrect | **Critical** | `core/engine.rs:336-347`, `core/bm25.rs` (no remove), `hnsw` (no delete) |
| G4 | No update/upsert operation. Re-insert of same UUID re-inserts vectors into HNSW (duplicate graph nodes) and BM25 (double-counted doc) | **Critical** | `core/engine.rs:243-266`, `collection.rs:56-66` |
| G5 | Two uncoordinated WALs; engine WAL never replayed; LSN resets | **Critical** | `storage/wal.rs`, `core/engine.rs:145`, `document_store.rs:228` |
| G6 | WAL framing has no length prefix; CRC covers payload only; silent skip on replay; no segmentation/retention; no version | **High** | `storage/wal.rs` |
| G7 | Transactions not atomic; no durable commit marker | **High** | `core/transaction.rs`, `engine.rs:294` |
| G8 | SSTables written directly at final path (no tmp+rename); open() silently skips unreadable SSTs | **High** | `storage/sstable.rs:27`, `document_store.rs:227` |
| G9 | SST load order = filename sort, last-write-wins ignoring `timestamp`: `compacted_*` always sorts before `sstable_*`, so a **stale record in an older unmerged sstable can overwrite a newer compacted record** (stale reads / resurrection across compaction generations) | **High** | `document_store.rs:207-240`, `compaction.rs:31-63` |
| G10 | Compaction GCs tombstones from **partial** merges (older unmerged files may still hold pre-delete state) | **High** | `compaction.rs:97-100` |
| G11 | Server falls back to in-memory engine when storage fails to open | **Critical** | `api/src/bin/server.rs:53-59` |
| G12 | Readiness not gated on recovery; no startup state machine | High | `api/src/rest.rs:297` |
| G13 | No checkpoint; WAL and SST state have no defined meeting point; store.wal grows forever; engine.wal grows forever | **High** | — |
| G14 | `TEMPORAL_DECAY`/`MIN_WEIGHT` parsed but never applied; gating network never trained/loaded in server path (uniform 1/N fusion) — *out of Phase 1 scope, documented for honesty* | Medium | `query/parser.rs:175-176`, `core/collection.rs:70-82` |
| G15 | HNSW persistence code exists but unused; no index load/validate/rebuild at startup | High | `hnsw/persistence/*`, `head_index.rs:96` |
| G16 | Block cache: insert/delete paths keep it coherent (tombstone cached), but it is a second copy of records with no invalidation on direct SST changes; memory accounting is approximate (`estimate_record_size` heuristics) | Medium | `document_store.rs:36-134` |
| G17 | Errors: `unwrap/expect/panic` in parsing paths; no unified error taxonomy; malformed files can panic rather than return `Corruption` | Medium | various (see error-handling audit in final report) |
| G18 | No backup consistency guarantee: `admin.rs` copies the live directory while writes proceed (no checkpoint/quiesce) | High | `api/src/admin.rs` |

## 8. What Phase 1 must therefore build (summary)

1. **One authoritative WAL** (segmented, framed, versioned, CRC over full record, torn-tail vs
   corruption distinction, replay that drives all subsystems) — replaces the dual-WAL arrangement.
2. **Durable catalog + manifest** (CURRENT/MANIFEST, atomic tmp+fsync+rename install, versioned).
3. **Checkpoint** binding WAL sequence ↔ SST/idmap/catalog state; WAL segment retention by checkpoint.
4. **Deterministic HNSW + BM25 rebuild** from authoritative document state at startup (hnsw_rs
   cannot delete; persisted-graph fast path deferred — see final report), with per-head accounting.
5. **Delete/update/upsert** as first-class, fan-out-to-all-subsystems operations with retirement of
   stale vector ids (tombstone filtering) since the graph cannot physically delete.
6. **IdMapper persistence** with exact next-id and retired-id state.
7. **Transactional commit markers** in the WAL; replay applies only committed txns, idempotently.
8. **Recovery state machine** with explicit failure (no READY-with-missing-data, no in-memory fallback).
9. Timestamp-correct SST merge + generation-safe compaction (tombstones retained unless full compaction).
10. Consistency checker, backup/restore via checkpoint quiesce, format versions, tests, docs.

These items are implemented in this phase; see `docs/consistency-model.md` (invariants) and
`docs/phase1-final-report.md` (what shipped, what remains).
