# AttentionDB Consistency Model (Phase 1, single node)

Status: **Authoritative** for Phase 1. Every mutation path in `AttentionEngine` must preserve these
invariants. The consistency checker (`attentiondb-check`) validates them on demand; recovery
validates them before READY.

## 1. The authoritative logical state

AttentionDB has **exactly one authoritative logical state**, defined by:

```
Database
 └── Collection (catalog entry: id, name, dim, heads, settings, schema fields)
      └── Document (uuid ↔ numeric_id, one live Record: fields, k_vecs, tags, version)
           ├── HNSW membership: for every (head, vector) in k_vecs, exactly one live
           │   vector with id == numeric_id in that head's index
           ├── BM25 membership: exactly one posting set entry for numeric_id, built
           │   from the live record's text fields
           └── IdMapper: uuid ↔ numeric_id bijection over live documents
```

The durable representation of this state is:

```
MANIFEST (catalog: collections, settings, next ids, checkpoint_seq)
 + SSTables (records incl. tombstones, timestamped)
 + WAL segments (every logical mutation after/between checkpoints)
 + META/idmap.bin (snapshot of IdMapper at checkpoint time)
 + INDEX/<coll>/<head>.meta (index generation marker; graph rebuilt deterministically)
```

**Rule of authority:** on any conflict between in-memory caches, index side-stores, or SST
contents, the state materialized by `(MANIFEST + SSTables + committed WAL prefix)` wins. Caches
and index side-stores are derivable and never authoritative.

## 2. Mutation rule

Every logical mutation is applied in this order:

1. Append a versioned, checksummed record to the authoritative WAL (with the mode's durability:
   `Sync` ⇒ fsync before ack; `GroupCommit`/`Async` ⇒ flush semantics documented in `docs/wal.md`).
2. Apply the mutation to every affected subsystem in a fixed order:
   DocumentStore → IdMapper → per-head HNSW → BM25 → caches.
3. Acknowledge success only after (1) succeeded; (2) failures abort the server process
   (fail-stop: the WAL is truth, an unapplied WAL replays on restart).

Transactions wrap (1) in `BeginTxn … CommitTxn` markers; subsystem application happens only after
the `CommitTxn` record is durable. Recovery applies a transaction's ops only if its commit marker
is present, and does so idempotently.

## 3. Invariants

- **INV-1 (vector⇒document):** Every live vector in every HNSW head corresponds to a live logical
  document in the same collection.
- **INV-2 (document⇒index):** Every live document's `k_vecs` entry is represented in the
  corresponding head index under its current `numeric_id` (modulo retired-id filtering until the
  next rebuild; retired ids can never be *returned*).
- **INV-3 (durable delete):** After a delete is acknowledged, no subsystem (DocumentStore, HNSW
  results, BM25, IdMapper, caches, restored backup) returns that document, across restarts.
- **INV-4 (update replaces):** After an update is acknowledged, searches never return the old
  vector/old content for that document; the old numeric id is retired and filtered.
- **INV-5 (update uniqueness):** An update never creates a second logical document; the uuid keeps
  exactly one live numeric id.
- **INV-6 (id bijection):** Every numeric id maps to exactly one uuid and vice versa;
  `next_id` is strictly monotonic across restarts; retired ids are never reused.
- **INV-7 (catalog durability):** Collection existence, ids, names, dims, heads, settings survive
  restart and crash.
- **INV-8 (schema durability):** Collection-level schema metadata (field names/types where
  declared) survives restart; per-document fields always survive with the record.
- **INV-9 (idempotent replay):** Replaying any committed WAL prefix (twice, or after a partial
  apply followed by crash) yields the same logical state. Replay applies each record at most once
  per recovery using sequence numbers.
- **INV-10 (recovery = clean shutdown):** After recovery from any crash point, the logical state
  equals the state after the last *acknowledged* mutation (Sync), or the last *flushed* mutation
  (GroupCommit/Async, documented), never a partial mixture of a transaction.
- **INV-11 (no phantom READY):** The server binds ports and reports ready only after recovery
  reached READY. Any recovery failure ⇒ RECOVERY_FAILED ⇒ process exits non-zero.
- **INV-12 (compaction safety):** Compaction may only remove data (tombstones or shadowed
  versions) that no reachable state can need: tombstones are dropped only when compaction inputs
  include *every* SST that predates the checkpoint that durably observed the delete.
- **INV-13 (cache discipline):** Caches return the same logical value as the authoritative state
  for any key, at any time, including after update/delete/restart.

## 4. Concurrency model (single node)

- All mutations serialize through the engine's WAL append (sequence order = commit order).
- Collection-level `RwLock`s guard subsystem application; readers (search) take read locks.
- Conflicting concurrent writes to the same document are ordered by WAL sequence;
  **last-committed-write-wins**. There are no write-write conflicts errors in Phase 1.
- Reads may race with writes; a read observes all mutations committed before it acquired its
  collection read lock (no snapshots in Phase 1 — documented limitation).
- Checkpoint and compaction take the same write path locks; they may run while traffic is served
  but serialize against mutations (correctness over concurrency).

## 5. Durability contract exposed by the API

| Mode | Ack means | Risk window |
|---|---|---|
| `SYNC` | record is in WAL file and `fsync`'d | none (machine-level durable) |
| `GROUP_COMMIT` (default) | record was written+flushed into OS page cache | machine crash/power loss may lose recent acks; process crash does not |
| `ASYNC` | record was handed to the in-process buffered writer | process crash may lose recent acks |

These semantics are returned in write responses (`durability` field) and documented in
`docs/wal.md`. ASYNC is never reported as fsync-durable.
