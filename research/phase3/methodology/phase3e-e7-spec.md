# Phase 3E — E7 Spec: Concurrency & Isolation Boundaries

Run family: `PH3E-CONC-001` · baseline commit `3e65d50` (E6 final) · charter:
determine experimentally what consistency/isolation AttentionDB actually
provides under concurrent readers, writers, and transactions. The answer may
be "no formal isolation level" — that is a valid result. No MVCC/isolation
implementation work is in scope.

## Part I — Concurrency implementation audit (from actual code, commit 3e65d50)

### Synchronization inventory

| structure | type | protects |
|---|---|---|
| `mutation_gate` | `parking_lot::Mutex<()>` (engine.rs:322) | mutation/commit/ckpt/compact/backup critical sections |
| `collections` | `RwLock<HashMap<String, Arc<Collection>>>` | collection catalog map |
| `document_store` | `RwLock<DocumentStore>` | authoritative documents |
| `id_mapper` | `RwLock<IdMapper>` | uuid↔numeric mapping, retired set |
| `catalog` | `Mutex<Catalog>` | SST manifest |
| `wal` | `Mutex<Option<Wal>>` | WAL append/fsync/rotate |
| `state` | `RwLock<EngineState>` | open/closed transitions |
| `TransactionManager.transactions` | `Mutex<HashMap<u64, Transaction>>` | staged txn buffers |
| head indexes | `head_manager: RwLock<HeadIndexManager>` per collection | HNSW graphs |
| bm25 | internal lock per collection | BM25 stats |

### Which operations acquire the mutation gate?

`create_collection*`, `drop_collection`, `insert_document`,
`update_document`, `delete_document`, `commit_transaction`,
`checkpoint`, `compact_storage`, `backup_to` (gate held across
checkpoint+copy). NOT gated: `begin_transaction`,
`record_transaction_operation`, `rollback_transaction`, all read APIs
(`attend*`, `scan_filtered`, `get_document_fields`), WAL replay at open.

### Locks on the write path (insert, gated)

gate → validate → `id_mapper.write()` (register) → `wal.lock()` append
(per-mode durability inside) → apply: `document_store.write()` insert_nolog →
`collection.insert_vector` (head write) → bm25 insert → ACK. Locks are
acquired and released **per step**: the gate does not exclude readers, and
readers take none of the writer's locks except briefly in shared mode.

### Locks on the read path

`attend` → collection `attend_detailed`: `head_manager.read()` per head +
exact rerank reads (store_vectors copies); `scan_filtered`:
`id_mapper.read()` + `document_store.read()`; `get_document_fields`:
`id_mapper.read()` then `document_store.read()` (two acquisitions — a writer
may mutate between them; each read is individually consistent).

### Audit answers (§4 of the prompt)

- Two readers simultaneously? **YES** — all read paths take RwLock read guards.
- Two writers simultaneously? **NO** — every mutation path holds the
  mutation gate for its whole critical section (write→apply→ACK).
- Reader during writer? **YES** — readers never take the gate; they share
  per-structure read locks with the writer's per-step exclusive locks.
- Txn stage while another commits? **YES** — staging takes only the
  txn-manager Mutex briefly; commit holds the gate.
- Two txns stage simultaneously? **YES** (E6j evidence; Mutex on the buffer map).
- Two txns commit simultaneously? **NO** — commit holds the mutation gate;
  WAL order = commit order (E6j).
- Backup/checkpoint/compaction while a txn is staged? **YES** — staging does
  not take the gate; the snapshot/checkpoint sees only committed state
  because staged state lives in the txn-manager buffer, invisible to
  doc-store/ckpts (E6o/E6q; re-verified under true concurrency in E7).
- Reader observes staged txn state? **NO** — staging is memory-private to
  the txn buffer; no read path consults it (E7d/E7k prove with barriers).
- Reader observes a txn between WAL append and apply? **YES** — readers do
  not take the gate; the apply of a committed txn happens after the CommitTxn
  append, per-op, under per-op locks (E7m measures the boundary).
- Reader observes PARTIAL effects of a committing txn? **EXPECTED YES** —
  the commit apply loop releases/reacquires `document_store.write()` per op,
  so a concurrent multi-read can interleave between ops (E7e/E7f measure).
- Txn observes another txn's effects? **N/A — TxnOp has no Read variant**;
  transactions cannot read (API limitation, drives E7i/E7j/E7v
  classifications).

## Part II — Invariants (contract candidates, not predeclared results)

- **I1 visibility** — staged state invisible; committed state visible at some
  measured point between CommitTxn apply and ACK.
- **I2 atomic transaction visibility** — per-op atomic vs per-txn snapshot
  visibility to concurrent readers: measured, not assumed.
- **I3 read/write synchronization** — readers never block on the gate;
  blocking only on per-structure RwLock contention.
- **I4 same-key conflict behavior** — commit-order-wins; no conflict
  detection (measured, incl. lost-update shape).
- **I5 transaction ordering** — WAL order = commit order; stage order ≠ commit order.
- **I6 collection isolation** — concurrent txns on different collections never
  contaminate.
- **I7 backup/checkpoint/compaction interaction** — gate serialization under
  true concurrent scheduling; never partial-txn snapshots.
- **I8 concurrency safety** — no torn reads, no malformed records, no lost
  ACKed writes, checker clean after every concurrent scenario.
- **I9 isolation presence/absence** — classification ONLY from §12–§32
  experiments.
- **I10 unsupported stronger guarantees** — explicit list (MVCC,
  serializability-if-unmeasurable, linearizability-if-unmeasurable).

## Part III — Reference model & event log

Independent concurrency-aware model in the harness (the DB is never its own
oracle): `E7Model` maps key → (uuid_str, num) for the single collection;
transitions driven ONLY by **commit events in commit order** (stage/rollback
never touch it). Every operation records an event:

```
event{ seq, thread_id, txn_id, op, key, t_invoke_us, t_complete_us, result, commit_status }
```

`seq` assigned under a log Mutex at invocation; timestamps from
`std::time::Instant` (MONOTONIC — wall clock never used for ordering).
Happens-before reconstructible: seq + per-thread intervals. The model is
compared to `export_state`-derived DB state after every scenario (and after
restart for persistence-tagged scenarios). A linearizability analysis for the
single-register subset checks each history's real-time ordering against the
allowed histories.

## Part IV — Families (PH3E-CONC-001)

| family | experiment | status |
|---|---|---|
| E7a | concurrent readers ×1/2/4/8/16 (attend + scan), errors/p50/p95/p99 | planned |
| E7b | readers + single writer flipping key A between two values; reader sees only committed values, never torn/intermediate | planned |
| E7c | repeated read: ordinary reads of A around a concurrent commit (old/old vs old/new); N reps, barriers | planned |
| E7d | dirty read: T1 stages A=X (no commit), barrier, T2 reads A; then rollback | planned |
| E7e | atomic visibility: T1 commits A=X,B=X; concurrent reader reads A,B; enumerate observed (A,B) combinations | planned |
| E7f | delete/insert visibility: T1 commits DELETE A + INSERT B; reader observes all 4 combos? | planned |
| E7g | write/write same key, both commit orders, WAL order + recovery | planned |
| E7h | lost update shape (read-outside-txn → blind write), commit-order-wins, NO conflict detection | planned |
| E7i | write skew | UNSUPPORTED BY API (no transactional reads) |
| E7j | phantom/range in-txn | UNSUPPORTED BY API (no transactional query reads) |
| E7k | staged visibility (barrier, N reps) | planned |
| E7l | rollback visibility (read before/after rollback) | planned |
| E7m | commit visibility boundary (reader loop across commit; visibility vs ACK ordering, monotonic clocks) | planned |
| E7n | same-key delete/reinsert both orders + compaction + restart | planned |
| E7o | collection concurrency (T1→A, T2→B; read A / write B / backup / ckpt / compact) | planned |
| E7p | checkpoint + concurrent staged txn (deterministic barriers) | planned |
| E7q | compaction + concurrent commit | planned |
| E7r | backup + staged / committing txn | planned |
| E7s | concurrent staging T1/T2/T3, commit T3,T1,T2, txn-id distinctness | planned |
| E7t | commit contention: N txns at barrier, all commit, order recorded | planned |
| E7u | single-op linearizability (invoke/complete histories, register subset, history check) | planned |
| E7v | bounded serializability histories (2–4 txns; write-only txns — API limitation stated) | planned |
| E7w | schedule enumeration (controlled interleavings via barriers) | planned |
| E7x | bounded randomized histories (fixed seeds, 2–4 txns, small keyspace, full event logs) | planned |
| E7y | targeted crash/concurrency: T1 committed + T2 committing at gate → groupkill → fresh-process recovery | planned (2 windows × sync/group) |

## Part V — Rules

- Barriers/channels for synchronization; sleeps only inside bounded
  observation loops and documented as such.
- Raw immutable; new run ID for any rerun; failing seeds/histories preserved.
- Randomized campaign bounded (fixed seeds, seconds not hours — E8 is soak).
- Crash cells reuse existing crashgate machinery only.
- Regressions E1–E6 under NEW run IDs after any code change.
- No claim without its experiment: no dirty-read-absence claim without E7d;
  no serializability without history analysis (E7v); no linearizability
  without real-time histories (E7u).
